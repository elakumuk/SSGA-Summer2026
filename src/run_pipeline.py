"""End-to-end pipeline orchestration."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.backtest import returns_wide_from_panel, run_all_strategies
from src.config import (
    apply_config_overrides,
    apply_split_overrides,
    clone_config_with_m1_allow_short,
    load_config,
    save_config_snapshot,
)
from src.data_providers import ingest_macro_data, ingest_market_data
from src.data_validation import (
    build_modeling_panel,
    ticker_coverage_summary,
    validate_model_panel,
    validate_price_panel,
)
from src.asset_analysis import (
    asset_analysis_markdown_sections,
    build_asset_analysis,
    generate_asset_component_report,
    strategy_overlays_from_mode_results,
)
from src.diagnostics import generate_dual_mode_report, run_diagnostics
from src.feature_engineering import build_features, get_feature_columns, save_model_panel
from src.labels import add_forward_returns, build_m1_target, build_meta_labels
from src.model_m1 import build_m1_model, split_train_test
from src.model_m2 import fit_m2, predict_m2
from src.research_logger import ResearchLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

M1_MODES: list[tuple[str, bool]] = [
    ("long_only", False),
    ("long_short", True),
]


@dataclass
class PipelineRunSummary:
    run_dir: Path
    effective_start: str | None
    effective_end: str | None
    used_cache: bool


@dataclass
class ModeRunResult:
    mode_name: str
    allow_short: bool
    results: dict
    panel: pd.DataFrame
    metrics_table: pd.DataFrame
    m2_metrics: dict
    backtests_dir: Path
    m1_signal_analysis: dict | None = None
    m1_signal_chart: str | None = None
    m1_exposure_analysis: dict | None = None
    per_asset_ic: pd.DataFrame | None = None
    m1_exposure_chart_rel: str | None = None
    m1_sens_chart_rel: str | None = None


def _cleanup_stale_reports_root(reports_root: Path) -> None:
    """Remove legacy entries from reports/ root; only final_report.md and subdirs may remain."""
    if not reports_root.exists():
        return
    allowed = {"final_report.md", "final", "mode_comparison", "assets"}
    for path in list(reports_root.iterdir()):
        if path.name in allowed:
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def create_run_dir(cfg, base: Path | None = None) -> Path:
    root = base or Path.cwd()
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = root / cfg.path("runs", root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_m1_mode(
    cfg,
    base_panel: pd.DataFrame,
    feature_cols: list[str],
    returns_wide: pd.DataFrame,
    *,
    mode_name: str,
    allow_short: bool,
    root: Path,
    run_dir: Path,
) -> ModeRunResult:
    """Train M1/M2 and backtest for one shorting mode."""
    mode_cfg = clone_config_with_m1_allow_short(cfg, allow_short)
    save_config_snapshot(mode_cfg, run_dir / f"config_{mode_name}.yaml")

    logger.info("=== M1 mode: %s (allow_short=%s) ===", mode_name, allow_short)

    panel = base_panel.copy()
    train, test = split_train_test(panel, mode_cfg)
    fwd_col = f"forward_return_{mode_cfg.labels.horizon_weeks}w"

    m1 = build_m1_model(mode_cfg)
    X_train = train[feature_cols].fillna(0)
    returns_train = returns_wide.loc[
        (returns_wide.index >= pd.Timestamp(mode_cfg.split.train_start))
        & (returns_wide.index <= pd.Timestamp(mode_cfg.split.train_end))
    ]
    m1.fit(
        X_train,
        train["m1_target"],
        forward_returns=train[fwd_col],
        panel=train,
        returns_wide=returns_train,
        portfolio_cfg=mode_cfg.portfolio,
    )
    X_panel = panel[feature_cols].fillna(0)
    m1_signals = m1.predict_signal(X_panel)
    m1_scores = m1.predict_score(X_panel)
    m1_conviction = m1.predict_conviction(X_panel)
    panel = build_meta_labels(panel, m1_signals, m1_scores, mode_cfg)
    panel["M1_conviction"] = m1_conviction.reindex(panel.index).fillna(0.0)

    m2_model, _ = fit_m2(panel, mode_cfg)
    panel = predict_m2(m2_model, panel, mode_cfg)

    predictions_dir = root / mode_cfg.paths.predictions / mode_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    panel.reset_index().to_parquet(predictions_dir / "panel_with_predictions.parquet", index=False)

    if mode_name == "long_only":
        features_dir = root / mode_cfg.paths.features
        save_model_panel(panel, features_dir / "model_panel.parquet")
        panel_report = validate_model_panel(panel.reset_index(), mode_cfg.assets.tickers)
        panel_report.save(run_dir / "model_panel_validation.json")

    train, test = split_train_test(panel, mode_cfg)
    train_proba = train.loc[train["M1_signal"] != 0, "p_success"]
    results = run_all_strategies(panel, returns_wide, mode_cfg, train_proba=train_proba)

    backtests_dir = root / mode_cfg.paths.backtests / mode_name
    backtests_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        res.returns.to_frame().to_parquet(backtests_dir / f"{name}_returns.parquet")

    diag_summary = run_diagnostics(
        results,
        panel,
        test,
        mode_cfg.m2.threshold,
        backtests_dir,
        cfg=mode_cfg,
        returns_wide=returns_wide,
        train_panel=train,
    )

    short_pct = (m1_signals == -1).mean() * 100
    long_pct = (m1_signals == 1).mean() * 100
    logger.info(
        "Mode %s complete: long=%.1f%% short=%.1f%% M1-only ann. return=%.4f",
        mode_name,
        long_pct,
        short_pct,
        pd.DataFrame(diag_summary["metrics_table"]).set_index("strategy").loc["m1_only", "annualized_return"],
    )

    exposure_chart = diag_summary.get("exposure_chart")
    sens_chart = diag_summary.get("sens_chart")
    return ModeRunResult(
        mode_name=mode_name,
        allow_short=allow_short,
        results=results,
        panel=panel,
        metrics_table=pd.DataFrame(diag_summary["metrics_table"]),
        m2_metrics=diag_summary["m2_metrics"],
        backtests_dir=backtests_dir,
        m1_signal_analysis=diag_summary.get("m1_signal_analysis"),
        m1_exposure_analysis=diag_summary.get("m1_exposure_analysis"),
        per_asset_ic=diag_summary.get("per_asset_ic"),
        m1_exposure_chart_rel=f"final/{mode_name}/figures/{exposure_chart}" if exposure_chart else None,
        m1_sens_chart_rel=f"final/{mode_name}/figures/{sens_chart}" if sens_chart else None,
    )


def _clip_panel_to_start(df: pd.DataFrame, start: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out[out["date"] >= start_ts].reset_index(drop=True)


def run_pipeline(
    config_path: str,
    *,
    project_root: Path | None = None,
    data_start: str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    require_full_universe: bool | None = None,
    config_overrides: dict | None = None,
    refresh_data: bool = False,
    skip_reports: bool = False,
) -> PipelineRunSummary:
    root = project_root or Path.cwd()
    cfg = load_config(root / config_path if not Path(config_path).is_absolute() else config_path)
    cfg = apply_split_overrides(
        cfg,
        data_start=data_start,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        require_full_universe=require_full_universe,
    )
    if config_overrides:
        cfg = apply_config_overrides(cfg, config_overrides)
    used_cache = not refresh_data
    run_dir = create_run_dir(cfg, root)
    save_config_snapshot(cfg, run_dir / "config_snapshot.yaml")
    ingest_start = cfg.data_start_resolved()
    universe_mode = "full 7-asset" if cfg.split.require_full_universe else "partial (per-ticker availability)"
    logger.info(
        "Chronological split — data: %s | train: %s to %s | test: %s to %s | universe: %s",
        ingest_start,
        cfg.split.train_start,
        cfg.split.train_end,
        cfg.split.test_start,
        cfg.split.test_end or "latest",
        universe_mode,
    )

    rlog = ResearchLogger(run_dir / "research_log.jsonl")
    rlog.log_stage("data_ingest", llm_used=False, output_used="yfinance and FRED providers")

    raw_dir = root / cfg.paths.raw
    processed_dir = root / cfg.paths.processed

    market = ingest_market_data(
        cfg.assets.tickers,
        cfg.assets.vix_ticker,
        ingest_start,
        cfg.split.test_end,
        raw_dir,
        processed_dir,
        use_cache=not refresh_data,
    )
    requested_start = pd.Timestamp(ingest_start)
    cached_start = pd.to_datetime(market["date"]).min()
    # Weekly data requested from a calendar date may naturally begin on the first Friday.
    # Refresh only when the cache is materially later than the requested history.
    if not refresh_data and cached_start > requested_start + pd.Timedelta(days=14):
        logger.warning(
            "Requested data_start %s is before cached market history (%s). "
            "Automatically refreshing market and macro data.",
            ingest_start,
            cached_start.date(),
        )
        market = ingest_market_data(
            cfg.assets.tickers,
            cfg.assets.vix_ticker,
            ingest_start,
            cfg.split.test_end,
            raw_dir,
            processed_dir,
            use_cache=False,
        )
        used_cache = False
    macro = ingest_macro_data(
        cfg.macro.fred_series,
        ingest_start,
        cfg.split.test_end,
        raw_dir,
        processed_dir,
        market_weekly=market,
        use_cache=not refresh_data and used_cache,
    )
    market = _clip_panel_to_start(market, ingest_start)
    macro = _clip_panel_to_start(macro, ingest_start)

    vix_label = cfg.assets.vix_ticker.replace("^", "")
    market_assets = build_modeling_panel(
        market[market["ticker"].isin(cfg.assets.tickers)],
        cfg.assets.tickers,
        require_full_universe=cfg.split.require_full_universe,
    )
    vix_rows = market[market["ticker"] == vix_label]
    market_for_features = pd.concat([market_assets, vix_rows], ignore_index=True)
    price_report = validate_price_panel(
        market_assets,
        cfg.assets.tickers,
        balanced=cfg.split.require_full_universe,
    )
    coverage = ticker_coverage_summary(market_assets, cfg.assets.tickers)
    coverage.to_csv(run_dir / "ticker_coverage.csv", index=False)
    price_report.save(run_dir / "validation_report.json")
    if not price_report.passed:
        logger.warning("Price validation reported issues; see %s", run_dir / "validation_report.json")
    if price_report.effective_start_date and pd.Timestamp(cfg.split.train_start) < pd.Timestamp(
        price_report.effective_start_date
    ):
        logger.info(
            "Requested train_start %s precedes effective panel start %s; "
            "training uses the overlapping window only.",
            cfg.split.train_start,
            price_report.effective_start_date,
        )
    rlog.log_stage("feature_engineering", llm_used=False, output_used="factor library with shift(1) and macro lag")
    base_panel = build_features(market_for_features, macro, cfg, vix_ticker=vix_label)
    base_panel = add_forward_returns(base_panel, cfg.labels.horizon_weeks)
    base_panel = build_m1_target(base_panel, cfg)

    feature_cols = get_feature_columns(base_panel)
    returns_wide = returns_wide_from_panel(base_panel.reset_index(), cfg.assets.tickers)

    mode_results: list[ModeRunResult] = []
    for mode_name, allow_short in M1_MODES:
        rlog.log_stage(
            "m1",
            llm_used=False,
            output_used=f"M1 mode={mode_name}, allow_short={allow_short}",
        )
        mode_results.append(
            run_m1_mode(
                cfg,
                base_panel,
                feature_cols,
                returns_wide,
                mode_name=mode_name,
                allow_short=allow_short,
                root=root,
                run_dir=run_dir,
            )
        )

    if not skip_reports:
        asset_analysis = build_asset_analysis(
            returns_wide,
            cfg,
            effective_start=price_report.effective_start_date,
            effective_end=price_report.effective_end_date,
            macro_series_loaded=cfg.macro.fred_series + [vix_label],
        )
        reports_root = root / "reports"
        assets_dir = reports_root / "assets"
        final_dir = reports_root / "final"
        mode_comparison_dir = reports_root / "mode_comparison"
        _cleanup_stale_reports_root(reports_root)

        strategy_overlays = strategy_overlays_from_mode_results(mode_results)
        generate_asset_component_report(
            asset_analysis,
            assets_dir / "asset_component_analysis.md",
            strategy_overlays=strategy_overlays,
        )
        asset_sections = asset_analysis_markdown_sections(
            asset_analysis,
            assets_dir,
            image_prefix="assets/",
            strategy_overlays=strategy_overlays,
        )

        rlog.log_stage(
            "diagnostics",
            llm_used=False,
            output_used="dual-mode report + per-asset component analysis",
        )
        generate_dual_mode_report(
            mode_results,
            reports_root / "final_report.md",
            final_dir=final_dir,
            mode_comparison_dir=mode_comparison_dir,
            cfg=cfg,
            effective_start=price_report.effective_start_date,
            effective_end=price_report.effective_end_date,
            asset_analysis_sections=asset_sections,
        )
    else:
        rlog.log_stage("diagnostics", llm_used=False, output_used="skipped report generation (grid search)")

    logger.info("Pipeline complete. Run directory: %s", run_dir)
    logger.info("Compared M1 modes: long_only (no shorts) and long_short (shorts enabled)")
    return PipelineRunSummary(
        run_dir=run_dir,
        effective_start=price_report.effective_start_date,
        effective_end=price_report.effective_end_date,
        used_cache=used_cache,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run meta-labeling research pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--data-start",
        metavar="DATE",
        help="Earliest date to download/store prices (ISO date). Overrides split.data_start.",
    )
    parser.add_argument(
        "--train-start",
        metavar="DATE",
        help="Chronological start of the train split (ISO date, e.g. 2006-01-01). Overrides config split.train_start.",
    )
    parser.add_argument(
        "--train-end",
        metavar="DATE",
        help="Chronological end of the train split (ISO date, e.g. 2020-12-31). Overrides config split.train_end.",
    )
    parser.add_argument(
        "--test-start",
        metavar="DATE",
        help="Chronological start of the test split (ISO date, e.g. 2021-01-01). Overrides config split.test_start.",
    )
    parser.add_argument(
        "--test-end",
        metavar="DATE",
        help="Optional end of the test split (ISO date). Overrides config split.test_end (default: open-ended).",
    )
    parser.add_argument(
        "--partial-universe",
        action="store_true",
        help="Allow partial ETF universes before all seven assets exist (sets require_full_universe=false).",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download market/macro data instead of using cached parquet files.",
    )
    args = parser.parse_args(argv)
    try:
        run_pipeline(
            args.config,
            data_start=args.data_start,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            require_full_universe=False if args.partial_universe else None,
            refresh_data=args.refresh_data,
        )
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
