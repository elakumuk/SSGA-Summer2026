"""Grid search runner for pipeline parameter sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import _deep_merge, apply_config_overrides, load_config, save_config_snapshot
from src.diagnostics import strategy_metrics_on_period
from src.run_pipeline import run_pipeline

logger = logging.getLogger(__name__)

STRATEGIES = ("m1_only", "m1_m2_linear", "m1_m2_ecdf")
MODES = ("long_only", "long_short")
BACKTEST_ARTIFACTS = (
    "metrics_table.csv",
    "diagnostics_summary.json",
    "m1_signal_m2_analysis.csv",
)


def _parse_dotted_path(key: str) -> list[str]:
    return key.split(".")


def _set_nested(target: dict[str, Any], parts: list[str], value: Any) -> None:
    cur = target
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _flatten_combo(combo: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in combo.items():
        parts = _parse_dotted_path(key)
        _set_nested(flat, parts, value)
    return flat


def derive_test_start_from_train_end(train_end: str) -> str:
    return (date.fromisoformat(train_end) + timedelta(days=1)).isoformat()


def load_grid_spec(spec_path: Path) -> dict[str, Any]:
    with spec_path.open() as f:
        return yaml.safe_load(f)


def expand_grid_spec(spec: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    """Expand factorial grid into a list of run definitions."""
    base_path = project_root / spec.get("base_config", "config/config.yaml")
    base_cfg = load_config(base_path)
    fixed = spec.get("fixed", {})
    grid: dict[str, list[Any]] = spec.get("grid", {})
    auto = spec.get("auto", {})

    if not grid:
        raise ValueError("grid_search_spec must define a non-empty 'grid' section")

    keys = list(grid.keys())
    values_product = list(product(*(grid[k] for k in keys)))

    runs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for combo_values in values_product:
        flat_combo = dict(zip(keys, combo_values))
        overrides = _flatten_combo(flat_combo)
        if fixed:
            overrides = _deep_merge(fixed, overrides)

        if auto.get("test_start_from_train_end", False):
            train_end = overrides.get("split", {}).get("train_end")
            if train_end:
                overrides.setdefault("split", {})["test_start"] = derive_test_start_from_train_end(train_end)

        merged_cfg = apply_config_overrides(base_cfg, overrides)
        param_sig = json.dumps(flat_combo, sort_keys=True, default=str)
        run_id = hashlib.md5(param_sig.encode()).hexdigest()[:12]

        if run_id in seen_ids:
            raise ValueError(f"Duplicate run_id generated for combo: {flat_combo}")
        seen_ids.add(run_id)

        runs.append(
            {
                "run_id": run_id,
                "run_index": len(runs) + 1,
                "overrides": overrides,
                "flat_params": flat_combo,
                "train_end": merged_cfg.split.train_end,
                "test_start": merged_cfg.split.test_start,
                "m2_threshold": merged_cfg.m2.threshold,
                "transaction_cost_bps": merged_cfg.portfolio.transaction_cost_bps,
            }
        )

    return runs


def _git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def warn_if_cache_used(project_root: Path, use_cache: bool) -> None:
    cache_market = project_root / "data/processed/market_weekly.parquet"
    cache_macro = project_root / "data/processed/macro_weekly.parquet"
    if use_cache and (cache_market.exists() or cache_macro.exists()):
        logger.warning(
            "Using cached market/macro parquet under data/processed/. "
            "Results may not reflect data_start changes. Pass --refresh-data to re-download."
        )


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _collect_strategy_metrics(
    backtests_root: Path,
    test_start: str,
    test_end: str | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in MODES:
        mode_dir = backtests_root / mode
        metrics_path = mode_dir / "metrics_table.csv"
        full_metrics: dict[str, dict[str, float]] = {}
        if metrics_path.exists():
            table = pd.read_csv(metrics_path).set_index("strategy")
            for strat in STRATEGIES:
                if strat in table.index:
                    row = table.loc[strat]
                    full_metrics[strat] = {
                        "ann_return": float(row["annualized_return"]),
                        "sharpe": float(row["sharpe"]),
                        "max_drawdown": float(row["max_drawdown"]),
                        "hit_rate": float(row["hit_rate"]),
                    }

        diag_path = mode_dir / "diagnostics_summary.json"
        m2_metrics: dict[str, Any] = {}
        if diag_path.exists():
            with diag_path.open() as f:
                diag = json.load(f)
            m2_metrics = diag.get("m2_metrics", {})

        for strat in STRATEGIES:
            prefix = f"{mode}_{strat}"
            if strat in full_metrics:
                for k, v in full_metrics[strat].items():
                    out[f"{prefix}_{k}"] = v

            ret_path = mode_dir / f"{strat}_returns.parquet"
            if ret_path.exists():
                returns = pd.read_parquet(ret_path).iloc[:, 0]
                period = strategy_metrics_on_period(returns, start=test_start, end=test_end)
                out[f"{prefix}_test_ann_return"] = period["annualized_return"]
                out[f"{prefix}_test_sharpe"] = period["sharpe"]
                out[f"{prefix}_test_max_drawdown"] = period["max_drawdown"]
                out[f"{prefix}_test_hit_rate"] = period["hit_rate"]
                out[f"{prefix}_test_n_weeks"] = period["n_weeks"]

        for mk in ("accuracy", "precision", "recall", "f1", "auc"):
            if mk in m2_metrics:
                out[f"{mode}_m2_{mk}"] = m2_metrics[mk]

    out["rank_score"] = out.get("long_only_m1_m2_linear_test_sharpe", float("nan"))
    out["rank_score_full"] = out.get("long_only_m1_m2_linear_sharpe", float("nan"))
    return out


def snapshot_run_artifacts(
    project_root: Path,
    run_out_dir: Path,
    pipeline_run_dir: Path,
    test_start: str,
    test_end: str | None,
) -> dict[str, Any]:
    run_out_dir.mkdir(parents=True, exist_ok=True)
    _copy_tree(pipeline_run_dir, run_out_dir / "pipeline_run")

    backtests_root = project_root / "data" / "backtests"
    for mode in MODES:
        src_mode = backtests_root / mode
        dst_mode = run_out_dir / "backtests" / mode
        dst_mode.mkdir(parents=True, exist_ok=True)
        for name in BACKTEST_ARTIFACTS:
            _copy_tree(src_mode / name, dst_mode / name)
        for strat in STRATEGIES:
            _copy_tree(src_mode / f"{strat}_returns.parquet", dst_mode / f"{strat}_returns.parquet")

    metrics = _collect_strategy_metrics(backtests_root, test_start, test_end)
    with (run_out_dir / "metrics_summary.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return metrics


def write_summary_md(results_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if results_df.empty:
        path.write_text("# Grid Search Summary\n\nNo successful runs.\n")
        return

    ranked = results_df.sort_values("rank_score", ascending=False, na_position="last")
    lines = [
        "# Grid Search Summary",
        "",
        f"Total runs: {len(results_df)}",
        "",
        "## Top 10 by test-set Sharpe (M1+M2 Linear, long-only)",
        "",
        "| Rank | run_id | train_end | test_start | m2_threshold | tc_bps | test_sharpe | full_sharpe |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, (_, row) in enumerate(ranked.head(10).iterrows(), start=1):
        lines.append(
            f"| {i} | {row.get('run_id', '')} | {row.get('train_end', '')} | {row.get('test_start', '')} | "
            f"{row.get('m2_threshold', '')} | {row.get('transaction_cost_bps', '')} | "
            f"{row.get('rank_score', float('nan')):.4f} | {row.get('rank_score_full', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Primary ranking uses **test-period** Sharpe (`rank_score`). "
            "Full-sample Sharpe (`rank_score_full`) is for reference only.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_grid_search(
    *,
    project_root: Path,
    spec_path: Path,
    use_cache: bool = True,
    dry_run: bool = False,
    resume: bool = False,
    max_runs: int | None = None,
    sweep_dir: Path | None = None,
) -> Path:
    project_root = project_root.resolve()
    spec = load_grid_spec(spec_path)
    runs = expand_grid_spec(spec, project_root)
    if max_runs is not None:
        runs = runs[:max_runs]

    if sweep_dir is not None:
        sweep_dir = sweep_dir.resolve()
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_id = sweep_dir.name
    else:
        sweep_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sweep_dir = project_root / "runs" / "grid_search" / sweep_id
        sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sweep_dir / "manifest.yaml"
    if not manifest_path.exists():
        manifest = {
            "sweep_id": sweep_id,
            "spec_path": str(spec_path),
            "n_runs": len(runs),
            "use_cache": use_cache,
            "git_commit": _git_commit(project_root),
            "created_at": datetime.utcnow().isoformat(),
        }
        with manifest_path.open("w") as f:
            yaml.safe_dump(manifest, f, default_flow_style=False)
        shutil.copy2(spec_path, sweep_dir / "grid_search_spec.yaml")

    results_csv = sweep_dir / "results.csv"
    results_jsonl = sweep_dir / "results.jsonl"
    failures_jsonl = sweep_dir / "failures.jsonl"

    completed_ids: set[str] = set()
    if resume and results_csv.exists():
        prev = pd.read_csv(results_csv)
        completed_ids = set(prev["run_id"].astype(str))

    warn_if_cache_used(project_root, use_cache)

    if dry_run:
        for run in runs:
            logger.info("DRY RUN %s: %s", run["run_id"], run["flat_params"])
        return sweep_dir

    base_config_path = project_root / spec.get("base_config", "config/config.yaml")
    rows: list[dict[str, Any]] = []

    for run in runs:
        run_id = run["run_id"]
        run_index = run["run_index"]
        run_label = f"run_{run_index:03d}"
        run_out_dir = sweep_dir / run_label

        if run_id in completed_ids:
            logger.info("Skipping completed run_id=%s (%s)", run_id, run_label)
            continue

        logger.info("=== Grid run %s / %s (id=%s) ===", run_index, len(runs), run_id)
        logger.info("Params: %s", run["flat_params"])

        run_config_path = run_out_dir / "config.yaml"
        run_out_dir.mkdir(parents=True, exist_ok=True)
        cfg_for_save = apply_config_overrides(load_config(base_config_path), run["overrides"])
        save_config_snapshot(cfg_for_save, run_config_path)

        t0 = time.perf_counter()
        try:
            summary = run_pipeline(
                str(base_config_path),
                project_root=project_root,
                config_overrides=run["overrides"],
                refresh_data=not use_cache,
                skip_reports=True,
            )
            elapsed = time.perf_counter() - t0
            test_end = cfg_for_save.split.test_end
            metrics = snapshot_run_artifacts(
                project_root,
                run_out_dir,
                summary.run_dir,
                run["test_start"],
                test_end,
            )
            row = {
                "run_id": run_id,
                "run_label": run_label,
                "run_index": run_index,
                "train_end": run["train_end"],
                "test_start": run["test_start"],
                "m2_threshold": run["m2_threshold"],
                "transaction_cost_bps": run["transaction_cost_bps"],
                "effective_start": summary.effective_start,
                "effective_end": summary.effective_end,
                "used_cache": summary.used_cache,
                "elapsed_sec": round(elapsed, 2),
                "pipeline_run_dir": str(summary.run_dir),
                "status": "ok",
                **metrics,
                **{f"param_{k}": v for k, v in run["flat_params"].items()},
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            row = {
                "run_id": run_id,
                "run_label": run_label,
                "run_index": run_index,
                "status": "failed",
                "elapsed_sec": round(elapsed, 2),
                "error": str(exc),
                **{f"param_{k}": v for k, v in run["flat_params"].items()},
            }
            with failures_jsonl.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            logger.exception("Run %s failed", run_label)
            continue

        rows.append(row)
        with results_jsonl.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

        if results_csv.exists():
            combined = pd.concat([pd.read_csv(results_csv), pd.DataFrame([row])], ignore_index=True)
        else:
            combined = pd.DataFrame([row])
        combined.to_csv(results_csv, index=False)

    if results_csv.exists():
        final_df = pd.read_csv(results_csv)
    else:
        final_df = pd.DataFrame(rows)
    write_summary_md(final_df, sweep_dir / "summary.md")
    logger.info("Grid search complete. Outputs: %s", sweep_dir)
    return sweep_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grid search over pipeline parameters (~40 runs)")
    parser.add_argument(
        "--spec",
        default="scripts/grid_search_spec.yaml",
        help="Path to grid search specification YAML",
    )
    parser.add_argument("--project-root", default=".", help="Repository root")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Re-download market/macro data on every pipeline run (default: use cache with warning)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print expanded grid without executing")
    parser.add_argument("--resume", action="store_true", help="Skip run_ids already in results.csv")
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Existing sweep directory to resume (e.g. runs/grid_search/20260101_120000)",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of runs (smoke test)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    use_cache = not args.refresh_data

    try:
        run_grid_search(
            project_root=Path(args.project_root),
            spec_path=Path(args.spec),
            use_cache=use_cache,
            dry_run=args.dry_run,
            resume=args.resume,
            max_runs=args.max_runs,
            sweep_dir=Path(args.sweep_dir) if args.sweep_dir else None,
        )
        return 0
    except Exception:
        logger.exception("Grid search failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
