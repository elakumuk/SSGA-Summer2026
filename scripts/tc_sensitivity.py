#!/usr/bin/env python3
"""Transaction cost sensitivity ladder.

Re-runs the backtest layer (M1+M2 panel held fixed) across a grid of
transaction costs and reports how net annualized return, Sharpe, and turnover
decay as costs rise.

The motivation is institutional realism. SSGA scale strategies trade at
materially higher round-trip costs than the 5 bps default that academic
backtests typically assume; PMs almost always want to know the alpha-decay
slope before greenlighting deployment. This script answers exactly that.

Usage
-----
    python scripts/tc_sensitivity.py \\
        --run-dir runs/<timestamp> \\
        --bps 5 10 15 25 50 \\
        --mode long_only

Outputs (written next to the script's --output-dir):
    tc_sensitivity_summary.csv  -- one row per (strategy, tc_bps)
    tc_sensitivity_chart.png    -- ann return + Sharpe vs cost curves
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import returns_wide_from_panel, run_all_strategies  # noqa: E402
from src.config import apply_config_overrides, load_config  # noqa: E402
from src.diagnostics import (  # noqa: E402
    annualized_return,
    annualized_volatility,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tc_sensitivity")

DEFAULT_BPS_GRID = [5.0, 10.0, 15.0, 25.0, 50.0]


def _metrics_row(name: str, tc_bps: float, returns: pd.Series, turnover: pd.Series) -> dict:
    dsr = deflated_sharpe_ratio(returns, n_trials=1)
    return {
        "strategy": name,
        "tc_bps": tc_bps,
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "dsr": dsr["dsr"],
        "max_drawdown": max_drawdown(returns),
        "ann_turnover": float(turnover.mean() * 52),
    }


def run_tc_ladder(
    run_dir: Path,
    *,
    mode: str = "long_only",
    bps_grid: list[float] | None = None,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Re-run all strategies at each transaction-cost level."""
    bps_grid = bps_grid or DEFAULT_BPS_GRID
    config_path = config_path or (run_dir / f"config_{mode}.yaml")
    if not config_path.exists():
        # Fall back to the snapshot if mode-specific config is missing.
        config_path = run_dir / "config_snapshot.yaml"
    cfg = load_config(config_path)

    panel_path = ROOT / cfg.paths.predictions / mode / "panel_with_predictions.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Could not find {panel_path}. Run the pipeline first so M1+M2 predictions are cached."
        )

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "ticker"]).sort_index()

    returns_wide = returns_wide_from_panel(panel.reset_index(), cfg.assets.tickers)

    rows: list[dict] = []
    for tc in bps_grid:
        cfg_tc = apply_config_overrides(cfg, {"portfolio": {"transaction_cost_bps": float(tc)}})
        # Mirror run_m1_mode's train-window p_success extraction for ECDF sizing.
        train_mask = (panel.index.get_level_values("date") >= pd.Timestamp(cfg_tc.split.train_start)) & (
            panel.index.get_level_values("date") <= pd.Timestamp(cfg_tc.split.train_end)
        )
        train_panel = panel[train_mask]
        train_proba = train_panel.loc[train_panel.get("M1_signal", pd.Series(0)) != 0, "p_success"]
        if train_proba.empty and "p_success" in train_panel.columns:
            train_proba = train_panel["p_success"]
        results = run_all_strategies(panel, returns_wide, cfg_tc, train_proba=train_proba)
        for name, res in results.items():
            rows.append(_metrics_row(name, tc, res.returns, res.turnover))
        logger.info("TC ladder: %s bps complete (%d strategies)", tc, len(results))

    table = pd.DataFrame(rows)

    if output_dir is None:
        output_dir = run_dir / "tc_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "tc_sensitivity_summary.csv", index=False)

    _plot_decay(table, output_dir / "tc_sensitivity_chart.png")
    logger.info("TC ladder written to %s", output_dir)
    return table


def _plot_decay(table: pd.DataFrame, dest: Path) -> None:
    if table.empty:
        return
    strategies = sorted(table["strategy"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for strat in strategies:
        sub = table[table["strategy"] == strat].sort_values("tc_bps")
        axes[0].plot(sub["tc_bps"], sub["ann_return"], marker="o", label=strat)
        axes[1].plot(sub["tc_bps"], sub["sharpe"], marker="o", label=strat)
    axes[0].set_title("Annualized return vs cost")
    axes[1].set_title("Sharpe vs cost")
    for ax in axes:
        ax.set_xlabel("Transaction cost (bps round trip)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Ann. return")
    axes[1].set_ylabel("Sharpe ratio")
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def alpha_decay_slope(table: pd.DataFrame, strategy: str) -> float:
    """Linear slope of Sharpe vs TC for a single strategy (units: Sharpe per bp)."""
    sub = table[table["strategy"] == strategy].sort_values("tc_bps").dropna(subset=["sharpe"])
    if len(sub) < 2:
        return float("nan")
    slope, _ = np.polyfit(sub["tc_bps"].values, sub["sharpe"].values, 1)
    return float(slope)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Existing runs/<timestamp> directory")
    parser.add_argument("--mode", default="long_only", choices=["long_only", "long_short"])
    parser.add_argument(
        "--bps",
        nargs="+",
        type=float,
        default=DEFAULT_BPS_GRID,
        help="Transaction cost levels to sweep (bps round trip)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config override path")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    table = run_tc_ladder(
        args.run_dir,
        mode=args.mode,
        bps_grid=args.bps,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
