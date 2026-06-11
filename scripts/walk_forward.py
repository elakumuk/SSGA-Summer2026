#!/usr/bin/env python3
"""Walk-forward validation across multiple chronological train/test windows.

Implements the highest-priority follow-up named in NEXT_STEPS.md §1:

    > Walk-forward robustness is not proven yet. The current test-period
    > table helps, but the most important next validation is rolling or
    > expanding walk-forward evaluation across multiple train/test windows.

Four expanding-window splits are run for both the baseline config and the
Champion config (Carry + HMM regime in M1 only, blend = 0.25). Each
invocation re-fits M1 thresholds, retrains M2, refits HMM (if enabled), and
backtests on the corresponding test window. The script snapshots metrics
from `data/backtests/` after each run because that path is shared across
runs.

Outputs
-------
    docs/ablation_results/walk_forward_summary.csv
    docs/visuals/walk_forward_sharpe_dsr.png
    docs/visuals/walk_forward_per_strategy.png

Usage
-----
    python scripts/walk_forward.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import (  # noqa: E402
    annualized_return,
    annualized_volatility,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
)
from src.run_pipeline import run_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("walk_forward")


@dataclass(frozen=True)
class Window:
    label: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str | None


WINDOWS: list[Window] = [
    Window("W1_test_2014_2016", "2006-01-01", "2013-12-31", "2014-01-01", "2016-12-31"),
    Window("W2_test_2016_2018", "2006-01-01", "2015-12-31", "2016-01-01", "2018-12-31"),
    Window("W3_test_2018_2020", "2006-01-01", "2017-12-31", "2018-01-01", "2020-12-31"),
    Window("W4_test_2021_now",  "2006-01-01", "2020-12-31", "2021-01-01", None),
]

# Where to read the pipeline's post-run outputs. These paths come from
# config.paths.backtests / predictions and are overwritten every run.
BACKTESTS_DIR = ROOT / "data" / "backtests"
PREDICTIONS_DIR = ROOT / "data" / "predictions"

STRATEGIES = ["m1_only", "m1_m2_binary", "m1_m2_linear", "m1_m2_ecdf"]


def _snapshot_run(snapshot_dir: Path) -> None:
    """Copy the freshly-written `data/backtests` + `data/predictions` into a snapshot."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for src in (BACKTESTS_DIR, PREDICTIONS_DIR):
        if not src.exists():
            continue
        dst = snapshot_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _read_returns(snapshot_dir: Path, mode: str, strategy: str) -> pd.Series:
    path = snapshot_dir / "backtests" / mode / f"{strategy}_returns.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s


def _slice_test(returns: pd.Series, window: Window) -> pd.Series:
    if returns.empty:
        return returns
    start = pd.Timestamp(window.test_start)
    end = pd.Timestamp(window.test_end) if window.test_end else returns.index.max()
    return returns.loc[(returns.index >= start) & (returns.index <= end)]


def _metrics_row(config_label: str, window: Window, mode: str, strategy: str,
                 returns_test: pd.Series) -> dict:
    if returns_test.empty:
        return {
            "config": config_label, "window": window.label, "mode": mode,
            "strategy": strategy, "test_start": window.test_start,
            "test_end": window.test_end or "latest",
            "n_weeks": 0, "ann_return": float("nan"),
            "ann_vol": float("nan"), "sharpe": float("nan"),
            "dsr": float("nan"), "max_drawdown": float("nan"),
        }
    dsr = deflated_sharpe_ratio(returns_test, n_trials=1)
    return {
        "config": config_label, "window": window.label, "mode": mode,
        "strategy": strategy, "test_start": window.test_start,
        "test_end": window.test_end or "latest",
        "n_weeks": int(returns_test.dropna().shape[0]),
        "ann_return": annualized_return(returns_test),
        "ann_vol": annualized_volatility(returns_test),
        "sharpe": sharpe_ratio(returns_test),
        "dsr": dsr["dsr"],
        "max_drawdown": max_drawdown(returns_test),
    }


def run_walk_forward(
    *,
    baseline_config: Path,
    champion_config: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for window in WINDOWS:
        for config_label, config_path in (
            ("BASE", baseline_config),
            ("CHAMPION", champion_config),
        ):
            logger.info("=" * 70)
            logger.info("Walk-forward: %s / %s", window.label, config_label)
            logger.info("  train %s -> %s  |  test %s -> %s",
                        window.train_start, window.train_end,
                        window.test_start, window.test_end or "latest")
            run_pipeline(
                str(config_path),
                project_root=ROOT,
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                skip_reports=True,
            )
            snap = output_dir / "snapshots" / f"{window.label}__{config_label}"
            _snapshot_run(snap)
            for mode in ("long_only", "long_short"):
                for strat in STRATEGIES:
                    r = _read_returns(snap, mode, strat)
                    rt = _slice_test(r, window)
                    rows.append(_metrics_row(config_label, window, mode, strat, rt))
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "walk_forward_summary.csv", index=False)
    return table


def _plot_sharpe_dsr(table: pd.DataFrame, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    focus_strategy = "m1_m2_ecdf"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for col_idx, mode in enumerate(("long_only", "long_short")):
        sub = table[(table["strategy"] == focus_strategy) & (table["mode"] == mode)].copy()
        pivot_sh = sub.pivot(index="window", columns="config", values="sharpe")
        pivot_dsr = sub.pivot(index="window", columns="config", values="dsr")
        x = np.arange(len(pivot_sh.index))
        w = 0.36
        for i, cfg, color in [(0, "BASE", "#5A5A5A"), (1, "CHAMPION", "#C7522A")]:
            axes[0, col_idx].bar(x + (i - 0.5) * w, pivot_sh[cfg].values, w,
                                 label=cfg, color=color, edgecolor="white")
            axes[1, col_idx].bar(x + (i - 0.5) * w, pivot_dsr[cfg].values, w,
                                 label=cfg, color=color, edgecolor="white")
        # Delta annotations
        for j, win in enumerate(pivot_sh.index):
            d_sh = pivot_sh.loc[win, "CHAMPION"] - pivot_sh.loc[win, "BASE"]
            col = "#3E8E41" if d_sh > 0 else "#C7522A"
            axes[0, col_idx].annotate(
                f"{d_sh:+.2f}",
                xy=(j, max(pivot_sh.loc[win, "BASE"], pivot_sh.loc[win, "CHAMPION"])),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontweight="bold", color=col, fontsize=9,
            )
            d_dsr = pivot_dsr.loc[win, "CHAMPION"] - pivot_dsr.loc[win, "BASE"]
            col2 = "#3E8E41" if d_dsr > 0 else "#C7522A"
            axes[1, col_idx].annotate(
                f"{d_dsr:+.3f}",
                xy=(j, max(pivot_dsr.loc[win, "BASE"], pivot_dsr.loc[win, "CHAMPION"])),
                xytext=(0, 5), textcoords="offset points",
                ha="center", fontweight="bold", color=col2, fontsize=9,
            )
        axes[0, col_idx].set_title(f"Sharpe — {mode}", fontweight="bold")
        axes[1, col_idx].set_title(f"Deflated SR — {mode}", fontweight="bold")
        axes[0, col_idx].axhline(0, color="black", lw=0.8, alpha=0.4)
        axes[1, col_idx].axhline(0.95, color="#999", lw=0.8, linestyle="--", alpha=0.6)
        for row in (0, 1):
            axes[row, col_idx].set_xticks(x)
            axes[row, col_idx].set_xticklabels([w_.replace("_", "\n") for w_ in pivot_sh.index],
                                               fontsize=8)
            axes[row, col_idx].grid(axis="y", alpha=0.25)
            axes[row, col_idx].legend(framealpha=0.95, fontsize=9)
    fig.suptitle(f"Walk-forward validation — {focus_strategy}, BASE vs CHAMPION (4 expanding windows)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(dest_dir / "walk_forward_sharpe_dsr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_strategy(table: pd.DataFrame, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row")
    for col_idx, strat in enumerate(STRATEGIES):
        for row_idx, mode in enumerate(("long_only", "long_short")):
            sub = table[(table["strategy"] == strat) & (table["mode"] == mode)].copy()
            pivot = sub.pivot(index="window", columns="config", values="sharpe")
            x = np.arange(len(pivot.index))
            w = 0.36
            for i, cfg, color in [(0, "BASE", "#5A5A5A"), (1, "CHAMPION", "#C7522A")]:
                axes[row_idx, col_idx].bar(x + (i - 0.5) * w, pivot[cfg].values, w,
                                           label=cfg, color=color, edgecolor="white")
            for j, win in enumerate(pivot.index):
                d = pivot.loc[win, "CHAMPION"] - pivot.loc[win, "BASE"]
                col = "#3E8E41" if d > 0 else "#C7522A"
                ymax = max(pivot.loc[win, "BASE"], pivot.loc[win, "CHAMPION"])
                axes[row_idx, col_idx].annotate(
                    f"{d:+.2f}", xy=(j, ymax), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    color=col, fontsize=8, fontweight="bold",
                )
            axes[row_idx, col_idx].set_title(f"{strat} / {mode}", fontsize=10, fontweight="bold")
            axes[row_idx, col_idx].set_xticks(x)
            axes[row_idx, col_idx].set_xticklabels(
                [w_.split("_test_")[1].replace("_", "→") for w_ in pivot.index],
                fontsize=8,
            )
            axes[row_idx, col_idx].axhline(0, color="black", lw=0.8, alpha=0.4)
            axes[row_idx, col_idx].grid(axis="y", alpha=0.25)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(f"{mode}\nSharpe", fontweight="bold")
            if col_idx == 0 and row_idx == 0:
                axes[row_idx, col_idx].legend(loc="upper right", framealpha=0.95, fontsize=8)
    fig.suptitle("Walk-forward Sharpe — every strategy × mode, BASE vs CHAMPION",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(dest_dir / "walk_forward_per_strategy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_plots(table: pd.DataFrame, dest_dir: Path) -> None:
    _plot_sharpe_dsr(table, dest_dir)
    _plot_per_strategy(table, dest_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "config" / "config.yaml")
    parser.add_argument(
        "--champion-config",
        type=Path,
        default=ROOT / "config" / "config_experimental_carry_hmm.yaml",
        help="Experimental configuration to compare against baseline (carry + HMM regime).",
    )
    parser.add_argument("--ablation-dir", type=Path, default=ROOT / "docs" / "ablation_results")
    parser.add_argument("--visuals-dir", type=Path, default=ROOT / "docs" / "visuals")
    args = parser.parse_args(argv)

    table = run_walk_forward(
        baseline_config=args.baseline_config,
        champion_config=args.champion_config,
        output_dir=args.ablation_dir,
    )
    render_plots(table, args.visuals_dir)

    # Print a compact summary
    focus = table[(table["strategy"] == "m1_m2_ecdf")].copy()
    pivot = focus.pivot_table(index=["window", "mode"], columns="config", values=["sharpe", "dsr"])
    print()
    print("=" * 70)
    print("Walk-forward summary (m1_m2_ecdf strategy):")
    print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
