"""Backtest diagnostics and reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.backtest import BacktestResult
from src.config import PipelineConfig

logger = logging.getLogger(__name__)

WEEKS_PER_YEAR = 52


def annualized_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    cum = (1 + r).prod()
    years = len(r) / WEEKS_PER_YEAR
    if years <= 0:
        return 0.0
    return float(cum ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(r.std() * np.sqrt(WEEKS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    vol = annualized_volatility(returns)
    if vol == 0:
        return 0.0
    ann = annualized_return(returns) - rf
    return float(ann / vol)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0)).cumprod()
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def rolling_max_drawdown(returns: pd.Series, window: int = 52) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    roll_max = equity.rolling(window, min_periods=1).max()
    return equity / roll_max - 1


def information_ratio(strategy: pd.Series, benchmark: pd.Series) -> float:
    active = strategy - benchmark
    std = active.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(active.mean() * np.sqrt(WEEKS_PER_YEAR) / std)


def compute_ic(panel: pd.DataFrame, score_col: str = "M1_score", fwd_col: str | None = None) -> pd.Series:
    df = panel.reset_index()
    if fwd_col is None:
        fwd_cols = [c for c in df.columns if c.startswith("forward_return")]
        fwd_col = fwd_cols[0] if fwd_cols else None
    if fwd_col is None:
        return pd.Series(dtype=float)
    ics = []
    dates = []
    for date, grp in df.groupby("date"):
        if grp[score_col].notna().sum() < 2:
            continue
        ic = grp[score_col].corr(grp[fwd_col], method="spearman")
        ics.append(ic)
        dates.append(date)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="IC")


def strategy_metrics(result: BacktestResult, benchmark: BacktestResult | None = None) -> dict[str, float]:
    r = result.returns
    m: dict[str, float] = {
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "rolling_12m_max_drawdown": float(rolling_max_drawdown(r, 52).min()),
        "turnover": float(result.turnover.mean()),
        "annualized_turnover": float(result.turnover.mean() * WEEKS_PER_YEAR),
        "hit_rate": float((r > 0).mean()),
    }
    if benchmark is not None:
        m["excess_return_vs_benchmark"] = m["annualized_return"] - annualized_return(benchmark.returns)
        m["information_ratio"] = information_ratio(r, benchmark.returns)
    return m


def strategy_metrics_on_period(
    returns: pd.Series,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict[str, float]:
    """Compute strategy metrics on a date-filtered return slice."""
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    if r.empty:
        return {
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "n_weeks": 0,
        }
    return {
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "n_weeks": int(len(r)),
    }


def m2_classification_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, Any]:
    mask = y_true.notna() & y_prob.notna()
    y = y_true[mask].astype(int)
    p = y_prob[mask]
    if len(y) == 0:
        return {}
    pred = (p >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def build_metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    bench = results.get("equal_weight_1_7")
    rows = []
    for name, res in results.items():
        row = {"strategy": name, **strategy_metrics(res, bench)}
        rows.append(row)
    return pd.DataFrame(rows)


STRATEGY_LABELS: dict[str, str] = {
    "equal_weight_1_7": "Equal Weight (1/7)",
    "sixty_forty": "60/40 Benchmark",
    "m1_only": "M1 Only",
    "m1_m2_binary": "M1 + M2 (Binary)",
    "m1_m2_linear": "M1 + M2 (Linear)",
    "m1_m2_ecdf": "M1 + M2 (ECDF)",
}

REPORT_CHART_STRATEGIES = [
    "equal_weight_1_7",
    "sixty_forty",
    "m1_only",
    "m1_m2_linear",
    "m1_m2_ecdf",
]


def _strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def _fmt_pct(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def _fmt_num(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):.{decimals}f}"


def format_metrics_table_for_report(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """Return a display-friendly metrics table with readable labels and rounding."""
    df = metrics_table.copy()
    df["strategy"] = df["strategy"].map(_strategy_label)
    display = pd.DataFrame(
        {
            "Strategy": df["strategy"],
            "Ann. Return": df["annualized_return"].map(lambda x: _fmt_pct(x)),
            "Ann. Volatility": df["annualized_volatility"].map(lambda x: _fmt_pct(x)),
            "Sharpe": df["sharpe"].map(lambda x: _fmt_num(x)),
            "Max Drawdown": df["max_drawdown"].map(lambda x: _fmt_pct(x)),
            "Excess vs EW": df["excess_return_vs_benchmark"].map(lambda x: _fmt_pct(x)),
            "Info Ratio": df["information_ratio"].map(lambda x: _fmt_num(x)),
            "Weekly Hit Rate": df["hit_rate"].map(lambda x: _fmt_pct(x)),
        }
    )
    return display


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def _build_executive_summary(metrics_table: pd.DataFrame, m2_metrics: dict[str, Any]) -> list[str]:
    raw = metrics_table.set_index("strategy")
    ew = raw.loc["equal_weight_1_7"] if "equal_weight_1_7" in raw.index else None
    m1 = raw.loc["m1_only"] if "m1_only" in raw.index else None
    m2_linear = raw.loc["m1_m2_linear"] if "m1_m2_linear" in raw.index else None

    lines = [
        "This report compares a **meta-labeling pipeline** against standard benchmarks on seven global ETFs "
        "(SPY, TLT, GLD, VEA, VWO, HYG, VNQ). M1 proposes long/short/flat signals; M2 estimates trade quality "
        "and scales position size.",
        "",
        "**Research use only — not investment advice.**",
        "",
    ]

    if ew is not None:
        lines.append(
            f"- The **equal-weight benchmark** returned {_fmt_pct(ew['annualized_return'])} per year "
            f"with Sharpe {_fmt_num(ew['sharpe'])} and max drawdown {_fmt_pct(ew['max_drawdown'])}."
        )
    if m1 is not None:
        lines.append(
            f"- **M1 alone** produced lower return ({_fmt_pct(m1['annualized_return'])}) but also lower volatility "
            f"({_fmt_pct(m1['annualized_volatility'])}) than buy-and-hold benchmarks."
        )
    if m2_linear is not None and m1 is not None:
        vol_drop = m1["annualized_volatility"] - m2_linear["annualized_volatility"]
        lines.append(
            f"- **M1 + M2 (linear sizing)** improved risk-adjusted metrics: Sharpe {_fmt_num(m2_linear['sharpe'])} "
            f"vs {_fmt_num(m1['sharpe'])} for M1-only, with max drawdown {_fmt_pct(m2_linear['max_drawdown'])}. "
            "Much of the improvement comes from **lower exposure**, not higher raw returns."
        )
        if vol_drop > 0:
            lines.append(
                f"- M2 filtering reduced annualized volatility by roughly {_fmt_pct(vol_drop)} relative to M1-only."
            )
        lines.append(
            f"- Max drawdown moved from {_fmt_pct(m1['max_drawdown'])} (M1-only) to "
            f"{_fmt_pct(m2_linear['max_drawdown'])} (M1 + M2 linear)."
        )

    if m2_metrics:
        recall = m2_metrics.get("recall", float("nan"))
        precision = m2_metrics.get("precision", float("nan"))
        auc = m2_metrics.get("auc", float("nan"))
        lines.extend(
            [
                "",
                "**M2 meta-labeling (out-of-sample test period):**",
                f"- Precision {_fmt_num(precision)} — when M2 approves a trade, about {_fmt_pct(precision)} are profitable.",
                f"- Recall {_fmt_num(recall)} — M2 approves {_fmt_pct(recall)} of truly profitable M1 signals.",
                f"- AUC {_fmt_num(auc)} — modest discrimination between winning and losing trades.",
            ]
        )

    return lines


def save_report_charts(
    results: dict[str, BacktestResult],
    m2_metrics: dict[str, Any],
    reports_dir: Path,
    *,
    subdir: str | None = None,
) -> list[str]:
    """Create presentation-ready charts saved alongside final_report.md."""
    out_dir = reports_dir / subdir if subdir else reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir
    saved: list[str] = []
    palette = {
        "equal_weight_1_7": "#4C72B0",
        "sixty_forty": "#55A868",
        "m1_only": "#C44E52",
        "m1_m2_linear": "#8172B3",
        "m1_m2_ecdf": "#CCB974",
    }

    # 1. Cumulative returns (key strategies only)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in REPORT_CHART_STRATEGIES:
        if name not in results:
            continue
        cum = (1 + results[name].returns.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=_strategy_label(name), color=palette.get(name), linewidth=2)
    ax.set_title("Cumulative Growth of $1", fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio value")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    p = reports_dir / "strategy_cumulative_returns.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 2. Drawdown (key strategies)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in REPORT_CHART_STRATEGIES:
        if name not in results:
            continue
        eq = (1 + results[name].returns.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, label=_strategy_label(name), color=palette.get(name), linewidth=1.8)
    ax.set_title("Drawdown Over Time", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    p = reports_dir / "strategy_drawdown.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 3. Sharpe comparison bar chart
    metrics = build_metrics_table(results)
    metrics["label"] = metrics["strategy"].map(_strategy_label)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [palette.get(s, "#888888") for s in metrics["strategy"]]
    ax.bar(metrics["label"], metrics["sharpe"], color=colors, edgecolor="white")
    ax.set_title("Sharpe Ratio by Strategy", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(metrics["sharpe"]):
        ax.text(i, v + 0.02, _fmt_num(v, 2), ha="center", fontsize=8)
    p = reports_dir / "strategy_sharpe_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 4. Risk vs return scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    vol_pct = metrics["annualized_volatility"] * 100
    ret_pct = metrics["annualized_return"] * 100
    for _, row in metrics.iterrows():
        name = row["strategy"]
        ax.scatter(
            row["annualized_volatility"] * 100,
            row["annualized_return"] * 100,
            s=120,
            color=palette.get(name, "#888888"),
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            _strategy_label(name),
            (row["annualized_volatility"] * 100, row["annualized_return"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    sharpe_raw = metrics["annualized_return"] / metrics["annualized_volatility"].replace(0, np.nan)
    best_idx = sharpe_raw.idxmax()
    best = metrics.loc[best_idx]
    bx = float(best["annualized_volatility"] * 100)
    by = float(best["annualized_return"] * 100)
    if np.isfinite(bx) and np.isfinite(by) and (bx != 0 or by != 0):
        x_end = max(float(vol_pct.max()) * 1.1, bx * 1.05, bx + 1.0)
        scale = x_end / bx if bx != 0 else 1.0
        ax.plot(
            [0, bx * scale],
            [0, by * scale],
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            alpha=0.75,
            zorder=2,
            label=f"Best return/risk ({_strategy_label(best['strategy'])})",
        )
        ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Risk vs Return", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    p = reports_dir / "strategy_risk_return.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 5. M2 classification summary
    if m2_metrics and "confusion_matrix" in m2_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        cm = np.array(m2_metrics["confusion_matrix"])
        im = axes[0].imshow(cm, cmap="Blues")
        axes[0].set_title("M2 Confusion Matrix (Test)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=axes[0], fraction=0.046)

        cls_names = ["Precision", "Recall", "F1", "AUC"]
        cls_vals = [
            m2_metrics.get("precision", 0),
            m2_metrics.get("recall", 0),
            m2_metrics.get("f1", 0),
            m2_metrics.get("auc", 0),
        ]
        axes[1].barh(cls_names, cls_vals, color="#8172B3")
        axes[1].set_xlim(0, 1)
        axes[1].set_title("M2 Test Metrics")
        for i, v in enumerate(cls_vals):
            axes[1].text(v + 0.02, i, _fmt_num(v, 2), va="center", fontsize=9)
        p = reports_dir / "m2_classification_summary.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    return saved


def save_figures(
    results: dict[str, BacktestResult],
    panel: pd.DataFrame,
    m2_metrics: dict[str, Any],
    ic_series: pd.Series,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # 1. Cumulative returns
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        cum = (1 + res.returns.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Cumulative Returns")
    p = figures_dir / "cumulative_returns.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 2. Drawdown
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        eq = (1 + res.returns.fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Drawdown")
    p = figures_dir / "drawdown.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 3. Rolling Sharpe
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        roll = res.returns.rolling(52).mean() / res.returns.rolling(52).std() * np.sqrt(WEEKS_PER_YEAR)
        ax.plot(roll.index, roll.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Rolling 52-week Sharpe")
    p = figures_dir / "rolling_sharpe.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 4. Rolling 12m max drawdown
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        rdd = rolling_max_drawdown(res.returns, 52)
        ax.plot(rdd.index, rdd.values, label=name)
    ax.legend(fontsize=8)
    ax.set_title("Rolling 52-week Max Drawdown")
    p = figures_dir / "rolling_max_drawdown.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 5. Turnover
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in results.items():
        ax.plot(res.turnover.index, res.turnover.values, label=name, alpha=0.7)
    ax.legend(fontsize=8)
    ax.set_title("Turnover")
    p = figures_dir / "turnover.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 6. Asset weights (m1_m2_linear if present)
    key = "m1_m2_linear" if "m1_m2_linear" in results else next(iter(results))
    w = results[key].weights
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in w.columns:
        ax.plot(w.index, w[col], label=col, alpha=0.7)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Asset Weights ({key})")
    p = figures_dir / "asset_weights.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))

    # 7. M1 signal heatmap
    df = panel.reset_index()
    if "M1_signal" in df.columns:
        sig = df.pivot(index="date", columns="ticker", values="M1_signal")
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(sig.T, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_yticks(range(len(sig.columns)))
        ax.set_yticklabels(sig.columns)
        ax.set_title("M1 Signal Heatmap")
        fig.colorbar(im, ax=ax)
        p = figures_dir / "m1_signal_heatmap.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 8. M2 probability histogram
    if "p_success" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["p_success"].dropna(), bins=30, edgecolor="black")
        ax.set_title("M2 Probability Histogram")
        p = figures_dir / "m2_probability_histogram.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 9. M2 calibration / ROC
    if m2_metrics and "auc" in m2_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].set_title("ROC placeholder")
        axes[1].set_title(f"AUC={m2_metrics.get('auc', 'n/a')}")
        p = figures_dir / "m2_roc_calibration.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 10. IC time series
    if not ic_series.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(ic_series.index, ic_series.values)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title("Information Coefficient Time Series")
        p = figures_dir / "ic_timeseries.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    return saved


def generate_final_report(
    metrics_table: pd.DataFrame,
    m2_metrics: dict[str, Any],
    report_path: Path,
    *,
    effective_start: str | None = None,
    effective_end: str | None = None,
    results: dict[str, BacktestResult] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = report_path.parent

    chart_files: list[str] = []
    if results is not None:
        chart_files = save_report_charts(results, m2_metrics, reports_dir)

    display_table = format_metrics_table_for_report(metrics_table)
    summary_lines = _build_executive_summary(metrics_table, m2_metrics)

    m2_display = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": _fmt_num(m2_metrics.get("accuracy", float("nan"))), "Meaning": "Share of correct meta-label predictions"},
            {"Metric": "Precision", "Value": _fmt_num(m2_metrics.get("precision", float("nan"))), "Meaning": "Approved trades that were actually profitable"},
            {"Metric": "Recall", "Value": _fmt_num(m2_metrics.get("recall", float("nan"))), "Meaning": "Profitable trades that M2 approved"},
            {"Metric": "F1 Score", "Value": _fmt_num(m2_metrics.get("f1", float("nan"))), "Meaning": "Balance of precision and recall"},
            {"Metric": "AUC", "Value": _fmt_num(m2_metrics.get("auc", float("nan"))), "Meaning": "Ranking quality of M2 probabilities"},
            {"Metric": "Brier Score", "Value": _fmt_num(m2_metrics.get("brier_score", float("nan"))), "Meaning": "Probability calibration error (lower is better)"},
            {
                "Metric": "Mean IC",
                "Value": _fmt_num(m2_metrics.get("information_coefficient_mean", float("nan"))),
                "Meaning": "Spearman rank correlation of M1 scores vs forward returns",
            },
        ]
    )

    lines = [
        "# Final Report: AI-Augmented Multi-Asset Meta-Labeling Pipeline",
        "",
        "## Executive Summary",
        "",
        *summary_lines,
        "",
        "## Sample Period",
        "",
        f"| Item | Value |",
        f"| --- | --- |",
        f"| Effective start | {effective_start or 'N/A'} |",
        f"| Effective end | {effective_end or 'N/A'} |",
        f"| Train period | 2006-01-01 to 2020-12-31 |",
        f"| Test period (M2 evaluation) | 2021-01-01 onward |",
        f"| Assets | SPY, TLT, GLD, VEA, VWO, HYG, VNQ |",
        "",
        "## Strategy Comparison",
        "",
        "All return and risk figures are **annualized** from weekly data. Benchmark for excess return and information ratio is **equal-weight 1/7**.",
        "",
        _markdown_table(display_table),
        "",
        "### How to read the metrics",
        "",
        "| Metric | Interpretation |",
        "| --- | --- |",
        "| **Ann. Return** | Geometric average yearly portfolio return after transaction costs |",
        "| **Ann. Volatility** | Standard deviation of weekly returns, scaled to a year |",
        "| **Sharpe** | Return per unit of risk (higher is better; assumes 0% risk-free rate) |",
        "| **Max Drawdown** | Largest peak-to-trough loss over the full sample |",
        "| **Excess vs EW** | Strategy return minus equal-weight benchmark return |",
        "| **Info Ratio** | Consistency of outperformance vs equal-weight (mean active return / tracking error) |",
        "| **Weekly Hit Rate** | Fraction of weeks with positive net strategy return |",
        "",
        "## Visual Summary",
        "",
    ]

    chart_descriptions = {
        "strategy_cumulative_returns.png": "Growth of $1 invested — compares benchmarks, M1-only, and meta-labeled strategies.",
        "strategy_drawdown.png": "Peak-to-trough losses over time — shows how deeply each strategy declined.",
        "strategy_sharpe_comparison.png": "Sharpe ratio bar chart — risk-adjusted performance at a glance.",
        "strategy_risk_return.png": "Return vs volatility scatter; dashed line from origin through the best return/risk (Sharpe) strategy.",
        "m2_classification_summary.png": "M2 confusion matrix and precision/recall on the out-of-sample test set.",
    }
    for chart in chart_files:
        desc = chart_descriptions.get(chart, "")
        lines.append(f"### {chart.replace('_', ' ').replace('.png', '').title()}")
        lines.append("")
        if desc:
            lines.append(desc)
            lines.append("")
        lines.append(f"![{chart}]({chart})")
        lines.append("")

    lines.extend(
        [
            "## M2 Meta-Labeling Quality (Test Set)",
            "",
            "M2 predicts whether each non-zero M1 signal will be profitable after a 4-week horizon. "
            "Metrics below are computed on **out-of-sample** dates from 2021 onward.",
            "",
            _markdown_table(m2_display),
            "",
        ]
    )

    if "confusion_matrix" in m2_metrics:
        cm = m2_metrics["confusion_matrix"]
        lines.extend(
            [
                "**Confusion matrix** (rows = actual, columns = predicted):",
                "",
                f"| | Predicted 0 | Predicted 1 |",
                f"| --- | ---: | ---: |",
                f"| Actual 0 | {cm[0][0]} | {cm[0][1]} |",
                f"| Actual 1 | {cm[1][0]} | {cm[1][1]} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Key Takeaways",
            "",
            "1. **Benchmarks** (equal-weight, 60/40) delivered the highest raw returns but with larger drawdowns.",
            "2. **M1-only** generated more active exposure with mixed results versus passive benchmarks.",
            "3. **M2 meta-labeling** improved Sharpe ratios mainly by **reducing position size** on low-confidence signals.",
            "4. M2 has **low recall** — it filters aggressively, trading less often but with better risk control.",
            "5. Results are **historical** and sensitive to data source quality (yfinance/FRED).",
            "",
            "## Pipeline Architecture",
            "",
            "```",
            "Raw Data → Ingest → Validation → Features → M1 → M2 → Sizing → Backtest → Diagnostics",
            "```",
            "",
            "## Look-Ahead Controls",
            "",
            "- Features use only data available at signal time (`shift(1)` on rolling windows)",
            "- Macro series lagged 4 weeks to approximate release delay",
            "- Strict chronological train/test split (train ≤ 2020, test ≥ 2021)",
            "- Label columns excluded from model feature matrices",
            "",
            "## AI / LLM Usage",
            "",
            "LLM-derived features are **disabled** in this run. See `runs/*/research_log.jsonl` for design decisions.",
            "",
            "## Limitations",
            "",
            "- yfinance and FRED are research-grade fallbacks, not institutional data",
            "- Partial FRED series may be missing when downloads time out",
            "- ETF backtests may include survivorship and product-history effects",
            "- Past performance does not predict future results",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


MODE_LABELS = {
    "long_only": "Long Only (no shorts)",
    "long_short": "Long / Short",
}


def _m2_metrics_table(m2_metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": _fmt_num(m2_metrics.get("accuracy", float("nan"))), "Meaning": "Share of correct meta-label predictions"},
            {"Metric": "Precision", "Value": _fmt_num(m2_metrics.get("precision", float("nan"))), "Meaning": "Approved trades that were actually profitable"},
            {"Metric": "Recall", "Value": _fmt_num(m2_metrics.get("recall", float("nan"))), "Meaning": "Profitable trades that M2 approved"},
            {"Metric": "F1 Score", "Value": _fmt_num(m2_metrics.get("f1", float("nan"))), "Meaning": "Balance of precision and recall"},
            {"Metric": "AUC", "Value": _fmt_num(m2_metrics.get("auc", float("nan"))), "Meaning": "Ranking quality of M2 probabilities"},
            {"Metric": "Brier Score", "Value": _fmt_num(m2_metrics.get("brier_score", float("nan"))), "Meaning": "Probability calibration error (lower is better)"},
            {
                "Metric": "Mean IC",
                "Value": _fmt_num(m2_metrics.get("information_coefficient_mean", float("nan"))),
                "Meaning": "Spearman rank correlation of M1 scores vs forward returns",
            },
        ]
    )


M1_SIGNAL_LABELS: dict[int, str] = {
    -1: "Short (−1)",
    0: "Flat (0)",
    1: "Long (+1)",
}


def analyze_m1_signal_m2_performance(
    panel: pd.DataFrame,
    threshold: float,
    *,
    period_label: str = "test",
) -> dict[str, Any]:
    """Summarize trade outcomes and M2 quality grouped by M1 signal (−1, 0, +1)."""
    df = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    if df.empty:
        return {"period": period_label, "threshold": threshold, "by_signal": pd.DataFrame()}

    rows: list[dict[str, Any]] = []
    for sig in sorted(df["M1_signal"].dropna().unique()):
        sig_int = int(sig)
        sub = df[df["M1_signal"] == sig_int]
        row: dict[str, Any] = {
            "m1_signal": sig_int,
            "signal_label": M1_SIGNAL_LABELS.get(sig_int, str(sig_int)),
            "observations": len(sub),
            "share_of_panel": len(sub) / len(df),
        }
        if sig_int == 0:
            rows.append(row)
            continue

        trades = sub[sub["meta_label"].notna() & sub["p_success"].notna()]
        row["labeled_trades"] = len(trades)
        if trades.empty:
            rows.append(row)
            continue

        approved = trades[trades["p_success"] >= threshold]
        rejected = trades[trades["p_success"] < threshold]

        row["m1_hit_rate"] = float(trades["meta_label"].mean())
        row["mean_trade_return"] = float(trades["trade_return"].mean())
        row["median_trade_return"] = float(trades["trade_return"].median())
        row["m2_approval_rate"] = float(len(approved) / len(trades))
        row["hit_rate_m2_approved"] = float(approved["meta_label"].mean()) if len(approved) else float("nan")
        row["hit_rate_m2_rejected"] = float(rejected["meta_label"].mean()) if len(rejected) else float("nan")
        row["mean_return_m2_approved"] = float(approved["trade_return"].mean()) if len(approved) else float("nan")
        row["mean_return_m2_rejected"] = float(rejected["trade_return"].mean()) if len(rejected) else float("nan")

        m2_group = m2_classification_metrics(trades["meta_label"], trades["p_success"], threshold)
        row["m2_accuracy"] = m2_group.get("accuracy", float("nan"))
        row["m2_precision"] = m2_group.get("precision", float("nan"))
        row["m2_recall"] = m2_group.get("recall", float("nan"))
        row["m2_f1"] = m2_group.get("f1", float("nan"))
        row["m2_auc"] = m2_group.get("auc", float("nan"))
        rows.append(row)

    by_signal = pd.DataFrame(rows)
    return {"period": period_label, "threshold": threshold, "by_signal": by_signal}


def format_m1_signal_analysis_table(analysis: dict[str, Any]) -> pd.DataFrame:
    """Display table for M1-signal-grouped M2 analysis."""
    df = analysis.get("by_signal", pd.DataFrame())
    if df.empty:
        return pd.DataFrame()

    display = pd.DataFrame(
        {
            "M1 Signal": df["signal_label"],
            "Observations": df["observations"],
            "Share": df["share_of_panel"].map(lambda x: _fmt_pct(x)),
            "Labeled Trades": df.get("labeled_trades", pd.Series(dtype=float)).fillna(0).astype(int),
            "M1 Hit Rate": df.get("m1_hit_rate", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "Mean Trade Return": df.get("mean_trade_return", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "M2 Approval Rate": df.get("m2_approval_rate", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "Hit Rate (M2 Approved)": df.get("hit_rate_m2_approved", pd.Series(dtype=float)).map(lambda x: _fmt_pct(x)),
            "M2 Precision": df.get("m2_precision", pd.Series(dtype=float)).map(lambda x: _fmt_num(x)),
            "M2 Recall": df.get("m2_recall", pd.Series(dtype=float)).map(lambda x: _fmt_num(x)),
            "M2 F1": df.get("m2_f1", pd.Series(dtype=float)).map(lambda x: _fmt_num(x)),
        }
    )
    return display


def save_m1_signal_m2_chart(analysis: dict[str, Any], output_path: Path) -> str:
    """Visualize M1 signal grouping and M2 filtering on the test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = analysis.get("by_signal", pd.DataFrame())
    if df.empty:
        return ""

    active = df[df["m1_signal"] != 0].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: observation counts by M1 signal
    ax = axes[0, 0]
    colors_count = {1: "#55A868", 0: "#CCCCCC", -1: "#C44E52"}
    bar_colors = [colors_count.get(int(s), "#888888") for s in df["m1_signal"]]
    ax.bar(df["signal_label"], df["observations"], color=bar_colors)
    ax.set_title("Observations by M1 Signal", fontweight="bold")
    ax.set_ylabel("Count (asset-weeks)")
    ax.tick_params(axis="x", rotation=15)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["observations"], f"{int(row['observations'])}", ha="center", va="bottom", fontsize=8)

    # Panel B: mean trade return for active signals
    ax = axes[0, 1]
    if not active.empty and "mean_trade_return" in active.columns:
        ret_pct = active["mean_trade_return"].fillna(0) * 100
        bar_colors_a = [colors_count.get(int(s), "#888888") for s in active["m1_signal"]]
        bars = ax.bar(active["signal_label"], ret_pct, color=bar_colors_a)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title("Mean Forward Trade Return by M1 Signal", fontweight="bold")
        ax.set_ylabel("Mean return (%)")
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, ret_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )

    # Panel C: M1 hit rate vs M2-approved hit rate
    ax = axes[1, 0]
    if not active.empty:
        x = np.arange(len(active))
        width = 0.35
        m1_hr = active["m1_hit_rate"].fillna(0) * 100
        m2_hr = active["hit_rate_m2_approved"].fillna(0) * 100
        ax.bar(x - width / 2, m1_hr, width, label="M1 (all trades)", color="#4C72B0")
        ax.bar(x + width / 2, m2_hr, width, label="M2 approved only", color="#8172B3")
        ax.set_xticks(x)
        ax.set_xticklabels(active["signal_label"], rotation=15, ha="right")
        ax.set_ylabel("Hit rate (%)")
        ax.set_title("Profitability: M1 vs M2-Filtered", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(100, float(m1_hr.max()) * 1.15, float(m2_hr.max()) * 1.15))

    # Panel D: M2 classification metrics by active signal group
    ax = axes[1, 1]
    if not active.empty:
        metric_names = ["m2_precision", "m2_recall", "m2_f1", "m2_auc"]
        metric_labels = ["Precision", "Recall", "F1", "AUC"]
        x = np.arange(len(active))
        n_metrics = len(metric_names)
        width = 0.8 / n_metrics
        palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
        for j, (col, label) in enumerate(zip(metric_names, metric_labels)):
            if col not in active.columns:
                continue
            offset = (j - (n_metrics - 1) / 2) * width
            vals = active[col].fillna(0)
            ax.bar(x + offset, vals, width, label=label, color=palette[j % len(palette)])
        ax.set_xticks(x)
        ax.set_xticklabels(active["signal_label"], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("M2 Classifier Quality by M1 Signal", fontweight="bold")
        ax.legend(fontsize=7, ncol=2)

    threshold = analysis.get("threshold", 0.5)
    fig.suptitle(
        f"M2 Performance by M1 Signal Group (test set, M2 threshold={threshold})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    chart_name = output_path.name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_name


def save_m1_signal_m2_mode_comparison_chart(
    mode_analyses: list[tuple[str, dict[str, Any]]],
    output_path: Path,
) -> str:
    """Compare long-only vs long/short M1 signal outcomes and M2 filtering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mode_colors = {"long_only": "#4C72B0", "long_short": "#C44E52"}

    # Left: mean trade return by signal and mode
    ax = axes[0]
    plot_rows = []
    for mode_name, analysis in mode_analyses:
        df = analysis.get("by_signal", pd.DataFrame())
        active = df[df["m1_signal"] != 0]
        for _, row in active.iterrows():
            plot_rows.append(
                {
                    "mode": MODE_LABELS.get(mode_name, mode_name),
                    "mode_key": mode_name,
                    "signal": row["signal_label"],
                    "mean_return_pct": float(row.get("mean_trade_return", 0) or 0) * 100,
                }
            )
    if plot_rows:
        plot_df = pd.DataFrame(plot_rows)
        signals = plot_df["signal"].unique()
        modes = [MODE_LABELS.get(m, m) for m, _ in mode_analyses]
        x = np.arange(len(signals))
        width = 0.8 / max(len(modes), 1)
        for i, (mode_name, _) in enumerate(mode_analyses):
            mode_label = MODE_LABELS.get(mode_name, mode_name)
            subset = plot_df[plot_df["mode"] == mode_label]
            vals = [subset.loc[subset["signal"] == s, "mean_return_pct"].iloc[0] if s in subset["signal"].values else 0 for s in signals]
            offset = (i - (len(modes) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=mode_label, color=mode_colors.get(mode_name, "#888888"))
        ax.set_xticks(x)
        ax.set_xticklabels(signals, rotation=15, ha="right")
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_ylabel("Mean trade return (%)")
        ax.set_title("M1 Trade Return by Signal & Mode", fontweight="bold")
        ax.legend(fontsize=8)

    # Right: M1 vs M2-approved hit rate for Long (+1) across modes
    ax = axes[1]
    hit_rows = []
    for mode_name, analysis in mode_analyses:
        df = analysis.get("by_signal", pd.DataFrame())
        for sig_val, label in [(1, "Long (+1)"), (-1, "Short (−1)")]:
            row = df[df["m1_signal"] == sig_val]
            if row.empty or pd.isna(row.iloc[0].get("m1_hit_rate", np.nan)):
                continue
            hit_rows.append(
                {
                    "group": f"{label}\n({MODE_LABELS.get(mode_name, mode_name)})",
                    "mode_key": mode_name,
                    "m1_hit": float(row.iloc[0]["m1_hit_rate"]) * 100,
                    "m2_hit": float(row.iloc[0].get("hit_rate_m2_approved", 0) or 0) * 100,
                }
            )
    if hit_rows:
        hit_df = pd.DataFrame(hit_rows)
        x = np.arange(len(hit_df))
        width = 0.35
        ax.bar(x - width / 2, hit_df["m1_hit"], width, label="M1 all trades", color="#4C72B0")
        ax.bar(x + width / 2, hit_df["m2_hit"], width, label="M2 approved", color="#8172B3")
        ax.set_xticks(x)
        ax.set_xticklabels(hit_df["group"], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Hit rate (%)")
        ax.set_title("Hit Rate: M1 vs M2-Filtered by Mode", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)

    fig.suptitle("M2 Exploration: M1 Signal Groups (Long-Only vs Long/Short)", fontweight="bold", y=1.02)
    fig.tight_layout()
    chart_name = output_path.name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_name


def build_m1_signal_m2_report_section(
    mode_results: list[Any],
    *,
    comparison_chart: str | None = None,
) -> list[str]:
    """Markdown section for M1-signal-grouped M2 analysis."""
    lines = [
        "## M2 Performance by M1 Signal",
        "",
        "M1 outputs three signal types per asset-week: **short (−1)**, **flat (0)**, or **long (+1)**. "
        "M2 only trains and predicts on non-zero signals. Below we break out **test-set** trade outcomes "
        "and classifier quality within each M1 group.",
        "",
        "- **M1 hit rate**: share of trades with positive forward return (after cost hurdle)",
        "- **M2 approval rate**: share of trades where `p_success` ≥ threshold",
        "- **Hit rate (M2 approved)**: profitability among trades M2 kept",
        "",
    ]
    if comparison_chart:
        lines.extend(
            [
                "### Long-Only vs Long/Short Comparison",
                "",
                f"![M2 by M1 signal comparison]({comparison_chart})",
                "",
                "*Left: mean forward trade return by M1 signal. Right: M1 vs M2-filtered hit rates "
                "(long-only has no short bucket).*",
                "",
            ]
        )

    for mode in mode_results:
        analysis = getattr(mode, "m1_signal_analysis", None)
        if not analysis:
            continue
        label = MODE_LABELS.get(mode.mode_name, mode.mode_name)
        chart_name = getattr(mode, "m1_signal_chart", None)
        lines.extend(
            [
                f"### {label}",
                "",
                f"`allow_short={mode.allow_short}` — M2 threshold = {analysis.get('threshold', 'N/A')}",
                "",
            ]
        )
        table = format_m1_signal_analysis_table(analysis)
        if not table.empty:
            lines.append(_markdown_table(table))
            lines.append("")
        chart_rel = getattr(mode, "m1_signal_chart_rel", None)
        if chart_rel:
            lines.append(f"![M2 by M1 signal — {label}]({chart_rel})")
        elif chart_name:
            lines.append(f"![M2 by M1 signal — {label}]({chart_name})")
            lines.append("")
        if mode.mode_name == "long_only":
            lines.extend(
                [
                    "*Long-only mode: M1 never emits −1; shorts are disabled at the signal layer.*",
                    "",
                ]
            )

    return lines


def save_m1_mode_comparison_chart(mode_results: list[Any], reports_dir: Path) -> str:
    """Chart comparing M1-only performance across long-only vs long-short modes."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    colors = {"long_only": "#4C72B0", "long_short": "#C44E52"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for mode in mode_results:
        name = mode.mode_name
        res = mode.results["m1_only"]
        cum = (1 + res.returns.fillna(0)).cumprod()
        axes[0].plot(
            cum.index,
            cum.values,
            label=MODE_LABELS.get(name, name),
            color=colors.get(name, "#888888"),
            linewidth=2,
        )
    axes[0].set_title("M1 Only — Cumulative Growth of $1", fontweight="bold")
    axes[0].set_ylabel("Portfolio value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rows = []
    for mode in mode_results:
        m1_row = mode.metrics_table.set_index("strategy").loc["m1_only"]
        rows.append(
            {
                "Mode": MODE_LABELS.get(mode.mode_name, mode.mode_name),
                "Ann. Return (%)": m1_row["annualized_return"] * 100,
                "Sharpe": m1_row["sharpe"],
                "Max DD (%)": m1_row["max_drawdown"] * 100,
            }
        )
    compare = pd.DataFrame(rows)
    x = np.arange(len(compare))
    width = 0.25
    axes[1].bar(x - width, compare["Ann. Return (%)"], width, label="Ann. Return %", color="#55A868")
    axes[1].bar(x, compare["Sharpe"] * 10, width, label="Sharpe (×10)", color="#8172B3")
    axes[1].bar(x + width, compare["Max DD (%)"].abs(), width, label="|Max DD| %", color="#CCB974")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(compare["Mode"], rotation=15, ha="right")
    axes[1].set_title("M1 Only — Key Metrics by Mode", fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)

    chart_name = "m1_mode_comparison.png"
    fig.tight_layout()
    fig.savefig(reports_dir / chart_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_name


def build_performance_parameters_section(cfg: PipelineConfig) -> list[str]:
    """Document tunable parameters that influence strategy performance."""
    m1 = cfg.m1
    w = m1.weights
    test_end_disp = cfg.split.test_end or "latest (open-ended)"
    return [
        "## Configuration Parameters Affecting Performance",
        "",
        "The pipeline reads defaults from `config/config.yaml`. **Split dates** can also be set "
        "at runtime without editing the file (see CLI below). Other parameters require config edits.",
        "",
        "### Train / Test Split",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `split.data_start` | {cfg.data_start_resolved()} | Earliest downloaded price date (can precede train for feature warmup) |",
        f"| `split.train_start` | {cfg.split.train_start} | Intended train window start (clipped to effective panel start) |",
        f"| `split.train_end` | {cfg.split.train_end} | Last in-sample date; **primary knob for tuning in-sample fit** |",
        f"| `split.test_start` | {cfg.split.test_start} | Out-of-sample evaluation begins here (M2 metrics, IC, reported Sharpe) |",
        f"| `split.test_end` | {test_end_disp} | Optional cap on the evaluation window |",
        f"| `split.require_full_universe` | {cfg.split.require_full_universe} | If true, only weeks with all 7 ETFs (~2007+); if false, partial groups allowed |",
        "",
        "**Can train_start be before 2006?** Yes in config/CLI, but with `require_full_universe: true` "
        "(default) the **effective** sample usually starts ~**2007-07** when VEA and HYG (youngest ETFs) "
        "both exist. Dates before that are dropped. Set `require_full_universe: false` or `--partial-universe` "
        "to train on subsets (e.g. SPY/TLT/GLD/VNQ/VWO from 2005).",
        "",
        "**CLI overrides** (ISO dates, applied after loading config):",
        "",
        "```bash",
        "# Shorter/longer train, earlier/later test — compare Sharpe in reports/final_report.md",
        "python -m src.run_pipeline --train-end 2018-12-31 --test-start 2019-01-01",
        "python -m src.run_pipeline --train-end 2015-12-31 --test-start 2016-01-01",
        "python -m src.run_pipeline --train-start 2008-01-01 --train-end 2012-12-31 --test-start 2013-01-01",
        "",
        "# Earlier history: partial universe before all seven ETFs existed",
        "python -m src.run_pipeline --data-start 2004-01-01 --train-start 2005-01-01 --train-end 2006-12-31 "
        "--test-start 2007-01-01 --partial-universe --refresh-data",
        "```",
        "",
        "Shorter train windows reduce overfitting risk but give fewer M2 labels; varying `train_end` is the "
        "fastest way to test whether performance is stable across in-sample cutoffs.",
        "",
        "### M1 Rule-Based Side Model",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `models.m1.weights` | momentum={w['momentum']}, trend={w['trend']}, macro={w['macro']}, risk={w['risk_penalty']} | Relative importance of factor families in the composite score |",
        f"| `models.m1.optimize_thresholds` | {m1.optimize_thresholds} | When true, long/short cutoffs are tuned on the train set only |",
        f"| `models.m1.long_quantile` / `short_quantile` | {m1.long_quantile} / {m1.short_quantile} | Starting quantiles for threshold search (higher long quantile → fewer longs) |",
        f"| `models.m1.allow_short` | {m1.allow_short} | Default shorting flag; pipeline always runs both long-only and long/short modes |",
        f"| `models.m1.asset_class_tilts` | {m1.asset_class_tilts} | Macro tilts by asset class (equity, bonds, credit, gold, REIT) |",
        "",
        "### M2 Meta-Labeling",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `models.m2.threshold` | {cfg.m2.threshold} | Minimum P(success) to take full size; higher → fewer trades, often lower turnover |",
        f"| `models.m2.calibrate` | {cfg.m2.calibrate} | Probability calibration on train data; improves threshold interpretability |",
        f"| `models.m2.type` | {cfg.m2.type} | Classifier used for meta-labels |",
        "",
        "### Labels (M1 targets & M2 supervision)",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `labels.horizon_weeks` | {cfg.labels.horizon_weeks} | Forward return horizon for profitability labels |",
        f"| `labels.positive_threshold` | {cfg.labels.positive_threshold} | Minimum forward return to label a long as successful |",
        f"| `labels.negative_threshold` | {cfg.labels.negative_threshold} | Forward return threshold for short success |",
        f"| `labels.transaction_cost_threshold` | {cfg.labels.transaction_cost_threshold} | Cost hurdle embedded in label construction |",
        "",
        "### Portfolio & Costs",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `portfolio.transaction_cost_bps` | {cfg.portfolio.transaction_cost_bps} | Round-trip cost per unit turnover; higher values drag net returns |",
        f"| `portfolio.max_gross_exposure` | {cfg.portfolio.max_gross_exposure} | Cap on sum of absolute weights |",
        f"| `portfolio.max_abs_asset_weight` | {cfg.portfolio.max_abs_asset_weight} | Per-asset weight ceiling |",
        f"| `portfolio.sizing_mode` | {cfg.portfolio.sizing_mode} | How M2 probability maps to position size (binary / linear / ecdf) |",
        "",
        "### Features",
        "",
        "| Parameter | Current value | Performance impact |",
        "| --- | --- | --- |",
        f"| `features.momentum_windows` | {cfg.features.momentum_windows} | Lookback weeks for momentum factors |",
        f"| `features.macro_lag_weeks` | {cfg.features.macro_lag_weeks} | Release lag applied to macro series (reduces look-ahead) |",
        f"| `features.winsorize_pct` | {cfg.features.winsorize_pct} | Train-set winsorization of extreme feature values |",
        "",
    ]


def generate_dual_mode_report(
    mode_results: list[Any],
    report_path: Path,
    *,
    final_dir: Path | None = None,
    mode_comparison_dir: Path | None = None,
    cfg: PipelineConfig | None = None,
    effective_start: str | None = None,
    effective_end: str | None = None,
    asset_analysis_sections: list[str] | None = None,
) -> None:
    """Build a final report comparing long-only and long-short M1 runs."""
    reports_root = report_path.parent
    report_path.parent.mkdir(parents=True, exist_ok=True)
    final_dir = final_dir or reports_root / "final"
    mode_comparison_dir = mode_comparison_dir or reports_root / "mode_comparison"
    final_dir.mkdir(parents=True, exist_ok=True)
    mode_comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_chart = f"mode_comparison/{save_m1_mode_comparison_chart(mode_results, mode_comparison_dir)}"
    m1_m2_comparison_chart = (
        f"mode_comparison/{save_m1_signal_m2_mode_comparison_chart(
            [(m.mode_name, m.m1_signal_analysis) for m in mode_results if getattr(m, 'm1_signal_analysis', None)],
            mode_comparison_dir / 'm2_m1_signal_comparison.png',
        )}"
    )

    for mode in mode_results:
        analysis = getattr(mode, "m1_signal_analysis", None)
        if analysis:
            mode_chart_dir = final_dir / mode.mode_name
            mode_chart_dir.mkdir(parents=True, exist_ok=True)
            chart_path = mode_chart_dir / "m2_m1_signal_analysis.png"
            save_m1_signal_m2_chart(analysis, chart_path)
            mode.m1_signal_chart = chart_path.name
            mode.m1_signal_chart_rel = f"final/{mode.mode_name}/{chart_path.name}"

    lines = [
        "# Final Report: AI-Augmented Multi-Asset Meta-Labeling Pipeline",
        "",
        "This run executes the pipeline **twice**: once with M1 **long-only** (no short signals) "
        "and once with M1 **long/short** enabled.",
        "",
        "**Research use only — not investment advice.**",
        "",
        "## Sample Period",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Effective start | {effective_start or 'N/A'} |",
        f"| Effective end | {effective_end or 'N/A'} |",
    ]
    if cfg is not None:
        test_end_disp = cfg.split.test_end or "latest"
        universe = "all 7 ETFs each week" if cfg.split.require_full_universe else "partial (per-ticker availability)"
        lines.extend(
            [
                f"| Data download from | {cfg.data_start_resolved()} |",
                f"| Train period (requested) | {cfg.split.train_start} to {cfg.split.train_end} |",
                f"| Test period (M2 evaluation) | {cfg.split.test_start} to {test_end_disp} |",
                f"| Universe mode | {universe} |",
                f"| Assets | {', '.join(cfg.assets.tickers)} |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"| Train period | (see config) |",
                f"| Test period (M2 evaluation) | (see config) |",
                f"| Assets | SPY, TLT, GLD, VEA, VWO, HYG, VNQ |",
                "",
            ]
        )

    if cfg is not None:
        lines.extend(build_performance_parameters_section(cfg))

    if asset_analysis_sections:
        lines.extend(asset_analysis_sections)

    lines.extend(build_m1_signal_m2_report_section(mode_results, comparison_chart=m1_m2_comparison_chart))

    lines.extend(
        [
        "## M1 Mode Comparison (M1 Only)",
        "",
        "| Mode | Ann. Return | Sharpe | Max Drawdown |",
        "| --- | --- | --- | --- |",
        ]
    )

    for mode in mode_results:
        m1_row = mode.metrics_table.set_index("strategy").loc["m1_only"]
        label = MODE_LABELS.get(mode.mode_name, mode.mode_name)
        lines.append(
            f"| {label} | {_fmt_pct(m1_row['annualized_return'])} | {_fmt_num(m1_row['sharpe'])} | {_fmt_pct(m1_row['max_drawdown'])} |"
        )

    lines.extend(
        [
            "",
            f"![M1 mode comparison]({comparison_chart})",
            "",
            "*Left: cumulative M1-only returns. Right: return, Sharpe (×10), and drawdown by mode.*",
            "",
        ]
    )

    for mode in mode_results:
        label = MODE_LABELS.get(mode.mode_name, mode.mode_name)
        chart_prefix = f"final/{mode.mode_name}/"
        save_report_charts(mode.results, mode.m2_metrics, final_dir, subdir=mode.mode_name)
        display_table = format_metrics_table_for_report(mode.metrics_table)

        lines.extend(
            [
                f"## Results: {label}",
                "",
                f"`allow_short={mode.allow_short}` — outputs in `data/backtests/{mode.mode_name}/`",
                "",
                _markdown_table(display_table),
                "",
                f"### Charts ({label})",
                "",
            ]
        )
        for chart in [
            "strategy_cumulative_returns.png",
            "strategy_drawdown.png",
            "strategy_sharpe_comparison.png",
            "strategy_risk_return.png",
            "m2_classification_summary.png",
            "m2_m1_signal_analysis.png",
        ]:
            lines.append(f"![{chart}]({chart_prefix}{chart})")
            lines.append("")

        lines.extend(
            [
                f"### M2 Quality — {label} (Test Set)",
                "",
                _markdown_table(_m2_metrics_table(mode.m2_metrics)),
                "",
            ]
        )

    lines.extend(
        [
            "### How to read the metrics",
            "",
            "| Metric | Interpretation |",
            "| --- | --- |",
            "| **Ann. Return** | Geometric average yearly portfolio return after transaction costs |",
            "| **Ann. Volatility** | Standard deviation of weekly returns, scaled to a year |",
            "| **Sharpe** | Return per unit of risk (higher is better; assumes 0% risk-free rate) |",
            "| **Max Drawdown** | Largest peak-to-trough loss over the full sample |",
            "| **Excess vs EW** | Strategy return minus equal-weight benchmark return |",
            "| **Info Ratio** | Consistency of outperformance vs equal-weight |",
            "| **Weekly Hit Rate** | Fraction of weeks with positive net strategy return |",
            "",
            "## Key Takeaways",
            "",
            "1. **Long-only M1** avoids short exposure, which often hurts in upward-trending ETF samples.",
            "2. **Long/short M1** can increase activity but shorts may reduce returns if poorly timed.",
            "3. **M2 meta-labeling** adjusts position size on top of whichever M1 mode is used.",
            "4. Compare both modes above to see whether shorts add value in this universe.",
            "",
            "## Look-Ahead Controls",
            "",
            "- Features use only data available at signal time (`shift(1)` on rolling windows)",
            f"- Macro series lagged {cfg.features.macro_lag_weeks} weeks to approximate release delay"
            if cfg is not None
            else "- Macro series lagged (see config features.macro_lag_weeks) to approximate release delay",
            f"- Strict chronological train/test split (train {cfg.split.train_start}–{cfg.split.train_end}, "
            f"test {cfg.split.test_start}–{cfg.split.test_end or 'latest'})"
            if cfg is not None
            else "- Strict chronological train/test split (see config split section)",
            "",
            "## Limitations",
            "",
            "- yfinance and FRED are research-grade fallbacks, not institutional data",
            "- Past performance does not predict future results",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def run_diagnostics(
    results: dict[str, BacktestResult],
    panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    cfg_threshold: float,
    output_dir: Path,
) -> dict[str, Any]:
    metrics_table = build_metrics_table(results)
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False)

    ic = compute_ic(test_panel)
    ic_mean = float(ic.mean()) if not ic.empty else float("nan")

    m2_metrics = m2_classification_metrics(
        test_panel["meta_label"],
        test_panel["p_success"],
        threshold=cfg_threshold,
    )
    if not ic.empty:
        m2_metrics["information_coefficient_mean"] = ic_mean

    m1_signal_analysis = analyze_m1_signal_m2_performance(
        test_panel, cfg_threshold, period_label="test"
    )
    m1_signal_analysis["by_signal"].to_csv(output_dir / "m1_signal_m2_analysis.csv", index=False)

    figures_dir = output_dir / "figures"
    save_figures(results, test_panel, m2_metrics, ic, figures_dir)
    save_m1_signal_m2_chart(m1_signal_analysis, figures_dir / "m2_m1_signal_analysis.png")

    summary = {
        "metrics_table": metrics_table.to_dict(orient="records"),
        "m2_metrics": m2_metrics,
        "m1_signal_analysis": m1_signal_analysis,
        "ic_mean": ic_mean,
    }
    json_summary = {
        **summary,
        "m1_signal_analysis": {
            "period": m1_signal_analysis["period"],
            "threshold": m1_signal_analysis["threshold"],
            "by_signal": m1_signal_analysis["by_signal"].to_dict(orient="records"),
        },
    }
    with (output_dir / "diagnostics_summary.json").open("w") as f:
        json.dump(json_summary, f, indent=2, default=str)

    return summary
