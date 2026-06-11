"""Individual asset buy-and-hold analysis and data component documentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.diagnostics import (
    _fmt_num,
    _fmt_pct,
    _markdown_table,
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)

ASSET_CATALOG: dict[str, dict[str, str]] = {
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "benchmark": "S&P 500 (proxy)",
        "asset_class": "U.S. Equities",
        "role": "U.S. large-cap equity beta and growth exposure",
        "source": "yfinance — adjusted close, weekly",
    },
    "TLT": {
        "name": "iShares 20+ Year Treasury Bond ETF",
        "benchmark": "Long-duration U.S. Treasuries",
        "asset_class": "Government Bonds",
        "role": "Duration and defensive interest-rate exposure",
        "source": "yfinance — adjusted close, weekly",
    },
    "GLD": {
        "name": "SPDR Gold Shares",
        "benchmark": "Gold spot price (proxy)",
        "asset_class": "Commodities / Gold",
        "role": "Inflation hedge and safe-haven commodity exposure",
        "source": "yfinance — adjusted close, weekly",
    },
    "VEA": {
        "name": "Vanguard FTSE Developed Markets ETF",
        "benchmark": "Developed ex-U.S. equities",
        "asset_class": "International Equities",
        "role": "Geographic diversification outside the U.S.",
        "source": "yfinance — adjusted close, weekly",
    },
    "VWO": {
        "name": "Vanguard FTSE Emerging Markets ETF",
        "benchmark": "Emerging market equities",
        "asset_class": "Emerging Market Equities",
        "role": "Emerging market growth and risk premia",
        "source": "yfinance — adjusted close, weekly",
    },
    "HYG": {
        "name": "iShares iBoxx High Yield Corporate Bond ETF",
        "benchmark": "U.S. high-yield corporate bonds",
        "asset_class": "Credit / High Yield",
        "role": "Credit risk and income exposure",
        "source": "yfinance — adjusted close, weekly",
    },
    "VNQ": {
        "name": "Vanguard Real Estate ETF",
        "benchmark": "U.S. REITs",
        "asset_class": "Real Estate (REITs)",
        "role": "Real estate and rate-sensitive income exposure",
        "source": "yfinance — adjusted close, weekly",
    },
}

MACRO_CATALOG: dict[str, dict[str, str]] = {
    "CPIAUCSL": {
        "name": "Consumer Price Index",
        "role": "Inflation trend and regime indicator",
        "source": "FRED — lagged 4 weeks in features",
    },
    "UNRATE": {
        "name": "Unemployment Rate",
        "role": "Labor market / growth proxy",
        "source": "FRED — lagged 4 weeks in features",
    },
    "INDPRO": {
        "name": "Industrial Production Index",
        "role": "Economic growth proxy",
        "source": "FRED — lagged 4 weeks in features",
    },
    "FEDFUNDS": {
        "name": "Federal Funds Rate",
        "role": "Monetary policy stance",
        "source": "FRED — lagged 4 weeks in features",
    },
    "DGS10": {
        "name": "10-Year Treasury Yield",
        "role": "Long-term interest rate level",
        "source": "FRED — lagged 4 weeks in features",
    },
    "T10Y2Y": {
        "name": "10Y–2Y Treasury Spread",
        "role": "Yield curve slope / recession signal",
        "source": "FRED — lagged 4 weeks in features",
    },
    "BAA10Y": {
        "name": "Baa–10Y Credit Spread",
        "role": "Credit stress indicator",
        "source": "FRED — lagged 4 weeks in features",
    },
    "VIX": {
        "name": "CBOE Volatility Index",
        "role": "Equity risk sentiment (risk-on / risk-off)",
        "source": "yfinance (^VIX) — used in features, not traded",
    },
}


@dataclass
class AssetAnalysisResult:
    metrics_full: pd.DataFrame
    metrics_train: pd.DataFrame
    metrics_test: pd.DataFrame
    returns_wide: pd.DataFrame
    effective_start: str | None
    effective_end: str | None
    macro_series_loaded: list[str]
    train_period_label: str
    test_period_label: str


@dataclass
class StrategyOverlay:
    """Portfolio strategy series to compare against individual asset buy-and-hold."""

    label: str
    returns: pd.Series
    annualized_return: float
    sharpe: float


MODE_LABELS_SHORT = {
    "long_only": "Long Only",
    "long_short": "Long/Short",
}


def strategy_overlays_from_mode_results(mode_results: list[Any]) -> list[StrategyOverlay]:
    """Build M1 and M1+M2 strategy overlays from dual-mode pipeline results."""
    overlays: list[StrategyOverlay] = []
    for mode in mode_results:
        mode_label = MODE_LABELS_SHORT.get(mode.mode_name, mode.mode_name)
        for key, short_name in (("m1_only", "M1"), ("m1_m2_linear", "M1+M2")):
            result = mode.results.get(key)
            if result is None:
                continue
            m = _metrics_for_returns(result.returns)
            overlays.append(
                StrategyOverlay(
                    label=f"{short_name} ({mode_label})",
                    returns=result.returns,
                    annualized_return=m["annualized_return"],
                    sharpe=m["sharpe"],
                )
            )
    return overlays


def _metrics_for_returns(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "total_return": float("nan"),
            "weekly_hit_rate": float("nan"),
        }
    return {
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "total_return": float((1 + r).prod() - 1),
        "weekly_hit_rate": float((r > 0).mean()),
    }


def compute_asset_buy_hold_metrics(
    returns_wide: pd.DataFrame,
    tickers: list[str],
    *,
    period_label: str = "full",
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        if ticker not in returns_wide.columns:
            continue
        m = _metrics_for_returns(returns_wide[ticker])
        meta = ASSET_CATALOG.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "period": period_label,
                "name": meta.get("name", ticker),
                "asset_class": meta.get("asset_class", ""),
                "benchmark": meta.get("benchmark", ""),
                **m,
            }
        )
    return pd.DataFrame(rows)


def build_asset_analysis(
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    effective_start: str | None = None,
    effective_end: str | None = None,
    macro_series_loaded: list[str] | None = None,
) -> AssetAnalysisResult:
    tickers = cfg.assets.tickers
    train_start = pd.Timestamp(cfg.split.train_start)
    train_end = pd.Timestamp(cfg.split.train_end)
    test_start = pd.Timestamp(cfg.split.test_start)
    train_mask = (returns_wide.index >= train_start) & (returns_wide.index <= train_end)
    test_mask = returns_wide.index >= test_start
    if cfg.split.test_end is not None:
        test_mask &= returns_wide.index <= pd.Timestamp(cfg.split.test_end)

    train_label = f"{cfg.split.train_start} to {cfg.split.train_end}"
    test_label = f"{cfg.split.test_start} to {cfg.split.test_end or 'latest'}"

    full = compute_asset_buy_hold_metrics(returns_wide, tickers, period_label="full")
    train = compute_asset_buy_hold_metrics(
        returns_wide.loc[train_mask], tickers, period_label="train"
    )
    test = compute_asset_buy_hold_metrics(
        returns_wide.loc[test_mask], tickers, period_label="test"
    )

    return AssetAnalysisResult(
        metrics_full=full,
        metrics_train=train,
        metrics_test=test,
        returns_wide=returns_wide,
        effective_start=effective_start,
        effective_end=effective_end,
        macro_series_loaded=macro_series_loaded or list(cfg.macro.fred_series) + ["VIX"],
        train_period_label=train_label,
        test_period_label=test_label,
    )


def format_asset_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": df["ticker"],
            "Asset": df["name"],
            "Class": df["asset_class"],
            "Ann. Return": df["annualized_return"].map(lambda x: _fmt_pct(x)),
            "Ann. Volatility": df["annualized_volatility"].map(lambda x: _fmt_pct(x)),
            "Sharpe": df["sharpe"].map(lambda x: _fmt_num(x)),
            "Max Drawdown": df["max_drawdown"].map(lambda x: _fmt_pct(x)),
            "Total Return": df["total_return"].map(lambda x: _fmt_pct(x)),
            "Weekly Hit Rate": df["weekly_hit_rate"].map(lambda x: _fmt_pct(x)),
        }
    )


def save_asset_analysis_charts(
    analysis: AssetAnalysisResult,
    reports_dir: Path,
    *,
    strategy_overlays: list[StrategyOverlay] | None = None,
) -> list[str]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    returns = analysis.returns_wide
    metrics = analysis.metrics_full
    strategies = strategy_overlays or []

    palette = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    strategy_styles = [
        {"color": "#333333", "linestyle": "--", "linewidth": 2.4},
        {"color": "#111111", "linestyle": "-", "linewidth": 2.4},
        {"color": "#C44E52", "linestyle": "--", "linewidth": 2.4},
        {"color": "#E24A4A", "linestyle": "-", "linewidth": 2.4},
    ]

    # 1. Cumulative buy-and-hold per asset + strategy portfolios
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ticker in enumerate(metrics["ticker"]):
        if ticker not in returns.columns:
            continue
        cum = (1 + returns[ticker].fillna(0)).cumprod()
        label = ASSET_CATALOG.get(ticker, {}).get("benchmark", ticker)
        ax.plot(cum.index, cum.values, label=f"{ticker} ({label})", color=palette[i], linewidth=1.8, alpha=0.9)
    for i, strat in enumerate(strategies):
        style = strategy_styles[i % len(strategy_styles)]
        cum = (1 + strat.returns.reindex(returns.index).fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=strat.label, zorder=5, **style)
    title = "Assets vs Strategy Models (Cumulative Growth of $1)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio value")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    p = reports_dir / "asset_cumulative_returns.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 2. Bar chart: ann return (left axis) and Sharpe (right axis) per asset + strategies
    n_assets = len(metrics)
    n_strategies = len(strategies)
    n_total = n_assets + n_strategies
    fig, ax = plt.subplots(figsize=(max(11, 1.2 * n_total), 5))
    x = np.arange(n_total)
    width = 0.35
    asset_labels = [f"{r['ticker']}\n{r['asset_class'][:12]}" for _, r in metrics.iterrows()]
    strategy_labels = [s.label.replace(" ", "\n") for s in strategies]
    all_labels = asset_labels + strategy_labels

    asset_ret = metrics["annualized_return"].values * 100
    asset_sharpe = metrics["sharpe"].values
    strat_ret = np.array([s.annualized_return * 100 for s in strategies])
    strat_sharpe = np.array([s.sharpe for s in strategies])

    ret_vals = np.concatenate([asset_ret, strat_ret])
    sharpe_vals = np.concatenate([asset_sharpe, strat_sharpe])

    ret_colors = ["#55A868"] * n_assets + ["#4C72B0", "#111111", "#C44E52", "#E24A4A"][:n_strategies]
    sharpe_colors = ["#8172B3"] * n_assets + ["#6A8FC7", "#555555", "#E07A7A", "#F08080"][:n_strategies]

    ax.bar(x - width / 2, ret_vals, width, label="Ann. Return %", color=ret_colors)
    ax.set_ylabel("Annualized Return (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=7)
    ax.set_title("Assets vs Strategy Models (Full Sample)", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8)
    if n_strategies:
        ax.axvline(n_assets - 0.5, color="gray", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(x + width / 2, sharpe_vals, width, label="Sharpe", color=sharpe_colors, alpha=0.85)
    ax2.set_ylabel("Sharpe Ratio")
    sharpe_max = float(np.nanmax(sharpe_vals)) if len(sharpe_vals) else 0.0
    if sharpe_max > 0:
        ax2.set_ylim(0, sharpe_max * 1.15)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", fontsize=8)
    p = reports_dir / "asset_metrics_bars.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    # 3. Train vs test return comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    train_ret = analysis.metrics_train.set_index("ticker")["annualized_return"] * 100
    test_ret = analysis.metrics_test.set_index("ticker")["annualized_return"] * 100
    tickers = [t for t in metrics["ticker"] if t in train_ret.index and t in test_ret.index]
    x = np.arange(len(tickers))
    train_label = analysis.train_period_label.replace(" to ", "–")
    test_label = analysis.test_period_label.replace(" to ", "–").replace("latest", "…")
    ax.bar(x - width / 2, [train_ret[t] for t in tickers], width, label=f"Train ({train_label})", color="#4C72B0")
    ax.bar(x + width / 2, [test_ret[t] for t in tickers], width, label=f"Test ({test_label})", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Individual Asset Returns: Train vs Test", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    p = reports_dir / "asset_train_test_returns.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    return saved


def _data_components_section(analysis: AssetAnalysisResult) -> list[str]:
    asset_rows = []
    for ticker, meta in ASSET_CATALOG.items():
        asset_rows.append(
            {
                "Ticker": ticker,
                "Instrument": meta["name"],
                "Proxy / Benchmark": meta["benchmark"],
                "Asset Class": meta["asset_class"],
                "Role in Portfolio": meta["role"],
                "Data Source": meta["source"],
            }
        )

    macro_rows = []
    for sid in analysis.macro_series_loaded:
        key = sid.replace("^", "")
        meta = MACRO_CATALOG.get(key, MACRO_CATALOG.get(sid, {}))
        if not meta:
            macro_rows.append({"Series": sid, "Description": sid, "Use": "Macro feature", "Source": "FRED"})
            continue
        macro_rows.append(
            {
                "Series": sid,
                "Description": meta["name"],
                "Use": meta["role"],
                "Source": meta["source"],
            }
        )

    lines = [
        "## Data & Components Used",
        "",
        "The pipeline combines **seven tradable ETF proxies** for major asset classes plus **macro/risk indicators** "
        "for regime features. Prices are resampled to **weekly** (Friday close) from daily adjusted-close data.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Sample start | {analysis.effective_start or 'N/A'} |",
        f"| Sample end | {analysis.effective_end or 'N/A'} |",
        f"| Frequency | Weekly (W-FRI) |",
        f"| Price field | Adjusted close |",
        "",
        "### Tradable ETF Components",
        "",
        _markdown_table(pd.DataFrame(asset_rows)),
        "",
        "### Macro & Risk Indicators (features only)",
        "",
        "These series are **not traded** in the backtest. They feed M1/M2 regime and false-positive features, "
        "lagged by 4 weeks to approximate publication delay.",
        "",
        _markdown_table(pd.DataFrame(macro_rows)),
        "",
    ]
    return lines


def _individual_asset_section(analysis: AssetAnalysisResult, *, image_prefix: str = "") -> list[str]:
    full_table = format_asset_metrics_table(analysis.metrics_full)
    train_table = format_asset_metrics_table(analysis.metrics_train)
    test_table = format_asset_metrics_table(analysis.metrics_test)

    lines = [
        "## Individual Asset Performance (Buy-and-Hold)",
        "",
        "Each row below is a **standalone buy-and-hold** of one ETF: 100% allocated to that asset, "
        "rebalanced weekly, **no transaction costs**, no M1/M2 overlay. This shows how each building block "
        "performed on its own before any strategy logic. "
        "Charts also overlay **M1** and **M1+M2** portfolio models (long-only and long/short) for comparison.",
        "",
        "### Full Sample",
        "",
        _markdown_table(full_table),
        "",
        f"![Individual asset cumulative returns]({image_prefix}asset_cumulative_returns.png)",
        "",
        f"![Individual asset metrics]({image_prefix}asset_metrics_bars.png)",
        "",
        f"### Train Period ({analysis.train_period_label})",
        "",
        _markdown_table(train_table),
        "",
        f"### Test Period ({analysis.test_period_label})",
        "",
        _markdown_table(test_table),
        "",
        f"![Train vs test asset returns]({image_prefix}asset_train_test_returns.png)",
        "",
        "### Per-Asset Highlights",
        "",
    ]

    ranked = analysis.metrics_full.sort_values("annualized_return", ascending=False)
    for _, row in ranked.iterrows():
        lines.append(
            f"- **{row['ticker']}** ({row['benchmark']}): "
            f"{_fmt_pct(row['annualized_return'])} annualized, "
            f"Sharpe {_fmt_num(row['sharpe'])}, "
            f"max drawdown {_fmt_pct(row['max_drawdown'])} — {ASSET_CATALOG[row['ticker']]['role']}."
        )

    lines.extend(["",])
    return lines


def generate_asset_component_report(
    analysis: AssetAnalysisResult,
    report_path: Path,
    *,
    strategy_overlays: list[StrategyOverlay] | None = None,
) -> None:
    """Write standalone asset/component analysis markdown."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    charts = save_asset_analysis_charts(analysis, report_path.parent, strategy_overlays=strategy_overlays)
    analysis.metrics_full.to_csv(report_path.parent / "asset_metrics_full.csv", index=False)
    analysis.metrics_train.to_csv(report_path.parent / "asset_metrics_train.csv", index=False)
    analysis.metrics_test.to_csv(report_path.parent / "asset_metrics_test.csv", index=False)

    lines = [
        "# Asset & Component Analysis",
        "",
        "Standalone buy-and-hold performance for each ETF in the universe, plus documentation of all data inputs.",
        "",
        "**Research use only — not investment advice.**",
        "",
        *_data_components_section(analysis),
        *_individual_asset_section(analysis, image_prefix=""),
        "## Notes",
        "",
        "- **SPY** is used as the practical proxy for the **S&P 500** in this ETF-only universe.",
        "- **TLT** and **HYG** represent the **bond** sleeve (duration government vs. high-yield credit).",
        "- **VEA** and **VWO** split developed vs. emerging international equities.",
        "- **GLD** provides commodity/gold exposure; **VNQ** provides listed real estate.",
        "- Strategy results in the main report combine these components via M1 signals and M2 sizing.",
        "- Data provenance, ETL, validation, cache behavior, and fallback caveats are documented in `../../DATA_SOURCES_AND_ETL.md`.",
        "",
    ]
    report_path.write_text("\n".join(lines))


def asset_analysis_markdown_sections(
    analysis: AssetAnalysisResult,
    assets_dir: Path,
    *,
    image_prefix: str = "assets/",
    strategy_overlays: list[StrategyOverlay] | None = None,
) -> list[str]:
    """Return markdown sections to embed in the main final report."""
    save_asset_analysis_charts(analysis, assets_dir, strategy_overlays=strategy_overlays)
    analysis.metrics_full.to_csv(assets_dir / "asset_metrics_full.csv", index=False)
    return [
        *_data_components_section(analysis),
        *_individual_asset_section(analysis, image_prefix=image_prefix),
        f"See also: [{image_prefix}asset_component_analysis.md]({image_prefix}asset_component_analysis.md) "
        "for the full standalone write-up.",
        "",
    ]
