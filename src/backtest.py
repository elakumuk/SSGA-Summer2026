"""Backtest engine with transaction costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.portfolio import apply_constraints_by_date, apply_vol_target_wide, weights_to_wide
from src.position_sizing import SizingMode, compute_sizes


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    gross_returns: pd.Series


def equal_weight_returns(
    returns_wide: pd.DataFrame,
    tickers: list[str],
    rebalance: str = "weekly",
) -> BacktestResult:
    sub = returns_wide.reindex(columns=[t for t in tickers if t in returns_wide.columns])
    available = sub.notna()
    n_active = available.sum(axis=1).replace(0, np.nan)
    w = available.div(n_active, axis=0).fillna(0.0)
    return _run_backtest("equal_weight_1_7", w, sub, transaction_cost_bps=0.0)


def sixty_forty_returns(
    returns_wide: pd.DataFrame,
    weights_map: dict[str, float],
    transaction_cost_bps: float = 0.0,
) -> BacktestResult:
    cols = [c for c in weights_map if c in returns_wide.columns]
    w = pd.DataFrame(0.0, index=returns_wide.index, columns=returns_wide.columns)
    for t, wt in weights_map.items():
        if t in w.columns:
            w[t] = wt
    return _run_backtest("sixty_forty", w, returns_wide, transaction_cost_bps=transaction_cost_bps)


def _run_backtest(
    name: str,
    weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    transaction_cost_bps: float,
) -> BacktestResult:
    aligned_ret = returns_wide.reindex(weights.index).fillna(0.0)
    w = weights.reindex(aligned_ret.index).ffill().fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    tc = turnover * transaction_cost_bps / 10000.0
    gross = (w.shift(1) * aligned_ret).sum(axis=1)
    net = gross - tc
    return BacktestResult(
        name=name,
        returns=net.rename("return"),
        weights=w,
        turnover=turnover.rename("turnover"),
        transaction_costs=tc.rename("transaction_cost"),
        gross_returns=gross.rename("gross_return"),
    )


def strategy_weights_from_panel(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    sizing_mode: SizingMode | str,
    *,
    use_m2: bool = True,
    m2_threshold: float | None = None,
    train_sorted: np.ndarray | None = None,
    train_proba: pd.Series | None = None,
) -> pd.DataFrame:
    df = panel.reset_index().copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"])
    if use_m2 and "p_success" in df.columns:
        sizes = compute_sizes(
            df["p_success"],
            sizing_mode,
            threshold=m2_threshold or cfg.m2.threshold,
            train_proba=train_proba,
            train_sorted=train_sorted,
        )
        df["size"] = sizes.reindex(df.index).fillna(0.0)
    else:
        df["size"] = 1.0

    if "M1_conviction" in df.columns:
        df["raw_weight"] = (
            df["M1_signal"] * df["size"] * df["M1_conviction"] * cfg.portfolio.base_budget_per_asset
        )
    else:
        df["raw_weight"] = df["M1_signal"] * df["size"] * cfg.portfolio.base_budget_per_asset
    df["weight"] = apply_constraints_by_date(df, cfg.portfolio)
    w_wide = weights_to_wide(df.reset_index())
    return apply_vol_target_wide(w_wide, returns_wide, cfg.portfolio)


def run_all_strategies(
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: PipelineConfig,
    train_proba: pd.Series | None = None,
) -> dict[str, BacktestResult]:
    train_sorted = None
    if train_proba is not None:
        from src.position_sizing import fit_ecdf

        train_sorted = fit_ecdf(train_proba)

    tc = cfg.portfolio.transaction_cost_bps
    results: dict[str, BacktestResult] = {}
    results["equal_weight_1_7"] = equal_weight_returns(returns_wide, cfg.assets.tickers)
    results["sixty_forty"] = sixty_forty_returns(
        returns_wide, cfg.benchmarks.get("sixty_forty", {}), transaction_cost_bps=tc
    )

    m1_w = strategy_weights_from_panel(panel, returns_wide, cfg, SizingMode.LINEAR, use_m2=False)
    results["m1_only"] = _run_backtest("m1_only", m1_w, returns_wide, tc)

    for mode, key in [
        (SizingMode.BINARY, "m1_m2_binary"),
        (SizingMode.LINEAR, "m1_m2_linear"),
        (SizingMode.ECDF, "m1_m2_ecdf"),
    ]:
        w = strategy_weights_from_panel(
            panel,
            returns_wide,
            cfg,
            mode,
            use_m2=True,
            train_proba=train_proba,
            train_sorted=train_sorted,
        )
        results[key] = _run_backtest(key, w, returns_wide, tc)

    return results


def returns_wide_from_panel(panel: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    df = panel.reset_index()
    if "return_1w" not in df.columns:
        raise ValueError("return_1w required")
    wide = df[df["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values="return_1w")
    return wide.sort_index()
