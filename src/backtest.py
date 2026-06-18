"""Backtest + metrics. Info ratio is the headline (benchmark-relative shop).

No look-ahead: weights decided at week t earn week t+1's return (weights.shift(1)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS = 52


def portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame, costs: pd.Series | None = None) -> pd.Series:
    gross = (weights.shift(1) * returns).sum(axis=1)
    if costs is not None:
        gross = gross - costs.shift(1).fillna(0.0)
    return gross.rename("ret")


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    w = returns.notna().astype(float)
    w = w.div(w.sum(axis=1), axis=0)
    return (w.shift(1) * returns).sum(axis=1).rename("ew")


def static_portfolio_returns(weights: dict[str, float], returns: pd.DataFrame) -> pd.Series:
    """Returns of a fixed-weight baseline portfolio (rebalanced weekly to target)."""
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    w = w / w.sum()
    return (returns.mul(w, axis=1)).sum(axis=1).rename("static")


def _max_drawdown(rets: pd.Series) -> float:
    curve = (1 + rets.fillna(0)).cumprod()
    return float((curve / curve.cummax() - 1).min())


def metrics(rets: pd.Series, benchmark: pd.Series | None = None) -> dict:
    r = rets.dropna()
    ann_ret = float((1 + r).prod() ** (WEEKS / len(r)) - 1) if len(r) else np.nan
    ann_vol = float(r.std() * np.sqrt(WEEKS))
    sharpe = float(ann_ret / ann_vol) if ann_vol else np.nan
    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(r),
    }
    if benchmark is not None:
        active = (r - benchmark.reindex(r.index)).dropna()
        te = float(active.std() * np.sqrt(WEEKS))
        out["tracking_error"] = te
        out["info_ratio"] = float((active.mean() * WEEKS) / te) if te else np.nan
        out["excess_return"] = float(active.mean() * WEEKS)
    return out
