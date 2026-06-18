"""Portfolio construction — BENCHMARK-RELATIVE active weights.

benchmark = equal weight (1/N). M1 score -> cross-sectional tilt around benchmark.
Optional M2 sizing multiplier shrinks the tilt where M2 distrusts M1. Then volatility
targeting + gross/asset caps. Shorting is unnecessary: an underweight IS the short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config


def _centered_rank(s: pd.Series) -> pd.Series:
    n = len(s)
    if n <= 1:
        return pd.Series(0.0, index=s.index)
    return 2 * (s.rank() - 1) / (n - 1) - 1            # best name +1, worst -1


def build_weights(score: pd.DataFrame, cfg: Config, size: pd.Series | None = None) -> pd.DataFrame:
    """Active weights = 1/N + max_tilt * centered_rank(score) * [M2 size].
    size is a long (date,ticker) Series in [0,1]; None -> M1-only."""
    max_tilt = cfg.portfolio.max_active_tilt
    max_abs = cfg.portfolio.max_abs_asset_weight
    size_wide = size.unstack() if size is not None else None

    out = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for date, row in score.iterrows():
        s = row.dropna()
        if s.empty:
            continue
        n = len(s)
        tilt = max_tilt * _centered_rank(s)
        if size_wide is not None and date in size_wide.index:
            m = size_wide.loc[date].reindex(s.index).fillna(0.0)
            tilt = tilt * m                            # M2 shrinks distrusted tilts
        w = (1.0 / n + tilt).clip(lower=0.0, upper=max_abs)
        if w.sum() > 0:
            w = w / w.sum()
        out.loc[date, w.index] = w.values
    return out


def apply_risk_layer(weights: pd.DataFrame, risk_score: pd.DataFrame, strength: float) -> pd.DataFrame:
    """SEPARATE risk layer (not an M1 factor): tilt weights toward lower-risk
    (higher risk_score) assets via a multiplicative overlay, then renormalize.
    risk_score is cross-sectionally z-scored (mean ~0); high = low risk = good."""
    mult = (1.0 + strength * risk_score.reindex_like(weights).fillna(0.0)).clip(lower=0.0)
    w = weights * mult
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def apply_vol_target(weights: pd.DataFrame, returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Scale gross exposure to hit the annualized vol target (trailing realized vol)."""
    target = cfg.portfolio.vol_target_ann
    lookback = max(4, cfg.portfolio.vol_target_lookback_weeks)
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    realized = port_ret.rolling(lookback).std() * np.sqrt(52)
    scale = (target / realized).clip(upper=2.0).fillna(1.0).replace([np.inf, -np.inf], 1.0)
    scaled = weights.mul(scale, axis=0)
    gross = scaled.abs().sum(axis=1).replace(0, np.nan)
    over = gross > cfg.portfolio.max_gross_exposure
    scaled.loc[over] = scaled.loc[over].div(gross[over], axis=0) * cfg.portfolio.max_gross_exposure
    return scaled.fillna(0.0)


def cost_drag(weights: pd.DataFrame, cfg: Config) -> pd.Series:
    """Two cost layers (State Street): (1) expense ratio carry, (2) transaction cost."""
    expense_weekly = (cfg.costs.expense_ratio_bps_annual / 1e4) / 52.0
    expense = weights.abs().sum(axis=1) * expense_weekly
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    txn = turnover * (cfg.costs.transaction_cost_bps / 1e4)
    return expense + txn
