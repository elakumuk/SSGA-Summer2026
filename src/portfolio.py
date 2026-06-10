"""Portfolio weight construction with risk constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PortfolioConfig


def raw_weights(
    signals: pd.Series,
    sizes: pd.Series,
    base_budget: float,
) -> pd.Series:
    return (signals * sizes * base_budget).rename("raw_weight")


def apply_constraints(
    weights: pd.Series,
    cfg: PortfolioConfig,
) -> pd.Series:
    w = weights.copy().astype(float)
    if not cfg.allow_short:
        w = w.clip(lower=0.0)
    w = w.clip(lower=-cfg.max_abs_asset_weight, upper=cfg.max_abs_asset_weight)
    gross = w.abs().sum()
    if gross > cfg.max_gross_exposure and gross > 0:
        w = w * (cfg.max_gross_exposure / gross)
    return w.rename("weight")


def apply_constraints_by_date(df: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series:
    """Apply portfolio constraints to each date's cross-section of raw weights."""
    out = []
    for date, grp in df.groupby(level="date"):
        w = apply_constraints(grp["raw_weight"], cfg)
        w.index = grp.index
        out.append(w)
    return pd.concat(out).sort_index()


def build_weights_panel(
    panel: pd.DataFrame,
    cfg: PortfolioConfig,
    sizes: pd.Series,
) -> pd.DataFrame:
    out = panel.copy()
    out["size"] = sizes.reindex(out.index)
    out["raw_weight"] = raw_weights(out["M1_signal"], out["size"], cfg.base_budget_per_asset)
    out["weight"] = apply_constraints_by_date(out, cfg)
    return out


def weights_to_wide(weights_long: pd.DataFrame) -> pd.DataFrame:
    df = weights_long.reset_index()
    return df.pivot(index="date", columns="ticker", values="weight").fillna(0.0).sort_index()
