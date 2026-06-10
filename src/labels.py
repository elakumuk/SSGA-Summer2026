"""Label construction for M1 and M2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig


def add_forward_returns(panel: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    out = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    out["date"] = pd.to_datetime(out["date"])
    prices = out.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    fwd = prices.pct_change(horizon_weeks).shift(-horizon_weeks)
    col = f"forward_return_{horizon_weeks}w"
    out[col] = out.apply(
        lambda r: fwd.loc[r["date"], r["ticker"]] if r["date"] in fwd.index and r["ticker"] in fwd.columns else np.nan,
        axis=1,
    )
    return out.set_index(["date", "ticker"]) if "ticker" in out.columns else out


def build_m1_target(panel: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    h = cfg.labels.horizon_weeks
    fwd_col = f"forward_return_{h}w"
    if fwd_col not in out.columns:
        out = add_forward_returns(out.set_index(["date", "ticker"]), h).reset_index()

    pos = cfg.labels.positive_threshold
    neg = cfg.labels.negative_threshold
    fwd = out[fwd_col]
    out["m1_target"] = np.where(fwd > pos, 1, np.where(fwd < neg, -1, 0))
    return out.set_index(["date", "ticker"]).sort_index()


def build_meta_labels(
    panel: pd.DataFrame,
    m1_signals: pd.Series,
    m1_scores: pd.Series,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    out = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    h = cfg.labels.horizon_weeks
    fwd_col = f"forward_return_{h}w"
    if fwd_col not in out.columns:
        out = add_forward_returns(out.set_index(["date", "ticker"]), h).reset_index()

    out["M1_signal"] = m1_signals.values
    out["M1_score"] = m1_scores.values
    out["trade_return"] = out["M1_signal"] * out[fwd_col]
    cost_thresh = cfg.labels.transaction_cost_threshold
    out["meta_label"] = np.where(
        out["M1_signal"] != 0,
        (out["trade_return"] > cost_thresh).astype(int),
        np.nan,
    )
    return out.set_index(["date", "ticker"]).sort_index()


def get_m2_training_mask(panel: pd.DataFrame) -> pd.Series:
    if "M1_signal" not in panel.columns:
        raise ValueError("M1_signal required for M2 training mask")
    return panel["M1_signal"] != 0
