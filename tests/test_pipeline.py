"""Correctness tests — the properties an institutional reviewer checks first:
no look-ahead, embargo, benchmark-relative weight constraints, label logic.
Uses synthetic data only (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import risk_score, technical_score
from src.m1 import M1Model
from src.m2 import build_meta_labels
from src.portfolio import build_weights, cost_drag


@pytest.fixture
def prices():
    dates = pd.date_range("2010-01-01", periods=300, freq="W-FRI")
    rng = np.random.default_rng(0)
    tickers = ["SPY", "TLT", "GLD", "VEA", "VWO", "HYG", "VNQ"]
    data = {}
    for i, t in enumerate(tickers):
        steps = rng.normal(0.001 + i * 0.0002, 0.02, len(dates))
        data[t] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def cfg():
    return load_config()


# ---------- no look-ahead ----------

def test_technical_score_no_lookahead(prices, cfg):
    """Perturbing the LAST price must not change EARLIER technical scores."""
    base = technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows)
    perturbed = prices.copy()
    perturbed.iloc[-1, 0] *= 1.5
    after = technical_score(perturbed, cfg.m1.momentum_windows, cfg.m1.trend_windows)
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_risk_score_no_lookahead(prices, cfg):
    base = risk_score(prices, cfg.risk_layer.vol_windows)
    perturbed = prices.copy()
    perturbed.iloc[-1, 1] *= 1.5
    after = risk_score(perturbed, cfg.risk_layer.vol_windows)
    pd.testing.assert_frame_equal(base.iloc[:-1], after.iloc[:-1])


def test_risk_layer_keeps_weights_valid(prices, cfg):
    """The separate risk layer must keep weights fully invested and long-only."""
    from src.portfolio import apply_risk_layer
    score = technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows)
    w = build_weights(score, cfg)
    rk = risk_score(prices, cfg.risk_layer.vol_windows)
    wr = apply_risk_layer(w, rk, cfg.risk_layer.strength)
    active = wr[wr.abs().sum(axis=1) > 0]
    assert np.allclose(active.sum(axis=1), 1.0, atol=1e-9)
    assert (wr >= -1e-12).all().all()


# ---------- benchmark-relative weight constraints ----------

def test_weights_fully_invested_and_long_only(prices, cfg):
    score = technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows)
    w = build_weights(score, cfg)
    active = w[w.abs().sum(axis=1) > 0]
    assert np.allclose(active.sum(axis=1), 1.0, atol=1e-9)   # fully invested
    assert (w >= -1e-12).all().all()                          # long-only
    assert (w <= cfg.portfolio.max_abs_asset_weight + 1e-9).all().all()  # asset cap


def test_weights_are_benchmark_relative(prices, cfg):
    """Highest-scored asset must get >= benchmark weight; lowest <= benchmark."""
    score = technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows)
    w = build_weights(score, cfg)
    d = score.dropna(how="all").index[-1]
    n = score.loc[d].notna().sum()
    bench = 1.0 / n
    best = score.loc[d].idxmax()
    worst = score.loc[d].idxmin()
    assert w.loc[d, best] >= bench - 1e-9
    assert w.loc[d, worst] <= bench + 1e-9


def test_m2_size_shrinks_tilts_not_expand(prices, cfg):
    """An M2 size in [0,1] must move active weights TOWARD the benchmark."""
    score = technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows)
    w_full = build_weights(score, cfg)
    half = pd.Series(0.5, index=pd.MultiIndex.from_product(
        [score.index, score.columns], names=["date", "ticker"]))
    w_half = build_weights(score, cfg, size=half)
    n = score.columns.size
    d = score.dropna(how="all").index[-1]
    assert (w_half.loc[d] - 1.0 / n).abs().sum() <= (w_full.loc[d] - 1.0 / n).abs().sum() + 1e-9


# ---------- meta-label logic ----------

def test_meta_labels_sign(prices, cfg):
    """Overweight that beats the basket -> 1; flat tilt -> NaN."""
    n = prices.shape[1]
    w = pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)
    w.iloc[:, 0] = 1.0 / n + 0.1          # asset 0 overweight
    labels = build_meta_labels(prices, w, 4, 1.0 / n, 0.0)
    fwd = prices.pct_change(4).shift(-4)
    active_fwd = fwd.sub(fwd.mean(axis=1), axis=0)
    valid = labels.iloc[:, 0].dropna().index
    expected = (active_fwd.iloc[:, 0].loc[valid] > 0).astype(float)
    pd.testing.assert_series_equal(labels.iloc[:, 0].loc[valid], expected, check_names=False)
    # equal-weight columns have ~zero tilt -> NaN label
    assert labels.iloc[:, 1].isna().all()


# ---------- costs ----------

def test_cost_drag_nonnegative(prices, cfg):
    score = technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows)
    w = build_weights(score, cfg)
    assert (cost_drag(w, cfg) >= -1e-12).all()


def test_m1_score_linear_combo(prices, cfg):
    """M1 score must be the exact fixed-ratio linear blend of its factor groups."""
    m1 = M1Model(cfg.m1)
    f = {"technical": technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows)}
    s = m1.score(f)
    expected = cfg.m1.factors["technical"] * f["technical"].fillna(0.0)
    pd.testing.assert_frame_equal(s, expected)
