"""Tests for the Week-2 additions: DSR, HMM regime, conformal M2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.conformal import M2Conformal, conformal_size_multiplier
from src.config import M2Config
from src.diagnostics import (
    CRISIS_WINDOWS,
    crisis_subperiod_table,
    deflated_sharpe_ratio,
    robust_performance_table,
    subperiod_metrics,
)
from src.regime import (
    REGIME_LABELS,
    bridgewater_regime_columns,
    fit_macro_regime,
    regime_features_long,
)


def _make_returns(seed: int = 0, n: int = 600, sigma: float = 0.012, drift: float = 0.0015) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-04", periods=n, freq="W-FRI")
    return pd.Series(rng.normal(drift, sigma, n), index=idx)


# ---------- Deflated Sharpe ----------------------------------------------------


def test_dsr_high_for_good_strategy_low_for_bad():
    good = _make_returns(seed=1, drift=0.003, sigma=0.008)
    bad = _make_returns(seed=2, drift=-0.001, sigma=0.012)
    dsr_good = deflated_sharpe_ratio(good, n_trials=1)["dsr"]
    dsr_bad = deflated_sharpe_ratio(bad, n_trials=1)["dsr"]
    assert dsr_good > 0.9
    assert dsr_bad < 0.5


def test_dsr_decreases_with_n_trials():
    r = _make_returns(seed=3, drift=0.0025, sigma=0.012)
    dsr_1 = deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    dsr_100 = deflated_sharpe_ratio(r, n_trials=100)["dsr"]
    assert dsr_1 >= dsr_100  # multiple-testing penalty must not increase DSR


def test_subperiod_table_has_full_sample_and_crisis_windows():
    r = _make_returns(seed=4)
    table = subperiod_metrics(r)
    assert "full_sample" in table["window"].values
    for label in CRISIS_WINDOWS:
        assert label in table["window"].values
    # full sample n_weeks should equal len(r)
    full = table.loc[table["window"] == "full_sample"].iloc[0]
    assert int(full["n_weeks"]) == len(r)


def test_robust_performance_table_runs():
    from src.backtest import BacktestResult

    r = _make_returns(seed=5)
    res = BacktestResult(
        name="strat_a",
        returns=r,
        weights=pd.DataFrame(),
        turnover=pd.Series(0.0, index=r.index),
        transaction_costs=pd.Series(0.0, index=r.index),
        gross_returns=r,
    )
    out = robust_performance_table({"strat_a": res}, n_trials=5)
    assert {"strategy", "sharpe_ann", "dsr"}.issubset(out.columns)
    assert len(out) == 1


def test_crisis_subperiod_table_runs():
    from src.backtest import BacktestResult

    r = _make_returns(seed=6)
    res = BacktestResult(
        name="strat_a",
        returns=r,
        weights=pd.DataFrame(),
        turnover=pd.Series(0.0, index=r.index),
        transaction_costs=pd.Series(0.0, index=r.index),
        gross_returns=r,
    )
    out = crisis_subperiod_table({"strat_a": res})
    assert (out["strategy"] == "strat_a").all()
    assert "full_sample" in out["window"].values


# ---------- HMM regime ---------------------------------------------------------


def _make_macro_panel(n: int = 800, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-07", periods=n, freq="W-FRI")
    cycle = np.sin(np.linspace(0, 6 * np.pi, n))
    infl = np.cos(np.linspace(0, 6 * np.pi, n))
    return pd.DataFrame(
        {
            "INDPRO": 100 * (1 + 0.0005 * np.cumsum(0.001 * cycle + 0.0003 * rng.normal(size=n))),
            "CPIAUCSL": 100 * (1 + 0.0005 * np.cumsum(0.001 * infl + 0.0003 * rng.normal(size=n))),
            "T10Y2Y": 0.5 + 0.4 * cycle + 0.1 * rng.normal(size=n),
            "FEDFUNDS": 2.0 + infl + 0.1 * rng.normal(size=n),
        },
        index=idx,
    )


def test_hmm_regime_produces_normalized_posteriors():
    macro = _make_macro_panel()
    fit = fit_macro_regime(macro, train_end="2012-12-31", n_states=4)
    row_sums = fit.posteriors.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert set(fit.posteriors.columns) == set(REGIME_LABELS)
    assert fit.top1.isin(REGIME_LABELS).all()


def test_bridgewater_columns_include_tilts():
    macro = _make_macro_panel()
    fit = fit_macro_regime(macro, train_end="2012-12-31", n_states=4)
    brw = bridgewater_regime_columns(fit)
    assert "regime_growth_tilt" in brw.columns
    assert "regime_inflation_tilt" in brw.columns
    # Tilts must lie in [-1, 1] up to floating-point error.
    eps = 1e-9
    assert brw["regime_growth_tilt"].abs().max() <= 1.0 + eps
    assert brw["regime_inflation_tilt"].abs().max() <= 1.0 + eps


def test_regime_features_long_has_one_row_per_date_ticker():
    macro = _make_macro_panel()
    fit = fit_macro_regime(macro, train_end="2012-12-31", n_states=4)
    long = regime_features_long(fit, tickers=["SPY", "TLT", "GLD"])
    assert {"date", "ticker"}.issubset(long.columns)
    assert long.groupby("ticker").size().nunique() == 1  # equal rows per ticker


# ---------- Conformal M2 -------------------------------------------------------


def _make_classification_panel(n_dates: int = 200, n_assets: int = 5, seed: int = 21) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=n_dates, freq="W-FRI")
    tickers = [f"A{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    logit = 0.8 * f1 - 0.4 * f2 + 0.2 * f3
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(size=n) < p).astype(int)
    X = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3}, index=idx)
    return X, pd.Series(y, index=idx, name="meta_label")


def test_conformal_calibrates_and_predicts_with_intervals():
    X, y = _make_classification_panel()
    model = M2Conformal(M2Config(), calibration_weeks=20, alpha=0.10)
    model.fit(X, y)
    out = model.predict_with_interval(X)
    assert {"p_success", "p_success_lo", "p_success_hi", "p_success_band_width"}.issubset(out.columns)
    assert (out["p_success_lo"] <= out["p_success"]).all()
    assert (out["p_success_hi"] >= out["p_success"]).all()
    # Bands clipped to [0, 1]
    assert (out["p_success_lo"] >= 0).all()
    assert (out["p_success_hi"] <= 1).all()


def test_conformal_size_multiplier_within_bounds():
    band = pd.Series([0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    mult = conformal_size_multiplier(band, max_shrink_band=0.40, floor=0.25)
    assert (mult >= 0.25).all() and (mult <= 1.0).all()
    # Wider bands → smaller multiplier
    assert mult.iloc[-1] <= mult.iloc[0]
