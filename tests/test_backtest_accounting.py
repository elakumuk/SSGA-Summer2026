"""Backtest accounting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import _run_backtest, run_all_strategies
from src.config import PortfolioConfig
from src.portfolio import apply_constraints


def test_turnover_formula():
    idx = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
    tickers = ["A", "B"]
    w = pd.DataFrame([[0.5, 0.5], [0.6, 0.4], [0.6, 0.4], [0.3, 0.7], [0.3, 0.7]], index=idx, columns=tickers)
    ret = pd.DataFrame(0.01, index=idx, columns=tickers)
    res = _run_backtest("test", w, ret, transaction_cost_bps=5.0)
    expected_turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    pd.testing.assert_series_equal(res.turnover, expected_turnover, check_names=False)


def test_transaction_cost_formula():
    idx = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    w = pd.DataFrame([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0]], index=idx, columns=["A", "B"])
    ret = pd.DataFrame(0.01, index=idx, columns=["A", "B"])
    bps = 10.0
    res = _run_backtest("test", w, ret, transaction_cost_bps=bps)
    turnover = res.turnover.iloc[1]
    expected_tc = turnover * bps / 10000.0
    assert abs(res.transaction_costs.iloc[1] - expected_tc) < 1e-12


def test_signal_earns_next_period_return():
    idx = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    w = pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], index=idx, columns=["A", "B"])
    ret = pd.DataFrame({"A": [0.0, 0.05, 0.02], "B": [0.0, 0.01, 0.01]}, index=idx)
    res = _run_backtest("test", w, ret, transaction_cost_bps=0.0)
    assert abs(res.gross_returns.iloc[2] - 0.02) < 1e-12


def test_weight_constraints():
    cfg = PortfolioConfig(max_abs_asset_weight=0.25, max_gross_exposure=1.0, allow_short=True)
    w = pd.Series({"A": 0.5, "B": 0.5, "C": 0.5})
    constrained = apply_constraints(w, cfg)
    assert constrained.abs().max() <= 0.25 + 1e-9
    assert constrained.abs().sum() <= 1.0 + 1e-9


def test_all_strategy_variants_exist(synthetic_panel, cfg):
    df = synthetic_panel.reset_index()
    returns_wide = df.pivot(index="date", columns="ticker", values="return_1w").sort_index()
    results = run_all_strategies(synthetic_panel, returns_wide, cfg, train_proba=synthetic_panel["p_success"])
    expected = {
        "equal_weight_1_7",
        "sixty_forty",
        "m1_only",
        "m1_m2_binary",
        "m1_m2_linear",
        "m1_m2_ecdf",
    }
    assert expected == set(results.keys())
