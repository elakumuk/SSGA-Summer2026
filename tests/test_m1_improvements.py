"""Tests for M1 improvements: top-K, conviction, exposure diagnostics, vol target."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestResult, strategy_weights_from_panel, _run_backtest
from src.config import M1Config, PipelineConfig, PortfolioConfig
from src.diagnostics import (
    analyze_m1_exposure,
    compute_per_asset_ic,
    threshold_sensitivity_summary,
)
from src.model_m1 import RuleBasedM1, _signals_from_thresholds, build_m1_model
from src.portfolio import apply_vol_target_wide, build_weights_from_signals


def _synthetic_panel(n_dates: int = 20, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["SPY", "TLT", "GLD"]
    dates = pd.date_range("2020-01-03", periods=n_dates, freq="W-FRI")
    rows = []
    for d in dates:
        for i, t in enumerate(tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "M1_signal": 1 if i == 0 else 0,
                    "M1_score": float(i),
                    "M1_conviction": 0.5 + 0.1 * i,
                    "p_success": 0.6,
                    "return_1w": 0.001,
                }
            )
    df = pd.DataFrame(rows)
    return df.set_index(["date", "ticker"]).sort_index()


def test_top_k_allocation_long_only():
    cfg = M1Config(allow_short=False, allocation_mode="top_k", top_k=2, optimize_thresholds=False)
    m1 = RuleBasedM1(cfg)
    dates = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    tickers = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    scores = pd.Series(
        [0.1, 0.5, 0.9, 0.3, 0.2, 0.8, 0.7, 0.4, 0.0, 0.6, 0.1, 0.2],
        index=idx,
        name="M1_score",
    )
    X = pd.DataFrame({"z_mom_12w": scores}, index=idx)
    # Override predict_score path by using internal method
    sig = m1._signals_top_k(scores)
    per_date = sig.groupby(level="date").sum()
    assert (per_date <= 2).all()
    assert (sig.groupby(level="date").apply(lambda s: (s == 1).sum()) <= 2).all()


def test_conviction_in_weights():
    panel = _synthetic_panel()
    returns_wide = panel.reset_index().pivot(index="date", columns="ticker", values="return_1w").fillna(0)
    cfg = PipelineConfig()
    cfg.portfolio.vol_target_ann = None
    w = strategy_weights_from_panel(panel, returns_wide, cfg, "linear", use_m2=False)
    assert (w.abs().sum(axis=1) > 0).any()


def test_vol_target_scales_exposure():
    dates = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    tickers = ["SPY", "TLT"]
    rng = np.random.default_rng(42)
    returns_wide = pd.DataFrame(
        {t: rng.normal(0.001, 0.02, len(dates)) for t in tickers},
        index=dates,
    )
    weights = pd.DataFrame(0.2, index=dates, columns=tickers)
    cfg = PortfolioConfig(vol_target_ann=0.15, vol_target_lookback_weeks=8, vol_target_max_scale=3.0)
    scaled = apply_vol_target_wide(weights, returns_wide, cfg)
    assert scaled.abs().sum(axis=1).mean() >= weights.abs().sum(axis=1).mean() * 0.5


def test_analyze_m1_exposure():
    dates = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    w = pd.DataFrame({"SPY": 0.3, "TLT": 0.2}, index=dates)
    r = pd.Series(0.001, index=dates)
    result = BacktestResult("m1_only", r, w, r * 0, r * 0, r)
    bench_w = pd.DataFrame({"SPY": 0.5, "TLT": 0.5}, index=dates)
    bench = BacktestResult("ew", r, bench_w, r * 0, r * 0, r)
    exp = analyze_m1_exposure(result, bench)
    assert exp["summary"]["mean_gross_exposure"] == pytest.approx(0.5)
    assert exp["summary"]["mean_cash_weight"] == pytest.approx(0.5)


def test_per_asset_ic():
    panel = _synthetic_panel(n_dates=30)
    panel = panel.reset_index()
    panel["forward_return_4w"] = panel.groupby("ticker")["return_1w"].shift(-4)
    panel = panel.set_index(["date", "ticker"])
    ic_df = compute_per_asset_ic(panel)
    assert "ticker" in ic_df.columns
    assert len(ic_df) >= 1


def test_ml_m1_build():
    cfg = PipelineConfig()
    cfg.models = {"m1": {"type": "ml", "allow_short": False, "allocation_mode": "top_k", "top_k": 2}}
    m1 = build_m1_model(cfg)
    assert m1.__class__.__name__ == "MLM1"


def test_portfolio_threshold_signals():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-03", periods=5, freq="W-FRI"), ["SPY"]],
        names=["date", "ticker"],
    )
    scores = pd.Series(np.linspace(-1, 1, 5), index=idx)
    sig = _signals_from_thresholds(scores, 0.0, -0.5, allow_short=True)
    assert set(sig.unique()).issubset({-1, 0, 1})
