"""Tests for M1-signal-grouped M2 performance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.diagnostics import analyze_m1_signal_m2_performance, format_m1_signal_analysis_table


def _synthetic_test_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=8, freq="W-FRI")
    tickers = ["SPY", "TLT"]
    rows = []
    for d in dates:
        for t in tickers:
            sig = rng.choice([-1, 0, 1], p=[0.2, 0.3, 0.5])
            trade_ret = rng.normal(0.002 if sig == 1 else -0.001, 0.01)
            meta = int(trade_ret > 0.001) if sig != 0 else np.nan
            p = rng.uniform(0.4, 0.9) if sig != 0 else np.nan
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "M1_signal": sig,
                    "trade_return": trade_ret if sig != 0 else 0.0,
                    "meta_label": meta,
                    "p_success": p,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_analyze_groups_by_m1_signal():
    panel = _synthetic_test_panel()
    analysis = analyze_m1_signal_m2_performance(panel, threshold=0.55)
    by_sig = analysis["by_signal"]
    assert set(by_sig["m1_signal"]) <= {-1, 0, 1}
    flat = by_sig[by_sig["m1_signal"] == 0].iloc[0]
    assert flat["observations"] > 0
    assert "labeled_trades" not in flat or pd.isna(flat.get("labeled_trades", np.nan))


def test_long_only_has_no_short_bucket():
    panel = _synthetic_test_panel().reset_index()
    panel.loc[panel["M1_signal"] == -1, "M1_signal"] = 0
    analysis = analyze_m1_signal_m2_performance(panel.set_index(["date", "ticker"]), 0.55)
    assert -1 not in set(analysis["by_signal"]["m1_signal"])


def test_format_table_includes_hit_rates():
    panel = _synthetic_test_panel()
    analysis = analyze_m1_signal_m2_performance(panel, threshold=0.55)
    table = format_m1_signal_analysis_table(analysis)
    assert "M1 Hit Rate" in table.columns
    assert "Hit Rate (M2 Approved)" in table.columns
