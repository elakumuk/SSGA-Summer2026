"""Tests for strategy overlays on asset analysis charts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.asset_analysis import strategy_overlays_from_mode_results


@dataclass
class _FakeResult:
    returns: pd.Series


@dataclass
class _FakeMode:
    mode_name: str
    results: dict


def test_strategy_overlays_from_mode_results():
    dates = pd.date_range("2020-01-03", periods=20, freq="W-FRI")
    returns = pd.Series(np.linspace(0.001, 0.002, len(dates)), index=dates)
    bt = _FakeResult(returns=returns)
    modes = [
        _FakeMode("long_only", {"m1_only": bt, "m1_m2_linear": bt}),
        _FakeMode("long_short", {"m1_only": bt, "m1_m2_linear": bt}),
    ]
    overlays = strategy_overlays_from_mode_results(modes)
    assert len(overlays) == 4
    assert overlays[0].label == "M1 (Long Only)"
