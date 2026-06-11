"""Tests for train/test split configuration and CLI overrides."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import apply_split_overrides, load_config, validate_split_dates

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config/config.yaml"


def test_default_split_valid():
    cfg = load_config(CONFIG)
    validate_split_dates(cfg)
    assert cfg.split.train_end == "2020-12-31"
    assert cfg.split.test_start == "2021-01-01"
    assert cfg.split.require_full_universe is True
    assert cfg.data_start_resolved() == "2000-01-01"


def test_apply_split_overrides():
    cfg = load_config(CONFIG)
    updated = apply_split_overrides(
        cfg,
        train_end="2018-12-31",
        test_start="2019-01-01",
    )
    assert updated.split.train_end == "2018-12-31"
    assert updated.split.test_start == "2019-01-01"
    assert updated.split.train_start == cfg.split.train_start


def test_invalid_split_raises():
    cfg = load_config(CONFIG)
    with pytest.raises(ValueError, match="train_end"):
        apply_split_overrides(cfg, train_end="2021-06-01", test_start="2021-01-01")


def test_partial_universe_override():
    cfg = load_config(CONFIG)
    updated = apply_split_overrides(cfg, require_full_universe=False, train_start="2005-01-01")
    assert updated.split.require_full_universe is False
    assert updated.split.train_start == "2005-01-01"


def test_build_available_panel_allows_partial_dates():
    from src.data_validation import build_available_panel, build_balanced_panel

    dates = pd.to_datetime(["2005-01-07", "2005-01-14", "2005-01-21"])
    rows = []
    for d in dates:
        rows.append({"date": d, "ticker": "SPY", "adj_close": 100.0})
        if d >= dates[1]:
            rows.append({"date": d, "ticker": "TLT", "adj_close": 90.0})
    df = pd.DataFrame(rows)
    balanced = build_balanced_panel(df, ["SPY", "TLT"])
    available = build_available_panel(df, ["SPY", "TLT"])
    assert len(balanced) == 4  # two tickers x two complete dates
    assert len(available) == 5  # SPY on all 3 weeks + TLT on 2


def test_train_date_mask_respects_start():
    import pandas as pd

    from src.model_m1 import split_train_test

    cfg = load_config(CONFIG)
    cfg = apply_split_overrides(cfg, train_start="2010-01-01", train_end="2015-12-31", test_start="2016-01-01")
    dates = pd.date_range("2008-01-04", periods=520, freq="W-FRI")
    tickers = ["SPY"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({"score": 0.0}, index=idx)
    train, test = split_train_test(panel, cfg)
    assert train.index.get_level_values("date").min() >= pd.Timestamp("2010-01-01")
    assert train.index.get_level_values("date").max() <= pd.Timestamp("2015-12-31")
    assert test.index.get_level_values("date").min() >= pd.Timestamp("2016-01-01")
