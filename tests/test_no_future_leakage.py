"""Tests for no future leakage."""

from __future__ import annotations

import pandas as pd

from src.feature_engineering import get_feature_columns
from src.model_m1 import split_train_test


def test_train_test_no_overlap(cfg, synthetic_panel):
    train, test = split_train_test(synthetic_panel, cfg)
    train_dates = train.index.get_level_values("date")
    test_dates = test.index.get_level_values("date")
    if len(train_dates) and len(test_dates):
        assert train_dates.max() < test_dates.min()


def test_feature_matrix_excludes_labels(synthetic_panel):
    feature_cols = get_feature_columns(synthetic_panel)
    forbidden = {"forward_return_4w", "m1_target", "meta_label", "trade_return"}
    assert forbidden.isdisjoint(set(feature_cols))


def test_no_label_columns_in_features(synthetic_panel):
    cols = get_feature_columns(synthetic_panel)
    for c in cols:
        assert not c.startswith("forward_return")
        assert c not in ("meta_label", "trade_return", "m1_target")


def test_chronological_dates(synthetic_panel):
    dates = synthetic_panel.reset_index()["date"]
    assert dates.is_monotonic_increasing or dates.sort_values().equals(dates)
