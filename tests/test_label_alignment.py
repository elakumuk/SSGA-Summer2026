"""Meta-label alignment tests."""

from __future__ import annotations

import numpy as np

from src.labels import get_m2_training_mask


def test_m2_training_only_nonzero_m1(synthetic_panel):
    mask = get_m2_training_mask(synthetic_panel)
    assert (synthetic_panel.loc[mask, "M1_signal"] != 0).all()


def test_meta_label_logic(synthetic_panel):
    nonzero = synthetic_panel[synthetic_panel["M1_signal"] != 0]
    expected = (nonzero["trade_return"] > 0.001).astype(int)
    actual = nonzero["meta_label"].dropna()
    np.testing.assert_array_equal(actual.values, expected.loc[actual.index].values)


def test_zero_m1_has_null_meta_label(synthetic_panel):
    zero = synthetic_panel[synthetic_panel["M1_signal"] == 0]
    assert zero["meta_label"].isna().all()
