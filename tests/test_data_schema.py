"""Data schema validation tests."""

from __future__ import annotations

import pandas as pd

from src.data_validation import validate_model_panel, validate_price_panel


def test_required_columns_present(synthetic_panel, cfg):
    df = synthetic_panel.reset_index()
    report = validate_model_panel(df, cfg.assets.tickers)
    assert "date" in df.columns
    assert "ticker" in df.columns
    assert "adj_close" in df.columns
    assert "return_1w" in df.columns
    assert not df.duplicated(subset=["date", "ticker"]).any()


def test_price_panel_validation(cfg, synthetic_panel):
    df = synthetic_panel.reset_index()
    report = validate_price_panel(df, cfg.assets.tickers, balanced=False)
    assert report.n_rows > 0
    assert report.n_tickers == len(cfg.assets.tickers)


def test_feature_columns_numeric(synthetic_panel, cfg):
    from src.feature_engineering import get_feature_columns

    df = synthetic_panel.reset_index()
    feature_cols = get_feature_columns(synthetic_panel)
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(df[col])
