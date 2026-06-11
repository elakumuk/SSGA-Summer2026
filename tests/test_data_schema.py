"""Data schema validation tests."""

from __future__ import annotations

import pandas as pd

from src.data_providers import MacroDataProvider, MarketDataProvider, ingest_macro_data, ingest_market_data
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


class DummyMarketProvider(MarketDataProvider):
    def __init__(self) -> None:
        self.called = False

    def get_prices(self, tickers, start, end=None, frequency="weekly"):
        self.called = True
        dates = pd.date_range(start, periods=3, freq="W-FRI")
        rows = []
        for ticker in tickers:
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker.replace("^", ""),
                        "open": 100 + i,
                        "high": 101 + i,
                        "low": 99 + i,
                        "close": 100 + i,
                        "adj_close": 100 + i,
                        "volume": 1000,
                    }
                )
        return pd.DataFrame(rows)


class PartialMacroProvider(MacroDataProvider):
    def get_macro(self, series, start, end=None, frequency="daily"):
        dates = pd.date_range(start, periods=3, freq="D")
        return pd.DataFrame({"date": dates, "series": [series[0]] * 3, "value": [1.0, 1.1, 1.2]})


def test_ingest_market_data_uses_injected_provider(tmp_path):
    provider = DummyMarketProvider()
    out = ingest_market_data(
        ["SPY"],
        "^VIX",
        "2020-01-01",
        None,
        tmp_path / "raw",
        tmp_path / "processed",
        provider=provider,
        use_cache=False,
    )
    assert provider.called
    assert {"SPY", "VIX"}.issubset(set(out["ticker"]))


def test_ingest_macro_data_fills_missing_series_from_proxy(tmp_path):
    market = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-03", periods=3, freq="W-FRI"),
            "ticker": ["VIX"] * 3,
            "adj_close": [20.0, 21.0, 22.0],
        }
    )
    out = ingest_macro_data(
        ["CPIAUCSL", "DGS10"],
        "2020-01-01",
        None,
        tmp_path / "raw",
        tmp_path / "processed",
        provider=PartialMacroProvider(),
        market_weekly=market,
        use_cache=False,
    )
    assert {"CPIAUCSL", "DGS10"} == set(out["series"])
