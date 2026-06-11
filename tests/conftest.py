"""Shared pytest fixtures with synthetic data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path
    config_src = (
        (pytest.importorskip("pathlib").Path(__file__).parent.parent / "config" / "config.yaml")
    )
    return load_config(config_src)


@pytest.fixture
def synthetic_panel(cfg) -> pd.DataFrame:
    tickers = cfg.assets.tickers
    dates = pd.date_range("2015-01-02", periods=400, freq="W-FRI")
    rows = []
    rng = np.random.default_rng(42)
    price = {t: 100.0 for t in tickers}
    for d in dates:
        for t in tickers:
            ret = rng.normal(0.001, 0.02)
            price[t] *= 1 + ret
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "adj_close": price[t],
                    "return_1w": ret,
                    "mom_12w": rng.normal(0, 1),
                    "z_mom_12w": rng.normal(0, 1),
                    "z_mom_26w": rng.normal(0, 1),
                    "z_mom_52w": rng.normal(0, 1),
                    "z_trend_signal": rng.normal(0, 1),
                    "z_vol_12w": rng.normal(0, 1),
                    "z_drawdown_26w": rng.normal(0, 1),
                    "inflation_trend": rng.normal(0, 1),
                    "growth_trend": rng.normal(0, 1),
                    "yield_curve": rng.normal(0, 1),
                    "credit_stress": rng.normal(0, 1),
                    "risk_off": rng.choice([0.0, 1.0]),
                    "vix_level": abs(rng.normal(20, 5)),
                    "cross_asset_dispersion_4w": abs(rng.normal(0.01, 0.005)),
                    "forward_return_4w": rng.normal(0, 0.03),
                }
            )
    df = pd.DataFrame(rows)
    df["m1_target"] = np.where(
        df["forward_return_4w"] > 0.005,
        1,
        np.where(df["forward_return_4w"] < -0.005, -1, 0),
    )
    df["M1_signal"] = np.where(df["z_mom_12w"] > 0.5, 1, np.where(df["z_mom_12w"] < -0.5, -1, 0))
    df["M1_score"] = df["z_mom_12w"]
    df["trade_return"] = df["M1_signal"] * df["forward_return_4w"]
    df["meta_label"] = np.where(df["M1_signal"] != 0, (df["trade_return"] > 0.001).astype(int), np.nan)
    df["p_success"] = np.where(df["M1_signal"] != 0, rng.uniform(0.3, 0.8, len(df)), np.nan)
    return df.set_index(["date", "ticker"]).sort_index()
