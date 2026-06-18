"""End-to-end smoke run for the M1 layer.

Loads config -> ingests data (cached) -> builds technical + macro features ->
computes the simple linear M1 score -> converts to benchmark-relative active
weights, and prints the latest week so we can SEE how the assets are allocated.

    python run_m1.py
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config
from src.data import ingest_macro_data, ingest_market_data
from src.features import (
    get_vix_series,
    macro_asset_tilt,
    macro_wide,
    pivot_prices,
    regime_features,
    technical_score,
)
from src.m1 import M1Model

logging.basicConfig(level=logging.WARNING, format="%(message)s")
pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def main() -> None:
    cfg = load_config()

    market = ingest_market_data(
        cfg.data.universe, cfg.vix_ticker, cfg.data.data_start, None,
        cfg.raw_dir, cfg.processed_dir,
    )
    macro = ingest_macro_data(
        cfg.data.macro_series, cfg.data.data_start, None,
        cfg.raw_dir, cfg.processed_dir, market_weekly=market,
    )

    prices = pivot_prices(market[market["ticker"].isin(cfg.data.universe)])
    vix = get_vix_series(market)
    regime = regime_features(macro_wide(macro), vix)

    m1 = M1Model(cfg.m1)
    score = m1.score({
        "technical": technical_score(prices, cfg.m1.momentum_windows, cfg.m1.trend_windows),
        "macro": macro_asset_tilt(prices, regime),
    })
    tilts = m1.active_tilts(score, cfg.portfolio.max_active_tilt, cfg.portfolio.max_abs_asset_weight)

    last = score.index[-1]
    print(f"\n=== M1 latest week: {last.date()} | universe {list(prices.columns)} ===\n")
    table = pd.DataFrame({
        "M1_score": score.loc[last],
        "active_weight": tilts.loc[last],
        "benchmark_1/N": 1.0 / len(prices.columns),
    }).sort_values("M1_score", ascending=False)
    table["active_tilt_vs_bench"] = table["active_weight"] - table["benchmark_1/N"]
    print(table.to_string())
    print(f"\ngross exposure = {tilts.loc[last].sum():.3f} (fully invested = 1.000)")
    print(f"weeks of history: {len(score)}  |  first: {score.index[0].date()}")


if __name__ == "__main__":
    main()
