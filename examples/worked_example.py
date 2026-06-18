"""Worked 3-asset example (Xuesong template): show, with real numbers, how each
factor scores 3 assets, how they are ranked, and how benchmark-relative weights
(over/underweight = synthetic long/short) are allocated. Prints a table we paste
into FACTORS.md.

    python examples/worked_example.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.data import ingest_market_data
from src.features import _cross_sectional_z, pivot_prices

logging.basicConfig(level=logging.ERROR)
pd.set_option("display.width", 120)
SUBSET = ["SPY", "TLT", "GLD"]   # equity / long bonds / gold — three distinct sleeves


def main() -> None:
    cfg = load_config()
    market = ingest_market_data(cfg.data.universe, cfg.vix_ticker, cfg.data.data_start, None,
                                cfg.raw_dir, cfg.processed_dir)
    prices = pivot_prices(market[market["ticker"].isin(SUBSET)])[SUBSET].dropna()

    # --- factor RAW values (no look-ahead: shift(1)) ---
    mom12 = prices.pct_change(12).shift(1)                       # 12-week momentum
    trend = (prices.rolling(10).mean() / prices.rolling(40).mean() - 1).shift(1)

    d = prices.index[-1]
    raw = pd.DataFrame({
        "price": prices.loc[d],
        "mom_12w": mom12.loc[d],
        "trend_10_40": trend.loc[d],
    })

    # --- cross-sectional z-scores (compete on one axis) ---
    z = pd.DataFrame({
        "z_mom": _cross_sectional_z(mom12).loc[d],
        "z_trend": _cross_sectional_z(trend).loc[d],
    })
    z["technical"] = 0.5 * z["z_mom"] + 0.5 * z["z_trend"]       # merged technical score

    # --- ranking -> benchmark-relative weights (3 assets, +/-10% tilt) ---
    n = len(SUBSET)
    bench = 1.0 / n
    max_tilt = cfg.portfolio.max_active_tilt
    s = z["technical"]
    ranks = s.rank()
    centered = 2 * (ranks - 1) / (n - 1) - 1                     # best +1, worst -1
    tilt = max_tilt * centered
    w = (bench + tilt).clip(lower=0)
    w = w / w.sum()

    out = raw.join(z)
    out["rank"] = ranks.astype(int)
    out["tilt_vs_bench"] = tilt
    out["weight"] = w
    out["bench_1/N"] = bench

    print(f"\nWorked example — date {d.date()}, universe {SUBSET}\n")
    print(out.to_string(float_format=lambda v: f"{v:,.3f}"))
    print(f"\nfully invested: {w.sum():.3f}  |  best={s.idxmax()} (overweight), "
          f"worst={s.idxmin()} (underweight = synthetic short)")


if __name__ == "__main__":
    main()
