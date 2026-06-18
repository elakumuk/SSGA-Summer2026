"""Convert an investing.com 'Historical Data' CSV to the pipeline index format.

investing.com gives: "Date"(MM/DD/YYYY), "Price"(comma thousands), ...
We need: date,adj_close (ISO date, numeric), sorted ascending.

    python examples/convert_investing.py "<downloaded.csv>" <TICKER>
e.g. python examples/convert_investing.py ~/Downloads/"MSCI Emerging Markets Historical Data.csv" VWO
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    src, ticker = sys.argv[1], sys.argv[2]
    df = pd.read_csv(src).rename(columns={"Date": "date", "Price": "adj_close"})
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["adj_close"] = df["adj_close"].astype(str).str.replace(",", "").astype(float)
    df = df[["date", "adj_close"]].sort_values("date")
    out = ROOT / "data" / "raw" / "index" / f"{ticker}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{ticker}: {len(df)} rows, {df.date.min().date()} -> {df.date.max().date()} -> {out}")
    if len(df) < 200:
        print("  ⚠️  very few rows — did you set a long date range (Daily) before downloading?")


if __name__ == "__main__":
    main()
