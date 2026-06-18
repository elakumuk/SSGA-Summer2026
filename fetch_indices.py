"""Auto-fetch the index series that have a public source (FRED / Yahoo) into
data/raw/index/<KEY>.csv. Manual sources (MSCI EAFE, MSCI EM, Treasury 7-10Y) are
listed for you to download. See DATA_SOURCES.md.

    python fetch_indices.py
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.config import load_config
from src.data import INDEX_SOURCES, _fetch_fred_csv


def main() -> None:
    cfg = load_config()
    out_dir = cfg.raw_dir / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = cfg.data.data_start

    for key in cfg.data.universe:
        kind, ident = INDEX_SOURCES.get(key, ("manual", key))
        try:
            if kind == "fred":
                df = _fetch_fred_csv(ident)
                df.columns = ["date", "adj_close"][: len(df.columns)] if len(df.columns) == 2 else df.columns
                df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "adj_close"})
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
            elif kind == "yahoo":
                raw = yf.download(ident, start=start, auto_adjust=False, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                col = "Adj Close" if "Adj Close" in raw.columns else "Close"
                df = raw[[col]].reset_index()
                df.columns = ["date", "adj_close"]
            else:
                print(f"  [manual] {key}: download {ident} -> {out_dir / (key + '.csv')}")
                continue
            df = df.dropna()
            df = df[df["date"] >= pd.Timestamp(start)]
            df.to_csv(out_dir / f"{key}.csv", index=False)
            print(f"  [{kind}] {key} <- {ident}: {len(df)} rows -> {out_dir / (key + '.csv')}")
        except Exception as exc:
            print(f"  [error] {key} ({ident}): {exc}")

    print("\nDone. Set use_index_signal: true in config to use index data "
          "(missing files fall back to ETF).")


if __name__ == "__main__":
    main()
