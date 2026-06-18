"""Market + macro data layer. PORTED nearly verbatim from Vitaly's pipeline
(it was correct and config-independent). yfinance + FRED, weekly Friday close,
parquet caching, proxy-macro fallback. BloombergProvider is the hook for index/
institutional data later (State Street: research on the INDEX, longer history).
"""

from __future__ import annotations

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class MarketDataProvider(ABC):
    @abstractmethod
    def get_prices(self, tickers, start, end=None, frequency="weekly"): ...


class MacroDataProvider(ABC):
    @abstractmethod
    def get_macro(self, series, start, end=None, frequency="weekly"): ...


class BloombergProvider(MarketDataProvider, MacroDataProvider):
    """Placeholder. Export index series from Bloomberg Terminal to data/raw/,
    or wrap the Bloomberg API here. This is where the INDEX-based signal lands."""

    def get_prices(self, tickers, start, end=None, frequency="weekly"):
        raise NotImplementedError(
            "BloombergProvider placeholder: export index data from Bloomberg to data/raw/."
        )

    def get_macro(self, series, start, end=None, frequency="weekly"):
        raise NotImplementedError("BloombergProvider placeholder: implement or use FredProvider.")


# Internal key -> underlying index source (see DATA_SOURCES.md).
# kind: "fred" / "yahoo" auto-fetchable; "manual" must be downloaded to data/raw/index/.
INDEX_SOURCES = {
    "SPY": ("yahoo", "^GSPC"),                      # S&P 500 (long history; FRED SP500 = ~10y alt)
    "HYG": ("fred", "BAMLHYH0A0HYM2TRIV"),         # ICE BofA US HY total return
    "VNQ": ("fred", "NASDAQNQUSB351020"),          # Nasdaq US Benchmark REIT
    "GLD": ("yahoo", "GC=F"),                       # Gold price
    "VEA": ("manual", "MSCI_EAFE"),                 # investing.com
    "VWO": ("manual", "MSCI_EM"),                   # investing.com
    "TLT": ("manual", "SP_USTREASURY_7_10Y"),       # Bloomberg
}


class IndexFileProvider(MarketDataProvider):
    """Reads index series exported to data/raw/index/<TICKER>.csv (columns:
    date, adj_close). This is the 'replace ETF with index data' path. Missing
    files fall back to the ETF (logged) so a partial index set still runs."""

    def __init__(self, index_dir: Path, fallback: MarketDataProvider | None = None) -> None:
        self.index_dir = index_dir
        self.fallback = fallback or YFinanceProvider()

    def get_prices(self, tickers, start, end=None, frequency="weekly"):
        have, missing = [], []
        frames = []
        for t in tickers:
            f = self.index_dir / f"{t}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df.columns = [c.lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                if "adj_close" not in df.columns and "close" in df.columns:
                    df["adj_close"] = df["close"]
                for c in PRICE_COLUMNS:
                    if c not in df.columns:
                        df[c] = df["adj_close"]
                df["ticker"] = t
                frames.append(df[["date", "ticker", *PRICE_COLUMNS]])
                have.append(t)
            else:
                missing.append(t)
        if missing:
            logger.warning("Index data missing for %s -> falling back to ETF", ", ".join(missing))
            fb = self.fallback.get_prices(missing, start=start, end=end, frequency="daily")
            frames.append(fb)
        if have:
            logger.warning("Using INDEX data for %s", ", ".join(have))
        daily = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
        daily = daily[daily["date"] >= pd.Timestamp(start)]
        return daily if frequency == "daily" else resample_to_weekly(daily)


class YFinanceProvider(MarketDataProvider):
    def get_prices(self, tickers, start, end=None, frequency="weekly"):
        frames = []
        for ticker in tickers:
            logger.info("Downloading %s from yfinance", ticker)
            raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            if raw.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
            })
            df = df.reset_index().rename(columns={"Date": "date"})
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df["ticker"] = ticker.replace("^", "")
            if "adj_close" not in df.columns and "close" in df.columns:
                df["adj_close"] = df["close"]
            frames.append(df[["date", "ticker", *PRICE_COLUMNS]])
        daily = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
        return daily if frequency == "daily" else resample_to_weekly(daily)


def _fetch_fred_csv(series_id: str, retries: int = 3, pause: float = 1.0, cache_path: Path | None = None) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["curl", "-fsSL", "--max-time", "30", url],
                capture_output=True, text=True, check=True,
            )
            df = pd.read_csv(StringIO(result.stdout))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
            return df
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            last_err = exc
            logger.warning("FRED fetch %s attempt %d failed: %s", series_id, attempt + 1, exc)
            time.sleep(pause)
    raise RuntimeError(f"Failed to fetch FRED series {series_id}") from last_err


def build_proxy_macro(market_weekly: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Fallback macro panel derived from market data when FRED is unavailable."""
    dates = pd.to_datetime(market_weekly["date"].unique())
    vix = market_weekly[market_weekly["ticker"].isin(["VIX", "^VIX"])]
    vix_series = vix.set_index("date")["adj_close"] if not vix.empty else pd.Series(20.0, index=dates)
    rows = []
    for d in dates:
        for sid in series:
            vix_val = float(vix_series.get(d, 20.0))
            rows.append({"date": d, "series": sid, "value": vix_val if sid == "BAA10Y" else vix_val / 100.0})
    return pd.DataFrame(rows)


class FredProvider(MacroDataProvider):
    """Fetch macro series from FRED public CSV endpoint (no API key required)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir

    def get_macro(self, series, start, end=None, frequency="weekly"):
        frames = []
        for sid in series:
            try:
                cache_path = (self.cache_dir / f"fred_{sid}.csv") if self.cache_dir else None
                s = _fetch_fred_csv(sid, cache_path=cache_path)
                s = s.rename(columns={"observation_date": "date", "DATE": "date", sid: "value"})
                s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
                s["value"] = pd.to_numeric(s["value"], errors="coerce")
                s = s[s["date"] >= pd.Timestamp(start)]
                if end is not None:
                    s = s[s["date"] <= pd.Timestamp(end)]
                s["series"] = sid
                frames.append(s[["date", "series", "value"]])
            except Exception as exc:
                logger.warning("Skipping FRED series %s: %s", sid, exc)
        if not frames:
            raise RuntimeError("No FRED series could be downloaded")
        daily = pd.concat(frames, ignore_index=True).sort_values(["series", "date"]).reset_index(drop=True)
        return daily if frequency == "daily" else resample_macro_to_weekly(daily)


def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV panel to weekly (Friday close)."""
    parts = []
    for ticker, grp in daily.groupby("ticker"):
        g = grp.set_index("date").sort_index()
        weekly = pd.DataFrame({
            "open": g["open"].resample("W-FRI").first(),
            "high": g["high"].resample("W-FRI").max(),
            "low": g["low"].resample("W-FRI").min(),
            "close": g["close"].resample("W-FRI").last(),
            "adj_close": g["adj_close"].resample("W-FRI").last(),
            "volume": g["volume"].resample("W-FRI").sum(),
        }).dropna(subset=["adj_close"]).reset_index()
        weekly["ticker"] = ticker
        parts.append(weekly)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def resample_macro_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for series, grp in daily.groupby("series"):
        g = grp.set_index("date").sort_index()
        weekly = g["value"].resample("W-FRI").last().dropna().reset_index()
        weekly["series"] = series
        parts.append(weekly)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "series"]).reset_index(drop=True)


def ingest_market_data(tickers, vix_ticker, start, end, raw_dir, processed_dir,
                       provider=None, *, use_cache=True) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_weekly = processed_dir / "market_weekly.parquet"
    if use_cache and cache_weekly.exists():
        logger.warning("Using CACHED market data from %s", cache_weekly)
        return pd.read_parquet(cache_weekly)
    all_tickers = list(dict.fromkeys(tickers + [vix_ticker]))
    daily = (provider or YFinanceProvider()).get_prices(all_tickers, start=start, end=end, frequency="daily")
    weekly = resample_to_weekly(daily)
    daily.to_parquet(raw_dir / "market_daily.parquet", index=False)
    weekly.to_parquet(cache_weekly, index=False)
    logger.info("Saved market data: %d weekly rows", len(weekly))
    return weekly


def ingest_macro_data(series, start, end, raw_dir, processed_dir, provider=None,
                      *, market_weekly=None, use_cache=True) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_weekly = processed_dir / "macro_weekly.parquet"
    if use_cache and cache_weekly.exists():
        logger.warning("Using CACHED macro data from %s", cache_weekly)
        return pd.read_parquet(cache_weekly)
    try:
        provider = provider or FredProvider(cache_dir=raw_dir)
        daily = provider.get_macro(series, start=start, end=end, frequency="daily")
        missing = [sid for sid in series if sid not in set(daily["series"])]
        if missing and market_weekly is not None:
            logger.warning("FRED missed %s; filling with market-derived proxies", ", ".join(missing))
            daily = pd.concat([daily, build_proxy_macro(market_weekly, missing)], ignore_index=True)
        weekly = resample_macro_to_weekly(daily)
        weekly.to_parquet(cache_weekly, index=False)
        logger.info("Saved macro data: %d weekly rows", len(weekly))
        return weekly
    except Exception as exc:
        if market_weekly is not None:
            logger.warning("Macro fetch failed (%s); using proxy macro", exc)
            weekly = resample_macro_to_weekly(build_proxy_macro(market_weekly, series))
            weekly.to_parquet(cache_weekly, index=False)
            return weekly
        raise
