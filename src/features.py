"""Feature engineering. No look-ahead everywhere (shift(1)).

Two clean factor families, matching State Street's directive:
  * TECHNICAL  -> momentum + trend MERGED into one cross-sectional score
                  (they are collinear; keeping them separate breaks attribution).
                  Feeds the STATIC linear M1.
  * MACRO      -> (a) a static asset-class tilt for M1, and
                  (b) regime features for the DYNAMIC M2 (regime-dating).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Asset-class map (only members present in the universe are used).
ASSET_CLASS = {
    "SPY": "equity", "VEA": "equity", "VWO": "equity",
    "TLT": "rates",
    "GLD": "gold",
    "HYG": "credit", "LQD": "credit",
    "VNQ": "reit",
    "DBC": "commodity",
}


# ---------- shared helpers ----------

def pivot_prices(panel: pd.DataFrame) -> pd.DataFrame:
    """Long market panel -> wide adj_close (index=date, cols=ticker)."""
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    return p.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change()


def _cross_sectional_z(wide: pd.DataFrame) -> pd.DataFrame:
    """Z-score across assets each week -> puts everything on a comparable scale."""
    mean = wide.mean(axis=1)
    std = wide.std(axis=1).replace(0, np.nan)
    return wide.sub(mean, axis=0).div(std, axis=0)


# ---------- TECHNICAL (momentum + trend, merged) ----------

def momentum_score(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Cross-sectional momentum score (avg of z-scored pct_change over windows)."""
    parts = [_cross_sectional_z(prices.pct_change(w).shift(1)) for w in windows]
    return sum(parts) / len(parts)


def trend_score(prices: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Cross-sectional trend score (fast/slow MA ratio, price-level independent)."""
    fast, slow = windows[0], windows[1]
    raw = (prices.rolling(fast).mean() / prices.rolling(slow).mean() - 1).shift(1)
    return _cross_sectional_z(raw)


def technical_score(prices: pd.DataFrame, momentum_windows: list[int], trend_windows: list[int]) -> pd.DataFrame:
    """M1's merged technical score: momentum and trend blended 50/50 (they are very
    close / collinear) -> ONE static signal for the simple linear M1. The INDIVIDUAL
    momentum and trend scores stay available separately for M2 to time dynamically."""
    return 0.5 * momentum_score(prices, momentum_windows) + 0.5 * trend_score(prices, trend_windows)


# ---------- RISK (volatility / asset-quality penalty) ----------

def risk_score(prices: pd.DataFrame, vol_windows: list[int]) -> pd.DataFrame:
    """Asset-quality / risk penalty for M1. HIGH score = LOW risk (good): low
    volatility and shallow drawdown. Cross-sectionally z-scored. No look-ahead."""
    returns = prices.pct_change()
    vol = sum(returns.rolling(w).std() for w in vol_windows) / len(vol_windows)
    vol = (vol * np.sqrt(52)).shift(1)
    drawdown = (prices / prices.rolling(26).max() - 1).shift(1)   # <= 0, deeper = worse
    quality = -_cross_sectional_z(vol) + _cross_sectional_z(drawdown)
    return _cross_sectional_z(quality)


# ---------- MACRO ----------

def macro_wide(macro_weekly: pd.DataFrame, lag_weeks: int = 4) -> pd.DataFrame:
    """Wide macro table, published with a 4-week lag (no look-ahead)."""
    wide = macro_weekly.pivot(index="date", columns="series", values="value").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide.ffill().shift(lag_weeks)


def regime_features(macro: pd.DataFrame, vix: pd.Series | None) -> pd.DataFrame:
    """Regime features for the DYNAMIC M2. Macro is used here to regime-date M2,
    NOT as an M1 factor (State Street directive)."""
    out = pd.DataFrame(index=macro.index)
    if "CPIAUCSL" in macro:
        out["inflation_trend"] = macro["CPIAUCSL"].pct_change(52)
    if "INDPRO" in macro:
        out["growth_trend"] = macro["INDPRO"].pct_change(52)
    if "T10Y2Y" in macro:
        out["yield_curve"] = macro["T10Y2Y"]
        out["curve_inverted"] = (macro["T10Y2Y"] < 0).astype(float)
    if "BAA10Y" in macro:
        out["credit_stress"] = macro["BAA10Y"]
    if "FEDFUNDS" in macro:
        out["policy_rate_change"] = macro["FEDFUNDS"].diff(4)
    if vix is not None:
        vix = vix.reindex(macro.index).ffill()
        out["vix"] = vix.shift(1)
        out["risk_off"] = (vix > vix.rolling(156).quantile(0.75)).astype(float).shift(1)
    return out


def macro_asset_tilt(prices: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """STATIC, rule-based asset-class macro tilt for M1 (no learning -> stays linear).
    Transparent regime->asset-class direction map. Returns a per-(date,ticker) score
    z-scored across assets so it lives on the same scale as the technical score."""
    idx, cols = prices.index, prices.columns
    tilt = pd.DataFrame(0.0, index=idx, columns=cols)
    reg = regime.reindex(idx).ffill()

    growth_up = (reg.get("growth_trend", pd.Series(0, idx)) > 0).astype(float)
    inflation_up = (reg.get("inflation_trend", pd.Series(0, idx)) > 0).astype(float)
    inverted = reg.get("curve_inverted", pd.Series(0.0, idx)).fillna(0.0)
    risk_off = reg.get("risk_off", pd.Series(0.0, idx)).fillna(0.0)

    for c in cols:
        ac = ASSET_CLASS.get(c)
        s = pd.Series(0.0, index=idx)
        if ac == "equity":
            s += growth_up - inverted - risk_off
        elif ac == "rates":
            s += (1 - growth_up) + inverted - inflation_up
        elif ac == "gold":
            s += inflation_up + risk_off
        elif ac == "credit":
            s += growth_up - risk_off
        elif ac == "reit":
            s += growth_up - inverted
        elif ac == "commodity":
            s += inflation_up + growth_up
        tilt[c] = s

    return _cross_sectional_z(tilt)


def get_vix_series(market_weekly: pd.DataFrame) -> pd.Series | None:
    vix = market_weekly[market_weekly["ticker"].isin(["VIX", "^VIX"])]
    if vix.empty:
        return None
    s = vix.copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["adj_close"].sort_index()
