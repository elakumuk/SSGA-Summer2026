"""Feature engineering with strict no-lookahead controls."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

LABEL_COLUMNS = {
    "forward_return",
    "m1_target",
    "meta_label",
    "trade_return",
}


def _pivot_prices(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def _momentum_features(prices: pd.DataFrame, windows: list[int]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for w in windows:
        mom = prices.pct_change(w).shift(1)
        out[f"mom_{w}w"] = mom
    return out


def _trend_features(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    ma_fast = prices.rolling(fast).mean()
    ma_slow = prices.rolling(slow).mean()
    trend = (ma_fast / ma_slow - 1).shift(1)
    return trend


def _volatility_features(returns: pd.DataFrame, windows: list[int]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for w in windows:
        out[f"vol_{w}w"] = (returns.rolling(w).std() * np.sqrt(52)).shift(1)
    return out


def _drawdown_feature(prices: pd.DataFrame, window: int = 26) -> pd.DataFrame:
    roll_max = prices.rolling(window).max()
    return (prices / roll_max - 1).shift(1)


def _corr_to_spy(returns: pd.DataFrame, window: int = 26) -> pd.DataFrame:
    if "SPY" not in returns.columns:
        return pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    spy = returns["SPY"]
    corrs = {}
    for col in returns.columns:
        corrs[col] = returns[col].rolling(window).corr(spy).shift(1)
    return pd.DataFrame(corrs)


def _cross_sectional_rank(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.rank(axis=1, pct=True)


def _cross_sectional_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    mean = wide.mean(axis=1)
    std = wide.std(axis=1).replace(0, np.nan)
    return wide.sub(mean, axis=0).div(std, axis=0)


def _macro_wide(macro_weekly: pd.DataFrame, lag_weeks: int) -> pd.DataFrame:
    wide = macro_weekly.pivot(index="date", columns="series", values="value").sort_index()
    wide = wide.ffill().shift(lag_weeks)
    wide.index = pd.to_datetime(wide.index)
    return wide


def _macro_regime_features(macro: pd.DataFrame, vix: pd.Series | None) -> pd.DataFrame:
    out = pd.DataFrame(index=macro.index)
    if "CPIAUCSL" in macro.columns:
        cpi_yoy = macro["CPIAUCSL"].pct_change(52)
        out["inflation_trend"] = cpi_yoy
        out["inflation_up"] = (cpi_yoy > cpi_yoy.rolling(156).median()).astype(float)
    if "INDPRO" in macro.columns:
        growth = macro["INDPRO"].pct_change(52)
        out["growth_trend"] = growth
        out["growth_down"] = (growth < growth.rolling(156).median()).astype(float)
    if "T10Y2Y" in macro.columns:
        out["yield_curve"] = macro["T10Y2Y"]
        out["curve_inverted"] = (macro["T10Y2Y"] < 0).astype(float)
    if "BAA10Y" in macro.columns:
        out["credit_stress"] = macro["BAA10Y"]
    if "FEDFUNDS" in macro.columns:
        out["policy_rate_change"] = macro["FEDFUNDS"].diff(4)
    if "UNRATE" in macro.columns:
        out["unemployment_change"] = macro["UNRATE"].diff(12)
    if vix is not None:
        vix = vix.reindex(macro.index).ffill()
        out["vix_level"] = vix.shift(1)
        out["vix_change_4w"] = vix.pct_change(4).shift(1)
        out["risk_off"] = (vix > vix.rolling(156).quantile(0.75)).astype(float).shift(1)
    return out


def _average_pairwise_correlation(returns: pd.DataFrame, window: int = 26) -> pd.Series:
    avg_corr = pd.Series(np.nan, index=returns.index)
    for i in range(window, len(returns)):
        corr = returns.iloc[i - window : i].corr()
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        avg_corr.iloc[i] = corr.where(mask).stack().mean()
    return avg_corr.shift(1)


def _dispersion_features(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    disp4 = returns.rolling(4).std().mean(axis=1).shift(1)
    disp12 = returns.rolling(12).std().mean(axis=1).shift(1)
    avg_corr = _average_pairwise_correlation(returns, 26)
    return {
        "cross_asset_dispersion_4w": pd.DataFrame({c: disp4 for c in returns.columns}, index=returns.index),
        "cross_asset_dispersion_12w": pd.DataFrame({c: disp12 for c in returns.columns}, index=returns.index),
        "average_pairwise_correlation_26w": pd.DataFrame({c: avg_corr for c in returns.columns}, index=returns.index),
    }


_CARRY_ASSET_CLASS: dict[str, str] = {
    "SPY": "equity",
    "VEA": "equity",
    "VWO": "em_equity",
    "VNQ": "reit",
    "TLT": "bond",
    "HYG": "credit",
    "GLD": "gold",
}


def _carry_features(
    macro_wide: pd.DataFrame,
    tickers: list[str],
    *,
    zscore_window: int = 156,
) -> dict[str, pd.DataFrame]:
    """Per-asset carry score derived from FRED macro proxies.

    Koijen, Moskowitz, Pedersen & Vrugt (2018) — *Carry*. Journal of Financial
    Economics 127(2). Carry is defined as the asset's expected return
    conditional on the underlying spot price staying constant. We use macro
    proxies because asset-level yield data (TIPS, futures roll, dividend yield
    surface) isn't in the FRED panel:

        TLT  → term premium       (DGS10 − FEDFUNDS)
        HYG  → credit spread      (BAA10Y)
        GLD  → negative real rate (CPI YoY − DGS10)
        SPY, VEA, VWO, VNQ → negative policy rate (-FEDFUNDS) as a coarse
            proxy for the carry-on-discount-rate logic of dividend-yielding
            risk assets in a low-rate regime. VWO gets a small EM premium add.

    Each per-asset proxy is rolling z-scored against its own 3-year history so
    the four pillars (mom / trend / macro / carry) live on compatible scales,
    then `.shift(1)` enforces no-lookahead.
    """
    if macro_wide.empty:
        return {}
    needed = {"DGS10", "FEDFUNDS", "BAA10Y", "CPIAUCSL"}
    if not needed.issubset(macro_wide.columns):
        return {}

    term_premium = macro_wide["DGS10"] - macro_wide["FEDFUNDS"]
    credit_spread = macro_wide["BAA10Y"]
    cpi_yoy = macro_wide["CPIAUCSL"].pct_change(52) * 100
    real_rate = macro_wide["DGS10"] - cpi_yoy
    neg_policy = -macro_wide["FEDFUNDS"]

    per_asset: dict[str, pd.Series] = {}
    for ticker in tickers:
        ac = _CARRY_ASSET_CLASS.get(ticker, "equity")
        if ac == "bond":
            s = term_premium
        elif ac == "credit":
            s = credit_spread
        elif ac == "gold":
            s = -real_rate
        elif ac == "em_equity":
            s = neg_policy + 2.0  # ~200 bps premium for EM yield differential
        elif ac == "reit":
            s = neg_policy - 0.5 * macro_wide["DGS10"]  # REITs hurt by long rates too
        else:  # equity
            s = neg_policy
        per_asset[ticker] = s

    raw = pd.DataFrame(per_asset, index=macro_wide.index)
    mu = raw.rolling(zscore_window, min_periods=52).mean()
    sd = raw.rolling(zscore_window, min_periods=52).std().replace(0, np.nan)
    z = ((raw - mu) / sd).shift(1)

    return {"carry_score": z}


def _wide_to_long(feature_dict: dict[str, pd.DataFrame], dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for name, wide in feature_dict.items():
        if isinstance(wide, pd.Series):
            wide = wide.to_frame("SPY") if wide.name is None else wide.to_frame(wide.name)
        long = wide.stack().reset_index()
        long.columns = ["date", "ticker", name]
        rows.append(long)
    if not rows:
        return pd.DataFrame(columns=["date", "ticker"])
    merged = rows[0]
    for r in rows[1:]:
        merged = merged.merge(r, on=["date", "ticker"], how="outer")
    merged["date"] = pd.to_datetime(merged["date"])
    return merged


def winsorize_train_features(
    panel: pd.DataFrame,
    feature_cols: list[str],
    train_end: str,
    pct: float = 0.01,
) -> pd.DataFrame:
    out = panel.copy()
    train_mask = out["date"] <= pd.Timestamp(train_end)
    for col in feature_cols:
        if col not in out.columns:
            continue
        train_vals = out.loc[train_mask, col].dropna()
        if train_vals.empty:
            continue
        lo = train_vals.quantile(pct)
        hi = train_vals.quantile(1 - pct)
        out[col] = out[col].clip(lo, hi)
    return out


def build_features(
    market_weekly: pd.DataFrame,
    macro_weekly: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    vix_ticker: str = "VIX",
) -> pd.DataFrame:
    tickers = cfg.assets.tickers
    panel = market_weekly[market_weekly["ticker"].isin(tickers)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"])

    prices = _pivot_prices(panel)
    returns = prices.pct_change()
    panel["return_1w"] = panel.apply(
        lambda r: returns.loc[r["date"], r["ticker"]] if r["date"] in returns.index and r["ticker"] in returns.columns else np.nan,
        axis=1,
    )

    feature_frames: dict[str, pd.DataFrame] = {}
    feature_frames.update(_momentum_features(prices, cfg.features.momentum_windows))
    feature_frames["trend_signal"] = _trend_features(prices, cfg.features.trend_windows[0], cfg.features.trend_windows[1])
    feature_frames.update(_volatility_features(returns, cfg.features.volatility_windows))
    feature_frames["drawdown_26w"] = _drawdown_feature(prices, 26)
    feature_frames["corr_to_spy_26w"] = _corr_to_spy(returns, 26)

    mom12 = prices.pct_change(12).shift(1)
    feature_frames["rank_mom_12w"] = _cross_sectional_rank(mom12)
    ew_ret = returns.mean(axis=1)
    ew_mom12 = ew_ret.rolling(12).sum().shift(1)
    rel_mom = mom12.sub(ew_mom12, axis=0)
    feature_frames["rel_mom_12w"] = rel_mom
    vol12 = feature_frames.get("vol_12w")
    if vol12 is not None and "mom_12w" in feature_frames:
        inv_vol = (1.0 / vol12.replace(0, np.nan)).clip(upper=5.0)
        feature_frames["mom_vol_interaction"] = feature_frames["mom_12w"] * inv_vol
    for key in ["mom_12w", "mom_26w", "mom_52w", "trend_signal", "vol_12w", "drawdown_26w"]:
        if key in feature_frames:
            feature_frames[f"z_{key}"] = _cross_sectional_zscore(feature_frames[key])

    feature_frames.update(_dispersion_features(returns))

    macro_wide = _macro_wide(macro_weekly, cfg.features.macro_lag_weeks)
    carry_frames = _carry_features(macro_wide, tickers)
    # Reindex the carry score to the price calendar so it merges cleanly.
    for name, frame in carry_frames.items():
        feature_frames[name] = frame.reindex(prices.index).ffill()
    if "DGS10" in macro_wide.columns:
        carry = macro_wide["DGS10"].reindex(prices.index).ffill()
        feature_frames["carry_yield_level"] = pd.DataFrame(
            {c: carry for c in prices.columns}, index=prices.index
        )
    if "BAA10Y" in macro_wide.columns:
        credit_chg = macro_wide["BAA10Y"].diff(4).reindex(prices.index).ffill()
        feature_frames["credit_carry_chg"] = pd.DataFrame(
            {c: credit_chg for c in prices.columns}, index=prices.index
        )
    vix_data = market_weekly[market_weekly["ticker"] == vix_ticker.replace("^", "")]
    vix_series = None
    if not vix_data.empty:
        vix_series = vix_data.set_index("date")["adj_close"].sort_index()
    regime = _macro_regime_features(macro_wide, vix_series)

    feat_long = _wide_to_long(feature_frames, prices.index, tickers)
    panel = panel.merge(feat_long, on=["date", "ticker"], how="left")

    for col in regime.columns:
        panel[col] = panel["date"].map(regime[col])

    if getattr(cfg.features, "hmm_regime_enabled", False):
        from src.regime import fit_macro_regime, regime_features_long

        try:
            regime_fit = fit_macro_regime(
                macro_wide,
                train_end=cfg.split.train_end,
                n_states=cfg.features.hmm_regime_states,
                random_state=cfg.features.hmm_regime_random_state,
            )
            regime_long = regime_features_long(regime_fit, tickers)
            panel = panel.merge(regime_long, on=["date", "ticker"], how="left")
            logger.info(
                "HMM regime features merged: %d posterior cols + 2 tilt cols",
                len(regime_fit.posteriors.columns),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("HMM regime fit failed (%s); continuing without regime features.", exc)

    feature_cols = [c for c in panel.columns if c not in {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "return_1w"}]
    panel = winsorize_train_features(panel, feature_cols, cfg.split.train_end, cfg.features.winsorize_pct)

    panel = panel.set_index(["date", "ticker"]).sort_index()
    logger.info("Built feature panel: %d rows, %d feature columns", len(panel), len(feature_cols))
    return panel


def save_model_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.reset_index().to_parquet(path, index=False)


def get_feature_columns(panel: pd.DataFrame, exclude_labels: bool = True) -> list[str]:
    exclude = {
        "adj_close",
        "return_1w",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "M1_signal",
        "M1_score",
        "M1_conviction",
        "p_success",
        "predicted_meta_label",
    }
    if exclude_labels:
        exclude |= LABEL_COLUMNS
        exclude |= {c for c in panel.columns if c.startswith("forward_return")}
    return [c for c in panel.columns if c not in exclude and panel[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
