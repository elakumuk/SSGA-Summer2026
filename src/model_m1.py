"""M1 primary side model."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.config import M1Config, PipelineConfig

logger = logging.getLogger(__name__)

ASSET_CLASS_MAP: dict[str, str] = {
    "SPY": "equity",
    "VEA": "equity",
    "VWO": "equity",
    "VNQ": "reit",
    "TLT": "bond",
    "HYG": "credit",
    "GLD": "gold",
}


class M1Model(ABC):
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        *,
        forward_returns: pd.Series | None = None,
    ) -> M1Model:
        ...

    @abstractmethod
    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        ...


class RuleBasedM1(M1Model):
    def __init__(self, cfg: M1Config) -> None:
        self.cfg = cfg
        self.long_threshold = cfg.long_threshold
        self.short_threshold = cfg.short_threshold
        self.weights = cfg.weights

    def _col(self, X: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        if name in X.columns:
            return X[name].fillna(default)
        return pd.Series(default, index=X.index)

    def _momentum_score(self, X: pd.DataFrame) -> pd.Series:
        z_cols = [c for c in X.columns if c in ("z_mom_12w", "z_mom_26w", "z_mom_52w")]
        parts: list[pd.Series] = []
        if z_cols:
            parts.append(X[z_cols].mean(axis=1))
        if "rank_mom_12w" in X.columns:
            # Cross-sectional rank in [0, 1] -> roughly [-0.5, 0.5]
            parts.append(X["rank_mom_12w"].fillna(0.5) - 0.5)
        if not parts:
            return pd.Series(0.0, index=X.index)
        return sum(parts) / len(parts)

    def _trend_score(self, X: pd.DataFrame) -> pd.Series:
        return self._col(X, "z_trend_signal")

    def _risk_penalty(self, X: pd.DataFrame) -> pd.Series:
        vol = self._col(X, "z_vol_12w")
        # Drawdown feature is negative in stress; penalize low (more negative) values
        dd = self._col(X, "drawdown_26w")
        dd_penalty = (-dd).clip(lower=0)
        if "z_drawdown_26w" in X.columns:
            dd_penalty = dd_penalty + self._col(X, "z_drawdown_26w").clip(lower=0)
        return vol + dd_penalty

    def _asset_class_macro_tilt(self, X: pd.DataFrame) -> pd.Series:
        """Asset-class-specific macro regime tilts instead of one global macro average."""
        if not self.cfg.asset_class_tilts:
            macro_cols = [c for c in X.columns if c in ("inflation_trend", "growth_trend", "yield_curve", "credit_stress", "risk_off")]
            if macro_cols:
                return X[macro_cols].mean(axis=1)
            return pd.Series(0.0, index=X.index)

        if not isinstance(X.index, pd.MultiIndex) or "ticker" not in X.index.names:
            macro_cols = [c for c in ("growth_trend", "risk_off", "inflation_up") if c in X.columns]
            return X[macro_cols].mean(axis=1) if macro_cols else pd.Series(0.0, index=X.index)

        tickers = X.index.get_level_values("ticker")
        tilt = pd.Series(0.0, index=X.index)
        growth = self._col(X, "growth_trend")
        risk_off = self._col(X, "risk_off")
        inflation_up = self._col(X, "inflation_up")
        curve_inv = self._col(X, "curve_inverted")
        credit = self._col(X, "credit_stress")

        for ticker in tickers.unique():
            mask = tickers == ticker
            asset_class = ASSET_CLASS_MAP.get(ticker, "equity")
            if asset_class == "equity":
                tilt.loc[mask] = (0.35 * growth - 0.35 * risk_off - 0.15 * inflation_up).loc[mask]
            elif asset_class == "reit":
                tilt.loc[mask] = (0.25 * growth - 0.30 * risk_off - 0.25 * inflation_up).loc[mask]
            elif asset_class == "bond":
                tilt.loc[mask] = (0.40 * risk_off + 0.30 * curve_inv - 0.30 * inflation_up).loc[mask]
            elif asset_class == "credit":
                tilt.loc[mask] = (0.30 * growth - 0.40 * credit - 0.20 * risk_off).loc[mask]
            elif asset_class == "gold":
                tilt.loc[mask] = (0.40 * inflation_up + 0.35 * risk_off - 0.15 * growth).loc[mask]
        return tilt

    def _component_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        comps = pd.DataFrame(index=X.index)
        comps["momentum_score"] = self._momentum_score(X)
        comps["trend_score"] = self._trend_score(X)
        comps["risk_penalty"] = self._risk_penalty(X)
        comps["macro_score"] = self._asset_class_macro_tilt(X)
        return comps

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        comps = self._component_scores(X)
        w = self.weights
        score = (
            w.get("momentum", 0.45) * comps["momentum_score"]
            + w.get("trend", 0.25) * comps["trend_score"]
            + w.get("macro", 0.20) * comps["macro_score"]
            - w.get("risk_penalty", 0.10) * comps["risk_penalty"]
        )
        return score.rename("M1_score")

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        score = self.predict_score(X)
        if self.cfg.allow_short:
            signals = np.where(
                score > self.long_threshold,
                1,
                np.where(score < self.short_threshold, -1, 0),
            )
        else:
            signals = np.where(score > self.long_threshold, 1, 0)
        return pd.Series(signals, index=X.index, name="M1_signal")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        *,
        forward_returns: pd.Series | None = None,
    ) -> RuleBasedM1:
        scores = self.predict_score(X)
        if self.cfg.optimize_thresholds and forward_returns is not None:
            self.long_threshold, self.short_threshold = tune_thresholds(
                scores, forward_returns, self.cfg
            )
        elif y is not None:
            self.long_threshold, self.short_threshold = tune_thresholds(
                scores, None, self.cfg, fallback_quantiles=True
            )
        nonzero = (self.predict_signal(X) != 0).sum()
        if nonzero < self.cfg.min_nonzero_signals:
            warnings.warn(
                f"M1 generated only {nonzero} non-zero signals (min {self.cfg.min_nonzero_signals})",
                stacklevel=2,
            )
        logger.info(
            "M1 thresholds: long=%.3f short=%.3f (allow_short=%s)",
            self.long_threshold,
            self.short_threshold,
            self.cfg.allow_short,
        )
        return self


def tune_thresholds(
    scores: pd.Series,
    forward_returns: pd.Series | None,
    cfg: M1Config,
    *,
    fallback_quantiles: bool = False,
) -> tuple[float, float]:
    """Tune thresholds on training data to maximize mean trade return."""
    valid = scores.dropna()
    if valid.empty:
        return cfg.long_threshold, cfg.short_threshold

    if fallback_quantiles or forward_returns is None:
        long_t = float(valid.quantile(cfg.long_quantile))
        short_t = float(valid.quantile(cfg.short_quantile))
        if not cfg.allow_short:
            short_t = float(valid.min() - 1.0)
        return long_t, short_t

    aligned = pd.DataFrame({"score": scores, "fwd": forward_returns}).dropna()
    if aligned.empty:
        return cfg.long_threshold, cfg.short_threshold

    best_long, best_short = cfg.long_threshold, cfg.short_threshold
    best_obj = -np.inf

    long_quantiles = np.arange(cfg.long_quantile_min, cfg.long_quantile_max + 1e-9, cfg.quantile_step)
    short_quantiles = np.arange(cfg.short_quantile_min, cfg.short_quantile_max + 1e-9, cfg.quantile_step)

    for long_q in long_quantiles:
        long_t = float(aligned["score"].quantile(long_q))
        if cfg.allow_short:
            short_candidates = short_quantiles
        else:
            short_candidates = [0.0]

        for short_q in short_candidates:
            if cfg.allow_short:
                short_t = float(aligned["score"].quantile(short_q))
                if long_t <= short_t:
                    continue
            else:
                short_t = float(aligned["score"].min() - 1.0)

            sig = np.where(
                aligned["score"] > long_t,
                1,
                np.where(aligned["score"] < short_t, -1, 0),
            )
            mask = sig != 0
            if mask.sum() < cfg.min_nonzero_signals:
                continue

            trade_ret = sig * aligned["fwd"].values
            mean_ret = trade_ret[mask].mean()
            hit_rate = (trade_ret[mask] > 0).mean()
            # Favor profitable signals with reasonable hit rate
            obj = mean_ret + 0.25 * (hit_rate - 0.5)

            if cfg.allow_short:
                long_mask = sig == 1
                short_mask = sig == -1
                if long_mask.sum() > 0 and short_mask.sum() > 0:
                    # Penalize shorts that lose money on average in training
                    short_mean = trade_ret[short_mask].mean()
                    if short_mean < 0:
                        obj += short_mean * 0.5

            if obj > best_obj:
                best_obj = obj
                best_long, best_short = long_t, short_t

    if best_obj == -np.inf:
        return tune_thresholds(scores, None, cfg, fallback_quantiles=True)

    logger.info("M1 threshold tuning objective (mean trade return + hit-rate bonus): %.6f", best_obj)
    return best_long, best_short


def build_m1_model(cfg: PipelineConfig) -> M1Model:
    m1_type = cfg.m1.type
    if m1_type == "rule_based":
        return RuleBasedM1(cfg.m1)
    raise ValueError(f"Unsupported M1 type: {m1_type}")


def train_date_mask(dates: pd.Index | pd.Series, cfg: PipelineConfig) -> pd.Series:
    d = pd.to_datetime(dates)
    mask = (d >= pd.Timestamp(cfg.split.train_start)) & (d <= pd.Timestamp(cfg.split.train_end))
    return pd.Series(mask, index=dates if isinstance(dates, pd.Index) else None)


def test_date_mask(dates: pd.Index | pd.Series, cfg: PipelineConfig) -> pd.Series:
    d = pd.to_datetime(dates)
    mask = d >= pd.Timestamp(cfg.split.test_start)
    if cfg.split.test_end is not None:
        mask &= d <= pd.Timestamp(cfg.split.test_end)
    return pd.Series(mask, index=dates if isinstance(dates, pd.Index) else None)


def split_train_test(panel: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(panel.index, pd.MultiIndex):
        dates = panel.index.get_level_values("date")
        train = panel[train_date_mask(dates, cfg).values]
        test = panel[test_date_mask(dates, cfg).values]
    else:
        train = panel[train_date_mask(panel["date"], cfg).values]
        test = panel[test_date_mask(panel["date"], cfg).values]
    return train, test
