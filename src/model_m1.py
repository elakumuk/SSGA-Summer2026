"""M1 primary side model."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
        panel: pd.DataFrame | None = None,
        returns_wide: pd.DataFrame | None = None,
        portfolio_cfg: object | None = None,
    ) -> M1Model:
        ...

    @abstractmethod
    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        ...

    @abstractmethod
    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        ...

    def predict_conviction(self, X: pd.DataFrame) -> pd.Series:
        """Normalized M1 conviction in [0, 1]; default uniform when inactive."""
        scores = self.predict_score(X)
        return _score_to_conviction(scores, getattr(self, "_train_scores", None))


def _score_to_conviction(scores: pd.Series, train_scores: pd.Series | None) -> pd.Series:
    """Map scores to [0, 1] via train ECDF (fallback: cross-sectional rank per date)."""
    if train_scores is not None and len(train_scores.dropna()) > 0:
        sorted_train = np.sort(train_scores.dropna().values)
        n = len(sorted_train)

        def _ecdf(val: float) -> float:
            if np.isnan(val):
                return 0.0
            return float(np.searchsorted(sorted_train, val, side="right") / n)

        return scores.apply(_ecdf).rename("M1_conviction")

    if isinstance(scores.index, pd.MultiIndex) and "date" in scores.index.names:
        dates = scores.index.get_level_values("date")
        return scores.groupby(dates).rank(pct=True).fillna(0.5).rename("M1_conviction")
    ranks = scores.rank(pct=True)
    return ranks.fillna(0.5).rename("M1_conviction")


class RuleBasedM1(M1Model):
    def __init__(self, cfg: M1Config) -> None:
        self.cfg = cfg
        self.long_threshold = cfg.long_threshold
        self.short_threshold = cfg.short_threshold
        self.weights = cfg.weights
        self._train_scores: pd.Series | None = None

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
            parts.append(X["rank_mom_12w"].fillna(0.5) - 0.5)
        if "rel_mom_12w" in X.columns:
            parts.append(X["rel_mom_12w"].fillna(0.0))
        if not parts:
            return pd.Series(0.0, index=X.index)
        return sum(parts) / len(parts)

    def _trend_score(self, X: pd.DataFrame) -> pd.Series:
        return self._col(X, "z_trend_signal")

    def _carry_score(self, X: pd.DataFrame) -> pd.Series:
        """Per-asset carry score from `feature_engineering._carry_features`.

        Already a rolling 3y z-score (shifted), so this is just a column read.
        Returns zeros if the carry pillar is not in the panel — keeps the
        weighted-sum scoring graceful when the feature is disabled upstream.
        """
        return self._col(X, "carry_score")

    def _risk_penalty(self, X: pd.DataFrame) -> pd.Series:
        vol = self._col(X, "z_vol_12w")
        dd = self._col(X, "drawdown_26w")
        dd_penalty = (-dd).clip(lower=0)
        if "z_drawdown_26w" in X.columns:
            dd_penalty = dd_penalty + self._col(X, "z_drawdown_26w").clip(lower=0)
        return vol + dd_penalty

    def _hmm_regime_tilt(self, X: pd.DataFrame) -> pd.Series | None:
        """Asset-class-specific tilt sourced from HMM Bridgewater grid posteriors.

        Returns None when the regime feature is absent, so the caller can fall
        back to the FRED-flag tilt without changing behavior.
        """
        if not getattr(self.cfg, "use_hmm_regime", False):
            return None
        if "regime_growth_tilt" not in X.columns or "regime_inflation_tilt" not in X.columns:
            return None
        if not isinstance(X.index, pd.MultiIndex) or "ticker" not in X.index.names:
            return None

        g = self._col(X, "regime_growth_tilt")
        i = self._col(X, "regime_inflation_tilt")
        tickers = X.index.get_level_values("ticker")
        tilt = pd.Series(0.0, index=X.index)
        # Coefficients are sanity-checked against the four Bridgewater quadrants
        # to keep the sign correct in each:
        #   overheat   (G+, I+), goldilocks (G+, I-),
        #   stagflation(G-, I+), deflation  (G-, I-).
        # In particular, nominal bonds are penalized strongly when I+ — stagflation
        # historically the worst environment for duration.
        for ticker in tickers.unique():
            mask = tickers == ticker
            asset_class = ASSET_CLASS_MAP.get(ticker, "equity")
            if asset_class == "equity":
                tilt.loc[mask] = (0.50 * g - 0.20 * i).loc[mask]
            elif asset_class == "reit":
                tilt.loc[mask] = (0.40 * g - 0.30 * i).loc[mask]
            elif asset_class == "bond":
                tilt.loc[mask] = (-0.30 * g - 0.60 * i).loc[mask]
            elif asset_class == "credit":
                tilt.loc[mask] = (0.45 * g - 0.20 * i).loc[mask]
            elif asset_class == "gold":
                tilt.loc[mask] = (-0.20 * g + 0.55 * i).loc[mask]
        return tilt

    def _asset_class_macro_tilt(self, X: pd.DataFrame) -> pd.Series:
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

        hmm_tilt = self._hmm_regime_tilt(X)
        if hmm_tilt is not None:
            alpha = float(getattr(self.cfg, "hmm_regime_blend", 0.5))
            alpha = max(0.0, min(1.0, alpha))
            tilt = (1.0 - alpha) * tilt + alpha * hmm_tilt
        return tilt

    def _component_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        comps = pd.DataFrame(index=X.index)
        comps["momentum_score"] = self._momentum_score(X)
        comps["trend_score"] = self._trend_score(X)
        comps["risk_penalty"] = self._risk_penalty(X)
        comps["macro_score"] = self._asset_class_macro_tilt(X)
        comps["carry_score"] = self._carry_score(X)
        return comps

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        comps = self._component_scores(X)
        w = self.weights
        score = (
            w.get("momentum", 0.45) * comps["momentum_score"]
            + w.get("trend", 0.25) * comps["trend_score"]
            + w.get("macro", 0.20) * comps["macro_score"]
            + w.get("carry", 0.0) * comps["carry_score"]
            - w.get("risk_penalty", 0.10) * comps["risk_penalty"]
        )
        return score.rename("M1_score")

    def _signals_threshold(self, score: pd.Series) -> pd.Series:
        if self.cfg.allow_short:
            signals = np.where(
                score > self.long_threshold,
                1,
                np.where(score < self.short_threshold, -1, 0),
            )
        else:
            signals = np.where(score > self.long_threshold, 1, 0)
        return pd.Series(signals, index=score.index, name="M1_signal")

    def _signals_top_k(self, score: pd.Series) -> pd.Series:
        """Weekly cross-sectional top-K long (and bottom-K short if enabled)."""
        k = max(1, int(self.cfg.top_k))
        min_score = float(self.cfg.top_k_min_score)
        signals = pd.Series(0, index=score.index, dtype=int, name="M1_signal")

        if not isinstance(score.index, pd.MultiIndex) or "date" not in score.index.names:
            ranked = score.rank(ascending=False)
            long_idx = ranked[ranked <= k].index
            if min_score > 0:
                long_idx = score.loc[long_idx][score.loc[long_idx] > min_score].index
            signals.loc[long_idx] = 1
            if self.cfg.allow_short:
                short_ranked = score.rank(ascending=True)
                short_idx = short_ranked[short_ranked <= k].index
                if min_score > 0:
                    short_idx = score.loc[short_idx][score.loc[short_idx] < -min_score].index
                signals.loc[short_idx] = -1
            return signals

        dates = score.index.get_level_values("date")
        for date in pd.Index(dates).unique():
            mask = dates == date
            grp = score[mask]
            if grp.empty:
                continue
            long_candidates = grp.nlargest(k)
            if min_score > 0:
                long_candidates = long_candidates[long_candidates > min_score]
            signals.loc[long_candidates.index] = 1
            if self.cfg.allow_short:
                short_candidates = grp.nsmallest(k)
                if min_score > 0:
                    short_candidates = short_candidates[short_candidates < -min_score]
                # Avoid overwriting longs
                for idx in short_candidates.index:
                    if signals.loc[idx] == 0:
                        signals.loc[idx] = -1
        return signals

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        score = self.predict_score(X)
        if self.cfg.allocation_mode == "top_k":
            return self._signals_top_k(score)
        return self._signals_threshold(score)

    def predict_conviction(self, X: pd.DataFrame) -> pd.Series:
        if not self.cfg.conviction_sizing:
            signals = self.predict_signal(X)
            return signals.abs().astype(float).rename("M1_conviction")
        return _score_to_conviction(self.predict_score(X), self._train_scores)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        *,
        forward_returns: pd.Series | None = None,
        panel: pd.DataFrame | None = None,
        returns_wide: pd.DataFrame | None = None,
        portfolio_cfg: object | None = None,
    ) -> RuleBasedM1:
        scores = self.predict_score(X)
        self._train_scores = scores

        if self.cfg.optimize_thresholds and self.cfg.allocation_mode == "threshold":
            if (
                self.cfg.tune_objective == "portfolio"
                and forward_returns is not None
                and panel is not None
                and returns_wide is not None
                and portfolio_cfg is not None
            ):
                self.long_threshold, self.short_threshold = tune_thresholds_portfolio(
                    scores,
                    forward_returns,
                    panel,
                    returns_wide,
                    self.cfg,
                    portfolio_cfg,
                )
            elif forward_returns is not None:
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
            "M1 allocation=%s top_k=%s thresholds: long=%.3f short=%.3f (allow_short=%s)",
            self.cfg.allocation_mode,
            self.cfg.top_k,
            self.long_threshold,
            self.short_threshold,
            self.cfg.allow_short,
        )
        return self


class MLM1(M1Model):
    """Logistic regression M1 using engineered features; outputs probability-based scores."""

    def __init__(self, cfg: M1Config) -> None:
        self.cfg = cfg
        self.long_threshold = cfg.long_threshold
        self.short_threshold = cfg.short_threshold
        self._train_scores: pd.Series | None = None
        self._model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._scaler = StandardScaler()
        self._feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        *,
        forward_returns: pd.Series | None = None,
        panel: pd.DataFrame | None = None,
        returns_wide: pd.DataFrame | None = None,
        portfolio_cfg: object | None = None,
    ) -> MLM1:
        if y is None:
            raise ValueError("MLM1 requires m1_target labels for training")
        self._feature_cols = list(X.columns)
        X_arr = self._scaler.fit_transform(X.fillna(0).values)
        y_bin = (y > 0).astype(int).values
        self._model.fit(X_arr, y_bin)
        scores = self.predict_score(X)
        self._train_scores = scores

        if self.cfg.optimize_thresholds and self.cfg.allocation_mode == "threshold":
            if (
                self.cfg.tune_objective == "portfolio"
                and panel is not None
                and returns_wide is not None
                and portfolio_cfg is not None
            ):
                self.long_threshold, self.short_threshold = tune_thresholds_portfolio(
                    scores,
                    forward_returns,
                    panel,
                    returns_wide,
                    self.cfg,
                    portfolio_cfg,
                )
            elif forward_returns is not None:
                self.long_threshold, self.short_threshold = tune_thresholds(scores, forward_returns, self.cfg)
            else:
                self.long_threshold, self.short_threshold = tune_thresholds(
                    scores, None, self.cfg, fallback_quantiles=True
                )
        return self

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        cols = [c for c in self._feature_cols if c in X.columns]
        X_use = X.reindex(columns=self._feature_cols, fill_value=0.0) if self._feature_cols else X.fillna(0)
        if self._feature_cols:
            X_use = X_use.fillna(0)
        proba = self._model.predict_proba(self._scaler.transform(X_use.values))[:, 1]
        # Center around 0 for compatibility with threshold / top_k logic
        centered = proba - 0.5
        return pd.Series(centered, index=X.index, name="M1_score")

    def predict_signal(self, X: pd.DataFrame) -> pd.Series:
        score = self.predict_score(X)
        if self.cfg.allocation_mode == "top_k":
            rb = RuleBasedM1(self.cfg)
            rb.long_threshold = self.long_threshold
            rb.short_threshold = self.short_threshold
            return rb._signals_top_k(score)
        if self.cfg.allow_short:
            signals = np.where(
                score > self.long_threshold,
                1,
                np.where(score < self.short_threshold, -1, 0),
            )
        else:
            signals = np.where(score > self.long_threshold, 1, 0)
        return pd.Series(signals, index=score.index, name="M1_signal")

    def predict_conviction(self, X: pd.DataFrame) -> pd.Series:
        if not self.cfg.conviction_sizing:
            return self.predict_signal(X).abs().astype(float).rename("M1_conviction")
        return _score_to_conviction(self.predict_score(X), self._train_scores)


def _signals_from_thresholds(
    scores: pd.Series,
    long_t: float,
    short_t: float,
    allow_short: bool,
) -> pd.Series:
    if allow_short:
        sig = np.where(scores > long_t, 1, np.where(scores < short_t, -1, 0))
    else:
        sig = np.where(scores > long_t, 1, 0)
    return pd.Series(sig, index=scores.index, name="M1_signal")


def _portfolio_objective(
    returns: pd.Series,
    turnover: pd.Series,
    *,
    turnover_penalty: float,
    drawdown_penalty: float,
    drawdown_cap: float,
) -> float:
    from src.diagnostics import annualized_return, max_drawdown, sharpe_ratio

    r = returns.dropna()
    if len(r) < 20:
        return -np.inf
    sh = sharpe_ratio(r)
    ann_turn = float(turnover.mean() * 52)
    dd = abs(max_drawdown(r))
    dd_pen = max(0.0, dd - drawdown_cap) * drawdown_penalty
    return sh - turnover_penalty * ann_turn - dd_pen


def tune_thresholds_portfolio(
    scores: pd.Series,
    forward_returns: pd.Series | None,
    panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    cfg: M1Config,
    portfolio_cfg: object,
) -> tuple[float, float]:
    """Tune thresholds by maximizing train-period portfolio Sharpe minus penalties."""
    from src.backtest import _run_backtest
    from src.portfolio import apply_vol_target_wide, build_weights_from_signals

    best_long, best_short = cfg.long_threshold, cfg.short_threshold
    best_obj = -np.inf

    long_quantiles = np.arange(cfg.long_quantile_min, cfg.long_quantile_max + 1e-9, cfg.quantile_step)
    short_quantiles = np.arange(cfg.short_quantile_min, cfg.short_quantile_max + 1e-9, cfg.quantile_step)

    score_vals = scores.dropna()
    if score_vals.empty:
        return best_long, best_short

    for long_q in long_quantiles:
        long_t = float(score_vals.quantile(long_q))
        short_candidates = short_quantiles if cfg.allow_short else [0.0]
        for short_q in short_candidates:
            if cfg.allow_short:
                short_t = float(score_vals.quantile(short_q))
                if long_t <= short_t:
                    continue
            else:
                short_t = float(score_vals.min() - 1.0)

            sig = _signals_from_thresholds(scores, long_t, short_t, cfg.allow_short)
            if (sig != 0).sum() < cfg.min_nonzero_signals:
                continue

            w = build_weights_from_signals(
                panel,
                sig,
                conviction=pd.Series(1.0, index=sig.index),
                portfolio_cfg=portfolio_cfg,
            )
            w = apply_vol_target_wide(w, returns_wide, portfolio_cfg)
            bt = _run_backtest("tune", w, returns_wide, getattr(portfolio_cfg, "transaction_cost_bps", 5.0))
            obj = _portfolio_objective(
                bt.returns,
                bt.turnover,
                turnover_penalty=cfg.tune_turnover_penalty,
                drawdown_penalty=cfg.tune_drawdown_penalty,
                drawdown_cap=cfg.tune_drawdown_cap,
            )
            if obj > best_obj:
                best_obj = obj
                best_long, best_short = long_t, short_t

    if best_obj == -np.inf:
        return tune_thresholds(scores, forward_returns, cfg, fallback_quantiles=True)

    logger.info("M1 portfolio threshold tuning objective: %.6f", best_obj)
    return best_long, best_short


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
            obj = mean_ret + 0.25 * (hit_rate - 0.5)

            if cfg.allow_short:
                long_mask = sig == 1
                short_mask = sig == -1
                if long_mask.sum() > 0 and short_mask.sum() > 0:
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
    if m1_type == "ml":
        return MLM1(cfg.m1)
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
