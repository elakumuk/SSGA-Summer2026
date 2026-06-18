"""M2 — the DYNAMIC, regime-aware meta-labeling layer.

This is where meta-labeling actually kicks in. M2 asks, per period and per asset:
"given the current REGIME, how much do I trust M1's view here?"

  * meta-label = did M1's benchmark-relative bet pay? (overweight that beat the
    basket, or underweight that lagged it -> success)
  * features   = M1 score + macro regime features (vix, curve, credit, growth,
    inflation). M2 learns in WHICH regimes the M1 score is trustworthy.
  * training   = rolling ~12-month (refit_window_weeks) window, refit periodically,
    strict no-look-ahead + embargo so the 4-week label cannot leak.
  * output     = P(success) per (date, ticker) -> a sizing multiplier in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import Config


def build_meta_labels(prices: pd.DataFrame, weights: pd.DataFrame, horizon: int,
                      benchmark_w: float, cost: float) -> pd.DataFrame:
    """1 if M1's active bet paid on a benchmark-relative basis, else 0 (wide)."""
    fwd = prices.pct_change(horizon).shift(-horizon)          # forward h-week return
    active_fwd = fwd.sub(fwd.mean(axis=1), axis=0)            # vs equal-weight basket
    tilt = weights - benchmark_w                              # over/underweight sign
    paid = np.sign(tilt) * active_fwd
    return (paid > cost).astype(float).where(tilt.abs() > 1e-9)


def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
    return wide.stack().rename(name)


def build_feature_matrix(factors: dict[str, pd.DataFrame], regime: pd.DataFrame) -> pd.DataFrame:
    """Long (date,ticker) feature matrix for M2: the INDIVIDUAL M1 factor scores
    (momentum, trend, macro -- kept separate, NOT pre-merged) + regime features.

    This is what lets M2 do the dynamic factor-timing the mentor described: "figure
    during which period which factor does better" -- M2 can only weight momentum vs
    trend separately if it sees them separately. Regime values are broadcast across
    tickers per date."""
    ref = next(iter(factors.values()))
    feats = None
    for name, frame in factors.items():
        s = _stack(frame, name)
        feats = s.to_frame() if feats is None else feats.join(s, how="outer")
    reg = regime.reindex(ref.index).ffill()
    for col in reg.columns:
        feats[col] = feats.index.get_level_values("date").map(reg[col]).astype(float).values
    return feats


class M2Model:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.horizon = cfg.labels.horizon_weeks
        self.embargo = cfg.split.embargo_weeks
        self.lookback = cfg.m2.refit_window_weeks
        self.refit_cadence = 4          # refit quarterly-ish; reuse model in between
        self.min_samples = 60

    def run_rolling(self, features: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
        """Walk forward week by week. At prediction date d, train only on bets whose
        4-week label fully resolved at least `embargo` weeks before d."""
        y = _stack(labels, "y")
        df = features.join(y, how="left")
        feat_cols = list(features.columns)
        dates = df.index.get_level_values("date")
        unique_dates = pd.Index(sorted(dates.unique()))

        proba = pd.Series(np.nan, index=df.index, name="m2_proba")
        model: LogisticRegression | None = None
        scaler: StandardScaler | None = None

        for i, d in enumerate(unique_dates):
            # labels are observable only `horizon` weeks after entry; add embargo
            cutoff = unique_dates[max(0, i - self.horizon - self.embargo)]
            train = df[(dates <= cutoff) & df["y"].notna()].dropna(subset=feat_cols)
            if self.lookback:  # rolling 12-month window
                lb_start = unique_dates[max(0, i - self.horizon - self.embargo - self.lookback)]
                train = train[train.index.get_level_values("date") >= lb_start]

            # (re)fit on cadence, if we have enough samples spanning both classes
            if (i % self.refit_cadence == 0) and len(train) >= self.min_samples and train["y"].nunique() == 2:
                scaler = StandardScaler().fit(train[feat_cols].values)
                model = LogisticRegression(max_iter=1000, C=1.0).fit(
                    scaler.transform(train[feat_cols].values), train["y"].values
                )

            if model is None:
                continue
            cur = df[(dates == d)].dropna(subset=feat_cols)
            if cur.empty:
                continue
            proba.loc[cur.index] = model.predict_proba(scaler.transform(cur[feat_cols].values))[:, 1]

        return proba

    def size(self, proba: pd.Series, train_proba: pd.Series | None = None) -> pd.Series:
        """Map P(success) -> sizing multiplier in [0, 1]."""
        mode = self.cfg.m2.sizing
        if mode == "binary":
            return (proba > 0.5).astype(float)
        if mode == "linear":
            return (2 * proba - 1).clip(lower=0.0)
        # ecdf: rank current proba against training-history distribution
        ref = (train_proba if train_proba is not None else proba).dropna().sort_values().values
        if len(ref) == 0:
            return proba.fillna(0.0)
        return proba.apply(lambda p: np.searchsorted(ref, p) / len(ref) if pd.notna(p) else 0.0)
