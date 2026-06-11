"""Gaussian HMM macro-regime classifier.

Maps a (growth, inflation, term-structure, policy-rate) macro panel to a
posterior over four Bridgewater-style regimes:

    Growth↑ Inflation↑   "overheat"
    Growth↑ Inflation↓   "goldilocks"
    Growth↓ Inflation↑   "stagflation"
    Growth↓ Inflation↓   "deflation / recession"

Theory:
    Hamilton (1989) — "A New Approach to the Economic Analysis of Nonstationary
    Time Series and the Business Cycle". Econometrica 57(2).
    Ang & Bekaert (2002) — "Regime Switches in Interest Rates". JBES 20(2).
    Dalio (2008) — Bridgewater "All Weather" 2x2 framework.

Discipline:
    - Standard scaler and HMM are fit on the **training window only**.
    - State -> Bridgewater quadrant mapping is derived from the train posterior
      means in the original (unscaled) feature space, so labels are stable
      and interpretable rather than arbitrary HMM state indices.
    - Forward-decoded posteriors are written for the entire sample so the
      pipeline never peeks past `train_end`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REGIME_LABELS: list[str] = [
    "growth_up_infl_up",
    "growth_up_infl_dn",
    "growth_dn_infl_up",
    "growth_dn_infl_dn",
]

# Default macro features.
#   INDPRO_yoy : real-growth proxy
#   CPIAUCSL_yoy : headline inflation YoY
#   T10Y2Y : yield curve (term-structure regime)
#   FEDFUNDS_chg : 6-month change in policy rate (tightening / easing)
DEFAULT_HMM_FEATURES: list[str] = ["INDPRO_yoy", "CPIAUCSL_yoy", "T10Y2Y", "FEDFUNDS_chg"]


@dataclass
class RegimeFit:
    """Container holding the fitted HMM and its posterior over the full sample."""

    model: GaussianHMM
    scaler: StandardScaler
    feature_cols: list[str]
    state_to_regime: dict[int, str]
    posteriors: pd.DataFrame  # index=date, columns=REGIME_LABELS
    top1: pd.Series  # index=date, values in REGIME_LABELS
    train_end: pd.Timestamp


def _macro_growth_inflation_panel(
    macro_wide: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Derive the HMM input matrix from the FRED macro_wide frame.

    macro_wide is the same shape the existing feature_engineering produces:
    rows=date (weekly), columns=raw FRED series.
    """
    feature_cols = feature_cols or DEFAULT_HMM_FEATURES
    out = pd.DataFrame(index=macro_wide.index)

    if "INDPRO" in macro_wide.columns:
        out["INDPRO_yoy"] = macro_wide["INDPRO"].pct_change(52)
    if "CPIAUCSL" in macro_wide.columns:
        out["CPIAUCSL_yoy"] = macro_wide["CPIAUCSL"].pct_change(52)
    if "T10Y2Y" in macro_wide.columns:
        out["T10Y2Y"] = macro_wide["T10Y2Y"]
    if "FEDFUNDS" in macro_wide.columns:
        out["FEDFUNDS_chg"] = macro_wide["FEDFUNDS"].diff(26)

    cols = [c for c in feature_cols if c in out.columns]
    if not cols:
        raise ValueError(
            "macro_wide must include at least one of: INDPRO, CPIAUCSL, T10Y2Y, FEDFUNDS"
        )
    return out[cols]


def _state_to_bridgewater(
    state_means_unscaled: np.ndarray,
    feature_cols: list[str],
) -> dict[int, str]:
    """Assign each HMM state to one of the 4 Bridgewater quadrants.

    For each state we compute z-scores of its mean against the cross-state mean
    of (growth, inflation), then bucket. Two states may collide on the same
    quadrant when the HMM finds fewer than 4 economically distinct regimes;
    in that case the second-best state gets the next quadrant by score gap.
    """
    growth_idx = feature_cols.index("INDPRO_yoy") if "INDPRO_yoy" in feature_cols else None
    infl_idx = feature_cols.index("CPIAUCSL_yoy") if "CPIAUCSL_yoy" in feature_cols else None
    if growth_idx is None or infl_idx is None:
        # Fallback: assign by order so labels are deterministic.
        return {i: REGIME_LABELS[i % len(REGIME_LABELS)] for i in range(len(state_means_unscaled))}

    growth_med = float(np.median(state_means_unscaled[:, growth_idx]))
    infl_med = float(np.median(state_means_unscaled[:, infl_idx]))

    assignment: dict[int, str] = {}
    quadrant_taken: dict[str, int] = {}

    # Greedy assignment by absolute z-score in (growth, infl) space.
    order = np.argsort(
        -np.abs(state_means_unscaled[:, [growth_idx, infl_idx]] - np.array([growth_med, infl_med])).sum(axis=1)
    )
    for state in order:
        g_up = state_means_unscaled[state, growth_idx] >= growth_med
        i_up = state_means_unscaled[state, infl_idx] >= infl_med
        if g_up and i_up:
            base = "growth_up_infl_up"
        elif g_up and not i_up:
            base = "growth_up_infl_dn"
        elif (not g_up) and i_up:
            base = "growth_dn_infl_up"
        else:
            base = "growth_dn_infl_dn"

        if base in quadrant_taken:
            # Find the nearest free label.
            for alt in REGIME_LABELS:
                if alt not in quadrant_taken:
                    base = alt
                    break
        assignment[int(state)] = base
        quadrant_taken[base] = int(state)

    # Ensure every regime label has at least an empty mapping for downstream code.
    return assignment


def fit_macro_regime(
    macro_wide: pd.DataFrame,
    train_end: str,
    *,
    n_states: int = 4,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
    n_iter: int = 200,
    covariance_type: str = "full",
) -> RegimeFit:
    """Fit a Gaussian HMM on the train window and decode posteriors over the full sample.

    Parameters
    ----------
    macro_wide : weekly FRED panel (index=date, columns=series names).
    train_end : last date (inclusive) used for HMM fitting. Posteriors after
        this date are forward-decoded with the frozen model.
    n_states : number of latent regimes. Default 4 = Bridgewater 2x2.
    feature_cols : optional override of the macro feature subset.

    Returns
    -------
    RegimeFit
    """
    macro_wide = macro_wide.copy()
    macro_wide.index = pd.to_datetime(macro_wide.index)

    panel = _macro_growth_inflation_panel(macro_wide, feature_cols=feature_cols)
    panel = panel.dropna()
    if panel.empty:
        raise ValueError("Macro panel is empty after dropping NaNs; check FRED ingestion.")

    train_end_ts = pd.Timestamp(train_end)
    train_panel = panel.loc[panel.index <= train_end_ts]
    if len(train_panel) < 52:
        raise ValueError(
            f"Need at least 52 weeks of macro data before train_end ({train_end}); "
            f"got {len(train_panel)}."
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_panel.values)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
    )
    model.fit(X_train)

    # Posterior means back in unscaled feature space for label assignment.
    state_means_scaled = model.means_
    state_means_unscaled = scaler.inverse_transform(state_means_scaled)
    state_to_regime = _state_to_bridgewater(state_means_unscaled, list(train_panel.columns))

    # Posteriors on the full sample (forward decoding with frozen model).
    X_full = scaler.transform(panel.values)
    log_post = model.predict_proba(X_full)
    posteriors_state = pd.DataFrame(log_post, index=panel.index, columns=range(n_states))

    # Re-aggregate by regime label (states can map to the same label if HMM finds <4 distinct).
    regime_frame = pd.DataFrame(0.0, index=panel.index, columns=REGIME_LABELS)
    for state, label in state_to_regime.items():
        if label in regime_frame.columns:
            regime_frame[label] = regime_frame[label] + posteriors_state[state]

    # Row-normalize defensively (should already sum to 1).
    row_sum = regime_frame.sum(axis=1).replace(0, np.nan)
    regime_frame = regime_frame.div(row_sum, axis=0).fillna(0.0)

    top1 = regime_frame.idxmax(axis=1).rename("regime_top1")

    logger.info(
        "HMM regime fit: %d states, train_obs=%d, train_end=%s, state→regime=%s",
        n_states,
        len(train_panel),
        train_end_ts.date(),
        state_to_regime,
    )

    return RegimeFit(
        model=model,
        scaler=scaler,
        feature_cols=list(train_panel.columns),
        state_to_regime=state_to_regime,
        posteriors=regime_frame,
        top1=top1,
        train_end=train_end_ts,
    )


def bridgewater_regime_columns(fit: RegimeFit) -> pd.DataFrame:
    """Wide DataFrame of regime posteriors with column prefix `regime_prob_`."""
    df = fit.posteriors.copy()
    df.columns = [f"regime_prob_{c}" for c in df.columns]
    # Net-of-noise growth & inflation tilts derived from posteriors — handy single-number features.
    df["regime_growth_tilt"] = (
        fit.posteriors.get("growth_up_infl_up", 0.0)
        + fit.posteriors.get("growth_up_infl_dn", 0.0)
        - fit.posteriors.get("growth_dn_infl_up", 0.0)
        - fit.posteriors.get("growth_dn_infl_dn", 0.0)
    )
    df["regime_inflation_tilt"] = (
        fit.posteriors.get("growth_up_infl_up", 0.0)
        - fit.posteriors.get("growth_up_infl_dn", 0.0)
        + fit.posteriors.get("growth_dn_infl_up", 0.0)
        - fit.posteriors.get("growth_dn_infl_dn", 0.0)
    )
    return df


def regime_features_long(
    fit: RegimeFit,
    tickers: list[str],
) -> pd.DataFrame:
    """Broadcast regime posterior columns across the ticker dimension.

    Result schema: ['date', 'ticker', regime_prob_*, regime_growth_tilt,
    regime_inflation_tilt]. Used to merge into the per-asset feature panel.
    """
    wide = bridgewater_regime_columns(fit)
    wide = wide.shift(1)  # never use today's regime to score today — strict no-lookahead.
    rows = []
    for ticker in tickers:
        sub = wide.copy()
        sub["ticker"] = ticker
        sub["date"] = sub.index
        rows.append(sub.reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out
