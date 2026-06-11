"""Split-conformal prediction wrapper for the M2 secondary model.

Background
----------
Vovk, Gammerman & Shafer (2005) — *Algorithmic Learning in a Random World*.
Lei et al. (2018) — "Distribution-Free Predictive Inference for Regression".
Romano, Patterson & Candès (2019) — "Conformalized Quantile Regression".

For meta-labeling we already have calibrated probabilities from M2. What we
*lack* is a finite-sample guarantee on how reliable any particular probability
estimate is — that is, a way of attaching a confidence band around `p_success`
that carries a distribution-free coverage statement.

Split-conformal gives exactly that. We hold out a calibration slice from the
training window, fit M2 on the remainder, compute non-conformity scores on the
calibration slice, and use the (1 - α) empirical quantile of those scores as
a symmetric interval half-width around each new prediction.

This module is **additive** to the existing M2 path: pass an `M2Conformal`
instead of the raw `SklearnM2` and you get `p_success_lo` / `p_success_hi`
alongside the existing point probability. Sizing can then optionally use the
interval width to shrink positions whose predictions are uncertain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import M2Config, PipelineConfig
from src.model_m1 import train_date_mask
from src.model_m2 import SklearnM2, build_m2_features
from src.labels import get_m2_training_mask

logger = logging.getLogger(__name__)


@dataclass
class ConformalCalibration:
    """Cached calibration scores + quantile lookup."""

    alpha: float
    scores: np.ndarray  # nonconformity values on the calibration set
    quantile: float  # the (1 - alpha) quantile used as the interval half-width
    n_calib: int


class M2Conformal:
    """Split-conformal wrapper around `SklearnM2`.

    Parameters
    ----------
    cfg : M2 config (passed through to the underlying SklearnM2).
    calibration_weeks : how many weeks at the tail of the training window to
        hold out for calibration. Default 26 (~6 months).
    alpha : miscoverage level. 0.10 = 90% coverage bands. Default 0.10.
    """

    def __init__(
        self,
        cfg: M2Config,
        *,
        calibration_weeks: int = 26,
        alpha: float = 0.10,
    ) -> None:
        self.cfg = cfg
        self.calibration_weeks = int(calibration_weeks)
        self.alpha = float(alpha)
        self.inner = SklearnM2(cfg)
        self.calibration: ConformalCalibration | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "M2Conformal":
        """Fit inner M2 on (train minus tail) and calibrate on the tail."""
        if not isinstance(X.index, pd.MultiIndex) or "date" not in X.index.names:
            raise ValueError("Conformal calibration requires a MultiIndex with a 'date' level.")

        dates = X.index.get_level_values("date")
        unique_dates = pd.Index(sorted(dates.unique()))
        if len(unique_dates) <= self.calibration_weeks + 4:
            raise ValueError(
                f"Need more than {self.calibration_weeks + 4} weeks of train data; got {len(unique_dates)}."
            )
        cutoff_date = unique_dates[-self.calibration_weeks]
        train_mask = dates < cutoff_date
        calib_mask = dates >= cutoff_date

        self.inner.fit(X.loc[train_mask], y.loc[train_mask])

        proba_calib = self.inner.predict_proba(X.loc[calib_mask])
        y_calib = y.loc[calib_mask]
        scores = (y_calib.values - proba_calib.values) ** 2  # squared residual non-conformity
        scores = np.abs(y_calib.values - proba_calib.values)  # absolute residual is simpler + monotone

        n_calib = len(scores)
        # Finite-sample exact quantile per Vovk et al.: index = ceil((1 - alpha)(n+1)) / n
        rank = int(np.ceil((1.0 - self.alpha) * (n_calib + 1)))
        rank = max(1, min(rank, n_calib))
        q = float(np.sort(scores)[rank - 1])
        self.calibration = ConformalCalibration(
            alpha=self.alpha,
            scores=scores,
            quantile=q,
            n_calib=n_calib,
        )
        logger.info(
            "M2 conformal calibration: n=%d, alpha=%.2f, interval half-width=%.4f",
            n_calib,
            self.alpha,
            q,
        )
        return self

    def predict_with_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.calibration is None:
            raise RuntimeError("M2Conformal not calibrated.")
        proba = self.inner.predict_proba(X)
        q = self.calibration.quantile
        lo = (proba - q).clip(lower=0.0, upper=1.0)
        hi = (proba + q).clip(lower=0.0, upper=1.0)
        return pd.DataFrame(
            {
                "p_success": proba,
                "p_success_lo": lo,
                "p_success_hi": hi,
                "p_success_band_width": hi - lo,
            },
            index=X.index,
        )


def fit_m2_conformal(
    panel: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    calibration_weeks: int = 26,
    alpha: float = 0.10,
    train_mask: pd.Series | None = None,
) -> tuple[M2Conformal, pd.DataFrame]:
    """End-to-end fit + calibration matching the existing `fit_m2` API."""
    if train_mask is None:
        dates = panel.index.get_level_values("date")
        train_mask = train_date_mask(dates, cfg).values
    m2_mask = get_m2_training_mask(panel) & train_mask
    train_panel = panel.loc[m2_mask]
    y = train_panel["meta_label"].dropna()
    X = build_m2_features(train_panel, cfg).loc[y.index]
    model = M2Conformal(cfg.m2, calibration_weeks=calibration_weeks, alpha=alpha)
    model.fit(X, y)
    return model, X


def predict_m2_conformal(
    model: M2Conformal,
    panel: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    out = panel.copy()
    m2_rows = get_m2_training_mask(out)
    X = build_m2_features(out, cfg)
    for col in ("p_success", "p_success_lo", "p_success_hi", "p_success_band_width"):
        out[col] = np.nan
    if m2_rows.any():
        intervals = model.predict_with_interval(X.loc[m2_rows])
        out.loc[m2_rows, "p_success"] = intervals["p_success"].values
        out.loc[m2_rows, "p_success_lo"] = intervals["p_success_lo"].values
        out.loc[m2_rows, "p_success_hi"] = intervals["p_success_hi"].values
        out.loc[m2_rows, "p_success_band_width"] = intervals["p_success_band_width"].values
        out.loc[m2_rows, "predicted_meta_label"] = (
            intervals["p_success"] >= cfg.m2.threshold
        ).astype(int).values
    return out


def conformal_size_multiplier(
    band_width: pd.Series,
    *,
    max_shrink_band: float = 0.40,
    floor: float = 0.25,
) -> pd.Series:
    """Map M2 conformal interval width → linear position-size multiplier in [floor, 1].

    Wide bands (≥ max_shrink_band) → multiplier = floor.
    Narrow bands (≈ 0) → multiplier = 1.
    Linear interpolation between, then clipped.
    """
    width = band_width.clip(lower=0.0)
    raw = 1.0 - (1.0 - floor) * (width / max(max_shrink_band, 1e-9))
    return raw.clip(lower=floor, upper=1.0).rename("conformal_size_mult")
