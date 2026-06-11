"""M2 meta-labeling model."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import M2Config, PipelineConfig
from src.model_m1 import train_date_mask
from src.feature_engineering import get_feature_columns
from src.labels import get_m2_training_mask

logger = logging.getLogger(__name__)


class M2Model(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> M2Model:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        ...

    def predict_meta_label(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int).rename("predicted_meta_label")


class SklearnM2(M2Model):
    def __init__(self, cfg: M2Config) -> None:
        self.cfg = cfg
        self.feature_cols: list[str] = []
        self.pipeline: Pipeline | CalibratedClassifierCV | None = None

    def _build_estimator(self) -> Pipeline:
        if self.cfg.type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SklearnM2:
        self.feature_cols = list(X.columns)
        base = self._build_estimator()
        if self.cfg.calibrate:
            self.pipeline = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        else:
            self.pipeline = base
        self.pipeline.fit(X[self.feature_cols].values, y.values)
        logger.info("M2 fitted on %d rows, %d features", len(X), len(self.feature_cols))
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if self.pipeline is None:
            raise RuntimeError("M2 model not fitted")
        cols = [c for c in self.feature_cols if c in X.columns]
        proba = self.pipeline.predict_proba(X[cols].values)[:, 1]
        return pd.Series(proba, index=X.index, name="p_success")


def build_m2_features(panel: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    base_features = get_feature_columns(panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel)
    extra = [c for c in ("M1_signal", "M1_score") if c in panel.columns]
    m2_cols = list(dict.fromkeys(base_features + extra))
    available = [c for c in m2_cols if c in panel.columns]
    return panel[available].copy()


def fit_m2(
    panel: pd.DataFrame,
    cfg: PipelineConfig,
    train_mask: pd.Series | None = None,
) -> tuple[SklearnM2, pd.DataFrame]:
    if train_mask is None:
        dates = panel.index.get_level_values("date")
        train_mask = train_date_mask(dates, cfg).values
    m2_mask = get_m2_training_mask(panel) & train_mask
    train_panel = panel.loc[m2_mask]
    y = train_panel["meta_label"].dropna()
    X = build_m2_features(train_panel, cfg).loc[y.index]
    model = SklearnM2(cfg.m2)
    model.fit(X, y)
    return model, X


def predict_m2(model: SklearnM2, panel: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = panel.copy()
    m2_rows = get_m2_training_mask(out)
    X = build_m2_features(out, cfg)
    out["p_success"] = np.nan
    if m2_rows.any():
        out.loc[m2_rows, "p_success"] = model.predict_proba(X.loc[m2_rows]).values
        out.loc[m2_rows, "predicted_meta_label"] = model.predict_meta_label(
            X.loc[m2_rows], threshold=cfg.m2.threshold
        ).values
    return out
