"""M2 evaluation suite. State Street: keep the model simple (logistic regression),
put the effort into MULTIPLE evaluations rather than a fancier model.

Classifier view (is M2 a good meta-label?):  F1, precision, recall, AUC-ROC,
AUC-PR, calibration by probability bucket, base rate.
Economic view (does M2 help the portfolio?):  info-ratio / Sharpe uplift, handled
in run_strategy via the M1-only vs M1+M2 comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _align(proba: pd.Series, labels: pd.DataFrame) -> pd.DataFrame:
    y = labels.stack().rename("y")
    df = pd.concat([proba.rename("p"), y], axis=1).dropna()
    return df


def classifier_metrics(proba: pd.Series, labels: pd.DataFrame, oos_start: str | None = None) -> dict:
    df = _align(proba, labels)
    if oos_start is not None:
        df = df[df.index.get_level_values("date") >= oos_start]
    if df.empty or df["y"].nunique() < 2:
        return {"n": len(df), "note": "insufficient data / one class"}
    y, p = df["y"].values, df["p"].values
    pred = (p > 0.5).astype(int)
    return {
        "n": int(len(df)),
        "base_rate": float(y.mean()),
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
    }


def calibration_table(proba: pd.Series, labels: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Are predicted probabilities honest? Mean predicted vs realized success per bucket."""
    df = _align(proba, labels)
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = pd.qcut(df["p"], q=min(bins, df["p"].nunique()), duplicates="drop")
    g = df.groupby("bucket", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "mean_pred": g["p"].mean(),
        "realized": g["y"].mean(),
    })


def print_report(proba: pd.Series, labels: pd.DataFrame, oos_start: str | None = None) -> None:
    print("\n--- M2 classifier evaluation (full sample) ---")
    full = classifier_metrics(proba, labels)
    for k, v in full.items():
        print(f"  {k:>10}: {v:.3f}" if isinstance(v, float) else f"  {k:>10}: {v}")
    if oos_start:
        print(f"--- M2 classifier evaluation (OOS from {oos_start}) ---")
        for k, v in classifier_metrics(proba, labels, oos_start).items():
            print(f"  {k:>10}: {v:.3f}" if isinstance(v, float) else f"  {k:>10}: {v}")
    print("--- calibration (mean predicted vs realized) ---")
    cal = calibration_table(proba, labels)
    if not cal.empty:
        print(cal.to_string(float_format=lambda v: f"{v:,.3f}"))
