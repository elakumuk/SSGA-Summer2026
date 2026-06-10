"""Position sizing from M2 probabilities."""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class SizingMode(str, Enum):
    BINARY = "binary"
    LINEAR = "linear"
    ECDF = "ecdf"


def binary_size(proba: pd.Series, threshold: float) -> pd.Series:
    return (proba >= threshold).astype(float).rename("size")


def linear_size(proba: pd.Series) -> pd.Series:
    return pd.Series(np.maximum(0.0, 2.0 * proba - 1.0), index=proba.index, name="size")


def fit_ecdf(train_proba: pd.Series) -> np.ndarray:
    clean = train_proba.dropna().values
    if len(clean) == 0:
        return np.array([0.5])
    return np.sort(clean)


def ecdf_size(proba: pd.Series, train_sorted: np.ndarray) -> pd.Series:
    n = len(train_sorted)

    def _map(p: float) -> float:
        if np.isnan(p):
            return 0.0
        return float(np.searchsorted(train_sorted, p, side="right") / n)

    return proba.apply(_map).rename("size")


def compute_sizes(
    proba: pd.Series,
    mode: SizingMode | str,
    *,
    threshold: float = 0.55,
    train_proba: pd.Series | None = None,
    train_sorted: np.ndarray | None = None,
) -> pd.Series:
    mode = SizingMode(mode)
    if mode == SizingMode.BINARY:
        return binary_size(proba, threshold)
    if mode == SizingMode.LINEAR:
        return linear_size(proba)
    if mode == SizingMode.ECDF:
        if train_sorted is None:
            if train_proba is None:
                raise ValueError("train_proba or train_sorted required for ECDF sizing")
            train_sorted = fit_ecdf(train_proba)
        return ecdf_size(proba, train_sorted)
    raise ValueError(f"Unknown sizing mode: {mode}")
