"""Tests for apply_config_overrides."""

from __future__ import annotations

from pathlib import Path

from src.config import apply_config_overrides, load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config/config.yaml"


def test_apply_config_overrides_m2_threshold():
    cfg = load_config(CONFIG)
    updated = apply_config_overrides(cfg, {"models": {"m2": {"threshold": 0.62}}})
    assert updated.m2.threshold == 0.62


def test_apply_config_overrides_transaction_cost():
    cfg = load_config(CONFIG)
    updated = apply_config_overrides(
        cfg,
        {"portfolio": {"transaction_cost_bps": 3}, "split": {"train_end": "2018-12-31", "test_start": "2019-01-01"}},
    )
    assert updated.portfolio.transaction_cost_bps == 3
    assert updated.split.train_end == "2018-12-31"
    assert updated.split.test_start == "2019-01-01"
