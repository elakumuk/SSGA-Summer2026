"""Tests for grid search spec expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.grid_search import derive_test_start_from_train_end, expand_grid_spec, load_grid_spec

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "scripts/grid_search_spec.yaml"


def test_spec_expands_to_40_runs():
    spec = load_grid_spec(SPEC)
    runs = expand_grid_spec(spec, ROOT)
    assert len(runs) == 40
    run_ids = [r["run_id"] for r in runs]
    assert len(run_ids) == len(set(run_ids))


def test_test_start_derived_from_train_end():
    spec = load_grid_spec(SPEC)
    runs = expand_grid_spec(spec, ROOT)
    for run in runs:
        expected = derive_test_start_from_train_end(run["train_end"])
        assert run["test_start"] == expected
        assert run["overrides"]["split"]["test_start"] == expected


def test_train_end_test_start_valid():
    from src.config import validate_split_dates, apply_config_overrides, load_config

    spec = load_grid_spec(SPEC)
    base = load_config(ROOT / spec["base_config"])
    runs = expand_grid_spec(spec, ROOT)
    for run in runs:
        cfg = apply_config_overrides(base, run["overrides"])
        validate_split_dates(cfg)


def test_expand_requires_grid():
    with pytest.raises(ValueError, match="non-empty"):
        expand_grid_spec({"grid": {}}, ROOT)
