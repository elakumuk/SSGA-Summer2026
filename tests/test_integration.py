"""Integration test requiring network access."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_full_pipeline_smoke(tmp_path):
    from pathlib import Path
    import shutil

    import yaml

    from src.run_pipeline import run_pipeline

    root = Path(__file__).parent.parent
    cfg_path = tmp_path / "config.yaml"
    base_cfg = yaml.safe_load((root / "config" / "config.yaml").read_text())
    base_cfg["split"]["train_start"] = "2018-01-01"
    base_cfg["split"]["train_end"] = "2019-12-31"
    base_cfg["split"]["test_start"] = "2020-01-01"
    base_cfg["paths"] = {
        "raw": str(tmp_path / "data/raw"),
        "processed": str(tmp_path / "data/processed"),
        "features": str(tmp_path / "data/features"),
        "predictions": str(tmp_path / "data/predictions"),
        "backtests": str(tmp_path / "data/backtests"),
        "runs": str(tmp_path / "runs"),
    }
    cfg_path.write_text(yaml.dump(base_cfg))

    # Reuse cached market/macro parquet from repo data/ if available
    src_processed = root / "data" / "processed"
    dst_processed = tmp_path / "data" / "processed"
    dst_processed.mkdir(parents=True, exist_ok=True)
    for name in ("market_weekly.parquet", "macro_weekly.parquet"):
        src = src_processed / name
        if src.exists():
            shutil.copy(src, dst_processed / name)

    summary = run_pipeline(str(cfg_path), project_root=tmp_path)
    run_dir = summary.run_dir
    assert run_dir.exists()
    assert (run_dir / "config_snapshot.yaml").exists()
    assert (tmp_path / "data" / "backtests" / "long_only" / "metrics_table.csv").exists()
    assert (tmp_path / "data" / "backtests" / "long_short" / "metrics_table.csv").exists()
    assert (tmp_path / "reports" / "final_report.md").exists()
    assert (tmp_path / "reports" / "mode_comparison" / "m1_mode_comparison.png").exists()
    assert (tmp_path / "reports" / "final" / "long_only" / "strategy_cumulative_returns.png").exists()
    assert (tmp_path / "reports" / "assets" / "asset_component_analysis.md").exists()
