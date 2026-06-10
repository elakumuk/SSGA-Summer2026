"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str = "finance-meta-labeling-pipeline"
    frequency: str = "weekly"
    rebalance: str = "weekly"


@dataclass
class AssetsConfig:
    tickers: list[str] = field(default_factory=lambda: ["SPY", "TLT", "GLD", "VEA", "VWO", "HYG", "VNQ"])
    vix_ticker: str = "^VIX"


@dataclass
class MacroConfig:
    fred_series: list[str] = field(
        default_factory=lambda: ["CPIAUCSL", "UNRATE", "INDPRO", "FEDFUNDS", "DGS10", "T10Y2Y", "BAA10Y"]
    )


@dataclass
class SplitConfig:
    # Earliest date to download/store prices (can precede train_start for feature warmup).
    data_start: str | None = None
    train_start: str = "2006-01-01"
    train_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str | None = None
    # When True (default), keep only weeks where all 7 ETFs have prices (~2007+).
    # When False, allow partial universes so train_start can predate the youngest ETF.
    require_full_universe: bool = True


@dataclass
class FeaturesConfig:
    momentum_windows: list[int] = field(default_factory=lambda: [4, 12, 26, 52])
    volatility_windows: list[int] = field(default_factory=lambda: [4, 12, 26])
    trend_windows: list[int] = field(default_factory=lambda: [10, 40])
    macro_lag_weeks: int = 4
    winsorize_pct: float = 0.01


@dataclass
class LabelsConfig:
    horizon_weeks: int = 4
    positive_threshold: float = 0.005
    negative_threshold: float = -0.005
    transaction_cost_threshold: float = 0.001


@dataclass
class M1Config:
    type: str = "rule_based"
    long_threshold: float = 0.50
    short_threshold: float = -0.50
    weights: dict[str, float] = field(
        default_factory=lambda: {"momentum": 0.45, "trend": 0.25, "macro": 0.20, "risk_penalty": 0.10}
    )
    min_nonzero_signals: int = 100
    optimize_thresholds: bool = True
    asset_class_tilts: bool = True
    allow_short: bool = True
    long_quantile: float = 0.58
    short_quantile: float = 0.22
    long_quantile_min: float = 0.52
    long_quantile_max: float = 0.68
    short_quantile_min: float = 0.12
    short_quantile_max: float = 0.32
    quantile_step: float = 0.02


@dataclass
class M2Config:
    type: str = "logistic_regression"
    threshold: float = 0.55
    calibrate: bool = True


@dataclass
class PortfolioConfig:
    allow_short: bool = True
    max_abs_asset_weight: float = 0.25
    max_gross_exposure: float = 1.0
    transaction_cost_bps: float = 5.0
    base_budget_per_asset: float = 1.0 / 7.0
    sizing_mode: str = "linear"


@dataclass
class PathsConfig:
    raw: str = "data/raw"
    processed: str = "data/processed"
    features: str = "data/features"
    predictions: str = "data/predictions"
    backtests: str = "data/backtests"
    runs: str = "runs"


@dataclass
class LLMFeaturesConfig:
    enabled: bool = False
    cache_dir: str = "data/features/llm_cache"


@dataclass
class PipelineConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    models: dict[str, Any] = field(default_factory=dict)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm_features: LLMFeaturesConfig = field(default_factory=LLMFeaturesConfig)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def m1(self) -> M1Config:
        m1 = self.models.get("m1", {})
        return M1Config(**m1) if isinstance(m1, dict) else m1

    @property
    def m2(self) -> M2Config:
        m2 = self.models.get("m2", {})
        return M2Config(**m2) if isinstance(m2, dict) else m2

    def path(self, key: str, base_dir: Path | None = None) -> Path:
        root = base_dir or Path.cwd()
        return root / getattr(self.paths, key)

    def train_start_date(self) -> date:
        return date.fromisoformat(self.split.train_start)

    def train_end_date(self) -> date:
        return date.fromisoformat(self.split.train_end)

    def test_start_date(self) -> date:
        return date.fromisoformat(self.split.test_start)

    def data_start_resolved(self) -> str:
        """Ingest start date: explicit data_start, else train_start."""
        return self.split.data_start or self.split.train_start


def _build_config(data: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        project=ProjectConfig(**data.get("project", {})),
        assets=AssetsConfig(**data.get("assets", {})),
        macro=MacroConfig(**data.get("macro", {})),
        split=SplitConfig(**data.get("split", {})),
        features=FeaturesConfig(**data.get("features", {})),
        labels=LabelsConfig(**data.get("labels", {})),
        models=data.get("models", {}),
        portfolio=PortfolioConfig(**data.get("portfolio", {})),
        benchmarks=data.get("benchmarks", {}),
        paths=PathsConfig(**data.get("paths", {})),
        llm_features=LLMFeaturesConfig(**data.get("llm_features", {})),
        _raw=data,
    )


def validate_split_dates(cfg: PipelineConfig) -> None:
    """Ensure chronological train/test split with no overlap."""
    train_start = date.fromisoformat(cfg.split.train_start)
    train_end = date.fromisoformat(cfg.split.train_end)
    test_start = date.fromisoformat(cfg.split.test_start)
    if not (train_start < train_end):
        raise ValueError(f"train_start ({cfg.split.train_start}) must be before train_end ({cfg.split.train_end})")
    if not (train_end < test_start):
        raise ValueError(f"train_end ({cfg.split.train_end}) must be before test_start ({cfg.split.test_start})")
    if cfg.split.test_end is not None:
        test_end = date.fromisoformat(cfg.split.test_end)
        if not (test_start <= test_end):
            raise ValueError(f"test_start must be on or before test_end ({cfg.split.test_end})")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into a copy of base."""
    import copy

    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _validate_portfolio(cfg: PipelineConfig) -> None:
    if cfg.portfolio.max_abs_asset_weight <= 0 or cfg.portfolio.max_abs_asset_weight > 1:
        raise ValueError("max_abs_asset_weight must be in (0, 1]")
    if cfg.portfolio.max_gross_exposure <= 0:
        raise ValueError("max_gross_exposure must be positive")


def apply_config_overrides(cfg: PipelineConfig, overrides: dict[str, Any]) -> PipelineConfig:
    """Return config with nested overrides merged (split, models, portfolio, etc.)."""
    data = _deep_merge(cfg._raw, overrides)
    updated = _build_config(data)
    validate_split_dates(updated)
    _validate_portfolio(updated)
    return updated


def apply_split_overrides(
    cfg: PipelineConfig,
    *,
    data_start: str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    require_full_universe: bool | None = None,
) -> PipelineConfig:
    """Return config with CLI or runtime overrides applied to the train/test split."""
    overrides: dict[str, Any] = {}
    split: dict[str, Any] = {}
    if data_start is not None:
        split["data_start"] = data_start
    if train_start is not None:
        split["train_start"] = train_start
    if train_end is not None:
        split["train_end"] = train_end
    if test_start is not None:
        split["test_start"] = test_start
    if test_end is not None:
        split["test_end"] = test_end
    if require_full_universe is not None:
        split["require_full_universe"] = require_full_universe
    if split:
        overrides["split"] = split
    if not overrides:
        return cfg
    return apply_config_overrides(cfg, overrides)


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    cfg = _build_config(data)
    _validate_portfolio(cfg)
    validate_split_dates(cfg)
    return cfg


def save_config_snapshot(cfg: PipelineConfig, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        yaml.safe_dump(cfg._raw, f, default_flow_style=False)


def clone_config_with_m1_allow_short(cfg: PipelineConfig, allow_short: bool) -> PipelineConfig:
    """Return a copy of config with M1 and portfolio shorting enabled or disabled."""
    import copy

    data = copy.deepcopy(cfg._raw)
    data.setdefault("models", {}).setdefault("m1", {})["allow_short"] = allow_short
    data.setdefault("portfolio", {})["allow_short"] = allow_short
    return _build_config(data)
