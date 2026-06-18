"""Minimal config loader. Maps config/config.yaml -> typed dataclasses.

Deliberately small: M1 simple/static/linear, M2 dynamic/regime-aware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    universe: list[str]
    source: str = "yfinance"
    use_index_signal: bool = False
    frequency: str = "weekly"
    data_start: str = "2000-01-01"
    macro_series: list[str] = field(default_factory=list)
    vix: bool = True


@dataclass
class SplitConfig:
    train_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    embargo_weeks: int = 4


@dataclass
class LabelsConfig:
    horizon_weeks: int = 4
    positive_threshold: float = 0.005
    negative_threshold: float = -0.005


@dataclass
class M1Config:
    # PURE static price signal: momentum + trend merged (the only M1 factor).
    # Macro and risk are dynamic -> M2. (static -> M1, dynamic -> M2)
    factors: dict[str, float] = field(default_factory=lambda: {"technical": 1.0})
    momentum_windows: list[int] = field(default_factory=lambda: [4, 12, 26, 52])
    trend_windows: list[int] = field(default_factory=lambda: [10, 40])
    allow_short: bool = False


@dataclass
class RiskLayerConfig:
    enabled: bool = True
    vol_windows: list[int] = field(default_factory=lambda: [13, 26])
    strength: float = 0.5


@dataclass
class M2Config:
    regime_features: list[str] = field(default_factory=list)
    refit_window_weeks: int = 52
    sizing: str = "ecdf"


@dataclass
class PortfolioConfig:
    benchmark: str = "equal_weight"
    max_active_tilt: float = 0.10
    max_abs_asset_weight: float = 0.25
    max_gross_exposure: float = 1.00
    vol_target_ann: float = 0.12
    vol_target_lookback_weeks: int = 26


@dataclass
class CostsConfig:
    expense_ratio_bps_annual: float = 9
    transaction_cost_bps: float = 5


@dataclass
class Config:
    data: DataConfig
    split: SplitConfig
    labels: LabelsConfig
    m1: M1Config
    risk_layer: RiskLayerConfig
    m2: M2Config
    portfolio: PortfolioConfig
    costs: CostsConfig
    baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def vix_ticker(self) -> str:
        return "^VIX"

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"


def load_config(path: str | Path | None = None) -> Config:
    root = Path(__file__).resolve().parent.parent
    path = Path(path) if path else root / "config" / "config.yaml"
    raw = yaml.safe_load(path.read_text())
    return Config(
        data=DataConfig(**raw["data"]),
        split=SplitConfig(**raw["split"]),
        labels=LabelsConfig(**raw["labels"]),
        m1=M1Config(**raw["m1"]),
        risk_layer=RiskLayerConfig(**raw.get("risk_layer", {})),
        m2=M2Config(**raw["m2"]),
        portfolio=PortfolioConfig(**raw["portfolio"]),
        costs=CostsConfig(**raw["costs"]),
        baselines=raw.get("baselines", {}),
        root=root,
    )
