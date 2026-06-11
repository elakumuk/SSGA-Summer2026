"""Optional LLM-derived features with timestamp and cache controls."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import LLMFeaturesConfig

logger = logging.getLogger(__name__)

FEATURE_SCHEMA = {
    "risk_sentiment": float,
    "inflation_pressure": float,
    "growth_slowdown": float,
    "policy_tightness": float,
    "credit_stress": float,
    "macro_uncertainty": float,
    "dominant_narrative": str,
    "confidence": float,
}


@dataclass
class TextSource:
    source_id: str
    document_date: datetime
    text: str


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def validate_source_timestamp(source_date: datetime, prediction_date: datetime) -> None:
    if source_date > prediction_date:
        raise ValueError(
            f"Source date {source_date} is after prediction date {prediction_date}; rejected to prevent leakage"
        )


def cache_path(cache_dir: Path, date: str, source_id: str, model: str, phash: str) -> Path:
    return cache_dir / f"{date}_{source_id}_{model}_{phash}.json"


def load_cached_feature(path: Path) -> dict | None:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return None


def save_cached_feature(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def build_prompt(source: TextSource, prediction_date: datetime) -> str:
    return (
        "You are extracting structured macro-regime features for a quantitative backtest.\n"
        "IMPORTANT: Use only the text below. Do not use any knowledge after the document date.\n"
        f"Document date: {source.document_date.date()}\n"
        f"Prediction date: {prediction_date.date()}\n"
        f"Source ID: {source.source_id}\n"
        "Return JSON only with risk_sentiment, inflation_pressure, growth_slowdown, "
        "policy_tightness, credit_stress, macro_uncertainty, dominant_narrative, confidence.\n"
        f"Text:\n{source.text}"
    )


class LLMFeatureExtractor:
    """LLM feature interface. Disabled by default; no API calls unless explicitly configured."""

    def __init__(self, cfg: LLMFeaturesConfig, model_name: str = "disabled") -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.cache_dir = Path(cfg.cache_dir)

    def extract(
        self,
        source: TextSource,
        prediction_date: datetime,
        *,
        structured_output: dict | None = None,
    ) -> dict:
        if not self.cfg.enabled:
            logger.debug("LLM features disabled; returning empty feature dict")
            return {}

        validate_source_timestamp(source.document_date, prediction_date)
        prompt = build_prompt(source, prediction_date)
        phash = prompt_hash(prompt)
        cpath = cache_path(
            self.cache_dir,
            str(prediction_date.date()),
            source.source_id,
            self.model_name,
            phash,
        )
        cached = load_cached_feature(cpath)
        if cached is not None:
            return cached

        if structured_output is None:
            raise RuntimeError(
                "LLM features enabled but no API client configured. "
                "Provide structured_output manually or implement API integration."
            )

        payload = {
            "date": str(prediction_date.date()),
            "source_id": source.source_id,
            "model": self.model_name,
            "prompt_hash": phash,
            **structured_output,
        }
        save_cached_feature(cpath, payload)
        return payload

    def features_to_frame(self, records: list[dict]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
