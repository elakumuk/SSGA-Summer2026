"""Data validation for market and modeling panels."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_PRICE_COLUMNS = ["date", "ticker", "adj_close"]
REQUIRED_PANEL_COLUMNS = ["date", "ticker", "adj_close", "return_1w"]


@dataclass
class ValidationReport:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    missing_value_summary: dict[str, float] = field(default_factory=dict)
    effective_start_date: str | None = None
    effective_end_date: str | None = None
    n_rows: int = 0
    n_tickers: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        ok = bool(passed)
        self.checks.append({"name": name, "passed": ok, "detail": detail})
        if not ok:
            self.passed = False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json.dumps(asdict(self), default=str))
        with path.open("w") as f:
            json.dump(payload, f, indent=2)


def _check_datetime_sorted(df: pd.DataFrame, report: ValidationReport) -> None:
    dates = pd.to_datetime(df["date"])
    report.add_check("datetime_index", pd.api.types.is_datetime64_any_dtype(dates), "date column is datetime")
    unique_dates = pd.Series(pd.to_datetime(df["date"].unique())).sort_values()
    report.add_check("sorted_ascending", unique_dates.is_monotonic_increasing, "dates sorted")
    report.add_check("no_duplicate_dates_per_ticker", not df.duplicated(subset=["date", "ticker"]).any(), "unique date/ticker pairs")


def validate_price_panel(
    df: pd.DataFrame,
    required_tickers: list[str],
    *,
    balanced: bool = True,
) -> ValidationReport:
    report = ValidationReport(passed=True)
    report.n_rows = len(df)
    report.n_tickers = df["ticker"].nunique()

    for col in REQUIRED_PRICE_COLUMNS:
        report.add_check(f"column_{col}", col in df.columns, f"missing {col}")

    if "adj_close" in df.columns:
        report.add_check("positive_adj_close", (df["adj_close"] > 0).all(), "adj_close must be positive")
        report.add_check("no_negative_prices", (df["adj_close"] > 0).all())

    present = set(df["ticker"].unique())
    missing = set(required_tickers) - present
    report.add_check("required_tickers_present", len(missing) == 0, f"missing tickers: {missing}")

    _check_datetime_sorted(df, report)

    today = pd.Timestamp(datetime.utcnow().date())
    report.add_check("no_future_dates", (pd.to_datetime(df["date"]) <= today).all(), "no future-dated rows")

    if balanced and not df.empty:
        counts = df.groupby("date")["ticker"].nunique()
        report.add_check(
            "balanced_panel",
            (counts == len(required_tickers)).all(),
            f"unbalanced dates: {(counts != len(required_tickers)).sum()}",
        )

    missing_pct = df.isna().mean().to_dict()
    report.missing_value_summary = {k: float(v) for k, v in missing_pct.items()}

    if not df.empty:
        report.effective_start_date = str(pd.to_datetime(df["date"]).min().date())
        report.effective_end_date = str(pd.to_datetime(df["date"]).max().date())

    return report


def validate_model_panel(df: pd.DataFrame, required_tickers: list[str]) -> ValidationReport:
    report = validate_price_panel(df.reset_index() if isinstance(df.index, pd.MultiIndex) else df, required_tickers)
    panel = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()

    for col in REQUIRED_PANEL_COLUMNS:
        if col not in panel.columns:
            report.add_check(f"panel_column_{col}", False, f"missing {col}")

    label_cols = [c for c in panel.columns if "forward_return" in c or c in ("m1_target", "meta_label", "trade_return")]
    feature_cols = [
        c
        for c in panel.columns
        if c not in label_cols
        and c not in {"date", "ticker", "adj_close", "return_1w", "M1_signal", "M1_score", "p_success", "predicted_meta_label"}
        and not c.startswith("weight_")
    ]
    for col in feature_cols:
        if col in panel.columns and panel[col].dtype == object:
            report.add_check(f"feature_numeric_{col}", False, f"feature {col} should be numeric")

    return report


def build_balanced_panel(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Keep only dates where all required tickers have data."""
    wide = df[df["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values="adj_close")
    complete_dates = wide.dropna().index
    out = df[df["date"].isin(complete_dates) & df["ticker"].isin(tickers)].copy()
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_available_panel(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Keep every ticker-date row with a valid price (partial universes allowed)."""
    sub = df[df["ticker"].isin(tickers)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub[sub["adj_close"].notna() & (sub["adj_close"] > 0)]
    return sub.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_modeling_panel(df: pd.DataFrame, tickers: list[str], *, require_full_universe: bool = True) -> pd.DataFrame:
    if require_full_universe:
        return build_balanced_panel(df, tickers)
    return build_available_panel(df, tickers)


def ticker_coverage_summary(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Per-ticker first/last available dates in the modeling panel."""
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        sub = df[df["ticker"] == ticker]
        if sub.empty:
            rows.append({"ticker": ticker, "first_date": None, "last_date": None, "n_weeks": 0})
            continue
        dates = pd.to_datetime(sub["date"])
        rows.append(
            {
                "ticker": ticker,
                "first_date": str(dates.min().date()),
                "last_date": str(dates.max().date()),
                "n_weeks": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)
