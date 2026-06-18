# Multi-Asset Meta-Labeling Research Pipeline

A weekly, multi-asset allocation framework using a two-stage meta-labeling design.
Research and educational use only — not investment advice.

## Overview

The pipeline allocates across a seven-sleeve global asset universe and separates the
allocation decision into two stages plus a portfolio layer:

- **M1 — static directional model.** A simple, linear signal that decides which
  assets to favour. It uses only static price factors (momentum + trend), kept
  deliberately lean.
- **M2 — dynamic meta-label.** A logistic-regression layer that, conditioned on the
  market regime, estimates whether an M1 view is likely to pay and sizes accordingly.
  It receives the individual factors (momentum, trend, macro, volatility) and regime
  features separately, so it can weight them dynamically.
- **Portfolio.** Benchmark-relative active weights (benchmark ± bounded tilt),
  volatility targeting, position caps, and a two-layer cost model (expense ratio +
  transaction cost). The headline metric is the information ratio.

Design principle: **static factors live in M1, dynamic factors live in M2.**

## Asset universe

S&P 500 · MSCI EAFE · MSCI Emerging Markets · U.S. Treasury 7–10Y · U.S. High Yield ·
Gold · U.S. REITs. Signals are computed on index series where available; instrument
substitution (ETFs) is used at the implementation stage. See `DATA_SOURCES.md`.

## Usage

```bash
pip install -r requirements.txt      # pandas, numpy, scikit-learn, yfinance, pyyaml
python fetch_indices.py              # download index/proxy series into data/raw/index/
python run_all.py                    # strategy + attribution + walk-forward -> reports/
```

Individual stages:
```bash
python run_m1.py            # M1 allocation for the latest week
python run_strategy.py      # full M1 / M1+M2 backtest + M2 evaluation suite
python run_attribution.py   # factor and cost attribution
python run_walkforward.py   # walk-forward validation across windows
python -m pytest tests/     # correctness tests (no look-ahead, embargo, constraints)
```

Configuration lives in `config/config.yaml` (universe, factor weights, split,
risk/cost parameters, baseline portfolios).

## Repository layout

| Path | Contents |
|---|---|
| `src/data.py` | market + macro ingestion (yfinance, FRED), weekly resampling, index loader |
| `src/features.py` | no-look-ahead factors: momentum, trend, macro tilt, regime, volatility |
| `src/m1.py` | static linear directional model |
| `src/m2.py` | dynamic regime-aware meta-label (rolling logistic, embargo) |
| `src/portfolio.py` | benchmark-relative weights, vol targeting, two-layer costs |
| `src/backtest.py` | returns, Sharpe, drawdown, information ratio, baselines |
| `src/evaluation.py` | M2 classifier metrics (F1, AUC-ROC, AUC-PR, calibration) |
| `config/config.yaml` | all parameters |
| `tests/` | correctness tests |
| `reports/` | generated results and write-ups |

## Methodology notes

- No look-ahead: rolling features are shifted; macro is lagged four weeks; a
  four-week embargo separates train and test to prevent label leakage.
- Train through 2020, test from 2021 onward; walk-forward across multiple windows.
- Shorting is not required: an underweight relative to the benchmark is an implicit
  short, so the strategy expresses views through bounded active tilts.

## Current results (summary)

- **M1** is a modest but sound directional sleeve: it improves risk-adjusted return
  over an equal-weight benchmark and roughly halves drawdown, with an edge that is
  concentrated in recent regimes rather than uniform across history.
- **M2**, as currently specified, does not add value — its classifier metrics are weak
  (out-of-sample AUC ≈ 0.46, flat calibration) and it underperforms M1 across
  walk-forward windows. Reformulating the meta-label is the open research question.

## Limitations

- Research-grade data (yfinance, FRED, ETF proxies); free index history is limited,
  so a true long-history index study requires institutional data.
- Historical simulation only — no capacity, market impact, borrow, or live execution.
- Some diagnostics are full-sample; production validation would extend the
  walk-forward and purged cross-validation.
