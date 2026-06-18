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

## Results

Full sample 2000–2026, out-of-sample (OOS) from 2021.

| Strategy | Sharpe | Max DD | Sharpe (OOS) | Info Ratio (OOS) |
|---|---:|---:|---:|---:|
| Equal-Weight | 0.62 | -30% | 0.76 | — |
| Moderate Growth | 0.56 | -34% | 0.71 | -0.32 |
| Institutional | 0.63 | -29% | 0.74 | -0.64 |
| **M1-only** | **0.65** | **-22%** | **0.81** | **+0.18** |
| M1 + M2 | 0.61 | -25% | 0.78 | -0.00 |

Walk-forward Sharpe by window: equal-weight 0.30 / 0.59 / 0.48 / 0.76 vs M1-only
0.05 / 0.53 / 0.15 / **0.81** across 2014-16 / 2016-18 / 2018-20 / 2021-now.

### What the results mean

- **M1 works, modestly.** It beats all three baselines on risk-adjusted return and
  cuts drawdown the most (-22% vs -30% for equal-weight), with a positive OOS
  information ratio. Its two sub-signals (momentum, trend) reinforce each other
  (positive interaction), so they are complementary rather than redundant.
- **M1's edge is regime-concentrated.** Walk-forward shows it clearly beats
  equal-weight only in the 2021+ window; full-sample numbers flatter it than the
  regime-by-regime view does. The edge is real but not uniform across history.
- **M2 does not add value yet.** Its probabilities carry no information — realized
  success is flat across every predicted bucket (predicted 0.14 → realized 0.44;
  predicted 0.82 → realized 0.43), AUC-ROC ≈ 0.50 full / 0.46 OOS, and it
  underperforms M1 in every walk-forward window. This held under both proxy and real
  macro data, so it is a robust finding, not a tuning artifact. Reformulating the
  meta-label is the open research question.
- **Costs are modest and turnover-driven:** gross 6.48% → net of expense 6.39% → net
  of all costs 6.19% annualized (transaction ≈ 29 bps vs expense ratio ≈ 9 bps).

## Open questions / discussion

- **M2 reformulation** — should the meta-label become (a) a regime gate that scales
  total active risk down in stress regimes, (b) a factor-timer (momentum vs trend by
  regime), or (c) a different prediction target / horizon? The current per-asset,
  4-week, benchmark-relative success label extracts no signal.
- **M1 robustness** — is a recent-regime-concentrated edge acceptable, or do we need
  an M1 that is robust across regimes?
- **Data** — prioritize Bloomberg true-index history (pre-2012)? Which sleeves first?
- **Universe** — stay at 7 sleeves, or add investment-grade credit + broad commodity (9)?
- **Benchmark / tracking-error budget** for the information-ratio framing?

## Limitations

- Research-grade data (yfinance, FRED, ETF proxies); free index history is limited,
  so a true long-history index study requires institutional data.
- Historical simulation only — no capacity, market impact, borrow, or live execution.
- Some diagnostics are full-sample; production validation would extend the
  walk-forward and purged cross-validation.
