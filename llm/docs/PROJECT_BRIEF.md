# Project Brief: AI-Augmented Research Pipeline for Multi-Asset Meta-Labeling

## Project Goal

Build a reusable, end-to-end quantitative research pipeline for a multi-asset portfolio strategy. The pipeline should integrate AI and large language models throughout the research workflow while preserving strict time-series discipline.

The system should behave like a research decision engine:

- M1 proposes trade direction.
- M2 estimates whether M1 is likely to be correct.
- The sizing layer decides how much capital to allocate.
- The backtest engine evaluates the resulting strategy.

The system must not execute live trades.

---

## Problem Framing

Traditional single-model strategies often combine two separate questions:

1. **Side:** Should we be long, short, or flat?
2. **Size:** How much capital should we allocate?

This project separates those decisions.

### M1: Side Decision

M1 outputs:

```text
-1 = short signal
 0 = close/sell/flat signal
 1 = long signal
```

### M2: Meta-Labeling / Sizing Decision

M2 predicts whether M1's non-zero trade signal will be profitable.

```text
M2 output = P(M1 signal is profitable)
```

The probability is then used to size positions. Higher probability means larger allocation, subject to risk constraints.

---

## Research Motivation

Meta-labeling is useful because the primary model can focus on generating candidate opportunities, while the secondary model focuses on filtering false positives and sizing positions. In the attached meta-labeling paper, the framework separates information advantage, false-positive modeling, and position sizing as distinct components to test.

For factor research, candidate signals should be selected based on:

1. Meaning: economic, financial, or behavioral rationale.
2. Significance: measurable predictive performance.
3. Stability: persistence across time and market regimes.

---

## Asset Universe

Use these seven ETFs as a generic multi-asset testing universe:

| Ticker | Asset Class | Role |
|---|---|---|
| SPY | U.S. equities | U.S. growth and equity beta |
| TLT | Long-term U.S. Treasuries | Duration and defensive exposure |
| GLD | Gold | Inflation hedge and safe-haven commodity |
| VEA | Developed international equities | Geographic diversification |
| VWO | Emerging market equities | Emerging market growth/risk premia |
| HYG | High-yield corporate bonds | Credit risk exposure |
| VNQ | Real estate / REITs | Real estate and rate sensitivity |

---

## Data Sources

### Preferred Academic / Class Setting

- Bloomberg Terminal data export or API if available.

### Open-Source Fallback

- ETF prices: yfinance.
- Macro data: FRED through pandas-datareader.
- VIX: Yahoo Finance ticker `^VIX`, if available.

Data should be saved locally after ingest so results are reproducible.

---

## Time Period

Default:

```text
Train: 2006-01-01 to 2020-12-31
Test:  2021-01-01 to latest available date
```

Use weekly data. If the balanced panel starts later because one asset lacks earlier data, report the actual start date.

---

## Modeling Scope

### Candidate M1 Features

- 1-month, 3-month, 6-month, and 12-month momentum.
- Moving-average trend indicators.
- Rolling volatility.
- Drawdown from recent highs.
- Cross-sectional relative strength.
- Carry/valuation proxies where possible.
- Macro regime indicators: inflation, growth, rates, credit, risk sentiment.
- Optional LLM-derived macro narrative features.

### Candidate M2 Features

- M1 input features.
- M1 signal strength.
- Rolling M1 performance statistics.
- VIX level and change.
- Cross-asset dispersion.
- Liquidity and volume proxies.
- Regime indicators.
- LLM-derived narrative or sentiment features.

---

## Strategy Evaluation

Compare:

1. Equal-weight 1/7 benchmark.
2. 60/40 benchmark.
3. M1-only strategy.
4. M1 + M2 binary filter.
5. M1 + M2 probability sizing.
6. M1 + M2 ECDF/rank sizing.

Report:

- Return and excess return.
- Annualized volatility.
- Sharpe ratio.
- Information ratio.
- Information coefficient.
- 12-month maximum drawdown.
- Turnover.
- Hit rate.
- Classification metrics.
- Calibration metrics.

---

## Final Report Structure

The final report should contain:

1. Executive summary.
2. Pipeline architecture.
3. Data sources and cleaning method.
4. Feature engineering rationale.
5. M1 methodology.
6. M2 methodology.
7. Position sizing methodology.
8. Portfolio construction and rebalancing.
9. Backtest results.
10. AI/LLM integration documentation.
11. Limitations and future improvements.

---

## Important Limitations

- yfinance data is convenient but should be treated as research-grade fallback, not institutional source of truth.
- ETF backtests can include survivorship and product-history limitations.
- Macro data release lags must be handled explicitly.
- LLM-derived features can be unstable and should be cached, audited, and compared against non-LLM baselines.
- Backtest performance is not evidence of future performance.
