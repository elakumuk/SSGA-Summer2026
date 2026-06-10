# AI-Augmented Multi-Asset Meta-Labeling Research Pipeline

**Purpose:** Build a reusable Python research pipeline for a multi-asset portfolio strategy where:

- **M1 / Primary model** decides the trade side for each asset: `-1 = open/maintain SHORT`, `0 = close/sell existing position`, `1 = open/maintain BUY/LONG`.
- **M2 / Meta-labeling model** estimates the probability that the M1 signal will be profitable.
- **Position sizing layer** converts M2 probabilities into portfolio weights / dollar allocation.
- **Backtest diagnostics** compare the strategy against equal-weight and 60/40 benchmarks.
- **AI/LLM features and logs** are integrated deliberately, with timestamp controls to prevent future leakage.

This is a **research and educational project**. It must not place live trades or be treated as investment advice. All outputs should be backtested, documented, and reviewed before any real-world use.

---

## Core Research Question

Can a modular meta-labeling pipeline improve a multi-asset allocation strategy by filtering false-positive M1 signals and sizing positions according to estimated trade quality?

The project is inspired by meta-labeling literature, where a secondary model is layered on top of a primary strategy to filter false positives and size positions, with the M2 output interpreted as the probability of a profitable trade.

---

## Asset Universe

Use a generic global multi-asset ETF universe of at least 7 assets:

| Ticker | Asset Class | Economic Role |
|---|---|---|
| SPY | U.S. equities | U.S. growth / equity benchmark |
| TLT | Long-term U.S. Treasuries | Duration / defensive interest-rate exposure |
| GLD | Gold | Inflation hedge / safe-haven commodity |
| VEA | Developed international equities | Non-U.S. developed market exposure |
| VWO | Emerging market equities | Emerging market growth and risk premia |
| HYG | High-yield corporate bonds | Credit risk exposure |
| VNQ | Real estate / REITs | Real estate and rate-sensitive income exposure |

Preferred frequency: **weekly**.

Default split:

- Training: `2006-01-01` through `2020-12-31`
- Testing: `2021-01-01` through latest available date, ideally through 2026 if data exists

---

## Benchmarks

1. **Equal-weight benchmark:** 1/7 in each asset, rebalanced weekly or monthly.
2. **60/40 benchmark:** 60% equities and 40% bonds.
   - Default implementation: equities = `SPY, VEA, VWO, VNQ`; bonds = `TLT, HYG`.
   - Gold is excluded from the strict 60/40 benchmark, but an optional 55/35/10 stock/bond/gold benchmark can be added.

---

## Repository Layout to Build

```text
finance-meta-labeling-pipeline/
  README.md
  pyproject.toml
  requirements.txt
  config/
    config.yaml
  data/
    raw/
    processed/
    features/
    predictions/
    backtests/
  docs/
    PROJECT_BRIEF.md
    ARCHITECTURE_SPEC.md
    DATA_FEATURES_SPEC.md
    MODELING_SPEC.md
    BACKTEST_EVALUATION_SPEC.md
    AI_LLM_INTEGRATION_LOG.md
    ACCEPTANCE_TESTS.md
  notebooks/
    01_data_ingest.ipynb
    02_feature_engineering.ipynb
    03_modeling_meta_labeling.ipynb
    04_backtest_diagnostics.ipynb
  src/
    __init__.py
    config.py
    data_providers.py
    data_validation.py
    feature_engineering.py
    labels.py
    model_m1.py
    model_m2.py
    position_sizing.py
    portfolio.py
    backtest.py
    diagnostics.py
    llm_features.py
    research_logger.py
  tests/
    test_no_future_leakage.py
    test_data_schema.py
    test_label_alignment.py
    test_backtest_accounting.py
```

---

## Quick Start for the Coding LLM

Give the LLM coding agent the files in `/docs`, especially:

1. `LLM_MASTER_PROMPT.md`
2. `PROJECT_BRIEF.md`
3. `ARCHITECTURE_SPEC.md`
4. `DATA_FEATURES_SPEC.md`
5. `MODELING_SPEC.md`
6. `BACKTEST_EVALUATION_SPEC.md`
7. `ACCEPTANCE_TESTS.md`

The master prompt instructs the agent to implement the pipeline step-by-step with tests, data validation, no look-ahead leakage, and complete diagnostics.

---

## Design Rules

- Never shuffle time-series samples.
- At prediction time `t`, features may only use data available at or before `t`.
- Labels may use future returns, but labels must never leak into features.
- M1 and M2 must be fit only on training data for out-of-sample evaluation.
- Use walk-forward / expanding-window validation when doing cross-validation.
- Save every intermediate output with timestamps and schema validation.
- Treat LLM-generated features as data: cache them, version them, and record prompts/model/date/source.
- Include baseline strategies before comparing meta-labeling improvements.

---

## Primary Deliverables

- Clean weekly dataset for all assets and macro features.
- Factor library: momentum, volatility, valuation/carry proxies, macro regime, and optional LLM-derived regime/sentiment features.
- M1 primary model producing `{-1, 0, 1}` signals.
- M2 meta-labeling model producing probability of a profitable M1 signal.
- Position sizing model mapping M2 probability to dollar weights.
- Backtest engine with transaction costs and turnover.
- Diagnostics report with returns, excess returns, volatility, Sharpe, IR, IC, max drawdown, turnover, hit rate, precision/recall/F1/AUC, and calibration.
- AI/LLM usage log.

---

## Source Notes

- Attached paper: `JFDS-2022-Joubert-31-44.pdf`, *Meta-Labeling: Theory and Framework*.
- Attached paper: `document_pdf.pdf`, *The A-Z of Quant*.
- Reference GitHub repo: `https://github.com/hudson-and-thames/meta-labeling`
- Time-series validation reference: scikit-learn `TimeSeriesSplit`.
- Market data fallback: `yfinance`.
- Macro data fallback: FRED via `pandas-datareader`.
