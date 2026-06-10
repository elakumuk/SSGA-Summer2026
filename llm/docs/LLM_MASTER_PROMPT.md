# LLM Master Prompt: Build the AI-Augmented Multi-Asset Meta-Labeling Pipeline

You are an expert quantitative research engineer. Build a production-quality, research-grade Python project named `finance-meta-labeling-pipeline`.

## Mission

Create a reusable, observable, reproducible, and extensible multi-asset research pipeline that tests whether a meta-labeling model improves a primary multi-asset allocation model.

The pipeline must:

1. Ingest and clean weekly multi-asset time-series data.
2. Build non-leaky factor features.
3. Train a primary model, **M1**, that outputs trade side per asset: `-1`, `0`, or `1`.
4. Train a secondary meta-labeling model, **M2**, that predicts whether M1's proposed trade will be profitable.
5. Convert M2 probabilities into position sizes / portfolio weights.
6. Backtest the resulting strategy against benchmarks.
7. Produce diagnostics and research logs.
8. Include optional AI/LLM-derived features with strict timestamp and leakage controls.

This project is for research and education only. Do not implement live brokerage execution.

---

## Non-Negotiable Constraints

### Time-Series Integrity

At prediction timestamp `t`, the pipeline may only use information available at or before `t`. Future returns are only allowed when constructing labels, never as features.

Implement automated tests that fail when:

- A feature uses `return_{t+1}` or later.
- Train/test dates overlap.
- Cross-validation shuffles data.
- Macro data is forward-filled before its release/availability date without an explicit lag assumption.
- LLM-derived text features are not timestamped or cached.

### Data Frequency

Use weekly data by default. Weekly data should be created from daily adjusted-close data using end-of-week prices, preferably Friday close or the last trading day of the week.

### Asset Universe

Use these tickers:

```text
SPY, TLT, GLD, VEA, VWO, HYG, VNQ
```

### Train/Test Split

Use:

```text
train_start = 2006-01-01
train_end   = 2020-12-31
test_start  = 2021-01-01
test_end    = latest available date
```

If a ticker starts after 2006, begin the balanced panel on the first date where all assets have valid adjusted prices, and report the effective start date.

---

## Build Order

Implement in this order:

1. Project skeleton, config, and logging.
2. Data ingest layer.
3. Data validation layer.
4. Feature engineering layer.
5. Label construction layer.
6. M1 primary model.
7. M2 meta-labeling model.
8. Position sizing and portfolio construction.
9. Backtest accounting.
10. Diagnostics and reporting.
11. Optional LLM-derived feature module.
12. Unit tests and final report notebook.

Stop after each stage and verify tests before moving on.

---

## Expected Python Stack

Use:

```text
python >= 3.11
pandas
numpy
scipy
scikit-learn
matplotlib
pyyaml
pydantic or pandera
yfinance
pandas-datareader
statsmodels
joblib
pytest
```

Optional:

```text
xgboost or lightgbm
shap
ruff
black
mypy
```

Do not make the project dependent on paid data. Create a provider interface so Bloomberg can be added later, but use yfinance/FRED fallback by default.

---

## M1: Primary Model Requirements

M1 decides whether each asset should be long, short, or closed:

```text
-1 = open/maintain SHORT
 0 = close/sell existing position / no active position
 1 = open/maintain BUY/LONG
```

M1 may be implemented as either:

1. A rule-based factor score model, or
2. A supervised classifier/regressor trained on forward returns.

The first working implementation should be simple, robust, and explainable:

```text
M1_score = weighted combination of normalized features:
  + momentum
  + trend
  + carry/valuation proxy
  - volatility/risk penalty
  + macro regime tilt
```

Then convert score to side:

```text
if score > long_threshold: signal = 1
elif score < short_threshold: signal = -1
else: signal = 0
```

Tune thresholds using training data only. Prefer higher recall at M1 because M2 will filter false positives.

---

## M2: Meta-Labeling Model Requirements

M2 predicts whether M1's suggested trade will be profitable.

For each asset and date where M1 proposes a non-zero side:

```text
meta_label = 1 if M1_signal_t * forward_return_{t:t+h} > transaction_cost_threshold else 0
meta_label = 0 otherwise
```

Default horizon:

```text
h = 4 weeks
```

M2 input features should include:

1. M1 features, called **information advantage features**.
2. M1 score and signal strength.
3. False-positive features:
   - VIX level/change if available
   - Cross-asset dispersion
   - Rolling volatility
   - Rolling correlation / diversification stress
   - Rolling M1 hit rate
   - Rolling M1 precision/recall by asset
   - Macro regime indicators
   - Liquidity/volume proxy if available
4. Optional LLM-derived features:
   - Macro narrative regime
   - Inflation narrative score
   - Growth slowdown score
   - Risk-on/risk-off sentiment
   - Policy uncertainty narrative

M2 output:

```text
p_success = P(meta_label = 1 | features)
```

Use logistic regression or random forest as the first version. Include calibration diagnostics. Probability calibration can be added after the baseline is working.

---

## Position Sizing Requirements

Position sizing converts M2 probability into a portfolio weight.

Implement at least three sizing modes:

### 1. Binary Filter

```text
size = 1 if p_success >= threshold else 0
```

### 2. Linear Probability Sizing

```text
raw_size = max(0, 2 * p_success - 1)
```

### 3. ECDF / Rank-Based Sizing

Rank M2 probabilities against the training probability distribution and use percentile rank as size.

```text
size = empirical_cdf_train(p_success)
```

Then apply risk constraints:

```text
max_abs_asset_weight <= 0.25
max_gross_exposure <= 1.00 by default
max_single_trade_turnover <= configurable
transaction_cost_bps = configurable
```

Final signed weight:

```text
weight_asset_t = M1_signal_asset_t * size_asset_t * base_risk_budget_asset_t
```

Normalize weights to respect gross exposure limits.

---

## Backtest Requirements

Backtest should use next-period returns after signals are formed.

At date `t`:

1. Compute features using data available through `t`.
2. Generate M1 signal for each asset.
3. Generate M2 probability and position size.
4. Form target weights.
5. Apply transaction costs based on change in weights.
6. Earn returns from `t` to `t+1`.

Default rebalancing: weekly.

---

## Diagnostics Requirements

Compute and report:

- Cumulative return
- Annualized return
- Excess return vs equal-weight benchmark
- Excess return vs 60/40 benchmark
- Annualized volatility
- Sharpe ratio
- Information ratio
- Information coefficient / rank IC
- Rolling 12-month maximum drawdown
- Full-period maximum drawdown
- Turnover rate
- Hit rate
- Precision, recall, F1, accuracy, AUC for M2
- Confusion matrix
- Probability calibration / Brier score
- Asset contribution to return and risk
- Comparison of M1-only vs M1+M2 vs benchmarks

Create figures and save them under `data/backtests/figures/`.

---

## AI/LLM Integration Requirements

AI/LLM use must be documented and reproducible.

Implement `research_logger.py` that records:

```yaml
stage: data_ingest | cleaning | feature_engineering | m1 | m2 | backtest | diagnostics
llm_used: true/false
tool_or_model: string
prompt_summary: string
human_decision: string
output_used: string
risk_or_limitation: string
timestamp: ISO-8601
```

If implementing LLM-derived features, enforce:

- Every input text document has a timestamp.
- The feature timestamp is no later than the prediction date.
- Outputs are structured JSON.
- Features are cached by date, source, model, and prompt hash.
- No future news or future macro summaries are allowed.

---

## Deliverables

Create:

1. Working Python package in `src/`.
2. Config file in `config/config.yaml`.
3. Four notebooks under `notebooks/`.
4. Tests under `tests/`.
5. A final markdown report under `reports/final_report.md`.
6. A reproducible command:

```bash
python -m src.run_pipeline --config config/config.yaml
```

7. A test command:

```bash
pytest -q
```

---

## Final Acceptance Criteria

The project is complete only if:

- Data downloads and is converted to a clean weekly panel.
- Data validation passes.
- Features are shifted/aligned correctly.
- M1 outputs only `-1`, `0`, or `1`.
- M2 outputs probabilities in `[0, 1]`.
- Position weights respect configured constraints.
- Backtest has no look-ahead bias.
- Strategy, benchmark, and M1-only comparisons are reported.
- Diagnostics are saved.
- Tests pass.
- The final report explains choices, alternatives, limitations, and AI/LLM usage.
