# Architecture Specification

## High-Level Pipeline

```text
Raw Data
  ↓
Data Ingest
  ↓
Data Cleaning + Validation
  ↓
Feature Engineering
  ↓
Label Construction
  ↓
M1 Primary Side Model
  ↓
M2 Meta-Labeling Model
  ↓
Position Sizing
  ↓
Portfolio Construction
  ↓
Backtest Engine
  ↓
Diagnostics + Final Report
```

---

## Module Responsibilities

### `src/config.py`

Load and validate configuration.

Must include:

- tickers
- date ranges
- frequency
- feature windows
- label horizon
- model parameters
- risk constraints
- transaction costs
- output paths

---

### `src/data_providers.py`

Create a provider interface:

```python
class MarketDataProvider:
    def get_prices(self, tickers, start, end, frequency):
        ...

class MacroDataProvider:
    def get_macro(self, series, start, end, frequency):
        ...
```

Implement:

- `YFinanceProvider`
- `FredProvider`
- placeholder `BloombergProvider`

The Bloomberg provider can raise `NotImplementedError` with clear instructions for future implementation.

---

### `src/data_validation.py`

Validate all panel data.

Checks:

- datetime index sorted ascending
- no duplicate dates
- required tickers present
- required columns present
- no negative prices
- missing value report
- no future-dated rows
- weekly frequency approximately consistent
- balanced panel option

Expected cleaned price schema:

```text
index: date
columns:
  ticker
  open
  high
  low
  close
  adj_close
  volume
```

Preferred modeling format:

```text
MultiIndex: date, ticker
columns:
  adj_close
  return_1w
  volume
  feature_...
  label_...
```

---

### `src/feature_engineering.py`

Build features using historical data only.

All rolling features must be shifted if needed so that the strategy does not use information unavailable at signal time.

Example:

```python
returns = prices.pct_change()
momentum_12w = prices.pct_change(12).shift(1)
vol_12w = returns.rolling(12).std().shift(1)
```

---

### `src/labels.py`

Construct M1 and M2 labels.

M1 target options:

```text
future_return_h > positive_threshold  ->  1
future_return_h < negative_threshold  -> -1
otherwise                             ->  0
```

M2 target:

```text
meta_label = 1 if M1_signal * future_return_h > transaction_cost_threshold else 0
```

Only create M2 rows where M1 signal is non-zero.

---

### `src/model_m1.py`

M1 must expose:

```python
fit(X_train, y_train)
predict_signal(X) -> {-1, 0, 1}
predict_score(X) -> continuous score
```

Implement baseline:

- `RuleBasedM1`
- Optional `LogisticM1` or `RandomForestM1`

---

### `src/model_m2.py`

M2 must expose:

```python
fit(X_train, y_meta_train)
predict_proba(X) -> probability in [0, 1]
predict_meta_label(X, threshold=0.5) -> {0, 1}
```

Implement baseline:

- Logistic regression
- Random forest
- optional probability calibration

---

### `src/position_sizing.py`

Convert M2 probability into position size.

Required methods:

1. binary filter
2. linear probability sizing
3. ECDF / rank-based sizing

---

### `src/portfolio.py`

Convert per-asset signal and size into target weights.

Constraints:

```yaml
max_abs_asset_weight: 0.25
max_gross_exposure: 1.0
allow_short: true
cash_weight_allowed: true
```

---

### `src/backtest.py`

Backtest target weights and returns.

Accounting:

```text
strategy_return_t+1 = sum(weight_t * asset_return_t+1) - transaction_cost_t
turnover_t = sum(abs(weight_t - weight_t-1))
transaction_cost_t = turnover_t * transaction_cost_bps / 10000
```

---

### `src/diagnostics.py`

Compute metrics and generate plots.

Required outputs:

- metrics table
- cumulative return plot
- drawdown plot
- rolling Sharpe plot
- rolling 12-month max drawdown plot
- turnover plot
- M2 confusion matrix
- M2 ROC/PR curve
- M2 calibration curve

---

### `src/llm_features.py`

Optional module for LLM-derived features.

Must be disabled by default.

If enabled, the module must:

- accept only timestamped historical text
- return structured JSON
- cache every feature output
- record prompt hash and model name
- reject any source with timestamp after prediction date

---

### `src/research_logger.py`

Log AI usage, human decisions, and key design choices.

Use append-only JSONL or YAML.

---

## Configuration Example

```yaml
project:
  name: finance-meta-labeling-pipeline
  frequency: weekly
  rebalance: weekly

assets:
  tickers: [SPY, TLT, GLD, VEA, VWO, HYG, VNQ]

split:
  train_start: '2006-01-01'
  train_end: '2020-12-31'
  test_start: '2021-01-01'
  test_end: null

features:
  momentum_windows: [4, 12, 26, 52]
  volatility_windows: [4, 12, 26]
  trend_windows: [10, 40]
  macro_lag_weeks: 4

labels:
  horizon_weeks: 4
  positive_threshold: 0.005
  negative_threshold: -0.005
  transaction_cost_threshold: 0.001

models:
  m1:
    type: rule_based
    long_threshold: 0.50
    short_threshold: -0.50
  m2:
    type: logistic_regression
    threshold: 0.55
    calibrate: true

portfolio:
  allow_short: true
  max_abs_asset_weight: 0.25
  max_gross_exposure: 1.00
  transaction_cost_bps: 5

llm_features:
  enabled: false
  cache_dir: data/features/llm_cache
```

---

## Observability Requirements

Every pipeline run must save:

```text
run_id
config snapshot
data ingest log
validation report
feature schema
training dates
test dates
model parameters
metrics table
plots
research log
```

Use a timestamped run directory:

```text
runs/YYYYMMDD_HHMMSS/
```
