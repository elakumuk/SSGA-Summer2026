# Acceptance Tests and Quality Checklist

## Goal

These tests define whether the project is actually complete. The LLM coding agent should implement these as `pytest` tests where possible and as manual checklist items where code tests are not sufficient.

---

## 1. Data Ingest Tests

### Test: required tickers present

```text
Given the configured asset universe,
When data ingest completes,
Then every ticker should be present in the raw and processed dataset.
```

### Test: date index valid

```text
- index is datetime
- index is sorted ascending
- no duplicate dates
- no future dates beyond run date
```

### Test: adjusted close valid

```text
- adjusted close exists
- adjusted close is positive
- returns can be calculated
```

---

## 2. Data Schema Tests

Expected modeling panel columns:

```text
date
ticker
adj_close
return_1w
forward_return_h
M1_signal
M1_score
meta_label
feature columns
```

Test that:

```text
- no required column is missing
- each ticker/date pair is unique
- feature columns are numeric unless explicitly categorical
```

---

## 3. No Future Leakage Tests

### Feature Alignment

For every feature column:

```text
feature_timestamp <= prediction_timestamp
```

### Label Exclusion

Ensure feature matrix does not contain:

```text
forward_return
future_return
meta_label
trade_return
any label-like column
```

### Train/Test Split

```text
max(train_date) < min(test_date)
```

### Cross-Validation

```text
For every fold:
max(train_date) < min(validation_date)
```

No shuffling allowed.

---

## 4. M1 Tests

M1 predictions must satisfy:

```text
set(predictions).issubset({-1, 0, 1})
```

M1 output must include:

```text
M1_signal
M1_score
```

M1 should generate enough non-zero signals for M2 training. If fewer than a configurable minimum, raise a warning.

---

## 5. M2 Tests

M2 probabilities must satisfy:

```text
0 <= p_success <= 1
```

M2 training rows should only include non-zero M1 signals.

M2 test output must include:

```text
p_success
predicted_meta_label
```

---

## 6. Position Sizing Tests

Weights must satisfy:

```text
abs(weight_i) <= max_abs_asset_weight
sum(abs(weights)) <= max_gross_exposure
```

If `allow_short = false`, then:

```text
weight_i >= 0 for all assets
```

---

## 7. Backtest Tests

### Timing Test

Signal at `t` must earn return from `t` to `t+1`, not return ending at `t`.

### Turnover Test

```text
turnover_t = sum(abs(weight_t - weight_t_minus_1))
```

### Transaction Cost Test

```text
net_return = gross_return - turnover * cost_bps / 10000
```

---

## 8. Diagnostics Tests

Final diagnostics must include:

```text
annualized_return
annualized_volatility
Sharpe
information_ratio
information_coefficient
max_drawdown
rolling_12m_max_drawdown
turnover
hit_rate
precision
recall
F1
AUC
Brier score
```

---

## 9. Benchmark Tests

Backtest must produce these comparison series:

```text
equal_weight_1_7
sixty_forty
m1_only
m1_m2_binary
m1_m2_linear
m1_m2_ecdf
```

---

## 10. Report Checklist

The final report must answer:

1. What data was used?
2. What was the actual available sample period?
3. How were features constructed?
4. How was look-ahead bias avoided?
5. How was M1 trained or specified?
6. How were meta-labels defined?
7. What did M2 predict?
8. How were positions sized?
9. Which benchmark did the strategy beat or fail to beat?
10. Was the improvement from return, volatility reduction, drawdown reduction, or lower exposure?
11. How sensitive are results to transaction costs?
12. How did LLM features help or fail to help?
13. What are the main limitations?

---

## Definition of Done

The project is done only when:

```text
pytest -q passes
pipeline runs from raw data to final report
backtest results are reproducible
all major assumptions are documented
AI/LLM usage is logged
no live trading execution exists
```
