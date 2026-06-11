# Backtest and Evaluation Specification

## Backtest Timing

At each weekly rebalance date `t`:

1. Use features available through `t`.
2. Generate M1 signals for each asset.
3. Generate M2 probability for non-zero M1 signals.
4. Convert M2 probability into target weights.
5. Apply transaction costs based on weight changes.
6. Earn asset returns from `t` to `t+1`.

This prevents look-ahead bias.

---

## Return Accounting

For asset returns:

```text
r_asset_{t+1} = adj_close_{t+1} / adj_close_t - 1
```

For strategy return:

```text
gross_strategy_return_{t+1} = Σ_i weight_{i,t} * r_{i,t+1}
turnover_t = Σ_i abs(weight_{i,t} - weight_{i,t-1})
transaction_cost_t = turnover_t * transaction_cost_bps / 10000
net_strategy_return_{t+1} = gross_strategy_return_{t+1} - transaction_cost_t
```

Default transaction cost:

```text
5 bps per unit turnover
```

Run sensitivity tests at:

```text
0 bps, 5 bps, 10 bps, 25 bps
```

---

## Benchmarks

### Equal-Weight Benchmark

```text
weight_i = 1 / 7
```

Rebalance weekly or monthly.

### 60/40 Benchmark

Default:

```text
Equity bucket = SPY, VEA, VWO, VNQ
Bond bucket = TLT, HYG
Gold = excluded
```

Weights:

```text
SPY = 15%
VEA = 15%
VWO = 15%
VNQ = 15%
TLT = 20%
HYG = 20%
GLD = 0%
```

Optional alternative:

```text
55/35/10 = equities / bonds / gold
```

---

## Required Metrics

### Return Metrics

```text
cumulative_return = cumulative product of (1 + returns) - 1
annualized_return = geometric annualized return
excess_return = strategy_return - benchmark_return
```

### Risk Metrics

Weekly annualization:

```text
annualized_volatility = std(weekly_returns) * sqrt(52)
```

### Sharpe Ratio

```text
Sharpe = annualized_mean_excess_return_over_cash / annualized_volatility
```

If cash/risk-free data is not available, use zero risk-free rate and document that simplification.

### Information Ratio

```text
active_return = strategy_return - benchmark_return
IR = annualized_mean(active_return) / annualized_std(active_return)
```

### Information Coefficient

For M1 score:

```text
IC_t = SpearmanRankCorr(M1_score_{assets,t}, forward_return_{assets,t+h})
```

Report:

```text
mean IC
IC t-stat
12-month rolling IC
IC hit rate = % of IC_t > 0
```

### Maximum Drawdown

Full-period max drawdown:

```text
drawdown_t = equity_curve_t / running_max_equity_curve_t - 1
max_drawdown = min(drawdown_t)
```

12-month max drawdown:

```text
rolling_52w_max_drawdown
```

### Turnover

```text
turnover_t = Σ_i abs(weight_{i,t} - weight_{i,t-1})
average_turnover = mean(turnover_t)
annualized_turnover = average_turnover * 52
```

### Hit Rate

Trade-level hit rate:

```text
hit_rate = number of profitable non-zero M1 trades / total non-zero M1 trades
```

Strategy week hit rate:

```text
weekly_hit_rate = % of weeks with net strategy return > 0
```

---

## M2 Classification Metrics

Report on out-of-sample test set:

```text
accuracy
precision
recall
F1 score
AUC
confusion matrix
Brier score
calibration curve
```

Also report:

```text
mean p_success for winners
mean p_success for losers
probability decile table
```

---

## Required Charts

Save as PNG and include in final report:

1. Cumulative return: strategy variants vs benchmarks.
2. Drawdown curve.
3. Rolling 52-week Sharpe.
4. Rolling 52-week max drawdown.
5. Monthly/weekly turnover.
6. Asset weights through time.
7. M1 signal heatmap by asset.
8. M2 probability histogram.
9. M2 calibration curve.
10. IC time series and rolling IC.

---

## Results Table Template

| Strategy | Ann. Return | Ann. Vol | Sharpe | Excess Return vs EW | IR vs EW | Max DD | 12M Max DD | Turnover | Hit Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal Weight | | | | | | | | | |
| 60/40 | | | | | | | | | |
| M1 Only | | | | | | | | | |
| M1 + M2 Binary | | | | | | | | | |
| M1 + M2 Linear | | | | | | | | | |
| M1 + M2 ECDF | | | | | | | | | |

---

## Robustness Checks

Run at least these checks:

1. Different M2 probability thresholds: `0.50, 0.55, 0.60, 0.65`.
2. Different transaction costs: `0, 5, 10, 25 bps`.
3. Different label horizons: `1, 4, 8, 12 weeks`.
4. With and without macro features.
5. With and without LLM features.
6. Long-only version vs long/short version.
7. Equal risk budget vs inverse-volatility risk budget.

---

## Interpretation Rules

When writing the final report:

- Separate M1 signal quality from M2 filtering quality.
- Explain whether performance improved because of better selection, lower exposure, lower volatility, or luck.
- Discuss drawdown and turnover, not only return.
- Report cases where M2 hurts performance.
- Clearly state that results are historical and not a prediction of future performance.
