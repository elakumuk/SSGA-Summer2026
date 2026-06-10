# Data and Feature Engineering Specification

## Data Sources

### Market Data

Default tickers:

```text
SPY, TLT, GLD, VEA, VWO, HYG, VNQ
```

Preferred fields:

```text
open, high, low, close, adjusted close, volume
```

Use adjusted close for return calculations.

### Macro / Regime Data

Use FRED or Bloomberg equivalents.

Suggested FRED series:

| Series | Meaning | Use |
|---|---|---|
| CPIAUCSL | Consumer price index | Inflation trend |
| UNRATE | Unemployment rate | Labor/growth proxy |
| INDPRO | Industrial production | Growth proxy |
| FEDFUNDS | Fed funds rate | Policy stance |
| DGS10 | 10-year Treasury yield | Rate level |
| T10Y2Y | Yield curve slope | Growth/recession risk |
| BAA10Y | Credit spread proxy | Credit stress |

Market risk proxy:

```text
^VIX from Yahoo Finance, if available
```

---

## Frequency Conversion

Use daily data as raw input. Convert to weekly data:

```python
weekly_price = daily_adj_close.resample('W-FRI').last()
weekly_return = weekly_price.pct_change()
```

If Friday is not a trading day, use the last available trading date in that week.

---

## Missing Data Rules

1. Drop assets with insufficient history only if necessary.
2. Prefer a balanced panel for modeling.
3. Report the effective start date after aligning all assets.
4. For macro data, use release-lag assumptions. Default: lag macro features by 4 weeks unless exact release date is available.
5. Never backfill future macro observations into past dates.

---

## Feature Families

### 1. Momentum / Relative Strength

For each asset:

```text
mom_4w  = price_t / price_t-4  - 1
mom_12w = price_t / price_t-12 - 1
mom_26w = price_t / price_t-26 - 1
mom_52w = price_t / price_t-52 - 1
```

Shift features by one period if the signal is formed before execution.

Cross-sectional relative strength:

```text
rank_mom_12w = percentile rank of mom_12w across assets at date t
```

---

### 2. Trend

```text
ma_fast = rolling_mean(price, 10 weeks)
ma_slow = rolling_mean(price, 40 weeks)
trend_signal = ma_fast / ma_slow - 1
```

---

### 3. Volatility / Risk

```text
vol_4w  = std(weekly_return, 4 weeks)  * sqrt(52)
vol_12w = std(weekly_return, 12 weeks) * sqrt(52)
vol_26w = std(weekly_return, 26 weeks) * sqrt(52)
```

Other risk features:

```text
drawdown_26w = price / rolling_max(price, 26 weeks) - 1
skew_26w = rolling skewness
corr_to_spy_26w = rolling correlation to SPY
```

---

### 4. Valuation / Carry Proxies

ETF-level valuation may be limited. Use proxies carefully:

- TLT: yield level / yield change as rate environment proxy.
- HYG: credit spread proxy such as BAA10Y.
- GLD: real-rate proxy using nominal rates minus inflation trend.
- Equities: earnings yield would require external data; optional if Bloomberg data is available.

If the data is not available, document the omission rather than inventing values.

---

### 5. Macro Regime Features

Create standardized regime features:

```text
inflation_trend = CPI YoY change, lagged
policy_rate_change = FEDFUNDS change, lagged
yield_curve = T10Y2Y, lagged
credit_stress = BAA10Y, lagged
growth_trend = INDPRO YoY change, lagged
unemployment_change = UNRATE 3-month change, lagged
```

Create binary / categorical regimes:

```text
inflation_up = inflation_trend > rolling_median(inflation_trend, 3 years)
growth_down = growth_trend < rolling_median(growth_trend, 3 years)
risk_off = VIX > rolling_75th_percentile(VIX, 3 years)
curve_inverted = T10Y2Y < 0
```

---

### 6. False-Positive Features for M2

These features help M2 identify when M1 is likely to be wrong:

```text
vix_level
vix_change_4w
cross_asset_dispersion_4w
cross_asset_dispersion_12w
average_pairwise_correlation_26w
rolling_m1_hit_rate_26w
rolling_m1_precision_26w
rolling_m1_return_26w
rolling_strategy_drawdown_26w
macro_regime_flags
```

M1 rolling performance features must be computed using past predictions only.

---

### 7. Optional LLM-Derived Features

Only use timestamped historical text, such as archived market commentary, Bloomberg headlines, FOMC statements, or dated macro summaries.

Structured output schema:

```json
{
  "date": "YYYY-MM-DD",
  "source_id": "string",
  "risk_sentiment": -0.2,
  "inflation_pressure": 0.5,
  "growth_slowdown": 0.1,
  "policy_tightness": 0.4,
  "macro_uncertainty": 0.7,
  "dominant_narrative": "inflation / growth / policy / credit / geopolitical / other",
  "confidence": 0.6
}
```

Rules:

- Feature date must be less than or equal to prediction date.
- Save raw source metadata, prompt, model name, and output.
- Cache outputs; do not regenerate silently.
- Compare strategy with and without LLM features.

---

## Feature Normalization

For each date, normalize cross-sectionally where appropriate:

```text
z_asset_feature = (feature - cross_sectional_mean) / cross_sectional_std
```

For time-series features, use rolling z-scores based only on historical data:

```text
z_ts = (feature_t - rolling_mean_t) / rolling_std_t
```

Winsorize extreme values using training data thresholds only.

---

## Feature Store Output

Save a modeling panel:

```text
data/features/model_panel.parquet
```

Expected schema:

```text
date: datetime
ticker: string
adj_close: float
return_1w: float
forward_return_4w: float
m1_target: int
meta_label: int or null
feature columns: float / category
```

---

## Leakage Checklist

Before modeling, assert:

- All feature dates are <= prediction dates.
- Forward returns appear only in label columns.
- No label columns are included in feature matrices.
- Train/test split is strictly chronological.
- Cross-validation uses expanding or rolling windows.
- LLM features have source timestamps.
