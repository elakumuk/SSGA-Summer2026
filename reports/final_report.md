# Final Report: AI-Augmented Multi-Asset Meta-Labeling Pipeline

This run executes the pipeline **twice**: once with M1 **long-only** (no short signals) and once with M1 **long/short** enabled.

**Research use only — not investment advice.**

## Sample Period

| Item | Value |
| --- | --- |
| Effective start | 2007-07-27 |
| Effective end | 2026-06-12 |
| Train period | 2006-01-01 to 2020-12-31 |
| Test period (M2 evaluation) | 2021-01-01 to latest |
| Assets | SPY, TLT, GLD, VEA, VWO, HYG, VNQ |

## Configuration Parameters Affecting Performance

The pipeline reads defaults from `config/config.yaml`. **Split dates** can also be set at runtime without editing the file (see CLI below). Other parameters require config edits.

### Train / Test Split

| Parameter | Current value | Performance impact |
| --- | --- | --- |
| `split.train_start` | 2006-01-01 | Earliest date for M1 threshold tuning, feature winsorization, and M2 training |
| `split.train_end` | 2020-12-31 | Last in-sample date; defines how much history the models learn from |
| `split.test_start` | 2021-01-01 | Out-of-sample evaluation begins here (M2 metrics, IC, reported Sharpe) |
| `split.test_end` | latest (open-ended) | Optional cap on the evaluation window |

**CLI overrides** (ISO dates, applied after loading config):

```bash
python -m src.run_pipeline --train-end 2018-12-31 --test-start 2019-01-01
python -m src.run_pipeline --train-start 2008-01-01 --train-end 2015-12-31 --test-start 2016-01-01
```

Shorter train windows reduce overfitting risk but give fewer signals for M2; earlier test starts include more regimes (e.g. 2022 rate hikes) in evaluation.

### M1 Rule-Based Side Model

| Parameter | Current value | Performance impact |
| --- | --- | --- |
| `models.m1.weights` | momentum=0.45, trend=0.25, macro=0.2, risk=0.1 | Relative importance of factor families in the composite score |
| `models.m1.optimize_thresholds` | True | When true, long/short cutoffs are tuned on the train set only |
| `models.m1.long_quantile` / `short_quantile` | 0.58 / 0.22 | Starting quantiles for threshold search (higher long quantile → fewer longs) |
| `models.m1.allow_short` | False | Default shorting flag; pipeline always runs both long-only and long/short modes |
| `models.m1.asset_class_tilts` | True | Macro tilts by asset class (equity, bonds, credit, gold, REIT) |

### M2 Meta-Labeling

| Parameter | Current value | Performance impact |
| --- | --- | --- |
| `models.m2.threshold` | 0.55 | Minimum P(success) to take full size; higher → fewer trades, often lower turnover |
| `models.m2.calibrate` | True | Probability calibration on train data; improves threshold interpretability |
| `models.m2.type` | logistic_regression | Classifier used for meta-labels |

### Labels (M1 targets & M2 supervision)

| Parameter | Current value | Performance impact |
| --- | --- | --- |
| `labels.horizon_weeks` | 4 | Forward return horizon for profitability labels |
| `labels.positive_threshold` | 0.005 | Minimum forward return to label a long as successful |
| `labels.negative_threshold` | -0.005 | Forward return threshold for short success |
| `labels.transaction_cost_threshold` | 0.001 | Cost hurdle embedded in label construction |

### Portfolio & Costs

| Parameter | Current value | Performance impact |
| --- | --- | --- |
| `portfolio.transaction_cost_bps` | 5 | Round-trip cost per unit turnover; higher values drag net returns |
| `portfolio.max_gross_exposure` | 1.0 | Cap on sum of absolute weights |
| `portfolio.max_abs_asset_weight` | 0.25 | Per-asset weight ceiling |
| `portfolio.sizing_mode` | linear | How M2 probability maps to position size (binary / linear / ecdf) |

### Features

| Parameter | Current value | Performance impact |
| --- | --- | --- |
| `features.momentum_windows` | [4, 12, 26, 52] | Lookback weeks for momentum factors |
| `features.macro_lag_weeks` | 4 | Release lag applied to macro series (reduces look-ahead) |
| `features.winsorize_pct` | 0.01 | Train-set winsorization of extreme feature values |

## Data & Components Used

The pipeline combines **seven tradable ETF proxies** for major asset classes plus **macro/risk indicators** for regime features. Prices are resampled to **weekly** (Friday close) from daily adjusted-close data.

| Field | Value |
| --- | --- |
| Sample start | 2007-07-27 |
| Sample end | 2026-06-12 |
| Frequency | Weekly (W-FRI) |
| Price field | Adjusted close |

### Tradable ETF Components

| Ticker | Instrument | Proxy / Benchmark | Asset Class | Role in Portfolio | Data Source |
| --- | --- | --- | --- | --- | --- |
| SPY | SPDR S&P 500 ETF Trust | S&P 500 (proxy) | U.S. Equities | U.S. large-cap equity beta and growth exposure | yfinance — adjusted close, weekly |
| TLT | iShares 20+ Year Treasury Bond ETF | Long-duration U.S. Treasuries | Government Bonds | Duration and defensive interest-rate exposure | yfinance — adjusted close, weekly |
| GLD | SPDR Gold Shares | Gold spot price (proxy) | Commodities / Gold | Inflation hedge and safe-haven commodity exposure | yfinance — adjusted close, weekly |
| VEA | Vanguard FTSE Developed Markets ETF | Developed ex-U.S. equities | International Equities | Geographic diversification outside the U.S. | yfinance — adjusted close, weekly |
| VWO | Vanguard FTSE Emerging Markets ETF | Emerging market equities | Emerging Market Equities | Emerging market growth and risk premia | yfinance — adjusted close, weekly |
| HYG | iShares iBoxx High Yield Corporate Bond ETF | U.S. high-yield corporate bonds | Credit / High Yield | Credit risk and income exposure | yfinance — adjusted close, weekly |
| VNQ | Vanguard Real Estate ETF | U.S. REITs | Real Estate (REITs) | Real estate and rate-sensitive income exposure | yfinance — adjusted close, weekly |

### Macro & Risk Indicators (features only)

These series are **not traded** in the backtest. They feed M1/M2 regime and false-positive features, lagged by 4 weeks to approximate publication delay.

| Series | Description | Use | Source |
| --- | --- | --- | --- |
| CPIAUCSL | Consumer Price Index | Inflation trend and regime indicator | FRED — lagged 4 weeks in features |
| UNRATE | Unemployment Rate | Labor market / growth proxy | FRED — lagged 4 weeks in features |
| INDPRO | Industrial Production Index | Economic growth proxy | FRED — lagged 4 weeks in features |
| FEDFUNDS | Federal Funds Rate | Monetary policy stance | FRED — lagged 4 weeks in features |
| DGS10 | 10-Year Treasury Yield | Long-term interest rate level | FRED — lagged 4 weeks in features |
| T10Y2Y | 10Y–2Y Treasury Spread | Yield curve slope / recession signal | FRED — lagged 4 weeks in features |
| BAA10Y | Baa–10Y Credit Spread | Credit stress indicator | FRED — lagged 4 weeks in features |
| VIX | CBOE Volatility Index | Equity risk sentiment (risk-on / risk-off) | yfinance (^VIX) — used in features, not traded |

## Individual Asset Performance (Buy-and-Hold)

Each row below is a **standalone buy-and-hold** of one ETF: 100% allocated to that asset, rebalanced weekly, **no transaction costs**, no M1/M2 overlay. This shows how each building block performed on its own before any strategy logic. Charts also overlay **M1** and **M1+M2** portfolio models (long-only and long/short) for comparison.

### Full Sample

| Ticker | Asset | Class | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Total Return | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | SPDR S&P 500 ETF Trust | U.S. Equities | 10.8782% | 18.2811% | 0.5951 | -54.6130% | 607.1129% | 57.3604% |
| TLT | iShares 20+ Year Treasury Bond ETF | Government Bonds | 2.9717% | 14.3522% | 0.2071 | -47.8267% | 74.1417% | 53.5025% |
| GLD | SPDR Gold Shares | Commodities / Gold | 9.6506% | 17.1340% | 0.5632 | -44.7446% | 472.6647% | 54.9239% |
| VEA | Vanguard FTSE Developed Markets ETF | International Equities | 5.1290% | 19.6234% | 0.2614 | -59.0021% | 157.9133% | 55.5330% |
| VWO | Vanguard FTSE Emerging Markets ETF | Emerging Market Equities | 4.0303% | 22.5913% | 0.1784 | -63.8086% | 111.3700% | 52.7919% |
| HYG | iShares iBoxx High Yield Corporate Bond ETF | Credit / High Yield | 5.3554% | 11.2039% | 0.4780 | -33.0009% | 168.6410% | 58.4772% |
| VNQ | Vanguard Real Estate ETF | Real Estate (REITs) | 6.5646% | 26.0759% | 0.2518 | -70.2120% | 233.4693% | 55.3299% |

![Individual asset cumulative returns](assets/asset_cumulative_returns.png)

![Individual asset metrics](assets/asset_metrics_bars.png)

### Train Period (2006-01-01 to 2020-12-31)

| Ticker | Asset | Class | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Total Return | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | SPDR S&P 500 ETF Trust | U.S. Equities | 9.3926% | 19.1490% | 0.4905 | -54.6130% | 234.8409% | 57.8571% |
| TLT | iShares 20+ Year Treasury Bond ETF | Government Bonds | 7.6818% | 14.2777% | 0.5380 | -25.1822% | 170.8210% | 56.0000% |
| GLD | SPDR Gold Shares | Commodities / Gold | 7.6458% | 17.4949% | 0.4370 | -44.7446% | 169.6071% | 54.2857% |
| VEA | Vanguard FTSE Developed Markets ETF | International Equities | 3.0057% | 20.8513% | 0.1442 | -59.0021% | 48.9824% | 55.1429% |
| VWO | Vanguard FTSE Emerging Markets ETF | Emerging Market Equities | 3.1832% | 24.6520% | 0.1291 | -63.8086% | 52.4754% | 52.0000% |
| HYG | iShares iBoxx High Yield Corporate Bond ETF | Credit / High Yield | 6.0127% | 12.5519% | 0.4790 | -33.0009% | 119.4601% | 60.2857% |
| VNQ | Vanguard Real Estate ETF | Real Estate (REITs) | 6.5635% | 28.6584% | 0.2290 | -70.2120% | 135.3171% | 56.7143% |

### Test Period (2021-01-01 to latest)

| Ticker | Asset | Class | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Total Return | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPY | SPDR S&P 500 ETF Trust | U.S. Equities | 14.6131% | 15.9747% | 0.9148 | -23.9272% | 111.1788% | 56.1404% |
| TLT | iShares 20+ Year Treasury Bond ETF | Government Bonds | -7.7410% | 14.4461% | -0.5359 | -43.7988% | -35.6986% | 47.3684% |
| GLD | SPDR Gold Shares | Commodities / Gold | 14.7345% | 16.2274% | 0.9080 | -22.5674% | 112.4071% | 56.4912% |
| VEA | Vanguard FTSE Developed Markets ETF | International Equities | 10.5316% | 16.2355% | 0.6487 | -29.4775% | 73.1166% | 56.4912% |
| VWO | Vanguard FTSE Emerging Markets ETF | Emerging Market Equities | 6.1403% | 16.5087% | 0.3719 | -33.4800% | 38.6257% | 54.7368% |
| HYG | iShares iBoxx High Yield Corporate Bond ETF | Credit / High Yield | 3.7583% | 6.8592% | 0.5479 | -15.3951% | 22.4099% | 54.0351% |
| VNQ | Vanguard Real Estate ETF | Real Estate (REITs) | 6.5674% | 18.2858% | 0.3592 | -34.2941% | 41.7106% | 51.9298% |

![Train vs test asset returns](assets/asset_train_test_returns.png)

### Per-Asset Highlights

- **SPY** (S&P 500 (proxy)): 10.8782% annualized, Sharpe 0.5951, max drawdown -54.6130% — U.S. large-cap equity beta and growth exposure.
- **GLD** (Gold spot price (proxy)): 9.6506% annualized, Sharpe 0.5632, max drawdown -44.7446% — Inflation hedge and safe-haven commodity exposure.
- **VNQ** (U.S. REITs): 6.5646% annualized, Sharpe 0.2518, max drawdown -70.2120% — Real estate and rate-sensitive income exposure.
- **HYG** (U.S. high-yield corporate bonds): 5.3554% annualized, Sharpe 0.4780, max drawdown -33.0009% — Credit risk and income exposure.
- **VEA** (Developed ex-U.S. equities): 5.1290% annualized, Sharpe 0.2614, max drawdown -59.0021% — Geographic diversification outside the U.S..
- **VWO** (Emerging market equities): 4.0303% annualized, Sharpe 0.1784, max drawdown -63.8086% — Emerging market growth and risk premia.
- **TLT** (Long-duration U.S. Treasuries): 2.9717% annualized, Sharpe 0.2071, max drawdown -47.8267% — Duration and defensive interest-rate exposure.

See also: [assets/asset_component_analysis.md](assets/asset_component_analysis.md) for the full standalone write-up.

## M2 Performance by M1 Signal

M1 outputs three signal types per asset-week: **short (−1)**, **flat (0)**, or **long (+1)**. M2 only trains and predicts on non-zero signals. Below we break out **test-set** trade outcomes and classifier quality within each M1 group.

- **M1 hit rate**: share of trades with positive forward return (after cost hurdle)
- **M2 approval rate**: share of trades where `p_success` ≥ threshold
- **Hit rate (M2 approved)**: profitability among trades M2 kept

### Long-Only vs Long/Short Comparison

![M2 by M1 signal comparison](mode_comparison/m2_m1_signal_comparison.png)

*Left: mean forward trade return by M1 signal. Right: M1 vs M2-filtered hit rates (long-only has no short bucket).*

### Long Only (no shorts)

`allow_short=False` — M2 threshold = 0.55

| M1 Signal | Observations | Share | Labeled Trades | M1 Hit Rate | Mean Trade Return | M2 Approval Rate | Hit Rate (M2 Approved) | M2 Precision | M2 Recall | M2 F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flat (0) | 1129 | 56.5915% | 0 | — | — | — | — | — | — | — |
| Long (+1) | 866 | 43.4085% | 866 | 59.9307% | 0.7690% | 95.9584% | 60.0481% | 0.6005 | 0.9615 | 0.7393 |

![M2 by M1 signal — Long Only (no shorts)](final/long_only/m2_m1_signal_analysis.png)
*Long-only mode: M1 never emits −1; shorts are disabled at the signal layer.*

### Long / Short

`allow_short=True` — M2 threshold = 0.55

| M1 Signal | Observations | Share | Labeled Trades | M1 Hit Rate | Mean Trade Return | M2 Approval Rate | Hit Rate (M2 Approved) | M2 Precision | M2 Recall | M2 F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short (−1) | 234 | 11.7293% | 234 | 42.7350% | -0.4143% | 6.4103% | 33.3333% | 0.3333 | 0.0500 | 0.0870 |
| Flat (0) | 895 | 44.8622% | 0 | — | — | — | — | — | — | — |
| Long (+1) | 866 | 43.4085% | 866 | 59.9307% | 0.7690% | 86.3741% | 59.0909% | 0.5909 | 0.8516 | 0.6977 |

![M2 by M1 signal — Long / Short](final/long_short/m2_m1_signal_analysis.png)
## M1 Mode Comparison (M1 Only)

| Mode | Ann. Return | Sharpe | Max Drawdown |
| --- | --- | --- | --- |
| Long Only (no shorts) | 3.7800% | 0.6605 | -17.1473% |
| Long / Short | 2.2935% | 0.4037 | -13.0652% |

![M1 mode comparison](mode_comparison/m1_mode_comparison.png)

*Left: cumulative M1-only returns. Right: return, Sharpe (×10), and drawdown by mode.*

## Results: Long Only (no shorts)

`allow_short=False` — outputs in `data/backtests/long_only/`

| Strategy | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Excess vs EW | Info Ratio | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal Weight (1/7) | 7.3796% | 12.8982% | 0.5721 | -39.4430% | 0.0000% | 0.0000 | 56.0852% |
| 60/40 Benchmark | 6.5640% | 13.2675% | 0.4947 | -43.1363% | -0.8156% | -0.2850 | 56.4909% |
| M1 Only | 3.7800% | 5.7230% | 0.6605 | -17.1473% | -3.5997% | -0.4362 | 56.7951% |
| M1 + M2 (Binary) | 4.1713% | 5.1729% | 0.8064 | -11.5267% | -3.2083% | -0.3755 | 56.8966% |
| M1 + M2 (Linear) | 0.9854% | 1.1686% | 0.8432 | -2.4647% | -6.3943% | -0.5751 | 56.8966% |
| M1 + M2 (ECDF) | 2.6017% | 3.0197% | 0.8616 | -5.3080% | -4.7780% | -0.4734 | 57.8093% |

### Charts (Long Only (no shorts))

![strategy_cumulative_returns.png](final/long_only/strategy_cumulative_returns.png)

![strategy_drawdown.png](final/long_only/strategy_drawdown.png)

![strategy_sharpe_comparison.png](final/long_only/strategy_sharpe_comparison.png)

![strategy_risk_return.png](final/long_only/strategy_risk_return.png)

![m2_classification_summary.png](final/long_only/m2_classification_summary.png)

![m2_m1_signal_analysis.png](final/long_only/m2_m1_signal_analysis.png)

### M2 Quality — Long Only (no shorts) (Test Set)

| Metric | Value | Meaning |
| --- | --- | --- |
| Accuracy | 0.5935 | Share of correct meta-label predictions |
| Precision | 0.6005 | Approved trades that were actually profitable |
| Recall | 0.9615 | Profitable trades that M2 approved |
| F1 Score | 0.7393 | Balance of precision and recall |
| AUC | 0.4963 | Ranking quality of M2 probabilities |
| Brier Score | 0.2410 | Probability calibration error (lower is better) |
| Mean IC | 0.0957 | Spearman rank correlation of M1 scores vs forward returns |

## Results: Long / Short

`allow_short=True` — outputs in `data/backtests/long_short/`

| Strategy | Ann. Return | Ann. Volatility | Sharpe | Max Drawdown | Excess vs EW | Info Ratio | Weekly Hit Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal Weight (1/7) | 7.3796% | 12.8982% | 0.5721 | -39.4430% | 0.0000% | 0.0000 | 56.0852% |
| 60/40 Benchmark | 6.5640% | 13.2675% | 0.4947 | -43.1363% | -0.8156% | -0.2850 | 56.4909% |
| M1 Only | 2.2935% | 5.6811% | 0.4037 | -13.0652% | -5.0861% | -0.4604 | 53.5497% |
| M1 + M2 (Binary) | 3.5312% | 4.9457% | 0.7140 | -12.9937% | -3.8485% | -0.4034 | 56.3895% |
| M1 + M2 (Linear) | 0.6652% | 0.9810% | 0.6781 | -2.7018% | -6.7144% | -0.5913 | 56.5923% |
| M1 + M2 (ECDF) | 2.2807% | 3.3591% | 0.6790 | -8.6466% | -5.0989% | -0.5028 | 56.7951% |

### Charts (Long / Short)

![strategy_cumulative_returns.png](final/long_short/strategy_cumulative_returns.png)

![strategy_drawdown.png](final/long_short/strategy_drawdown.png)

![strategy_sharpe_comparison.png](final/long_short/strategy_sharpe_comparison.png)

![strategy_risk_return.png](final/long_short/strategy_risk_return.png)

![m2_classification_summary.png](final/long_short/m2_classification_summary.png)

![m2_m1_signal_analysis.png](final/long_short/m2_m1_signal_analysis.png)

### M2 Quality — Long / Short (Test Set)

| Metric | Value | Meaning |
| --- | --- | --- |
| Accuracy | 0.5564 | Share of correct meta-label predictions |
| Precision | 0.5858 | Approved trades that were actually profitable |
| Recall | 0.7221 | Profitable trades that M2 approved |
| F1 Score | 0.6469 | Balance of precision and recall |
| AUC | 0.5491 | Ranking quality of M2 probabilities |
| Brier Score | 0.2442 | Probability calibration error (lower is better) |
| Mean IC | 0.0957 | Spearman rank correlation of M1 scores vs forward returns |

### How to read the metrics

| Metric | Interpretation |
| --- | --- |
| **Ann. Return** | Geometric average yearly portfolio return after transaction costs |
| **Ann. Volatility** | Standard deviation of weekly returns, scaled to a year |
| **Sharpe** | Return per unit of risk (higher is better; assumes 0% risk-free rate) |
| **Max Drawdown** | Largest peak-to-trough loss over the full sample |
| **Excess vs EW** | Strategy return minus equal-weight benchmark return |
| **Info Ratio** | Consistency of outperformance vs equal-weight |
| **Weekly Hit Rate** | Fraction of weeks with positive net strategy return |

## Key Takeaways

1. **Long-only M1** avoids short exposure, which often hurts in upward-trending ETF samples.
2. **Long/short M1** can increase activity but shorts may reduce returns if poorly timed.
3. **M2 meta-labeling** adjusts position size on top of whichever M1 mode is used.
4. Compare both modes above to see whether shorts add value in this universe.

## Look-Ahead Controls

- Features use only data available at signal time (`shift(1)` on rolling windows)
- Macro series lagged 4 weeks to approximate release delay
- Strict chronological train/test split (train 2006-01-01–2020-12-31, test 2021-01-01–latest)

## Limitations

- yfinance and FRED are research-grade fallbacks, not institutional data
- Past performance does not predict future results
