# Next Steps and Review Notes

This branch is now in a good reviewer-facing state: the project narrative is documented, the data/ETL caveats are explicit, M1/M2 behavior is easier to explain, and the generated report separates full-sample and test-period strategy metrics.

## Current Assessment

The strongest current interpretation is a **long-only weekly top-K allocator**:

- **M1** ranks the seven ETF universe and selects the strongest names each week.
- **M2** should be framed as an exposure/risk-shaping layer, not a strong standalone alpha filter.
- **Portfolio controls** apply transaction costs, caps, and volatility targeting.
- **Long-only** is the main research sleeve; long/short remains diagnostic because short timing has hurt in this ETF universe.

The headline result is promising but should remain carefully worded. M1-only is now close to equal-weight return with better Sharpe and much lower drawdown on the full sample. The final report also shows 2021+ test-period portfolio metrics, where long-only M1-only performs competitively versus equal-weight. M1+M2 ECDF remains the best risk-adjusted combination, while M2 linear is too defensive to be the main return sleeve.

## What Is Strong

1. **The branch has a clear story.** `README.md`, `PROJECT_SUMMARY.md`, `ARCHITECTURE_BRIEFING.md`, and `reports/final_report.md` now describe the same M1/M2 architecture and current results.

2. **The model roles are clearer.** M1 is the opportunity selector. M2 is a probability-based sizing layer. This is more defensible than claiming M2 is a powerful profitability filter.

3. **The data process is documented.** `DATA_SOURCES_AND_ETL.md` explains yfinance/FRED sources, cache behavior, macro fallback logic, validation, and research-grade limitations.

4. **The reporting is more reviewable.** `reports/final_report.md` now separates full-sample strategy metrics from test-period strategy metrics and includes M1 exposure / IC diagnostics.

5. **The implementation is tested.** The latest validation passed the full pipeline run and the full pytest suite.

## Main Remaining Risks

1. **Walk-forward robustness is not proven yet.** The current test-period table helps, but the most important next validation is rolling or expanding walk-forward evaluation across multiple train/test windows.

2. **The grid search is historical.** The checked-in 40-run grid search predates the current top-K / 12% vol-target / no-conviction defaults. It is useful sensitivity evidence, not final tuning proof for the latest configuration.

3. **M2 classifier quality is modest.** M2 improves risk-adjusted behavior through sizing, especially ECDF, but its AUC and binary filtering behavior are not strong enough to describe it as a reliable trade rejector.

4. **Benchmark comparisons need context.** Equal-weight is shown with 0 bps transaction costs, while strategies pay configured turnover costs. M1 also uses lower average gross exposure than equal-weight, so comparisons should emphasize risk efficiency rather than unconditional alpha.

5. **Data are research-grade.** yfinance and FRED are acceptable for the class/research setting but not enough for institutional claims without point-in-time and vendor-quality data.

## Recommended Next Work

### 1. Add Walk-Forward Validation

This is the highest-priority next step. Implement a script or module that runs the pipeline over multiple chronological windows, for example:

- Train 2007-2013, test 2014-2016
- Train 2007-2015, test 2016-2018
- Train 2007-2017, test 2018-2020
- Train 2007-2020, test 2021-latest

For each window, report M1-only, M1+M2 binary, M1+M2 linear, M1+M2 ECDF, equal-weight, and 60/40. The key output should be a compact table of annualized return, Sharpe, max drawdown, hit rate, and excess return versus equal-weight.

Success criteria:

- M1-only should remain competitive with equal-weight on risk-adjusted metrics across most windows.
- M1+M2 ECDF should consistently improve Sharpe or drawdown versus M1-only.
- Any weak regime should be clearly identified rather than hidden.

### 2. Re-Run Grid Search On Current Defaults

After walk-forward validation exists, re-run the grid search using the current top-K / volatility-targeted M1 design. The old sweep should not be treated as current tuning evidence.

Recommended grid dimensions:

- `top_k`: 2, 3, 4
- `portfolio.vol_target_ann`: 0.10, 0.12, 0.14
- M2 sizing mode: binary, linear, ECDF
- Train/test split windows, ideally using the walk-forward setup

Do not spend much grid budget on `m2.threshold` when ranking linear sizing, because linear sizing uses continuous probability rather than the binary threshold.

### 3. Add A Fairer Benchmark Comparison

Add at least one benchmark adjustment:

- Equal-weight with the same transaction-cost assumption.
- Equal-weight scaled to the same realized volatility as M1.
- Equal-weight scaled to similar gross exposure.

This will make it easier to separate real selection value from lower exposure and lower realized risk.

### 4. Improve M2 Diagnostics

M2 should be evaluated as a sizing/risk model:

- Calibration by probability bucket.
- ECDF decile returns.
- Approval rate by regime and asset.
- Return and drawdown contribution by M2 probability bucket.

The central question should be: "Does M2 assign larger sizes to trades that improve portfolio risk-adjusted outcomes?"

### 5. Separate Short-Side Research

Do not force the same M1 logic to work symmetrically for long and short trades. If shorting remains a goal, treat it as a separate research task:

- Separate short labels.
- Separate short thresholds or top-K bottom selection.
- Regime filters for equity/bond/risk-on markets.
- Borrow/cost assumptions if moving beyond research simulation.

### 6. Tighten Reproducibility

Before a final review or presentation:

- Keep one canonical run directory or manifest for the current baseline.
- Avoid committing every timestamped experiment run.
- Add a short command section that explains exactly how to regenerate `reports/final_report.md`.
- Consider storing a lightweight `CURRENT_BASELINE.md` with the config hash, run timestamp, and key metrics.

## Suggested Reviewer Framing

Use this framing when presenting the branch:

> This is a research-grade multi-asset ETF allocation pipeline. M1 is the main weekly selector, M2 is a meta-labeling layer used mostly for exposure shaping, and the portfolio layer applies costs and risk controls. The current long-only M1 result is promising because it reaches similar return to equal-weight with better Sharpe and lower drawdown. The main remaining work is to prove robustness through walk-forward validation and fairer benchmark comparisons.

## Immediate Priority

The next commit should implement **walk-forward validation** and add a generated walk-forward summary table to the final report. That is the most important step before making stronger out-of-sample claims.
