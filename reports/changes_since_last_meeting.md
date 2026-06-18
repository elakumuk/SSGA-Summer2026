# Changes Since Last Review

A recap of the feedback from the previous meeting and the concrete changes made in
the code in response.

## Feedback → change

| # | Feedback | What changed in the code | Where |
|---|---|---|---|
| 1 | Keep **M1 as simple as possible**; static parameters. | M1 reduced to a single static factor — `technical` (momentum + trend, merged). No macro, no risk weighting inside M1. Linear, fixed ratio. | `config/config.yaml` → `m1.factors = {technical: 1.0}`, `src/m1.py`, `src/features.py` |
| 2 | ML layer = **logistic regression only**. | M2 is a single logistic regression, refit on a rolling ~12-month window. No more complex model. | `src/m2.py` |
| 3 | **Static factors → M1, dynamic factors → M2** (signals / interesting market moments go to M2). | Macro tilt, volatility, and regime signals moved out of M1 into M2 as features. M1 keeps only the static price signal. | `src/features.py`, `src/m2.py`, `run_strategy.py` |
| 4 | Use the **macro / regime datasets** in M2 (previous groups ignored them). | M2's feature set now includes macro and regime features: VIX, yield-curve slope, credit spread, growth trend, inflation trend — alongside the individual factors. | `src/features.py` (`regime_features`), `src/m2.py` |
| 5 | Add **multiple, simple evaluations** (e.g. mean difference, MAPE). | Evaluation suite reports F1, AUC-ROC, AUC-PR and calibration, plus simple error metrics: **mean difference, MAE, Brier, and a calibration MAPE.** | `src/evaluation.py` |
| 6 | Set up an **out-of-sample test** to confirm the model; don't over-engineer the data split. | Chronological train/test split (train ≤2020, test 2021+), a 4-week embargo to prevent label leakage, and a walk-forward across multiple windows. | `config/config.yaml`, `src/m2.py`, `run_walkforward.py` |
| 7 | Prefer the **index** over the ETF for research. | Signals computed on index series where available (`^GSPC`, FRED REIT) and the longest free proxy elsewhere; one-flag switch (`use_index_signal`). | `src/data.py`, `DATA_SOURCES.md`, `fetch_indices.py` |
| 8 | Shorting is not critical early; think benchmark-relative / information ratio. | Portfolio expresses views as bounded active tilts around the benchmark (an underweight is an implicit short); information ratio is the headline metric. | `src/portfolio.py`, `src/backtest.py` |

## What the changes produced

- **M1** beats all baselines on risk-adjusted return and cuts drawdown the most
  (-22% vs -30% for equal-weight); its edge is concentrated in recent regimes.
- **M2** does not add value yet — the new simple metrics confirm it: low average bias
  (mean difference ≈ 0.01–0.04) but MAE ≈ 0.50, Brier ≈ 0.31, calibration MAPE ≈ 0.49,
  and AUC ≈ 0.46 out-of-sample. The probabilities carry no conditional signal.

## Open items / for discussion

- **M2 reformulation** — how should the meta-label be redefined so it extracts signal
  (regime gate vs factor-timer vs different target/horizon)? How should the evaluation
  metrics feed back into M2? (No standard answer — exploratory.)
- **Possible additional layer ("M3")** — to be clarified.
- **Data** — whether to source true long-history index data (pre-2012) from Bloomberg.
