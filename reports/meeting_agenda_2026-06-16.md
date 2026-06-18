# SSGA Meta-Labeling — Meeting Notes
**Date:** 2026-06-16 · Ela Kumuk

## Agenda

### Project Updates
- Rebuilt the pipeline from a clean, owned codebase instead of extending the
  inherited one. The prior version over-loaded M1 (Carry pillar, HMM regime,
  conviction sizing, ~100 experiment runs) — the "too much info on M1" pattern.
- Kept the correct infrastructure (data layer, no-look-ahead features, backtest);
  rebuilt M1/M2 to the agreed architecture: M1 simple/static/linear, M2 dynamic/
  regime-aware (logistic regression), portfolio benchmark-relative / info-ratio.
- End-to-end run is live (~35s, single canonical run).
- First clean result (note: FRED macro partly proxied locally — real-data rerun
  needed before judging M2):

  | Strategy | Ann.Ret | Sharpe | MaxDD | Info Ratio (OOS) |
  |---|---:|---:|---:|---:|
  | Equal-Weight | 6.0% | 0.45 | -43% | — |
  | M1-only | 6.6% | 0.68 | -26% | +0.44 |
  | M1+M2 | 6.1% | 0.63 | -29% | +0.13 |

  M1 already improves Sharpe and roughly halves drawdown vs equal-weight. M2 does
  not add value yet (shrinks tilts uniformly, not selectively).

### Asset Universe Review
- Current 7: SPY, TLT, GLD, VEA, VWO, HYG, VNQ.
- Gaps: no investment-grade credit, no broad commodity / inflation hedge.
- Candidate 9: add LQD (IG credit) + DBC (broad commodity).
- **Replace ETF with index data** (firm direction): the signal is computed on the
  index — a pure series, not a product. ETFs embed costs (expense ratio) and
  trading frictions (liquidity / bid-ask) that pollute attribution. Index data is
  also more available and has longer history (S&P 500 to 1957 vs SPY 1993), which
  fixes the 2007 start limit. The ETF can remain the traded instrument; costs are
  added back deliberately in two layers (expense ratio + transaction/liquidity).

### M1 Modeling Framework
- Static **linear** model: combine factor GROUPS by fixed ratios (not necessarily
  equal). No dynamics — kept deliberately simple to leave "valuation margin" for M2.
- Groups:
  - **Technical** — momentum + trend merged into one score (they are very close /
    collinear, so same dimension), cross-sectionally z-scored. (May revisit the
    internal momentum/trend split later.)
  - **Risk** — penalty from asset quality and volatility; down-weights high-vol /
    stressed names.
  - **Macro** — transparent, rule-based asset-class tilt (no learning). Detailed
    macro regime signal is reserved for M2.
- Benchmark-relative output: rank weekly, tilt up to ±10% around the 1/N
  benchmark. An underweight is a synthetic short → no outright shorting needed;
  shorting is not critical now (we mostly mirror the benchmark for info ratio).
- Show the mechanics for the assets: 4/12-week score → ranking → weight allocation,
  so each factor's behavior is visible.
- No look-ahead (shift(1)), macro lagged 4 weeks, 4-week embargo at train/test.

### M2 Modeling Framework
- Intentionally **dynamic**, regime-aware: takes M1 layers + market data and learns,
  per period, which M1 factor/signal to trust — this is where meta-labeling kicks in.
- **Logistic regression only** in the ML layer (keep the model simple); put the
  effort into **multiple evaluation methods** rather than a more complex model.
- **Macro is used to regime-date M2.**
- Updated on rolling ~12-month (weeks) windows.
- AI-augmentation, if used, scoped to a clearly defined layer / task.

### Next Step
1. Convert data from ETF to index (and lock 7 vs 9 instruments — no back-and-forth).
2. Rerun on real macro to fairly evaluate M2.
3. Add the **Risk** group to M1 (volatility / asset-quality penalty).
4. Attribution reporting — decompose P&L by factor / interaction / cost, to know
   the driver (factor itself vs interaction vs noise / lag).
5. M2: keep logistic regression, build **multiple evaluations** (F1, AUC-PR,
   calibration, info ratio uplift), then make sizing selective / regime-gated.
6. Walk-forward validation; automate the run if capacity allows.

### Questions
- Lock the universe at 7 or 9? Any preferred instruments? Heading toward many
  instruments later?
- Index data: Bloomberg access for total-return index series? Which index per sleeve?
- M1 factor-group ratios (Technical / Risk / Macro) — any prior, or start and tune?
- M2: confirm logistic-only + multiple-evaluation approach.
- Cost assumptions (expense ratio + ~5 bps transaction) reasonable for this universe?

---

## Summary / Key Decisions
*(to fill during the meeting)*
- Universe locked at:
- ETF → index data source / access:
- M1 factor groups + ratios:
- M2 approach confirmed:

## Next Steps
*(owners / dates to fill during the meeting)*
- [ ]
