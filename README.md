# SSGA Multi-Asset Meta-Labeling — Clean Rebuild

**Owner:** Ela Kumuk · **Field project:** State Street Global Advisors, Summer 2026
**Codebase advisor (original pipeline):** Vitaly · **Research direction:** State Street mentor
**Status:** clean rebuild started 2026-06-15. Research/education only — not investment advice.

---

## Why this repo exists

This is a deliberate, owned rebuild of the meta-labeling pipeline. The original
codebase (Vitaly's `SSGA-Summer2026-vitaly_week2`) over-loaded the M1 layer
(Carry pillar + HMM regime + conviction sizing + 6 variants + 98 runs), which is
exactly the mistake State Street flagged ("last semester focused too much info on
M1"). We salvage the correct, boring infrastructure and rebuild the M1/M2 factor
architecture to match State Street's research philosophy.

## Core architecture (State Street directive, 2026-06-15)

> **Be solid on each step. Don't chase performance at the beginning. Stay in the
> right direction.**

**Organizing principle: STATIC factors → M1, DYNAMIC factors → M2.**

| Layer | Character | Contents | Job |
|---|---|---|---|
| **M1** | **STATIC, linear** | `technical` only = momentum + trend merged (the static price signal) | "Which side?" — a lean directional signal |
| **M2** | **DYNAMIC, regime-aware** | sees momentum, trend, **macro**, **vol/risk** SEPARATELY + regime; logistic regression, rolling ~12-month refit | "In *this* regime, which signal do I trust + how much risk?" — meta-labeling |
| **Portfolio** | benchmark-relative | active weights = benchmark ± bounded tilt | risk budget + costs |

Macro and volatility/risk are dynamic → they live in M2 (as features), not M1.
Transaction-cost thresholds are likewise a separate, later layer — not built now.

Key principles:
- **M1 stays lean** so there is "valuation margin" (headroom) to layer M2 on top.
- **Momentum & trend are collinear** → treated as ONE "technical" group, not two weights.
- **Macro → M2 regime-dating**, not an M1 factor.
- **Dynamic stuff → M2, static/directional stuff → M1.**
- **All intelligence/time-variation lives in M2.** This is richer than textbook
  binary-filter meta-labeling — it is dynamic, regime-conditional factor-timing.

## Research philosophy (why index, not ETF)

- **Research on the INDEX** (pure signal, no expense ratio / tracking error /
  liquidity), trade instrument can be ETF. Index data also has far longer history
  (S&P 500 → 1957 vs SPY → 1993), fixing the 2007 sample-start limit.
- **Cost in two layers:** (1) expense ratio (ETF-embedded), (2) transaction/
  liquidity (bid-ask + impact on trade).
- **Goal = ATTRIBUTION.** For any factor / factor-combo / cost, isolate the
  driver: the factor itself, factor interaction, noise, or lag.
- **Shorting not critical yet.** Benchmark-relative active weights (e.g. 33.33% ±
  ~10% tilts) mean an underweight is already a synthetic short. Target metric =
  **information ratio**. What matters is having many instruments, not long vs short.

## Instrument universe — LOCKED at 7 INDICES

S&P 500 · MSCI EAFE · MSCI EM · S&P US Treasury 7–10Y · ICE BofA US HY · Gold ·
Nasdaq US REIT. Research on the index; ETF only at implementation. Sources +
auto-fetch in **`DATA_SOURCES.md`** / `fetch_indices.py`. Three baseline portfolios
(Equal-Weight, Moderate Growth, Institutional) in `config/config.yaml`.

## Build order

1. ✅ Foundation: clean repo, README, salvage map, config
2. ✅ Port salvaged infra (data, backtest, validation, no-lookahead feature utils)
3. ✅ Build **M1** — simple static linear (technical + macro, fixed ratios). `run_m1.py`
4. ✅ Build **M2** — dynamic regime-aware (macro-dated), rolling refit + embargo. `src/m2.py`
5. ✅ Benchmark-relative portfolio + 2-layer costs + information ratio. `run_strategy.py`
5b. ✅ M1 **Risk** group (vol / asset-quality penalty); ETF→**index** data path
    (`IndexFileProvider`, Bloomberg-ready); M2 **multiple-evaluation** suite
    (`src/evaluation.py`: F1, AUC-ROC, AUC-PR, calibration).
6. ✅ Attribution reporting (per-factor / interaction / per-cost). `run_attribution.py`
7. ✅ Walk-forward validation across chronological windows. `run_walkforward.py`
8. ⬜ (Bonus) automation / AI-augmentation — explicit about which layer, which job

### Clean result — M1 = technical+macro, Risk as separate layer (FRED partly proxied in dev env — needs real macro to judge M2)
| Strategy | Sharpe | MaxDD | Sharpe (OOS) |
|---|---:|---:|---:|
| base: Equal-Weight | 0.58 | -39% | 0.75 |
| base: Moderate Growth | 0.56 | -40% | 0.65 |
| base: Institutional | 0.62 | -35% | 0.62 |
| **M1-only** | **0.66** | **-25%** | **0.87** |
| M1+M2 | 0.61 | -28% | 0.81 |

M1 (linear technical+macro + separate Risk layer) beats all three baselines on
Sharpe, drawdown, and OOS Sharpe. Attribution: the Risk layer lifts Sharpe
0.607 → 0.659; technical & macro reinforce (interaction > 0).
M2 adds no value yet — classifier weak (OOS AUC-ROC ~0.46, calibration flat). Make
M2 selective and re-evaluate with real macro. Eval suite: `src/evaluation.py`.

See `SALVAGE_MAP.md` for what is ported from the original vs rebuilt.
