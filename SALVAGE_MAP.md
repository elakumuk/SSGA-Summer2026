# Salvage Map — what we port from Vitaly's pipeline vs rebuild

Original repo: `~/Downloads/SSGA-Summer2026-vitaly_week2/`

| Original file | Lines | Decision | Reason / what to do |
|---|---:|---|---|
| `src/data_providers.py` | 326 | ✅ **PORT** | yfinance + FRED work; has a `BloombergProvider` placeholder — index/institutional data plugs in here |
| `src/backtest.py` | 155 | ✅ **PORT** | clean: equal-weight, 60/40, `run_all_strategies` |
| `src/data_validation.py` | 155 | ✅ **PORT** | no-lookahead / data-integrity checks |
| `src/feature_engineering.py` | 357 | 🔧 **PARTIAL** | keep no-lookahead utils + momentum/trend/vol calcs; **rebuild factor grouping** (technical = mom+trend, risk overlay, macro→M2) |
| `src/research_logger.py` | 72 | ✅ **PORT** | lightweight run logging |
| `src/labels.py` | 65 | 🔧 **PARTIAL** | keep fixed-horizon forward-return label; revisit meta-label for dynamic M2 |
| `src/portfolio.py` | 119 | 🔧 **REWRITE** | move to **benchmark-relative active weights** (benchmark ± tilt), info-ratio oriented |
| `src/position_sizing.py` | 62 | 🔧 **PARTIAL** | ECDF/linear sizing useful; reframe under M2 |
| `src/model_m1.py` | 631 | ❌ **REBUILD** | over-loaded (Carry/HMM/conviction/top-k). New M1 = simple static linear |
| `src/model_m2.py` | 114 | ❌ **REBUILD** | new M2 = dynamic, regime-aware (macro-dated), rolling refit |
| `src/diagnostics.py` | 2065 | 🔧 **SELECTIVE** | extract only attribution / decomposition reporting; drop the rest |
| `src/config.py` | 309 | 🔧 **REWRITE** | minimal clean config matching new architecture |
| `src/asset_analysis.py` | 546 | ⏸️ **HOLD** | revisit later if needed |
| `src/conformal.py` | 195 | ⏸️ **HOLD** | not in current scope |
| `src/llm_features.py` | 130 | ⏸️ **HOLD** | AI-augmentation — only after core is solid, and only at a clearly-defined layer |
| `src/grid_search.py` | 453 | ⏸️ **HOLD** | State Street: don't spend time on threshold/param tweaking now |
| `scripts/walk_forward.py` | — | ✅ **PORT (later)** | reuse the walk-forward harness once M1/M2 are stable |

**Dropped entirely:** the 98 timestamped `runs/`, the 6 `variant_*` dirs, the
inflated marketing docs. Keep one canonical run + clean reports.
