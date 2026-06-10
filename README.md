# SSGA Summer 2026 — AI-Augmented Multi-Asset Meta-Labeling Pipeline

Research-grade Python pipeline for a multi-asset portfolio strategy using meta-labeling (M1 side decision + M2 profitability filter + position sizing). **For education and research only — not investment advice.**

## Asset Universe

SPY, TLT, GLD, VEA, VWO, HYG, VNQ (weekly, 7-asset global ETF basket).

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run full pipeline (downloads yfinance + FRED data)
# Runs M1 twice: long_only (no shorts) and long_short (shorts enabled)
python -m src.run_pipeline --config config/config.yaml

# Override train/test split dates (ISO format; defaults: train through 2020, test from 2021)
python -m src.run_pipeline --train-end 2018-12-31 --test-start 2019-01-01
python -m src.run_pipeline --train-start 2008-01-01 --train-end 2015-12-31 --test-start 2016-01-01

# Train before all 7 ETFs existed (partial universe; re-download if cache is too short)
python -m src.run_pipeline --data-start 2004-01-01 --train-start 2005-01-01 --train-end 2006-12-31 \
  --test-start 2007-01-01 --partial-universe --refresh-data

# Run tests (synthetic fixtures, no network)
pytest -q

# Integration test (requires network)
pytest -q -m integration
```

## Project Layout

```text
config/config.yaml     # Pipeline configuration
src/                   # Core modules
tests/                 # Unit tests
notebooks/             # Exploratory notebooks
data/                  # Raw, processed, features, backtests
runs/                  # Timestamped pipeline runs
reports/               # final_report.md + final/, mode_comparison/, assets/
docs/                  # Specifications (copied from llm/docs)
llm/                   # LLM instruction bundle
```

## Pipeline Stages

1. Data ingest (yfinance + FRED)
2. Validation and balanced panel
3. Feature engineering (no look-ahead)
4. M1 rule-based side model
5. M2 meta-labeling model
6. Position sizing (binary, linear, ECDF)
7. Backtest vs equal-weight and 60/40 benchmarks
8. Diagnostics and final report

## Design Rules

- No time-series shuffling
- Features use only data available at or before signal time
- Train/test split configurable in `config/config.yaml` or via `--data-start`, `--train-start`, `--train-end`, `--test-start`, `--test-end`
- `train_start` may be before 2006, but the default **full 7-asset** panel starts ~2007 when VEA/HYG exist; use `--partial-universe` for earlier subsets
- Default split: train 2006–2020; test 2021–latest
- LLM features disabled by default

See [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) and [docs/ACCEPTANCE_TESTS.md](docs/ACCEPTANCE_TESTS.md) for full specifications.

## Reports

- [`reports/final_report.md`](reports/final_report.md) — strategy comparison (long-only vs long/short M1)
- [`reports/assets/asset_component_analysis.md`](reports/assets/asset_component_analysis.md) — per-asset buy-and-hold (SPY/S&P 500, bonds, gold, etc.) and data source documentation

## Grid Search (40 runs)

Sweep train-end dates, M2 threshold, and transaction costs. Results are ranked by **test-set** Sharpe (M1+M2 Linear, long-only); full-sample metrics are stored for reference.

```bash
# Preview 40 combinations (no execution)
python scripts/grid_search.py --dry-run

# Full sweep using cached data (~25–35 min locally)
python scripts/grid_search.py

# Smoke test (2 runs)
python scripts/grid_search.py --max-runs 2

# Force fresh downloads
python scripts/grid_search.py --refresh-data

# Resume an interrupted sweep
python scripts/grid_search.py --resume --sweep-dir runs/grid_search/<sweep_id>
```

Outputs land in `runs/grid_search/<sweep_id>/`:

- `results.csv` / `results.jsonl` — master comparison table
- `summary.md` — top 10 by test-set Sharpe
- `run_NNN/` — per-run config, metrics, and backtest snapshots

Edit [`scripts/grid_search_spec.yaml`](scripts/grid_search_spec.yaml) to change the parameter grid.
