# Asset Universe & Data Sources (index-based)

Per State Street direction: research on the **index** (pure series), exploit the ETF
only at the implementation stage. Seven indices spanning equity / rates / credit /
inflation / real assets. Internal ticker keys (left) map to these indices.

| Key | Asset class | Index | Source | Auto-fetch |
|---|---|---|---|---|
| SPY | U.S. Equity | S&P 500 | FRED `SP500` (≈10y) — or Yahoo `^GSPC` for long history | yes |
| VEA | Developed Intl Equity | MSCI EAFE | investing.com (download CSV) | **manual** |
| VWO | Emerging Mkts Equity | MSCI Emerging Markets | investing.com (download CSV) | **manual** |
| TLT | U.S. Treasury | S&P U.S. Treasury Bond 7–10Y | Bloomberg (recommended) | **manual** |
| HYG | High Yield Credit | ICE BofA U.S. HY total-return index | FRED `BAMLHYH0A0HYM2TRIV` | yes |
| GLD | Gold | Gold price | Yahoo `GC=F` (or LBMA) | yes |
| VNQ | Real Estate | Nasdaq U.S. Benchmark REIT | FRED `NASDAQNQUSB351020` | yes |

## How to load index data
1. Run `python fetch_indices.py` — pulls the auto-fetchable series (SPY, HYG, VNQ, GLD)
   into `data/raw/index/<KEY>.csv` (columns: `date,adj_close`).
2. For the **manual** ones (VEA MSCI EAFE, VWO MSCI EM, TLT Treasury 7–10Y), download
   from the source above and save as `data/raw/index/VEA.csv` etc. (same two columns).
3. Set `use_index_signal: true` in `config/config.yaml`. Missing files fall back to the
   ETF automatically (logged), so a partial index set still runs.

## ⚠️ History caveat (decide with the mentor)
FRED `SP500` only goes back ~10 years. The other series go back further. If the panel
is restricted to the common date range, the backtest shrinks to ~10y. Options:
- Use Yahoo `^GSPC` (S&P price index, 1990s+) for S&P to keep long history, **or**
- Accept the ~10y common window for a fully index-consistent study.
The HY (`BAMLHYH0A0HYM2TRIV`, ~1998+) and REIT series cover more history.

## Baseline portfolios (for comparison) — see `config/config.yaml` → `baselines`
- **Equal Weight** — 1/7 each.
- **Moderate Growth** — Equity 50% / Fixed income 35% / Alternatives 15%.
- **Institutional Multi-Asset** — Equity 45% / Fixed income 35% / Alternatives 20%.
