# Asset Universe & Data Sources (index-based)

Per State Street direction: research on the **index** (pure series), exploit the ETF
only at the implementation stage. Seven indices spanning equity / rates / credit /
inflation / real assets. Internal ticker keys (left) map to these indices.

**Reality check (2026-06-18):** free *true-index* history is poor — the FRED HY
total-return index starts only **2023**, MSCI EAFE/EM on investing.com only **2012**.
So free index data is SHORTER than the ETFs, not longer. We therefore use the longest
free series per sleeve (ETF trackers where the free index is too short / unavailable),
and reserve true long-history index for **Bloomberg** at the implementation stage.
Switching to this data did NOT change conclusions vs the ETF run.

| Key | Asset class | Ideal index | What we actually fetch (free, auto) | History |
|---|---|---|---|---|
| SPY | U.S. Equity | S&P 500 | Yahoo `^GSPC` (the index) | 2000+ |
| VEA | Developed Intl Equity | MSCI EAFE | Yahoo `EFA` (ETF proxy) | 2001+ |
| VWO | Emerging Mkts Equity | MSCI EM | Yahoo `EEM` (ETF proxy) | 2003+ |
| TLT | U.S. Treasury 7–10Y | S&P UST 7–10Y (Bloomberg) | Yahoo `IEF` (ETF proxy) | 2002+ |
| HYG | High Yield Credit | ICE BofA US HY TR | Yahoo `HYG` (ETF proxy; FRED TR index only 2023+) | 2007+ |
| GLD | Gold | Gold spot | Yahoo `GC=F` (futures) | 2000+ |
| VNQ | Real Estate | Nasdaq US REIT | FRED `NASDAQNQUSB351020` (the index) | 2011+ |

Binding constraint = VNQ REIT index (2011). All auto-fetched by `fetch_indices.py`.
To upgrade any sleeve to the true Bloomberg index, drop a `date,adj_close` CSV into
`data/raw/index/<KEY>.csv` (overwrites the proxy) — or use `examples/convert_investing.py`
for investing.com exports.

## How to load index data
1. Run `python fetch_indices.py` — pulls the auto-fetchable series (SPY, HYG, VNQ, GLD)
   into `data/raw/index/<KEY>.csv` (columns: `date,adj_close`).
2. For the **manual** ones (VEA MSCI EAFE, VWO MSCI EM, TLT Treasury 7–10Y), download
   from the source above and save as `data/raw/index/VEA.csv` etc. (same two columns).
3. Set `use_index_signal: true` in `config/config.yaml`. Missing files fall back to the
   ETF automatically (logged), so a partial index set still runs.

## ⚠️ History caveat
FRED `SP500` only goes back ~10 years. The other series go back further. If the panel
is restricted to the common date range, the backtest shrinks to ~10y. Options:
- Use Yahoo `^GSPC` (S&P price index, 1990s+) for S&P to keep long history, **or**
- Accept the ~10y common window for a fully index-consistent study.
The HY (`BAMLHYH0A0HYM2TRIV`, ~1998+) and REIT series cover more history.

## Baseline portfolios (for comparison) — see `config/config.yaml` → `baselines`
- **Equal Weight** — 1/7 each.
- **Moderate Growth** — Equity 50% / Fixed income 35% / Alternatives 15%.
- **Institutional Multi-Asset** — Equity 45% / Fixed income 35% / Alternatives 20%.
