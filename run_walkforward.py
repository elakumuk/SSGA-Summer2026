"""Walk-forward validation across chronological windows.

M2's rolling refit is walk-forward BY CONSTRUCTION (at each week it trains only on
resolved-and-embargoed past labels), so evaluating consecutive windows is honest
walk-forward. For each window we use an ECDF sizing reference taken strictly from
BEFORE that window. Reports Equal-Weight / M1-only / M1+M2 per window.

    python run_walkforward.py
"""

from __future__ import annotations

import logging

import pandas as pd

from src.backtest import equal_weight_returns, metrics, portfolio_returns
from src.config import load_config
from src.data import IndexFileProvider, ingest_macro_data, ingest_market_data
from src.features import (
    get_vix_series, macro_asset_tilt, macro_wide, momentum_score, pivot_prices,
    regime_features, risk_score, trend_score,
)
from src.m1 import M1Model
from src.m2 import M2Model, build_feature_matrix, build_meta_labels
from src.portfolio import apply_risk_layer, apply_vol_target, build_weights, cost_drag

logging.basicConfig(level=logging.ERROR, format="%(message)s")
pd.set_option("display.width", 160)

WINDOWS = [
    ("W1_2014_2016", "2014-01-01", "2016-12-31"),
    ("W2_2016_2018", "2016-01-01", "2018-12-31"),
    ("W3_2018_2020", "2018-01-01", "2020-12-31"),
    ("W4_2021_now", "2021-01-01", None),
]


def main() -> None:
    cfg = load_config()
    provider = IndexFileProvider(cfg.raw_dir / "index") if cfg.data.use_index_signal else None
    market = ingest_market_data(cfg.data.universe, cfg.vix_ticker, cfg.data.data_start, None,
                                cfg.raw_dir, cfg.processed_dir, provider=provider,
                                use_cache=not cfg.data.use_index_signal)
    macro = ingest_macro_data(cfg.data.macro_series, cfg.data.data_start, None,
                              cfg.raw_dir, cfg.processed_dir, market_weekly=market)

    prices = pivot_prices(market[market["ticker"].isin(cfg.data.universe)])
    returns = prices.pct_change()
    regime = regime_features(macro_wide(macro), get_vix_series(market))
    ew = equal_weight_returns(returns)
    benchmark_w = 1.0 / len(prices.columns)

    mom = momentum_score(prices, cfg.m1.momentum_windows)
    trd = trend_score(prices, cfg.m1.trend_windows)
    mac = macro_asset_tilt(prices, regime)
    vol = risk_score(prices, cfg.risk_layer.vol_windows)

    m1 = M1Model(cfg.m1)
    score = m1.score({"technical": 0.5 * mom + 0.5 * trd})
    risk = vol

    def sleeve(size=None):
        w = build_weights(score, cfg, size=size)
        if cfg.risk_layer.enabled:
            w = apply_risk_layer(w, risk, cfg.risk_layer.strength)
        return apply_vol_target(w, returns, cfg)

    w_m1 = sleeve()
    r_m1 = portfolio_returns(w_m1, returns, cost_drag(w_m1, cfg))

    # M2 proba once (walk-forward by construction)
    m2 = M2Model(cfg)
    labels = build_meta_labels(prices, w_m1, cfg.labels.horizon_weeks, benchmark_w,
                               cfg.labels.positive_threshold)
    proba = m2.run_rolling(build_feature_matrix({"momentum": mom, "trend": trd, "macro": mac, "vol": vol}, regime), labels)

    rows = []
    for name, start, end in WINDOWS:
        ref = proba[proba.index.get_level_values("date") < start]   # ECDF ref from BEFORE window
        size = m2.size(proba, ref)
        w_m2 = sleeve(size)
        r_m2 = portfolio_returns(w_m2, returns, cost_drag(w_m2, cfg))

        def slice_(r):
            s = r[r.index >= start]
            return s[s.index <= end] if end else s

        for strat, r in [("EqualWeight", ew), ("M1_only", r_m1), ("M1_M2", r_m2)]:
            m = metrics(slice_(r), slice_(ew))
            rows.append({"window": name, "strategy": strat,
                         "ann_ret": m["ann_return"], "sharpe": m["sharpe"],
                         "maxDD": m["max_drawdown"], "info_ratio": m.get("info_ratio", float("nan"))})

    out = pd.DataFrame(rows)
    table = out.pivot(index="window", columns="strategy", values=["sharpe", "info_ratio"])
    print("\n=== WALK-FORWARD: Sharpe & Info Ratio by window ===\n")
    print(table.to_string(float_format=lambda v: f"{v:,.2f}"))
    path = cfg.root / "reports" / "walk_forward.csv"
    out.to_csv(path, index=False)
    print(f"\nfull table -> {path}")


if __name__ == "__main__":
    main()
