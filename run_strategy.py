"""Full strategy: M1 (static linear) -> M2 (dynamic regime) -> benchmark-relative
portfolio -> backtest. Compares Equal-Weight vs M1-only vs M1+M2 on info ratio.

    python run_strategy.py
"""

from __future__ import annotations

import logging

import pandas as pd

from src import evaluation
from src.backtest import equal_weight_returns, metrics, portfolio_returns, static_portfolio_returns
from src.config import load_config
from src.data import IndexFileProvider, ingest_macro_data, ingest_market_data
from src.features import (
    get_vix_series, macro_asset_tilt, macro_wide, momentum_score, pivot_prices,
    regime_features, risk_score, technical_score, trend_score,
)
from src.m1 import M1Model
from src.m2 import M2Model, build_feature_matrix, build_meta_labels
from src.portfolio import apply_risk_layer, apply_vol_target, build_weights, cost_drag

logging.basicConfig(level=logging.ERROR, format="%(message)s")
pd.set_option("display.width", 140)


def main() -> None:
    cfg = load_config()
    # ETF vs INDEX: index data (Bloomberg export in data/raw/index/) when enabled,
    # else ETF via yfinance. Missing index files fall back to ETF automatically.
    provider = IndexFileProvider(cfg.raw_dir / "index") if cfg.data.use_index_signal else None
    market = ingest_market_data(cfg.data.universe, cfg.vix_ticker, cfg.data.data_start, None,
                                cfg.raw_dir, cfg.processed_dir, provider=provider,
                                use_cache=not cfg.data.use_index_signal)
    macro = ingest_macro_data(cfg.data.macro_series, cfg.data.data_start, None,
                              cfg.raw_dir, cfg.processed_dir, market_weekly=market)

    prices = pivot_prices(market[market["ticker"].isin(cfg.data.universe)])
    returns = prices.pct_change()
    regime = regime_features(macro_wide(macro), get_vix_series(market))

    # --- factors. M1 = PURE static technical (momentum+trend). Macro + vol/risk are
    #     DYNAMIC -> M2 features only (static -> M1, dynamic -> M2). ---
    mom = momentum_score(prices, cfg.m1.momentum_windows)
    trd = trend_score(prices, cfg.m1.trend_windows)
    mac = macro_asset_tilt(prices, regime)
    vol = risk_score(prices, cfg.risk_layer.vol_windows)

    m1 = M1Model(cfg.m1)
    score = m1.score({"technical": 0.5 * mom + 0.5 * trd})
    risk = vol   # only used if risk_layer.enabled (off by default)

    def sleeve(size=None):
        """M1 weights -> SEPARATE risk layer -> vol target."""
        w = build_weights(score, cfg, size=size)
        if cfg.risk_layer.enabled:
            w = apply_risk_layer(w, risk, cfg.risk_layer.strength)
        return apply_vol_target(w, returns, cfg)

    w_m1 = sleeve()
    benchmark_w = 1.0 / len(prices.columns)

    # --- M2: dynamic regime-aware meta-label ---
    m2 = M2Model(cfg)
    labels = build_meta_labels(prices, w_m1, cfg.labels.horizon_weeks, benchmark_w,
                               cfg.labels.positive_threshold)
    feats = build_feature_matrix({"momentum": mom, "trend": trd, "macro": mac, "vol": vol}, regime)
    proba = m2.run_rolling(feats, labels)
    train_ref = proba[proba.index.get_level_values("date") < cfg.split.test_start]
    size = m2.size(proba, train_ref)

    w_m2 = sleeve(size)

    # --- backtest ---
    ew = equal_weight_returns(returns)
    r_m1 = portfolio_returns(w_m1, returns, cost_drag(w_m1, cfg))
    r_m2 = portfolio_returns(w_m2, returns, cost_drag(w_m2, cfg))

    test = cfg.split.test_start
    series = [(f"base:{name}", static_portfolio_returns(w, returns)) for name, w in cfg.baselines.items()]
    series += [("M1-only", r_m1), ("M1+M2", r_m2)]
    rows = {}
    for label, r in series:
        full = metrics(r, ew)
        oos = metrics(r[r.index >= test], ew[ew.index >= test])
        rows[label] = {
            "ann_ret": full["ann_return"], "sharpe": full["sharpe"], "maxDD": full["max_drawdown"],
            "info_ratio": full.get("info_ratio", float("nan")),
            "IR_oos": oos.get("info_ratio", float("nan")), "sharpe_oos": oos["sharpe"],
        }
    table = pd.DataFrame(rows).T
    print(f"\n=== Full sample {prices.index[0].date()}–{prices.index[-1].date()} | OOS from {test} ===\n")
    print(table.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("\nM2 coverage:", f"{proba.notna().mean():.1%} of asset-weeks scored")

    # M2 multiple-evaluation suite (simple model -> many evaluations)
    evaluation.print_report(proba, labels, oos_start=test)


if __name__ == "__main__":
    main()
