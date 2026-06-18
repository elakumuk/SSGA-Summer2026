"""Attribution — the central research goal: know the DRIVER.

Decomposes performance so we can tell whether return comes from a single factor,
from factor INTERACTION, or from costs:
  * each factor group ALONE (technical / risk / macro) as a benchmark-relative sleeve
  * the COMBINED M1
  * INTERACTION = combined active return - sum(standalone active returns)
  * COST layers  = gross vs net-of-expense vs net-of-transaction

    python run_attribution.py
"""

from __future__ import annotations

import logging

import pandas as pd

from src.backtest import equal_weight_returns, metrics, portfolio_returns
from src.config import load_config
from src.data import IndexFileProvider, ingest_macro_data, ingest_market_data
from src.features import (
    get_vix_series, macro_wide, momentum_score, pivot_prices, regime_features, trend_score,
)
from src.m1 import M1Model
from src.portfolio import apply_vol_target, build_weights, cost_drag

logging.basicConfig(level=logging.ERROR, format="%(message)s")
pd.set_option("display.width", 140)


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
    m1 = M1Model(cfg.m1)

    # M1 is PURE static technical -> attribute its two sub-signals: momentum vs trend.
    mom = momentum_score(prices, cfg.m1.momentum_windows)
    trd = trend_score(prices, cfg.m1.trend_windows)
    subfactors = {"momentum": mom, "trend": trd}

    def sleeve_return(score: pd.DataFrame) -> pd.Series:
        w = apply_vol_target(build_weights(score, cfg), returns, cfg)
        return portfolio_returns(w, returns, cost_drag(w, cfg))

    # --- 1. M1 sub-factor isolation (momentum vs trend) ---
    print("\n=== M1 ATTRIBUTION (momentum vs trend; macro & risk are now in M2) ===\n")
    standalone = {}
    rows = {}
    for name, frame in subfactors.items():
        r = sleeve_return(m1.score({"technical": frame}))
        standalone[name] = r
        m = metrics(r, ew)
        rows[name] = {"ann_ret": m["ann_return"], "sharpe": m["sharpe"],
                      "info_ratio": m["info_ratio"], "excess_ret": m["excess_return"]}
    technical = 0.5 * mom + 0.5 * trd
    combined_r = sleeve_return(m1.score({"technical": technical}))
    mc = metrics(combined_r, ew)
    rows["TECHNICAL (M1)"] = {"ann_ret": mc["ann_return"], "sharpe": mc["sharpe"],
                              "info_ratio": mc["info_ratio"], "excess_ret": mc["excess_return"]}
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:,.3f}"))

    # --- 2. interaction ---
    sum_standalone_excess = sum(metrics(r, ew)["excess_return"] for r in standalone.values())
    interaction = mc["excess_return"] - sum_standalone_excess
    print(f"\nINTERACTION (technical excess - sum of momentum+trend excess): {interaction:+.4f}")
    print("  >0 => momentum & trend reinforce; <0 => they partly cancel / overlap")

    # --- 3. cost layers ---
    w = build_weights(m1.score({"technical": technical}), cfg)
    w = apply_vol_target(w, returns, cfg)
    gross = portfolio_returns(w, returns, None)
    expense_only = portfolio_returns(w, returns, _expense(w, cfg))
    net = portfolio_returns(w, returns, cost_drag(w, cfg))
    print("\n=== COST LAYER ATTRIBUTION (annualized return) ===")
    print(f"  gross                : {metrics(gross, ew)['ann_return']:.4f}")
    print(f"  net of expense ratio : {metrics(expense_only, ew)['ann_return']:.4f}")
    print(f"  net of all costs     : {metrics(net, ew)['ann_return']:.4f}")


def _expense(w, cfg):
    weekly = (cfg.costs.expense_ratio_bps_annual / 1e4) / 52.0
    return w.abs().sum(axis=1) * weekly


if __name__ == "__main__":
    main()
