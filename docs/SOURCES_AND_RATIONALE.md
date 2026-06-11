# Sources and Rationale Notes

## Attached Source: Meta-Labeling Theory and Framework

Key ideas to carry into the project:

- Meta-labeling is a machine-learning layer on top of a primary strategy.
- The primary model focuses on trade side.
- The secondary model predicts whether the primary signal will be profitable.
- M2 probabilities can be used to size positions.
- Useful M2 feature groups include information advantage, false-positive indicators, regime features, and position-sizing logic.
- The secondary model filters poor M1 signals; it does not create new independent signals.

## Attached Source: The A-Z of Quant

Key ideas to carry into the project:

- Factor signals should have meaning, significance, and stability.
- Factor testing can include information coefficients, IC decay, fractiles, and pure factor returns.
- Data pitfalls include matching across data sources, survivorship bias, look-ahead bias, and corporate-action adjustment issues.
- Portfolio construction should include realistic constraints such as active bet limits, risk exposure, liquidity, and transaction costs.

## Public Source: Hudson & Thames Meta-Labeling Repository

Use as a conceptual reference for meta-labeling architecture and position sizing examples:

```text
https://github.com/hudson-and-thames/meta-labeling
```

## Public Source: yfinance

Use as open-source market data fallback:

```text
https://ranaroussi.github.io/yfinance/
```

## Public Source: scikit-learn TimeSeriesSplit

Use as reference for time-ordered cross-validation:

```text
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
```

## Public Source: pandas-datareader FRED

Use as open-source macroeconomic data fallback:

```text
https://pandas-datareader.readthedocs.io/en/latest/readers/fred.html
```

## Design Choice Summary

| Choice | Rationale | Alternative |
|---|---|---|
| Weekly data | Larger sample than monthly while reducing daily noise | Daily data with higher turnover controls |
| 7 ETF universe | Covers broad global risk premia with manageable complexity | Add commodities, currencies, or more fixed income |
| M1 high-recall model | Meta-labeling benefits from candidate signals that M2 can filter | Stricter M1 with fewer signals |
| M2 probability output | Needed for probability-based sizing | Binary-only filter |
| ECDF sizing | Maps model confidence into relative historical confidence | Linear or sigmoid sizing |
| Time-series split | Prevents future leakage | Random split is not acceptable |
| LLM features optional | Reduces complexity and leakage risk in MVP | Full news/sentiment pipeline |
