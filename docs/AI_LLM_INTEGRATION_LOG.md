# AI / LLM Integration Plan and Research Log Template

## Purpose

The project should use AI and LLMs deliberately in two ways:

1. **Productivity support:** coding, debugging, data validation, documentation, research planning.
2. **Research features:** timestamped text-derived macro/regime/sentiment features used by M1 or M2.

Every AI use should be documented so the final project can explain what was automated, what was human-reviewed, and what risks remain.

---

## LLM Usage by Pipeline Stage

| Stage | Allowed AI Use | Required Human Check |
|---|---|---|
| Literature review | summarize papers, extract methodology | verify against source paper |
| Data ingest | generate API wrapper code | inspect schema and missing values |
| Cleaning | suggest validation checks | run tests and review data report |
| Feature engineering | brainstorm features | verify no look-ahead leakage |
| M1 | suggest model alternatives | compare baseline and explain choice |
| M2 | suggest meta-label features | ensure labels are correctly aligned |
| Position sizing | implement sizing methods | verify risk constraints |
| Backtest | generate metrics code | verify timing and transaction costs |
| Diagnostics | draft charts and report | verify interpretations |
| LLM features | extract structured narratives | timestamp, cache, audit, compare non-LLM baseline |

---

## LLM-Derived Feature Rules

LLM-derived features are optional and disabled by default. If implemented, they must satisfy the following rules:

1. Text source must have a timestamp.
2. Source timestamp must be less than or equal to the prediction date.
3. Prompt must instruct the model not to infer from future events.
4. Output must be structured JSON.
5. Output must be cached and versioned.
6. Prompt, model, temperature, source ID, and output must be saved.
7. Strategy must be tested both with and without LLM features.

---

## LLM Feature Prompt Template

```text
You are extracting structured macro-regime features for a quantitative backtest.

IMPORTANT: Use only the text below. Do not use any knowledge after the document date.

Document date: {document_date}
Prediction date: {prediction_date}
Source ID: {source_id}

Return JSON only with these fields:
{
  "risk_sentiment": float from -1 to 1,
  "inflation_pressure": float from -1 to 1,
  "growth_slowdown": float from -1 to 1,
  "policy_tightness": float from -1 to 1,
  "credit_stress": float from -1 to 1,
  "macro_uncertainty": float from 0 to 1,
  "dominant_narrative": one of ["inflation", "growth", "policy", "credit", "geopolitical", "liquidity", "mixed", "other"],
  "confidence": float from 0 to 1,
  "short_rationale": string with max 30 words
}

Text:
{historical_text}
```

---

## Research Log Entry Template

Use one YAML block per AI-assisted decision.

```yaml
- date: YYYY-MM-DD
  stage: feature_engineering
  llm_used: true
  tool_or_model: GPT/Claude/Copilot/etc.
  prompt_summary: "Asked for ideas for false-positive features for M2."
  output_used: "Added cross-asset dispersion and rolling M1 hit rate."
  human_decision: "Kept only features that can be computed from past data."
  risk_or_limitation: "Feature may overlap with volatility; will check correlation."
  verification: "Unit test confirms feature is shifted by one week."
```

---

## Weekly Research Log Template

```markdown
# Weekly Research Log: Week of YYYY-MM-DD

## What We Built

- 

## Data Issues Found

- 

## Modeling Decisions

- 

## AI/LLM Assistance Used

- Stage:
- Prompt summary:
- Output used:
- Human review:
- Limitation:

## Backtest Observations

- 

## Risks / Open Questions

- 

## Next Steps

- 
```

---

## Final AI Documentation Section

The final report should include:

1. Where AI/LLMs were used.
2. Where AI/LLMs were not used.
3. Which features were LLM-derived.
4. Leakage controls.
5. Comparison of strategy with and without LLM-derived features.
6. Known risks: hallucination, unstable outputs, hidden knowledge leakage, prompt sensitivity.
