# ADR 0001 — Composite Bias Score formula & weights

- **Status:** Accepted
- **Date:** 2026-05

## Context
The STP specifies a single Composite Bias Score aggregating Disparate Impact (DI), Statistical
Parity Difference (SPD) and Equalized Odds (EO) with weights 0.40 / 0.35 / 0.25, and pins
`compute_composite_bias_score(0.6, -0.25, 0.15) == 0.7150`. Its original formula was undefined for
DI > 1 (a *favoured* group), where a naive linear term would reward over-selection.

## Decision
Normalise each input to a 0–1 fairness-*goodness* sub-score before weighting:

```
g_DI = min(DI, 1/DI)   g_SPD = 1 - min(|SPD|, 1)   g_EO = 1 - min(|EO|, 1)
Composite = clamp(0.40·g_DI + 0.35·g_SPD + 0.25·g_EO, 0, 1)
```

`min(DI, 1/DI)` makes the DI term **symmetric** about parity, so a group favoured by 25% scores
the same as one disadvantaged by the mirror amount. The weights remain configurable; the defaults
preserve the STP's pinned golden value (verified in `test_metrics_known_values.py`).

## Consequences
- Higher composite = fairer (documented prominently to avoid the "bias score" naming trap).
- The score is explicitly a **triage convenience, not a legal standard**; raw metrics + CIs are
  always reported alongside it (see `LIMITATIONS.md`).
- Changing the weights changes the band a model lands in — a product decision, recorded here.
