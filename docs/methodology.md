# Methodology

This document defines every metric EthicLens computes, the statistical machinery around it, and —
most importantly — the **methodological correction** the project makes over its original
specification.

## The flaw we fixed

The original System Test Plan ([`original-stp.pdf`](original-stp.pdf)) proposed to audit a model
by feeding it **synthetic demographic personas** and computing fairness metrics on the resulting
predictions. This cannot work:

- Synthetic personas have **no ground-truth label `Y`**. Every error-based metric the spec names
  (Equalized Odds, Equal Opportunity, predictive parity, calibration) is therefore
  **mathematically uncomputable** on that data.
- Because the demographic distribution is authored by the tester (`config`), the measured
  "selection rate" reflects **sampling choices and out-of-distribution model behaviour**, not
  real-world disparate impact.

### The fix

1. **Audit on real, labelled benchmarks** with known protected attributes and true outcomes —
   Folktables/ACSIncome (primary), COMPAS, German Credit.
2. **Repurpose the Synthetic Profile Generator** into two legitimate roles:
   - **Counterfactual probing** — flip *only* the protected attribute on *real* records and
     measure the change in prediction (an individual-fairness signal; Dwork 2012, Kusner 2017).
   - **A bias-injection oracle** — manufacture data with a *known* disparate impact to verify that
     the engine *recovers* it (the golden-reference test).
3. **Report uncertainty** — bootstrap CIs, significance tests, minimum-subgroup floors — and flag
   a group only when its CI excludes 0.80.

## Notation

Privileged group `a = 1`, unprivileged `a = 0`. Prediction `Ŷ`, ground truth `Y`,
selection rate `r_a = P(Ŷ = 1 | A = a)`.

## Group metrics

| Metric | Definition | Needs `Y`? | Library check |
|---|---|:---:|---|
| **Disparate Impact** | `r_0 / r_1` | no | `fairlearn.demographic_parity_ratio` |
| **Statistical Parity Difference** | `r_0 − r_1` | no | `fairlearn.demographic_parity_difference` |
| **Equalized Odds (difference)** | `max(\|TPR_1−TPR_0\|, \|FPR_1−FPR_0\|)` | **yes** | `fairlearn.equalized_odds_difference` |
| **Equal Opportunity** | `\|TPR_1 − TPR_0\|` | **yes** | derived from `MetricFrame` |
| **Predictive Parity** | `\|PPV_1 − PPV_0\|` | **yes** | derived from `MetricFrame` |
| **FPR balance** | `\|FPR_1 − FPR_0\|` | **yes** | derived from `MetricFrame` |
| **Calibration / ECE** | binned `\|accuracy − confidence\|` | **yes** | from scratch |
| **Theil index** | generalised entropy of the benefit vector | yes | from scratch |

Every from-scratch implementation is asserted **equal to Fairlearn within 1e-9** in
[`test_vs_fairlearn.py`](../packages/fairness-core/tests/test_vs_fairlearn.py). That equality is the
project's correctness proof.

### Disparate Impact and the 4/5ths rule
`calculate_disparate_impact(privileged_rate, unprivileged_rate)` returns `r_0 / r_1`, raising
`ValueError` when `r_1 == 0`. A value below **0.80** is evidence of adverse impact (EEOC Uniform
Guidelines on Employee Selection Procedures, 1978).

### The impossibility theorem (why we surface PPV *and* FPR)
When base rates differ between groups, predictive parity and equalized odds **cannot both hold**
(Chouldechova 2017; Kleinberg, Mullainathan & Raghavan 2016). EthicLens shows PPV and FPR side by
side on COMPAS precisely so this trade-off is visible rather than hidden behind a single metric.

## Composite Bias Score

A 0–1 fairness-*goodness* score (higher = fairer). Each component is normalised to a sub-score,
then weighted:

```
g_DI  = min(DI, 1/DI)          # symmetric; 1.0 at parity; clamps the favoured-flip case
g_SPD = 1 − min(|SPD|, 1)
g_EO  = 1 − min(|EO|, 1)
Composite = clamp(0.40·g_DI + 0.35·g_SPD + 0.25·g_EO, 0, 1)
```

Bands: `< 0.60` High Risk, `< 0.80` Medium Risk, `≥ 0.80` Low Risk. Worked example (the value
pinned in `TS-UNIT-004`): `compute_composite_bias_score(0.6, −0.25, 0.15) = 0.7150`. The composite
is a **triage convenience, not a legal standard** — see [`../LIMITATIONS.md`](../LIMITATIONS.md).
The weighting rationale is recorded in [`adr/0001-composite-weights.md`](adr/0001-composite-weights.md).

## Statistical rigour

- **Bootstrap confidence intervals** — BCa (bias-corrected & accelerated) by default, percentile
  on request. Acceleration uses a jackknife, skipped for very large subgroups (falling back to
  bias-corrected percentile) and reported in the interval's `method` field. No SciPy dependency:
  the normal CDF / inverse-CDF come from `statistics.NormalDist`.
- **Significance** — a pooled two-proportion z-test on the selection-rate gap.
- **Minimum subgroups** — ≥ 100 for stable rates; ≥ 30 positives **and** ≥ 30 negatives for error
  metrics. Below the floor → `INSUFFICIENT_DATA`.
- **Flagging policy** — a group is flagged when its DI **confidence interval lies entirely below
  0.80**, *or* when its **Equalized-Odds CI lower bound exceeds 0.10**. This catches both
  selection-rate bias (hiring/lending) *and* error-rate bias (e.g. COMPAS, where the DI rule alone
  is blind), while subgroup noise alone never flips the verdict.

## References

- Ammann, P., & Offutt, J. (2016). *Introduction to Software Testing* (2nd ed.).
- Barocas, Hardt, & Narayanan (2023). *Fairness and Machine Learning*. fairmlbook.org.
- Chouldechova, A. (2017). Fair prediction with disparate impact. *Big Data*, 5(2).
- Ding, Hardt, Miller, & Schmidt (2021). Retiring Adult: New Datasets for Fair ML. *NeurIPS*.
- Dwork, Hardt, Pitassi, Reingold, & Zemel (2012). Fairness Through Awareness. *ITCS*.
- Hardt, Price, & Srebro (2016). Equality of Opportunity in Supervised Learning. *NeurIPS*.
- Kleinberg, Mullainathan, & Raghavan (2016). Inherent Trade-Offs in Fair Determination of Risk.
- Mitchell et al. (2019). Model Cards for Model Reporting. *FAccT*.
