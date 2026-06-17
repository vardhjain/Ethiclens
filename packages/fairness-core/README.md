# fairness-core

The pure-compute heart of **EthicLens** — an audited, framework-agnostic fairness engine.

`fairness-core` has **zero web/database dependencies** and is the single source of truth for
every fairness metric in the system. The API service, the CLI, and the analysis notebooks all
import the *same* metric implementations, so the numbers can never drift.

## Why it's trustworthy

- **Implemented from scratch** (Disparate Impact, Statistical Parity Difference, Equalized Odds,
  Equal Opportunity, Predictive Parity, calibration, composite score) — and then
  **cross-validated against [Fairlearn](https://fairlearn.org/) to a 1e-9 tolerance** in CI.
- **Statistical rigour the spec lacked:** BCa bootstrap confidence intervals, two-proportion
  significance tests, and minimum-subgroup floors. A group is flagged only when its CI excludes
  the 0.80 threshold — not on a noisy point estimate.
- **Property-tested** with [Hypothesis](https://hypothesis.readthedocs.io/) (e.g. DI = 1 at
  parity, SPD ∈ [-1, 1], the composite score is bounded and monotone).
- A **golden-reference model** with a CI-asserted Disparate Impact is checked on every push.

```python
from fairness_core.metrics import calculate_disparate_impact, classify_disparate_impact

di = calculate_disparate_impact(privileged_rate=0.80, unprivileged_rate=0.40)  # 0.50
classify_disparate_impact(di)  # "FAIL" — below the 4/5ths (0.80) rule
```

See the repository root `README.md` for the full project and methodology.
