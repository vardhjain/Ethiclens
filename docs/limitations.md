# Limitations — what EthicLens does *not* prove

A fairness tool that oversells itself is worse than none. This page states, plainly,
the boundaries of what EthicLens can and cannot establish. Read it before trusting any
number this system produces.

## 1. A passing audit is not a legal clearance
EthicLens computes statistical fairness metrics (Disparate Impact, Statistical Parity
Difference, Equalized Odds, …). Meeting the 0.80 four-fifths threshold is **evidence**, not a
legal safe harbour. Adverse-impact law (EEOC Uniform Guidelines; EU AI Act conformity) involves
job-relatedness, business necessity, and the availability of less-discriminatory alternatives —
none of which a metric can settle. **Use EthicLens to triage and document, not to certify.**

## 2. The Composite Bias Score is a convenience, not a standard
The composite folds three metrics into one number with chosen weights (0.40 / 0.35 / 0.25). The
weights are a product decision, not a law of nature. Always read the **raw per-group metrics and
their confidence intervals** — the composite can mask a single severely-harmed group. (Despite
the name, a *higher* composite is *fairer*.)

## 3. The golden Disparate Impact is empirically pinned, not analytically known
The golden-reference model's DI (≈ 0.55) is a **regression anchor reproduced under a frozen
seed, dataset, and code version** — not a closed-form truth. A trained classifier's selection-rate
ratio is an emergent property of the model, features, threshold and fit; it cannot be derived in
closed form. The CI band (±0.02) exists to absorb platform-level floating-point non-determinism.
The value is meaningful as a *drift detector*, and that is all it claims to be.

## 4. Error-based metrics require ground-truth labels
Equalized Odds, Equal Opportunity, predictive parity and calibration compare error rates, which
need true outcomes `Y`. On a label-free cohort (e.g. purely synthetic personas) these are
**uncomputable**, and EthicLens returns `INSUFFICIENT_DATA` rather than a fabricated value. This
is the central correction over the original specification — see
[`docs/methodology.md`](docs/methodology.md).

## 5. Confidence intervals depend on subgroup size
Bootstrap CIs are only as trustworthy as the sample they resample. Below the minimum-subgroup
floors (100 for rates; 30 positives **and** 30 negatives for error metrics) EthicLens reports
`INSUFFICIENT_DATA`. Intersectional subgroups thin out fast; a wide CI means "we don't know," not
"it's fair."

## 6. The model sandbox is defense-in-depth, not a guarantee
Uploaded models are deserialised inside a network-isolated, resource-limited container because
unpickling untrusted files is remote code execution. This raises the bar substantially but is
**not** a guarantee against a determined attacker with a sandbox-escape. Prefer the safe formats
(`safetensors`, `skops`, ONNX-direct); treat pickle as a discouraged fallback. See
[`docs/adr/0002-onnx-keystone-and-sandbox.md`](docs/adr/0002-onnx-keystone-and-sandbox.md).

## 7. Counterfactual probing tests a narrow notion of fairness
Flipping a single protected attribute and holding all else fixed measures *ceteris paribus*
individual sensitivity. Real causal pathways are entangled (a protected attribute correlates with
many features), so a low counterfactual flip-rate does not imply the absence of structural bias.

## 8. Benchmarks are not your data
The bundled datasets (Folktables/ACS, COMPAS, German Credit) illustrate the engine. Fairness
findings on them say nothing about *your* model on *your* population. Re-run on representative,
governed data before drawing conclusions.

---

*If you find a place where the code claims more than this document allows, that is a bug — please
open an issue.*
