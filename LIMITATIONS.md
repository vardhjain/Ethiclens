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

## 9. The hosted agent is a demonstration deployment, not production infrastructure
The live agent (Cloud Run + Supabase + Vercel, see [`DEPLOYMENT.md`](DEPLOYMENT.md)) runs on free
tiers with **no uptime guarantee**. Cloud Run scales to zero between requests, so the first
request after idle time pays a cold-start cost (a few seconds). Treat it as a portfolio artifact
to click through, not a system to depend on.

## 10. The agent's narrative and Q&A can silently downgrade to numeric-only
The LLM layer (schema inference, narrative, chat) degrades to a plain numeric summary — never a
fabricated explanation — under any of: no `GROQ_API_KEY`/`GEMINI_API_KEY` configured, the daily
LLM call budget exhausted (`agent_daily_llm_call_cap` in
[`config.py`](services/api/src/ethiclens_api/config.py)), or two consecutive generations failing
the number-grounding validator ([`agent/grounding.py`](services/api/src/ethiclens_api/agent/grounding.py)).
Every response carries `grounded`/`degraded` flags so this is never silent to the API caller — the
UI surfaces them as badges — but it means two audits of the same data can produce different
*prose* (never different *numbers*) depending on quota state.

## 11. Measured mitigation is one strategy, not a menu
The agent's before/after mitigation chart is *measured* (fit on a train split, evaluated on
held-out data) only for **group-specific decision thresholds** (Fairlearn's `ThresholdOptimizer`),
and only when the uploaded CSV has both a continuous score column and true labels. Reweighing and
constrained-retraining strategies — available in the core `fairness_core.mitigation` engine and
the enterprise `/sessions` workbench — need a trainable model object, which a predictions-only CSV
structurally never provides. This is a design constraint of the hosted agent, not an oversight; see
[`agent/executor.py`](services/api/src/ethiclens_api/agent/executor.py).

## 12. The agent never persists what you upload
Predictions CSVs are parsed in memory and discarded; only the derived scorecard JSON and narrative
are stored ([`AgentAuditRecord`](services/api/src/ethiclens_api/models.py)). Uploads are capped at
5MB / 50,000 rows. The three demo datasets are the only data that ships with the deployment.

## 13. The hosted agent intentionally exposes a smaller surface than the full repo
Model-file ingestion, MLflow tracking, and the arq/Redis worker queue all exist in this repository
and are exercised by the enterprise `/sessions` flow and its test suite, but are **not** present in
the agent's hosted deployment: it accepts predictions CSVs only (never model files — see
[`agent/csv_ingest.py`](services/api/src/ethiclens_api/agent/csv_ingest.py)), stores audit records
in Postgres instead of MLflow, and runs every job in-process (`EAGER_TASKS=true`) instead of
queuing to a worker. Each is a deliberate scope decision for a public, free-tier deployment, not a
missing feature — see [`DEPLOYMENT.md`](DEPLOYMENT.md#whats-intentionally-not-enabled-here).

---

*If you find a place where the code claims more than this document allows, that is a bug — please
open an issue.*
