<div align="center">

# ⚖️ EthicLens

### An AI Bias Detection & Mitigation Workbench for regulated decision models

Unifies fairness **detection**, prescriptive **mitigation**, and formal **governance sign-off**
in one tool — for compliance officers as well as ML engineers. Built for EU AI Act / EEOC
audit requirements.

**[▶ Try the hosted agent](https://ethiclens-dun.vercel.app)** &nbsp;·&nbsp;
**[▶ Try the Gradio demo](https://huggingface.co/spaces/vardhjain20/Ethiclens)** &nbsp;·&nbsp;
**[📖 Docs](https://vardhjain.github.io/Ethiclens/)**

[![Live Agent](https://img.shields.io/badge/agent-live%20on%20Vercel-blue)](https://ethiclens-dun.vercel.app)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow)](https://huggingface.co/spaces/vardhjain20/Ethiclens)
[![ci-python](https://github.com/vardhjain/Ethiclens/actions/workflows/ci-python.yml/badge.svg)](https://github.com/vardhjain/Ethiclens/actions/workflows/ci-python.yml)
[![golden-audit](https://github.com/vardhjain/Ethiclens/actions/workflows/golden-audit.yml/badge.svg)](https://github.com/vardhjain/Ethiclens/actions/workflows/golden-audit.yml)
[![security](https://github.com/vardhjain/Ethiclens/actions/workflows/security.yml/badge.svg)](https://github.com/vardhjain/Ethiclens/actions/workflows/security.yml)
[![coverage](https://img.shields.io/badge/coverage-96%25-2ea44f)](.github/workflows/ci-python.yml)
[![python](https://img.shields.io/badge/python-3.11%20|%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![code style: ruff](https://img.shields.io/badge/style-ruff-261230)](https://github.com/astral-sh/ruff)

</div>

> [!NOTE]
> **Read this first:** [`LIMITATIONS.md`](LIMITATIONS.md) — exactly what this tool does and does
> **not** prove. Honesty about epistemics is a feature, not an afterthought.

---

## Why this exists

Organisations deploying automated hiring, lending and risk models face real legal and
reputational exposure when those models discriminate. Existing tools force a trade-off:

| Tool | Detection | **Mitigation** | Governance | Non-technical UI |
|---|:---:|:---:|:---:|:---:|
| IBM AI Fairness 360 | ✅ (code-only) | ✅ (code-only) | ❌ | ❌ |
| Google What-If Tool | ✅ (viz-only) | ❌ | ❌ | ⚠️ |
| **EthicLens** | ✅ | ✅ **executable + measured** | ✅ sign-off + audit trail | ✅ |

EthicLens closes the loop from *"this model is biased"* to *"here is a measured fix and a
signed, immutable audit record."*

## What makes it credible (the three things to look at)

1. **Verifiable numeric correctness.** Every fairness metric is implemented **from scratch** and
   **cross-validated against [Fairlearn](https://fairlearn.org/) to a 1e-9 tolerance**, with
   [Hypothesis](https://hypothesis.readthedocs.io/) property tests. A **golden-reference model**
   with an empirically-pinned Disparate Impact (≈ 0.55) is **asserted in CI** — if the bias math
   ever drifts, the build goes red. *Almost no portfolio repo can prove its math is correct.*
2. **Real, measured mitigation — never faked.** A held-out **accuracy-vs-fairness Pareto frontier**
   (with bootstrap CI error bars) feeds a ranked recommender; the reported improvement is a
   *measured* delta on data the mitigation never touched, and the re-audit actually crosses the
   0.80 threshold.
3. **Security & responsible-AI maturity.** Uploaded models are deserialised in a **sandbox** (the
   original spec's `pickle.load` of untrusted files is a remote-code-execution hole — fixed here),
   metrics ship with **bootstrap confidence intervals** and minimum-subgroup floors, and every
   audit emits a **Model Card** + **Datasheet** + an honesty banner.

## 🤖 Hosted agent

**[ethiclens-dun.vercel.app](https://ethiclens-dun.vercel.app)** — upload a predictions CSV (or
pick one of three built-in demo datasets) and get a bias audit with confidence intervals, a
plain-English narrative, ranked mitigation recommendations, and a grounded Q&A chat, in about a
minute. It's a brain on top of the same `fairness-core` engine described above — an agentic layer,
not a rebuild.

```mermaid
flowchart TD
    U[Upload predictions CSV] --> S1
    subgraph LLM["LLM stages"]
        S1["Stage 1 — Schema inference"]
        S4["Stage 4 — Narrative"]
        S5["Stage 5 — Grounded Q&A"]
    end
    subgraph DET["Deterministic stages — no LLM"]
        S2["Stage 2 — Audit-planning rules engine"]
        S3["Stage 3 — fairness-core execution"]
    end
    S1 --> H{{"Human confirms<br/>or edits the proposal"}}
    H --> S2 --> S3 --> S4
    S4 --> G{{"Number-grounding<br/>validator"}}
    G -- pass --> R["Scorecard + narrative"]
    G -- "fail twice" --> N["Numeric-only fallback"]
    R --> S5 --> G2{{"Number-grounding<br/>validator"}}
    G2 --> A["Grounded answer"]
```

**Guardrails** — the same discipline the core engine uses, applied to the LLM layer:

| Guardrail | Enforced in |
|---|---|
| LLM never computes a metric or threshold | `agent/audit_planner.py`, `agent/executor.py` — no LLM import in either |
| Every number the LLM writes is traceable to the scorecard | `agent/grounding.py`; regenerate once, then degrade to numeric-only |
| An audit never runs on an unconfirmed schema guess | two-endpoint split (`/propose-schema` → `/run-audit`) in `routers/agent.py` |
| LLM cost has a hard daily ceiling, degrades gracefully | `agent/quota.py`, `usage_counter` table |
| Uploaded data is never persisted, only the derived scorecard | `AgentAuditRecord` model; 5MB / 50,000-row upload caps |

**What's deliberately not in the hosted deployment, and why:** the arq/Redis worker queue
(seconds-long jobs don't justify a distributed queue — `EAGER_TASKS=true` runs everything
in-process, see `ethiclens_api/tasks.py`), MLflow (Postgres stores audit records, not
experiments), and model-file uploads (predictions-only by design — a public endpoint accepting
arbitrary model files is an attack surface the enterprise `/sessions` flow's sandbox exists to
manage, not something to also expose here). Full deployment guide: [`DEPLOYMENT.md`](DEPLOYMENT.md).
See [`LIMITATIONS.md`](LIMITATIONS.md#9-the-hosted-agent-is-a-demonstration-deployment-not-production-infrastructure)
for what this deployment does not guarantee.

<details>
<summary>Deployment topology (Cloud Run, Supabase, Vercel — all free tier)</summary>

```mermaid
flowchart LR
    B["Browser"] -->|static assets| V["Vercel"]
    B -->|"/api/* rewrite"| V
    V -->|"proxied, same-origin — no CORS"| C["Cloud Run: FastAPI + agent"]
    C --> S[("Supabase Postgres")]
    C -->|primary| Q["Groq"]
    C -.->|fallback| G["Gemini"]
```

Originally targeted Hugging Face Spaces (Docker) — the free tier's 2 vCPU/16GB RAM is the best
free ceiling for safely importing pandas + scikit-learn + fairlearn. Pivoted to Cloud Run mid-build
when HF began gating the Docker SDK behind a paid plan with no announcement. Cloud Run's free tier
(2M requests + generous vCPU/memory-seconds per month, scales to zero) covers a low-traffic demo
just as well.
</details>

## ▶ Quickstart

```bash
# 1. The engine, proven correct, in 30 seconds (no Docker needed)
uv venv && uv pip install -e "packages/fairness-core[validation,viz,cli]"
uv run ethiclens-audit demo          # trains a biased model, audits it, prints a scorecard
make audit-golden                    # reproduces the CI-pinned golden DI ≈ 0.55

# 2. The full stack (API + Postgres + React + MLflow)
cp .env.example .env
docker compose up --build            # → web http://localhost:5173 · api http://localhost:8000/docs
```

## Example: the scorecard the CLI prints

```
==============================================================================
                         EthicLens Fairness Scorecard
==============================================================================
Composite Bias Score: 0.612  [Medium Risk]   (higher = fairer)
Worst-group Disparate Impact: 0.55   Labels available: yes
------------------------------------------------------------------------------
Group                      DI          95% CI     SPD      EO    Flag
------------------------------------------------------------------------------
race:Black              0.553   [0.51,0.60]  -0.282   0.141    FLAG
race:Hispanic           0.910   [0.86,0.96]  -0.058   0.044      ok
race:Asian              1.020   [0.97,1.07]   0.014   0.031      ok
==============================================================================
[!] 1 flagged group(s): race:Black
    A group is flagged only when its DI confidence interval is below 0.80.
==============================================================================
```

`ethiclens-audit mitigate` then ranks fixes and applies the top one, re-auditing on held-out
data — e.g. ThresholdOptimizer drives **race:Black DI 0.36 → 0.91 (crosses 0.80)** for ~1% accuracy.

## Auditing real benchmarks — why one metric isn't enough

The same engine audits any classifier on real labelled benchmarks
(`python -m ml.cli.audit_dataset compas`). Running it on two famous datasets shows why a serious
audit needs **both** selection-rate *and* error-rate metrics:

| Dataset | Outcome | Most-affected group | DI | Equalized Odds | Flagged by |
|---|---|---|---:|---:|---|
| **COMPAS** — recidivism (*adverse*) | "will reoffend" | race: African-American | 2.50 | **0.34** | **Equalized Odds** |
| **Adult** — income (*favorable*) | "earns > $50k" | sex: Female | **0.51** | 0.01 | **Disparate Impact** |

On **COMPAS** the Disparate-Impact rule is *blind*: African-American defendants are flagged
high-risk *more* often, which the 4/5ths under-selection rule ignores — but their **false-positive
rate is ~2× higher (0.24 vs 0.11)**, the real [ProPublica finding](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing),
caught by **Equalized Odds**. On **Adult** it's the reverse — women are predicted to earn >$50k at
half the rate of men, caught by **Disparate Impact**. EthicLens flags on *either*, so it covers
both hiring/lending-style (favorable) and risk-scoring (adverse) decisions. See
[methodology](docs/methodology.md#worked-example-compas).

## Architecture

A three-tier system around one shared, audited fairness engine.

```
┌──────────────┐   REST/JSON    ┌────────────────────────┐   imports   ┌────────────────────┐
│  React 18 UI │ ─────────────▶ │  FastAPI service       │ ──────────▶ │  fairness-core     │
│  (Mantine)   │ ◀───────────── │  + arq async workers   │             │  (pure, mypy-strict│
└──────────────┘   poll status  │  + sandboxed ingestion │             │   Fairlearn-proven)│
                                 └───────────┬────────────┘             └────────────────────┘
                                             │  SQLAlchemy 2.0 (async)         ▲   imports
                                       ┌──────▼───────┐                  ┌─────┴──────┐
                                       │ PostgreSQL 16│                  │  ml/ + CLI │
                                       │ 5-entity model│                 │ notebooks  │
                                       └──────────────┘                  └────────────┘
```

The **same `fairness_core` code** powers the API workers, the CLI, and the notebooks, so the
numbers can never diverge between "what the demo shows" and "what the service computes."

## Repository layout

| Path | What |
|---|---|
| `packages/fairness-core/` | ★ The audited metric + mitigation engine (zero web/db deps) |
| `services/api/` | FastAPI app, async ORM, arq workers, sandboxed model ingestion |
| `services/api/.../agent/` | The hosted agent: LLM schema inference, planner, executor, narrative, Q&A |
| `apps/web/` | React 18 + Vite + TypeScript workbench (incl. the agent's upload → report flow) |
| `ml/` | Dataset loaders, model training, the golden oracle, notebooks, CLI |
| `docs/` | Methodology, ADRs, STP traceability matrix, per-persona guides |
| `infra/` · `.github/` | Docker/compose · CI, golden-audit gate, security scans |

## Live demo & notebook

- 🤖 **[Hosted agent](https://ethiclens-dun.vercel.app)** — the full agentic flow described above:
  upload a CSV or pick a canned dataset (COMPAS, Adult Income, or a synthetic hiring screen), get a
  narrated audit and grounded chat.
- ▶ **[Gradio demo](https://huggingface.co/spaces/vardhjain20/Ethiclens)** — the original workbench
  demo (source: [`ml/demo/app.py`](ml/demo/app.py)) that audits a biased model and applies a
  measured mitigation in the browser.
- 📓 **Narrative notebook** — [*Why synthetic-persona auditing is broken — and the fix*](ml/notebooks/why_synthetic_auditing_is_broken.py),
  including the **impossibility theorem** demonstration:

  ![Impossibility theorem](docs/impossibility-theorem.png)

- 🧾 Generated outputs: [sample Fairness Scorecard PDF](docs/sample-scorecard.pdf) ·
  [Model Card](docs/sample-model-card.md) · [Datasheet](docs/sample-datasheet.md).

## Documentation

- 📐 [Methodology](docs/methodology.md) — metric formulas, the composite rationale, and the
  methodological fix over the original spec.
- 🧾 [STP traceability matrix](docs/traceability-matrix.md) — every requirement → code + test.
- 🏛️ [Architecture Decision Records](docs/adr/) — the choices and why.
- 🚀 [Deployment guide](DEPLOYMENT.md) — how the hosted agent gets to Cloud Run + Supabase + Vercel.
- ⚠️ [Limitations](LIMITATIONS.md) — what this does **not** prove.

## Provenance

EthicLens began as a graduate **System Test Plan** (GWU SEAS) that was never implemented. This
repository is the implementation — and, deliberately, a **correction**: the original design
audited models on synthetic personas, which cannot measure real disparate impact. See
[`docs/methodology.md`](docs/methodology.md) and the original document in
[`docs/original-stp.pdf`](docs/original-stp.pdf).

## License

[MIT](LICENSE) © 2026 Vardh Jain
