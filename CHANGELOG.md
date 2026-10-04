# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **API service** (`services/api`) — FastAPI with async SQLAlchemy and Alembic migrations:
  accounts and JWT sign-in, sandboxed model upload, audit sessions, mitigation with re-audit,
  Fairness Scorecard PDF, and a role-restricted governance sign-off that locks the session.
- **Web app** (`apps/web`) — React + TypeScript workbench for running audits, reading results,
  applying a mitigation and signing off; dark mode, error boundary, mobile layout, Markdown export.
- **Hosted agent** — upload a predictions CSV (or pick a demo dataset) and get an audit, a
  plain-English narrative, ranked recommendations and a Q&A chat. Every number the language
  model writes is checked against the computed scorecard; if it can't be verified, the response
  falls back to numbers only. Daily usage caps per user and overall.
- **Mitigation engine** — held-out accuracy-vs-fairness comparison, a ranked recommender, and
  measured before/after results.
- **Reporting** — Scorecard PDF, Model Card and Datasheet generators; sample outputs in `docs/`.
- **ML tooling** (`ml/`) — real-benchmark loaders (COMPAS, Adult, German Credit), the golden
  reference model, a dataset-audit CLI and a Gradio demo.
- **Deployment** — Cloud Run (API), Supabase (Postgres), Vercel (web), with a GitHub Actions
  deploy workflow and a deployment guide.
- **Project hygiene** — `SECURITY.md`, `CODE_OF_CONDUCT.md`, web lint/test/build in CI,
  `npm audit` and `pip-audit` gates, committed `uv.lock`.

### Changed
- Sign-in tokens now use PyJWT (replacing the unmaintained python-jose); the unused passlib
  dependency was removed.
- The agent's fallback language model moved to `gemini-3.5-flash` after `gemini-2.0-flash` was
  retired.
- `react-router-dom` upgraded to v7 to clear two security advisories.
- `fairness-core` now ships type information (`py.typed`).

### Fixed
- `/run` no longer reports a stale pre-audit status.
- Intermittent database errors in the test suite (SQLite connection pooling).
- A malformed login token now returns 401 instead of a server error.
- Stale status markers in the STP traceability matrix.

### Engine (first milestone)
- **`fairness-core` engine** — Disparate Impact, Statistical Parity Difference, Equalized Odds,
  Equal Opportunity, predictive parity, FPR balance, calibration/ECE, Theil index, and the
  Composite Bias Score, all implemented from scratch.
- **Statistical rigour** — BCa bootstrap confidence intervals, two-proportion significance test,
  and minimum-subgroup floors; groups are flagged on the CI, not the point estimate.
- **Synthetic profile generator** (FR-002) and a **counterfactual fairness probe** on real records.
- **`run_audit`** end-to-end pipeline (FR-003) that returns `INSUFFICIENT_DATA` for error-based
  metrics when ground-truth labels are absent.
- **`ethiclens-audit` CLI** that prints a Fairness Scorecard for a freshly trained biased model.
- **Correctness proof** — Fairlearn 1e-9 parity tests, Hypothesis property tests, and the STP's
  hard-coded unit values (`TS-UNIT-001/003/004`) reproduced verbatim. 96% coverage.
- Monorepo scaffold (uv workspace), Ruff + mypy-strict + pytest tooling, GitHub Actions CI
  (lint/type/test matrix), the golden-audit gate, and CodeQL/pip-audit security workflows.

### Documentation
- `README`, `LIMITATIONS.md` ("what this does *not* prove"), `docs/methodology.md`, and the STP
  traceability matrix.

[Unreleased]: https://github.com/vardhjain/Ethiclens/commits/main
