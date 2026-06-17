# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
