# STP Traceability Matrix

Every requirement and named test script from the original System Test Plan, mapped to the code and
tests that satisfy it. Status legend: ✅ done · 🟡 in progress · ⬜ planned.

## Functional Requirements

| Req | Description | Implemented in | Test (STP script) | Status |
|---|---|---|---|---|
| FR-001 | Upload models (.h5/.pt/.pkl) | `ethiclens_api/ingestion.py` + `POST /api/models/upload` (sandboxed) | `TS-FUNC-001` → `test_models_and_governance.py` | ✅ |
| FR-002 | Synthetic personas, configurable | `fairness_core.profiles.generate_profiles` | `TS-UNIT-002` → `test_generator.py` | ✅ |
| FR-003 | Compute DI / SPD / EO + composite | `fairness_core.metrics`, `audit.run_audit` | `TS-UNIT-001/003/004` → `test_metrics_known_values.py` | ✅ |
| FR-004 | Create audit session | `audit_session` table + `POST /api/sessions/create` | `TS-FUNC-002` → `test_sessions_e2e.py` | ✅ |
| FR-005 | Ranked mitigations where DI<0.8 | `fairness_core.mitigation.recommender` + `GET /api/sessions/{id}/recommendations` | `TS-INT-002`, `TS-FUNC-004` → `test_mitigation.py`, `test_sessions_e2e.py` | ✅ |
| FR-006 | Explanation + estimated impact | `mitigation.recommender` | `TS-FUNC-004` → `test_mitigation.py` | ✅ |
| FR-007 | Fairness Scorecard PDF | `fairness_core.reporting.scorecard` + `GET /api/sessions/{id}/report` (ReportLab) | `TS-FUNC-003` → `test_reporting.py`, `test_sessions_e2e.py` | ✅ |
| FR-008 | Escalation workflow | descoped — sign-off (FR-010) is the governance gate | `TS-UAT-001` | ⬜ descoped |
| FR-009 | Re-audit after mitigation | `parent_session_id` lineage + `POST /api/sessions/{id}/mitigate` | `TS-UAT-002` → `test_sessions_e2e.py` | ✅ |
| FR-010 | Governance sign-off portal | `POST /api/sessions/{id}/sign-off` (RBAC) | `TS-UAT-003` → `test_models_and_governance.py` | ✅ |
| FR-011 | Lock + clearance notification | server-enforced lock ✅; notification is a log line only (`routers/governance.py`) | `TS-UAT-003` → `test_models_and_governance.py` | 🟡 lock done, notifier roadmap |

## Non-Functional Requirements

| Req | Description | Approach | Test | Status |
|---|---|---|---|---|
| NFR-PERF-001 | ≤90s audit for ≤500MB model | arq async + batched ONNX inference | `TS-PERF-001` | ⬜ not yet benchmarked |
| NFR-PERF-002 | ≥50 concurrent, <20% degradation | arq queue + DB pool | `TS-PERF-002` | ⬜ not yet benchmarked |
| NFR-PERF-003 | PDF ≤15s for 10 attributes | async report job, precomputed metrics | `TS-PERF-003` | ⬜ not yet benchmarked |
| NFR-DB-001 | Persist results ≤5s | worker writes in completion handler | `TS-INT-003` step 5 | 🟡 implemented, not timed |

## Golden reference & superseded tests

| Item | Where | Status |
|---|---|---|
| Golden `calibrated_bias_model.pkl` (DI ≈ 0.55, pinned) | `models/golden/` + `.github/workflows/golden-audit.yml` | ✅ |
| Fairlearn 1e-9 parity proof | `test_vs_fairlearn.py` | ✅ |
| Hypothesis property invariants | `test_properties.py` | ✅ |

> **Consciously superseded:** any STP assertion that treats the *synthetic-persona audit* as
> ground truth is deliberately replaced — see [`methodology.md`](methodology.md). The **pure-math**
> unit scripts (`TS-UNIT-001/003/004`) pass **verbatim**; we do not claim 100% pass-through of the
> synthetic-audit path, because that path was methodologically unsound.
