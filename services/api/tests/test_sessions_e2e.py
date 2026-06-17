"""End-to-end audit pipeline (reproduces STP TS-INT-003 over HTTP)."""

from __future__ import annotations

from httpx import AsyncClient


async def _completed_flagged_session(client: AsyncClient, headers: dict[str, str]) -> str:
    create = await client.post(
        "/api/sessions/create",
        headers=headers,
        json={
            "name": "Lending audit",
            "dataset": "synthetic",
            "protected_attributes": [{"name": "race"}],
            "target": "approved",
        },
    )
    assert create.status_code == 201
    session_id = create.json()["id"]

    run = await client.post(f"/api/sessions/{session_id}/run", headers=headers)
    assert run.status_code == 202

    status = await client.get(f"/api/sessions/{session_id}/status", headers=headers)
    assert status.json()["status"] in {"COMPLETED", "FLAGGED"}
    return session_id


async def test_full_pipeline(client: AsyncClient, auth) -> None:
    headers = await auth(client)
    session_id = await _completed_flagged_session(client, headers)

    # Metrics: race:Black should be flagged (DI < 0.80) with a confidence interval.
    metrics = await client.get(f"/api/sessions/{session_id}/metrics", headers=headers)
    assert metrics.status_code == 200
    body = metrics.json()
    black_di = next(
        m
        for m in body["metrics"]
        if m["group_label"] == "race:Black" and m["metric_type"] == "disparate_impact"
    )
    assert black_di["value"] < 0.80
    assert black_di["ci_low"] is not None and black_di["ci_high"] is not None
    assert body["has_labels"] is True  # Equalized Odds was computable

    # Recommendations only for the flagged group.
    recs = await client.get(f"/api/sessions/{session_id}/recommendations", headers=headers)
    assert recs.json()["flagged"] is True
    assert "race:Black" in recs.json()["recommendations"]

    # Fairness Scorecard PDF (FR-007).
    req = await client.post(f"/api/sessions/{session_id}/report", headers=headers)
    assert req.status_code == 202
    pdf = await client.get(f"/api/sessions/{session_id}/report", headers=headers)
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content[:5] == b"%PDF-"

    # Apply mitigation -> child session whose DI crosses 0.80.
    mit = await client.post(
        f"/api/sessions/{session_id}/mitigate",
        headers=headers,
        json={"strategy": "threshold_optimizer"},
    )
    assert mit.status_code == 202
    child_id = mit.json()["result_session_id"]
    assert child_id is not None

    child_metrics = await client.get(f"/api/sessions/{child_id}/metrics", headers=headers)
    child_di = next(m for m in child_metrics.json()["metrics"] if m["group_label"] == "race:Black")
    assert child_di["value"] >= 0.80  # mitigation crossed the threshold
    assert child_di["classification"] == "PASS"


async def test_session_ownership_isolation(client: AsyncClient, auth) -> None:
    owner = await auth(client, "owner@example.com")
    other = await auth(client, "other@example.com")
    session_id = await _completed_flagged_session(client, owner)
    # A different user cannot read someone else's session.
    resp = await client.get(f"/api/sessions/{session_id}", headers=other)
    assert resp.status_code == 404
