"""Model ingestion (FR-001) and the governance workflow (FR-008/010/011)."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient

_GOLDEN = Path(__file__).resolve().parents[3] / "models" / "golden" / "calibrated_bias_model.pkl"


async def test_upload_rejects_unsupported_type(client: AsyncClient, auth) -> None:
    headers = await auth(client)
    resp = await client.post(
        "/api/models/upload",
        headers=headers,
        files={"file": ("data.csv", b"a,b,c\n1,2,3\n", "text/csv")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@pytest.mark.skipif(not _GOLDEN.exists(), reason="golden model not built")
async def test_upload_and_audit_golden_model(client: AsyncClient, auth) -> None:
    headers = await auth(client)
    upload = await client.post(
        "/api/models/upload",
        headers=headers,
        files={"file": ("calibrated_bias_model.pkl", _GOLDEN.read_bytes(), "application/octet")},
    )
    assert upload.status_code == 202
    model = upload.json()
    assert model["framework"] == "sklearn"
    model_id = model["id"]

    create = await client.post(
        "/api/sessions/create",
        headers=headers,
        json={
            "dataset": "golden",
            "model_id": model_id,
            "protected_attributes": [
                {"name": "race", "privileged_value": "White", "unprivileged_values": ["Black"]}
            ],
            "target": "approved",
        },
    )
    session_id = create.json()["id"]
    await client.post(f"/api/sessions/{session_id}/run", headers=headers)

    metrics = await client.get(f"/api/sessions/{session_id}/metrics", headers=headers)
    black = next(
        m
        for m in metrics.json()["metrics"]
        if m["group_label"] == "race:Black" and m["metric_type"] == "disparate_impact"
    )
    assert black["value"] < 0.80  # the golden model is biased by construction


async def test_governance_signoff_locks_session(client: AsyncClient, auth) -> None:
    eng = await auth(client, "eng2@example.com", role="ml_engineer")
    approver = await auth(client, "exec@example.com", role="governance_approver")

    create = await client.post(
        "/api/sessions/create",
        headers=eng,
        json={
            "dataset": "synthetic",
            "protected_attributes": [{"name": "race"}],
            "target": "approved",
        },
    )
    session_id = create.json()["id"]
    await client.post(f"/api/sessions/{session_id}/run", headers=eng)

    # An ML engineer may NOT sign off (RBAC).
    denied = await client.post(
        f"/api/sessions/{session_id}/sign-off", headers=eng, json={"note": "ok"}
    )
    assert denied.status_code == 403

    # The governance approver can.
    ok = await client.post(
        f"/api/sessions/{session_id}/sign-off", headers=approver, json={"note": "Approved."}
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "SIGNED_OFF"
    assert ok.json()["locked"] is True

    # A signed-off (locked) session cannot be re-run, and cannot be signed off twice.
    rerun = await client.post(f"/api/sessions/{session_id}/run", headers=eng)
    assert rerun.status_code == 409
    again = await client.post(
        f"/api/sessions/{session_id}/sign-off", headers=approver, json={"note": "again"}
    )
    assert again.status_code == 409
