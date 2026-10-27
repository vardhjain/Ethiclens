"""Governance state machine and audit snapshot integrity regression tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from httpx import AsyncClient

from ethiclens_api.audit_service import execute_audit
from ethiclens_api.db import SessionLocal
from ethiclens_api.mitigation_service import apply_mitigation
from ethiclens_api.models import AuditSession, FairnessMetric, SessionStatus


async def _session(client: AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post(
        "/api/sessions/create",
        headers=headers,
        json={"dataset": "synthetic", "protected_attributes": [{"name": "race"}]},
    )
    assert response.status_code == 201
    return response.json()["id"]


async def _complete(client: AsyncClient, headers: dict[str, str], session_id: str) -> None:
    response = await client.post(f"/api/sessions/{session_id}/run", headers=headers)
    assert response.status_code == 202
    assert response.json()["status"] in {"COMPLETED", "FLAGGED"}


async def test_only_finished_audits_with_metrics_can_be_approved(client: AsyncClient, auth) -> None:
    owner = await auth(client, "governance-owner@example.com")
    approver = await auth(client, "governance-approver@example.com", "governance_approver")
    session_id = await _session(client, owner)
    url = f"/api/sessions/{session_id}/sign-off"

    # Every unfinished state is rejected, including failed audits.
    for state in (
        SessionStatus.DRAFT,
        SessionStatus.QUEUED,
        SessionStatus.RUNNING,
        SessionStatus.FAILED,
    ):
        async with SessionLocal() as db:
            obj = await db.get(AuditSession, UUID(session_id))
            assert obj is not None
            obj.status = state
            await db.commit()
        result = await client.post(url, headers=approver, json={"note": "Too early"})
        assert result.status_code == 409

    # A forged "completed" status without a real scorecard is also rejected.
    async with SessionLocal() as db:
        obj = await db.get(AuditSession, UUID(session_id))
        assert obj is not None
        obj.status = SessionStatus.COMPLETED
        obj.completed_at = datetime.now(UTC)
        await db.commit()
    result = await client.post(url, headers=approver, json={"note": "No metrics"})
    assert result.status_code == 409

    integrity = await client.get(f"/api/sessions/{session_id}/integrity", headers=owner)
    assert integrity.json()["status"] == "not_signed"


async def test_approver_cannot_sign_off_own_audit(client: AsyncClient, auth) -> None:
    approver = await auth(client, "self-approval@example.com", "governance_approver")
    session_id = await _session(client, approver)
    await _complete(client, approver, session_id)
    response = await client.post(
        f"/api/sessions/{session_id}/sign-off", headers=approver, json={"note": "Self"}
    )
    assert response.status_code == 403


async def test_signed_off_audit_has_verifiable_fingerprint_and_is_frozen(
    client: AsyncClient, auth
) -> None:
    owner = await auth(client, "fingerprint-owner@example.com")
    approver = await auth(client, "fingerprint-approver@example.com", "governance_approver")
    stranger = await auth(client, "fingerprint-stranger@example.com")
    session_id = await _session(client, owner)
    await _complete(client, owner, session_id)
    url = f"/api/sessions/{session_id}"

    signed = await client.post(
        f"{url}/sign-off", headers=approver, json={"note": "Reviewed the metrics"}
    )
    assert signed.status_code == 200
    digest = signed.json()["signoff_digest"]
    assert isinstance(digest, str) and len(digest) == 64
    assert signed.json()["status"] == "SIGNED_OFF"
    assert signed.json()["locked"] is True

    verification = await client.get(f"{url}/integrity", headers=owner)
    assert verification.status_code == 200
    assert verification.json() == {
        "status": "verified",
        "expected_digest": digest,
        "current_digest": digest,
    }
    assert (await client.get(f"{url}/integrity", headers=stranger)).status_code == 404
    assert (await client.post(f"{url}/run", headers=owner)).status_code == 409
    assert (
        await client.post(
            f"{url}/mitigate", headers=owner, json={"strategy": "threshold_optimizer"}
        )
    ).status_code == 409

    # Guard a previously queued worker job and a direct mitigation task.
    await execute_audit(UUID(session_id))
    with pytest.raises(ValueError, match="locked or unfinished"):
        await apply_mitigation(UUID(session_id), "threshold_optimizer", None)
    assert (await client.get(f"{url}/integrity", headers=owner)).json()["status"] == "verified"

    # Simulate database-level tampering: changing a metric invalidates the saved
    # fingerprint; this is not a claim of protection against a privileged DBA.
    async with SessionLocal() as db:
        from sqlalchemy import select

        metric = (
            (
                await db.execute(
                    select(FairnessMetric).where(FairnessMetric.session_id == UUID(session_id))
                )
            )
            .scalars()
            .first()
        )
        assert metric is not None
        metric.value = (metric.value or 0.0) + 0.1
        await db.commit()
    check = await client.get(f"{url}/integrity", headers=owner)
    assert check.json()["status"] == "mismatch"
    assert check.json()["expected_digest"] == digest
    assert check.json()["current_digest"] != digest


async def test_legacy_signed_audit_is_not_silently_rebaselined(client: AsyncClient, auth) -> None:
    owner = await auth(client, "legacy-approval-owner@example.com")
    session_id = await _session(client, owner)
    async with SessionLocal() as db:
        obj = await db.get(AuditSession, UUID(session_id))
        assert obj is not None
        obj.status = SessionStatus.SIGNED_OFF
        obj.locked = True
        obj.signoff_digest = None
        await db.commit()

    response = await client.get(f"/api/sessions/{session_id}/integrity", headers=owner)
    assert response.status_code == 200
    assert response.json()["status"] == "legacy_unverifiable"
