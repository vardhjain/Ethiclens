"""Public model-file execution must fail closed, including legacy sessions."""

from __future__ import annotations

from uuid import UUID, uuid4

import pandas as pd
import pytest
from httpx import AsyncClient

from ethiclens_api.audit_service import _AuditInputs, _resolve_model
from ethiclens_api.config import get_settings
from ethiclens_api.db import SessionLocal
from ethiclens_api.models import AuditSession, ModelFramework, SessionStatus, UploadedModel


async def test_model_routes_absent_by_default(client: AsyncClient) -> None:
    assert get_settings().enable_model_file_ingestion is False
    assert (await client.get("/api/models")).status_code == 404
    assert (await client.get(f"/api/models/{uuid4()}")).status_code == 404
    response = await client.post(
        "/api/models/upload",
        files={"file": ("untrusted.pkl", b"untrusted", "application/octet-stream")},
    )
    assert response.status_code == 404
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert not any(path.startswith("/api/models") for path in paths)


async def test_public_cannot_create_model_backed_session(client: AsyncClient, auth) -> None:
    headers = await auth(client)
    response = await client.post(
        "/api/sessions/create",
        headers=headers,
        json={"dataset": "synthetic", "model_id": str(uuid4())},
    )
    assert response.status_code == 403


async def test_legacy_model_session_cannot_run_or_mitigate(client: AsyncClient, auth) -> None:
    headers = await auth(client)
    owner = UUID((await client.get("/api/auth/me", headers=headers)).json()["id"])
    async with SessionLocal() as db:
        model = UploadedModel(
            owner_id=owner, filename="legacy.pkl", framework=ModelFramework.SKLEARN,
            file_uri="/no/such/model.pkl",
        )
        db.add(model)
        await db.flush()
        session = AuditSession(
            owner_id=owner, model_id=model.id,
            dataset="synthetic", status=SessionStatus.DRAFT,
        )
        db.add(session)
        await db.commit()
        session_id = session.id

    assert (await client.post(f"/api/sessions/{session_id}/run", headers=headers)).status_code == 403
    response = await client.post(
        f"/api/sessions/{session_id}/mitigate",
        headers=headers, json={"strategy": "threshold_optimizer"},
    )
    assert response.status_code == 403
    async with SessionLocal() as db:
        saved = await db.get(AuditSession, session_id)
        assert saved is not None and saved.status == SessionStatus.DRAFT


def test_internal_worker_refuses_existing_model_file(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "enable_model_file_ingestion", False)
    inputs = _AuditInputs(
        dataset="synthetic", seed=42, model_uri="/no/such/model.pkl",
        model_framework="sklearn", target="approved", feature_columns=[], specs=[],
    )
    with pytest.raises(PermissionError, match="disabled"):
        _resolve_model(inputs, pd.DataFrame(), "approved", [])
