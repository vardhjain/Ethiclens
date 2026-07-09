"""End-to-end tests for the one-click canned-dataset demo endpoints."""

from __future__ import annotations

import json

from httpx import AsyncClient

import ethiclens_api.routers.agent as agent_router


class _StubClient:
    def complete(self, system: str, prompt: str) -> str:
        if "Question:" in prompt:
            return json.dumps({"answer": "grounded answer"})
        return json.dumps(
            {"narrative": "The audit found a disparity.", "mitigation_summary": "See recs."}
        )


def _patch_llm(monkeypatch) -> None:
    monkeypatch.setattr(agent_router, "build_default_client_or_none", lambda *_: _StubClient())


async def test_list_demo_datasets(client: AsyncClient, auth) -> None:
    headers = await auth(client)
    resp = await client.get("/api/agent/demo-datasets", headers=headers)
    assert resp.status_code == 200
    keys = {d["key"] for d in resp.json()}
    assert keys == {"compas", "adult_income", "synthetic_hiring"}


async def test_demo_audit_unknown_key_returns_404(client: AsyncClient, auth, monkeypatch) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/demo-audit", headers=headers, json={"dataset_key": "does-not-exist"}
    )
    assert resp.status_code == 404


async def test_demo_audit_compas_has_measured_mitigation(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/demo-audit", headers=headers, json={"dataset_key": "compas"}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["grounded"] is True
    sc = body["scorecard"]
    assert sc["has_labels"] is True
    assert sc["measured_mitigation"] is not None
    assert sc["measured_mitigation"]["strategy"] == "threshold_optimizer"


async def test_demo_audit_adult_income_has_labels_no_measured_mitigation(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/demo-audit", headers=headers, json={"dataset_key": "adult_income"}
    )
    assert resp.status_code == 201
    sc = resp.json()["scorecard"]
    assert sc["has_labels"] is True
    assert sc["measured_mitigation"] is None


async def test_demo_audit_synthetic_hiring_skips_equalized_odds(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/demo-audit", headers=headers, json={"dataset_key": "synthetic_hiring"}
    )
    assert resp.status_code == 201
    sc = resp.json()["scorecard"]
    assert sc["has_labels"] is False
    assert sc["measured_mitigation"] is None
    assert "skipped" in sc["metrics_reason"]


async def test_demo_audit_allows_overriding_proposal(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """The human-confirmation step still applies to demos: a caller may edit the
    stored proposal before running (e.g. drop a feature column)."""
    _patch_llm(monkeypatch)
    headers = await auth(client)
    listing = await client.get("/api/agent/demo-datasets", headers=headers)
    compas = next(d for d in listing.json() if d["key"] == "compas")
    edited = {**compas["proposal"], "feature_columns": []}

    resp = await client.post(
        "/api/agent/demo-audit",
        headers=headers,
        json={"dataset_key": "compas", "proposal": edited},
    )
    assert resp.status_code == 201
    assert resp.json()["plan"]["feature_columns"] == []


async def test_demo_audit_persists_and_is_retrievable(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/demo-audit", headers=headers, json={"dataset_key": "synthetic_hiring"}
    )
    record_id = resp.json()["record_id"]
    fetched = await client.get(f"/api/agent/records/{record_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["scorecard"] == resp.json()["scorecard"]
