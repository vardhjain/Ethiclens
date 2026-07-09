"""End-to-end test of the agent HTTP endpoints (propose-schema -> run-audit -> get record)."""

from __future__ import annotations

import json

from httpx import AsyncClient

import ethiclens_api.routers.agent as agent_router

_CSV = "\n".join(
    ["race,flag,score"]
    + [f"A,{'1' if i < 6 else '0'},{i}" for i in range(60)]
    + [f"B,{'1' if i < 30 else '0'},{i}" for i in range(60)]
)

_PROPOSAL = {
    "protected_attribute_columns": ["race"],
    "outcome_column": "flag",
    "outcome_direction": {
        "positive_label_meaning": "flagged as high risk",
        "positive_is_favorable": False,
    },
    "true_label_column": None,
    "feature_columns": ["score"],
    "reasoning": "test fixture",
}


class _StubClient:
    def complete(self, system: str, prompt: str) -> str:
        if "Question:" in prompt:
            return json.dumps({"answer": "race:Black was flagged based on the scorecard."})
        if "Scorecard JSON" in prompt:
            return json.dumps(
                {
                    "narrative": "The audit found a disparity between groups.",
                    "mitigation_summary": "Consider applying a mitigation strategy.",
                }
            )
        return json.dumps(_PROPOSAL)


def _patch_llm(monkeypatch) -> None:
    monkeypatch.setattr(agent_router, "build_default_client_or_none", lambda *_: _StubClient())


async def test_propose_schema_returns_proposal(client: AsyncClient, auth, monkeypatch) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/propose-schema",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome_column"] == "flag"
    assert body["outcome_direction"]["positive_is_favorable"] is False


async def test_propose_schema_without_llm_key_returns_503(
    client: AsyncClient, auth, monkeypatch
) -> None:
    monkeypatch.setattr(agent_router, "build_default_client_or_none", lambda *_: None)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/propose-schema",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
    )
    assert resp.status_code == 503


async def test_run_audit_persists_and_returns_report(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)

    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["grounded"] is True
    assert body["degraded"] is False
    assert "disparity" in body["narrative"]
    assert body["scorecard"]["groups"]
    record_id = body["record_id"]

    fetched = await client.get(f"/api/agent/records/{record_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["narrative"] == body["narrative"]


async def test_run_audit_rejects_unknown_columns(client: AsyncClient, auth, monkeypatch) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    bad_proposal = {**_PROPOSAL, "outcome_column": "does_not_exist"}
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(bad_proposal)},
    )
    assert resp.status_code == 400


async def test_run_audit_degrades_without_llm_key(client: AsyncClient, auth, monkeypatch) -> None:
    monkeypatch.setattr(agent_router, "build_default_client_or_none", lambda *_: None)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["degraded"] is True
    assert body["grounded"] is False


async def test_ask_about_record_returns_grounded_answer(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    run = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )
    record_id = run.json()["record_id"]

    resp = await client.post(
        f"/api/agent/records/{record_id}/ask",
        headers=headers,
        json={"question": "Why was race:Black flagged?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["degraded"] is False
    assert "flagged" in body["answer"]


async def test_ask_about_missing_record_returns_404(client: AsyncClient, auth, monkeypatch) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/records/00000000-0000-0000-0000-000000000000/ask",
        headers=headers,
        json={"question": "Anything?"},
    )
    assert resp.status_code == 404


async def test_run_audit_degrades_once_daily_quota_is_exhausted(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    monkeypatch.setattr(agent_router.get_settings(), "agent_daily_llm_call_cap", 0)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["degraded"] is True
    assert body["grounded"] is False


async def test_propose_schema_returns_429_once_daily_quota_is_exhausted(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    monkeypatch.setattr(agent_router.get_settings(), "agent_daily_llm_call_cap", 0)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/propose-schema",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
    )
    assert resp.status_code == 429


def _labeled_csv() -> str:
    import numpy as np

    rng = np.random.RandomState(7)
    n = 100
    score_a = rng.uniform(0.3, 0.9, n)
    score_b = rng.uniform(0.1, 0.6, n)
    score = np.concatenate([score_a, score_b])
    flag = (score >= 0.5).astype(int)
    label = (score + rng.normal(0, 0.15, n * 2) >= 0.5).astype(int)
    feature = rng.uniform(0, 1, n * 2)
    race = ["A"] * n + ["B"] * n
    rows = ["race,flag,risk_score,actual,feature"]
    rows += [
        f"{race[i]},{flag[i]},{score[i]},{label[i]},{feature[i]}" for i in range(2 * n)
    ]
    return "\n".join(rows)


_LABELED_PROPOSAL = {
    "protected_attribute_columns": ["race"],
    "outcome_column": "flag",
    "outcome_direction": {"positive_label_meaning": "approved", "positive_is_favorable": True},
    "true_label_column": "actual",
    "score_column": "risk_score",
    "feature_columns": ["feature"],
    "reasoning": "test fixture",
}


async def test_run_audit_includes_measured_mitigation_with_score_and_labels(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _labeled_csv(), "text/csv")},
        data={"proposal_json": json.dumps(_LABELED_PROPOSAL)},
    )
    assert resp.status_code == 201
    measured = resp.json()["scorecard"]["measured_mitigation"]
    assert measured is not None
    assert measured["strategy"] == "threshold_optimizer"
    assert 0.0 <= measured["accuracy_before"] <= 1.0
    assert 0.0 <= measured["accuracy_after"] <= 1.0


async def test_run_audit_measured_mitigation_none_without_score_column(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )
    assert resp.status_code == 201
    assert resp.json()["scorecard"]["measured_mitigation"] is None


async def test_ask_without_llm_key_returns_503(client: AsyncClient, auth, monkeypatch) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    run = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )
    record_id = run.json()["record_id"]

    monkeypatch.setattr(agent_router, "build_default_client_or_none", lambda *_: None)
    resp = await client.post(
        f"/api/agent/records/{record_id}/ask",
        headers=headers,
        json={"question": "Anything?"},
    )
    assert resp.status_code == 503
