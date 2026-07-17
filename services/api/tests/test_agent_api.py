"""End-to-end test of the agent HTTP endpoints (propose-schema -> run-audit -> get record)."""

from __future__ import annotations

import asyncio
import json
import time

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


class _AlwaysInvalidClient:
    """Simulates every configured provider failing: never returns valid schema JSON,
    so ``complete_json`` exhausts its retries and raises ``LLMCallError``."""

    def complete(self, system: str, prompt: str) -> str:
        return "not valid json"


def _patch_llm(monkeypatch) -> None:
    monkeypatch.setattr(agent_router, "build_default_client_or_none", lambda *_: _StubClient())


def _patch_llm_always_failing(monkeypatch) -> None:
    monkeypatch.setattr(
        agent_router, "build_default_client_or_none", lambda *_: _AlwaysInvalidClient()
    )


class _SlowClient:
    """Simulates a slow real network call — the SDK clients this stands in for
    (Groq/Gemini) block synchronously on I/O, just like ``time.sleep`` does here."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    def complete(self, system: str, prompt: str) -> str:
        time.sleep(self._delay_seconds)
        return json.dumps(
            {
                "narrative": "The audit found a disparity between groups.",
                "mitigation_summary": "Consider applying a mitigation strategy.",
            }
        )


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


async def test_propose_schema_returns_503_when_all_llm_providers_fail(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm_always_failing(monkeypatch)
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


async def test_list_records_returns_empty_list_for_new_user(client: AsyncClient, auth) -> None:
    headers = await auth(client)
    resp = await client.get("/api/agent/records", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_records_returns_summary_of_owned_records(
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

    resp = await client.get("/api/agent/records", headers=headers)
    assert resp.status_code == 200
    summaries = resp.json()
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["record_id"] == record_id
    assert summary["outcome_column"] == "flag"
    assert summary["grounded"] is True
    assert summary["degraded"] is False
    assert summary["composite_band"] is not None
    assert isinstance(summary["composite_score"], float)


async def test_list_records_orders_most_recent_first(
    client: AsyncClient, auth, monkeypatch
) -> None:
    _patch_llm(monkeypatch)
    headers = await auth(client)
    ids = []
    for _ in range(3):
        run = await client.post(
            "/api/agent/run-audit",
            headers=headers,
            files={"file": ("data.csv", _CSV, "text/csv")},
            data={"proposal_json": json.dumps(_PROPOSAL)},
        )
        ids.append(run.json()["record_id"])

    resp = await client.get("/api/agent/records", headers=headers)
    returned_ids = [r["record_id"] for r in resp.json()]
    assert returned_ids == list(reversed(ids))


async def test_list_records_does_not_return_another_users_records(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """Security-critical: a user must never see another user's audit records."""
    _patch_llm(monkeypatch)
    owner_headers = await auth(client, email="owner@example.com")
    await client.post(
        "/api/agent/run-audit",
        headers=owner_headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )

    other_headers = await auth(client, email="other@example.com")
    resp = await client.get("/api/agent/records", headers=other_headers)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_run_audit_does_not_block_the_event_loop(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """A slow LLM call must not stall other concurrent requests on the same process —
    regression test for the asyncio.to_thread offload in agent.py.

    Timing this reliably takes care: everything before the blocking LLM call (auth,
    multipart parsing, the has_budget DB check) is itself made of await points, so a
    lightweight concurrent request fired with no delay at all can race ahead and finish
    during that legitimate preamble — appearing "unblocked" regardless of whether the
    fix is applied. STARTUP_MARGIN below is chosen well above that preamble's observed
    cost so the lightweight request is reliably dispatched only once the slow request
    has already reached (and, if unfixed, is stuck in) the blocking call — then elapsed
    time is measured from one fixed, absolute start so an internally-delayed asyncio.sleep
    can't make a blocked wait look artificially fast relative to its own delayed start.
    """
    block_seconds = 2.0
    startup_margin = 0.5
    monkeypatch.setattr(
        agent_router, "build_default_client_or_none", lambda *_: _SlowClient(block_seconds)
    )
    headers = await auth(client)
    t_start = time.monotonic()

    async def slow_request() -> float:
        resp = await client.post(
            "/api/agent/run-audit",
            headers=headers,
            files={"file": ("data.csv", _CSV, "text/csv")},
            data={"proposal_json": json.dumps(_PROPOSAL)},
        )
        assert resp.status_code == 201
        return time.monotonic() - t_start

    async def lightweight_request() -> float:
        await asyncio.sleep(startup_margin)
        resp = await client.get("/api/agent/demo-datasets")
        assert resp.status_code == 200
        return time.monotonic() - t_start

    slow_elapsed, lightweight_elapsed = await asyncio.gather(slow_request(), lightweight_request())
    assert slow_elapsed >= block_seconds
    # If the slow request's blocking call had monopolized the event loop, this would
    # also be dragged out to ~block_seconds — nothing else could run until it cleared.
    assert lightweight_elapsed < block_seconds - startup_margin


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


async def test_run_audit_rejects_unknown_true_label_column(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """A confirmed proposal naming a nonexistent true_label_column must 400, not crash
    downstream with a KeyError when the executor reads it."""
    _patch_llm(monkeypatch)
    headers = await auth(client)
    bad_proposal = {**_PROPOSAL, "true_label_column": "does_not_exist"}
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(bad_proposal)},
    )
    assert resp.status_code == 400


async def test_run_audit_rejects_unknown_feature_column(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """A confirmed proposal naming a nonexistent feature column must 400, not crash
    downstream with a KeyError when fairness_core selects X = data[features]."""
    _patch_llm(monkeypatch)
    headers = await auth(client)
    bad_proposal = {**_PROPOSAL, "feature_columns": ["score", "does_not_exist"]}
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _CSV, "text/csv")},
        data={"proposal_json": json.dumps(bad_proposal)},
    )
    assert resp.status_code == 400


_NON_BINARY_OUTCOME_CSV = "\n".join(
    ["race,flag,score"]
    + [f"A,{'yes' if i < 6 else 'no'},{i}" for i in range(60)]
    + [f"B,{'yes' if i < 30 else 'no'},{i}" for i in range(60)]
)


async def test_run_audit_rejects_non_binary_outcome_column(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """A predictions column with non-0/1 values (e.g. "yes"/"no") must 400 instead of
    crashing deep inside _PassthroughModel.predict's unconditional .astype(int)."""
    _patch_llm(monkeypatch)
    headers = await auth(client)
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", _NON_BINARY_OUTCOME_CSV, "text/csv")},
        data={"proposal_json": json.dumps(_PROPOSAL)},
    )
    assert resp.status_code == 400
    assert "flag" in resp.json()["detail"]


async def test_run_audit_rejects_non_binary_true_label_column(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """Same guard for a confirmed true_label_column with non-0/1 values."""
    _patch_llm(monkeypatch)
    headers = await auth(client)
    csv = "\n".join(
        ["race,flag,score,actual"]
        + [f"A,{'1' if i < 6 else '0'},{i},{'yes' if i < 6 else 'no'}" for i in range(60)]
        + [f"B,{'1' if i < 30 else '0'},{i},{'yes' if i < 30 else 'no'}" for i in range(60)]
    )
    bad_proposal = {**_PROPOSAL, "true_label_column": "actual"}
    resp = await client.post(
        "/api/agent/run-audit",
        headers=headers,
        files={"file": ("data.csv", csv, "text/csv")},
        data={"proposal_json": json.dumps(bad_proposal)},
    )
    assert resp.status_code == 400
    assert "actual" in resp.json()["detail"]


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


async def test_run_audit_degrades_when_all_llm_providers_fail(
    client: AsyncClient, auth, monkeypatch
) -> None:
    """The deterministic stages (2-4) already succeeded before narrative generation is
    attempted; an LLM outage must degrade to the numeric-only narrative, not 500 and
    discard a completed audit."""
    _patch_llm_always_failing(monkeypatch)
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
    assert body["scorecard"]["groups"]

    fetched = await client.get(f"/api/agent/records/{body['record_id']}", headers=headers)
    assert fetched.status_code == 200


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
    rows += [f"{race[i]},{flag[i]},{score[i]},{label[i]},{feature[i]}" for i in range(2 * n)]
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


async def test_ask_returns_503_when_all_llm_providers_fail(
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

    _patch_llm_always_failing(monkeypatch)
    resp = await client.post(
        f"/api/agent/records/{record_id}/ask",
        headers=headers,
        json={"question": "Anything?"},
    )
    assert resp.status_code == 503
