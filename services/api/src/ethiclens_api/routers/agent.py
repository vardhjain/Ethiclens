"""HTTP endpoints for the agent pipeline (Stages 1-4).

Two-step, stateless-until-confirmed flow, matching the "never auto-run an audit on an
unconfirmed guess" and "no data persistence" guardrails:

1. ``POST /api/agent/propose-schema`` — upload a CSV, get back the LLM's proposed
   column roles. Nothing is persisted; the CSV is parsed in memory and discarded.
2. ``POST /api/agent/run-audit`` — upload the *same* CSV plus a (user-reviewed,
   possibly edited) confirmed proposal. Runs Stages 2-4 and persists only the
   resulting scorecard/narrative, never the raw rows.
3. ``POST /api/agent/records/{id}/ask`` — Stage 5: ask a grounded question about a
   stored record; retrieves from its scorecard JSON, never from model memory.
   ``GET /api/agent/records`` lists the caller's own past records (summary fields
   only); ``GET /api/agent/records/{id}`` returns one in full.
4. ``GET /api/agent/demo-datasets`` / ``POST /api/agent/demo-audit`` — one-click
   canned datasets shipped with the app, so a visitor never has to bring their own
   CSV. Skips Stage 1 (the proposal is hand-verified, checked into
   ``agent/demo_datasets.py``) but otherwise runs the identical Stages 2-4 pipeline
   as ``run-audit``, via the same ``_run_and_persist`` helper.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.agent.audit_planner import plan_audit
from ethiclens_api.agent.csv_ingest import read_predictions_csv
from ethiclens_api.agent.demo_datasets import get_demo_dataset, list_demo_datasets
from ethiclens_api.agent.executor import execute_plan, measure_top_mitigation
from ethiclens_api.agent.llm_client import LLMCallError, build_default_client_or_none
from ethiclens_api.agent.narrative import build_scorecard, generate_narrative, unavailable_result
from ethiclens_api.agent.qa import ask
from ethiclens_api.agent.quota import has_budget, record_llm_call
from ethiclens_api.agent.schema_inference import infer_schema
from ethiclens_api.agent.schemas import AskRequest, SchemaInferenceProposal
from ethiclens_api.config import get_settings
from ethiclens_api.db import get_session
from ethiclens_api.models import AgentAuditRecord, UserAccount
from ethiclens_api.security import get_current_user

_log = logging.getLogger("ethiclens.agent")

router = APIRouter(prefix="/api/agent", tags=["agent"])


async def _owned_record(record_id: UUID, user: UserAccount, db: AsyncSession) -> AgentAuditRecord:
    record = await db.get(AgentAuditRecord, record_id)
    if record is None or record.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audit record not found")
    return record


def _non_binary_values(series: pd.Series, limit: int = 5) -> list[str]:
    """Non-null values in ``series`` that don't coerce to 0/1 (int/float/bool/"0"/"1" all do).

    Stage 3 (:func:`ethiclens_api.agent.executor.execute_plan`) unconditionally does
    ``.astype(int)`` on the outcome and true-label columns; anything that doesn't coerce
    cleanly either raises deep inside the engine (a bare 500) or, worse, silently coerces
    to something meaningless. Caught here instead, against the raw upload, with a message
    naming the offending column.
    """
    bad: list[str] = []
    for value in series.dropna().unique():
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            bad.append(str(value))
        else:
            if coerced not in (0.0, 1.0):
                bad.append(str(value))
        if len(bad) >= limit:
            break
    return bad


async def _run_and_persist(
    proposal: SchemaInferenceProposal,
    data: pd.DataFrame,
    db: AsyncSession,
    user: UserAccount,
) -> dict:
    """Stages 2-4 shared by ``/run-audit`` and ``/demo-audit``: plan, execute, measure
    mitigation, generate narrative, persist only the derived scorecard/narrative."""
    # score_column isn't checked here: plan_audit() already nulls it out below if it's
    # missing or not genuinely continuous, so it can never reach a downstream KeyError.
    required_columns = [
        proposal.outcome_column,
        *proposal.protected_attribute_columns,
        *proposal.feature_columns,
    ]
    if proposal.true_label_column is not None:
        required_columns.append(proposal.true_label_column)
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Columns not found in CSV: {missing}")

    binary_columns = [proposal.outcome_column]
    if proposal.true_label_column is not None:
        binary_columns.append(proposal.true_label_column)
    for col in binary_columns:
        bad_values = _non_binary_values(data[col])
        if bad_values:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Column '{col}' must be binary (0/1) but contains non-binary values, "
                f"e.g. {bad_values}",
            )

    settings = get_settings()
    plan = plan_audit(proposal, data)
    # execute_plan (a bootstrap over up to agent_max_rows rows) and generate_narrative
    # (a blocking Groq/Gemini SDK call) both run in a worker thread, not inline on the
    # event loop — otherwise one large audit or one slow LLM call blocks every other
    # request this process is serving, /health included. Same pattern already used by
    # audit_service.execute_audit for the enterprise /sessions flow.
    result = await asyncio.to_thread(execute_plan, plan, data, proposal.outcome_direction)
    # Real, held-out-measured before/after — only possible while the CSV is still in
    # memory (never persisted); unavailable without a score column + true labels.
    measured_mitigation = await asyncio.to_thread(
        measure_top_mitigation, plan, data, proposal.outcome_direction, result
    )
    scorecard, _recommendations = build_scorecard(result, plan, measured_mitigation)

    client = build_default_client_or_none(settings.groq_api_key, settings.gemini_api_key)
    if client is not None and await has_budget(db, settings.agent_daily_llm_call_cap):
        try:
            narrative_result = await asyncio.to_thread(generate_narrative, client, scorecard)
        except LLMCallError as exc:
            # All configured providers failed (outage, bad key, rate limit). The
            # deterministic stages above already succeeded — degrade to a numeric-only
            # narrative rather than discard a completed audit with a bare 500.
            _log.warning("Narrative generation failed, degrading to numeric-only: %s", exc)
            narrative_result = unavailable_result(scorecard)
        await record_llm_call(db)
    else:
        if client is None:
            _log.info("No LLM provider configured; narrative will be numeric-only")
        else:
            _log.info("Daily LLM call budget exhausted; narrative will be numeric-only")
        narrative_result = unavailable_result(scorecard)

    record = AgentAuditRecord(
        owner_id=user.id,
        outcome_column=plan.outcome_column,
        protected_attribute_columns=plan.protected_attribute_columns,
        true_label_column=plan.true_label_column,
        plan=plan.model_dump(),
        scorecard=scorecard,
        narrative=narrative_result.output.narrative,
        mitigation_summary=narrative_result.output.mitigation_summary,
        grounded=narrative_result.grounded,
        degraded=narrative_result.degraded,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return {
        "record_id": str(record.id),
        "plan": plan.model_dump(),
        "scorecard": scorecard,
        "narrative": narrative_result.output.narrative,
        "mitigation_summary": narrative_result.output.mitigation_summary,
        "grounded": narrative_result.grounded,
        "degraded": narrative_result.degraded,
    }


@router.post("/propose-schema", response_model=SchemaInferenceProposal)
async def propose_schema(
    file: UploadFile,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> SchemaInferenceProposal:
    """Stage 1: propose column roles for a predictions CSV. Proposal only — not confirmed."""
    settings = get_settings()
    client = build_default_client_or_none(settings.groq_api_key, settings.gemini_api_key)
    if client is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Schema inference requires an LLM provider; none is configured "
            "(set GROQ_API_KEY or GEMINI_API_KEY).",
        )
    if not await has_budget(db, settings.agent_daily_llm_call_cap):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Daily LLM call budget exhausted; try again tomorrow.",
        )
    data = await read_predictions_csv(
        file, max_mb=settings.agent_max_upload_mb, max_rows=settings.agent_max_rows
    )
    try:
        proposal = await asyncio.to_thread(infer_schema, client, data)
    except LLMCallError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Schema inference is temporarily unavailable (all LLM providers failed). "
            "Try again shortly.",
        ) from exc
    await record_llm_call(db)
    return proposal


@router.post("/run-audit", status_code=status.HTTP_201_CREATED)
async def run_audit_endpoint(
    file: UploadFile,
    proposal_json: str = Form(
        ..., description="JSON-encoded, human-confirmed SchemaInferenceProposal"
    ),
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> dict:
    """Stages 2-4: run the confirmed plan and persist only the derived scorecard/narrative.

    ``proposal_json`` must be the (possibly human-edited) result of ``/propose-schema`` —
    this endpoint treats it as already confirmed; it never re-derives or second-guesses it.
    Sent as a form field (not a JSON body) because it accompanies a file upload.
    """
    try:
        proposal = SchemaInferenceProposal.model_validate_json(proposal_json)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid proposal: {exc}") from exc

    settings = get_settings()
    data = await read_predictions_csv(
        file, max_mb=settings.agent_max_upload_mb, max_rows=settings.agent_max_rows
    )
    return await _run_and_persist(proposal, data, db, user)


class DemoDatasetOut(BaseModel):
    key: str
    title: str
    blurb: str
    proposal: SchemaInferenceProposal


class DemoAuditRequest(BaseModel):
    dataset_key: str
    proposal: SchemaInferenceProposal | None = None


@lru_cache(maxsize=8)
def _load_demo_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(Path(csv_path))


@router.get("/demo-datasets", response_model=list[DemoDatasetOut])
async def demo_datasets_list() -> list[DemoDatasetOut]:
    """Canned datasets shipped with the app — the primary, upload-free demo path."""
    return [
        DemoDatasetOut(key=d.key, title=d.title, blurb=d.blurb, proposal=d.proposal)
        for d in list_demo_datasets()
    ]


@router.post("/demo-audit", status_code=status.HTTP_201_CREATED)
async def demo_audit_endpoint(
    body: DemoAuditRequest,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> dict:
    """Run Stages 2-4 against a canned dataset. Skips Stage 1 (LLM schema inference) —
    the proposal is hand-verified and checked into ``agent/demo_datasets.py`` — but a
    caller may still pass an edited ``proposal``, preserving the human-confirmation step.
    """
    dataset = get_demo_dataset(body.dataset_key)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown demo dataset '{body.dataset_key}'")
    proposal = body.proposal or dataset.proposal
    data = _load_demo_dataframe(str(dataset.csv_path))
    return await _run_and_persist(proposal, data, db, user)


class AgentRecordSummary(BaseModel):
    record_id: str
    created_at: str
    outcome_column: str
    composite_score: float | None
    composite_band: str | None
    grounded: bool
    degraded: bool


@router.get("/records", response_model=list[AgentRecordSummary])
async def list_records(
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> list[AgentRecordSummary]:
    """Every agent audit record the caller owns, most recent first.

    Without this, a completed audit is a dead end: there is no way back to a report
    once the user navigates away from it. Capped at 50 — a personal history list, not
    a paginated archive.
    """
    rows = (
        (
            await db.execute(
                select(AgentAuditRecord)
                .where(AgentAuditRecord.owner_id == user.id)
                .order_by(AgentAuditRecord.created_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    return [
        AgentRecordSummary(
            record_id=str(r.id),
            created_at=r.created_at.isoformat(),
            outcome_column=r.outcome_column,
            composite_score=r.scorecard.get("composite_score"),
            composite_band=r.scorecard.get("composite_band"),
            grounded=r.grounded,
            degraded=r.degraded,
        )
        for r in rows
    ]


@router.get("/records/{record_id}")
async def get_record(
    record_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> dict:
    record = await _owned_record(record_id, user, db)
    return {
        "record_id": str(record.id),
        "plan": record.plan,
        "scorecard": record.scorecard,
        "narrative": record.narrative,
        "mitigation_summary": record.mitigation_summary,
        "grounded": record.grounded,
        "degraded": record.degraded,
        "created_at": record.created_at.isoformat(),
    }


@router.post("/records/{record_id}/ask")
async def ask_about_record(
    record_id: UUID,
    body: AskRequest,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> dict:
    """Stage 5: answer a question grounded only in the stored record's scorecard JSON."""
    record = await _owned_record(record_id, user, db)
    settings = get_settings()
    client = build_default_client_or_none(settings.groq_api_key, settings.gemini_api_key)
    if client is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Q&A requires an LLM provider; none is configured "
            "(set GROQ_API_KEY or GEMINI_API_KEY).",
        )
    if not await has_budget(db, settings.agent_daily_llm_call_cap):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Daily LLM call budget exhausted; try again tomorrow.",
        )
    try:
        result = await asyncio.to_thread(ask, client, record.scorecard, body.question)
    except LLMCallError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Q&A is temporarily unavailable (all LLM providers failed). Try again shortly.",
        ) from exc
    await record_llm_call(db)
    return {"answer": result.answer, "grounded": result.grounded, "degraded": result.degraded}
