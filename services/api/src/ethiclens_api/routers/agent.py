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
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.agent.audit_planner import plan_audit
from ethiclens_api.agent.csv_ingest import read_predictions_csv
from ethiclens_api.agent.executor import execute_plan, measure_top_mitigation
from ethiclens_api.agent.llm_client import build_default_client_or_none
from ethiclens_api.agent.narrative import build_scorecard, generate_narrative, unavailable_result
from ethiclens_api.agent.qa import ask
from ethiclens_api.agent.quota import has_budget, record_llm_call
from ethiclens_api.agent.schema_inference import infer_schema
from ethiclens_api.agent.schemas import AskRequest, SchemaInferenceProposal
from ethiclens_api.config import get_settings
from ethiclens_api.db import get_session
from ethiclens_api.models import AgentAuditRecord, UserAccount
from ethiclens_api.security import get_current_user

router = APIRouter(prefix="/api/agent", tags=["agent"])


async def _owned_record(
    record_id: UUID, user: UserAccount, db: AsyncSession
) -> AgentAuditRecord:
    record = await db.get(AgentAuditRecord, record_id)
    if record is None or record.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audit record not found")
    return record


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
    proposal = infer_schema(client, data)
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

    missing = [
        col
        for col in [proposal.outcome_column, *proposal.protected_attribute_columns]
        if col not in data.columns
    ]
    if missing:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Columns not found in CSV: {missing}")

    plan = plan_audit(proposal, data)
    result = execute_plan(plan, data, proposal.outcome_direction)
    # Real, held-out-measured before/after — only possible while the CSV is still in
    # memory (never persisted); unavailable without a score column + true labels.
    measured_mitigation = measure_top_mitigation(plan, data, proposal.outcome_direction, result)
    scorecard, _recommendations = build_scorecard(result, plan, measured_mitigation)

    client = build_default_client_or_none(settings.groq_api_key, settings.gemini_api_key)
    if client is not None and await has_budget(db, settings.agent_daily_llm_call_cap):
        narrative_result = generate_narrative(client, scorecard)
        await record_llm_call(db)
    else:
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
    result = ask(client, record.scorecard, body.question)
    await record_llm_call(db)
    return {"answer": result.answer, "grounded": result.grounded, "degraded": result.degraded}
