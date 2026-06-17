"""Audit session lifecycle (FR-002/003/004/005/009)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.db import get_session
from ethiclens_api.models import (
    AuditSession,
    FairnessMetric,
    ProtectedAttribute,
    SessionStatus,
    UserAccount,
)
from ethiclens_api.schemas import (
    MetricsResponse,
    MitigateRequest,
    RecommendationOut,
    RecommendationsResponse,
    SessionCreate,
    SessionOut,
    SessionStatusOut,
)
from ethiclens_api.security import get_current_user
from ethiclens_api.tasks import enqueue_audit, enqueue_mitigation
from fairness_core import get_recommendations
from fairness_core.types import AuditResult, GroupAuditResult, MetricName, MetricResult

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_PROGRESS = {
    SessionStatus.DRAFT: 0.0,
    SessionStatus.QUEUED: 0.1,
    SessionStatus.RUNNING: 0.5,
    SessionStatus.COMPLETED: 1.0,
    SessionStatus.FLAGGED: 1.0,
    SessionStatus.FAILED: 1.0,
    SessionStatus.ESCALATED: 1.0,
    SessionStatus.SIGNED_OFF: 1.0,
}


async def _owned(session_id: UUID, user: UserAccount, db: AsyncSession) -> AuditSession:
    obj = await db.get(AuditSession, session_id)
    if obj is None or obj.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return obj


@router.post("/create", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> AuditSession:
    obj = AuditSession(
        owner_id=user.id,
        model_id=body.model_id,
        name=body.name,
        dataset=body.dataset,
        target_column=body.target,
        feature_columns=body.feature_columns,
        profile_config={"n_profiles": body.n_profiles},
        seed=body.seed,
        status=SessionStatus.DRAFT,
    )
    for pa in body.protected_attributes:
        obj.protected_attributes.append(
            ProtectedAttribute(
                name=pa.name,
                privileged_value=pa.privileged_value,
                unprivileged_values=pa.unprivileged_values,
            )
        )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


@router.post("/{session_id}/run", status_code=status.HTTP_202_ACCEPTED)
async def run_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> dict:
    obj = await _owned(session_id, user, db)
    if obj.locked:
        raise HTTPException(status.HTTP_409_CONFLICT, "Session is locked")
    obj.status = SessionStatus.QUEUED
    await db.commit()
    job_id = await enqueue_audit(session_id)
    fresh = await db.get(AuditSession, session_id)
    return {"session_id": str(session_id), "status": fresh.status.value, "job_id": job_id}


@router.get("/{session_id}/status", response_model=SessionStatusOut)
async def session_status(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> SessionStatusOut:
    obj = await _owned(session_id, user, db)
    return SessionStatusOut(
        status=obj.status.value, progress=_PROGRESS.get(obj.status, 0.0), error=obj.error
    )


@router.get("/{session_id}/metrics", response_model=MetricsResponse)
async def session_metrics(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> MetricsResponse:
    obj = await _owned(session_id, user, db)
    rows = list(
        (
            await db.execute(select(FairnessMetric).where(FairnessMetric.session_id == session_id))
        ).scalars()
    )
    has_labels = any(
        m.metric_type == MetricName.EQUALIZED_ODDS.value and m.value is not None for m in rows
    )
    return MetricsResponse(
        session_id=session_id,
        composite_score=obj.composite_score,
        composite_band=obj.composite_band,
        min_di=obj.min_di,
        has_labels=has_labels,
        metrics=rows,
    )


@router.get("/{session_id}/recommendations", response_model=RecommendationsResponse)
async def session_recommendations(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> RecommendationsResponse:
    obj = await _owned(session_id, user, db)
    rows = list(
        (
            await db.execute(
                select(FairnessMetric).where(
                    FairnessMetric.session_id == session_id,
                    FairnessMetric.metric_type == MetricName.DISPARATE_IMPACT.value,
                )
            )
        ).scalars()
    )
    groups = [
        GroupAuditResult(
            attribute=m.group_label.split(":")[0],
            group_label=m.group_label,
            privileged_value="",
            unprivileged_value=m.group_label.split(":")[-1],
            n_privileged=0,
            n_unprivileged=m.n or 0,
            metrics={MetricName.DISPARATE_IMPACT.value: MetricResult(name="di", value=m.value)},
            flagged=(m.value is not None and m.value < 0.80),
        )
        for m in rows
    ]
    audit = AuditResult(
        composite_score=obj.composite_score,
        composite_band=obj.composite_band,
        min_di=obj.min_di,
        groups=groups,
    )
    recs = get_recommendations(audit)
    payload = {
        g: [
            RecommendationOut(
                rank=r.rank,
                strategy=r.strategy,
                strategy_name=r.strategy_name,
                description=r.description,
                estimated_di_improvement=r.estimated_di_improvement,
                stage=r.stage,
                measured=r.measured,
            )
            for r in rs
        ]
        for g, rs in recs.items()
    }
    return RecommendationsResponse(flagged=bool(payload), recommendations=payload)


@router.post("/{session_id}/mitigate", status_code=status.HTTP_202_ACCEPTED)
async def mitigate_session(
    session_id: UUID,
    body: MitigateRequest,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> dict:
    await _owned(session_id, user, db)
    child_id = await enqueue_mitigation(session_id, body.strategy, body.group_label)
    return {"result_session_id": str(child_id) if child_id else None, "status": "QUEUED"}


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> list[AuditSession]:
    result = await db.execute(
        select(AuditSession)
        .where(AuditSession.owner_id == user.id)
        .order_by(AuditSession.created_at.desc())
    )
    return list(result.scalars())


@router.get("/{session_id}", response_model=SessionOut)
async def get_session_detail(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> AuditSession:
    return await _owned(session_id, user, db)
