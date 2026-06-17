"""Fairness Scorecard PDF endpoints (FR-007)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.db import get_session
from ethiclens_api.models import AuditSession, FairnessMetric, UserAccount
from ethiclens_api.security import get_current_user
from fairness_core.reporting import generate_scorecard_pdf
from fairness_core.reporting.scorecard import ScorecardMeta
from fairness_core.types import (
    AuditResult,
    ConfidenceInterval,
    GroupAuditResult,
    MetricResult,
)

router = APIRouter(prefix="/api/sessions", tags=["reports"])


async def _owned(session_id: UUID, user: UserAccount, db: AsyncSession) -> AuditSession:
    obj = await db.get(AuditSession, session_id)
    if obj is None or obj.owner_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return obj


async def _reconstruct(session_id: UUID, obj: AuditSession, db: AsyncSession) -> AuditResult:
    rows = list(
        (
            await db.execute(select(FairnessMetric).where(FairnessMetric.session_id == session_id))
        ).scalars()
    )
    by_group: dict[str, GroupAuditResult] = {}
    for m in rows:
        g = by_group.get(m.group_label)
        if g is None:
            g = GroupAuditResult(
                attribute=m.group_label.split(":")[0],
                group_label=m.group_label,
                privileged_value="",
                unprivileged_value=m.group_label.split(":")[-1],
                n_privileged=0,
                n_unprivileged=m.n or 0,
                metrics={},
            )
            by_group[m.group_label] = g
        ci = (
            ConfidenceInterval(m.ci_low, m.ci_high)
            if m.ci_low is not None and m.ci_high is not None
            else None
        )
        g.metrics[m.metric_type] = MetricResult(
            name=m.metric_type, value=m.value, ci=ci, classification=m.classification, n=m.n
        )
        if m.metric_type == "disparate_impact" and m.value is not None and m.value < 0.80:
            g.flagged = True
    return AuditResult(
        composite_score=obj.composite_score,
        composite_band=obj.composite_band,
        min_di=obj.min_di,
        groups=list(by_group.values()),
    )


@router.post("/{session_id}/report", status_code=status.HTTP_202_ACCEPTED)
async def request_report(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> dict:
    await _owned(session_id, user, db)
    return {"report_job_id": str(session_id), "status": "ready"}


@router.get("/{session_id}/report")
async def download_report(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> Response:
    obj = await _owned(session_id, user, db)
    audit = await _reconstruct(session_id, obj, db)
    if not audit.groups:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No audit results to report yet")
    pdf = generate_scorecard_pdf(
        audit,
        ScorecardMeta(
            model_name=obj.name,
            dataset=obj.dataset,
            session_id=str(session_id),
            audit_date=(obj.completed_at or obj.created_at).date().isoformat(),
            seed=obj.seed,
        ),
    )
    filename = f"EthicLens_Scorecard_{session_id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
