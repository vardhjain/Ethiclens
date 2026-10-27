"""Governance approval: separation of duties, finalized state, and integrity checks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.db import get_session
from ethiclens_api.governance_integrity import snapshot_digest
from ethiclens_api.models import (
    AuditSession,
    FairnessMetric,
    SessionStatus,
    UserAccount,
    UserRole,
)
from ethiclens_api.schemas import IntegrityOut, SessionOut, SignOffRequest
from ethiclens_api.security import get_current_user, require_role

router = APIRouter(prefix="/api/sessions", tags=["governance"])
_log = logging.getLogger("ethiclens.governance")
_FINAL_STATES = (SessionStatus.COMPLETED, SessionStatus.FLAGGED)


@router.post("/{session_id}/sign-off", response_model=SessionOut)
async def sign_off(
    session_id: UUID,
    body: SignOffRequest,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(require_role(UserRole.GOVERNANCE_APPROVER)),
) -> AuditSession:
    # PostgreSQL holds a row lock until commit; the conditional UPDATE also
    # protects SQLite and rejects a second approver racing for the same audit.
    obj = await db.scalar(
        select(AuditSession).where(AuditSession.id == session_id).with_for_update()
    )
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if obj.owner_id == user.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Self-approval is not allowed")
    if obj.locked:
        raise HTTPException(status.HTTP_409_CONFLICT, "Session already signed off and locked")
    if obj.status not in _FINAL_STATES or obj.completed_at is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Only completed audits can be signed off")

    has_metrics = await db.scalar(
        select(FairnessMetric.id).where(FairnessMetric.session_id == session_id).limit(1)
    )
    if has_metrics is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Cannot approve an audit without metrics")

    approved_at = datetime.now(UTC)
    changes = {
        "status": SessionStatus.SIGNED_OFF,
        "locked": True,
        "signed_off_by": user.id,
        "signed_off_at": approved_at,
        "approver_note": body.note,
    }
    digest = await snapshot_digest(db, obj, changes)
    result = await db.execute(
        update(AuditSession)
        .where(
            AuditSession.id == session_id,
            AuditSession.locked.is_(False),
            AuditSession.status.in_(_FINAL_STATES),
            AuditSession.completed_at.is_not(None),
        )
        .values(**changes, signoff_digest=digest)
    )
    if result.rowcount != 1:
        await db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Audit changed during approval")
    await db.commit()
    await db.refresh(obj)
    _log.info("DEPLOYMENT CLEARANCE: session %s signed off by %s", session_id, user.email)
    return obj


@router.get("/{session_id}/integrity", response_model=IntegrityOut)
async def check_integrity(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> IntegrityOut:
    obj = await db.get(AuditSession, session_id)
    if obj is None or (
        obj.owner_id != user.id and user.role not in (UserRole.GOVERNANCE_APPROVER, UserRole.ADMIN)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if not obj.locked or obj.status != SessionStatus.SIGNED_OFF:
        return IntegrityOut(status="not_signed", expected_digest=None, current_digest=None)
    if not obj.signoff_digest:
        # Older approved rows have no prior fingerprint: don't silently re-baseline them.
        return IntegrityOut(status="legacy_unverifiable", expected_digest=None, current_digest=None)
    current = await snapshot_digest(db, obj)
    return IntegrityOut(
        status="verified" if current == obj.signoff_digest else "mismatch",
        expected_digest=obj.signoff_digest,
        current_digest=current,
    )
