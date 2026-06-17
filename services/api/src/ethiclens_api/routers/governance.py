"""Governance workflow (FR-008/010/011): escalation, sign-off, immutable lock."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.db import get_session
from ethiclens_api.models import AuditSession, SessionStatus, UserAccount, UserRole
from ethiclens_api.schemas import SessionOut, SignOffRequest
from ethiclens_api.security import get_current_user, require_role

router = APIRouter(prefix="/api/sessions", tags=["governance"])
_log = logging.getLogger("ethiclens.governance")


async def _owned(session_id: UUID, user: UserAccount, db: AsyncSession) -> AuditSession:
    obj = await db.get(AuditSession, session_id)
    if obj is None or (obj.owner_id != user.id and user.role != UserRole.ADMIN):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return obj


@router.post("/{session_id}/escalate", response_model=SessionOut)
async def escalate(
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(get_current_user),
) -> AuditSession:
    obj = await _owned(session_id, user, db)
    if obj.locked:
        raise HTTPException(status.HTTP_409_CONFLICT, "Session is locked")
    obj.status = SessionStatus.ESCALATED
    await db.commit()
    await db.refresh(obj)
    _log.info("Session %s escalated to engineering by %s", session_id, user.email)
    return obj


@router.post("/{session_id}/sign-off", response_model=SessionOut)
async def sign_off(
    session_id: UUID,
    body: SignOffRequest,
    db: AsyncSession = Depends(get_session),
    user: UserAccount = Depends(require_role(UserRole.GOVERNANCE_APPROVER)),
) -> AuditSession:
    obj = await db.get(AuditSession, session_id)
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if obj.locked:
        raise HTTPException(status.HTTP_409_CONFLICT, "Session already signed off and locked")

    obj.status = SessionStatus.SIGNED_OFF
    obj.locked = True
    obj.signed_off_by = user.id
    obj.signed_off_at = datetime.now(UTC)
    obj.approver_note = body.note
    await db.commit()
    await db.refresh(obj)
    # FR-011: deployment-clearance notification (a full notifier is roadmap).
    _log.info("DEPLOYMENT CLEARANCE: session %s signed off by %s", session_id, user.email)
    return obj
