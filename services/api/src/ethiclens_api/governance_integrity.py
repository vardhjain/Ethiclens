"""Deterministic, reproducible SHA-256 fingerprints for approved audit snapshots.

These detect changes to the approved record *relative to a previously exported
fingerprint*. They are NOT a digital signature or an immutable external log:
a database administrator who can rewrite both the data and stored fingerprint
can conceal changes. Export the digest to an independent system for stronger
tamper evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.models import AuditSession, FairnessMetric, MitigationLog, ProtectedAttribute


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        # SQLite strips timezones; timestamps are stored as UTC in this app.
        return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _columns(row: Any, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    changes = overrides or {}
    return {
        column.name: _canonical(changes.get(column.name, getattr(row, column.name)))
        for column in type(row).__table__.columns
        if column.name != "signoff_digest"
    }


async def snapshot_digest(
    db: AsyncSession, obj: AuditSession, changes: dict[str, Any] | None = None
) -> str:
    """Hash all session fields and its protected attributes, metrics, mitigations."""
    related: dict[str, list[dict[str, Any]]] = {}
    for model in (ProtectedAttribute, FairnessMetric, MitigationLog):
        rows = (
            await db.execute(
                select(model).where(model.session_id == obj.id).order_by(model.id)
            )
        ).scalars().all()
        related[model.__tablename__] = [_columns(row) for row in rows]

    snapshot = {"session": _columns(obj, changes), "related": related}
    data = json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
