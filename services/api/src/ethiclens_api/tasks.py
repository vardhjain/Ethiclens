"""Task execution: eager (dev/tests) or queued via arq (production).

Long-running audits return ``202 Accepted`` immediately and run on a worker; the
client polls ``/status``. Setting ``EAGER_TASKS=true`` runs them in-process so the
whole system works (and is testable) with no Redis/worker — the same code path,
just synchronous.
"""

from __future__ import annotations

from typing import ClassVar
from uuid import UUID

from ethiclens_api.audit_service import execute_audit
from ethiclens_api.config import get_settings
from ethiclens_api.mitigation_service import apply_mitigation


async def _pool():  # pragma: no cover - requires Redis
    from arq import create_pool
    from arq.connections import RedisSettings

    return await create_pool(RedisSettings.from_dsn(get_settings().redis_url))


async def enqueue_audit(session_id: UUID) -> str | None:
    """Run or queue an audit. Returns an arq job id (queued) or ``None`` (eager)."""
    if get_settings().eager_tasks:
        await execute_audit(session_id)
        return None
    pool = await _pool()  # pragma: no cover
    job = await pool.enqueue_job("run_audit_task", str(session_id))  # pragma: no cover
    return job.job_id if job else None  # pragma: no cover


async def enqueue_mitigation(
    parent_id: UUID, strategy: str, group_label: str | None
) -> UUID | None:
    """Run or queue a mitigation re-audit. Returns the child session id when eager."""
    if get_settings().eager_tasks:
        return await apply_mitigation(parent_id, strategy, group_label)
    pool = await _pool()  # pragma: no cover
    # pragma: no cover
    await pool.enqueue_job("reaudit_task", str(parent_id), strategy, group_label)
    return None  # pragma: no cover


# --- arq worker entrypoints -----------------------------------------------


async def run_audit_task(ctx: dict, session_id: str) -> None:  # pragma: no cover
    await execute_audit(UUID(session_id))


async def reaudit_task(  # pragma: no cover
    ctx: dict, parent_id: str, strategy: str, group_label: str | None
) -> None:
    await apply_mitigation(UUID(parent_id), strategy, group_label)


class WorkerSettings:  # pragma: no cover - production worker config
    """``arq worker ethiclens_api.tasks.WorkerSettings``."""

    functions: ClassVar = [run_audit_task, reaudit_task]

    @staticmethod
    def redis_settings():
        from arq.connections import RedisSettings

        return RedisSettings.from_dsn(get_settings().redis_url)
