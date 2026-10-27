"""Daily LLM call budget: a hard stop with graceful degradation, not an error page.

Each outbound attempt is reserved atomically before a provider call. Once the cap
is hit, callers should degrade (numeric-only narrative, "Q&A unavailable", 429 on schema inference)
rather than let costs run past a free-tier quota.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.db import engine
from ethiclens_api.models import UsageCounter, UserUsageCounter

_log = logging.getLogger("ethiclens.agent.quota")


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def llm_calls_remaining(db: AsyncSession, daily_cap: int) -> int:
    """How many more LLM calls are allowed today (never negative)."""
    counter = await db.get(UsageCounter, _today())
    used = counter.llm_calls if counter is not None else 0
    return max(0, daily_cap - used)


async def user_llm_calls_remaining(db: AsyncSession, user_id: UUID, daily_cap: int) -> int:
    """How many more LLM calls this user may make today (never negative)."""
    counter = await db.get(UserUsageCounter, (user_id, _today()))
    used = counter.llm_calls if counter is not None else 0
    return max(0, daily_cap - used)


async def reserve_llm_attempt(
    db: AsyncSession, user_id: UUID, global_cap: int, user_cap: int
) -> bool:
    """Atomically reserve one outbound provider attempt against BOTH daily caps.

    The two conditional UPSERTs run inside the same database transaction. If
    either cap has been reached, a rollback undoes the other reservation. A
    competing request cannot oversubscribe the counters: PostgreSQL/SQLite
    serialize the conflicting update and its WHERE predicate. Reserve before
    invoking the SDK, including on retries, fallback and provider errors.

    An attempt is counted even when the upstream provider fails, because the
    request may already have reached a paid endpoint. No refunds are issued.
    """
    if global_cap <= 0 or user_cap <= 0:
        return False

    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    today = _today()
    global_stmt = insert(UsageCounter).values(day=today, llm_calls=1)
    global_stmt = global_stmt.on_conflict_do_update(
        index_elements=[UsageCounter.day],
        set_={"llm_calls": UsageCounter.llm_calls + 1},
        where=UsageCounter.llm_calls < global_cap,
    ).returning(UsageCounter.llm_calls)
    user_stmt = insert(UserUsageCounter).values(user_id=user_id, day=today, llm_calls=1)
    user_stmt = user_stmt.on_conflict_do_update(
        index_elements=[UserUsageCounter.user_id, UserUsageCounter.day],
        set_={"llm_calls": UserUsageCounter.llm_calls + 1},
        where=UserUsageCounter.llm_calls < user_cap,
    ).returning(UserUsageCounter.llm_calls)

    try:
        global_reserved = (await db.execute(global_stmt)).scalar_one_or_none()
        if global_reserved is None:
            await db.rollback()
            _log.warning("Global LLM attempt quota exhausted (cap=%d)", global_cap)
            return False

        user_reserved = (await db.execute(user_stmt)).scalar_one_or_none()
        if user_reserved is None:
            await db.rollback()
            _log.warning("Per-user LLM attempt quota exhausted (cap=%d)", user_cap)
            return False

        await db.commit()
    except Exception:
        await db.rollback()
        raise
    return True


async def record_llm_call(db: AsyncSession, user_id: UUID) -> None:
    """Atomically increment today's global and per-user counters.

    Legacy accounting helper kept for compatibility with existing callers/tests.
    Production agent calls now use reserve_llm_attempt for capped reservations.
    Call once per actual LLM call that goes out. A read-modify-write (``db.get`` ->
    ``+= 1`` -> commit) loses increments under concurrency: two requests can both read
    the same value and one increment vanishes, silently under-counting usage against
    the daily cap. An upsert makes each increment a single statement the database
    serializes, so no update is lost. Dialect-specific because SQLAlchemy's
    ``ON CONFLICT DO UPDATE`` construct isn't generic SQL — tests run on SQLite,
    production on Postgres.
    """
    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    today = _today()

    global_stmt = insert(UsageCounter).values(day=today, llm_calls=1)
    global_stmt = global_stmt.on_conflict_do_update(
        index_elements=[UsageCounter.day],
        set_={"llm_calls": UsageCounter.llm_calls + 1},
    )
    await db.execute(global_stmt)

    user_stmt = insert(UserUsageCounter).values(user_id=user_id, day=today, llm_calls=1)
    user_stmt = user_stmt.on_conflict_do_update(
        index_elements=[UserUsageCounter.user_id, UserUsageCounter.day],
        set_={"llm_calls": UserUsageCounter.llm_calls + 1},
    )
    await db.execute(user_stmt)

    await db.commit()


async def has_budget(db: AsyncSession, daily_cap: int) -> bool:
    remaining = await llm_calls_remaining(db, daily_cap)
    if remaining <= 0:
        _log.warning("Daily LLM call budget exhausted (cap=%d)", daily_cap)
    return remaining > 0


async def has_user_budget(db: AsyncSession, user_id: UUID, daily_cap: int) -> bool:
    remaining = await user_llm_calls_remaining(db, user_id, daily_cap)
    if remaining <= 0:
        _log.warning(
            "Per-user daily LLM call budget exhausted (user=%s, cap=%d)", user_id, daily_cap
        )
    return remaining > 0
