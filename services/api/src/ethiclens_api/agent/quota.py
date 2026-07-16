"""Daily LLM call budget: a hard stop with graceful degradation, not an error page.

Checked before every LLM call in the agent router. Once the daily cap is hit, callers
should degrade (numeric-only narrative, "Q&A unavailable", 429 on schema inference)
rather than let costs run past a free-tier quota.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.db import engine
from ethiclens_api.models import UsageCounter


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def llm_calls_remaining(db: AsyncSession, daily_cap: int) -> int:
    """How many more LLM calls are allowed today (never negative)."""
    counter = await db.get(UsageCounter, _today())
    used = counter.llm_calls if counter is not None else 0
    return max(0, daily_cap - used)


async def record_llm_call(db: AsyncSession) -> None:
    """Atomically increment today's counter. Call once per actual LLM call that goes out.

    A read-modify-write (``db.get`` -> ``+= 1`` -> commit) loses increments under
    concurrency: two requests can both read the same value and one increment vanishes,
    silently under-counting usage against the daily cap. An upsert makes the increment
    a single statement the database serializes, so no update is lost. Dialect-specific
    because SQLAlchemy's ``ON CONFLICT DO UPDATE`` construct isn't generic SQL — tests
    run on SQLite, production on Postgres.
    """
    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    stmt = insert(UsageCounter).values(day=_today(), llm_calls=1)
    stmt = stmt.on_conflict_do_update(
        index_elements=[UsageCounter.day],
        set_={"llm_calls": UsageCounter.llm_calls + 1},
    )
    await db.execute(stmt)
    await db.commit()


async def has_budget(db: AsyncSession, daily_cap: int) -> bool:
    return await llm_calls_remaining(db, daily_cap) > 0
