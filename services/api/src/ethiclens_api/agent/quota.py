"""Daily LLM call budget: a hard stop with graceful degradation, not an error page.

Checked before every LLM call in the agent router. Once the daily cap is hit, callers
should degrade (numeric-only narrative, "Q&A unavailable", 429 on schema inference)
rather than let costs run past a free-tier quota.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.models import UsageCounter


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


async def llm_calls_remaining(db: AsyncSession, daily_cap: int) -> int:
    """How many more LLM calls are allowed today (never negative)."""
    counter = await db.get(UsageCounter, _today())
    used = counter.llm_calls if counter is not None else 0
    return max(0, daily_cap - used)


async def record_llm_call(db: AsyncSession) -> None:
    """Increment today's counter. Call once per actual LLM call that goes out."""
    today = _today()
    counter = await db.get(UsageCounter, today)
    if counter is None:
        counter = UsageCounter(day=today, llm_calls=0)
        db.add(counter)
    counter.llm_calls += 1
    await db.commit()


async def has_budget(db: AsyncSession, daily_cap: int) -> bool:
    return await llm_calls_remaining(db, daily_cap) > 0
