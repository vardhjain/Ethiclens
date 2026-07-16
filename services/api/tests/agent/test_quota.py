from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.agent.quota import has_budget, llm_calls_remaining, record_llm_call
from ethiclens_api.db import SessionLocal

pytestmark = pytest.mark.asyncio


async def test_fresh_day_has_full_budget(db_session: AsyncSession):
    assert await llm_calls_remaining(db_session, daily_cap=5) == 5
    assert await has_budget(db_session, daily_cap=5) is True


async def test_record_llm_call_decrements_remaining(db_session: AsyncSession):
    await record_llm_call(db_session)
    await record_llm_call(db_session)
    assert await llm_calls_remaining(db_session, daily_cap=5) == 3


async def test_has_budget_false_once_cap_reached(db_session: AsyncSession):
    for _ in range(3):
        await record_llm_call(db_session)
    assert await has_budget(db_session, daily_cap=3) is False
    assert await llm_calls_remaining(db_session, daily_cap=3) == 0


async def test_record_llm_call_is_atomic_under_concurrency(db_session: AsyncSession):
    """A read-modify-write increment loses updates when requests race; each bump here
    uses its own session (the same shape as one-session-per-request in production), so
    this only passes if the increment is a genuinely atomic database operation."""
    n = 20

    async def bump() -> None:
        async with SessionLocal() as session:
            await record_llm_call(session)

    await asyncio.gather(*[bump() for _ in range(n)])

    assert await llm_calls_remaining(db_session, daily_cap=n) == 0
    assert await llm_calls_remaining(db_session, daily_cap=n + 5) == 5
