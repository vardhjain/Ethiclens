from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.agent.quota import has_budget, llm_calls_remaining, record_llm_call

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
