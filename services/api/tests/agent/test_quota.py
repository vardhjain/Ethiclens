from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ethiclens_api.agent.quota import (
    has_budget,
    has_user_budget,
    llm_calls_remaining,
    record_llm_call,
    reserve_llm_attempt,
    user_llm_calls_remaining,
)
from ethiclens_api.db import SessionLocal

pytestmark = pytest.mark.asyncio

_USER = uuid4()


async def test_fresh_day_has_full_budget(db_session: AsyncSession):
    assert await llm_calls_remaining(db_session, daily_cap=5) == 5
    assert await has_budget(db_session, daily_cap=5) is True


async def test_record_llm_call_decrements_remaining(db_session: AsyncSession):
    await record_llm_call(db_session, _USER)
    await record_llm_call(db_session, _USER)
    assert await llm_calls_remaining(db_session, daily_cap=5) == 3


async def test_has_budget_false_once_cap_reached(db_session: AsyncSession):
    for _ in range(3):
        await record_llm_call(db_session, _USER)
    assert await has_budget(db_session, daily_cap=3) is False
    assert await llm_calls_remaining(db_session, daily_cap=3) == 0


async def test_has_budget_false_logs_a_warning(db_session: AsyncSession, caplog):
    for _ in range(3):
        await record_llm_call(db_session, _USER)
    with caplog.at_level("WARNING", logger="ethiclens.agent.quota"):
        assert await has_budget(db_session, daily_cap=3) is False
    assert any("exhausted" in record.message for record in caplog.records)


async def test_record_llm_call_is_atomic_under_concurrency(db_session: AsyncSession):
    """A read-modify-write increment loses updates when requests race; each bump here
    uses its own session (the same shape as one-session-per-request in production), so
    this only passes if the increment is a genuinely atomic database operation."""
    n = 20

    async def bump() -> None:
        async with SessionLocal() as session:
            await record_llm_call(session, _USER)

    await asyncio.gather(*[bump() for _ in range(n)])

    assert await llm_calls_remaining(db_session, daily_cap=n) == 0
    assert await llm_calls_remaining(db_session, daily_cap=n + 5) == 5


async def test_fresh_user_has_full_budget(db_session: AsyncSession):
    assert await user_llm_calls_remaining(db_session, _USER, daily_cap=5) == 5
    assert await has_user_budget(db_session, _USER, daily_cap=5) is True


async def test_user_budget_decrements_independently_of_another_user(db_session: AsyncSession):
    other_user = uuid4()
    await record_llm_call(db_session, _USER)
    await record_llm_call(db_session, _USER)
    assert await user_llm_calls_remaining(db_session, _USER, daily_cap=5) == 3
    assert await user_llm_calls_remaining(db_session, other_user, daily_cap=5) == 5


async def test_has_user_budget_false_once_per_user_cap_reached(db_session: AsyncSession):
    for _ in range(3):
        await record_llm_call(db_session, _USER)
    assert await has_user_budget(db_session, _USER, daily_cap=3) is False
    # The global counter also advanced, but with a generous global cap this user's own
    # cap is what trips first — the two budgets are independent gates.
    assert await has_budget(db_session, daily_cap=1000) is True


async def test_record_llm_call_increments_both_global_and_user_counters(
    db_session: AsyncSession,
):
    await record_llm_call(db_session, _USER)
    assert await llm_calls_remaining(db_session, daily_cap=5) == 4
    assert await user_llm_calls_remaining(db_session, _USER, daily_cap=5) == 4


async def test_reservation_stops_at_both_caps(db_session: AsyncSession):
    assert await reserve_llm_attempt(db_session, _USER, global_cap=2, user_cap=10)
    assert await reserve_llm_attempt(db_session, _USER, global_cap=2, user_cap=10)
    assert not await reserve_llm_attempt(db_session, _USER, global_cap=2, user_cap=10)
    assert await llm_calls_remaining(db_session, daily_cap=2) == 0
    assert await user_llm_calls_remaining(db_session, _USER, daily_cap=10) == 8


async def test_user_denial_rolls_back_global_reservation(db_session: AsyncSession):
    assert await reserve_llm_attempt(db_session, _USER, global_cap=5, user_cap=1)
    assert not await reserve_llm_attempt(db_session, _USER, global_cap=5, user_cap=1)
    assert await llm_calls_remaining(db_session, daily_cap=5) == 4
    other = uuid4()
    assert await reserve_llm_attempt(db_session, other, global_cap=5, user_cap=1)
    assert await llm_calls_remaining(db_session, daily_cap=5) == 3


async def test_zero_cap_denies_without_creating_counters(db_session: AsyncSession):
    assert not await reserve_llm_attempt(db_session, _USER, global_cap=0, user_cap=9)
    assert not await reserve_llm_attempt(db_session, _USER, global_cap=9, user_cap=0)
    assert await llm_calls_remaining(db_session, daily_cap=9) == 9


async def test_atomic_reservation_never_oversubscribes_global_cap(db_session: AsyncSession):
    async def claim(user_id):
        async with SessionLocal() as session:
            return await reserve_llm_attempt(session, user_id, global_cap=7, user_cap=50)

    users = [uuid4() for _ in range(3)]
    results = await asyncio.gather(*(claim(users[i % 3]) for i in range(20)))
    assert sum(results) == 7
    assert await llm_calls_remaining(db_session, daily_cap=7) == 0
    total_users = sum(
        50 - await user_llm_calls_remaining(db_session, user, daily_cap=50)
        for user in users
    )
    assert total_users == 7


async def test_atomic_reservation_never_oversubscribes_user_cap(db_session: AsyncSession):
    async def claim():
        async with SessionLocal() as session:
            return await reserve_llm_attempt(session, _USER, global_cap=50, user_cap=4)

    results = await asyncio.gather(*(claim() for _ in range(12)))
    assert sum(results) == 4
    assert await llm_calls_remaining(db_session, daily_cap=50) == 46
    assert await user_llm_calls_remaining(db_session, _USER, daily_cap=4) == 0
