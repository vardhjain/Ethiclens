"""Async database engine, session factory, and the declarative base."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from ethiclens_api.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_settings = get_settings()
# SQLite (dev/test) gets NullPool: a fresh DBAPI connection per checkout, none held
# across checkouts. The default pool (AsyncAdaptedQueuePool) keeps its free-connection
# tracking in an asyncio.Queue that's lazily bound to whichever event loop first uses
# it — fine for a long-lived production process, but this engine is a module-level
# singleton reused across pytest-asyncio's function-scoped tests, each with its own
# fresh event loop. Once a test exercises real pool contention (concurrent checkouts
# exceeding the idle pool, e.g. the LLM-quota atomicity test's 20-way asyncio.gather)
# the Queue's blocking-wait path runs from a *different* loop than the one it was
# created on and raises "RuntimeError: <Queue ...> is bound to a different event
# loop" — reproduced deterministically standalone, and the likely cause of the
# intermittent StaleDataError seen in the full suite (a subtler corruption from the
# same cross-loop reuse, not always surfacing as the clean RuntimeError). NullPool
# sidesteps the whole class of bug for SQLite; Postgres in production has no
# per-test loop churn, so the default pool there is unaffected and still desirable.
_poolclass = NullPool if _settings.database_url.startswith("sqlite") else None
engine = create_async_engine(_settings.database_url, future=True, poolclass=_poolclass)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transactional session."""
    async with SessionLocal() as session:
        yield session


async def create_all() -> None:
    """Create tables directly (used for tests / first-run dev; prod uses Alembic)."""
    from ethiclens_api import models  # noqa: F401  (register mappers)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
