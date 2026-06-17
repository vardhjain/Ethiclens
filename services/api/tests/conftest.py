"""Async test harness: a fresh SQLite schema per test, eager task execution."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

# Configure the app for tests BEFORE importing it (settings are cached).
_DB_FILE = Path(tempfile.gettempdir()) / "ethiclens_api_test.db"
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_DB_FILE.as_posix()}")
os.environ.setdefault("EAGER_TASKS", "true")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("MODEL_STORAGE_DIR", str(Path(tempfile.gettempdir()) / "ethiclens_models"))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from ethiclens_api.db import Base, engine  # noqa: E402
from ethiclens_api.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _auth_headers(
    client: AsyncClient, email: str = "eng@example.com", role: str = "ml_engineer"
) -> dict[str, str]:
    """Register (idempotently) + log in, returning an Authorization header."""
    await client.post(
        "/api/auth/register",
        json={"email": email, "password": "password123", "role": role, "full_name": "Test"},
    )
    resp = await client.post("/api/auth/login", data={"username": email, "password": "password123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth():
    """Expose the auth helper as a fixture (avoids cross-package test imports)."""
    return _auth_headers
