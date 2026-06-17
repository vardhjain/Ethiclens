"""Authentication & RBAC basics."""

from __future__ import annotations

from httpx import AsyncClient


async def test_register_login_me(client: AsyncClient, auth) -> None:
    headers = await auth(client, "a@example.com")
    me = await client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["email"] == "a@example.com"


async def test_duplicate_register_conflict(client: AsyncClient, auth) -> None:
    await auth(client, "dup@example.com")
    resp = await client.post(
        "/api/auth/register",
        json={"email": "dup@example.com", "password": "password123"},
    )
    assert resp.status_code == 409


async def test_wrong_password_rejected(client: AsyncClient, auth) -> None:
    await auth(client, "x@example.com")
    resp = await client.post(
        "/api/auth/login", data={"username": "x@example.com", "password": "wrong"}
    )
    assert resp.status_code == 401


async def test_protected_route_requires_token(client: AsyncClient) -> None:
    assert (await client.get("/api/sessions")).status_code == 401
