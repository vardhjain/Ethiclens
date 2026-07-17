"""Authentication & RBAC basics."""

from __future__ import annotations

from httpx import AsyncClient

from ethiclens_api.config import get_settings


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


async def test_login_rate_limited_after_too_many_attempts(client: AsyncClient, monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "auth_login_rate_limit", "2/minute")
    await client.post(
        "/api/auth/register",
        json={"email": "rl-login@example.com", "password": "password123"},
    )
    data = {"username": "rl-login@example.com", "password": "password123"}
    assert (await client.post("/api/auth/login", data=data)).status_code == 200
    assert (await client.post("/api/auth/login", data=data)).status_code == 200
    resp = await client.post("/api/auth/login", data=data)
    assert resp.status_code == 429


async def test_register_rate_limited_after_too_many_attempts(
    client: AsyncClient, monkeypatch
) -> None:
    monkeypatch.setattr(get_settings(), "auth_register_rate_limit", "2/minute")
    for i in range(2):
        resp = await client.post(
            "/api/auth/register",
            json={"email": f"rl-register-{i}@example.com", "password": "password123"},
        )
        assert resp.status_code == 201
    resp = await client.post(
        "/api/auth/register",
        json={"email": "rl-register-over@example.com", "password": "password123"},
    )
    assert resp.status_code == 429


async def test_login_rate_limit_does_not_affect_registration(
    client: AsyncClient, monkeypatch
) -> None:
    """The two endpoints have independent limits — exhausting login must not block
    registering a brand new account."""
    monkeypatch.setattr(get_settings(), "auth_login_rate_limit", "1/minute")
    await client.post(
        "/api/auth/register",
        json={"email": "rl-independent@example.com", "password": "password123"},
    )
    data = {"username": "rl-independent@example.com", "password": "password123"}
    assert (await client.post("/api/auth/login", data=data)).status_code == 200
    assert (await client.post("/api/auth/login", data=data)).status_code == 429

    resp = await client.post(
        "/api/auth/register",
        json={"email": "rl-independent-2@example.com", "password": "password123"},
    )
    assert resp.status_code == 201
