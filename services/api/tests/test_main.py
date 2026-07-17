"""App-level behavior that isn't tied to a specific router: security headers, health."""

from __future__ import annotations

from httpx import AsyncClient


async def test_health_check(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "ethiclens-api"}


async def test_response_includes_security_headers(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "camera=()" in resp.headers["permissions-policy"]
    assert "max-age=" in resp.headers["strict-transport-security"]


async def test_security_headers_present_on_a_rate_limited_response(
    client: AsyncClient, monkeypatch
) -> None:
    """Security headers, like CORS headers, must survive SlowAPIMiddleware
    short-circuiting a request with a 429 before it reaches the router."""
    from ethiclens_api.config import get_settings

    monkeypatch.setattr(get_settings(), "auth_login_rate_limit", "1/minute")
    await client.post(
        "/api/auth/register",
        json={"email": "headers-429@example.com", "password": "password123"},
    )
    data = {"username": "headers-429@example.com", "password": "password123"}
    await client.post("/api/auth/login", data=data)
    resp = await client.post("/api/auth/login", data=data)
    assert resp.status_code == 429
    assert resp.headers["x-content-type-options"] == "nosniff"
