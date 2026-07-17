"""Standard defensive HTTP response headers.

This is a JSON API — no CSP here (would need to special-case /docs and /redoc,
which load assets from a CDN for the Swagger/ReDoc UI) — just the small set of
headers that are unconditionally safe to apply to every response and don't
require knowing anything about the response body.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import Request, Response

_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
}


async def add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    for name, value in _HEADERS.items():
        response.headers.setdefault(name, value)
    return response
