"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ethiclens_api.config import get_settings
from ethiclens_api.db import create_all
from ethiclens_api.rate_limit import limiter
from ethiclens_api.routers import agent, auth, governance, models, reports, sessions
from ethiclens_api.security_headers import add_security_headers

# Nothing else configures Python logging. uvicorn's own LOGGING_CONFIG only sets up
# its own "uvicorn"/"uvicorn.error"/"uvicorn.access" loggers (disable_existing_loggers
# is false there, so it never silences these) and never touches the root logger, so
# this is safe regardless of whether uvicorn's config runs before or after this module
# is imported. Without it, every ethiclens.* logger sits at the default WARNING level
# with no handler — every .info() call across the agent pipeline (LLM latency,
# provider fallback, quota exhaustion) is silently dropped.
logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DESCRIPTION = (
    "EthicLens API — AI bias detection & mitigation workbench. Upload a model, "
    "configure protected attributes, run an audit, review measured mitigations, and "
    "route findings through a governance sign-off. All fairness math is delegated to the "
    "audited `fairness-core` package."
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # For the zero-infrastructure SQLite default, create tables on startup.
    # Production (PostgreSQL) uses Alembic migrations instead.
    if settings.database_url.startswith("sqlite"):
        await create_all()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="EthicLens API", version="0.1.0", description=DESCRIPTION, lifespan=lifespan
    )
    app.state.limiter = limiter

    # slowapi's own handler is typed for RateLimitExceeded specifically, which
    # Starlette's add_exception_handler (typed for the general Exception) rejects
    # under strict mypy parameter contravariance; this thin wrapper satisfies both.
    async def _handle_rate_limit(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RateLimitExceeded)
        return _rate_limit_exceeded_handler(request, exc)

    app.add_exception_handler(RateLimitExceeded, _handle_rate_limit)
    # SlowAPIMiddleware can short-circuit a request with a 429 before it reaches the
    # router. Starlette's add_middleware() prepends, so the middleware added LAST ends
    # up OUTERMOST — CORSMiddleware must be added after SlowAPIMiddleware so it wraps
    # it and still gets a chance to add CORS headers to that 429; added the other way
    # around, a rate-limited response would bypass CORSMiddleware entirely and a
    # cross-origin caller's browser would see an opaque failure instead of a
    # readable 429.
    app.add_middleware(SlowAPIMiddleware)
    # No allow_credentials: the frontend authenticates with a Bearer token in a header,
    # never cookies or fetch's credentials:'include', so there's no ambient credential
    # for CORS to protect here. Pairing allow_origins=["*"] with allow_credentials=True
    # is also invalid per the Fetch spec for credentialed requests — Starlette sends a
    # literal "Access-Control-Allow-Origin: *" (not a reflected origin) in that
    # combination, which spec-compliant browsers must refuse for credentialed responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Outermost, so the headers land on every response, including a rate-limited 429.
    app.middleware("http")(add_security_headers)
    routers = (
        auth.router,
        models.router,
        sessions.router,
        reports.router,
        governance.router,
        agent.router,
    )
    for router in routers:
        app.include_router(router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "ethiclens-api"}

    return app


app = create_app()
