"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ethiclens_api.config import get_settings
from ethiclens_api.db import create_all
from ethiclens_api.routers import auth, governance, models, reports, sessions

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (auth.router, models.router, sessions.router, reports.router, governance.router):
        app.include_router(router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "ethiclens-api"}

    return app


app = create_app()
