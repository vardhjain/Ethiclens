"""Typed application settings (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    environment: str = "development"
    secret_key: str = "dev-secret-change-me"
    access_token_expire_minutes: int = 60
    algorithm: str = "HS256"

    # SQLite by default so the service runs (and tests) with zero infrastructure;
    # set DATABASE_URL to a postgresql+asyncpg URL in production.
    database_url: str = "sqlite+aiosqlite:///./ethiclens.db"
    redis_url: str = "redis://localhost:6379/0"

    # When true, audits run in-process instead of via the arq worker (dev/tests).
    eager_tasks: bool = True

    max_model_upload_mb: int = 512
    ingestion_sandbox_timeout_seconds: int = 60
    model_storage_dir: str = "./models/uploaded"


@lru_cache
def get_settings() -> Settings:
    return Settings()
