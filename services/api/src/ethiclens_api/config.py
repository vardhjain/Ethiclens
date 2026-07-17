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

    # Root log level. Nothing configures Python logging otherwise — every module-level
    # logger (ethiclens.agent, ethiclens.agent.quota, ...) would sit at the default
    # WARNING level with no handler, silently dropping every .info() call.
    log_level: str = "INFO"

    # SQLite by default so the service runs (and tests) with zero infrastructure;
    # set DATABASE_URL to a postgresql+asyncpg URL in production.
    database_url: str = "sqlite+aiosqlite:///./ethiclens.db"
    redis_url: str = "redis://localhost:6379/0"

    # When true, audits run in-process instead of via the arq worker (dev/tests).
    eager_tasks: bool = True

    max_model_upload_mb: int = 512
    ingestion_sandbox_timeout_seconds: int = 60
    model_storage_dir: str = "./models/uploaded"

    # --- Agent LLM layer (Stage 1 schema inference, Stage 4 narrative) ---
    # Groq is tried first (free-tier, fast); Gemini is the fallback. Either or both may
    # be unset in dev/tests — the agent degrades to "narrative unavailable" without them.
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    # Agent CSV upload caps (predictions-only ingestion; no model files, no persistence
    # of raw rows — only the resulting audit record is stored). agent_max_columns bounds
    # the Stage 1 prompt: every column name goes into it verbatim, and 100 is already
    # generous for a predictions CSV (protected attributes + outcome + features).
    agent_max_upload_mb: int = 5
    agent_max_rows: int = 50_000
    agent_max_columns: int = 100

    # Hard daily cap on LLM calls across all users (~80% of a free-tier quota in
    # practice). Once hit, the agent degrades instead of erroring: schema inference
    # returns 429, narrative/Q&A fall back to numeric-only / "unavailable" responses.
    agent_daily_llm_call_cap: int = 200

    # Tighter per-user daily cap, checked alongside the global one. On a shared public
    # demo, without this a single user looping propose-schema/run-audit/ask can exhaust
    # the whole day's budget for everyone else.
    agent_daily_llm_call_cap_per_user: int = 20

    # Per-IP rate limits (slowapi / limits syntax: "<count>/<period>") on the two
    # unauthenticated auth endpoints. Registration is open with no email verification
    # or captcha, and login has no lockout — without these both are open brute-force /
    # credential-stuffing / spam-registration vectors on a public demo.
    auth_login_rate_limit: str = "10/minute"
    auth_register_rate_limit: str = "5/minute"


@lru_cache
def get_settings() -> Settings:
    return Settings()
