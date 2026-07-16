"""Lock Supabase's auto-generated REST API out of the schema.

Supabase exposes every table in the ``public`` schema through PostgREST at
``https://<project>.supabase.co/rest/v1/`` using the anon key. EthicLens only
ever talks to Postgres over a direct connection as the table owner (which
bypasses RLS), so that API surface is pure risk: with RLS disabled it grants
anyone holding the anon key full read/write on ``user_account`` (email +
hashed_password) and every audit table.

Enabling RLS with no policies denies all PostgREST access while leaving the
app's owner connection untouched. Revoking the ``anon``/``authenticated``
grants is belt-and-suspenders for the same goal; both steps no-op gracefully
outside Supabase (local docker-compose Postgres has no such roles, SQLite
tests never run Alembic).

``downgrade`` only disables RLS — it does not restore the revoked grants,
since nothing in EthicLens ever needed them.

Revision ID: 0002_enable_rls
Revises: 0001_baseline
Create Date: 2026-07-16
"""

from __future__ import annotations

from alembic import op

from ethiclens_api import models  # noqa: F401  (register tables)
from ethiclens_api.db import Base

revision = "0002_enable_rls"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def _table_names() -> list[str]:
    return [t.name for t in Base.metadata.sorted_tables] + ["alembic_version"]


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for name in _table_names():
        op.execute(f'ALTER TABLE public."{name}" ENABLE ROW LEVEL SECURITY')
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public
                    REVOKE ALL ON TABLES FROM anon;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                REVOKE ALL ON ALL TABLES IN SCHEMA public FROM authenticated;
                ALTER DEFAULT PRIVILEGES IN SCHEMA public
                    REVOKE ALL ON TABLES FROM authenticated;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    for name in _table_names():
        op.execute(f'ALTER TABLE public."{name}" DISABLE ROW LEVEL SECURITY')
