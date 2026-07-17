"""Add user_usage_counter (per-user daily LLM call budget).

UsageCounter alone caps total cost across all users. On a shared public demo that
lets one user calling propose-schema/run-audit/ask in a loop exhaust the whole day's
budget for everyone else. This adds a per-(user, day) counter checked alongside the
existing global one.

Explicit ``op.create_table`` + RLS enable for just this one new table, rather than
``Base.metadata.create_all`` (0001's approach) — that would only be safe against a
completely fresh database; here it must compose with whatever 0001/0002 already
created in an already-migrated deployment.

The create is guarded by a ``has_table`` check: 0001 creates its schema from current
ORM metadata at whatever time it runs, so on a database that has never been migrated
before, 0001 alone already creates this table (it's in ``models.py`` by the time 0001
runs) and this revision would otherwise fail with "table already exists". On a
database that already ran 0001/0002 in the past (before this model existed), the
table is genuinely missing and this creates it.

Revision ID: 0003_user_usage_counter
Revises: 0002_enable_rls
Create Date: 2026-07-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_user_usage_counter"
down_revision = "0002_enable_rls"
branch_labels = None
depends_on = None

_TABLE = "user_usage_counter"


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(_TABLE):
        op.create_table(
            _TABLE,
            sa.Column("user_id", sa.Uuid(), sa.ForeignKey("user_account.id", ondelete="CASCADE")),
            sa.Column("day", sa.String(length=10)),
            sa.Column("llm_calls", sa.Integer(), nullable=False, server_default="0"),
            sa.PrimaryKeyConstraint("user_id", "day"),
        )

    if bind.dialect.name != "postgresql":
        return
    op.execute(f'ALTER TABLE public."{_TABLE}" ENABLE ROW LEVEL SECURITY')
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                REVOKE ALL ON public."{_TABLE}" FROM anon;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                REVOKE ALL ON public."{_TABLE}" FROM authenticated;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.drop_table(_TABLE)
