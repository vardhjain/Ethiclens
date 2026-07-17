"""Add agent_audit_record and usage_counter (agent pipeline persistence + cost cap).

Both models have existed in ``models.py`` for several prior features (the agent audit
record store and the daily LLM call budget) but neither ever got its own explicit
migration — an oversight caught while adding 0003_user_usage_counter, which is the
first migration to actually need one of these tables to already exist (its FK target,
user_account, predates it; agent_audit_record and usage_counter don't reference it).

Same guarded pattern as 0003: on a database that has never been migrated before, 0001's
``Base.metadata.create_all`` already creates both tables (they're in current
``models.py`` by the time 0001 runs), so this only actually creates anything on a
database that ran 0001/0002 in the past, before these models existed.

Revision ID: 0004_agent_tables
Revises: 0003_user_usage_counter
Create Date: 2026-07-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_agent_tables"
down_revision = "0003_user_usage_counter"
branch_labels = None
depends_on = None

_AGENT_AUDIT_RECORD = "agent_audit_record"
_USAGE_COUNTER = "usage_counter"


def _enable_rls(bind: sa.engine.Connection, table: str) -> None:
    if bind.dialect.name != "postgresql":
        return
    op.execute(f'ALTER TABLE public."{table}" ENABLE ROW LEVEL SECURITY')
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                REVOKE ALL ON public."{table}" FROM anon;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                REVOKE ALL ON public."{table}" FROM authenticated;
            END IF;
        END $$;
        """
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table(_AGENT_AUDIT_RECORD):
        op.create_table(
            _AGENT_AUDIT_RECORD,
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column(
                "owner_id",
                sa.Uuid(),
                sa.ForeignKey("user_account.id"),
                index=True,
                nullable=False,
            ),
            sa.Column("outcome_column", sa.String(length=128), nullable=False),
            sa.Column("protected_attribute_columns", sa.JSON(), nullable=False),
            sa.Column("true_label_column", sa.String(length=128), nullable=True),
            sa.Column("plan", sa.JSON(), nullable=False),
            sa.Column("scorecard", sa.JSON(), nullable=False),
            sa.Column("narrative", sa.Text(), nullable=False),
            sa.Column("mitigation_summary", sa.Text(), nullable=False),
            sa.Column("grounded", sa.Boolean(), nullable=False),
            sa.Column("degraded", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    _enable_rls(bind, _AGENT_AUDIT_RECORD)

    if not inspector.has_table(_USAGE_COUNTER):
        op.create_table(
            _USAGE_COUNTER,
            sa.Column("day", sa.String(length=10), primary_key=True),
            sa.Column("llm_calls", sa.Integer(), nullable=False, server_default="0"),
        )
    _enable_rls(bind, _USAGE_COUNTER)


def downgrade() -> None:
    op.drop_table(_USAGE_COUNTER)
    op.drop_table(_AGENT_AUDIT_RECORD)
