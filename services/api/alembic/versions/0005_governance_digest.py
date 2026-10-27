"""Record fingerprints for approved audit sessions.

Revision ID: 0005_governance_digest
Revises: 0004_agent_tables
Create Date: 2026-10-26
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_governance_digest"
down_revision = "0004_agent_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {col["name"] for col in sa.inspect(op.get_bind()).get_columns("audit_session")}
    if "signoff_digest" not in columns:
        op.add_column("audit_session", sa.Column("signoff_digest", sa.String(64), nullable=True))


def downgrade() -> None:
    columns = {col["name"] for col in sa.inspect(op.get_bind()).get_columns("audit_session")}
    if "signoff_digest" in columns:
        op.drop_column("audit_session", "signoff_digest")
