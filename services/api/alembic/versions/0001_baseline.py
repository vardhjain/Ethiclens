"""Baseline schema (the five STP entities + uploaded-model registry).

This first revision creates the whole schema from the ORM metadata. Subsequent
revisions are generated with ``alembic revision --autogenerate``.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-05-01
"""

from __future__ import annotations

from alembic import op

from ethiclens_api import models  # noqa: F401  (register tables)
from ethiclens_api.db import Base

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
