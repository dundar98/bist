"""add decision logs

Revision ID: b1c7d5a9f2e4
Revises: 9bf33a521c34
Create Date: 2026-05-19 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b1c7d5a9f2e4"
down_revision: Union[str, None] = "9bf33a521c34"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "decision_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("signal_id", sa.Integer(), nullable=True),
        sa.Column("symbol_id", sa.Integer(), nullable=False),
        sa.Column("decision_time", sa.DateTime(), nullable=False),
        sa.Column("signal_time", sa.DateTime(), nullable=False),
        sa.Column(
            "timeframe",
            postgresql.ENUM("DAILY", "HOURLY", "FIFTEEN_MIN", name="timeframe", create_type=False),
            nullable=False,
        ),
        sa.Column(
            "horizon",
            postgresql.ENUM("SHORT", "MEDIUM", "LONG", name="horizon", create_type=False),
            nullable=False,
        ),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column(
            "direction",
            postgresql.ENUM("BUY", "SELL", "HOLD", name="signaldirection", create_type=False),
            nullable=False,
        ),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("stop_price", sa.Float(), nullable=True),
        sa.Column("target_price", sa.Float(), nullable=True),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("model_score", sa.Float(), nullable=False),
        sa.Column("trend_score", sa.Float(), nullable=False),
        sa.Column("volume_score", sa.Float(), nullable=False),
        sa.Column("relative_strength_score", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("raw_json", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["signal_id"], ["signals.id"]),
        sa.ForeignKeyConstraint(["symbol_id"], ["symbols.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_decision_logs_signal", "decision_logs", ["signal_id"], unique=False)
    op.create_index("ix_decision_logs_symbol_time", "decision_logs", ["symbol_id", "decision_time"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_decision_logs_symbol_time", table_name="decision_logs")
    op.drop_index("ix_decision_logs_signal", table_name="decision_logs")
    op.drop_table("decision_logs")
