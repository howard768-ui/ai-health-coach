"""Schema for Signal Engine Phase 10 training run log.

Lightweight alternative to a full MLflow server. One row per training
run, logging parameters, metrics, and status. Provides the audit trail
needed for beta without the infrastructure cost of running MLflow.

See ``~/.claude/plans/harmonic-meandering-stardust.md`` Phase 10 for rationale.
"""

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class MLTrainingRun(Base):
    """One model training run. Logged before and after training."""

    __tablename__ = "ml_training_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_type: Mapped[str] = mapped_column(
        String(40),
        index=True,
        nullable=False,
        doc='"ranker" for XGBoost LambdaMART. Future: other model types.',
    )
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    params_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON dict of hyperparameters used.",
    )
    metrics_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="JSON dict of final metrics (val_ndcg, train_samples, etc.).",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="running | completed | failed",
    )
    model_version: Mapped[str | None] = mapped_column(
        String(40),
        nullable=True,
        doc="The model version produced by this run (set on completion).",
    )
