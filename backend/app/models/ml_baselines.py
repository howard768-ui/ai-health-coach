"""Schema for Signal Engine Phase 2 tables.

Four new ML tables, all prefixed ``ml_`` per the plan. These are written by
``backend/ml/discovery/baselines.py`` and ``backend/ml/forecasting/*`` and
read only by other ``backend/ml/`` modules. Like ``ml_features.py``, they
live under ``app/models/`` solely so alembic discovers them.

Tables:

- ``ml_baselines`` — per-(user, metric) STL decomposition snapshot. One row
  per (user, metric, computed_at). Trend + seasonal + residual_std give
  downstream layers (L2 associations, anomaly detection) the statistics
  they need to residualize a raw series.
- ``ml_change_points`` — detected changes in a metric's baseline. BOCPD
  writes here when the streaming probability crosses threshold.
- ``ml_forecasts`` — per-(user, metric, target_date, made_on) short-horizon
  forecast value with a 95% prediction interval. Many-to-one with user;
  stored forecasts are idempotent per (user, metric, target_date, made_on).
- ``ml_anomalies`` — flagged deviations ``|observed - forecasted|`` beyond
  the residual z-score threshold. One row per (user, metric, date) that
  trips the flag.

See ``~/.claude/plans/golden-floating-creek.md`` Phase 2 for the full spec.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.time import utcnow_naive
from app.database import Base


class MLBaseline(Base):
    """Per-(user, metric) seasonal decomposition + summary statistics.

    Written by L1 when there are enough observed days (see
    ``ml_shadow_baselines`` + ``l1_min_history_days`` in ``ml.config``).
    The actual trend / seasonal / residual series are not stored row-per-day
    here; callers that need them recompute from the feature frame. What's
    stored is enough to residualize a series and decide whether a new
    observation is anomalous.
    """

    __tablename__ = "ml_baselines"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "metric_key",
            "model_version",
            name="uq_ml_baselines_user_metric_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    metric_key: Mapped[str] = mapped_column(String(120), index=True)
    window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    # Rolling stats over the window the baseline was fit on.
    trend_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend_slope: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Slope of the STL trend component, units per day.",
    )
    seasonal_amplitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Range (max-min) of the seasonal component over one week.",
    )
    residual_std: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Standard deviation of STL residuals. Used as the per-metric scale for anomaly Z-scores.",
    )
    last_observed_date: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        doc="The most recent day included in the fit (YYYY-MM-DD).",
    )
    observed_days_in_window: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="How many of the ``window_days`` had an observed value. Lower = less reliable baseline.",
    )
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow_naive, nullable=False
    )


class MLChangePoint(Base):
    """A detected regime change in a metric's baseline.

    Populated by BOCPD (Bayesian Online Change Point Detection) and by the
    offline ``ruptures`` backstop. The two are complementary: BOCPD is fast
    and streaming, ``ruptures`` is used to re-certify historical change
    points on the weekly batch.
    """

    __tablename__ = "ml_change_points"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "metric_key",
            "change_date",
            "detector",
            name="uq_ml_change_points_user_metric_date_detector",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    metric_key: Mapped[str] = mapped_column(String(120), index=True)
    change_date: Mapped[str] = mapped_column(String(10), index=True)
    detector: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="bocpd | ruptures. Both can write the same change_date; uniqueness is per-detector.",
    )
    probability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Posterior probability the change happened on this date (BOCPD). None for ruptures.",
    )
    magnitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Absolute mean shift across the change, in the metric's native units.",
    )
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow_naive, nullable=False
    )


class MLForecast(Base):
    """Per-(user, metric, target_date) short-horizon forecast with 95% PI.

    One row per forecast-date / target-date pair. The ``made_on`` field is
    when the forecast was produced (typically yesterday or today); rerunning
    the same day overwrites that day's forecast for the same target.
    """

    __tablename__ = "ml_forecasts"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "metric_key",
            "target_date",
            "made_on",
            "model_version",
            name="uq_ml_forecasts_user_metric_target_made_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    metric_key: Mapped[str] = mapped_column(String(120), index=True)
    target_date: Mapped[str] = mapped_column(String(10), index=True)
    made_on: Mapped[str] = mapped_column(String(10), nullable=False)
    y_hat: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_hat_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_hat_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow_naive, nullable=False
    )


class MLAnomaly(Base):
    """A flagged anomaly: observed value is ``> threshold * residual_std`` away from forecast.

    These are candidates for the Phase 4 insight pipeline; nothing about this
    table is user-facing on its own. The coach / dashboard pulls from
    ``ml_insight_candidates`` in Phase 4, which normalizes anomalies into the
    same candidate shape as correlations and forecast warnings.
    """

    __tablename__ = "ml_anomalies"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "metric_key",
            "observation_date",
            "model_version",
            name="uq_ml_anomalies_user_metric_date_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    metric_key: Mapped[str] = mapped_column(String(120), index=True)
    observation_date: Mapped[str] = mapped_column(String(10), index=True)
    observed_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    forecasted_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    residual: Mapped[float] = mapped_column(Float, nullable=False)
    z_score: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(
        String(4),
        nullable=False,
        doc="high | low relative to forecast.",
    )
    confirmed_by_bocpd: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="True when a BOCPD change point fired within 48h of this anomaly. Two-signal gating per plan.",
    )
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow_naive, nullable=False
    )
