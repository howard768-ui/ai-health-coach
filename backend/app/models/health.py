from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.core.encryption import EncryptedString
from app.database import Base
from app.core.time import utcnow_naive


class OuraToken(Base):
    __tablename__ = "oura_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    # Encrypted at rest with Fernet (P1-1). Backward-compatible — legacy
    # plaintext rows still read fine and get re-encrypted on next refresh.
    access_token: Mapped[str] = mapped_column(EncryptedString(2000))
    refresh_token: Mapped[str] = mapped_column(EncryptedString(2000))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    # Last time the backend successfully called Oura's API for this user.
    # Used by _maybe_refresh_oura() in routers/health.py to throttle the
    # on-demand dashboard sync. Updated at the END of sync_user_data()
    # regardless of how many records were saved — the semantic is "we
    # talked to Oura", not "we got new rows". NULL = never synced.
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # MEL-45 part 1: Oura's user identifier (from /v2/usercollection/personal_info
    # `id` field, also sent in webhook payload as `body["user_id"]`). NULL until
    # backfilled on first sync. Indexed because the webhook receiver in MEL-45
    # part 2 will look up by this column on every incoming event.
    oura_user_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )


class SleepRecord(Base):
    __tablename__ = "sleep_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    efficiency: Mapped[float] = mapped_column(Float, nullable=True)
    total_sleep_seconds: Mapped[int] = mapped_column(Integer, nullable=True)
    deep_sleep_seconds: Mapped[int] = mapped_column(Integer, nullable=True)
    rem_sleep_seconds: Mapped[int] = mapped_column(Integer, nullable=True)
    light_sleep_seconds: Mapped[int] = mapped_column(Integer, nullable=True)
    hrv_average: Mapped[float] = mapped_column(Float, nullable=True)
    resting_hr: Mapped[float] = mapped_column(Float, nullable=True)
    readiness_score: Mapped[int] = mapped_column(Integer, nullable=True)
    bedtime_start: Mapped[str] = mapped_column(String(5), nullable=True)  # HH:MM when user fell asleep
    bedtime_end: Mapped[str] = mapped_column(String(5), nullable=True)    # HH:MM when user woke up
    # PHI: full raw vendor sleep payload. Encrypted at rest (audit C2).
    raw_json: Mapped[str] = mapped_column(EncryptedString(), nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    # Phase 4.5 synth-factory tag. Default False for all real ingestion paths.
    # Crisis evals and production aggregates MUST filter on is_synthetic = False.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)


class HealthMetricRecord(Base):
    """Unified metric store for all data sources.

    Every source writes here. The reconciliation engine marks
    is_canonical=True on the winning value per metric per date.
    """
    __tablename__ = "health_metric_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True)
    metric_type: Mapped[str] = mapped_column(String(50), index=True)  # sleep_duration, hrv, steps, etc.
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(20), nullable=True)
    source: Mapped[str] = mapped_column(String(30), index=True)  # oura, apple_health, garmin, peloton
    is_canonical: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[str] = mapped_column(String(20), default="primary")  # primary, fallback
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    # Phase 4.5 synth-factory tag. Default False for all real ingestion paths.
    # Crisis evals and production aggregates MUST filter on is_synthetic = False.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)


class ActivityRecord(Base):
    """Activity data from HealthKit, Garmin, or Peloton."""
    __tablename__ = "activity_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True)
    steps: Mapped[int] = mapped_column(Integer, nullable=True)
    active_calories: Mapped[int] = mapped_column(Integer, nullable=True)
    workout_type: Mapped[str] = mapped_column(String(50), nullable=True)
    workout_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(30))
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    # Phase 4.5 synth-factory tag. Default False for all real ingestion paths.
    # Crisis evals and production aggregates MUST filter on is_synthetic = False.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)


class SourcePriority(Base):
    """User-configurable data source priority per metric category."""
    __tablename__ = "source_priorities"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    metric_category: Mapped[str] = mapped_column(String(50))
    priority_order: Mapped[str] = mapped_column(Text)  # JSON list of source names
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
