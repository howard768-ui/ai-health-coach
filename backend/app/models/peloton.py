from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.core.encryption import EncryptedString
from app.database import Base
from app.core.time import utcnow_naive


class PelotonToken(Base):
    """Peloton credentials.

    `pylotoncycle.PylotonCycle` (the underlying client library) auths via
    username + password and does NOT expose a session token suitable for
    persistence. Pre-MEL-44, `session_id` stored a literal "oauth" placeholder
    that was unusable for re-authentication; the sync flow short-circuited
    with `needs_reauth` (PR #89). MEL-44 part 1 (this) adds the encrypted
    `password` column; MEL-44 part 2 will rewire the sync to login fresh on
    every sync cycle and deprecate the `session_id` placeholder.
    """
    __tablename__ = "peloton_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    peloton_user_id: Mapped[str] = mapped_column(String(255))
    # Encrypted at rest with Fernet (P1-1). Legacy from when the design was
    # cookie-based; pre-MEL-44 it stored a literal "oauth" placeholder.
    # Will be removed in a later cleanup migration once part 2 ships.
    session_id: Mapped[str] = mapped_column(EncryptedString(2000))
    username: Mapped[str] = mapped_column(String(255))
    # MEL-44 part 1: encrypted password for pylotoncycle re-auth on each sync.
    # NULL until backfilled (existing connections stay in `needs_reauth` until
    # user reconnects). EncryptedString uses Fernet AT-REST (see core/encryption).
    # Part 2 will populate this on connect and consume it on each sync.
    password: Mapped[str | None] = mapped_column(EncryptedString(2000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)


class WorkoutRecord(Base):
    """Workout data from any source (Peloton, Garmin, Apple Health)."""
    __tablename__ = "workout_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True)
    source: Mapped[str] = mapped_column(String(30))  # peloton, garmin, apple_health
    external_id: Mapped[str] = mapped_column(String(255), nullable=True)  # Source-specific workout ID
    workout_type: Mapped[str] = mapped_column(String(50))  # cycling, running, strength, yoga, etc.
    duration_seconds: Mapped[int] = mapped_column(Integer)
    calories: Mapped[int] = mapped_column(Integer, nullable=True)
    avg_heart_rate: Mapped[float] = mapped_column(Float, nullable=True)
    max_heart_rate: Mapped[float] = mapped_column(Float, nullable=True)
    avg_output: Mapped[float] = mapped_column(Float, nullable=True)  # Watts (Peloton-specific)
    instructor: Mapped[str] = mapped_column(String(100), nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    # PHI: full raw vendor workout payload. Encrypted at rest (audit C2).
    raw_json: Mapped[str] = mapped_column(EncryptedString(), nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
