from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.core.encryption import EncryptedString
from app.core.time import utcnow_naive


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Apple's stable per-app user identifier. This is our natural tenant key —
    # all tenant tables FK to users.apple_user_id (not users.id).
    apple_user_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    height_inches: Mapped[int] = mapped_column(Integer, nullable=True)
    weight_lbs: Mapped[float] = mapped_column(Float, nullable=True)
    target_weight_lbs: Mapped[float] = mapped_column(Float, nullable=True)
    goals: Mapped[str] = mapped_column(JSON, nullable=True)  # List of goal strings
    # Free-form "what do you want to get out of this?" text captured in onboarding
    # Goals step. Not required, but fed into the coach system prompt as additional
    # personalization context so it can address the user's actual situation rather
    # than only the canned chip goals.
    # PHI: free-form personal context fed into the coach prompt. Encrypted at
    # rest (audit C2).
    custom_goal_text: Mapped[str | None] = mapped_column(EncryptedString(), nullable=True)
    training_experience: Mapped[str] = mapped_column(String(50), nullable=True)
    training_days_per_week: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)

    # ── Auth fields (added 2026-04-10 for Sign in with Apple) ────────────
    # Deactivation flag — set to False to soft-disable an account without deletion.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # Apple private relay address (@privaterelay.appleid.com) — still routable,
    # but we flag it for compliance and UI display.
    is_private_email: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Last successful sign-in — useful for session hygiene and monitoring.
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Apple's refresh token from the original /auth/apple exchange.
    # Required for account deletion (we need to call Apple's /auth/revoke with it).
    # Nullable because older accounts may not have captured this.
    apple_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Set to True when the user reaches the main dashboard after completing all
    # onboarding steps. Used to restore `hasCompletedOnboarding` after reinstall.
    onboarding_complete: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
