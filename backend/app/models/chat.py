from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.core.encryption import EncryptedString
from app.core.time import utcnow_naive


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)


class ChatMessageRecord(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(Integer, index=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user" or "coach"
    # PHI: the actual user<->coach conversation. Encrypted at rest (audit C2).
    content: Mapped[str] = mapped_column(EncryptedString())
    routing_tier: Mapped[str] = mapped_column(String(20), nullable=True)  # rules/sonnet/opus
    routing_reason: Mapped[str] = mapped_column(String(255), nullable=True)
    model_used: Mapped[str] = mapped_column(String(100), nullable=True)
    safety_flagged: Mapped[bool] = mapped_column(default=False)
    metadata_json: Mapped[str] = mapped_column(JSON, nullable=True)  # Extra context
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)

    # Phase 4: Production monitoring
    feedback: Mapped[str] = mapped_column(String(10), nullable=True)  # "up", "down", or null
    feedback_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=True)  # Response time in ms
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=True)  # A/B test variant
    # PHI: health data snapshot sent to the model. Encrypted at rest (audit C2).
    health_context: Mapped[str] = mapped_column(EncryptedString(), nullable=True)
