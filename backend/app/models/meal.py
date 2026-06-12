from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.core.time import utcnow_naive


class MealRecord(Base):
    __tablename__ = "meal_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    meal_type: Mapped[str] = mapped_column(String(20))  # breakfast, lunch, dinner, snack
    source: Mapped[str] = mapped_column(String(20))  # photo, barcode, text, search, manual
    photo_hash: Mapped[str] = mapped_column(String(64), nullable=True)  # SHA256 for dedup
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive, onupdate=utcnow_naive)
    # Phase 4.5 synth-factory tag. Default False for all real ingestion paths.
    # Crisis evals and production aggregates MUST filter on is_synthetic = False.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)


class FoodItemRecord(Base):
    __tablename__ = "food_item_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    meal_id: Mapped[int] = mapped_column(Integer, ForeignKey("meal_records.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    serving_size: Mapped[str] = mapped_column(String(100))
    serving_count: Mapped[float] = mapped_column(Float, default=1.0)
    calories: Mapped[int] = mapped_column(Integer)
    protein: Mapped[float] = mapped_column(Float)
    carbs: Mapped[float] = mapped_column(Float)
    fat: Mapped[float] = mapped_column(Float)
    quality: Mapped[str] = mapped_column(String(20))  # whole, mixed, processed
    data_source: Mapped[str] = mapped_column(String(20))  # usda, usda_branded, off, ai_estimate
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    # Phase 4.5 synth-factory tag. Default False for all real ingestion paths.
    # Crisis evals and production aggregates MUST filter on is_synthetic = False.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
