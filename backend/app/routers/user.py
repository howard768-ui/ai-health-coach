"""User profile endpoints. All routes require authentication via CurrentUser."""

import json
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.database import get_db
from app.models.user import User
from app.models.health import OuraToken
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.user")

router = APIRouter(prefix="/api/user", tags=["user"])


class UserProfileResponse(BaseModel):
    name: str | None = None
    email: str | None = None
    age: int | None = None
    height_inches: int | None = None
    weight_lbs: float | None = None
    target_weight_lbs: float | None = None
    goals: list[str] = []
    custom_goal_text: str | None = None
    training_experience: str | None = None
    training_days_per_week: int | None = None
    member_since: str | None = None
    data_sources: list[dict] = []
    onboarding_complete: bool = False


class UserProfileUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
    age: int | None = None
    height_inches: int | None = None
    weight_lbs: float | None = None
    target_weight_lbs: float | None = None
    goals: list[str] | None = None
    custom_goal_text: str | None = None
    training_experience: str | None = None
    training_days_per_week: int | None = None
    onboarding_complete: bool | None = None


async def _build_profile_response(user: User, db: AsyncSession) -> UserProfileResponse:
    """Build a profile response from a User row. Shared by GET and PUT handlers."""
    # Data sources are tenant-scoped
    oura_result = await db.execute(
        select(OuraToken).where(OuraToken.user_id == user.apple_user_id).limit(1)
    )
    oura_token = oura_result.scalar_one_or_none()

    data_sources = []
    if oura_token:
        data_sources.append({
            "name": "Oura Ring",
            "connected": True,
            "last_synced": oura_token.created_at.isoformat() if oura_token.created_at else None,
        })

    goals = []
    if user.goals:
        try:
            goals = json.loads(user.goals) if isinstance(user.goals, str) else user.goals
        except (json.JSONDecodeError, TypeError):
            goals = []

    return UserProfileResponse(
        name=user.name,
        email=user.email,
        age=user.age,
        height_inches=user.height_inches,
        weight_lbs=user.weight_lbs,
        target_weight_lbs=user.target_weight_lbs,
        goals=goals,
        custom_goal_text=user.custom_goal_text,
        training_experience=user.training_experience,
        training_days_per_week=user.training_days_per_week,
        member_since=user.created_at.strftime("%B %Y") if user.created_at else None,
        data_sources=data_sources,
        onboarding_complete=user.onboarding_complete,
    )


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Get the current user's profile with connected data sources."""
    return await _build_profile_response(current_user, db)


@router.delete("/oura", status_code=204)
async def disconnect_oura(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Remove the current user's Oura token, revoking the connection."""
    existing = await db.execute(
        select(OuraToken).where(OuraToken.user_id == current_user.apple_user_id)
    )
    tokens = existing.scalars().all()
    for token in tokens:
        await db.delete(token)
    await db.commit()


@router.put("/profile", response_model=UserProfileResponse)
async def update_profile(
    update: UserProfileUpdate,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Update the current user's profile fields."""
    if update.name is not None:
        current_user.name = update.name
    if update.email is not None:
        current_user.email = update.email
    if update.age is not None:
        current_user.age = update.age
    if update.height_inches is not None:
        current_user.height_inches = update.height_inches
    if update.weight_lbs is not None:
        current_user.weight_lbs = update.weight_lbs
    if update.target_weight_lbs is not None:
        current_user.target_weight_lbs = update.target_weight_lbs
    if update.goals is not None:
        current_user.goals = json.dumps(update.goals)
    if update.custom_goal_text is not None:
        # Empty string is a valid clear; trim whitespace-only input to None.
        trimmed = update.custom_goal_text.strip()
        current_user.custom_goal_text = trimmed if trimmed else None
    if update.training_experience is not None:
        current_user.training_experience = update.training_experience
    if update.training_days_per_week is not None:
        current_user.training_days_per_week = update.training_days_per_week
    if update.onboarding_complete is not None:
        current_user.onboarding_complete = update.onboarding_complete
    current_user.updated_at = utcnow_naive()

    await db.commit()
    await db.refresh(current_user)
    return await _build_profile_response(current_user, db)
