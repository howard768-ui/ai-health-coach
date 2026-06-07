"""Meal logging and food search endpoints.

Handles meal CRUD, food photo recognition via Claude Vision,
food database search (USDA + OFF), and barcode lookup.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.core.ratelimit import limiter
from app.core.time import user_hour, user_today_iso
from app.database import get_db
from app.models.meal import MealRecord, FoodItemRecord
from app.schemas.meal import (
    MealCreate, MealResponse, DailyMealsResponse,
    FoodItemCreate, FoodItemResponse,
    FoodRecognitionRequest, FoodRecognitionResponse,
    FoodSearchRequest,
)
from app.services.food_recognition import food_recognition
from app.services.food_search import food_search
from app.services.openfoodfacts import off_client

logger = logging.getLogger("meld.meals")

router = APIRouter(prefix="/api", tags=["meals"])

# Rate limit AI-backed food recognition to prevent budget exhaustion (P1-4).
# Uses the single shared limiter (app.core.ratelimit), keyed on the real client
# IP, not the Cloudflare edge IP. Audit C2.


def _meal_type_from_time() -> str:
    """Auto-assign meal type based on the current hour in the user's timezone.

    Pre-PR-G this read `datetime.now().hour` (naive = UTC on Railway), so
    "breakfast" auto-classified at the wrong wall-clock hour for every
    user not in UTC. Now binds to `settings.user_timezone`.
    """
    hour = user_hour()
    if 5 <= hour < 11:
        return "breakfast"
    elif 11 <= hour < 15:
        return "lunch"
    elif 15 <= hour < 17:
        return "snack"
    else:
        return "dinner"


def _meal_to_response(meal: MealRecord, items: list[FoodItemRecord]) -> MealResponse:
    """Convert DB records to response model."""
    food_items = [
        FoodItemResponse(
            id=item.id,
            name=item.name,
            serving_size=item.serving_size,
            serving_count=item.serving_count,
            calories=item.calories,
            protein=item.protein,
            carbs=item.carbs,
            fat=item.fat,
            quality=item.quality,
            data_source=item.data_source,
            confidence=item.confidence,
        )
        for item in items
    ]
    return MealResponse(
        id=meal.id,
        date=meal.date,
        meal_type=meal.meal_type,
        source=meal.source,
        items=food_items,
        total_calories=sum(i.calories for i in items),
        total_protein=round(sum(i.protein for i in items), 1),
        total_carbs=round(sum(i.carbs for i in items), 1),
        total_fat=round(sum(i.fat for i in items), 1),
        created_at=meal.created_at.isoformat(),
    )


# ── Meal CRUD ───────────────────────────────────────────────

@router.post("/meals", response_model=MealResponse)
async def create_meal(
    meal_data: MealCreate,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Log a new meal with food items.

    PR-G: meal date is computed in the user's timezone (settings.user_timezone)
    rather than naive server time, so an EST user's 8pm dinner is dated to
    today's date instead of tomorrow's UTC date.
    """
    meal = MealRecord(
        user_id=current_user.apple_user_id,
        date=user_today_iso(),
        meal_type=meal_data.meal_type or _meal_type_from_time(),
        source=meal_data.source,
        photo_hash=meal_data.photo_hash,
    )
    db.add(meal)
    await db.flush()  # Get meal.id for foreign key

    items = []
    for item_data in meal_data.items:
        item = FoodItemRecord(
            meal_id=meal.id,
            name=item_data.name,
            serving_size=item_data.serving_size,
            serving_count=item_data.serving_count,
            calories=item_data.calories,
            protein=item_data.protein,
            carbs=item_data.carbs,
            fat=item_data.fat,
            quality=item_data.quality,
            data_source=item_data.data_source,
            confidence=item_data.confidence,
        )
        db.add(item)
        items.append(item)

    await db.commit()
    await db.refresh(meal)
    for item in items:
        await db.refresh(item)

    logger.info("Logged %s with %d items", meal.meal_type, len(items))
    return _meal_to_response(meal, items)


@router.get("/meals", response_model=DailyMealsResponse)
async def get_meals(
    current_user: CurrentUser,
    date: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Get all meals for a date (defaults to today in user's timezone)."""
    target_date = date or user_today_iso()

    result = await db.execute(
        select(MealRecord)
        .where(MealRecord.user_id == current_user.apple_user_id, MealRecord.date == target_date)
        .order_by(MealRecord.created_at)
    )
    meals = result.scalars().all()

    meal_responses = []
    total_cal, total_p, total_c, total_f = 0, 0.0, 0.0, 0.0

    for meal in meals:
        items_result = await db.execute(
            select(FoodItemRecord).where(FoodItemRecord.meal_id == meal.id)
        )
        items = items_result.scalars().all()
        response = _meal_to_response(meal, items)
        meal_responses.append(response)
        total_cal += response.total_calories
        total_p += response.total_protein
        total_c += response.total_carbs
        total_f += response.total_fat

    return DailyMealsResponse(
        date=target_date,
        meals=meal_responses,
        total_calories=total_cal,
        total_protein=round(total_p, 1),
        total_carbs=round(total_c, 1),
        total_fat=round(total_f, 1),
    )


@router.delete("/meals/{meal_id}")
async def delete_meal(
    meal_id: int,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Delete a meal and its food items."""
    result = await db.execute(
        select(MealRecord).where(
            MealRecord.id == meal_id,
            MealRecord.user_id == current_user.apple_user_id,
        )
    )
    meal = result.scalar_one_or_none()
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")

    # Delete food items first
    items_result = await db.execute(
        select(FoodItemRecord).where(FoodItemRecord.meal_id == meal_id)
    )
    for item in items_result.scalars().all():
        await db.delete(item)

    await db.delete(meal)
    await db.commit()
    return {"status": "ok"}


@router.put("/meals/{meal_id}/items/{item_id}")
async def update_food_item(
    meal_id: int,
    item_id: int,
    item_data: FoodItemCreate,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Update an individual food item within a meal owned by the caller."""
    # Verify the meal belongs to current user BEFORE touching the item
    meal_result = await db.execute(
        select(MealRecord).where(
            MealRecord.id == meal_id,
            MealRecord.user_id == current_user.apple_user_id,
        )
    )
    if meal_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Meal not found")

    result = await db.execute(
        select(FoodItemRecord).where(
            FoodItemRecord.id == item_id, FoodItemRecord.meal_id == meal_id
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Food item not found")

    item.name = item_data.name
    item.serving_size = item_data.serving_size
    item.serving_count = item_data.serving_count
    item.calories = item_data.calories
    item.protein = item_data.protein
    item.carbs = item_data.carbs
    item.fat = item_data.fat
    item.quality = item_data.quality
    item.data_source = item_data.data_source
    item.confidence = item_data.confidence

    await db.commit()
    return {"status": "ok"}


@router.delete("/meals/{meal_id}/items/{item_id}")
async def delete_food_item(
    meal_id: int,
    item_id: int,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Remove an individual food item from a meal owned by the caller."""
    # Verify the meal belongs to current user first
    meal_result = await db.execute(
        select(MealRecord).where(
            MealRecord.id == meal_id,
            MealRecord.user_id == current_user.apple_user_id,
        )
    )
    if meal_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Meal not found")

    result = await db.execute(
        select(FoodItemRecord).where(
            FoodItemRecord.id == item_id, FoodItemRecord.meal_id == meal_id
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Food item not found")

    await db.delete(item)
    await db.commit()
    return {"status": "ok"}


# ── Food Recognition ────────────────────────────────────────

@router.post("/food/recognize", response_model=FoodRecognitionResponse)
@limiter.limit("20/minute")
async def recognize_food(
    request: Request,
    body: FoodRecognitionRequest,
    current_user: CurrentUser,
):
    """Recognize food items from a photo using Claude Vision.

    Rate-limited to 20/min per IP — Claude Vision is the most expensive
    endpoint per request. Auth-gating + rate limit together cap exposure.
    """
    import asyncio
    # food_recognition uses the synchronous Anthropic SDK — must offload
    # to a thread to avoid blocking the event loop for the full API call.
    items = await asyncio.to_thread(
        food_recognition.recognize_from_photo,
        body.image_base64,
        body.media_type,
    )

    meal_type = body.meal_type or _meal_type_from_time()

    return FoodRecognitionResponse(
        items=[
            FoodItemResponse(
                name=item["name"],
                serving_size=item.get("serving_size", "1 serving"),
                calories=item.get("calories", 0),
                protein=item.get("protein", 0),
                carbs=item.get("carbs", 0),
                fat=item.get("fat", 0),
                quality=item.get("quality", "mixed"),
                data_source=item.get("data_source", "ai_estimate"),
                confidence=item.get("confidence", 0.7),
            )
            for item in items
        ],
        meal_type=meal_type,
    )


# ── Food Search (DOVA: USDA → OFF → AI) ────────────────────

@router.post("/food/search")
async def search_food(
    request: FoodSearchRequest,
    current_user: CurrentUser,
):
    """Search for food items using DOVA cascade.

    1. USDA FoodData Central (lab-analyzed, highest confidence)
    2. Open Food Facts (crowdsourced, good coverage)
    3. AI estimation fallback (Claude, lowest confidence)

    Auth-gated to prevent upstream API quota abuse (P1-4).
    """
    items = await food_search.search(request.query)
    return {
        "results": [
            FoodItemResponse(
                name=item["name"],
                serving_size=item.get("serving_size", "1 serving"),
                calories=item.get("calories", 0),
                protein=item.get("protein", 0),
                carbs=item.get("carbs", 0),
                fat=item.get("fat", 0),
                quality=item.get("quality", "mixed"),
                data_source=item.get("data_source", "ai_estimate"),
                confidence=item.get("confidence", 0.6),
            )
            for item in items
        ],
    }


# ── Barcode Lookup (Open Food Facts) ────────────────────────

@router.get("/food/barcode/{code}")
async def lookup_barcode(
    code: str,
    current_user: CurrentUser,
):
    """Look up a food product by barcode via Open Food Facts."""
    item = await off_client.get_by_barcode(code)
    if not item:
        raise HTTPException(status_code=404, detail="Product not found")

    return FoodItemResponse(
        name=item["name"],
        serving_size=item.get("serving_size", "1 serving"),
        calories=item.get("calories", 0),
        protein=item.get("protein", 0),
        carbs=item.get("carbs", 0),
        fat=item.get("fat", 0),
        quality=item.get("quality", "mixed"),
        data_source="off",
        confidence=0.9,
    )
