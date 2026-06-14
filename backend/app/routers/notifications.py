"""Notification management endpoints.

Handles device token registration, notification preferences,
test notifications, and open tracking.
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.core.constants import TEST_DEVICE_TOKEN
from app.database import get_db
from app.models.notification import DeviceToken, NotificationRecord, NotificationPreference
from app.services.apns import apns_client
from app.services.notification_engine import notification_engine
from app.services.notification_content import content_generator
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.notifications")

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _user_display_name(current_user) -> str:
    """Get the user's first name for notification templates."""
    if current_user.name:
        return current_user.name.split()[0]
    return "there"


# ── Request/Response Models ─────────────────────────────────

class RegisterTokenRequest(BaseModel):
    device_token: str
    platform: str = "ios"


class RegisterTokenResponse(BaseModel):
    status: str
    message: str


class NotificationPreferencesResponse(BaseModel):
    morning_brief: bool = True
    coaching_nudge: bool = True
    bedtime_coaching: bool = True
    streak_alerts: bool = True
    weekly_review: bool = True
    workout_reminders: bool = False
    health_alerts: bool = True
    nudge_frequency: str = "2x_week"  # daily, 2x_week, weekly
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "07:00"


class TestNotificationResponse(BaseModel):
    status: str
    notification_id: int | None = None
    apns_result: dict | None = None
    content: dict | None = None


# ── Endpoints ───────────────────────────────────────────────

@router.post("/register", response_model=RegisterTokenResponse)
async def register_device_token(
    request: RegisterTokenRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Register or update an APNs device token for the current user."""
    user_id = current_user.apple_user_id
    # Check if token already exists
    result = await db.execute(
        select(DeviceToken).where(DeviceToken.token == request.device_token)
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.user_id = user_id
        existing.is_active = True
        existing.updated_at = utcnow_naive()
        # Token is APNs device-token, deliberately redacted to prefix...suffix shape.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        logger.info("Updated device token %s...%s", request.device_token[:8], request.device_token[-4:])
    else:
        token = DeviceToken(
            user_id=user_id,
            token=request.device_token,
            platform=request.platform,
        )
        db.add(token)
        # Same redacted-shape token.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        logger.info("Registered new device token %s...%s", request.device_token[:8], request.device_token[-4:])

    await db.commit()
    return RegisterTokenResponse(status="ok", message="Token registered")


@router.post("/test", response_model=TestNotificationResponse)
async def send_test_notification(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Send a test morning brief notification (dev endpoint)."""
    user_id = current_user.apple_user_id
    # Get active device token (skip test tokens)
    result = await db.execute(
        select(DeviceToken).where(
            DeviceToken.user_id == user_id,
            DeviceToken.is_active,
            DeviceToken.token != TEST_DEVICE_TOKEN,
        )
    )
    token_row = result.scalar_one_or_none()
    if not token_row:
        return TestNotificationResponse(
            status="error",
            content={"message": "No registered device token found"},
        )

    # Get latest health data for content generation
    from app.models.health import SleepRecord
    from sqlalchemy import desc

    sleep_result = await db.execute(
        select(SleepRecord)
        .where(SleepRecord.user_id == user_id)
        .order_by(desc(SleepRecord.date))
        .limit(1)
    )
    sleep_record = sleep_result.scalar_one_or_none()

    health_data = {}
    if sleep_record:
        health_data = {
            "sleep_efficiency": sleep_record.efficiency,
            "hrv_average": sleep_record.hrv_average,
            "resting_hr": sleep_record.resting_hr,
            "readiness_score": sleep_record.readiness_score,
            "total_sleep_hours": (sleep_record.total_sleep_seconds or 0) / 3600,
        }

    # Generate content
    content = notification_engine.generate_morning_brief(health_data, user_name=_user_display_name(current_user))

    # Send via APNs
    apns_result = await apns_client.send_push(
        device_token=token_row.token,
        title=content["title"],
        body=content["body"],
        category=content["apns"]["category"],
        thread_id=content["apns"]["thread_id"],
        interruption_level=content["apns"]["interruption_level"],
        relevance_score=content["apns"]["relevance_score"],
        collapse_id=content["apns"]["collapse_id"],
        data=content["data"],
        media_url=content.get("media_url"),
    )

    # Log the notification
    record = NotificationRecord(
        user_id=user_id,
        device_token_id=token_row.id,
        category=content["category"],
        title=content["title"],
        body=content["body"],
        payload_json=content,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return TestNotificationResponse(
        status="sent" if apns_result.get("success") else "failed",
        notification_id=record.id,
        apns_result=apns_result,
        content={"title": content["title"], "body": content["body"]},
    )


@router.post("/test/{category}")
async def send_test_by_category(
    category: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Send a test notification for any category (dev endpoint)."""
    from sqlalchemy import desc
    from app.models.health import SleepRecord

    user_id = current_user.apple_user_id
    user_name = _user_display_name(current_user)

    result = await db.execute(
        select(DeviceToken).where(
            DeviceToken.user_id == user_id,
            DeviceToken.is_active,
            DeviceToken.token != TEST_DEVICE_TOKEN,
        )
    )
    token_row = result.scalar_one_or_none()
    if not token_row:
        return {"status": "error", "message": "No device token"}

    sleep_result = await db.execute(
        select(SleepRecord).where(SleepRecord.user_id == user_id).order_by(desc(SleepRecord.date)).limit(1)
    )
    sr = sleep_result.scalar_one_or_none()
    # Only build health_data from a real record — no fabricated fallbacks
    # (Original code used hardcoded 85/65/58/80 which is mock data in prod.)
    if sr is None:
        return {"status": "error", "message": "No sleep data available"}
    health_data = {
        "sleep_efficiency": sr.efficiency,
        "hrv_average": sr.hrv_average,
        "resting_hr": sr.resting_hr,
        "readiness_score": sr.readiness_score,
    }

    if category == "coaching_nudge":
        content = content_generator.generate_coaching_nudge(health_data, user_name=user_name)
    elif category == "bedtime_coaching":
        content = content_generator.generate_bedtime_coaching(health_data, user_name=user_name)
    elif category == "morning_brief":
        content = notification_engine.generate_morning_brief(health_data, user_name=user_name)
    elif category == "streak_saver":
        content = content_generator.generate_streak_saver(3, 5, user_name=user_name)
        if not content:
            return {"status": "skipped", "message": "No streak at risk"}
    elif category == "weekly_review":
        content = content_generator.generate_weekly_review(
            {"workout_days": 4, "sleep_trend": "improving"}, user_name=user_name
        )
    elif category == "health_alert":
        content = content_generator.generate_health_alert(
            health_data, ["HRV deviation detected"]
        )
        if not content:
            return {"status": "skipped", "message": "No concerning data"}
    else:
        return {"status": "error", "message": f"Unknown category: {category}"}

    apns_result = await apns_client.send_push(
        device_token=token_row.token,
        title=content["title"],
        body=content["body"],
        category=content["apns"]["category"],
        thread_id=content["apns"]["thread_id"],
        interruption_level=content["apns"]["interruption_level"],
        relevance_score=content["apns"]["relevance_score"],
        data=content["data"],
    )

    record = NotificationRecord(
        user_id=user_id, device_token_id=token_row.id,
        category=content["category"], title=content["title"],
        body=content["body"], payload_json=content,
    )
    db.add(record)
    await db.commit()

    return {
        "status": "sent" if apns_result.get("success") else "failed",
        "content": {"title": content["title"], "body": content["body"]},
        "apns_result": apns_result,
    }


@router.get("/preferences", response_model=NotificationPreferencesResponse)
async def get_notification_preferences(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Get notification preferences for the current user."""
    result = await db.execute(
        select(NotificationPreference).where(NotificationPreference.user_id == current_user.apple_user_id)
    )
    pref = result.scalar_one_or_none()

    if not pref:
        # Return defaults (no row yet — will be created on first PUT)
        return NotificationPreferencesResponse()

    return NotificationPreferencesResponse(
        morning_brief=pref.morning_brief,
        coaching_nudge=pref.coaching_nudge,
        bedtime_coaching=pref.bedtime_coaching,
        streak_alerts=pref.streak_alerts,
        weekly_review=pref.weekly_review,
        workout_reminders=pref.workout_reminders,
        health_alerts=pref.health_alerts,
        nudge_frequency=pref.nudge_frequency,
        quiet_hours_start=pref.quiet_hours_start,
        quiet_hours_end=pref.quiet_hours_end,
    )


@router.put("/preferences", response_model=NotificationPreferencesResponse)
async def update_notification_preferences(
    prefs: NotificationPreferencesResponse,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Update notification preferences. Creates row if it doesn't exist."""
    user_id = current_user.apple_user_id
    result = await db.execute(
        select(NotificationPreference).where(NotificationPreference.user_id == user_id)
    )
    pref = result.scalar_one_or_none()

    if pref:
        pref.morning_brief = prefs.morning_brief
        pref.coaching_nudge = prefs.coaching_nudge
        pref.bedtime_coaching = prefs.bedtime_coaching
        pref.streak_alerts = prefs.streak_alerts
        pref.weekly_review = prefs.weekly_review
        pref.workout_reminders = prefs.workout_reminders
        pref.health_alerts = prefs.health_alerts
        pref.nudge_frequency = prefs.nudge_frequency
        pref.quiet_hours_start = prefs.quiet_hours_start
        pref.quiet_hours_end = prefs.quiet_hours_end
    else:
        pref = NotificationPreference(
            user_id=user_id,
            morning_brief=prefs.morning_brief,
            coaching_nudge=prefs.coaching_nudge,
            bedtime_coaching=prefs.bedtime_coaching,
            streak_alerts=prefs.streak_alerts,
            weekly_review=prefs.weekly_review,
            workout_reminders=prefs.workout_reminders,
            health_alerts=prefs.health_alerts,
            nudge_frequency=prefs.nudge_frequency,
            quiet_hours_start=prefs.quiet_hours_start,
            quiet_hours_end=prefs.quiet_hours_end,
        )
        db.add(pref)

    await db.commit()
    logger.info("Updated notification preferences for user %s", user_id[:12])

    return prefs


class NotificationOpenedRequest(BaseModel):
    notification_id: int


@router.post("/opened")
async def report_notification_opened(
    payload: NotificationOpenedRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Track when a user opens a notification (for anti-fatigue metrics).

    Reads notification_id from the JSON body. iOS (APIClient.reportNotificationOpened
    -> APINotificationOpenedRequest) sends it in the body, but this used to declare
    it as a bare scalar = a query param, so every open report 422'd and open
    tracking was silently broken. Audit P2b (round-1 H).
    """
    result = await db.execute(
        select(NotificationRecord).where(
            NotificationRecord.id == payload.notification_id,
            NotificationRecord.user_id == current_user.apple_user_id,
        )
    )
    record = result.scalar_one_or_none()
    if record:
        record.opened_at = utcnow_naive()
        await db.commit()
        return {"status": "ok"}
    return {"status": "not_found"}
