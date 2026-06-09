"""Anti-fatigue mechanisms for push notifications.

Research-backed rules to prevent notification fatigue:
- Auto-throttle: 3 consecutive unopened → reduce frequency by 50%
- Auto-disable: 7 consecutive ignored → pause the category
- Daily budget: max 4 visible PNs per day across all categories
- Quiet hours: no notifications during user's sleep window

Sources:
- MyFitnessPal auto-disable pattern (Taplytics)
- Retenshun 2025: 6+/week = 3.4x uninstall risk
- Kidman et al. 2024: 70% abandon within 100 days, notification annoyance is a factor
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.notification import NotificationRecord, NotificationPreference
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.anti_fatigue")

# Hard limits
MAX_DAILY_NOTIFICATIONS = 4


def _user_now() -> datetime:
    """Return the current time in the user's configured timezone.

    P2-15 fix: previously used utcnow_naive() everywhere, which meant
    "today" rolled over at UTC midnight (8pm EST → double notifications at
    midnight local time). Now reads USER_TIMEZONE env var.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    from app.config import settings as _settings
    try:
        tz = ZoneInfo(_settings.user_timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        tz = ZoneInfo("UTC")
    return datetime.now(tz)


async def check_daily_budget(db: AsyncSession, user_id: str) -> bool:
    """Check if the user has hit the daily notification cap.

    Returns True if we CAN send, False if budget is exhausted.
    """
    # Day boundary in USER's timezone, not UTC (P2-15)
    user_now = _user_now()
    today_start_local = user_now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Convert to naive UTC for the DB query (DB stores naive UTC timestamps)
    from datetime import timezone as _tz
    today_start = today_start_local.astimezone(_tz.utc).replace(tzinfo=None)
    result = await db.execute(
        select(func.count(NotificationRecord.id)).where(
            NotificationRecord.user_id == user_id,
            NotificationRecord.sent_at >= today_start,
        )
    )
    count = result.scalar() or 0
    if count >= MAX_DAILY_NOTIFICATIONS:
        logger.info("Daily budget exhausted (%d/%d) for user %s", count, MAX_DAILY_NOTIFICATIONS, user_id)
        return False
    return True


async def check_throttle(db: AsyncSession, user_id: str, category: str) -> bool:
    """Check if a notification category should be throttled.

    If the last 3 notifications in this category were all ignored (no opened_at),
    skip every other send (50% frequency reduction).

    Returns True if we CAN send, False if throttled.
    """
    result = await db.execute(
        select(NotificationRecord)
        .where(
            NotificationRecord.user_id == user_id,
            NotificationRecord.category == category,
        )
        .order_by(desc(NotificationRecord.sent_at))
        .limit(3)
    )
    recent = result.scalars().all()

    if len(recent) < 3:
        return True  # Not enough history to throttle

    all_ignored = all(r.opened_at is None for r in recent)
    if not all_ignored:
        return True

    # 3 consecutive ignored — skip every other send
    # Use the count of records in this category to determine odd/even
    count_result = await db.execute(
        select(func.count(NotificationRecord.id)).where(
            NotificationRecord.user_id == user_id,
            NotificationRecord.category == category,
        )
    )
    total = count_result.scalar() or 0
    should_skip = total % 2 == 0  # Skip on even counts
    if should_skip:
        logger.info("Throttling %s for user %s (3 consecutive ignored, skipping this send)", category, user_id)
    return not should_skip


async def check_auto_disable(db: AsyncSession, user_id: str, category: str) -> bool:
    """Check if a category should be auto-disabled.

    If 7 consecutive notifications in this category were ignored,
    disable the category in preferences.

    Returns True if the category is still active, False if auto-disabled.
    """
    result = await db.execute(
        select(NotificationRecord)
        .where(
            NotificationRecord.user_id == user_id,
            NotificationRecord.category == category,
        )
        .order_by(desc(NotificationRecord.sent_at))
        .limit(7)
    )
    recent = result.scalars().all()

    if len(recent) < 7:
        return True  # Not enough history

    all_ignored = all(r.opened_at is None for r in recent)
    if not all_ignored:
        return True

    # Auto-disable this category
    pref_result = await db.execute(
        select(NotificationPreference).where(NotificationPreference.user_id == user_id)
    )
    pref = pref_result.scalar_one_or_none()
    if pref and hasattr(pref, category):
        setattr(pref, category, False)
        await db.commit()
        logger.info("Auto-disabled %s for user %s (7 consecutive ignored)", category, user_id)
        return False

    return True


async def check_quiet_hours(db: AsyncSession, user_id: str) -> bool:
    """Check if current time falls within the user's quiet hours.

    P2-15: quiet hours are stored as local wall-clock strings like "22:00",
    so we compare against the user's local time, not UTC.

    Returns True if we CAN send, False if within quiet hours.
    """
    pref_result = await db.execute(
        select(NotificationPreference).where(NotificationPreference.user_id == user_id)
    )
    pref = pref_result.scalar_one_or_none()
    if not pref:
        return True  # No preferences set, default to allowing

    now_time = _user_now().strftime("%H:%M")
    start = pref.quiet_hours_start
    end = pref.quiet_hours_end

    if start <= end:
        # Simple range (e.g., 01:00 - 06:00)
        if start <= now_time <= end:
            logger.info("Within quiet hours (%s-%s) for user %s", start, end, user_id)
            return False
    else:
        # Wraps midnight (e.g., 22:00 - 07:00)
        if now_time >= start or now_time <= end:
            logger.info("Within quiet hours (%s-%s) for user %s", start, end, user_id)
            return False

    return True


async def check_preference(db: AsyncSession, user_id: str, category: str) -> bool:
    """Check if a notification category is enabled in user preferences.

    Returns True if enabled (or no preferences exist), False if disabled.
    """
    pref_result = await db.execute(
        select(NotificationPreference).where(NotificationPreference.user_id == user_id)
    )
    pref = pref_result.scalar_one_or_none()
    if not pref:
        return True  # Defaults: everything on

    if hasattr(pref, category):
        enabled = getattr(pref, category)
        if not enabled:
            logger.info(
                "Category %s disabled in preferences for user %s",
                category,
                user_id[:12] + "...",
            )
        return enabled

    return True


async def can_send(db: AsyncSession, user_id: str, category: str) -> bool:
    """Run all anti-fatigue checks. Returns True if notification should be sent."""
    if not await check_preference(db, user_id, category):
        return False
    if not await check_quiet_hours(db, user_id):
        return False
    if not await check_daily_budget(db, user_id):
        return False
    if not await check_auto_disable(db, user_id, category):
        return False
    if not await check_throttle(db, user_id, category):
        return False
    return True
