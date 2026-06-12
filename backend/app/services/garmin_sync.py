"""Garmin data sync service.

Follows the same pattern as peloton_sync.py.
"""

import logging
from datetime import date, timedelta

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.garmin import GarminToken, GarminDailyRecord
from app.models.health import HealthMetricRecord
from app.services.garmin import GarminClient

logger = logging.getLogger("meld.garmin_sync")


async def sync_user_data(db: AsyncSession, user_id: str) -> dict:
    """Sync Garmin daily health data and workouts."""
    # Check for stored credentials
    result = await db.execute(
        select(GarminToken).where(GarminToken.user_id == user_id).order_by(desc(GarminToken.created_at)).limit(1)
    )
    token = result.scalar_one_or_none()
    if not token:
        return {"status": "error", "message": "No Garmin credentials. Connect your account."}

    client = GarminClient()
    try:
        if not token.session_data:
            # Legacy tokens may have empty session_data — force reconnect
            return {"status": "error", "message": "Garmin session expired. Please re-connect."}
        restored = await client.login_from_session(token.session_data)
        if not restored:
            return {"status": "error", "message": "Garmin session invalid. Please re-connect."}
    except (ConnectionError, TimeoutError, OSError, ValueError) as e:
        # Transport or malformed-session errors. Library-specific auth errors
        # are already caught and logged by GarminClient.login_from_session.
        logger.error("Garmin session restore failed: %s", e)
        return {"status": "error", "message": "Garmin login failed. Please re-connect."}

    records_saved = 0
    today = date.today()

    for day_offset in range(7):
        target = today - timedelta(days=day_offset)
        target_str = target.isoformat()

        # Check if already synced
        existing = await db.execute(
            select(GarminDailyRecord).where(
                GarminDailyRecord.user_id == user_id,
                GarminDailyRecord.date == target_str,
            )
        )
        if existing.scalar_one_or_none():
            continue

        # Fetch data
        hr_data = await client.get_heart_rate(target)
        steps_data = await client.get_steps(target)

        resting_hr = None
        avg_hr = None
        steps = None

        if hr_data:
            resting_hr = hr_data.get("restingHeartRate")
            avg_hr = hr_data.get("averageHeartRate")

        if steps_data:
            if isinstance(steps_data, list) and steps_data:
                steps = sum(s.get("steps", 0) for s in steps_data)
            elif isinstance(steps_data, dict):
                steps = steps_data.get("totalSteps")

        record = GarminDailyRecord(
            user_id=user_id,
            date=target_str,
            steps=steps,
            avg_heart_rate=avg_hr,
            resting_heart_rate=resting_hr,
        )
        db.add(record)
        records_saved += 1

        # Write to unified HealthMetricRecord for reconciliation
        if steps:
            db.add(HealthMetricRecord(user_id=user_id, date=target_str, metric_type="steps", value=steps, unit="count", source="garmin"))
        if resting_hr:
            db.add(HealthMetricRecord(user_id=user_id, date=target_str, metric_type="resting_hr", value=resting_hr, unit="bpm", source="garmin"))

    await db.commit()

    result = {"status": "ok", "records_saved": records_saved}
    logger.info("Garmin sync complete: %s", result)
    return result
