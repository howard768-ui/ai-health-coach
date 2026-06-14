"""Oura data sync service.

Refactored from the health router's inline sync logic into a reusable service
callable from both the API endpoint and the background scheduler.
Handles token refresh, sleep data, readiness, and HRV.
"""

import logging
from datetime import date, datetime, timedelta

import httpx
from sqlalchemy import select, desc
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.health import OuraToken, SleepRecord, HealthMetricRecord
from app.services.oura import OuraClient
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.oura_sync")


async def ensure_valid_token(db: AsyncSession, user_id: str) -> str | None:
    """Get a valid Oura access token, refreshing if expired.

    Returns the valid access_token string, or None if no token exists
    or refresh failed (user revoked access).
    """
    result = await db.execute(
        select(OuraToken)
        .where(OuraToken.user_id == user_id)
        .order_by(desc(OuraToken.created_at))
        .limit(1)
    )
    token = result.scalar_one_or_none()
    if not token:
        # Logs internal user_id only.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        logger.warning("No Oura token found for user %s", user_id[:12] + "...")
        return None

    # Check if token expires within 5 minutes
    if token.expires_at and token.expires_at < utcnow_naive() + timedelta(minutes=5):
        logger.info("Oura token expired or expiring soon — refreshing")
        try:
            client = OuraClient()
            new_tokens = await client.refresh_access_token(token.refresh_token)
            token.access_token = new_tokens["access_token"]
            token.refresh_token = new_tokens.get("refresh_token", token.refresh_token)
            token.expires_at = utcnow_naive() + timedelta(seconds=new_tokens.get("expires_in", 86400))
            await db.commit()
            logger.info("Oura token refreshed successfully")
        except (httpx.HTTPError, KeyError, ValueError, SQLAlchemyError) as e:
            logger.error("Oura token refresh failed: %s", e)
            try:
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("oura_action", "token_refresh")
                    scope.set_tag("user_id_prefix", (user_id or "")[:12])
                    sentry_sdk.capture_exception(e)
            except Exception:  # noqa: BLE001 -- never let Sentry crash a sync
                logger.debug("Sentry capture failed (non-fatal)", exc_info=True)
            return None

    return token.access_token


async def sync_user_data(db: AsyncSession, user_id: str) -> dict:
    """Sync latest Oura data for a user.

    Pulls sleep, readiness, and HRV data for the last 7 days.
    Deduplicates against existing records.
    Returns status dict with records_saved count.
    """
    access_token = await ensure_valid_token(db, user_id)
    if not access_token:
        return {"status": "error", "message": "No valid Oura token"}

    client = OuraClient(access_token=access_token)
    start = date.today() - timedelta(days=7)
    end = date.today() + timedelta(days=1)  # Include today (Oura API end date is exclusive)

    try:
        sleep_data = await client.get_daily_sleep(start, end)
        readiness_data = await client.get_daily_readiness(start, end)
        sleep_sessions = await client.get_sleep_sessions(start, end)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error("Oura API returned 401 — token may be revoked")
            return {"status": "error", "message": "Oura access revoked. Reconnect your ring."}
        logger.error("Oura API error: %s", e)
        return {"status": "error", "message": f"Oura API error: {e}"}
    except httpx.HTTPError as e:
        logger.error("Oura API network error: %s", e)
        return {"status": "error", "message": f"Oura API error: {e}"}
    except ValueError as e:
        logger.error("Oura API response parse error: %s", e)
        return {"status": "error", "message": "Oura returned unexpected data"}

    # Index sleep sessions by day for duration data
    session_by_day: dict[str, dict] = {}
    for session in sleep_sessions.get("data", []):
        sday = session.get("day", "")
        # Pick the longest sleep session per day (main sleep, not naps)
        existing_duration = session_by_day.get(sday, {}).get("total_sleep_duration", 0) or 0
        session_duration = session.get("total_sleep_duration", 0) or 0
        if session_duration > existing_duration:
            session_by_day[sday] = session

    records_saved = 0
    for day in sleep_data.get("data", []):
        day_date = day.get("day", "")

        # Skip if record already exists
        existing = await db.execute(
            select(SleepRecord).where(
                SleepRecord.user_id == user_id,
                SleepRecord.date == day_date,
            )
        )
        if existing.scalar_one_or_none():
            continue

        # Find matching readiness score
        readiness_score = None
        for r in readiness_data.get("data", []):
            if r.get("day") == day_date:
                readiness_score = r.get("score")
                break

        # Get session data for durations AND true efficiency percentage.
        # daily_sleep.contributors.efficiency is a 0-100 score that contributes
        # to the daily sleep score — NOT the actual sleep efficiency percentage.
        # The real percentage comes from the sleep_sessions endpoint.
        session = session_by_day.get(day_date, {})
        session_efficiency = session.get("efficiency")  # 0-100 percentage

        # Parse bedtime timestamps from session data
        bedtime_start = _parse_time(session.get("bedtime_start"))
        bedtime_end = _parse_time(session.get("bedtime_end"))

        record = SleepRecord(
            user_id=user_id,
            date=day_date,
            efficiency=session_efficiency,
            total_sleep_seconds=session.get("total_sleep_duration"),
            deep_sleep_seconds=session.get("deep_sleep_duration"),
            rem_sleep_seconds=session.get("rem_sleep_duration"),
            light_sleep_seconds=session.get("light_sleep_duration"),
            hrv_average=None,  # Patched by sync_hrv below
            resting_hr=session.get("lowest_heart_rate"),
            readiness_score=readiness_score,
            bedtime_start=bedtime_start,
            bedtime_end=bedtime_end,
            raw_json=str(day),
        )
        db.add(record)
        records_saved += 1

        # Also write to unified HealthMetricRecord for reconciliation
        if session_efficiency is not None:
            db.add(HealthMetricRecord(user_id=user_id, date=day_date, metric_type="sleep_efficiency", value=session_efficiency, unit="%", source="oura"))
        # Sleep duration comes from session data, not daily_sleep
        sleep_duration_secs = session.get("total_sleep_duration")
        if sleep_duration_secs:
            db.add(HealthMetricRecord(user_id=user_id, date=day_date, metric_type="sleep_duration", value=sleep_duration_secs / 3600, unit="hours", source="oura"))
        # Resting HR comes from session data
        session_rhr = session.get("lowest_heart_rate")
        if session_rhr:
            db.add(HealthMetricRecord(user_id=user_id, date=day_date, metric_type="resting_hr", value=session_rhr, unit="bpm", source="oura"))
        if readiness_score:
            db.add(HealthMetricRecord(user_id=user_id, date=day_date, metric_type="readiness", value=readiness_score, unit="score", source="oura"))

    await db.commit()

    # Sync HRV from heartrate endpoint
    hrv_updated = await sync_hrv(client, db, user_id, start, end)

    # Bump the token's last_synced_at so the dashboard on-demand refresh
    # logic can throttle us. This is updated on every successful call
    # regardless of how many records were written — the semantic is
    # "we successfully contacted Oura", not "we got new rows".
    token_result = await db.execute(
        select(OuraToken)
        .where(OuraToken.user_id == user_id)
        .order_by(desc(OuraToken.created_at))
        .limit(1)
    )
    token_row = token_result.scalar_one_or_none()
    if token_row is not None:
        token_row.last_synced_at = utcnow_naive()
        await db.commit()

    result = {
        "status": "ok",
        "records_saved": records_saved,
        "hrv_updated": hrv_updated,
        "days_synced": len(sleep_data.get("data", [])),
    }
    logger.info("Oura sync complete: %s", result)
    return result


async def sync_hrv(
    client: OuraClient, db: AsyncSession, user_id: str, start: date, end: date
) -> int:
    """Pull HRV data from Oura heartrate endpoint and patch SleepRecords.

    Computes average HRV from heart rate samples during the sleep window.
    Returns count of records updated.
    """
    try:
        hr_data = await client.get_heartrate(start, end)
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to fetch HRV data: %s", e)
        return 0

    # Group HR samples by date
    samples_by_date: dict[str, list[float]] = {}
    for sample in hr_data.get("data", []):
        ts = sample.get("timestamp", "")
        bpm = sample.get("bpm")
        source = sample.get("source", "")
        if not ts or not bpm:
            continue
        # Only use resting/sleep samples for HRV proxy
        if source in ("rest", "sleep"):
            day_str = ts[:10]  # "YYYY-MM-DD"
            samples_by_date.setdefault(day_str, []).append(bpm)

    # Patch SleepRecords that have hrv_average = None
    updated = 0
    records = await db.execute(
        select(SleepRecord).where(
            SleepRecord.user_id == user_id,
            SleepRecord.hrv_average.is_(None),
            SleepRecord.date >= str(start),
            SleepRecord.date <= str(end),
        )
    )
    for record in records.scalars().all():
        samples = samples_by_date.get(record.date, [])
        if samples:
            # NOTE: This stores average resting HR during sleep as an HRV proxy.
            # True RMSSD HRV requires the Oura v2 HRV endpoint or raw inter-beat
            # interval data, which is not available via the daily_sleep scope.
            # This proxy correlates with HRV trends but is not clinically accurate.
            avg_resting = sum(samples) / len(samples)
            record.hrv_average = round(avg_resting, 1)
            updated += 1

    if updated:
        await db.commit()
        logger.info("Updated HRV for %d records", updated)

    return updated


def _parse_time(iso_timestamp: str | None) -> str | None:
    """Parse an ISO 8601 timestamp into HH:MM format."""
    if not iso_timestamp:
        return None
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except (ValueError, AttributeError):
        return None
