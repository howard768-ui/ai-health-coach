"""Unified health data loader.

Single source of truth for all screens (Dashboard, Coach, Trends, Notifications).
Reads from the reconciled HealthMetricRecord table first (multi-source, best-quality),
falls back to SleepRecord for backward compatibility.

This is the key differentiator: data from Oura, Apple Health, Garmin, and Peloton
flows through the reconciliation layer, and every feature reads from here.
"""

import logging
from datetime import date, timedelta

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.health import HealthMetricRecord, SleepRecord

logger = logging.getLogger("meld.health_data")


SLEEP_METRIC_KEYS = ("sleep_efficiency", "sleep_duration", "hrv", "resting_hr", "readiness")


async def get_latest_health_data(db: AsyncSession, user_id: str) -> dict:
    """Get the latest reconciled health data for coaching and display.

    Reads canonical values from HealthMetricRecord (multi-source reconciled),
    falls back to SleepRecord if no reconciled data exists.

    Merges today's data (e.g. live step count from Apple Health) with last
    night's sleep data so the dashboard doesn't show 0% sleep efficiency when
    today's step record arrives before tonight's sleep.

    Returns dict with all metrics the coach engine needs.
    """
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Today may only have step data from Apple Health (no sleep until tomorrow morning).
    # Always merge yesterday's record on top when today lacks sleep metrics.
    today_metrics = await _get_reconciled_metrics(db, user_id, today) or {}
    yesterday_metrics = await _get_reconciled_metrics(db, user_id, yesterday) or {}

    # Start from yesterday (sleep + recovery baseline) and overlay today
    # (steps, active calories, anything that accumulates during the day).
    reconciled: dict = {}
    # Merge sources from both days — yesterday's sources are the fallback
    # for any metric that carries forward (e.g. steps when today has none).
    merged_sources: dict = {}
    if yesterday_metrics:
        reconciled.update(yesterday_metrics)
        merged_sources.update(yesterday_metrics.get("_sources", {}))
    if today_metrics:
        reconciled.update(today_metrics)
        merged_sources.update(today_metrics.get("_sources", {}))
        # If today is overwriting yesterday's sleep metrics with stale/missing
        # values, restore yesterday's sleep fields.
        for key in SLEEP_METRIC_KEYS:
            if key not in today_metrics and key in yesterday_metrics:
                reconciled[key] = yesterday_metrics[key]
                # Also restore the source attribution for carried-forward metrics
                yesterday_sources = yesterday_metrics.get("_sources", {})
                if key in yesterday_sources:
                    merged_sources[key] = yesterday_sources[key]

    if reconciled:
        # Get 7-day baselines from reconciled data
        baselines = await _get_reconciled_baselines(db, user_id, days=7)

        # Deep sleep isn't in HealthMetricRecord yet — pull from SleepRecord
        deep_sleep_min = 0
        sr = await db.execute(
            select(SleepRecord)
            .where(SleepRecord.user_id == user_id)
            .order_by(desc(SleepRecord.date))
            .limit(1)
        )
        latest_sleep = sr.scalar_one_or_none()
        if latest_sleep and latest_sleep.deep_sleep_seconds:
            deep_sleep_min = round(latest_sleep.deep_sleep_seconds / 60)

        return {
            "sleep_efficiency": reconciled.get("sleep_efficiency", 0),
            "sleep_duration_hours": reconciled.get("sleep_duration", 0),
            "deep_sleep_minutes": deep_sleep_min,
            "hrv_average": reconciled.get("hrv", 0),
            "baseline_hrv": baselines.get("hrv", 0),
            "resting_hr": reconciled.get("resting_hr", 0),
            "baseline_rhr": baselines.get("resting_hr", 0),
            "readiness_score": reconciled.get("readiness", 0),
            "steps": reconciled.get("steps", 0),
            "active_calories": reconciled.get("active_calories", 0),
            "data_sources": merged_sources,
            "baseline_days": baselines.get("_days", 0),
        }

    # Fallback: read directly from SleepRecord (Oura-only, pre-reconciliation)
    logger.info(
        "No reconciled data; falling back to SleepRecord for user %s",
        user_id[:12] + "...",
    )
    return await _get_sleep_record_data(db, user_id)


async def get_health_data_for_date(db: AsyncSession, user_id: str, target_date: str) -> dict:
    """Get reconciled health data for a specific date."""
    reconciled = await _get_reconciled_metrics(db, user_id, target_date)
    if reconciled:
        return reconciled
    return {}


async def get_health_data_range(db: AsyncSession, user_id: str, days: int = 7) -> list[dict]:
    """Get reconciled health data for a date range (for trends)."""
    start = (date.today() - timedelta(days=days - 1)).isoformat()

    result = await db.execute(
        select(HealthMetricRecord)
        .where(
            HealthMetricRecord.user_id == user_id,
            HealthMetricRecord.is_canonical,
            HealthMetricRecord.date >= start,
        )
        .order_by(HealthMetricRecord.date)
    )
    records = result.scalars().all()

    # Group by date
    by_date: dict[str, dict] = {}
    for r in records:
        if r.date not in by_date:
            by_date[r.date] = {"date": r.date}
        by_date[r.date][r.metric_type] = r.value
        by_date[r.date][f"{r.metric_type}_source"] = r.source

    return list(by_date.values())


# ── Private Helpers ─────────────────────────────────────────

async def _get_reconciled_metrics(db: AsyncSession, user_id: str, target_date: str) -> dict | None:
    """Get all canonical metrics for a specific date."""
    result = await db.execute(
        select(HealthMetricRecord).where(
            HealthMetricRecord.user_id == user_id,
            HealthMetricRecord.date == target_date,
            HealthMetricRecord.is_canonical,
        )
    )
    records = result.scalars().all()
    if not records:
        return None

    metrics = {}
    sources = {}
    for r in records:
        metrics[r.metric_type] = r.value
        sources[r.metric_type] = r.source

    metrics["_sources"] = sources
    return metrics


async def _get_reconciled_baselines(db: AsyncSession, user_id: str, days: int = 7) -> dict:
    """Compute 7-day rolling averages from canonical metrics.

    Returns a dict with metric averages plus a '_days' key indicating
    how many distinct days contributed. Consumers should treat baselines
    with < 3 days as unreliable (too noisy for deviation alerts).
    """
    start = (date.today() - timedelta(days=days - 1)).isoformat()

    result = await db.execute(
        select(HealthMetricRecord).where(
            HealthMetricRecord.user_id == user_id,
            HealthMetricRecord.is_canonical,
            HealthMetricRecord.date >= start,
        )
    )
    records = result.scalars().all()

    # Group values by metric type
    values_by_type: dict[str, list[float]] = {}
    distinct_dates: set[str] = set()
    for r in records:
        values_by_type.setdefault(r.metric_type, []).append(r.value)
        distinct_dates.add(r.date)

    # Compute averages
    baselines = {}
    for metric_type, values in values_by_type.items():
        if values:
            baselines[metric_type] = round(sum(values) / len(values), 1)

    baselines["_days"] = len(distinct_dates)
    return baselines


async def _get_sleep_record_data(db: AsyncSession, user_id: str) -> dict:
    """Fallback: read directly from SleepRecord (Oura-only)."""
    result = await db.execute(
        select(SleepRecord)
        .where(SleepRecord.user_id == user_id)
        .order_by(desc(SleepRecord.date))
        .limit(7)
    )
    records = list(result.scalars().all())
    if not records:
        return {}

    latest = records[0]
    avg_rhr = sum(r.resting_hr or 0 for r in records if r.resting_hr) / max(1, sum(1 for r in records if r.resting_hr))
    avg_hrv = sum(r.hrv_average or 0 for r in records if r.hrv_average) / max(1, sum(1 for r in records if r.hrv_average))

    return {
        "sleep_efficiency": latest.efficiency or 0,
        "sleep_duration_hours": round((latest.total_sleep_seconds or 0) / 3600, 1),
        "deep_sleep_minutes": round((latest.deep_sleep_seconds or 0) / 60),
        "hrv_average": latest.hrv_average or 0,
        "baseline_hrv": round(avg_hrv, 1),
        "resting_hr": latest.resting_hr or 0,
        "baseline_rhr": round(avg_rhr, 1),
        "readiness_score": latest.readiness_score or 0,
        "data_sources": {"all": "oura"},
    }
