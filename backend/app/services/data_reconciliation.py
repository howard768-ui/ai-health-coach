"""Data Reconciliation Layer.

Priority-based source selection for multi-source health data.
Industry standard approach (Apple HealthKit, Google Health Connect pattern):
pick one source's value as canonical per metric per time window.

Priority table is research-backed from polysomnography validation studies:
- PMC 10820351: Oura 5% more accurate than Apple Watch for sleep staging
- PMC 9412437: Oura/Garmin ~89% accuracy for sleep/wake
- PMC 12367097: Oura best for nocturnal HRV

Architecture: Store everything, never discard. Priority only affects
which value is marked as canonical for display and coaching.
"""

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.health import HealthMetricRecord, SourcePriority

logger = logging.getLogger("meld.reconciliation")

# Research-backed source priority defaults
# Order: highest priority first. Based on polysomnography validation studies.
DEFAULT_SOURCE_PRIORITY: dict[str, list[str]] = {
    "sleep_duration": ["oura", "apple_health", "garmin"],
    "sleep_efficiency": ["oura", "apple_health", "garmin"],
    "sleep_stages": ["oura", "apple_health", "garmin"],
    "resting_hr": ["oura", "garmin", "apple_health"],
    "hrv": ["oura", "garmin", "apple_health"],
    "steps": ["garmin", "apple_health", "oura"],
    "active_calories": ["garmin", "apple_health", "peloton"],
    "workouts": ["peloton", "garmin", "apple_health"],
    "readiness": ["oura"],  # Proprietary — do not merge across sources
    "body_battery": ["garmin"],  # Proprietary — Garmin only
}

# Divergence thresholds — flag for coach when sources disagree by more than this
DIVERGENCE_THRESHOLDS: dict[str, float] = {
    "sleep_duration": 0.15,  # 15% difference
    "sleep_efficiency": 0.10,  # 10 percentage points
    "resting_hr": 10.0,  # 10 BPM
    "hrv": 15.0,  # 15 ms
    "steps": 0.20,  # 20% difference
}


@dataclass
class CanonicalMetric:
    """The winning value for a metric at a specific time."""
    metric_type: str
    value: float
    unit: str
    source: str
    date: str
    is_primary: bool  # True if from highest-priority source, False if fallback
    confidence: str  # "primary", "fallback", "only_source"


@dataclass
class DivergenceAlert:
    """Alert when multiple sources meaningfully disagree."""
    metric_type: str
    date: str
    sources: dict[str, float]  # source → value
    divergence_pct: float
    message: str  # Human-readable explanation for coach


async def get_source_priority(
    db: AsyncSession, user_id: str, metric_category: str
) -> list[str]:
    """Get the source priority list for a metric category.

    Checks for user-customized priority first, falls back to research defaults.
    """
    result = await db.execute(
        select(SourcePriority).where(
            SourcePriority.user_id == user_id,
            SourcePriority.metric_category == metric_category,
        )
    )
    user_pref = result.scalar_one_or_none()
    if user_pref and user_pref.priority_order:
        import json
        try:
            return json.loads(user_pref.priority_order)
        except (json.JSONDecodeError, TypeError):
            pass

    return DEFAULT_SOURCE_PRIORITY.get(metric_category, ["oura", "apple_health", "garmin"])


async def get_canonical_value(
    db: AsyncSession, user_id: str, metric_type: str, date: str
) -> CanonicalMetric | None:
    """Get the best-quality value for a metric on a given date.

    Walks the priority chain: if highest-priority source has data, use it.
    If not, fall back to the next source. Marks confidence accordingly.
    """
    priority = await get_source_priority(db, user_id, metric_type)

    for i, source in enumerate(priority):
        result = await db.execute(
            select(HealthMetricRecord).where(
                HealthMetricRecord.user_id == user_id,
                HealthMetricRecord.metric_type == metric_type,
                HealthMetricRecord.date == date,
                HealthMetricRecord.source == source,
            )
        )
        record = result.scalar_one_or_none()
        if record:
            confidence = "primary" if i == 0 else "fallback"
            return CanonicalMetric(
                metric_type=metric_type,
                value=record.value,
                unit=record.unit or "",
                source=source,
                date=date,
                is_primary=(i == 0),
                confidence=confidence,
            )

    return None


async def detect_divergence(
    db: AsyncSession, user_id: str, metric_type: str, date: str
) -> DivergenceAlert | None:
    """Check if multiple sources report meaningfully different values.

    Returns a DivergenceAlert if divergence exceeds the threshold for this metric.
    """
    result = await db.execute(
        select(HealthMetricRecord).where(
            HealthMetricRecord.user_id == user_id,
            HealthMetricRecord.metric_type == metric_type,
            HealthMetricRecord.date == date,
        )
    )
    records = result.scalars().all()

    if len(records) < 2:
        return None

    sources = {r.source: r.value for r in records}
    values = list(sources.values())
    max_val = max(values)
    min_val = min(values)

    if max_val == 0:
        return None

    threshold = DIVERGENCE_THRESHOLDS.get(metric_type)
    if not threshold:
        return None

    # Calculate divergence
    if metric_type in ("resting_hr", "hrv"):
        # Absolute difference for BPM/ms metrics
        divergence = max_val - min_val
        is_divergent = divergence > threshold
    else:
        # Percentage difference for ratio metrics
        divergence = (max_val - min_val) / max_val
        is_divergent = divergence > threshold

    if not is_divergent:
        return None

    # Build human-readable message for coach
    source_list = ", ".join(f"{s}: {v}" for s, v in sources.items())
    message = (
        f"Your {metric_type.replace('_', ' ')} data varied between devices on {date} "
        f"({source_list}). Using the most reliable reading."
    )

    return DivergenceAlert(
        metric_type=metric_type,
        date=date,
        sources=sources,
        divergence_pct=divergence,
        message=message,
    )


async def reconcile_day(
    db: AsyncSession, user_id: str, date: str
) -> dict[str, CanonicalMetric]:
    """Run reconciliation for all metrics on a given date.

    Marks the winning record as is_canonical=True in the database.
    Returns dict of metric_type → CanonicalMetric.
    """
    metrics = list(DEFAULT_SOURCE_PRIORITY.keys())
    canonical = {}

    for metric_type in metrics:
        # Reset all records for this metric/date to non-canonical
        result = await db.execute(
            select(HealthMetricRecord).where(
                HealthMetricRecord.user_id == user_id,
                HealthMetricRecord.metric_type == metric_type,
                HealthMetricRecord.date == date,
            )
        )
        for record in result.scalars().all():
            record.is_canonical = False

        # Find the canonical value
        best = await get_canonical_value(db, user_id, metric_type, date)
        if best:
            canonical[metric_type] = best
            # Mark the winning record
            win_result = await db.execute(
                select(HealthMetricRecord).where(
                    HealthMetricRecord.user_id == user_id,
                    HealthMetricRecord.metric_type == metric_type,
                    HealthMetricRecord.date == date,
                    HealthMetricRecord.source == best.source,
                )
            )
            winner = win_result.scalar_one_or_none()
            if winner:
                winner.is_canonical = True

    await db.commit()
    logger.info("Reconciled %d metrics for %s on %s", len(canonical), user_id, date)
    return canonical
