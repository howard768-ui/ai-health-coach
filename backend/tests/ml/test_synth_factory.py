"""Phase 4.5 Commit 4: factory orchestrator tests.

End-to-end: build a small cohort, let the factory write rows, read them
back, and pin the load-bearing invariants:

- Every emitted row carries ``is_synthetic=True``.
- Every ``date`` column is a ``String(10)`` ``YYYY-MM-DD``.
- Seeded runs are deterministic.
- The manifest's ``user_ids`` match the rows actually written.
- Unknown generator names raise ``ValueError``; an honest message.

Run: ``cd backend && uv run python -m pytest tests/ml/test_synth_factory.py -v``
"""

from __future__ import annotations

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-must-be-long-enough-for-hs256-aaaaaaaa")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")


import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models import ml_synth as _ml_synth_models  # noqa: F401  (registers MLSynthRun on Base.metadata)
from app.models.health import ActivityRecord, HealthMetricRecord, SleepRecord
from app.models.meal import FoodItemRecord, MealRecord
from app.models.ml_synth import MLSynthRun
from ml.synth.factory import generate_cohort


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
    await engine.dispose()


# ─────────────────────────────────────────────────────────────────────────
# Manifest invariants
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_manifest_with_matching_user_ids(db: AsyncSession) -> None:
    manifest = await generate_cohort(db, n_users=3, days=14, seed=42)
    assert manifest.n_users == 3
    assert len(manifest.user_ids) == 3
    assert all(uid.startswith("synth-") for uid in manifest.user_ids)
    assert manifest.days == 14
    assert manifest.generator == "parametric"
    assert manifest.seed == 42
    # Date strings, not datetime objects.
    assert len(manifest.start_date) == 10
    assert len(manifest.end_date) == 10
    assert manifest.start_date < manifest.end_date


@pytest.mark.asyncio
async def test_run_id_is_uuid(db: AsyncSession) -> None:
    import uuid as _uuid

    manifest = await generate_cohort(db, n_users=1, days=5, seed=1)
    _uuid.UUID(manifest.run_id)  # raises if not a valid uuid


# ─────────────────────────────────────────────────────────────────────────
# Commit 6: ml_synth_runs manifest persistence
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manifest_persists_to_ml_synth_runs(db: AsyncSession) -> None:
    """Commit 6: factory writes the manifest to ml_synth_runs with
    run_id matching the returned CohortManifest so a scheduler log
    line referencing the run id can be cross-referenced to both the
    DB row and the raw rows it produced."""
    import json as _json

    manifest = await generate_cohort(db, n_users=3, days=7, seed=42)
    await db.flush()

    row = (
        await db.execute(select(MLSynthRun).where(MLSynthRun.run_id == manifest.run_id))
    ).scalar_one()
    assert row.run_id == manifest.run_id
    assert row.seed == 42
    assert row.generator == "parametric"
    assert row.n_users == 3
    assert row.days == 7
    assert row.start_date == manifest.start_date
    assert row.end_date == manifest.end_date
    assert row.created_at == manifest.created_at
    assert row.adversarial_fraction == manifest.adversarial_fraction
    assert _json.loads(row.user_ids_json) == manifest.user_ids


@pytest.mark.asyncio
async def test_manifest_row_and_raw_rows_share_a_transaction(db: AsyncSession) -> None:
    """Audit invariant: if the caller rolls back, BOTH the ml_synth_runs
    row AND every is_synthetic row disappear together. Rollback must
    not leave a dangling manifest pointing at no data."""
    manifest = await generate_cohort(db, n_users=2, days=5, seed=7)
    await db.rollback()

    # Manifest row gone.
    manifest_rows = (
        await db.execute(select(MLSynthRun).where(MLSynthRun.run_id == manifest.run_id))
    ).scalars().all()
    assert manifest_rows == []
    # Raw rows gone too (already covered by test_factory_does_not_commit
    # but pin the same session here for clarity).
    for model in (SleepRecord, ActivityRecord, MealRecord, FoodItemRecord, HealthMetricRecord):
        count = (
            await db.execute(select(func.count()).select_from(model))
        ).scalar_one()
        assert count == 0, f"{model.__name__} not cleared on rollback"


@pytest.mark.asyncio
async def test_every_manifest_run_id_is_unique(db: AsyncSession) -> None:
    """Two back-to-back calls on the same session land distinct run_ids."""
    a = await generate_cohort(db, n_users=1, days=5, seed=1)
    b = await generate_cohort(db, n_users=1, days=5, seed=1)
    # Same seed (identical cohort) but distinct run ids so the audit log
    # never collapses two runs into one.
    assert a.run_id != b.run_id
    rows = (await db.execute(select(MLSynthRun))).scalars().all()
    run_ids = [r.run_id for r in rows]
    assert len(set(run_ids)) == len(run_ids) == 2


@pytest.mark.asyncio
async def test_unseeded_run_stores_null_seed(db: AsyncSession) -> None:
    manifest = await generate_cohort(db, n_users=1, days=5, seed=None)
    await db.flush()
    row = (
        await db.execute(select(MLSynthRun).where(MLSynthRun.run_id == manifest.run_id))
    ).scalar_one()
    assert row.seed is None


# ─────────────────────────────────────────────────────────────────────────
# Rows written
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_writes_sleep_activity_meal_and_metric_rows(db: AsyncSession) -> None:
    """Full round-trip: the factory writes to every raw table the prep
    doc calls out."""
    await generate_cohort(db, n_users=2, days=10, seed=7)
    await db.flush()

    sleep_count = (await db.execute(select(func.count()).select_from(SleepRecord))).scalar_one()
    activity_count = (
        await db.execute(select(func.count()).select_from(ActivityRecord))
    ).scalar_one()
    meal_count = (await db.execute(select(func.count()).select_from(MealRecord))).scalar_one()
    food_count = (
        await db.execute(select(func.count()).select_from(FoodItemRecord))
    ).scalar_one()
    metric_count = (
        await db.execute(select(func.count()).select_from(HealthMetricRecord))
    ).scalar_one()

    # 2 users x 10 days = 20 potential activity rows; always emitted.
    assert activity_count == 20
    # Sleep and meal rows subject to missingness; just assert >0.
    assert sleep_count > 0
    assert meal_count > 0
    # Each meal has exactly one food item in the current generator.
    assert food_count == meal_count
    # HealthMetricRecord: up to 6 metric_types per user-day (hrv, rhr,
    # sleep_eff, sleep_duration, readiness, steps). Missingness prunes
    # some but we should see at least 3 * 20 = 60.
    assert metric_count >= 60


@pytest.mark.asyncio
async def test_every_row_tagged_is_synthetic_true(db: AsyncSession) -> None:
    """Invariant 1 in the factory docstring. Flipping a row to False
    anywhere would let synth leak into production aggregates."""
    await generate_cohort(db, n_users=3, days=10, seed=7)
    await db.flush()

    for model in (SleepRecord, ActivityRecord, MealRecord, FoodItemRecord, HealthMetricRecord):
        total = (await db.execute(select(func.count()).select_from(model))).scalar_one()
        synth = (
            await db.execute(
                select(func.count())
                .select_from(model)
                .where(model.is_synthetic.is_(True))
            )
        ).scalar_one()
        assert total == synth, f"{model.__name__}: {total - synth} rows not tagged synth"


@pytest.mark.asyncio
async def test_date_fields_are_iso_strings_not_datetimes(db: AsyncSession) -> None:
    """Load-bearing invariant #1 from the Phase 4.5 prep doc."""
    await generate_cohort(db, n_users=1, days=5, seed=7)
    await db.flush()
    sleep_rows = (await db.execute(select(SleepRecord))).scalars().all()
    assert sleep_rows
    for r in sleep_rows:
        assert isinstance(r.date, str)
        assert len(r.date) == 10
        assert r.date[4] == "-" and r.date[7] == "-"

    meal_rows = (await db.execute(select(MealRecord))).scalars().all()
    for r in meal_rows:
        assert isinstance(r.date, str)
        assert len(r.date) == 10


@pytest.mark.asyncio
async def test_sleep_durations_stored_in_seconds_not_minutes(db: AsyncSession) -> None:
    """Load-bearing invariant #2 from the Phase 4.5 prep doc.
    sleep_duration in HealthMetricRecord is stored as seconds; a typical
    7-hour sleep is ~25_200 seconds, way above any minutes-based value."""
    await generate_cohort(db, n_users=2, days=10, seed=7)
    await db.flush()
    rows = (
        await db.execute(
            select(HealthMetricRecord).where(
                HealthMetricRecord.metric_type == "sleep_duration"
            )
        )
    ).scalars().all()
    assert rows
    values = [r.value for r in rows]
    assert all(v > 3 * 3600 for v in values), (
        f"sleep_duration values look too small (minutes?): {values[:5]}"
    )
    assert all(v < 12 * 3600 for v in values), (
        f"sleep_duration values look too large: {values[:5]}"
    )


@pytest.mark.asyncio
async def test_dinner_meals_vary_by_hour_for_dinner_hour_feature(db: AsyncSession) -> None:
    """Phase 4.5 Commit 2's dinner_hour feature reads
    ``MealRecord.created_at.hour`` for meal_type='dinner'. If every
    synth user ate at the same hour, that feature would have zero
    variance and the discovery layers could not learn anything from it."""
    await generate_cohort(db, n_users=20, days=21, seed=7)
    await db.flush()
    dinner_rows = (
        await db.execute(
            select(MealRecord).where(MealRecord.meal_type == "dinner")
        )
    ).scalars().all()
    hours = {r.created_at.hour for r in dinner_rows}
    # At least 3 distinct dinner hours across the cohort.
    assert len(hours) >= 3, f"dinner hours show no variance: {hours}"
    # All within the published plausible range.
    assert all(17 <= h <= 22 for h in hours), f"dinner hours out of range: {hours}"


# ─────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seeded_runs_are_deterministic(db: AsyncSession) -> None:
    """Same seed, same user_ids emitted. (We can't diff rows across DB
    sessions cleanly because SQLite assigns its own PKs, but user_ids
    are derived from the seed and are a good proxy for the full
    determinism contract.)"""
    manifest_a = await generate_cohort(db, n_users=4, days=7, seed=42)

    # Fresh DB for the second run so PK collisions don't distort the test.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session2:
        manifest_b = await generate_cohort(session2, n_users=4, days=7, seed=42)
    await engine.dispose()

    assert manifest_a.user_ids == manifest_b.user_ids


# ─────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_generator_raises(db: AsyncSession) -> None:
    with pytest.raises(ValueError) as excinfo:
        await generate_cohort(db, n_users=1, days=5, seed=1, generator="nonsense")
    assert "parametric" in str(excinfo.value) or "gan" in str(excinfo.value)


@pytest.mark.asyncio
async def test_zero_days_raises(db: AsyncSession) -> None:
    with pytest.raises(ValueError):
        await generate_cohort(db, n_users=1, days=0, seed=1)


@pytest.mark.asyncio
async def test_zero_users_is_valid_and_writes_nothing(db: AsyncSession) -> None:
    """Empty cohort edge case: manifest is valid and no rows land."""
    manifest = await generate_cohort(db, n_users=0, days=5, seed=1)
    assert manifest.n_users == 0
    assert manifest.user_ids == []

    # Every raw table should be empty.
    for model in (SleepRecord, ActivityRecord, MealRecord, FoodItemRecord, HealthMetricRecord):
        count = (await db.execute(select(func.count()).select_from(model))).scalar_one()
        assert count == 0


@pytest.mark.asyncio
async def test_factory_does_not_commit(db: AsyncSession) -> None:
    """Caller owns the transaction. The factory adds rows to the session
    but must not commit. We verify by rolling back and asserting the
    rows disappear."""
    await generate_cohort(db, n_users=2, days=5, seed=7)
    await db.rollback()

    for model in (SleepRecord, ActivityRecord, MealRecord, FoodItemRecord, HealthMetricRecord):
        count = (await db.execute(select(func.count()).select_from(model))).scalar_one()
        assert count == 0, f"{model.__name__} should be empty after rollback"
