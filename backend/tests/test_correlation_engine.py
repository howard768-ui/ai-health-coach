"""Tests for the cross-domain correlation engine.

Covers the math primitives (pearson, spearman, BH-FDR) plus the
DB-touching `collect_metric_data` and `compute_correlations`. The
audit flagged this service for the "Compute_correlations -> Rank"
3-step flow with zero tests.

Run: cd backend && uv run python -m pytest tests/test_correlation_engine.py -v
"""

from __future__ import annotations

import os
import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from datetime import date, timedelta

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.database import Base
from app.models.health import HealthMetricRecord, SleepRecord
from app.models.meal import MealRecord, FoodItemRecord
from app.services import correlation_engine as ce


USER = "test-user-1"


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


# pearson_correlation ----------------------------------------------------


def test_pearson_returns_zero_for_under_three_samples():
    r, p = ce.pearson_correlation([1.0, 2.0], [3.0, 4.0])
    assert r == 0.0
    assert p == 1.0


def test_pearson_returns_one_for_perfectly_linear():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    r, _ = ce.pearson_correlation(x, y)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_pearson_returns_zero_when_std_is_zero():
    """Constant series can't correlate with anything; protects against div-by-zero."""
    x = [3.0, 3.0, 3.0, 3.0]
    y = [1.0, 2.0, 3.0, 4.0]
    r, p = ce.pearson_correlation(x, y)
    assert r == 0.0
    assert p == 1.0


def test_pearson_negative_for_inverse_relationship():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [5.0, 4.0, 3.0, 2.0, 1.0]
    r, _ = ce.pearson_correlation(x, y)
    assert r == pytest.approx(-1.0, abs=1e-6)


# spearman_correlation ---------------------------------------------------


def test_spearman_handles_monotonic_nonlinear():
    """Spearman ranks; perfectly monotonic non-linear is still rho=1."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 4.0, 9.0, 16.0, 25.0]
    r, _ = ce.spearman_correlation(x, y)
    assert r == pytest.approx(1.0, abs=1e-6)


# benjamini_hochberg -----------------------------------------------------


def test_bh_returns_empty_for_empty_input():
    assert ce.benjamini_hochberg([]) == []


def test_bh_single_value_passes_through():
    out = ce.benjamini_hochberg([0.05])
    assert out == [0.05]


def test_bh_is_monotonic_and_clamps_at_one():
    """Adjusted p-values must be non-decreasing once sorted, and cap at 1.0."""
    p_values = [0.001, 0.01, 0.04, 0.05, 0.5]
    adjusted = ce.benjamini_hochberg(p_values)
    sorted_adj = sorted(adjusted)
    assert sorted_adj == adjusted
    assert all(0.0 <= a <= 1.0 for a in adjusted)


# _describe_effect -------------------------------------------------------


def test_describe_effect_positive_phrasing():
    r = ce.CorrelationResult(
        source_metric="protein_intake",
        target_metric="deep_sleep_seconds",
        lag_days=0,
        pearson_r=0.5,
        spearman_r=0.5,
        p_value=0.01,
        sample_size=20,
        direction="positive",
        strength=0.5,
        methods_agree=True,
    )
    desc = ce._describe_effect(r)
    assert "higher" in desc
    assert "lower" not in desc
    assert "tends to" in desc, "must avoid causation language"


def test_describe_effect_negative_phrasing():
    r = ce.CorrelationResult(
        source_metric="resting_hr",
        target_metric="readiness",
        lag_days=0,
        pearson_r=-0.5,
        spearman_r=-0.5,
        p_value=0.01,
        sample_size=20,
        direction="negative",
        strength=0.5,
        methods_agree=True,
    )
    desc = ce._describe_effect(r)
    assert "lower" in desc
    assert "tends to" in desc


# collect_metric_data ----------------------------------------------------


@pytest.mark.asyncio
async def test_collect_protein_aggregates_food_items_per_day(db):
    today = date.today().isoformat()
    meal = MealRecord(user_id=USER, date=today, meal_type="dinner", source="manual")
    db.add(meal)
    await db.flush()
    db.add(FoodItemRecord(
        meal_id=meal.id, name="chicken", serving_size="6oz",
        calories=200, protein=40, carbs=0, fat=8,
        quality="whole", data_source="usda",
    ))
    db.add(FoodItemRecord(
        meal_id=meal.id, name="rice", serving_size="1cup",
        calories=300, protein=5, carbs=60, fat=2,
        quality="mixed", data_source="usda",
    ))
    await db.commit()

    result = await ce.collect_metric_data(db, USER, "protein_intake", days=30)
    assert result == {today: 45}


@pytest.mark.asyncio
async def test_collect_sleep_efficiency_skips_nulls(db):
    today = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    db.add(SleepRecord(user_id=USER, date=today, efficiency=92))
    db.add(SleepRecord(user_id=USER, date=yesterday, efficiency=None))
    await db.commit()

    result = await ce.collect_metric_data(db, USER, "sleep_efficiency", days=30)
    assert result == {today: 92}


@pytest.mark.asyncio
async def test_collect_steps_filters_canonical(db):
    """Only canonical HealthMetricRecord rows count for steps."""
    today = date.today().isoformat()
    db.add(HealthMetricRecord(
        user_id=USER, date=today, metric_type="steps", value=8000,
        unit="count", source="oura", is_canonical=True,
    ))
    db.add(HealthMetricRecord(
        user_id=USER, date=today, metric_type="steps", value=12000,
        unit="count", source="apple_health", is_canonical=False,
    ))
    await db.commit()

    result = await ce.collect_metric_data(db, USER, "steps", days=30)
    assert result == {today: 8000}, "non-canonical row must be ignored"


@pytest.mark.asyncio
async def test_collect_unknown_metric_returns_empty(db):
    assert await ce.collect_metric_data(db, USER, "made_up_metric", days=30) == {}


# compute_correlations ---------------------------------------------------


@pytest.mark.asyncio
async def test_compute_correlations_returns_empty_with_insufficient_data(db):
    """No data anywhere should produce zero significant results."""
    results = await ce.compute_correlations(db, USER, window_days=30)
    assert results == []


@pytest.mark.asyncio
async def test_compute_correlations_below_min_sample_filters_pair(db):
    """Fewer than MIN_SAMPLE_SIZE paired points means the pair is skipped."""
    today = date.today()
    for offset in range(ce.MIN_SAMPLE_SIZE - 1):
        d = (today - timedelta(days=offset)).isoformat()
        db.add(SleepRecord(user_id=USER, date=d, efficiency=90, resting_hr=55, readiness_score=80))
    await db.commit()

    results = await ce.compute_correlations(db, USER, window_days=30)
    assert results == []


@pytest.mark.asyncio
async def test_compute_correlations_surfaces_a_real_correlation(db):
    """Manufactured strong inverse correlation between resting_hr and
    readiness should produce a significant negative result with method agreement.
    """
    today = date.today()
    for i in range(ce.MIN_SAMPLE_SIZE + 4):
        d = (today - timedelta(days=i)).isoformat()
        rhr = 50 + i  # increasing
        readiness = 95 - i  # decreasing
        db.add(SleepRecord(
            user_id=USER, date=d, efficiency=90,
            resting_hr=rhr, readiness_score=readiness,
        ))
    await db.commit()

    results = await ce.compute_correlations(db, USER, window_days=60)
    rhr_pair = next(
        (r for r in results if r.source_metric == "resting_hr" and r.target_metric == "readiness"),
        None,
    )
    assert rhr_pair is not None, "perfectly inverse pair should clear the significance gate"
    assert rhr_pair.direction == "negative"
    assert rhr_pair.methods_agree is True
    assert rhr_pair.strength > 0.9
    assert rhr_pair.confidence_tier in ("emerging", "developing", "established")


@pytest.mark.asyncio
async def test_compute_correlations_results_sorted_by_strength_desc(db):
    """When multiple pairs clear the gate, strongest correlation comes first."""
    today = date.today()
    # Strong inverse: resting_hr vs readiness
    for i in range(ce.MIN_SAMPLE_SIZE + 4):
        d = (today - timedelta(days=i)).isoformat()
        db.add(SleepRecord(
            user_id=USER, date=d, efficiency=90,
            resting_hr=50 + i, readiness_score=95 - i,
        ))
    await db.commit()

    results = await ce.compute_correlations(db, USER, window_days=60)
    if len(results) < 2:
        pytest.skip("Only one significant pair in this fixture; ordering trivially holds.")
    strengths = [r.strength for r in results]
    assert strengths == sorted(strengths, reverse=True)
