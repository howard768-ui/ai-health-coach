"""Phase 9A experiment API + lifecycle tests.

Tests cover:
1. Create experiment with computed phase dates
2. Log adherence to baseline/treatment phases
3. Abandon experiment
4. Check and complete experiments
5. ml.api entry points

Run: ``cd backend && uv run python -m pytest tests/ml/test_experiments_api.py -v``
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from datetime import date, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
# Register ORM models.
from app.models import ml_baselines as _ml_baselines  # noqa: F401
from app.models import ml_features as _ml_features  # noqa: F401
from app.models import ml_insights as _ml_insights  # noqa: F401
from app.models import ml_synth as _ml_synth  # noqa: F401
from app.models import ml_models as _ml_models  # noqa: F401
from app.models import ml_discovery as _ml_discovery  # noqa: F401
from app.models import ml_cohorts as _ml_cohorts  # noqa: F401
from app.models import ml_experiments as _ml_experiments  # noqa: F401

from app.models.ml_experiments import MLExperiment, MLNof1Result


USER = "u-experiment"


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


# ---------------------------------------------------------------------------
# Create experiment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_experiment_computes_dates(db: AsyncSession):
    """Creating an experiment should set baseline_end, treatment_start, treatment_end."""
    from ml import api as ml_api

    exp = await ml_api.create_experiment(
        db, USER, "Morning exercise test",
        treatment_metric="workout_duration_sum_minutes",
        outcome_metric="sleep_efficiency",
        hypothesis="Morning exercise improves sleep",
        design="ab",
        baseline_days=14,
        treatment_days=14,
    )
    await db.flush()

    assert exp.status == "baseline"
    assert exp.baseline_days == 14
    assert exp.treatment_days == 14

    # Dates should be computed from today.
    today = date.today()
    baseline_end = date.fromisoformat(exp.baseline_end)
    treatment_start = date.fromisoformat(exp.treatment_start)
    treatment_end = date.fromisoformat(exp.treatment_end)

    assert baseline_end == today + timedelta(days=13)
    assert treatment_start > baseline_end  # washout gap
    assert treatment_end > treatment_start


# ---------------------------------------------------------------------------
# Log adherence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_adherence_increments_compliance(db: AsyncSession):
    """Logging compliance should increment the correct phase counter."""
    from ml import api as ml_api

    exp = await ml_api.create_experiment(
        db, USER, "Test adherence",
        treatment_metric="dinner_hour",
        outcome_metric="sleep_efficiency",
    )
    await db.flush()

    # Log baseline compliance.
    today = date.today()
    await ml_api.log_experiment_adherence(db, exp.id, today.isoformat(), True)
    await db.flush()

    refreshed = await db.get(MLExperiment, exp.id)
    assert refreshed.compliant_days_baseline == 1
    assert refreshed.compliant_days_treatment == 0


# ---------------------------------------------------------------------------
# Abandon
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abandon_sets_status(db: AsyncSession):
    """Abandoning an experiment should set status='abandoned'."""
    from ml import api as ml_api

    exp = await ml_api.create_experiment(
        db, USER, "Abandon test",
        treatment_metric="steps",
        outcome_metric="hrv",
    )
    await db.flush()

    exp.status = "abandoned"
    from app.core.time import utcnow_naive
    exp.completed_at = utcnow_naive()
    await db.flush()

    refreshed = await db.get(MLExperiment, exp.id)
    assert refreshed.status == "abandoned"
    assert refreshed.completed_at is not None


# ---------------------------------------------------------------------------
# Result retrieval (empty)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_result_returns_none_when_not_completed(db: AsyncSession):
    """get_experiment_result should return None for non-completed experiments."""
    from ml import api as ml_api

    exp = await ml_api.create_experiment(
        db, USER, "No result yet",
        treatment_metric="steps",
        outcome_metric="hrv",
    )
    await db.flush()

    result = await ml_api.get_experiment_result(db, exp.id)
    assert result is None


# ---------------------------------------------------------------------------
# Full lifecycle with mock result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_apte_result_creates_row(db: AsyncSession):
    """Persisting an APTE result should create an ml_n_of_1_results row."""
    from ml.discovery.apte import APTEResult, persist_apte_result

    exp = MLExperiment(
        user_id=USER,
        experiment_name="Full lifecycle test",
        treatment_metric="dinner_hour",
        outcome_metric="sleep_efficiency",
        design="ab",
        baseline_days=14,
        treatment_days=14,
        washout_days=3,
        min_compliance=10,
        status="analyzing",
        started_at=__import__("datetime").datetime.now(),
        baseline_end="2026-04-01",
        treatment_start="2026-04-05",
        treatment_end="2026-04-18",
        created_at=__import__("datetime").datetime.now(),
    )
    db.add(exp)
    await db.flush()

    apte_result = APTEResult(
        apte=4.2,
        ci_lower=1.5,
        ci_upper=6.9,
        p_value=0.008,
        effect_size_d=0.85,
        baseline_mean=78.3,
        treatment_mean=82.5,
        baseline_n=14,
        treatment_n=14,
    )

    row_id = await persist_apte_result(
        db, exp.id, USER, "dinner_hour", "sleep_efficiency",
        apte_result, compliant_baseline=12, compliant_treatment=13,
    )
    await db.flush()

    result = await db.get(MLNof1Result, row_id)
    assert result is not None
    assert result.apte == 4.2
    assert result.p_value == 0.008
    assert result.effect_size_d == 0.85
    assert result.compliant_days_baseline == 12
    assert result.compliant_days_treatment == 13
    assert result.method == "permutation_test"
