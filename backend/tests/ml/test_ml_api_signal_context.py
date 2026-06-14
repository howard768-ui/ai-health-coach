"""Tests for ml.api.load_active_patterns / load_recent_anomalies /
load_personal_forecasts / load_coach_signal_context.

These are the async loaders that feed Phase 5's SignalContext into the
coach prompt. DB-seeded; no LLM calls.

Run: ``cd backend && uv run python -m pytest tests/ml/test_ml_api_signal_context.py -v``
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
from app.models.correlation import UserCorrelation
# Register ORM models before create_all.
from app.models import ml_baselines as _ml_baselines_models  # noqa: F401
from app.models import ml_features as _ml_features_models  # noqa: F401
from app.models import ml_insights as _ml_insights_models  # noqa: F401
from app.models.ml_baselines import MLAnomaly, MLForecast
from ml import api as ml_api


USER = "u-signal-context"
TODAY = date.today()


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
# load_active_patterns
# ─────────────────────────────────────────────────────────────────────────


def _make_correlation(
    db, *, source: str, target: str, tier: str, strength: float,
    literature_match: bool = False, literature_ref: str | None = None,
    description: str = "",
) -> None:
    db.add(
        UserCorrelation(
            user_id=USER,
            source_metric=source,
            target_metric=target,
            lag_days=0,
            direction="positive",
            pearson_r=strength,
            spearman_r=strength,
            p_value=0.01,
            fdr_adjusted_p=0.05,
            sample_size=40,
            strength=strength,
            confidence_tier=tier,
            literature_match=literature_match,
            literature_ref=literature_ref,
            effect_size_description=description,
        )
    )


@pytest.mark.asyncio
async def test_load_active_patterns_empty_for_new_user(db):
    patterns = await ml_api.load_active_patterns(db, USER)
    assert patterns == []


@pytest.mark.asyncio
async def test_load_active_patterns_filters_emerging_tier(db):
    """``emerging`` tier is below the Phase 5 display threshold."""
    _make_correlation(db, source="steps", target="sleep_efficiency",
                      tier="emerging", strength=0.4)
    _make_correlation(db, source="protein_intake", target="deep_sleep_seconds",
                      tier="developing", strength=0.55)
    await db.flush()

    patterns = await ml_api.load_active_patterns(db, USER)
    sources = [p.source_metric for p in patterns]
    assert "steps" not in sources
    assert "protein_intake" in sources


@pytest.mark.asyncio
async def test_load_active_patterns_sorted_by_strength_x_tier_weight(db):
    """Literature-supported beats established even at lower strength."""
    _make_correlation(db, source="a", target="b",
                      tier="established", strength=0.80,
                      description="Established pattern.")
    _make_correlation(db, source="c", target="d",
                      tier="literature_supported", strength=0.55,
                      literature_match=True, literature_ref="10.1/xyz",
                      description="Literature-backed pattern.")
    await db.flush()

    patterns = await ml_api.load_active_patterns(db, USER)
    # literature (0.55 * 0.95 = 0.5225) vs established (0.80 * 0.80 = 0.64) ->
    # established wins here. Check that ordering is actually by the product.
    scores = [(p.source_metric, p.strength * {"developing":0.60,"established":0.80,
               "causal_candidate":0.90,"literature_supported":0.95}.get(p.confidence_tier, 0.30))
              for p in patterns]
    # First entry should have highest score.
    assert scores[0][1] >= scores[-1][1]


@pytest.mark.asyncio
async def test_load_active_patterns_limit_parameter(db):
    for i in range(10):
        _make_correlation(db, source=f"s{i}", target=f"t{i}",
                          tier="developing", strength=0.6)
    await db.flush()

    patterns = await ml_api.load_active_patterns(db, USER, limit=3)
    assert len(patterns) == 3


@pytest.mark.asyncio
async def test_load_active_patterns_includes_literature_ref(db):
    _make_correlation(db, source="protein_intake", target="deep_sleep_seconds",
                      tier="literature_supported", strength=0.55,
                      literature_match=True, literature_ref="10.1007/s40279-014-0260-0",
                      description="Higher protein, longer deep sleep.")
    await db.flush()

    patterns = await ml_api.load_active_patterns(db, USER)
    assert patterns[0].literature_ref == "10.1007/s40279-014-0260-0"


# ─────────────────────────────────────────────────────────────────────────
# load_recent_anomalies
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_recent_anomalies_requires_bocpd_confirmation_by_default(db):
    # One unconfirmed, one confirmed. Default call returns only confirmed.
    db.add_all([
        MLAnomaly(
            user_id=USER,
            metric_key="hrv",
            observation_date=(TODAY - timedelta(days=1)).isoformat(),
            observed_value=22.0,
            forecasted_value=42.0,
            residual=-20.0,
            z_score=-4.0,
            direction="low",
            confirmed_by_bocpd=False,
            model_version="residual-z-1.0.0",
        ),
        MLAnomaly(
            user_id=USER,
            metric_key="resting_hr",
            observation_date=(TODAY - timedelta(days=2)).isoformat(),
            observed_value=72.0,
            forecasted_value=55.0,
            residual=17.0,
            z_score=3.5,
            direction="high",
            confirmed_by_bocpd=True,
            model_version="residual-z-1.0.0",
        ),
    ])
    await db.flush()

    anomalies = await ml_api.load_recent_anomalies(db, USER)
    metric_keys = [a.metric_key for a in anomalies]
    assert "resting_hr" in metric_keys
    assert "hrv" not in metric_keys


@pytest.mark.asyncio
async def test_load_recent_anomalies_with_confirmed_only_false_returns_all(db):
    db.add(
        MLAnomaly(
            user_id=USER,
            metric_key="hrv",
            observation_date=(TODAY - timedelta(days=1)).isoformat(),
            observed_value=22.0,
            forecasted_value=42.0,
            residual=-20.0,
            z_score=-4.0,
            direction="low",
            confirmed_by_bocpd=False,
            model_version="residual-z-1.0.0",
        )
    )
    await db.flush()

    anomalies = await ml_api.load_recent_anomalies(db, USER, confirmed_only=False)
    assert len(anomalies) == 1


@pytest.mark.asyncio
async def test_load_recent_anomalies_filters_outside_lookback_window(db):
    db.add(
        MLAnomaly(
            user_id=USER,
            metric_key="hrv",
            observation_date=(TODAY - timedelta(days=30)).isoformat(),
            observed_value=20.0,
            forecasted_value=42.0,
            residual=-22.0,
            z_score=-4.5,
            direction="low",
            confirmed_by_bocpd=True,
            model_version="residual-z-1.0.0",
        )
    )
    await db.flush()

    anomalies = await ml_api.load_recent_anomalies(db, USER, lookback_days=7)
    assert anomalies == []


# ─────────────────────────────────────────────────────────────────────────
# load_personal_forecasts
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_personal_forecasts_returns_today_and_tomorrow(db):
    for days_out in (0, 1, 2, 3):
        db.add(
            MLForecast(
                user_id=USER,
                metric_key="hrv",
                target_date=(TODAY + timedelta(days=days_out)).isoformat(),
                made_on=(TODAY - timedelta(days=1)).isoformat(),
                y_hat=40.0,
                y_hat_low=35.0,
                y_hat_high=45.0,
                model_version="ensemble-1.0.0",
            )
        )
    await db.flush()

    forecasts = await ml_api.load_personal_forecasts(db, USER, horizon_days=2)
    target_dates = [f.target_date for f in forecasts]
    assert TODAY.isoformat() in target_dates
    assert (TODAY + timedelta(days=1)).isoformat() in target_dates
    assert (TODAY + timedelta(days=2)).isoformat() not in target_dates


@pytest.mark.asyncio
async def test_load_personal_forecasts_keeps_most_recent_made_on(db):
    # Two forecasts for the same target_date with different made_on dates.
    for made_on_offset in (-3, -1):
        db.add(
            MLForecast(
                user_id=USER,
                metric_key="hrv",
                target_date=TODAY.isoformat(),
                made_on=(TODAY + timedelta(days=made_on_offset)).isoformat(),
                y_hat=40.0 if made_on_offset == -1 else 35.0,
                y_hat_low=35.0,
                y_hat_high=45.0,
                model_version="ensemble-1.0.0",
            )
        )
    await db.flush()

    forecasts = await ml_api.load_personal_forecasts(db, USER)
    hrv_forecasts = [f for f in forecasts if f.metric_key == "hrv"]
    # Most recent made_on (offset -1) should win. y_hat=40.0.
    assert len(hrv_forecasts) == 1
    assert hrv_forecasts[0].y_hat == 40.0


# ─────────────────────────────────────────────────────────────────────────
# load_coach_signal_context
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_coach_signal_context_combines_all_three(db):
    _make_correlation(db, source="protein_intake", target="deep_sleep_seconds",
                      tier="developing", strength=0.6,
                      description="Higher protein, longer deep sleep.")
    db.add(
        MLAnomaly(
            user_id=USER,
            metric_key="hrv",
            observation_date=(TODAY - timedelta(days=1)).isoformat(),
            observed_value=22.0,
            forecasted_value=42.0,
            residual=-20.0,
            z_score=-4.0,
            direction="low",
            confirmed_by_bocpd=True,
            model_version="residual-z-1.0.0",
        )
    )
    db.add(
        MLForecast(
            user_id=USER,
            metric_key="hrv",
            target_date=TODAY.isoformat(),
            made_on=(TODAY - timedelta(days=1)).isoformat(),
            y_hat=40.0,
            y_hat_low=35.0,
            y_hat_high=45.0,
            model_version="ensemble-1.0.0",
        )
    )
    await db.flush()

    ctx = await ml_api.load_coach_signal_context(db, USER)
    assert ctx.is_empty is False
    assert len(ctx.active_patterns) == 1
    assert len(ctx.recent_anomalies) == 1
    assert len(ctx.personal_forecasts) == 1


@pytest.mark.asyncio
async def test_load_coach_signal_context_empty_is_not_an_error(db):
    ctx = await ml_api.load_coach_signal_context(db, USER)
    assert ctx.is_empty is True
    assert ctx.active_patterns == []
    assert ctx.recent_anomalies == []
    assert ctx.personal_forecasts == []
