"""Phase 7B: candidates + ranker-metadata endpoint tests.

Tests cover:
1. GET /api/insights/candidates returns today's candidates
2. GET /api/insights/ranker-metadata returns model info or 404

Run: ``cd backend && uv run python -m pytest tests/ml/test_insights_candidates_api.py -v``
"""

from __future__ import annotations

import json
import os

import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models.ml_insights import MLInsightCandidate
from app.models.ml_models import MLModel
# Register ORM models.
from app.models import ml_baselines as _ml_baselines  # noqa: F401
from app.models import ml_features as _ml_features  # noqa: F401
from app.models import ml_synth as _ml_synth  # noqa: F401
from app.models import ml_discovery as _ml_discovery  # noqa: F401


USER = "u-candidates-api"


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
# Candidates response shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidates_query_returns_todays_candidates(db: AsyncSession):
    """Candidates generated today should be returned by the endpoint query."""
    from app.core.time import utcnow_naive
    from datetime import date

    now = utcnow_naive()
    db.add(
        MLInsightCandidate(
            id="cand-001",
            user_id=USER,
            kind="correlation",
            subject_metrics_json=json.dumps(["steps", "sleep_efficiency"]),
            effect_size=0.65,
            confidence=0.80,
            novelty=1.0,
            recency_days=2,
            actionability_score=0.90,
            literature_support=True,
            directional_support=True,
            causal_support=False,
            payload_json=json.dumps({"source_metric": "steps", "target_metric": "sleep_efficiency"}),
            generated_at=now,
        )
    )
    await db.flush()

    # Mirror the route's query EXACTLY. Pre-MELD-BACKEND-G this used
    # `today.isoformat()` (a string), which SQLite happily compared against
    # a DateTime column but Postgres rejected with
    # `operator does not exist: timestamp without time zone >= character varying`.
    # The fix uses a real datetime; this test must keep using the same
    # pattern to catch any future regression.
    from datetime import datetime, time
    from sqlalchemy import select

    today_start = datetime.combine(date.today(), time.min)
    result = await db.execute(
        select(MLInsightCandidate).where(
            MLInsightCandidate.user_id == USER,
            MLInsightCandidate.generated_at >= today_start,
        )
    )
    candidates = result.scalars().all()
    assert len(candidates) == 1
    assert candidates[0].id == "cand-001"
    assert candidates[0].effect_size == 0.65
    assert candidates[0].literature_support is True


@pytest.mark.asyncio
async def test_candidates_route_uses_datetime_not_string_for_generated_at_compare():
    """Regression: MELD-BACKEND-G fired 14 times in production because the
    /api/insights/candidates handler compared a DateTime column against a
    string. Postgres rejects that with UndefinedFunctionError; SQLite was
    lenient.

    This test scans the handler's source for the broken pattern. We strip
    comments before checking so that a comment quoting the old pattern (for
    historical context) doesn't false-positive.
    """
    import inspect
    from app.routers import insights as insights_module

    src = inspect.getsource(insights_module.get_candidates)
    # Drop comments — they may legitimately mention the old pattern.
    no_comments = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert ".isoformat()" not in no_comments, (
        "MELD-BACKEND-G regression: the candidates handler must NOT pass a "
        "string into a `MLInsightCandidate.generated_at` comparison. Use a "
        "real datetime (Postgres rejects timestamp-vs-string compares)."
    )
    # Positive assertion: confirm the new datetime is actually constructed.
    assert "datetime.combine" in no_comments, (
        "Expected `datetime.combine(...)` in the candidates handler. Refactored "
        "to a different datetime construction? Update this assertion."
    )


# ---------------------------------------------------------------------------
# Ranker metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ranker_metadata_returns_none_when_no_model(db: AsyncSession):
    """With no active model, ranker_model_metadata returns None."""
    from ml import api as ml_api

    result = await ml_api.ranker_model_metadata(db)
    assert result is None


@pytest.mark.asyncio
async def test_ranker_metadata_returns_active_model(db: AsyncSession):
    """With an active model, ranker_model_metadata returns its info."""
    from ml import api as ml_api
    from app.core.time import utcnow_naive

    db.add(
        MLModel(
            model_type="ranker",
            model_version="ranker-test-001",
            file_hash="abc123def456",
            file_size_bytes=300000,
            r2_key="coreml/ranker-test-001.mlmodel",
            download_url="https://models.heymeld.com/coreml/ranker-test-001.mlmodel",
            train_samples=500,
            val_ndcg=0.75,
            feature_names_json=json.dumps(["effect_size", "confidence"]),
            is_active=True,
            created_at=utcnow_naive(),
        )
    )
    await db.flush()

    result = await ml_api.ranker_model_metadata(db)
    assert result is not None
    assert result.model_version == "ranker-test-001"
    assert result.file_hash == "abc123def456"
    assert result.file_size_bytes == 300000
    assert result.download_url == "https://models.heymeld.com/coreml/ranker-test-001.mlmodel"
