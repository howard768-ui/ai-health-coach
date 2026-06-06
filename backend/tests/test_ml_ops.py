"""Tests for /ops/ml/* endpoints.

Modeled on ``test_ops.py``. Each endpoint gets:

- A 200-response smoke test against the default app state.
- A schema check that the documented top-level keys are present.
- An empty-DB test against an in-memory SQLite session (all-null
  aggregates are acceptable).
- A seeded-DB test where applicable (ml_rankings feedback, ml_models rows)
  to verify aggregates actually count what they claim to count.

Run: cd backend && uv run python -m pytest tests/test_ml_ops.py -v
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Import every ml_ model side-effectfully so Base.metadata.create_all
# builds the ml_* tables in the test SQLite DB. app/models/__init__.py
# does not import these (they are written to only by backend/ml/ code),
# so the tests have to pull them in explicitly. These go before
# `from app.main import app` so the module imports don't shadow the
# ``app`` FastAPI instance we bind next.
from app.models import correlation  # noqa: F401
from app.models import ml_discovery  # noqa: F401
from app.models import ml_drift  # noqa: F401
from app.models import ml_experiments as _ml_experiments_module  # noqa: F401
from app.models import ml_insights as _ml_insights_module  # noqa: F401
from app.models import ml_models as _ml_models_module  # noqa: F401
from app.models import ml_synth  # noqa: F401
from app.models import ml_training_runs  # noqa: F401

from app.database import Base, get_db
from app.main import app


# -- fixtures --


@pytest_asyncio.fixture
async def empty_db():
    """In-memory SQLite with schema created, no seed data."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def seeded_db():
    """In-memory SQLite with a few rows across ml_rankings + ml_models.

    The seed is kept intentionally small and explicit so the aggregate
    counts in assertions are obvious at a glance.
    """
    from app.models.ml_insights import MLRanking
    from app.models.ml_models import MLModel

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        # Two shown rankings, one with thumbs_up.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        session.add(
            MLRanking(
                user_id="u1",
                surface_date="2026-04-10",
                candidate_id="c1",
                rank=1,
                score=0.9,
                ranker_version="v1",
                was_shown=True,
                shown_at=now,
                feedback="thumbs_up",
                feedback_at=now,
                created_at=now,
            )
        )
        session.add(
            MLRanking(
                user_id="u2",
                surface_date="2026-04-10",
                candidate_id="c2",
                rank=2,
                score=0.5,
                ranker_version="v1",
                was_shown=True,
                shown_at=now,
                created_at=now,
            )
        )
        # One active ranker model.
        session.add(
            MLModel(
                model_type="ranker",
                model_version="ranker-1.0.0",
                file_hash=None,
                file_size_bytes=None,
                r2_key=None,
                download_url=None,
                train_samples=100,
                val_ndcg=0.8,
                feature_names_json="[]",
                hyperparams_json=None,
                is_active=True,
                created_at=now,
            )
        )
        await session.commit()
        yield session
    await engine.dispose()


async def _get(path: str, db_session: AsyncSession | None = None):
    """Call an endpoint, optionally overriding the DB dependency."""
    if db_session is not None:

        async def _override():
            yield db_session

        app.dependency_overrides[get_db] = _override
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(path)
    finally:
        app.dependency_overrides.clear()
    return resp


# -- /ops/ml/signal-quality --


@pytest.mark.asyncio
async def test_signal_quality_200_and_keys():
    resp = await _get("/ops/ml/signal-quality")
    assert resp.status_code == 200
    data = resp.json()
    for k in (
        "timestamp",
        "l2_counts_by_tier",
        "l3_granger_pairs_count",
        "l4_causal_candidates_count",
        "ranker_ndcg_p50_last_7d",
        "insight_ctr_last_7d",
    ):
        assert k in data


@pytest.mark.asyncio
async def test_signal_quality_empty_db(empty_db):
    resp = await _get("/ops/ml/signal-quality", db_session=empty_db)
    assert resp.status_code == 200
    data = resp.json()
    # Every tier is present with a zero count on an empty DB.
    assert data["l2_counts_by_tier"] == {
        "emerging": 0,
        "developing": 0,
        "established": 0,
        "literature_supported": 0,
        "causal_candidate": 0,
    }
    assert data["l3_granger_pairs_count"] == 0
    assert data["l4_causal_candidates_count"] == 0
    assert data["ranker_ndcg_p50_last_7d"] is None
    assert data["insight_ctr_last_7d"] is None


@pytest.mark.asyncio
async def test_signal_quality_ctr_with_seed(seeded_db):
    resp = await _get("/ops/ml/signal-quality", db_session=seeded_db)
    assert resp.status_code == 200
    data = resp.json()
    # Two shown, one thumbs_up -> CTR 0.5
    assert data["insight_ctr_last_7d"] == 0.5
    # One thumbs_up at rank=1 -> median rank 1.0
    assert data["ranker_ndcg_p50_last_7d"] == 1.0


# -- /ops/ml/data-quality --


@pytest.mark.asyncio
async def test_data_quality_200_and_keys():
    resp = await _get("/ops/ml/data-quality")
    assert resp.status_code == 200
    data = resp.json()
    assert "timestamp" in data
    assert "sources" in data
    assert "canonical_freshness_days" in data
    # All four documented sources must be present.
    assert set(data["sources"].keys()) == {
        "oura",
        "garmin",
        "peloton",
        "apple_health",
    }
    for src in data["sources"].values():
        assert set(src.keys()) == {
            "last_ingest",
            "days_stale",
            "row_count_last_30d",
        }


@pytest.mark.asyncio
async def test_data_quality_empty_db_returns_nulls(empty_db):
    resp = await _get("/ops/ml/data-quality", db_session=empty_db)
    assert resp.status_code == 200
    data = resp.json()
    # Fresh schema with no rows: last_ingest None, days_stale None, count 0.
    for src in data["sources"].values():
        assert src["last_ingest"] is None
        assert src["days_stale"] is None
        assert src["row_count_last_30d"] == 0
    assert data["canonical_freshness_days"] is None


# -- /ops/ml/feature-drift --


@pytest.mark.asyncio
async def test_feature_drift_200_and_keys():
    resp = await _get("/ops/ml/feature-drift")
    assert resp.status_code == 200
    data = resp.json()
    for k in (
        "timestamp",
        "last_computed",
        "features_over_threshold",
        "total_features_checked",
        "drifted_count",
    ):
        assert k in data
    assert isinstance(data["features_over_threshold"], list)


@pytest.mark.asyncio
async def test_feature_drift_empty_db_has_empty_arrays(empty_db):
    resp = await _get("/ops/ml/feature-drift", db_session=empty_db)
    assert resp.status_code == 200
    data = resp.json()
    assert data["features_over_threshold"] == []
    assert data["total_features_checked"] == 0
    assert data["drifted_count"] == 0
    # Empty table -> no timestamp to surface.
    assert data["last_computed"] is None


@pytest.mark.asyncio
async def test_feature_drift_seeded_rows_appear(empty_db):
    """Insert ml_drift_results rows and verify the endpoint surfaces them.

    Seeds one batch of four features: two drifted, two not. The endpoint
    must return total_features_checked=4, drifted_count=2, and only the
    drifted entries in features_over_threshold.
    """
    from app.models.ml_drift import MLDriftResult

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    run_id = "run-alpha-0000000000000000000000000000"[:36]
    rows = [
        # (feature_key, ks_stat, threshold, drifted)
        ("hrv", 0.22, 0.15, True),
        ("resting_hr", 0.31, 0.15, True),
        ("sleep_efficiency", 0.08, 0.15, False),
        ("steps", 0.05, 0.15, False),
    ]
    for feature_key, ks_stat, threshold, drifted in rows:
        empty_db.add(
            MLDriftResult(
                synth_run_id=run_id,
                feature_key=feature_key,
                ks_statistic=ks_stat,
                ks_pvalue=0.0,  # arbitrary for this test
                threshold=threshold,
                drifted=drifted,
                sample_size_real=200,
                sample_size_synth=200,
                computed_at=now,
            )
        )
    await empty_db.commit()

    resp = await _get("/ops/ml/feature-drift", db_session=empty_db)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_features_checked"] == 4
    assert data["drifted_count"] == 2
    assert data["last_computed"] is not None
    drifted_features = {f["feature"] for f in data["features_over_threshold"]}
    assert drifted_features == {"hrv", "resting_hr"}
    # Each drifted entry carries ks_stat + threshold.
    for entry in data["features_over_threshold"]:
        assert 0.0 < entry["ks_stat"] <= 1.0
        assert entry["threshold"] == 0.15


@pytest.mark.asyncio
async def test_feature_drift_picks_most_recent_run_id(empty_db):
    """When multiple synth_run_ids exist, the endpoint returns only the
    most recent batch (by MAX(computed_at)).

    Seeds an older run (1 drifted feature) and a newer run (2 drifted +
    1 not drifted). The response must reflect the newer run's three rows
    only.
    """
    from app.models.ml_drift import MLDriftResult

    base = datetime.now(timezone.utc).replace(tzinfo=None)
    older = base - timedelta(days=2)
    newer = base

    empty_db.add(
        MLDriftResult(
            synth_run_id="older-run-aaaaaaaaaaaaaaaaaaaaaaaaaaa"[:36],
            feature_key="hrv",
            ks_statistic=0.9,
            ks_pvalue=0.0,
            threshold=0.15,
            drifted=True,
            sample_size_real=100,
            sample_size_synth=100,
            computed_at=older,
        )
    )
    for feature_key, ks_stat, drifted in [
        ("hrv", 0.2, True),
        ("resting_hr", 0.25, True),
        ("steps", 0.05, False),
    ]:
        empty_db.add(
            MLDriftResult(
                synth_run_id="newer-run-bbbbbbbbbbbbbbbbbbbbbbbbbbb"[:36],
                feature_key=feature_key,
                ks_statistic=ks_stat,
                ks_pvalue=0.0,
                threshold=0.15,
                drifted=drifted,
                sample_size_real=100,
                sample_size_synth=100,
                computed_at=newer,
            )
        )
    await empty_db.commit()

    resp = await _get("/ops/ml/feature-drift", db_session=empty_db)
    data = resp.json()
    # Newer batch has 3 rows; older batch's "hrv" must not be re-counted.
    assert data["total_features_checked"] == 3
    assert data["drifted_count"] == 2
    drifted_features = {f["feature"] for f in data["features_over_threshold"]}
    assert drifted_features == {"hrv", "resting_hr"}


# -- /ops/ml/experiments --


@pytest.mark.asyncio
async def test_experiments_200_and_keys():
    resp = await _get("/ops/ml/experiments")
    assert resp.status_code == 200
    data = resp.json()
    for k in ("timestamp", "active_experiments", "completed_last_30d"):
        assert k in data
    assert isinstance(data["active_experiments"], list)


@pytest.mark.asyncio
async def test_experiments_empty_db(empty_db):
    resp = await _get("/ops/ml/experiments", db_session=empty_db)
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_experiments"] == []
    assert data["completed_last_30d"] == 0


@pytest.mark.asyncio
async def test_experiments_active_row_reported(empty_db):
    """A single gathering-phase experiment appears in active_experiments."""
    from app.models.ml_experiments import MLExperiment

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    empty_db.add(
        MLExperiment(
            user_id="u1",
            experiment_name="Test exp",
            hypothesis=None,
            treatment_metric="dinner_hour",
            outcome_metric="sleep_efficiency",
            design="ab",
            baseline_days=14,
            treatment_days=14,
            washout_days=3,
            min_compliance=10,
            status="baseline",
            started_at=now - timedelta(days=3),
            baseline_end="2026-04-20",
            treatment_start="2026-04-23",
            treatment_end="2026-05-07",
        )
    )
    await empty_db.commit()

    resp = await _get("/ops/ml/experiments", db_session=empty_db)
    data = resp.json()
    assert len(data["active_experiments"]) == 1
    exp = data["active_experiments"][0]
    assert exp["phase"] == "gathering"
    # name carried user-authored free text (PHI); it was removed (audit C1).
    assert "name" not in exp
    # No user_id field leaks out of the endpoint.
    assert "user_id" not in exp


@pytest.mark.asyncio
async def test_experiments_does_not_leak_user_authored_name(empty_db):
    """Regression (audit C1 / MEL-45 class): the unauthenticated
    /ops/ml/experiments endpoint must never echo the user-authored
    experiment_name, which is PHI (users name experiments after conditions,
    meds, and context).
    """
    from app.models.ml_experiments import MLExperiment

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    secret = "cut caffeine for my afib"
    empty_db.add(
        MLExperiment(
            user_id="u1",
            experiment_name=secret,
            hypothesis=None,
            treatment_metric="dinner_hour",
            outcome_metric="sleep_efficiency",
            design="ab",
            baseline_days=14,
            treatment_days=14,
            washout_days=3,
            min_compliance=10,
            status="baseline",
            started_at=now - timedelta(days=3),
            baseline_end="2026-04-20",
            treatment_start="2026-04-23",
            treatment_end="2026-05-07",
        )
    )
    await empty_db.commit()

    resp = await _get("/ops/ml/experiments", db_session=empty_db)
    assert resp.status_code == 200
    # The user-authored free text must not appear anywhere in the response.
    assert secret not in resp.text
    data = resp.json()
    assert len(data["active_experiments"]) == 1
    exp = data["active_experiments"][0]
    assert "name" not in exp
    assert "user_id" not in exp
    assert exp["phase"] == "gathering"


# -- /ops/ml/retrain-readiness --


@pytest.mark.asyncio
async def test_retrain_readiness_200_and_keys():
    resp = await _get("/ops/ml/retrain-readiness")
    assert resp.status_code == 200
    data = resp.json()
    for k in (
        "timestamp",
        "last_training_run",
        "days_since_last_training",
        "labeled_feedback_since_last_training",
        "ndcg_p50_last_30d",
        "recommendation",
    ):
        assert k in data
    assert data["recommendation"] in {"retrain", "wait", "insufficient_data"}


@pytest.mark.asyncio
async def test_retrain_readiness_empty_db(empty_db):
    resp = await _get("/ops/ml/retrain-readiness", db_session=empty_db)
    assert resp.status_code == 200
    data = resp.json()
    # Zero labeled feedback -> insufficient_data.
    assert data["labeled_feedback_since_last_training"] == 0
    assert data["recommendation"] == "insufficient_data"


# -- /ops/ml/model-registry --


@pytest.mark.asyncio
async def test_model_registry_200_and_keys():
    resp = await _get("/ops/ml/model-registry")
    assert resp.status_code == 200
    data = resp.json()
    for k in ("timestamp", "models", "total_count", "latest_ranker_version"):
        assert k in data
    assert isinstance(data["models"], list)


@pytest.mark.asyncio
async def test_model_registry_empty_db(empty_db):
    resp = await _get("/ops/ml/model-registry", db_session=empty_db)
    assert resp.status_code == 200
    data = resp.json()
    assert data["models"] == []
    assert data["total_count"] == 0
    assert data["latest_ranker_version"] is None


@pytest.mark.asyncio
async def test_model_registry_seeded(seeded_db):
    resp = await _get("/ops/ml/model-registry", db_session=seeded_db)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_count"] == 1
    assert data["latest_ranker_version"] == "ranker-1.0.0"
    assert len(data["models"]) == 1
    entry = data["models"][0]
    assert entry["kind"] == "ranker"
    assert entry["active"] is True


# -- PHI hygiene: none of these responses should leak user_id --


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/ops/ml/signal-quality",
        "/ops/ml/data-quality",
        "/ops/ml/feature-drift",
        "/ops/ml/experiments",
        "/ops/ml/retrain-readiness",
        "/ops/ml/model-registry",
    ],
)
async def test_responses_never_include_user_id(path):
    resp = await _get(path)
    assert resp.status_code == 200
    # Serialized JSON body must not contain 'user_id' anywhere.
    assert "user_id" not in resp.text


# -- No auth required (same as /ops/status) --


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/ops/ml/signal-quality",
        "/ops/ml/data-quality",
        "/ops/ml/feature-drift",
        "/ops/ml/experiments",
        "/ops/ml/retrain-readiness",
        "/ops/ml/model-registry",
    ],
)
async def test_no_auth_required(path):
    resp = await _get(path)
    assert resp.status_code == 200
