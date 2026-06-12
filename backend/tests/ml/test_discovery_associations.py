"""Phase 3 L2 associations tests: parity vs legacy, scipy ground-truth,
dynamic pair generation, end-to-end via ``ml.api.run_associations``.

The parity test is the critical gate per the plan: the new scipy-based
engine must match the legacy hand-rolled engine within 2% on shared seed
pairs before we retire the legacy code.

Run: ``cd backend && uv run python -m pytest tests/ml/test_discovery_associations.py -v``
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.models.correlation import UserCorrelation
from app.models.health import HealthMetricRecord, SleepRecord, ActivityRecord
# Register ORM models with Base.metadata before fixture create_all.
from app.models import ml_baselines as _ml_baselines_models  # noqa: F401
from app.models import ml_features as _ml_features_models  # noqa: F401
from app.services import correlation_engine as legacy
from ml import api as ml_api
from ml.discovery import associations
from ml.features.store import materialize_for_user


USER = "u-assoc"


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
# Unit: BH-FDR + scipy ground truth
# ─────────────────────────────────────────────────────────────────────────


def test_bh_fdr_matches_statsmodels_on_known_pvals():
    """Pin the BH-FDR output against statsmodels' direct call on a known vector."""
    from statsmodels.stats.multitest import multipletests

    pvals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    _, expected, _, _ = multipletests(pvals, alpha=0.10, method="fdr_bh")

    # Route through associations._apply_fdr.
    results = [
        associations.AssociationResult(
            source_metric=f"s{i}",
            target_metric="t",
            lag_days=0,
            pearson_r=0.3,
            spearman_r=0.3,
            p_value=p,
            sample_size=30,
            direction="positive",
            strength=0.3,
            methods_agree=True,
        )
        for i, p in enumerate(pvals)
    ]
    associations._apply_fdr(results, alpha=0.10)
    for r, exp_p in zip(results, expected):
        assert abs(r.fdr_adjusted_p - round(float(exp_p), 6)) < 1e-4


def test_correlate_pair_returns_none_below_min_sample():
    """< MIN_SAMPLE_SIZE paired rows -> None."""
    import pandas as pd

    aligned = pd.DataFrame({"x": [1.0] * 10, "y": [2.0] * 10})
    res = associations._correlate_pair(aligned, "x", "y", lag_days=0)
    assert res is None


def test_correlate_pair_returns_none_on_constant_columns():
    """Zero variance -> None (r is NaN)."""
    import pandas as pd

    aligned = pd.DataFrame({"x": [3.0] * 30, "y": list(range(30))})
    res = associations._correlate_pair(aligned, "x", "y", lag_days=0)
    assert res is None


def test_correlate_pair_linear_perfect_positive():
    """Strictly increasing paired series -> Pearson r very close to 1."""
    import pandas as pd

    n = 30
    aligned = pd.DataFrame({"x": list(range(n)), "y": [i * 2 + 5 for i in range(n)]})
    res = associations._correlate_pair(aligned, "x", "y", lag_days=0)
    assert res is not None
    assert res.pearson_r > 0.999
    assert res.spearman_r > 0.999
    assert res.direction == "positive"
    assert res.methods_agree


def test_correlate_pair_inverse_perfect_negative():
    """y = -x -> r near -1."""
    import pandas as pd

    n = 30
    aligned = pd.DataFrame({"x": list(range(n)), "y": [-i for i in range(n)]})
    res = associations._correlate_pair(aligned, "x", "y", lag_days=0)
    assert res is not None
    assert res.pearson_r < -0.999
    assert res.direction == "negative"
    assert res.methods_agree


# ─────────────────────────────────────────────────────────────────────────
# _align_pair
# ─────────────────────────────────────────────────────────────────────────


def test_align_pair_applies_lag_correctly():
    """Lag 1: source[d] paired with target[d+1]."""
    import pandas as pd

    idx = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(5)]
    frame = pd.DataFrame(
        {
            "src": [1.0, 2.0, 3.0, 4.0, 5.0],
            "tgt": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=idx,
    )

    aligned = associations._align_pair(frame, "src", "tgt", lag_days=1)
    # Day 0 source=1, day 1 target=20 -> pair (1, 20)
    # Day 1 source=2, day 2 target=30 -> pair (2, 30)
    # Day 2 -> (3, 40), Day 3 -> (4, 50), Day 4 -> NaN (target[5] missing)
    assert list(aligned["x"]) == [1.0, 2.0, 3.0, 4.0]
    assert list(aligned["y"]) == [20.0, 30.0, 40.0, 50.0]


def test_align_pair_drops_nan_rows():
    import pandas as pd

    idx = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(5)]
    frame = pd.DataFrame(
        {
            "src": [1.0, None, 3.0, 4.0, 5.0],
            "tgt": [10.0, 20.0, 30.0, 40.0, None],
        },
        index=idx,
    )
    aligned = associations._align_pair(frame, "src", "tgt", lag_days=0)
    # day 0 (1,10), day 2 (3,30), day 3 (4,40). Day 1 drops (src NaN), day 4 drops (tgt NaN).
    assert aligned.shape[0] == 3


# ─────────────────────────────────────────────────────────────────────────
# Dynamic pair generator
# ─────────────────────────────────────────────────────────────────────────


def test_generate_dynamic_pairs_respects_cap():
    pairs = associations._generate_dynamic_pairs(exclude_keys=set(), max_pairs=10)
    assert len(pairs) <= 10


def test_generate_dynamic_pairs_excludes_seed_keys():
    """Pairs already in the seed set should not appear in dynamic output."""
    seed_keys = {(src, tgt, lag) for src, tgt, lag, _ in associations.SEED_PAIRS}
    pairs = associations._generate_dynamic_pairs(exclude_keys=seed_keys, max_pairs=500)
    for src, tgt, lag, _ in pairs:
        assert (src, tgt, lag) not in seed_keys


def test_generate_dynamic_pairs_emits_both_lags():
    pairs = associations._generate_dynamic_pairs(exclude_keys=set(), max_pairs=500)
    lags = {lag for _, _, lag, _ in pairs}
    assert lags == {0, 1}


def test_generate_dynamic_pairs_only_uses_base_feature_keys():
    """Derived keys like ``hrv.7d_rolling_mean`` should be excluded from
    dynamic pairs to avoid collinearity with their raw parents."""
    pairs = associations._generate_dynamic_pairs(exclude_keys=set(), max_pairs=500)
    for src, tgt, _, _ in pairs:
        assert "." not in src, f"Dynamic source should be a base feature, got {src}"
        assert "." not in tgt, f"Dynamic target should be a base feature, got {tgt}"


# ─────────────────────────────────────────────────────────────────────────
# End-to-end: compute_associations with seeded data
# ─────────────────────────────────────────────────────────────────────────


async def _seed_correlated_data(db, user_id: str, days: int = 60, noise: float = 1.0):
    """Seed upstream data so the feature store yields correlated series.

    - steps ~ N(8000, 500)
    - sleep_efficiency linearly related to steps: efficiency = 70 + 0.002 * steps + noise
      (strong positive correlation, should surface in SEED_PAIRS)
    """
    import random

    random.seed(17)
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=days - 1 - i)
        steps = 8000 + random.gauss(0, 500)
        efficiency = 70 + 0.002 * steps + random.gauss(0, noise)
        db.add_all([
            HealthMetricRecord(
                user_id=user_id,
                date=d.isoformat(),
                metric_type="steps",
                value=steps,
                source="apple_health",
                is_canonical=True,
            ),
            HealthMetricRecord(
                user_id=user_id,
                date=d.isoformat(),
                metric_type="sleep_efficiency",
                value=efficiency,
                source="oura",
                is_canonical=True,
            ),
            HealthMetricRecord(
                user_id=user_id,
                date=d.isoformat(),
                metric_type="resting_hr",
                value=55 + random.gauss(0, 2),
                source="oura",
                is_canonical=True,
            ),
            HealthMetricRecord(
                user_id=user_id,
                date=d.isoformat(),
                metric_type="readiness_score",
                value=78 + random.gauss(0, 5),
                source="oura",
                is_canonical=True,
            ),
        ])
        db.add(
            ActivityRecord(
                user_id=user_id,
                date=d.isoformat(),
                steps=int(steps),
                active_calories=250,
                source="apple_health",
            )
        )
    await db.flush()


@pytest.mark.asyncio
async def test_compute_associations_surfaces_planted_steps_sleep_correlation(db):
    """Plant a strong steps -> sleep_efficiency relationship. The new engine
    should find it with high strength and dual-method agreement."""
    await _seed_correlated_data(db, USER, days=60, noise=1.0)
    today = date.today()
    await materialize_for_user(db, USER, today - timedelta(days=59), today)

    results, report = await associations.compute_associations(
        db, USER, window_days=60, include_dynamic_pairs=False
    )
    # At least one significant result about sleep_efficiency + steps.
    steps_pair = [
        r
        for r in results
        if (r.source_metric, r.target_metric, r.lag_days) == ("steps", "sleep_efficiency", 0)
    ]
    assert steps_pair, f"Planted correlation not surfaced; got {[r.source_metric for r in results]}"
    assert steps_pair[0].strength > 0.5
    assert steps_pair[0].methods_agree
    assert steps_pair[0].direction == "positive"


@pytest.mark.asyncio
async def test_compute_associations_dynamic_pairs_expands_beyond_seeds(db):
    """Dynamic pair generation should produce > 20 pairs beyond the seed set."""
    await _seed_correlated_data(db, USER, days=30)
    today = date.today()
    await materialize_for_user(db, USER, today - timedelta(days=29), today)

    _, report = await associations.compute_associations(
        db, USER, window_days=30, include_dynamic_pairs=True, max_pairs=200
    )
    assert report.dynamic_pairs_generated >= 20
    assert report.pairs_tested >= len(associations.SEED_PAIRS) + 20


@pytest.mark.asyncio
async def test_compute_associations_empty_when_no_data(db):
    """No data -> zero results, no crash."""
    date.today()
    results, report = await associations.compute_associations(db, USER, window_days=30)
    assert results == []
    assert report.pairs_with_enough_data == 0


# ─────────────────────────────────────────────────────────────────────────
# Persistence via UserCorrelation + literature validation
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_associations_upserts_with_legacy_names(db):
    """Stored rows should use legacy names (``steps``, ``sleep_efficiency``)
    not feature-store keys (same in this case, but the protein example
    differs)."""
    await _seed_correlated_data(db, USER, days=60)
    today = date.today()
    await materialize_for_user(db, USER, today - timedelta(days=59), today)

    results, _ = await associations.compute_associations(
        db, USER, window_days=60, include_dynamic_pairs=False
    )
    await associations.persist_associations(db, USER, results)
    await db.commit()

    rows = (await db.execute(select(UserCorrelation).where(UserCorrelation.user_id == USER))).scalars().all()
    assert rows
    names = {(r.source_metric, r.target_metric, r.lag_days) for r in rows}
    # steps -> sleep_efficiency preserved with legacy names (same in this case).
    assert ("steps", "sleep_efficiency", 0) in names


@pytest.mark.asyncio
async def test_persist_associations_is_idempotent(db):
    """Rerunning persist should update in place, not create duplicates."""
    await _seed_correlated_data(db, USER, days=60)
    today = date.today()
    await materialize_for_user(db, USER, today - timedelta(days=59), today)
    results, _ = await associations.compute_associations(
        db, USER, window_days=60, include_dynamic_pairs=False
    )

    await associations.persist_associations(db, USER, results)
    first = (await db.execute(select(UserCorrelation).where(UserCorrelation.user_id == USER))).scalars().all()
    await associations.persist_associations(db, USER, results)
    second = (await db.execute(select(UserCorrelation).where(UserCorrelation.user_id == USER))).scalars().all()
    assert len(first) == len(second)


# ─────────────────────────────────────────────────────────────────────────
# ml.api.run_associations (public boundary)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_run_associations_returns_report(db):
    """Public boundary smoke test."""
    await _seed_correlated_data(db, USER, days=60)
    today = date.today()
    await materialize_for_user(db, USER, today - timedelta(days=59), today)

    report = await ml_api.run_associations(db, USER, window_days=60)
    assert report.user_id == USER
    assert report.window_days == 60
    assert report.pairs_tested >= 1
    assert report.rows_written >= 1


# ─────────────────────────────────────────────────────────────────────────
# Parity: legacy vs new on the same seed data
# ─────────────────────────────────────────────────────────────────────────


async def _seed_legacy_parity_data(db, user_id: str, days: int = 60):
    """Seed upstream rows in both the legacy format (SleepRecord +
    HealthMetricRecord + ActivityRecord) and the shape the new engine reads
    (feature store materializes from the same).

    Designed to exercise the overlapping seed pair
    ``steps -> sleep_efficiency``, lag 0. The correlation is seeded strong
    (noise 0.3 on a ~1.2 signal) so that **both** engines agree it's
    significant — legacy's hand-rolled p-value is a loose t-distribution
    approximation (p ≈ 0.07 on what scipy rates p << 0.001 for r ≈ 0.62),
    and the parity test would drop the pair on the legacy side if we chose
    a weaker correlation. Tighter noise gives r ≈ 0.97 where both engines
    clear the p < 0.05 gate.
    """
    import random

    random.seed(88)
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=days - 1 - i)
        steps = 8000 + random.gauss(0, 800)
        eff = 70 + 0.0015 * steps + random.gauss(0, 0.3)
        # HealthMetricRecord (read by new engine via feature store).
        db.add(
            HealthMetricRecord(
                user_id=user_id,
                date=d.isoformat(),
                metric_type="steps",
                value=steps,
                source="apple_health",
                is_canonical=True,
            )
        )
        db.add(
            HealthMetricRecord(
                user_id=user_id,
                date=d.isoformat(),
                metric_type="sleep_efficiency",
                value=eff,
                source="oura",
                is_canonical=True,
            )
        )
        # SleepRecord (read by legacy engine for sleep_efficiency).
        db.add(
            SleepRecord(
                user_id=user_id,
                date=d.isoformat(),
                efficiency=eff,
            )
        )
        # ActivityRecord (read by feature store for steps — in case reconciliation matters).
        db.add(
            ActivityRecord(
                user_id=user_id,
                date=d.isoformat(),
                steps=int(steps),
                active_calories=250,
                source="apple_health",
            )
        )
    await db.flush()


@pytest.mark.asyncio
async def test_parity_new_vs_legacy_on_steps_sleep_efficiency(db):
    """New engine's Pearson / Spearman / sample_size must match legacy's
    within 2% on the shared ``steps -> sleep_efficiency`` seed pair."""
    await _seed_legacy_parity_data(db, USER, days=60)
    today = date.today()
    await materialize_for_user(db, USER, today - timedelta(days=59), today)

    legacy_results = await legacy.compute_correlations(db, USER, window_days=60)
    legacy_pair = next(
        (r for r in legacy_results if (r.source_metric, r.target_metric, r.lag_days) == ("steps", "sleep_efficiency", 0)),
        None,
    )
    assert legacy_pair is not None, "Legacy engine did not surface the steps -> sleep_efficiency pair"

    new_results, _ = await associations.compute_associations(
        db, USER, window_days=60, include_dynamic_pairs=False
    )
    new_pair = next(
        (r for r in new_results if (r.source_metric, r.target_metric, r.lag_days) == ("steps", "sleep_efficiency", 0)),
        None,
    )
    assert new_pair is not None, "New engine did not surface the steps -> sleep_efficiency pair"

    # Pearson / Spearman must agree within 2% (correlations are scale-invariant,
    # so this should be tight even if units differ between the two reads).
    # Note: we do NOT assert parity on p_value. Legacy uses a rough t-distribution
    # approximation (p ≈ 0.07 on what scipy rates p << 0.001); one of the
    # primary wins of the migration is replacing that hand-rolled estimate
    # with scipy.stats.pearsonr's exact two-tailed p.
    assert abs(new_pair.pearson_r - legacy_pair.pearson_r) / max(abs(legacy_pair.pearson_r), 0.01) < 0.02
    assert abs(new_pair.spearman_r - legacy_pair.spearman_r) / max(abs(legacy_pair.spearman_r), 0.01) < 0.02
    # Sample sizes must match exactly — both should align all 60 days.
    assert new_pair.sample_size == legacy_pair.sample_size
    assert new_pair.direction == legacy_pair.direction
    assert new_pair.methods_agree == legacy_pair.methods_agree
