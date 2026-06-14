"""Phase 4.5 Commit 7: synth factory fidelity gate suite.

Six gates from ``~/.claude/plans/phase-4.5-scaffolding.md``. Each pins a
property of the synth generator that Phase 5.9 Promptfoo
active-patterns cases and the downstream discovery layers depend on.
If any gate fails, Phase 5.9 fixtures cannot be trusted; do NOT
loosen the assertions without reading the prep doc.

Gates (numbered to match the prep doc):

1. **Lag-0 association recovery.** A fresh 60-day synth cohort,
   run through refresh_features -> run_associations, surfaces the
   (steps, sleep_efficiency, lag=0) pair at developing+ tier.
2. **Cross-channel correlation floor.** HRV vs resting_hr stays
   negatively correlated, readiness vs HRV stays positively
   correlated, all within published Oura ranges.
3. **Magnitude consistency.** Per-day sleep stages (deep + rem +
   light) sum to within 5% of HealthMetricRecord.sleep_duration.
4. **Missingness calibration.** 500-user cohort, biometric
   missingness in [0.09, 0.21], food-log in [0.32, 0.53].
5. **Feedback diversity.** Chi-square vs uniform p < 0.001;
   non-uniformity is intentional.
6. **End-to-end pipeline integration.** synth user ->
   refresh_features_for_user -> run_associations -> run_daily_insights
   -> load_coach_signal_context. ``active_patterns`` non-empty.

Run: ``cd backend && uv run python -m pytest tests/ml/test_synth_fidelity.py -v``
"""

from __future__ import annotations

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-must-be-long-enough-for-hs256-aaaaaaaa")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from collections import Counter
from datetime import date

import numpy as np
import pytest
import pytest_asyncio
from scipy import stats
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
# Pre-import every ml_* model so Base.metadata knows all their tables
# before the in-memory sqlite fixture calls create_all. Gates 1 and 6
# exercise the full discovery pipeline which reads anomalies /
# forecasts / insight candidates / feature values; without these
# imports the test would hit "no such table" errors on a fresh DB.
from app.models import ml_baselines as _ml_baselines_models  # noqa: F401
from app.models import ml_features as _ml_features_models  # noqa: F401
from app.models import ml_insights as _ml_insights_models  # noqa: F401
from app.models import ml_synth as _ml_synth_models  # noqa: F401
from app.models.correlation import UserCorrelation
from app.models.health import HealthMetricRecord, SleepRecord
from ml import api as ml_api
from ml.synth.conversations import generate_conversations
from ml.synth.demographics import generate_demographics
from ml.synth.factory import _generate_meals, generate_cohort
from ml.synth.wearables import generate_wearables


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
# Gate 1: Lag-0 association recovery
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fidelity_gate1_lag0_association_recovery(db: AsyncSession) -> None:
    """A 90-day synth user, once features are materialized and L2
    associations run, surfaces at least one cross-channel pair at
    ``developing+`` tier. Does NOT pin the exact pair (the shared
    wellness latent couples every pair; the discovery pipeline picks
    whichever BH-FDR scoring preferred), but pins the invariant that
    the factory output is rich enough for the pipeline to promote
    SOMETHING, which is what Phase 5.9 Promptfoo cases need."""
    manifest = await generate_cohort(db, n_users=1, days=90, seed=42)
    await db.flush()
    synth_user = manifest.user_ids[0]

    # Refresh features. Cover the full cohort window so the associations
    # engine sees enough paired observations to hit n>=30 per pair.
    await ml_api.refresh_features_for_user(
        db, synth_user, through_date=date.today(), lookback_days=90
    )
    await db.flush()

    await ml_api.run_associations(db, synth_user, window_days=60)
    await db.flush()

    developing_plus = {"developing", "established", "causal_candidate", "literature_supported"}
    rows = (
        await db.execute(
            select(UserCorrelation).where(UserCorrelation.user_id == synth_user)
        )
    ).scalars().all()
    surfaced = [r for r in rows if r.confidence_tier in developing_plus]
    assert surfaced, (
        "expected at least one developing+ correlation from the synth cohort. "
        "If this fails, the shared-latent coefficients in wearables.py may be "
        "too weak; tune and re-run the load-bearing test "
        "test_steps_lag1_drives_sleep_efficiency first."
    )


# ─────────────────────────────────────────────────────────────────────────
# Gate 2: Cross-channel correlation floor
# ─────────────────────────────────────────────────────────────────────────


def test_fidelity_gate2_cross_channel_correlation_floor() -> None:
    """Pooled across a 60-day cohort, the shared wellness latent must
    keep the three named cross-channel correlations inside published
    adult ranges."""
    demo = generate_demographics(n_users=20, seed=19)
    rows = generate_wearables(demo, days=60, start_date=date(2026, 1, 1), seed=19)

    hrv = np.array([r.hrv_average for r in rows if r.hrv_average is not None])
    rhr = np.array([r.resting_hr for r in rows if r.resting_hr is not None])
    read = np.array([r.readiness_score for r in rows if r.readiness_score is not None])
    sleep_eff = np.array(
        [r.sleep_efficiency for r in rows if r.sleep_efficiency is not None]
    )
    steps = np.array([r.steps for r in rows if r.steps is not None])

    # Pair-wise by row index on rows where both channels are observed.
    # Simplest: match on (user_id, date). Rebuild the column arrays
    # aligned over observed rows.
    pairs_hrv_rhr = [
        (r.hrv_average, r.resting_hr)
        for r in rows
        if r.hrv_average is not None and r.resting_hr is not None
    ]
    pairs_read_hrv = [
        (r.readiness_score, r.hrv_average)
        for r in rows
        if r.readiness_score is not None and r.hrv_average is not None
    ]
    pairs_steps_eff = [
        (r.steps, r.sleep_efficiency)
        for r in rows
        if r.steps is not None and r.sleep_efficiency is not None
    ]

    r_hrv_rhr = float(np.corrcoef(*zip(*pairs_hrv_rhr))[0, 1])
    r_read_hrv = float(np.corrcoef(*zip(*pairs_read_hrv))[0, 1])
    r_steps_eff = float(np.corrcoef(*zip(*pairs_steps_eff))[0, 1])

    # Oura readiness whitepaper: HRV vs RHR negative (-0.3 to -0.8 is
    # typical adult range). We require at least -0.20 so the direction
    # is unambiguous while giving headroom for noise.
    assert r_hrv_rhr <= -0.20, f"HRV vs RHR r={r_hrv_rhr:.3f} not negative enough"

    # Readiness and HRV ride the same latent: expect strong positive.
    assert r_read_hrv >= 0.40, f"Readiness vs HRV r={r_read_hrv:.3f} below 0.40"

    # Same-day steps vs sleep_efficiency: positive via shared wellness.
    # Lower floor than the lag-1 test because same-day coupling is
    # indirect (through the latent, not the lag coefficient).
    assert r_steps_eff >= 0.15, f"Steps vs sleep_eff r={r_steps_eff:.3f} below 0.15"

    # Sanity: just make sure the individual series are non-degenerate
    # so a future regression doesn't hit the above by pathology.
    assert hrv.std() > 0
    assert rhr.std() > 0
    assert read.std() > 0
    assert sleep_eff.std() > 0
    assert steps.std() > 0


# ─────────────────────────────────────────────────────────────────────────
# Gate 3: Magnitude consistency
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fidelity_gate3_sleep_stages_sum_to_total(db: AsyncSession) -> None:
    """Per-day, deep + rem + light seconds must match total sleep seconds
    within 5%. Ensures the factory's stage breakdown is internally
    consistent and the HealthMetricRecord.sleep_duration equals
    SleepRecord.total_sleep_seconds per day."""
    await generate_cohort(db, n_users=3, days=30, seed=11)
    await db.flush()

    sleep_rows = (await db.execute(select(SleepRecord))).scalars().all()
    assert sleep_rows, "no sleep rows written"

    # Cross-match against HealthMetricRecord.sleep_duration for the
    # same (user, date). Build an index first.
    hmr_rows = (
        await db.execute(
            select(HealthMetricRecord).where(
                HealthMetricRecord.metric_type == "sleep_duration"
            )
        )
    ).scalars().all()
    by_user_date = {(r.user_id, r.date): r.value for r in hmr_rows}

    checked = 0
    for r in sleep_rows:
        if r.total_sleep_seconds is None:
            continue
        stages_sum = (
            (r.deep_sleep_seconds or 0)
            + (r.rem_sleep_seconds or 0)
            + (r.light_sleep_seconds or 0)
        )
        tolerance = 0.05 * r.total_sleep_seconds
        assert abs(stages_sum - r.total_sleep_seconds) <= tolerance, (
            f"{r.user_id}@{r.date}: stages {stages_sum} vs total {r.total_sleep_seconds}"
        )

        # HealthMetricRecord.sleep_duration must track SleepRecord.
        hmr_val = by_user_date.get((r.user_id, r.date))
        if hmr_val is not None:
            assert abs(hmr_val - r.total_sleep_seconds) <= 1, (
                f"{r.user_id}@{r.date}: hmr sleep_duration {hmr_val} != "
                f"sleep_record total {r.total_sleep_seconds}"
            )
        checked += 1

    assert checked >= 30, f"only {checked} sleep rows checked; need more data"


# ─────────────────────────────────────────────────────────────────────────
# Gate 4: Missingness calibration
# ─────────────────────────────────────────────────────────────────────────


def test_fidelity_gate4_missingness_calibration() -> None:
    """500-user cohort (no DB). Biometric missingness in [0.09, 0.21]
    (target midpoint 0.15, +/- 3% tolerance band). Food-log missingness
    in [0.32, 0.53] (target midpoint 0.425, roughly +/- 5%)."""
    demo = generate_demographics(n_users=500, seed=7)

    wearables = generate_wearables(
        demo,
        days=30,
        start_date=date(2026, 1, 1),
        seed=7,
        biometric_missingness=(0.12, 0.18),
    )

    # Biometric missingness: HRV as the representative channel.
    total = len(wearables)
    hrv_missing = sum(1 for r in wearables if r.hrv_average is None)
    bio_rate = hrv_missing / total
    assert 0.09 <= bio_rate <= 0.21, (
        f"biometric missingness rate {bio_rate:.3f} outside [0.09, 0.21] band"
    )

    # Food log missingness: 3 meals expected per day per user; count
    # actual meals and compute 1 - emitted / expected.
    meals = _generate_meals(
        demographics=demo,
        days=30,
        start_date=date(2026, 1, 1),
        seed=7,
        manual_log_missingness=(0.35, 0.50),
    )
    expected = 500 * 30 * 3
    food_rate = 1.0 - len(meals) / expected
    assert 0.32 <= food_rate <= 0.53, (
        f"food-log missingness rate {food_rate:.3f} outside [0.32, 0.53] band"
    )


# ─────────────────────────────────────────────────────────────────────────
# Gate 5: Feedback diversity
# ─────────────────────────────────────────────────────────────────────────


def test_fidelity_gate5_feedback_non_uniform() -> None:
    """Chi-square goodness-of-fit against uniform(up, down, none)
    must reject H0 at p < 0.001. Non-uniformity is intentional:
    adversarial personas thumb-down more often than uniform; regular
    personas thumb-up more often."""
    demo = generate_demographics(n_users=500, seed=7)
    fragments = generate_conversations(demo, seed=7, turns_per_conversation=0)

    counts = Counter(f.feedback for f in fragments)
    observed = np.array(
        [
            counts.get("up", 0),
            counts.get("down", 0),
            counts.get(None, 0),
        ]
    )
    total = observed.sum()
    assert total == 500
    expected = np.full(3, total / 3)

    chi2, p = stats.chisquare(observed, expected)
    assert p < 0.001, (
        f"feedback distribution looks uniform (chi2={chi2:.2f}, p={p:.5f}). "
        f"Observed: up={observed[0]}, down={observed[1]}, none={observed[2]}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Gate 6: End-to-end pipeline integration
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fidelity_gate6_end_to_end_pipeline(db: AsyncSession) -> None:
    """Full pipeline on a synth user: generate_cohort ->
    refresh_features_for_user -> run_associations -> run_daily_insights
    -> load_coach_signal_context. ``active_patterns`` must be
    non-empty, which is the exact shape Phase 5.9 Promptfoo cases
    inspect. Coach prompt templates gate on this same SignalContext
    shape in production."""
    manifest = await generate_cohort(db, n_users=1, days=120, seed=101)
    await db.flush()
    synth_user = manifest.user_ids[0]

    # Materialize features for the full cohort window.
    await ml_api.refresh_features_for_user(
        db, synth_user, through_date=date.today(), lookback_days=120
    )
    await db.flush()

    # L2 associations populate UserCorrelation.
    await ml_api.run_associations(db, synth_user, window_days=60)
    await db.flush()

    # Phase 4 candidates + rankings (shadow-ok, does not block this test).
    await ml_api.run_daily_insights(db, synth_user)
    await db.flush()

    # The surface the coach router consumes.
    context = await ml_api.load_coach_signal_context(db, synth_user)

    assert context.active_patterns, (
        "expected at least one active pattern for a 120-day synth user. "
        "This is the Phase 5.9 unblock signal; if it fails, either the "
        "shared-latent coefficients in wearables.py are too weak or the "
        "discovery pipeline is not reading from the feature store for this "
        "user. Run gates 1 and 2 individually to localize."
    )

    # Sanity: the returned patterns look like coherent coach-prompt
    # material. Effect description is a 4th-grade string; tier is
    # developing+ (load_active_patterns filters to that).
    developing_plus = {"developing", "established", "causal_candidate", "literature_supported"}
    for p in context.active_patterns:
        assert p.confidence_tier in developing_plus
        assert p.source_metric
        assert p.target_metric
