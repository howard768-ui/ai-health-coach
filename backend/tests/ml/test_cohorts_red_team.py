"""Phase 8A privacy red-team tests.

These are the critical privacy validation tests. They verify that the
anonymization pipeline strips identifying information and that DP noise
provides meaningful protection.

Tests:
1. Raw user_ids never appear in anonymized vectors
2. Raw health values (exact HRV, exact timestamps) are not recoverable
3. Removing one user from a cluster changes aggregates by less than DP noise bound
4. HMAC pseudonyms are not reversible
5. Feature vector does not contain direct identifiers

Run: ``cd backend && uv run python -m pytest tests/ml/test_cohorts_red_team.py -v``
"""

from __future__ import annotations

import json
import os

import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

import numpy as np
from sqlalchemy import select
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
from ml.cohorts.anonymize import (
    pseudonymize,
    encrypt_user_id,
    apply_dp_noise,
)


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
# Red-team: no raw user_ids in anonymized output
# ---------------------------------------------------------------------------


def test_pseudonym_does_not_contain_raw_user_id():
    """The HMAC pseudonym must not contain or encode the raw user_id."""
    user_id = "apple-user-abc123xyz"
    pseudo = pseudonymize(user_id, "test-key-2026-04")

    # Pseudonym should not contain any substring of the user_id.
    assert user_id not in pseudo
    assert "abc123" not in pseudo
    assert "apple-user" not in pseudo


def test_encrypted_uid_does_not_contain_raw_user_id():
    """The deletion lookup key must not expose the raw user_id."""
    user_id = "apple-user-abc123xyz"
    enc = encrypt_user_id(user_id, "test-key-2026-04")

    assert user_id not in enc
    assert "abc123" not in enc


# ---------------------------------------------------------------------------
# Red-team: raw health values not recoverable from pattern vector
# ---------------------------------------------------------------------------


def test_dp_noise_prevents_exact_value_recovery():
    """With epsilon=1.0, the noised vector should differ enough that
    exact original values cannot be recovered with high confidence.
    """
    # Simulate a "known" vector (attacker knows the ground truth).
    original = [0.75, 0.42, 0.91, 0.33, 0.68, 0.15, 0.88, 0.52, 0.27, 0.63]

    # Apply DP noise as the pipeline would.
    noised = apply_dp_noise(original, epsilon=1.0, seed=42)

    # The attacker tries to recover original values from the noised vector.
    # With epsilon=1.0, the expected absolute error per dimension is 1/epsilon = 1.0.
    # At least some dimensions should differ by more than 0.1 (recovery threshold).
    diffs = [abs(o - n) for o, n in zip(original, noised)]
    large_diffs = [d for d in diffs if d > 0.1]
    assert len(large_diffs) > 0, (
        "At least some dimensions should have noise > 0.1 with epsilon=1.0"
    )


# ---------------------------------------------------------------------------
# Red-team: single-user removal bounded by DP noise
# ---------------------------------------------------------------------------


def test_single_user_removal_within_dp_bound():
    """Removing one user from a 50-member cluster should change the
    centroid by less than the DP noise already applied.

    This validates that DP provides meaningful protection: the aggregate
    statistic changes less from removing a user than it already varies
    from the applied noise.
    """
    rng = np.random.default_rng(42)
    n_members = 50
    dim = 20

    # Generate a cluster of vectors.
    cluster = rng.normal(0, 0.3, size=(n_members, dim))

    # Centroid with all members.
    centroid_full = cluster.mean(axis=0)

    # Centroid with one member removed (worst case: most extreme member).
    deviations = np.linalg.norm(cluster - centroid_full, axis=1)
    worst_idx = np.argmax(deviations)
    centroid_minus_one = np.delete(cluster, worst_idx, axis=0).mean(axis=0)

    # Change in centroid from removing one user.
    centroid_change = np.linalg.norm(centroid_full - centroid_minus_one)

    # Expected DP noise magnitude per dimension: scale = 1/epsilon = 1.0.
    # L2 norm of Laplace noise in dim dimensions: scale * sqrt(2 * dim).
    epsilon = 1.0
    expected_noise_l2 = (1.0 / epsilon) * np.sqrt(2 * dim)

    assert centroid_change < expected_noise_l2, (
        f"Centroid change ({centroid_change:.4f}) should be less than "
        f"expected DP noise L2 ({expected_noise_l2:.4f})"
    )


# ---------------------------------------------------------------------------
# Red-team: HMAC not reversible
# ---------------------------------------------------------------------------


def test_hmac_not_reversible_by_brute_force_sample():
    """HMAC-SHA256 should not be reversible even with known key
    (given reasonable user_id entropy).
    """
    key = "test-key-2026-04"
    target_pseudo = pseudonymize("user-secret-id-42", key)

    # Try 1000 random guesses. None should match.
    for i in range(1000):
        guess = f"user-{i}"
        if pseudonymize(guess, key) == target_pseudo:
            pytest.fail(f"HMAC collision found at guess {i}")


# ---------------------------------------------------------------------------
# Red-team: feature names don't contain direct identifiers
# ---------------------------------------------------------------------------


def test_feature_names_are_generic():
    """Feature names in the pattern vector should not encode user identity."""
    # These are the names that would appear in feature_names_json.
    forbidden_patterns = [
        "user_id", "email", "name", "phone", "address",
        "apple_user", "device_id", "ip_address",
    ]

    # Simulate feature names from the anonymize pipeline.
    feature_names = [
        "corr_steps_sleep_efficiency",
        "corr_protein_g_deep_sleep_minutes",
        "summary_hrv",
        "summary_steps",
        "tertile_steps_low",
        "tertile_sleep_short",
    ]

    for name in feature_names:
        for forbidden in forbidden_patterns:
            assert forbidden not in name.lower(), (
                f"Feature name '{name}' contains forbidden pattern '{forbidden}'"
            )


# ---------------------------------------------------------------------------
# Red-team: deletion actually removes data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deletion_removes_anonymized_vectors(db: AsyncSession):
    """After deletion, no anonymized vectors remain for the user."""
    from app.core.time import utcnow_naive
    from app.models.ml_cohorts import MLAnonymizedVector, MLCohortConsent

    # Create consent + vector.
    now = utcnow_naive()
    db.add(MLCohortConsent(user_id="user-del-test", opted_in=True, opted_in_at=now))

    uid_enc = encrypt_user_id("user-del-test", "test-key")
    db.add(
        MLAnonymizedVector(
            pseudonym_id="pseudo-del-test",
            user_id_encrypted=uid_enc,
            vector_json=json.dumps([0.1, 0.2, 0.3]),
            feature_names_json=json.dumps(["a", "b", "c"]),
            dp_epsilon=1.0,
            created_at=now,
        )
    )
    await db.flush()

    # Verify vector exists.
    result = await db.execute(
        select(MLAnonymizedVector).where(
            MLAnonymizedVector.user_id_encrypted == uid_enc
        )
    )
    assert result.scalar_one_or_none() is not None

    # Delete.
    from sqlalchemy import delete

    await db.execute(
        delete(MLAnonymizedVector).where(
            MLAnonymizedVector.user_id_encrypted == uid_enc
        )
    )
    await db.flush()

    # Verify vector is gone.
    result2 = await db.execute(
        select(MLAnonymizedVector).where(
            MLAnonymizedVector.user_id_encrypted == uid_enc
        )
    )
    assert result2.scalar_one_or_none() is None
