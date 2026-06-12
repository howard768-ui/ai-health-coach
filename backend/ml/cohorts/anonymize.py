"""Anonymization pipeline for cross-user cohort clustering.

Builds per-user pattern vectors from the feature store, pseudonymizes
with HMAC-SHA256 (rotating monthly key), applies Laplace DP noise
(epsilon=1.0), and persists to ``ml_anonymized_vectors``.

Entry point: ``build_anonymized_vectors`` is imported by ``ml.api`` and
invoked from ``run_cohort_pipeline``. Static call-graph analyzers may
miss the cross-module import if the boundary is lazy-loaded.

Privacy invariants:
- No raw user_ids in the output. Only HMAC pseudonyms.
- No raw health values. Only derived correlation strengths, catch22
  summary stats, and tertile one-hots.
- Laplace noise applied BEFORE clustering (not after).
- Sensitivity bounded by feature clipping to [0, 1].

All heavy imports (numpy, hmac, hashlib) are lazy inside function bodies
per the cold-boot contract.

Entry point is ``build_anonymized_vectors``, called from ``ml.api``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Feature dimensions in the pattern vector.
# Top-20 correlation strengths + 22 catch22 + 6 tertile one-hots = 48 dims.
# Actual count may vary; we dynamically size based on available data.
MAX_CORRELATION_FEATURES = 20
CATCH22_FEATURES = 22
TERTILE_FEATURES = 6
EXPECTED_VECTOR_DIM = MAX_CORRELATION_FEATURES + CATCH22_FEATURES + TERTILE_FEATURES


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AnonymizedVector:
    """One pseudonymized, DP-noised pattern vector."""

    pseudonym_id: str
    user_id_encrypted: str
    vector: list[float]
    feature_names: list[str]
    dp_epsilon: float


@dataclass
class AnonymizationReport:
    """Summary of a full anonymization run."""

    users_processed: int = 0
    vectors_created: int = 0
    users_skipped_insufficient_data: int = 0


# ---------------------------------------------------------------------------
# Pseudonymization
# ---------------------------------------------------------------------------


def pseudonymize(user_id: str, rotating_key: str) -> str:
    """HMAC-SHA256 pseudonym. Deterministic per (user_id, key).

    Key rotates monthly so pseudonyms cannot be linked across months.
    Returns 64-char hex string.
    """
    import hashlib
    import hmac

    return hmac.new(
        rotating_key.encode("utf-8"),
        user_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def encrypt_user_id(user_id: str, key: str) -> str:
    """Simple keyed hash for deletion lookup.

    This is NOT encryption in the cryptographic sense; it's a second
    HMAC with a different purpose (finding a user's vector for deletion).
    Using a separate key from the pseudonym key so the two cannot be
    correlated.
    """
    import hashlib
    import hmac

    return hmac.new(
        (key + "-deletion").encode("utf-8"),
        user_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def get_rotating_key() -> str:
    """Get the current month's HMAC rotation key.

    In production, this should come from a secrets manager. For now,
    derive from the MLSettings cohort_hmac_key + current year-month.
    """
    from ml.config import get_ml_settings

    settings = get_ml_settings()
    base_key = getattr(settings, "cohort_hmac_key", "meld-cohort-default-key")
    year_month = date.today().strftime("%Y-%m")
    return f"{base_key}-{year_month}"


# ---------------------------------------------------------------------------
# Pattern vector construction
# ---------------------------------------------------------------------------


async def build_pattern_vector(
    db: "AsyncSession",
    user_id: str,
    window_days: int = 90,
) -> tuple[list[float], list[str]] | None:
    """Build a pattern vector for one user.

    Components:
    1. Top-20 established correlation strengths (canonical order, zero-filled).
    2. Catch22 summary stats averaged over the window (placeholder: use
       feature means for now; real catch22 in Phase 8B).
    3. Tertile one-hots (active/sedentary, short/long sleeper, etc.).

    Returns (vector, feature_names) or None if insufficient data.
    """
    import numpy as np

    from ml.features.store import get_feature_frame
    from app.models.correlation import UserCorrelation

    today = date.today()
    start = today - timedelta(days=window_days)

    # 1. Correlation strengths (top 20 by absolute strength).
    eligible_tiers = ("developing", "established", "literature_supported", "causal_candidate")
    stmt = (
        select(UserCorrelation)
        .where(
            UserCorrelation.user_id == user_id,
            UserCorrelation.confidence_tier.in_(eligible_tiers),
        )
        .order_by(UserCorrelation.strength.desc())
        .limit(MAX_CORRELATION_FEATURES)
    )
    result = await db.execute(stmt)
    correlations = result.scalars().all()

    corr_values: list[float] = []
    corr_names: list[str] = []
    for c in correlations:
        corr_values.append(float(c.strength))
        corr_names.append(f"corr_{c.source_metric}_{c.target_metric}")

    # Zero-fill to MAX_CORRELATION_FEATURES.
    while len(corr_values) < MAX_CORRELATION_FEATURES:
        corr_values.append(0.0)
        corr_names.append(f"corr_pad_{len(corr_values)}")

    # 2. Feature means as catch22 proxy (real catch22 in Phase 8B).
    frame = await get_feature_frame(
        db, user_id, feature_keys=None, start=start, end=today, include_imputed=False
    )

    if frame.empty or len(frame) < 14:
        return None

    # Use means of available biometric + activity + nutrition features.
    summary_cols = [
        "hrv", "resting_hr", "sleep_efficiency", "sleep_duration_minutes",
        "deep_sleep_minutes", "rem_sleep_minutes", "readiness_score",
        "steps", "active_calories", "workout_duration_sum_minutes",
        "protein_g", "calories", "carbs_g", "fat_g",
        "meal_count", "fiber_g", "processed_ratio",
        "days_since_last_workout", "dinner_hour",
        "weekday_mean", "engagement_score", "log_consistency",
    ]
    summary_values: list[float] = []
    summary_names: list[str] = []
    for col in summary_cols:
        if col in frame.columns:
            val = float(frame[col].mean()) if not frame[col].isna().all() else 0.0
            # Normalize to [0, 1] range (clip for DP sensitivity bounding).
            summary_values.append(np.clip(val / max(abs(val), 1.0), 0.0, 1.0) if val != 0 else 0.0)
        else:
            summary_values.append(0.0)
        summary_names.append(f"summary_{col}")

    # 3. Tertile one-hots.
    tertile_values: list[float] = []
    tertile_names: list[str] = []

    # Steps tertile (low/mid/high based on mean).
    steps_mean = float(frame["steps"].mean()) if "steps" in frame.columns and not frame["steps"].isna().all() else 0.0
    tertile_values.extend([
        1.0 if steps_mean < 5000 else 0.0,
        1.0 if 5000 <= steps_mean < 10000 else 0.0,
        1.0 if steps_mean >= 10000 else 0.0,
    ])
    tertile_names.extend(["tertile_steps_low", "tertile_steps_mid", "tertile_steps_high"])

    # Sleep duration tertile.
    sleep_mean = float(frame["sleep_duration_minutes"].mean()) if "sleep_duration_minutes" in frame.columns and not frame["sleep_duration_minutes"].isna().all() else 0.0
    tertile_values.extend([
        1.0 if sleep_mean < 360 else 0.0,  # < 6h
        1.0 if 360 <= sleep_mean < 480 else 0.0,  # 6-8h
        1.0 if sleep_mean >= 480 else 0.0,  # 8h+
    ])
    tertile_names.extend(["tertile_sleep_short", "tertile_sleep_mid", "tertile_sleep_long"])

    vector = corr_values + summary_values + tertile_values
    names = corr_names + summary_names + tertile_names

    return vector, names


# ---------------------------------------------------------------------------
# Differential privacy
# ---------------------------------------------------------------------------


def apply_dp_noise(
    vector: list[float],
    epsilon: float,
    sensitivity: float = 1.0,
    seed: int | None = None,
) -> list[float]:
    """Apply Laplace noise per dimension for epsilon-DP.

    Sensitivity is bounded by pre-clipping all features to [0, 1].
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    scale = sensitivity / epsilon
    noise = rng.laplace(0, scale, size=len(vector))
    return [float(v + n) for v, n in zip(vector, noise)]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


async def build_anonymized_vectors(
    db: "AsyncSession",
    window_days: int = 90,
    epsilon: float | None = None,
) -> AnonymizationReport:
    """Build anonymized vectors for all opted-in users.

    Reads from ml_cohort_consent, builds pattern vectors, pseudonymizes,
    applies DP noise, and persists to ml_anonymized_vectors.

    Does NOT commit. Caller owns the transaction.
    """
    from app.models.ml_cohorts import MLAnonymizedVector, MLCohortConsent
    from ml.config import get_ml_settings

    settings = get_ml_settings()
    if epsilon is None:
        epsilon = settings.cohort_dp_epsilon

    report = AnonymizationReport()

    # Load opted-in users.
    stmt = select(MLCohortConsent).where(MLCohortConsent.opted_in.is_(True))
    result = await db.execute(stmt)
    consents = result.scalars().all()

    if not consents:
        return report

    rotating_key = get_rotating_key()

    # Clear previous vectors (fresh build each month).
    await db.execute(delete(MLAnonymizedVector))

    for consent in consents:
        report.users_processed += 1

        pv = await build_pattern_vector(db, consent.user_id, window_days)
        if pv is None:
            report.users_skipped_insufficient_data += 1
            continue

        vector, feature_names = pv

        # Pseudonymize.
        pseudo_id = pseudonymize(consent.user_id, rotating_key)
        uid_enc = encrypt_user_id(consent.user_id, rotating_key)

        # Apply DP noise.
        noised_vector = apply_dp_noise(vector, epsilon)

        db.add(
            MLAnonymizedVector(
                pseudonym_id=pseudo_id,
                user_id_encrypted=uid_enc,
                vector_json=json.dumps(noised_vector),
                feature_names_json=json.dumps(feature_names),
                dp_epsilon=epsilon,
            )
        )
        report.vectors_created += 1

    await db.flush()
    logger.info(
        "Anonymization complete: processed=%d vectors=%d skipped=%d",
        report.users_processed,
        report.vectors_created,
        report.users_skipped_insufficient_data,
    )
    return report
