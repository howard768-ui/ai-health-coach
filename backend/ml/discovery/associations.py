"""L2 associations: dynamic cross-domain correlation discovery.

Replaces the hand-rolled statistics in
``backend/app/services/correlation_engine.py`` with scipy + statsmodels and
expands the fixed 8-pair test to dynamic pair generation from the Phase 1
feature store.

Key behaviors preserved from the legacy engine:
- Dual-method agreement: both Pearson AND Spearman must agree on sign.
- Benjamini-Hochberg FDR correction across every pair tested per run.
- Minimum paired sample size = 14 before a pair is considered.
- Confidence tier thresholds: developing >= 30, established >= 60.
- Output persists to the same ``UserCorrelation`` table with legacy names
  in ``source_metric`` / ``target_metric`` so downstream services that
  read those strings (e.g., Knowledge Graph seeds, future narrator code)
  keep working without a rename migration.

Key behaviors added in Phase 3:
- scipy.stats.pearsonr / spearmanr (exact two-tailed p, no hand-rolled
  t approximation).
- statsmodels.stats.multitest.multipletests(method="fdr_bh") (monotonic,
  vectorized).
- Dynamic pair generation: every (driver, outcome) pair across the
  activity / nutrition / contextual feature space vs the biometric
  feature space, at lag 0 and lag 1, capped at ``max_pairs=200`` per run.

See ~/.claude/plans/golden-floating-creek.md Phase 3 and the Cross-Domain
Correlation Research wiki page for the statistical rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.ext.asyncio import AsyncSession


MODEL_VERSION = "associations-1.0.0"

MIN_SAMPLE_SIZE = 14
TIER_DEVELOPING = 30
TIER_ESTABLISHED = 60
FDR_ALPHA = 0.10

# Seed pairs. feature-store names; kept stable across versions. The legacy
# ``METRIC_PAIRS`` referenced raw-schema names like ``"protein_intake"`` and
# ``"hrv_next_day"``; the translation layer is ``LEGACY_NAME`` below, which
# preserves the user-facing names stored in ``UserCorrelation``.
#
# Note: dinner_hour is intentionally skipped — it is not in the v1 catalog.
# Adding it is a follow-up (needs a per-meal timestamp feature).
SEED_PAIRS: tuple[tuple[str, str, int, str | None], ...] = (
    ("protein_g", "deep_sleep_minutes", 0, "positive"),
    ("calories", "sleep_efficiency", 0, None),
    ("steps", "sleep_efficiency", 0, "positive"),
    ("workout_duration_sum_minutes", "hrv", 1, "positive"),
    ("workout_duration_sum_minutes", "readiness_score", 1, None),
    ("sleep_efficiency", "steps", 1, "positive"),
    ("resting_hr", "readiness_score", 0, "negative"),
)


# Stored names in ``UserCorrelation`` for backwards compatibility with the
# legacy engine's output. When persisting, we look these up; when the pair
# uses a lag, we suffix ``_next_day`` on the target (matching legacy).
LEGACY_NAME: dict[str, str] = {
    "protein_g": "protein_intake",
    "calories": "total_calories",
    "deep_sleep_minutes": "deep_sleep_seconds",
    "sleep_efficiency": "sleep_efficiency",
    "steps": "steps",
    "workout_duration_sum_minutes": "workout_duration",
    "hrv": "hrv",
    "readiness_score": "readiness",
    "resting_hr": "resting_hr",
    "active_calories": "active_calories",
    "protein_g.7d_rolling_mean": "protein_intake_7d_rolling_mean",
    "hrv.7d_rolling_mean": "hrv_7d_rolling_mean",
}


@dataclass
class AssociationResult:
    """Output of the associations pipeline. Shape preserved from legacy
    ``CorrelationResult`` so the scheduler's upsert logic can be reused.
    """

    # Feature-store names. Mapped to legacy names on persist.
    source_metric: str
    target_metric: str
    lag_days: int
    pearson_r: float
    spearman_r: float
    p_value: float
    sample_size: int
    direction: str  # positive | negative
    strength: float
    methods_agree: bool
    fdr_adjusted_p: float = 0.0
    confidence_tier: str = "emerging"  # emerging | developing | established | literature_supported
    literature_match: bool = False
    literature_ref: str | None = None
    effect_size_description: str = ""


@dataclass
class AssociationsReport:
    """Summary of a single ``compute_associations`` + ``persist_associations`` cycle."""

    user_id: str
    window_days: int
    pairs_tested: int = 0
    pairs_with_enough_data: int = 0
    significant_results: int = 0
    dynamic_pairs_generated: int = 0
    rows_written: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Pair generation
# ─────────────────────────────────────────────────────────────────────────


def _generate_dynamic_pairs(
    exclude_keys: set[tuple[str, str, int]],
    max_pairs: int,
) -> list[tuple[str, str, int, str | None]]:
    """Cross all activity / nutrition features with all biometric features.

    Lag 0 and lag 1 are both emitted. ``exclude_keys`` is the set of
    (source, target, lag) triples already in the seed set so we do not
    duplicate. Cap at ``max_pairs`` to bound the test count for FDR.
    """
    from ml.features import catalog

    drivers: list[str] = []
    outcomes: list[str] = []
    for spec in catalog.iter_catalog():
        if "." in spec.key:
            # Skip derived features for now — they introduce collinearity
            # (e.g., 7d_rolling_mean is a smoothed version of the raw).
            # Phase 4+ can reintroduce if SHAP signals they help.
            continue
        if spec.category in {"activity", "nutrition"}:
            drivers.append(spec.key)
        elif spec.category == "biometric_raw":
            outcomes.append(spec.key)

    pairs: list[tuple[str, str, int, str | None]] = []
    for lag in (0, 1):
        for driver in drivers:
            for outcome in outcomes:
                key = (driver, outcome, lag)
                if key in exclude_keys:
                    continue
                pairs.append((driver, outcome, lag, None))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


# ─────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────


def _align_pair(
    frame: "pd.DataFrame",
    source: str,
    target: str,
    lag_days: int,
) -> "pd.DataFrame":
    """Return a 2-col DataFrame with aligned (source[d], target[d + lag]) rows.

    NaNs dropped. Order preserved. Empty frame if either column is missing.
    """
    import pandas as pd

    if source not in frame.columns or target not in frame.columns:
        return pd.DataFrame(columns=["x", "y"])

    x = frame[source].astype(float)
    # target[d + lag] paired with source[d] means shift target BACK by lag
    # in time — i.e., at index d look at the value that was originally at
    # index d + lag. pandas Series.shift(-lag) does exactly that.
    y = frame[target].astype(float).shift(-lag_days)
    aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    return aligned


def _correlate_pair(
    aligned: "pd.DataFrame",
    source: str,
    target: str,
    lag_days: int,
) -> AssociationResult | None:
    """Compute scipy pearson + spearman on an aligned pair.

    Returns ``None`` when sample size is below threshold or either column
    is constant (zero variance -> NaN r).
    """
    import numpy as np
    from scipy import stats

    n = aligned.shape[0]
    if n < MIN_SAMPLE_SIZE:
        return None

    x_arr = aligned["x"].to_numpy(dtype=float)
    y_arr = aligned["y"].to_numpy(dtype=float)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return None

    pr = stats.pearsonr(x_arr, y_arr)
    sr = stats.spearmanr(x_arr, y_arr)

    # scipy returns objects with statistic + pvalue attributes (and a tuple API).
    p_r = float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0])
    p_p = float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1])
    s_r = float(sr.statistic) if hasattr(sr, "statistic") else float(sr[0])
    s_p = float(sr.pvalue) if hasattr(sr, "pvalue") else float(sr[1])

    if np.isnan(p_r) or np.isnan(s_r):
        return None

    methods_agree = (p_r > 0 and s_r > 0) or (p_r < 0 and s_r < 0)
    direction = "positive" if p_r > 0 else "negative"
    strength = abs(p_r)

    return AssociationResult(
        source_metric=source,
        target_metric=target,
        lag_days=lag_days,
        pearson_r=round(p_r, 4),
        spearman_r=round(s_r, 4),
        # Conservative: use the larger of the two p-values so we don't
        # report significance that only one method supports.
        p_value=round(max(p_p, s_p), 6),
        sample_size=int(n),
        direction=direction,
        strength=round(strength, 3),
        methods_agree=methods_agree,
    )


def _apply_fdr(results: list[AssociationResult], alpha: float = FDR_ALPHA) -> None:
    """In-place: set ``fdr_adjusted_p`` using BH-FDR across all results."""
    if not results:
        return
    from statsmodels.stats.multitest import multipletests

    pvals = [r.p_value for r in results]
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    for r, adj in zip(results, pvals_corrected):
        r.fdr_adjusted_p = round(float(adj), 6)


def _describe_effect(r: AssociationResult) -> str:
    """Natural-language description. Voice-compliant (no em dashes)."""
    source = r.source_metric.replace("_", " ")
    target = r.target_metric.replace("_", " ")
    if r.lag_days > 0:
        target = f"{target} the next day" if r.lag_days == 1 else f"{target} {r.lag_days} days later"
    if r.direction == "positive":
        return f"When your {source} is higher, your {target} tends to be higher too."
    return f"When your {source} is higher, your {target} tends to be lower."


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────


async def compute_associations(
    db: "AsyncSession",
    user_id: str,
    window_days: int = 30,
    include_seed_pairs: bool = True,
    include_dynamic_pairs: bool = True,
    max_pairs: int = 200,
) -> tuple[list[AssociationResult], AssociationsReport]:
    """Discover cross-domain associations over the trailing window.

    Reads the Phase 1 feature store (no direct SQL to the raw tables),
    computes dual-method correlations on every eligible pair, applies
    BH-FDR, and returns significant results plus a report for telemetry.

    Does NOT commit — the caller owns the transaction. For the full
    persist + literature-validate cycle, call
    ``persist_associations`` after this.
    """
    import time

    from ml.features.store import get_feature_frame

    report = AssociationsReport(user_id=user_id, window_days=window_days)

    # 1. Build pair list.
    t0 = time.perf_counter()
    pair_list: list[tuple[str, str, int, str | None]] = []
    seed_keys: set[tuple[str, str, int]] = set()
    if include_seed_pairs:
        for src, tgt, lag, dir_hint in SEED_PAIRS:
            pair_list.append((src, tgt, lag, dir_hint))
            seed_keys.add((src, tgt, lag))
    if include_dynamic_pairs:
        remaining = max(0, max_pairs - len(pair_list))
        dynamic = _generate_dynamic_pairs(exclude_keys=seed_keys, max_pairs=remaining)
        pair_list.extend(dynamic)
        report.dynamic_pairs_generated = len(dynamic)
    pair_list = pair_list[:max_pairs]
    report.pairs_tested = len(pair_list)
    report.timings_ms["pair_generation"] = (time.perf_counter() - t0) * 1000

    if not pair_list:
        return [], report

    # 2. Pull feature frame. window_days + 1 so lag-1 has enough tail.
    t0 = time.perf_counter()
    needed_features = {src for src, _, _, _ in pair_list} | {tgt for _, tgt, _, _ in pair_list}
    today = date.today()
    start = today - timedelta(days=window_days)
    frame = await get_feature_frame(
        db,
        user_id,
        feature_keys=sorted(needed_features),
        start=start,
        end=today,
        include_imputed=False,
    )
    report.timings_ms["feature_fetch"] = (time.perf_counter() - t0) * 1000

    # 3. Compute every pair.
    t0 = time.perf_counter()
    results: list[AssociationResult] = []
    for src, tgt, lag, _dir_hint in pair_list:
        aligned = _align_pair(frame, src, tgt, lag)
        res = _correlate_pair(aligned, src, tgt, lag)
        if res is None:
            continue
        report.pairs_with_enough_data += 1
        results.append(res)
    report.timings_ms["correlate"] = (time.perf_counter() - t0) * 1000

    # 4. BH-FDR across every pair we successfully tested.
    _apply_fdr(results)

    # 5. Filter + tier-assign.
    significant: list[AssociationResult] = []
    for r in results:
        if r.p_value >= 0.05 or not r.methods_agree:
            continue
        if r.sample_size >= TIER_ESTABLISHED and r.fdr_adjusted_p < FDR_ALPHA:
            r.confidence_tier = "established"
        elif r.sample_size >= TIER_DEVELOPING and r.fdr_adjusted_p < FDR_ALPHA:
            r.confidence_tier = "developing"
        else:
            r.confidence_tier = "emerging"
        r.effect_size_description = _describe_effect(r)
        significant.append(r)

    significant.sort(key=lambda x: x.strength, reverse=True)
    report.significant_results = len(significant)
    return significant, report


# ─────────────────────────────────────────────────────────────────────────
# Persistence (writes to UserCorrelation + literature validation)
# ─────────────────────────────────────────────────────────────────────────


def _to_legacy_name(feature_key: str, as_target: bool = False, lag_days: int = 0) -> str:
    """Map a feature-store key to the legacy string stored in UserCorrelation.

    Matches the historical naming (``protein_intake``, ``readiness``, etc.) so
    the Knowledge Graph and anything else reading ``source_metric`` /
    ``target_metric`` keeps working. Appends ``_next_day`` to the target when
    the pair is lagged, mirroring legacy METRIC_PAIRS.
    """
    base = LEGACY_NAME.get(feature_key, feature_key)
    if as_target and lag_days > 0:
        return f"{base}_next_day"
    return base


async def persist_associations(
    db: "AsyncSession",
    user_id: str,
    results: list[AssociationResult],
) -> int:
    """Upsert results into ``UserCorrelation`` with legacy-compatible naming.

    Validates each result against ``literature_service``; on a hit, sets
    ``literature_match=True``, stores the DOI, and upgrades the confidence
    tier to ``literature_supported``.

    Returns the count of rows inserted or updated.
    """
    from app.models.correlation import UserCorrelation
    from app.services.literature import literature_service
    from app.core.time import utcnow_naive

    if not results:
        return 0

    touched = 0
    for r in results:
        legacy_source = _to_legacy_name(r.source_metric)
        legacy_target = _to_legacy_name(r.target_metric, as_target=True, lag_days=r.lag_days)

        lit = literature_service.validate_correlation(legacy_source, legacy_target, r.direction)
        if lit:
            # Record the match unconditionally (it is informational), but only
            # UPGRADE the tier when the pair cleared this run's BH-FDR gate.
            # Previously any literature hit jumped straight to the highest
            # surfaceable tier even when fdr_adjusted_p failed the gate, so a
            # statistically non-significant correlation could be presented as
            # literature-backed evidence. Audit P3/D2 (round-1 #34). A
            # literature match compensates for sample size, not for failing
            # the user's own significance correction.
            r.literature_match = True
            r.literature_ref = lit.doi
            if r.fdr_adjusted_p < FDR_ALPHA:
                r.confidence_tier = "literature_supported"

        # Upsert by (user, legacy_source, legacy_target, lag_days).
        existing = await db.execute(
            select(UserCorrelation).where(
                UserCorrelation.user_id == user_id,
                UserCorrelation.source_metric == legacy_source,
                UserCorrelation.target_metric == legacy_target,
                UserCorrelation.lag_days == r.lag_days,
            )
        )
        record = existing.scalar_one_or_none()
        now = utcnow_naive()

        if record:
            record.pearson_r = r.pearson_r
            record.spearman_r = r.spearman_r
            record.p_value = r.p_value
            record.fdr_adjusted_p = r.fdr_adjusted_p
            record.sample_size = r.sample_size
            record.strength = r.strength
            record.confidence_tier = r.confidence_tier
            record.literature_match = r.literature_match
            record.effect_size_description = r.effect_size_description
            record.last_validated_at = now
            if r.literature_ref is not None:
                record.literature_ref = r.literature_ref
        else:
            db.add(
                UserCorrelation(
                    user_id=user_id,
                    source_metric=legacy_source,
                    target_metric=legacy_target,
                    lag_days=r.lag_days,
                    direction=r.direction,
                    pearson_r=r.pearson_r,
                    spearman_r=r.spearman_r,
                    p_value=r.p_value,
                    fdr_adjusted_p=r.fdr_adjusted_p,
                    sample_size=r.sample_size,
                    strength=r.strength,
                    confidence_tier=r.confidence_tier,
                    literature_match=r.literature_match,
                    literature_ref=r.literature_ref,
                    effect_size_description=r.effect_size_description,
                    discovered_at=now,
                    last_validated_at=now,
                )
            )
        touched += 1

    await db.flush()
    return touched


async def run_associations_for_user(
    db: "AsyncSession",
    user_id: str,
    window_days: int = 30,
) -> AssociationsReport:
    """Convenience orchestrator: compute + persist in one call.

    The scheduler uses this; tests can either use this or split the two
    calls for finer assertions.
    """
    results, report = await compute_associations(db, user_id, window_days=window_days)
    report.rows_written = await persist_associations(db, user_id, results)
    return report
