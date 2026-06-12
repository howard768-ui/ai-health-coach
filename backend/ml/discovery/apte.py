"""L5 APTE: Average Period Treatment Effect for n-of-1 experiments.

MVP uses scipy permutation test on designed AB experiments (14+14 days).
Phase 9B adds the full Daza g-formula with RandomForest for observational
passive experiments.

All heavy imports (scipy, numpy) are lazy inside function bodies per the
cold-boot contract.

Entry point is ``run_apte_for_experiment``, called from ``ml.api``.

See ``~/.claude/plans/golden-floating-creek.md`` line 174 (L5 row).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MODEL_VERSION = "apte-permutation-1.0.0"

MIN_OBSERVATIONS = 5


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class APTEResult:
    """Output of a single APTE estimation."""

    apte: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effect_size_d: float
    baseline_mean: float
    treatment_mean: float
    baseline_n: int
    treatment_n: int
    method: str = "permutation_test"


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def compute_apte_permutation(
    baseline_values: "np.ndarray",
    treatment_values: "np.ndarray",
    n_resamples: int = 9999,
) -> APTEResult | None:
    """Compute APTE via scipy permutation test.

    Returns None if either phase has fewer than MIN_OBSERVATIONS values.
    """
    import numpy as np
    from scipy.stats import permutation_test

    baseline = baseline_values[~np.isnan(baseline_values)]
    treatment = treatment_values[~np.isnan(treatment_values)]

    if len(baseline) < MIN_OBSERVATIONS or len(treatment) < MIN_OBSERVATIONS:
        return None

    baseline_mean = float(np.mean(baseline))
    treatment_mean = float(np.mean(treatment))
    apte = treatment_mean - baseline_mean

    # Cohen's d: standardized effect size.
    pooled_std = float(np.sqrt(
        ((len(baseline) - 1) * np.var(baseline, ddof=1)
         + (len(treatment) - 1) * np.var(treatment, ddof=1))
        / (len(baseline) + len(treatment) - 2)
    ))
    effect_size_d = apte / pooled_std if pooled_std > 0 else 0.0

    # Permutation test for the mean difference.
    def statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = permutation_test(
        (treatment, baseline),
        statistic,
        n_resamples=n_resamples,
        alternative="two-sided",
    )

    # Bootstrap CI (2.5th and 97.5th percentiles of the null distribution
    # shifted by the observed effect).
    null_dist = result.null_distribution
    ci_lower = float(np.percentile(null_dist, 2.5))
    ci_upper = float(np.percentile(null_dist, 97.5))
    # Shift CI to be centered on the observed APTE.
    ci_lower = apte - (ci_upper - ci_lower) / 2
    ci_upper = apte + (ci_upper - ci_lower) / 2

    return APTEResult(
        apte=round(apte, 6),
        ci_lower=round(ci_lower, 6),
        ci_upper=round(ci_upper, 6),
        p_value=round(float(result.pvalue), 6),
        effect_size_d=round(effect_size_d, 4),
        baseline_mean=round(baseline_mean, 4),
        treatment_mean=round(treatment_mean, 4),
        baseline_n=len(baseline),
        treatment_n=len(treatment),
    )


# ---------------------------------------------------------------------------
# Autocorrelation assessment
# ---------------------------------------------------------------------------


def assess_autocorrelation(series: "np.ndarray") -> tuple[float, int]:
    """Compute lag-1 autocorrelation and effective sample size.

    Returns (rho, n_effective).
    """
    import numpy as np

    clean = series[~np.isnan(series)]
    n = len(clean)
    if n < 5:
        return 0.0, n

    mean = np.mean(clean)
    demeaned = clean - mean
    # Lag-1 autocorrelation.
    numerator = np.sum(demeaned[:-1] * demeaned[1:])
    denominator = np.sum(demeaned ** 2)
    rho = float(numerator / denominator) if denominator > 0 else 0.0
    rho = max(-0.99, min(0.99, rho))  # clamp to avoid division by zero

    # Effective sample size.
    n_eff = max(1, int(n * (1 - rho) / (1 + rho)))

    return round(rho, 4), n_eff


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def persist_apte_result(
    db: "AsyncSession",
    experiment_id: int,
    user_id: str,
    treatment_metric: str,
    outcome_metric: str,
    result: APTEResult,
    compliant_baseline: int,
    compliant_treatment: int,
) -> int:
    """Write APTE result to ml_n_of_1_results. Returns the row id."""
    from app.core.time import utcnow_naive
    from app.models.ml_experiments import MLNof1Result

    row = MLNof1Result(
        experiment_id=experiment_id,
        user_id=user_id,
        treatment_metric=treatment_metric,
        outcome_metric=outcome_metric,
        apte=result.apte,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        p_value=result.p_value,
        effect_size_d=result.effect_size_d,
        baseline_mean=result.baseline_mean,
        treatment_mean=result.treatment_mean,
        baseline_n=result.baseline_n,
        treatment_n=result.treatment_n,
        compliant_days_baseline=compliant_baseline,
        compliant_days_treatment=compliant_treatment,
        method=result.method,
        model_version=MODEL_VERSION,
        computed_at=utcnow_naive(),
    )
    db.add(row)
    await db.flush()
    return row.id


# ---------------------------------------------------------------------------
# Experiment orchestrator
# ---------------------------------------------------------------------------


async def run_apte_for_experiment(
    db: "AsyncSession",
    experiment_id: int,
) -> APTEResult | None:
    """Pull feature data for an experiment's phases, compute APTE, persist.

    Returns the APTEResult or None if data is insufficient.
    Does NOT commit. Caller owns the transaction.
    """
    import numpy as np
    from datetime import date


    from app.models.ml_experiments import MLExperiment
    from ml.features.store import get_feature_frame

    experiment = await db.get(MLExperiment, experiment_id)
    if experiment is None:
        logger.warning("Experiment %d not found", experiment_id)
        return None

    # Parse phase dates.
    baseline_start = date.fromisoformat(experiment.started_at.strftime("%Y-%m-%d"))
    baseline_end = date.fromisoformat(experiment.baseline_end)
    treatment_start = date.fromisoformat(experiment.treatment_start)
    treatment_end = date.fromisoformat(experiment.treatment_end)

    # Check compliance gate.
    if (experiment.compliant_days_baseline < experiment.min_compliance
            or experiment.compliant_days_treatment < experiment.min_compliance):
        logger.info(
            "Experiment %d: insufficient compliance (baseline=%d, treatment=%d, min=%d)",
            experiment_id,
            experiment.compliant_days_baseline,
            experiment.compliant_days_treatment,
            experiment.min_compliance,
        )
        return None

    # Pull outcome metric for both phases.
    frame_baseline = await get_feature_frame(
        db, experiment.user_id,
        feature_keys=[experiment.outcome_metric],
        start=baseline_start, end=baseline_end,
        include_imputed=False,
    )
    frame_treatment = await get_feature_frame(
        db, experiment.user_id,
        feature_keys=[experiment.outcome_metric],
        start=treatment_start, end=treatment_end,
        include_imputed=False,
    )

    if experiment.outcome_metric not in frame_baseline.columns:
        return None
    if experiment.outcome_metric not in frame_treatment.columns:
        return None

    baseline_vals = frame_baseline[experiment.outcome_metric].values.astype(np.float64)
    treatment_vals = frame_treatment[experiment.outcome_metric].values.astype(np.float64)

    result = compute_apte_permutation(baseline_vals, treatment_vals)
    if result is None:
        return None

    # Persist.
    await persist_apte_result(
        db, experiment_id, experiment.user_id,
        experiment.treatment_metric, experiment.outcome_metric,
        result,
        experiment.compliant_days_baseline,
        experiment.compliant_days_treatment,
    )

    # Update experiment status.
    from app.core.time import utcnow_naive
    experiment.status = "completed"
    experiment.completed_at = utcnow_naive()
    await db.flush()

    logger.info(
        "APTE for experiment %d: apte=%.4f p=%.4f d=%.4f",
        experiment_id, result.apte, result.p_value, result.effect_size_d,
    )
    return result
