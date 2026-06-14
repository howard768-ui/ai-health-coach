"""L3 Granger causality: directional evidence for L2 developing+ pairs.

Takes ``developing`` or higher pairs from ``UserCorrelation``, tests whether
the source metric Granger-causes the target using
``statsmodels.tsa.stattools.grangercausalitytests``. An ADF stationarity gate
precedes each test; non-stationary series are first-differenced once and
retried. If still non-stationary, the pair is skipped (logged, not surfaced).

When the F-test passes at p < 0.05, the corresponding ``UserCorrelation``
row gets ``directional_support=True`` and a row is written to
``ml_directional_tests``. Failed tests also get a row (for audit).

All heavy imports (numpy, pandas, statsmodels) are lazy inside function
bodies per the cold-boot contract.

Entry point is ``run_granger_for_user``, called from ``ml.api`` through
the weekly ``correlation_engine_job`` in ``app/tasks/scheduler.py``.

See ``~/.claude/plans/golden-floating-creek.md`` line 172 (L3 row).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    import numpy as np
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

MODEL_VERSION = "granger-1.0.0"

# Granger configuration.
GRANGER_MAX_LAG = 7
GRANGER_P_THRESHOLD = 0.05
ADF_P_THRESHOLD = 0.05
MIN_OBSERVATIONS = 30


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GrangerResult:
    """One Granger test outcome for a single (source, target, lag) triple."""

    source_metric: str
    target_metric: str
    lag_days: int
    is_stationary: bool
    differencing_order: int
    f_statistic: float | None
    p_value: float | None
    max_lag_tested: int
    optimal_lag: int | None
    significant: bool


@dataclass
class GrangerReport:
    """Summary of a full Granger run for one user."""

    user_id: str
    pairs_tested: int = 0
    pairs_stationary: int = 0
    pairs_significant: int = 0
    pairs_skipped_non_stationary: int = 0
    pairs_skipped_insufficient_data: int = 0
    rows_written: int = 0
    correlations_updated: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------


def _check_stationarity(series: "np.ndarray") -> tuple[bool, float]:
    """Run ADF test and return (is_stationary, p_value)."""
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series, autolag="AIC")
    p_val = result[1]
    return p_val < ADF_P_THRESHOLD, p_val


def _ensure_stationary(
    x: "np.ndarray", y: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray", bool, int]:
    """Return (x, y, is_stationary, differencing_order).

    Tries levels first. If either series is non-stationary, first-differences
    both and retries. If still non-stationary, returns (x, y, False, 1).
    """
    import numpy as np

    x_stat, _ = _check_stationarity(x)
    y_stat, _ = _check_stationarity(y)
    if x_stat and y_stat:
        return x, y, True, 0

    # First-difference both (keeps alignment).
    x_diff = np.diff(x)
    y_diff = np.diff(y)
    if len(x_diff) < MIN_OBSERVATIONS:
        return x_diff, y_diff, False, 1

    x_stat2, _ = _check_stationarity(x_diff)
    y_stat2, _ = _check_stationarity(y_diff)
    if x_stat2 and y_stat2:
        return x_diff, y_diff, True, 1

    return x_diff, y_diff, False, 1


# ---------------------------------------------------------------------------
# Core Granger test
# ---------------------------------------------------------------------------


def _run_granger_test(
    source: "np.ndarray",
    target: "np.ndarray",
    max_lag: int,
) -> tuple[float | None, float | None, int | None]:
    """Run Granger causality: does source Granger-cause target?

    Returns (best_f, best_p, optimal_lag) across lags 1..max_lag, where
    best_p is Bonferroni-corrected by the number of lags actually evaluated
    (selecting the minimum raw p across several lags is a multiple-comparison
    problem; downstream significance gates consume this value directly).
    Returns (None, None, None) if the test cannot be run.
    """
    import numpy as np
    from statsmodels.tsa.stattools import grangercausalitytests

    n = len(source)
    effective_max_lag = min(max_lag, n // 3)
    if effective_max_lag < 1:
        return None, None, None

    # grangercausalitytests expects a 2D array: [target, source] columns.
    data = np.column_stack([target, source])

    try:
        results = grangercausalitytests(data, maxlag=effective_max_lag, verbose=False)
    except Exception:
        logger.debug(
            "Granger test failed for series pair (n=%d, max_lag=%d)",
            n,
            effective_max_lag,
            exc_info=True,
        )
        return None, None, None

    best_f = None
    best_p = None
    best_lag = None
    lags_evaluated = 0

    for lag_order in range(1, effective_max_lag + 1):
        if lag_order not in results:
            continue
        test_result = results[lag_order]
        # test_result is a tuple: (dict of test results, [ols_restricted, ols_unrestricted])
        f_test = test_result[0].get("ssr_ftest")
        if f_test is None:
            continue
        lags_evaluated += 1
        f_stat, p_val = f_test[0], f_test[1]
        if best_p is None or p_val < best_p:
            best_f = float(f_stat)
            best_p = float(p_val)
            best_lag = lag_order

    # Bonferroni-correct for lag selection. Taking the minimum p across up
    # to 7 correlated tests without correction inflates the false-positive
    # rate to roughly 30% under independence; the uncorrected minimum then
    # gated L4 causal discovery and flowed into coach-cited insights.
    # Audit P3/D2 (round-1 #33). Correcting by the number of lags actually
    # evaluated is conservative for correlated tests, which is the right
    # direction for evidence the coach presents as causal support.
    if best_p is not None and lags_evaluated > 1:
        best_p = min(1.0, best_p * lags_evaluated)

    return best_f, best_p, best_lag


# ---------------------------------------------------------------------------
# Single-pair orchestrator
# ---------------------------------------------------------------------------


def compute_granger_for_pair(
    source_series: "np.ndarray",
    target_series: "np.ndarray",
    source_name: str,
    target_name: str,
    lag_days: int,
) -> GrangerResult | None:
    """Run the full Granger pipeline for one pair.

    Returns None if there are fewer than MIN_OBSERVATIONS aligned points.
    Otherwise returns a GrangerResult (significant or not, for persistence).
    """
    import numpy as np

    # Drop NaNs from aligned pair.
    mask = ~(np.isnan(source_series) | np.isnan(target_series))
    x = source_series[mask]
    y = target_series[mask]

    if len(x) < MIN_OBSERVATIONS:
        return None

    # Stationarity gate.
    x_s, y_s, is_stationary, diff_order = _ensure_stationary(x, y)

    if not is_stationary:
        return GrangerResult(
            source_metric=source_name,
            target_metric=target_name,
            lag_days=lag_days,
            is_stationary=False,
            differencing_order=diff_order,
            f_statistic=None,
            p_value=None,
            max_lag_tested=GRANGER_MAX_LAG,
            optimal_lag=None,
            significant=False,
        )

    # Run Granger test.
    f_stat, p_val, opt_lag = _run_granger_test(x_s, y_s, GRANGER_MAX_LAG)

    if f_stat is None:
        return GrangerResult(
            source_metric=source_name,
            target_metric=target_name,
            lag_days=lag_days,
            is_stationary=is_stationary,
            differencing_order=diff_order,
            f_statistic=None,
            p_value=None,
            max_lag_tested=GRANGER_MAX_LAG,
            optimal_lag=None,
            significant=False,
        )

    return GrangerResult(
        source_metric=source_name,
        target_metric=target_name,
        lag_days=lag_days,
        is_stationary=is_stationary,
        differencing_order=diff_order,
        f_statistic=f_stat,
        p_value=p_val,
        max_lag_tested=GRANGER_MAX_LAG,
        optimal_lag=opt_lag,
        significant=(p_val is not None and p_val < GRANGER_P_THRESHOLD),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def persist_granger_results(
    db: "AsyncSession",
    user_id: str,
    results: list[GrangerResult],
) -> tuple[int, int]:
    """Write results to ``ml_directional_tests`` and update ``UserCorrelation``.

    Returns (rows_written, correlations_updated).
    """
    from app.core.time import utcnow_naive
    from app.models.correlation import UserCorrelation
    from app.models.ml_discovery import MLDirectionalTest

    now = utcnow_naive()
    rows_written = 0
    correlations_updated = 0

    for r in results:
        db.add(
            MLDirectionalTest(
                user_id=user_id,
                source_metric=r.source_metric,
                target_metric=r.target_metric,
                lag_days=r.lag_days,
                is_stationary=r.is_stationary,
                differencing_order=r.differencing_order,
                f_statistic=r.f_statistic,
                p_value=r.p_value,
                max_lag_tested=r.max_lag_tested,
                optimal_lag=r.optimal_lag,
                significant=r.significant,
                model_version=MODEL_VERSION,
                computed_at=now,
            )
        )
        rows_written += 1

        if r.significant:
            existing = await db.execute(
                select(UserCorrelation).where(
                    UserCorrelation.user_id == user_id,
                    UserCorrelation.source_metric == r.source_metric,
                    UserCorrelation.target_metric == r.target_metric,
                    UserCorrelation.lag_days == r.lag_days,
                )
            )
            record = existing.scalar_one_or_none()
            if record is not None:
                record.directional_support = True
                record.last_validated_at = now
                correlations_updated += 1

    await db.flush()
    return rows_written, correlations_updated


# ---------------------------------------------------------------------------
# User-level orchestrator
# ---------------------------------------------------------------------------


async def run_granger_for_user(
    db: "AsyncSession",
    user_id: str,
    window_days: int = 90,
) -> GrangerReport:
    """Run L3 Granger on all developing+ pairs for a user.

    Reads ``UserCorrelation`` rows at ``developing`` tier or above,
    pulls the feature matrix from the Phase 1 store, runs Granger
    per pair, and persists results.

    Does NOT commit. Caller owns the transaction.
    """
    import time

    import numpy as np

    from ml.features.store import get_feature_frame
    from app.models.correlation import UserCorrelation

    report = GrangerReport(user_id=user_id)

    # 1. Load developing+ pairs.
    t0 = time.perf_counter()
    eligible_tiers = ("developing", "established", "literature_supported")
    stmt = select(UserCorrelation).where(
        UserCorrelation.user_id == user_id,
        UserCorrelation.confidence_tier.in_(eligible_tiers),
    )
    result = await db.execute(stmt)
    pairs = result.scalars().all()
    report.timings_ms["load_pairs"] = (time.perf_counter() - t0) * 1000

    if not pairs:
        logger.info("run_granger_for_user(%s): no developing+ pairs, skipping", user_id)
        return report

    # 2. Collect needed feature keys.
    feature_keys: set[str] = set()
    for p in pairs:
        feature_keys.add(p.source_metric)
        feature_keys.add(p.target_metric)

    # 3. Pull feature frame (wider window for Granger).
    t0 = time.perf_counter()
    today = date.today()
    start = today - timedelta(days=window_days)
    frame = await get_feature_frame(
        db,
        user_id,
        feature_keys=sorted(feature_keys),
        start=start,
        end=today,
        include_imputed=False,
    )
    report.timings_ms["feature_fetch"] = (time.perf_counter() - t0) * 1000

    # 4. Run Granger for each pair.
    t0 = time.perf_counter()
    granger_results: list[GrangerResult] = []
    for p in pairs:
        src_col = p.source_metric
        tgt_col = p.target_metric

        if src_col not in frame.columns or tgt_col not in frame.columns:
            report.pairs_skipped_insufficient_data += 1
            continue

        source_arr = frame[src_col].values.astype(np.float64)
        target_arr = frame[tgt_col].values.astype(np.float64)

        gr = compute_granger_for_pair(
            source_arr, target_arr, src_col, tgt_col, p.lag_days
        )
        if gr is None:
            report.pairs_skipped_insufficient_data += 1
            continue

        report.pairs_tested += 1
        granger_results.append(gr)

        if gr.is_stationary or gr.differencing_order == 0:
            report.pairs_stationary += 1
        if not gr.is_stationary:
            report.pairs_skipped_non_stationary += 1
        if gr.significant:
            report.pairs_significant += 1

    report.timings_ms["granger_compute"] = (time.perf_counter() - t0) * 1000

    # 5. Persist.
    t0 = time.perf_counter()
    rows, updated = await persist_granger_results(db, user_id, granger_results)
    report.rows_written = rows
    report.correlations_updated = updated
    report.timings_ms["persist"] = (time.perf_counter() - t0) * 1000

    logger.info(
        "run_granger_for_user(%s): tested=%d stationary=%d significant=%d "
        "skipped_nonstat=%d skipped_data=%d rows=%d updated=%d",
        user_id,
        report.pairs_tested,
        report.pairs_stationary,
        report.pairs_significant,
        report.pairs_skipped_non_stationary,
        report.pairs_skipped_insufficient_data,
        report.rows_written,
        report.correlations_updated,
    )
    return report
