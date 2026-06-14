"""L1 baselines: per-(user, metric) STL decomposition + BOCPD change points.

Takes a biometric feature series from the Phase 1 feature store, fits an STL
(Season-Trend-Loess) model when there's enough history, and runs streaming
BOCPD to detect regime shifts. Also re-certifies historical change points
with ``ruptures.Pelt`` as a backstop — BOCPD is fast but approximate; ruptures
gives deterministic, paper-ready segmentation.

All heavy imports (pandas, numpy, statsmodels, ruptures) are lazy.

Entry point is ``compute_baselines_for_user``, which writes to
``ml_baselines`` + ``ml_change_points``. Called by the nightly
``baselines_job`` in ``app/tasks/scheduler.py`` through ``ml.api``.

See ``~/.claude/plans/golden-floating-creek.md`` Phase 2 for the broader
design. The plan cites ``maxCPs λ=100`` from the 2025 BOCPD evaluation paper
as the winning streaming config, which corresponds to a geometric hazard
rate of 1/100 per day (one expected change every ~3 months). That is our
default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.ext.asyncio import AsyncSession


MODEL_VERSION = "1.0.0"

# Metrics we build baselines for. Keep this small in Phase 2 — every metric
# adds compute; start with the five headline biometrics.
BASELINE_METRICS: tuple[str, ...] = (
    "hrv",
    "resting_hr",
    "sleep_efficiency",
    "sleep_duration_minutes",
    "readiness_score",
)


# ─────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BaselineStats:
    """Output of ``compute_baseline_for_series``.

    Every field may be None when there is not enough data; the caller
    decides whether to persist (typically only if the first five are set).
    """

    metric_key: str
    window_days: int
    trend_mean: float | None
    trend_slope: float | None
    seasonal_amplitude: float | None
    residual_std: float | None
    last_observed_date: str  # YYYY-MM-DD
    observed_days_in_window: int


@dataclass
class ChangePointEvent:
    """Output of BOCPD or ruptures detection."""

    metric_key: str
    change_date: str  # YYYY-MM-DD
    probability: float | None  # None for ruptures
    magnitude: float  # |mean_after - mean_before|, metric's native units
    detector: str  # "bocpd" | "ruptures"


@dataclass
class BaselineRun:
    """Summary of one compute_baselines_for_user call."""

    user_id: str
    through_date: date
    baselines_written: int = 0
    change_points_written: int = 0
    metrics_skipped_short_history: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# STL baseline
# ─────────────────────────────────────────────────────────────────────────


def compute_baseline_for_series(
    series: "pd.Series",
    metric_key: str,
    min_history_days: int = 28,
) -> BaselineStats | None:
    """Fit STL on ``series`` and summarize trend + seasonality + residual.

    Returns ``None`` if fewer than ``min_history_days`` observed (non-NaN)
    values. Otherwise returns a ``BaselineStats`` whose last four fields
    summarize the decomposition.

    Implementation notes:
    - Weekly seasonality (period=7) is the right shape for daily biometrics.
    - Robust STL (``robust=True``) downweights outliers so a single travel
      day does not skew the trend.
    - Missing days are linearly interpolated within the observed span
      before fit — STL does not natively accept NaN but is tolerant of
      short interpolated gaps. Leading / trailing NaNs are dropped so we
      don't extrapolate.
    """
    import numpy as np
    from statsmodels.tsa.seasonal import STL

    # Drop leading / trailing NaNs so we fit only on the observed span.
    valid = series.dropna()
    if valid.empty:
        return None
    obs_span = series.loc[valid.index[0] : valid.index[-1]]
    observed_days = int(valid.shape[0])
    last_observed_date = str(valid.index[-1])
    window_days = int(obs_span.shape[0])

    if observed_days < min_history_days:
        # Caller treats None as "skip" (counts toward metrics_skipped_short_history).
        return None

    # Interpolate across short gaps within the observed span so STL can run.
    interpolated = obs_span.astype(float).interpolate(
        method="linear", limit_direction="both"
    )

    # period=7 for weekly seasonality on daily data.
    try:
        stl = STL(interpolated.values, period=7, robust=True).fit()
    except (ValueError, np.linalg.LinAlgError):
        # Degenerate series (constant, too short after interpolation, etc.).
        # Return a stats row with the metadata but None for computed fields so
        # the caller can log + decide whether to retry. Caller also checks
        # residual_std before running change-point detection.
        return BaselineStats(
            metric_key=metric_key,
            window_days=window_days,
            trend_mean=None,
            trend_slope=None,
            seasonal_amplitude=None,
            residual_std=None,
            last_observed_date=last_observed_date,
            observed_days_in_window=observed_days,
        )

    trend = stl.trend
    seasonal = stl.seasonal
    residual = stl.resid

    trend_mean = float(np.nanmean(trend))
    # Slope in units per day via a simple linear fit over the trend.
    if len(trend) >= 2:
        x = np.arange(len(trend), dtype=float)
        slope, _intercept = np.polyfit(x, trend, 1)
        trend_slope = float(slope)
    else:
        trend_slope = None

    # Seasonal amplitude — range over the most recent 7 days.
    last_week = seasonal[-7:] if len(seasonal) >= 7 else seasonal
    seasonal_amplitude = float(np.nanmax(last_week) - np.nanmin(last_week))

    residual_std = float(np.nanstd(residual, ddof=1)) if len(residual) > 1 else None

    return BaselineStats(
        metric_key=metric_key,
        window_days=window_days,
        trend_mean=trend_mean,
        trend_slope=trend_slope,
        seasonal_amplitude=seasonal_amplitude,
        residual_std=residual_std,
        last_observed_date=last_observed_date,
        observed_days_in_window=observed_days,
    )


# ─────────────────────────────────────────────────────────────────────────
# BOCPD (Bayesian Online Change Point Detection)
# ─────────────────────────────────────────────────────────────────────────


def fit_bocpd(
    series: "pd.Series",
    hazard_rate: float = 1.0 / 100.0,
    threshold_prob: float = 0.5,
    sigma: float | None = None,
) -> list[ChangePointEvent]:
    """Run streaming BOCPD on a Gaussian likelihood with known variance.

    Implementation follows Adams & MacKay 2007 with a simple Normal model
    (mean unknown, variance estimated once from the whole series). For
    daily biometric series on the order of months, that approximation is
    plenty accurate and keeps the inner loop vectorizable.

    Parameters
    ----------
    series : pandas Series indexed by date strings (YYYY-MM-DD)
        Must be float-valued. NaNs are skipped (the run length increments
        without a predictive update).
    hazard_rate : float
        Constant hazard per step. ``1/100`` corresponds to roughly one
        expected change every three months on daily data, matching the
        ``λ=100`` config recommended by the 2025 BOCPD evaluation paper.
    threshold_prob : float
        Emit a change point when ``P(r_t = 0 | x_{1:t}) > threshold_prob``.
        0.5 is a conservative default — tune per-metric if false positives
        creep up.
    sigma : float or None
        Measurement noise std. If ``None``, estimated as the empirical std
        of the series. Fixed across the run.

    Returns
    -------
    list of ``ChangePointEvent`` with ``detector="bocpd"``. Empty if no
    change point exceeds the threshold.
    """
    import numpy as np

    values = series.to_numpy(dtype=float)
    dates = list(series.index)
    n = len(values)
    if n < 3:
        return []

    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return []
    if sigma is None:
        sigma = float(np.std(finite, ddof=1))
        if sigma == 0.0:
            return []

    # Empirical-Bayes prior mean: grand mean of the series. Using 0 here (as
    # an earlier version did) makes the ``r=0`` predictive vanish for any
    # real biometric (HRV ~40 does not look at all like N(0, sigma)) and the
    # change branch never fires. Using the grand mean keeps the prior
    # plausible for values anywhere in the observed range.
    mu_0 = float(np.mean(finite))

    # Run-length probability distribution. R[r] = P(run_length = r).
    # We keep a dense vector of length t+1 at each step.
    R = np.array([1.0])
    run_means = np.array([mu_0])
    run_counts = np.array([0])

    events: list[ChangePointEvent] = []

    for t, x in enumerate(values):
        if not np.isfinite(x):
            # Missing observation. Increment run lengths, keep run_means.
            # Prepend r=0 slot with tiny mass so we can still reset later.
            growth = R * (1 - hazard_rate)
            change = np.array([R.sum() * hazard_rate])
            R = np.concatenate([change, growth])
            run_means = np.concatenate([[mu_0], run_means])
            run_counts = np.concatenate([[0], run_counts])
            total = R.sum()
            if total > 0:
                R /= total
            continue

        # Predictive Gaussian for each surviving run length.
        # Predictive variance inflates with 1/r for small r; this approximates
        # the full Student-t predictive.
        inflated_var = sigma**2 * (1 + 1.0 / np.maximum(run_counts, 1))
        pred_std = np.sqrt(inflated_var)
        z = (x - run_means) / pred_std
        predictive = np.exp(-0.5 * z * z) / (pred_std * np.sqrt(2 * np.pi))

        # Growth: run continues with probability (1 - hazard_rate).
        growth = R * predictive * (1 - hazard_rate)
        # Change: new run at r=0, total mass is sum of all transitions.
        change_mass = float((R * predictive * hazard_rate).sum())

        R_new = np.concatenate([[change_mass], growth])
        total = R_new.sum()
        if total > 0:
            R_new /= total

        # Update sufficient statistics: each run length r becomes r+1 with
        # mean updated by x. The new r=0 run starts at the empirical-Bayes
        # prior mean (grand mean of the series) so the next-step predictive
        # stays in a plausible range for real biometrics.
        new_counts = run_counts + 1
        new_means = (run_means * run_counts + x) / new_counts
        run_means = np.concatenate([[mu_0], new_means])
        run_counts = np.concatenate([[0], new_counts])

        # Detect change: cumulative posterior mass on small run lengths. A
        # single-step r=0 spike is rare because the new run takes a few steps
        # to accumulate mass; looking at the mass in the first few slots
        # captures the "MAP run length just reset" signal more robustly. See
        # Adams & MacKay 2007 figure 3 — the bright low-r band is the
        # canonical change-point visualization.
        short_run_slots = min(5, len(R_new))
        short_run_mass = float(R_new[:short_run_slots].sum())
        if t >= 10 and short_run_mass > threshold_prob:
            pre_mean = run_means[-1] if len(run_means) > 1 else float(x)
            events.append(
                ChangePointEvent(
                    metric_key="",  # caller fills in
                    change_date=str(dates[t]),
                    probability=short_run_mass,
                    magnitude=float(abs(x - pre_mean)),
                    detector="bocpd",
                )
            )
            # Pragmatic refractory: after firing, collapse to r=0 so we don't
            # emit 10 change points on consecutive days for the same shift.
            R_new = np.array([1.0])
            run_means = np.array([mu_0])
            run_counts = np.array([0])

        R = R_new

    return events


def fit_ruptures(
    series: "pd.Series",
    metric_key: str,
    penalty: float = 10.0,
    min_size: int = 7,
) -> list[ChangePointEvent]:
    """Offline ruptures.Pelt for deterministic segmentation.

    Used as the weekly re-certification backstop for BOCPD's streaming
    output. Returns at most one event per detected change.
    """
    import numpy as np
    import ruptures as rpt

    valid = series.dropna()
    if valid.shape[0] < min_size * 2:
        return []
    values = valid.to_numpy(dtype=float)
    dates = list(valid.index)

    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(values.reshape(-1, 1))
        breakpoints = algo.predict(pen=penalty)
    except (ValueError, np.linalg.LinAlgError):
        return []

    events: list[ChangePointEvent] = []
    for bp in breakpoints[:-1]:  # last breakpoint is always the final index
        if bp <= 0 or bp >= len(values):
            continue
        before = values[max(0, bp - min_size) : bp]
        after = values[bp : min(len(values), bp + min_size)]
        if len(before) == 0 or len(after) == 0:
            continue
        magnitude = float(abs(np.mean(after) - np.mean(before)))
        events.append(
            ChangePointEvent(
                metric_key=metric_key,
                change_date=str(dates[bp]),
                probability=None,
                magnitude=magnitude,
                detector="ruptures",
            )
        )
    return events


# ─────────────────────────────────────────────────────────────────────────
# User-level orchestrator
# ─────────────────────────────────────────────────────────────────────────


async def compute_baselines_for_user(
    db: "AsyncSession",
    user_id: str,
    through_date: date,
    window_days: int = 180,
    min_history_days: int = 28,
) -> BaselineRun:
    """Fit baselines + detect change points for every baseline metric.

    Reads feature values from the Phase 1 store, fits STL per metric, runs
    BOCPD + ruptures, persists results. Caller owns the transaction.
    """
    import time

    from ml.features.store import get_feature_frame

    run = BaselineRun(user_id=user_id, through_date=through_date)
    start = through_date - timedelta(days=window_days - 1)

    t0 = time.perf_counter()
    frame = await get_feature_frame(
        db,
        user_id,
        feature_keys=list(BASELINE_METRICS),
        start=start,
        end=through_date,
        include_imputed=False,
    )
    run.timings_ms["feature_fetch"] = (time.perf_counter() - t0) * 1000

    baselines_to_write: list[BaselineStats] = []
    change_points_to_write: list[ChangePointEvent] = []

    for metric in BASELINE_METRICS:
        if metric not in frame.columns:
            run.metrics_skipped_short_history.append(metric)
            continue
        series = frame[metric]

        t0 = time.perf_counter()
        stats = compute_baseline_for_series(series, metric_key=metric, min_history_days=min_history_days)
        run.timings_ms[f"stl:{metric}"] = (time.perf_counter() - t0) * 1000

        if stats is None:
            # Fewer than min_history_days of observed data.
            run.metrics_skipped_short_history.append(metric)
            continue

        baselines_to_write.append(stats)

        if stats.residual_std is None or stats.trend_mean is None:
            # No usable baseline, skip change detection too.
            continue

        t0 = time.perf_counter()
        bocpd_events = fit_bocpd(series, sigma=stats.residual_std)
        for e in bocpd_events:
            e.metric_key = metric
            change_points_to_write.append(e)
        run.timings_ms[f"bocpd:{metric}"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ruptures_events = fit_ruptures(series, metric_key=metric)
        change_points_to_write.extend(ruptures_events)
        run.timings_ms[f"ruptures:{metric}"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    await _upsert_baselines(db, user_id, baselines_to_write)
    await _upsert_change_points(db, user_id, change_points_to_write)
    run.timings_ms["upsert"] = (time.perf_counter() - t0) * 1000

    run.baselines_written = len(baselines_to_write)
    run.change_points_written = len(change_points_to_write)
    return run


async def _upsert_baselines(
    db: "AsyncSession",
    user_id: str,
    stats: list[BaselineStats],
) -> None:
    from app.core.time import utcnow_naive
    from app.models.ml_baselines import MLBaseline

    if not stats:
        return

    metric_keys = {s.metric_key for s in stats}
    # Replace existing rows for these (user, metric, version) tuples.
    await db.execute(
        delete(MLBaseline).where(
            MLBaseline.user_id == user_id,
            MLBaseline.metric_key.in_(metric_keys),
            MLBaseline.model_version == MODEL_VERSION,
        )
    )
    now = utcnow_naive()
    for s in stats:
        db.add(
            MLBaseline(
                user_id=user_id,
                metric_key=s.metric_key,
                window_days=s.window_days,
                trend_mean=s.trend_mean,
                trend_slope=s.trend_slope,
                seasonal_amplitude=s.seasonal_amplitude,
                residual_std=s.residual_std,
                last_observed_date=s.last_observed_date,
                observed_days_in_window=s.observed_days_in_window,
                model_version=MODEL_VERSION,
                computed_at=now,
            )
        )


async def _upsert_change_points(
    db: "AsyncSession",
    user_id: str,
    events: list[ChangePointEvent],
) -> None:
    from app.core.time import utcnow_naive
    from app.models.ml_baselines import MLChangePoint

    if not events:
        return

    # Use INSERT OR IGNORE semantics via query-before-insert; simpler than
    # dialect-specific upserts for the event volume we expect (<< 1000/run).
    existing_result = await db.execute(
        select(MLChangePoint.metric_key, MLChangePoint.change_date, MLChangePoint.detector).where(
            MLChangePoint.user_id == user_id,
        )
    )
    existing = {
        (r.metric_key, r.change_date, r.detector)
        for r in existing_result.all()
    }

    now = utcnow_naive()
    for e in events:
        key = (e.metric_key, e.change_date, e.detector)
        if key in existing:
            continue
        db.add(
            MLChangePoint(
                user_id=user_id,
                metric_key=e.metric_key,
                change_date=e.change_date,
                detector=e.detector,
                probability=e.probability,
                magnitude=e.magnitude,
                model_version=MODEL_VERSION,
                detected_at=now,
            )
        )
        existing.add(key)
