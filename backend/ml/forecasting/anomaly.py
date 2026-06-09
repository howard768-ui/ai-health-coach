"""Residual-based anomaly detection on recently-observed biometrics.

Given yesterday's (or today's) observation and its corresponding forecast,
flag the day as anomalous when ``|observed - forecasted|`` exceeds
``threshold * residual_std``. Optionally confirm with a BOCPD-fired change
point in the same 48-hour window (two-signal gating from the plan).

Writes to ``ml_anomalies``. Read by Phase 4's insight candidate pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import and_, delete, select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


MODEL_VERSION = "residual-z-1.0.0"
ANOMALY_Z_THRESHOLD = 2.5  # |residual / residual_std| > 2.5 flags the day
BOCPD_CONFIRMATION_WINDOW_DAYS = 2


@dataclass
class AnomalyEvent:
    """One flagged anomaly, ready for persistence."""

    metric_key: str
    observation_date: str
    observed_value: float
    forecasted_value: float
    residual: float
    z_score: float
    direction: str  # "high" | "low"
    confirmed_by_bocpd: bool


@dataclass
class AnomalyRun:
    """Summary of one ``detect_anomalies_for_user`` invocation."""

    user_id: str
    through_date: date
    anomalies_written: int = 0
    metrics_scanned: list[str] = field(default_factory=list)


async def detect_anomalies_for_user(
    db: "AsyncSession",
    user_id: str,
    through_date: date,
    lookback_days: int = 7,
    threshold_z: float = ANOMALY_Z_THRESHOLD,
) -> AnomalyRun:
    """Scan the last ``lookback_days`` and emit anomaly rows.

    The z denominator is a robust (MAD-based) scale of the ensemble's own
    forecast residuals in the scan window, so numerator and denominator come
    from the same model; with fewer than 10 forecast/observation pairs it
    falls back to the STL in-sample ``residual_std`` from ``ml_baselines``.
    Metrics with no baseline are skipped entirely. Cross-references
    ``ml_change_points`` for two-signal BOCPD confirmation within a 48h
    window of each anomaly.
    """
    import numpy as np

    from app.models.ml_baselines import MLBaseline, MLChangePoint, MLForecast
    from ml.features.store import get_feature_frame
    from ml.forecasting.residuals import FORECAST_METRICS

    run = AnomalyRun(user_id=user_id, through_date=through_date)

    # 1. Load baselines to get per-metric residual_std.
    baseline_result = await db.execute(
        select(MLBaseline).where(
            MLBaseline.user_id == user_id,
            MLBaseline.metric_key.in_(list(FORECAST_METRICS)),
        )
    )
    baselines = {b.metric_key: b for b in baseline_result.scalars().all()}
    if not baselines:
        # Nothing to check against — baselines need >= 28 days of data first.
        return run

    # 2. Load the last N days of observed feature values.
    start = through_date - timedelta(days=lookback_days - 1)
    frame = await get_feature_frame(
        db,
        user_id,
        feature_keys=list(FORECAST_METRICS),
        start=start,
        end=through_date,
        include_imputed=False,
    )

    # 3. For each candidate day x metric, find a forecast that was made
    # BEFORE the observation and check the residual.
    forecasts_result = await db.execute(
        select(MLForecast).where(
            MLForecast.user_id == user_id,
            MLForecast.metric_key.in_(list(FORECAST_METRICS)),
            MLForecast.target_date >= start.isoformat(),
            MLForecast.target_date <= through_date.isoformat(),
            MLForecast.made_on < through_date.isoformat(),
        )
    )
    # Prefer the most recent made_on per (metric, target_date).
    best_forecast: dict[tuple[str, str], "MLForecast"] = {}
    for f in forecasts_result.scalars().all():
        key = (f.metric_key, f.target_date)
        current = best_forecast.get(key)
        if current is None or (f.made_on > current.made_on):
            best_forecast[key] = f

    # 4. Look up BOCPD change points within +/- 2 days for two-signal gate.
    cp_result = await db.execute(
        select(MLChangePoint).where(
            MLChangePoint.user_id == user_id,
            MLChangePoint.detector == "bocpd",
            MLChangePoint.change_date >= (start - timedelta(days=BOCPD_CONFIRMATION_WINDOW_DAYS)).isoformat(),
            MLChangePoint.change_date <= (through_date + timedelta(days=BOCPD_CONFIRMATION_WINDOW_DAYS)).isoformat(),
        )
    )
    cp_by_metric: dict[str, list[str]] = {}
    for cp in cp_result.scalars().all():
        cp_by_metric.setdefault(cp.metric_key, []).append(cp.change_date)

    events: list[AnomalyEvent] = []
    for metric in FORECAST_METRICS:
        if metric not in frame.columns:
            continue
        baseline = baselines.get(metric)
        if baseline is None or baseline.residual_std is None or baseline.residual_std == 0:
            continue
        run.metrics_scanned.append(metric)

        series = frame[metric]

        # First pass: collect every (date, observed, forecast) pair so the
        # z denominator can come from the SAME model that produces the
        # numerator. Dividing the ensemble's forecast error by the STL
        # in-sample residual std was a train/serve mismatch: in-sample STL
        # residuals understate out-of-sample ensemble error, inflating z
        # and the false-positive anomaly rate. Audit P3/D2 (round-2 A49).
        paired: list[tuple[object, float, float]] = []
        for feature_date, observed in series.items():
            if observed is None:
                continue
            try:
                observed_val = float(observed)
            except (TypeError, ValueError):
                continue
            if observed_val != observed_val:  # NaN
                continue
            forecast = best_forecast.get((metric, feature_date))
            if forecast is None or forecast.y_hat is None:
                continue
            paired.append((feature_date, observed_val, forecast.y_hat))

        if not paired:
            continue

        # Robust out-of-sample scale: MAD of the ensemble residuals in the
        # scan window (x1.4826 ~ std under normality). MAD instead of std so
        # the very anomalies being hunted cannot inflate the denominator and
        # mask themselves. Below 10 pairs the estimate is too noisy; fall
        # back to the documented STL-in-sample std (legacy behavior).
        residuals = np.array([obs - y_hat for _, obs, y_hat in paired], dtype=float)
        sigma = baseline.residual_std
        if len(residuals) >= 10:
            mad = float(np.median(np.abs(residuals - np.median(residuals))))
            robust_sigma = 1.4826 * mad
            if robust_sigma > 1e-9:
                sigma = robust_sigma

        for feature_date, observed_val, y_hat in paired:
            residual = observed_val - y_hat
            z = residual / sigma
            if abs(z) < threshold_z:
                continue

            direction = "high" if z > 0 else "low"

            # Two-signal check: BOCPD fired within +/- 2 days of this observation?
            confirmed = False
            for cp_date in cp_by_metric.get(metric, []):
                try:
                    delta = abs(
                        (date.fromisoformat(cp_date) - date.fromisoformat(str(feature_date))).days
                    )
                except ValueError:
                    continue
                if delta <= BOCPD_CONFIRMATION_WINDOW_DAYS:
                    confirmed = True
                    break

            events.append(
                AnomalyEvent(
                    metric_key=metric,
                    observation_date=str(feature_date),
                    observed_value=observed_val,
                    forecasted_value=y_hat,
                    residual=residual,
                    z_score=z,
                    direction=direction,
                    confirmed_by_bocpd=confirmed,
                )
            )

    await _upsert_anomalies(db, user_id, events)
    run.anomalies_written = len(events)
    return run


async def _upsert_anomalies(
    db: "AsyncSession",
    user_id: str,
    events: list[AnomalyEvent],
) -> None:
    from app.core.time import utcnow_naive
    from app.models.ml_baselines import MLAnomaly

    if not events:
        return

    metric_keys = {e.metric_key for e in events}
    min_date = min(e.observation_date for e in events)
    max_date = max(e.observation_date for e in events)

    # Replace any existing rows in this window so we do not double-count.
    await db.execute(
        delete(MLAnomaly).where(
            and_(
                MLAnomaly.user_id == user_id,
                MLAnomaly.metric_key.in_(metric_keys),
                MLAnomaly.observation_date >= min_date,
                MLAnomaly.observation_date <= max_date,
                MLAnomaly.model_version == MODEL_VERSION,
            )
        )
    )

    now = utcnow_naive()
    for e in events:
        db.add(
            MLAnomaly(
                user_id=user_id,
                metric_key=e.metric_key,
                observation_date=e.observation_date,
                observed_value=e.observed_value,
                forecasted_value=e.forecasted_value,
                residual=e.residual,
                z_score=e.z_score,
                direction=e.direction,
                confirmed_by_bocpd=e.confirmed_by_bocpd,
                model_version=MODEL_VERSION,
                detected_at=now,
            )
        )
