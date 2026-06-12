"""Explanations for surfaced insights.

Dispatches by candidate kind:

- ``correlation`` -> the correlation row itself IS the explanation. We
  return the source/target/direction/pearson_r as "contributions" with no
  model fit required.
- ``anomaly`` -> rank the user's recent context features by absolute
  deviation from their 28-day baseline on the observation date. Surfaces
  the behavioral deltas most likely to have driven the anomaly. Full
  XGBoost + SHAP is a Phase 5.1 follow-up; this heuristic is honest for
  the data scale we have.
- ``forecast_warning`` -> same shape as anomaly (not yet wired — Phase 2
  forecasts don't emit warning candidates in v1).
- Other kinds -> generic note directing the user to the insight card
  payload. Phase 9+ fills these in.

The full SHAP path (real XGBoost surrogate + shap.TreeExplainer) is
scoped in the plan at ``~/.claude/plans/golden-floating-creek.md`` but
deliberately left as a follow-up: SHAP pulls heavy deps that we want to
defer until Phase 7 learned-ranker training needs them anyway.

Heavy imports stay lazy per the cold-boot budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class Contribution:
    """One driver of an explanation."""

    feature: str
    contribution: float  # signed; positive = pushed the value UP vs baseline
    observed_value: float | None = None
    baseline_value: float | None = None


@dataclass
class ExplanationResult:
    """Internal shape; api.explain_insight maps this to InsightExplanation."""

    insight_id: str
    user_id: str
    kind: str
    contributions: list[Contribution] = field(default_factory=list)
    historical_examples: list[dict] = field(default_factory=list)


# Context features used when explaining anomalies / forecast warnings. Kept
# deliberately small so the explanation is digestible and the baseline
# lookup stays cheap.
_CONTEXT_FEATURES: tuple[str, ...] = (
    "hrv",
    "resting_hr",
    "sleep_efficiency",
    "sleep_duration_minutes",
    "readiness_score",
    "steps",
    "workout_duration_sum_minutes",
    "training_load_7d",
    "protein_g",
    "calories",
    "dinner_hour",  # Phase 4.5 synth-factory will introduce this
)
_TOP_N_CONTRIBUTIONS = 3


# ─────────────────────────────────────────────────────────────────────────
# Public dispatch
# ─────────────────────────────────────────────────────────────────────────


async def explain(
    db: "AsyncSession",
    user_id: str,
    insight_id: str,
) -> ExplanationResult | None:
    """Return an explanation for the insight, or None if it does not exist."""
    from app.models.ml_insights import MLInsightCandidate

    candidate = await db.get(MLInsightCandidate, insight_id)
    if candidate is None or candidate.user_id != user_id:
        return None

    payload = json.loads(candidate.payload_json) if candidate.payload_json else {}

    if candidate.kind == "correlation":
        return _explain_correlation(candidate.id, user_id, payload)
    if candidate.kind == "anomaly":
        return await _explain_anomaly(db, candidate.id, user_id, payload)
    if candidate.kind == "forecast_warning":
        return await _explain_anomaly(db, candidate.id, user_id, payload)

    # Unknown / future kinds: return an empty explanation. The endpoint
    # still responds 200 so the iOS client does not see a 500.
    return ExplanationResult(
        insight_id=candidate.id,
        user_id=user_id,
        kind=candidate.kind,
    )


# ─────────────────────────────────────────────────────────────────────────
# Correlation path
# ─────────────────────────────────────────────────────────────────────────


def _explain_correlation(
    insight_id: str,
    user_id: str,
    payload: dict,
) -> ExplanationResult:
    """The correlation row IS the explanation. No model fit required.

    Surface the source -> target -> pearson_r as contributions, plus
    anything else from the payload that helps the user understand why
    we're telling them. Literature reference is included when present.
    """
    contributions: list[Contribution] = []

    source = payload.get("source_metric")
    target = payload.get("target_metric")
    pearson = payload.get("pearson_r")
    spearman = payload.get("spearman_r")

    if source and pearson is not None:
        contributions.append(
            Contribution(feature=f"{source} (source)", contribution=float(pearson))
        )
    if target:
        contributions.append(
            Contribution(feature=f"{target} (target)", contribution=float(pearson or 0.0))
        )
    if spearman is not None and pearson is not None:
        contributions.append(
            Contribution(
                feature="dual-method agreement",
                contribution=float(spearman) if pearson * spearman > 0 else 0.0,
            )
        )

    historical: list[dict] = []
    if sample_size := payload.get("sample_size"):
        historical.append(
            {
                "kind": "sample_size",
                "days": int(sample_size),
                "description": f"Based on {sample_size} paired days of your data.",
            }
        )
    if literature_ref := payload.get("literature_ref"):
        historical.append(
            {
                "kind": "literature",
                "doi": literature_ref,
                "description": f"Published research supports this pattern: {literature_ref}.",
            }
        )

    return ExplanationResult(
        insight_id=insight_id,
        user_id=user_id,
        kind="correlation",
        contributions=contributions[:_TOP_N_CONTRIBUTIONS],
        historical_examples=historical,
    )


# ─────────────────────────────────────────────────────────────────────────
# Anomaly path (baseline-delta heuristic)
# ─────────────────────────────────────────────────────────────────────────


async def _explain_anomaly(
    db: "AsyncSession",
    insight_id: str,
    user_id: str,
    payload: dict,
) -> ExplanationResult:
    """Rank context features by absolute deviation from 28-day baseline.

    On the observation date, compute ``(observed - rolling_28d_mean) /
    rolling_28d_std`` for each context feature and return the top-N by
    absolute value. These are the features whose personal-baseline-scaled
    distance from normal is largest on the anomaly day — the most likely
    behavioral drivers.

    Not a true causal attribution, but a useful narrative primitive for
    Phase 5 while real SHAP waits for Phase 7 XGBoost training data.
    """
    from ml.features.store import get_feature_frame

    observation_date = payload.get("observation_date")
    metric_key = payload.get("metric_key")
    if not observation_date:
        return ExplanationResult(
            insight_id=insight_id,
            user_id=user_id,
            kind=payload.get("kind", "anomaly"),
        )

    try:
        obs_dt = date.fromisoformat(str(observation_date))
    except ValueError:
        return ExplanationResult(
            insight_id=insight_id, user_id=user_id, kind="anomaly"
        )

    # 28-day trailing window ending on observation_date.
    start = obs_dt - timedelta(days=27)

    frame = await get_feature_frame(
        db,
        user_id,
        feature_keys=list(_CONTEXT_FEATURES),
        start=start,
        end=obs_dt,
        include_imputed=False,
    )

    # Compute z-score of observation_date against the 28-day window.
    import numpy as np
    import pandas as pd  # noqa: F401 — pulled transitively by get_feature_frame

    obs_row = frame.loc[obs_dt.isoformat()] if obs_dt.isoformat() in frame.index else None
    if obs_row is None:
        return ExplanationResult(
            insight_id=insight_id, user_id=user_id, kind="anomaly"
        )

    contributions: list[Contribution] = []
    for feature in _CONTEXT_FEATURES:
        # Skip the metric that triggered the anomaly itself — not useful
        # to tell the user "your HRV was low because your HRV was low".
        if feature == metric_key:
            continue
        if feature not in frame.columns:
            continue
        column = frame[feature].dropna()
        if column.shape[0] < 7:
            continue
        observed = obs_row[feature]
        if observed is None or (
            isinstance(observed, float) and np.isnan(observed)
        ):
            continue
        baseline_values = column[column.index < obs_dt.isoformat()]
        if baseline_values.shape[0] < 3:
            continue
        mean = float(baseline_values.mean())
        std = float(baseline_values.std(ddof=1)) if baseline_values.shape[0] > 1 else 0.0
        if std == 0:
            continue
        z = (float(observed) - mean) / std
        contributions.append(
            Contribution(
                feature=feature,
                contribution=round(z, 3),
                observed_value=float(observed),
                baseline_value=round(mean, 3),
            )
        )

    # Rank by magnitude, keep top N.
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)
    contributions = contributions[:_TOP_N_CONTRIBUTIONS]

    historical: list[dict] = []
    if observation_date:
        historical.append(
            {
                "kind": "observation_date",
                "date": str(observation_date),
                "description": f"Anomaly observed on {observation_date}.",
            }
        )
    if payload.get("confirmed_by_bocpd"):
        historical.append(
            {
                "kind": "two_signal_confirmation",
                "description": "A change-point detector also fired in the same window, which is our two-signal gate.",
            }
        )

    return ExplanationResult(
        insight_id=insight_id,
        user_id=user_id,
        kind="anomaly",
        contributions=contributions,
        historical_examples=historical,
    )
