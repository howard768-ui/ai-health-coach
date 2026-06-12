"""Public API for the Meld Signal Engine.

This module is the **sole** import target allowed from outside ``backend.ml``.
Anything deeper is a boundary violation and will fail
``backend/tests/ml/test_boundary.py``.

All heavy third-party imports (pandas, scipy, statsmodels, xgboost, prophet,
dowhy, econml, shap, mlflow, evidently, nannyml, coremltools) are lazy-imported
inside function bodies. This keeps FastAPI cold boot under the 4-second budget
on Railway. The cold-boot test (``backend/tests/ml/test_cold_boot.py``) enforces
this by measuring import times of the main app.

**Phase 0 status**: every function below is a stub that raises
``NotImplementedError``. Each phase of the rollout will implement these in turn.
See ``~/.claude/plans/golden-floating-creek.md`` for the full plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only imports. Do NOT move out of this block; doing so would
    # trigger heavy sqlalchemy/pydantic chains at module load on cold boot.
    from sqlalchemy.ext.asyncio import AsyncSession


# ─────────────────────────────────────────────────────────────────────────
# Shadow-mode gate (exposed so scheduler doesn't import ml.config directly)
# ─────────────────────────────────────────────────────────────────────────


def is_shadow_enabled(feature: str) -> bool:
    """Check whether a shadow-mode flag is enabled.

    ``feature`` is the suffix after ``ml_shadow_``: e.g. ``"granger_causal"``
    maps to ``MLSettings.ml_shadow_granger_causal``.
    """
    from ml.config import get_ml_settings

    return getattr(get_ml_settings(), f"ml_shadow_{feature}", False)


# ─────────────────────────────────────────────────────────────────────────
# Public dataclasses (the shapes the rest of the app is allowed to see).
# These intentionally avoid importing pandas / numpy. Any field that would
# need those types is returned as a plain Python primitive or dict.
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ActivePattern:
    """A surfaceable cross-domain pattern for one user.

    Replaces the hardcoded ``HealthConnection`` entries in ``coach_engine.py``
    starting in Phase 5. Loaded from ``UserCorrelation`` at ``developing+`` tier.
    """

    source_metric: str
    target_metric: str
    direction: str  # "positive" | "negative"
    strength: float  # 0-1, abs(r)
    confidence_tier: str  # "developing" | "established" | "causal_candidate" | "literature_supported"
    sample_size: int
    effect_description: str  # natural-language, 4th-grade, no em dashes
    literature_ref: str | None = None


@dataclass
class RecentAnomaly:
    """An anomaly row recent enough to be worth surfacing to the coach prompt.

    Phase 5 pulls from ``ml_anomalies`` where ``observation_date`` is within
    the last 7 days AND ``confirmed_by_bocpd`` is true (two-signal gate —
    see plan "Forecasting and Anomaly Detection" section).
    """

    metric_key: str
    observation_date: str  # YYYY-MM-DD
    direction: str  # "high" | "low"
    z_score: float
    observed_value: float | None
    forecasted_value: float | None


@dataclass
class PersonalForecast:
    """Short-horizon forecast for today + tomorrow, rendered into the coach prompt.

    Not a full Forecast (that is returned by ``forecast_metric``); this is a
    compact per-metric summary built for prompt inclusion. Populated only for
    the five headline metrics the coach is likely to be asked about.
    """

    metric_key: str
    target_date: str  # YYYY-MM-DD
    y_hat: float | None
    y_hat_low: float | None
    y_hat_high: float | None


@dataclass
class SignalContext:
    """Everything the coach prompt needs from the Signal Engine for one query.

    Built by ``load_coach_signal_context``. The coach router pre-loads this in
    the async request path, then passes it into the synchronous
    ``CoachEngine.process_query``. Keeping ``process_query`` sync matters
    because the Anthropic SDK is sync and the existing call site uses
    ``asyncio.to_thread``.

    Notification callers (``notification_engine``, ``notification_content``)
    pass ``None`` — notifications do not need active patterns.
    """

    active_patterns: list[ActivePattern] = field(default_factory=list)
    recent_anomalies: list[RecentAnomaly] = field(default_factory=list)
    personal_forecasts: list[PersonalForecast] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            not self.active_patterns
            and not self.recent_anomalies
            and not self.personal_forecasts
        )


@dataclass
class DiscoveryReport:
    """Summary of a single ``run_discovery_pipeline`` invocation."""

    user_id: str
    run_started_at: str  # ISO-8601 UTC
    run_finished_at: str
    layers_run: list[str] = field(default_factory=list)
    patterns_tested: int = 0
    patterns_surfaced: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    shadow_mode: bool = True


@dataclass
class InsightCandidate:
    """Everything the ranker sees.

    Populated by ``ml.ranking.candidates.generate_candidates``. This is the
    **public** shape — the internal builder uses a tuple for
    ``subject_metrics`` for immutability, but at the boundary we normalize
    to a plain list so JSON serialization is clean.
    """

    id: str
    user_id: str
    kind: str  # correlation | anomaly | forecast_warning | experiment_result | streak | regression
    subject_metrics: list[str]
    effect_size: float
    confidence: float
    novelty: float
    recency_days: int
    actionability_score: float
    literature_support: bool = False
    directional_support: bool = False
    causal_support: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedInsight:
    """A candidate after the ranker has scored it."""

    candidate: InsightCandidate
    score: float
    rank: int
    ranker_model_version: str


@dataclass
class DailyInsightReport:
    """Phase 4 summary returned by ``run_daily_insights``."""

    user_id: str
    surface_date: str  # YYYY-MM-DD
    candidates_generated: int
    rankings_written: int
    top_candidate_id: str | None  # None when no candidates exist today
    shadow_mode: bool


@dataclass
class InsightExplanation:
    """Response payload for ``/api/coach/explain-finding``.

    ``shap_values`` is a plain list of (feature_name, value) tuples so the
    router can serialize it without pulling pandas/numpy into its import chain.
    """

    insight_id: str
    user_id: str
    explanation_kind: str  # "correlation" | "anomaly" | "forecast_warning"
    top_contributing_features: list[tuple[str, float]] = field(default_factory=list)
    historical_examples: list[dict[str, Any]] = field(default_factory=list)
    shap_values: list[tuple[str, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "user_id": self.user_id,
            "explanation_kind": self.explanation_kind,
            "top_contributing_features": [
                {"feature": name, "value": val}
                for name, val in self.top_contributing_features
            ],
            "historical_examples": self.historical_examples,
            "shap_values": (
                [{"feature": n, "value": v} for n, v in self.shap_values]
                if self.shap_values is not None
                else None
            ),
        }


@dataclass
class Forecast:
    """Short-horizon forecast for a single metric."""

    user_id: str
    metric: str
    horizon_days: int
    points: list[dict[str, Any]] = field(default_factory=list)  # [{date, y_hat, y_hat_low, y_hat_high}]
    model_version: str = ""
    made_on: str = ""  # ISO-8601


@dataclass
class RankerMetadata:
    """Returned by ``ranker_model_metadata()`` for the iOS CoreML sync endpoint."""

    model_version: str
    file_hash: str  # SHA256 of the .mlpackage for conditional GET
    file_size_bytes: int
    download_url: str  # R2 signed URL, valid for TTL
    expires_at: str  # ISO-8601
    min_ios_version: str = "17.0"


@dataclass
class UserContext:
    """Minimal signal passed into ``rank_candidates`` for the server-side path."""

    user_id: str
    current_readiness_bucket: str = "mid"  # "low" | "mid" | "high"
    user_engagement_7d: int = 0
    thumbs_up_rate_30d: float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Public async interface. Every entry point below is a Phase 0 stub.
# ─────────────────────────────────────────────────────────────────────────


async def refresh_features_for_user(
    db: "AsyncSession",
    user_id: str,
    through_date: date,
    lookback_days: int = 30,
) -> int:
    """Materialize the feature frame for a user through ``through_date``.

    Idempotent: rerunning on the same day with the same builder versions
    recomputes identically. The underlying table enforces
    ``UNIQUE (user_id, feature_key, feature_date, feature_version)``.

    Does NOT commit. The caller owns the transaction (typically the
    scheduler job wraps the call in ``async with db.begin()``).

    Returns the number of rows written to ``ml_feature_values``.
    """
    from datetime import timedelta

    # Lazy import keeps cold-boot under the Railway budget. See
    # ``backend/ml/__init__.py`` for the full rationale.
    from ml.features.store import materialize_for_user

    start = through_date - timedelta(days=lookback_days - 1)
    result = await materialize_for_user(db, user_id, start, through_date)
    return result.rows_written


async def run_discovery_pipeline(
    db: "AsyncSession",
    user_id: str,
) -> DiscoveryReport:
    """Run the full L1 -> L5 discovery pipeline for one user.

    Phase 2 implements L1 only (baselines + change points + forecasts + anomaly
    detection). Phase 3 adds L2 associations, Phase 6 adds L3 Granger + L4
    DoWhy, Phase 9 adds L5 APTE.

    Does NOT commit — caller owns the transaction.
    """
    from datetime import date, datetime, timezone

    # Lazy imports per boundary rules + cold-boot budget.
    from ml.discovery.baselines import compute_baselines_for_user
    from ml.forecasting.anomaly import detect_anomalies_for_user
    from ml.forecasting.residuals import compute_forecasts_for_user

    started = datetime.now(timezone.utc).isoformat()
    through_date = date.today()
    layers_run: list[str] = []
    tier_counts: dict[str, int] = {}

    # L1 baselines + change points.
    baseline_run = await compute_baselines_for_user(db, user_id, through_date)
    layers_run.append("baselines")
    tier_counts["baselines_written"] = baseline_run.baselines_written
    tier_counts["change_points_written"] = baseline_run.change_points_written

    # Forecasts for the next 7 days.
    forecasts = await compute_forecasts_for_user(db, user_id, made_on=through_date)
    layers_run.append("forecasts")
    tier_counts["forecast_metrics"] = len(forecasts)

    # Residual anomaly detection over the last week.
    anomaly_run = await detect_anomalies_for_user(db, user_id, through_date)
    layers_run.append("anomalies")
    tier_counts["anomalies_written"] = anomaly_run.anomalies_written

    return DiscoveryReport(
        user_id=user_id,
        run_started_at=started,
        run_finished_at=datetime.now(timezone.utc).isoformat(),
        layers_run=layers_run,
        patterns_tested=baseline_run.baselines_written,
        patterns_surfaced=0,  # Phase 2 is shadow-only; nothing is surfaced.
        tier_counts=tier_counts,
        shadow_mode=True,
    )


async def generate_insight_candidates(
    db: "AsyncSession",
    user_id: str,
) -> list[InsightCandidate]:
    """Normalize all surfaceable findings into ``InsightCandidate`` objects.

    Reads from ``UserCorrelation`` (developing+ tier), ``ml_anomalies``
    within the last 7 days, and (future) ``ml_n_of_1_results``. Persists
    to ``ml_insight_candidates`` (idempotent upsert).

    Does NOT commit — caller owns the transaction.
    """
    from datetime import date as _date

    # Lazy import keeps cold boot clean. See backend/ml/__init__.py.
    from ml.ranking.candidates import generate_candidates

    internal = await generate_candidates(db, user_id, _date.today())
    # Map internal dataclass (tuple subject_metrics, ``payload`` field) to
    # public shape (list subject_metrics, same-named ``payload``).
    return [
        InsightCandidate(
            id=c.id,
            user_id=c.user_id,
            kind=c.kind,
            subject_metrics=list(c.subject_metrics),
            effect_size=c.effect_size,
            confidence=c.confidence,
            novelty=c.novelty,
            recency_days=c.recency_days,
            actionability_score=c.actionability_score,
            literature_support=c.literature_support,
            directional_support=c.directional_support,
            causal_support=c.causal_support,
            payload=dict(c.payload),
        )
        for c in internal
    ]


async def rank_candidates(
    candidates: list[InsightCandidate],
    user_context: "UserContext | None" = None,
) -> list[RankedInsight]:
    """Score + order candidates. Heuristic ranker in Phase 4, learned in Phase 7.

    In Phase 4 this is a pure weighted-sum scorer: the ``user_context``
    parameter is reserved for Phase 7 when the learned ranker consumes
    user engagement features. Present here to keep the signature stable.

    This is the **server-side** rank path (used by ``run_daily_insights``
    and the notification picker). Phase 7 adds an on-device CoreML copy
    for dashboard-side ranking.
    """
    # Lazy import keeps cold boot clean.
    from ml.ranking.candidates import InsightCandidate as InternalCandidate
    from ml.ranking.heuristic import RANKER_VERSION, rank_candidates as _rank

    internal = [
        InternalCandidate(
            id=c.id,
            user_id=c.user_id,
            kind=c.kind,
            subject_metrics=tuple(c.subject_metrics),
            effect_size=c.effect_size,
            confidence=c.confidence,
            novelty=c.novelty,
            recency_days=c.recency_days,
            actionability_score=c.actionability_score,
            literature_support=c.literature_support,
            directional_support=c.directional_support,
            causal_support=c.causal_support,
            payload=dict(c.payload),
        )
        for c in candidates
    ]
    ranked = _rank(internal)
    return [
        RankedInsight(
            candidate=candidates[
                next(i for i, c in enumerate(candidates) if c.id == r.candidate.id)
            ],
            score=r.score,
            rank=r.rank,
            ranker_model_version=RANKER_VERSION,
        )
        for r in ranked
    ]


def is_insight_card_shadow_mode() -> bool:
    """Whether the Phase 4 ``SignalInsightCard`` surface is in shadow mode.

    Exposed through the public boundary so the router does not need to
    import ``ml.config`` directly (that would trip the boundary AST check
    in ``tests/ml/test_boundary.py``).
    """
    from ml.config import get_ml_settings

    return get_ml_settings().ml_shadow_insight_card


async def can_surface_insight_today(
    db: "AsyncSession",
    user_id: str,
    surface_date: "date | None" = None,
) -> tuple[bool, str]:
    """Cap check for today's insight card.

    Returns ``(allowed, reason)``. Thin wrapper around
    ``ml.ranking.heuristic.can_surface_today`` so the router stays behind
    the public boundary.
    """
    from datetime import date as _date

    from ml.ranking.heuristic import can_surface_today

    resolved = surface_date or _date.today()
    return await can_surface_today(db, user_id, resolved)


async def run_daily_insights(
    db: "AsyncSession",
    user_id: str,
) -> DailyInsightReport:
    """Orchestrate Phase 4: generate candidates, rank, persist rankings.

    Called by the ``insight_candidate_job`` scheduler at ~07:00 local. The
    API endpoint at ``GET /api/insights/daily`` reads from the persisted
    ``ml_rankings`` rows; this job is what populates them.

    Idempotent: if a slate already exists for today,
    ``materialize_daily_ranking`` returns empty and we read the existing
    rank=1 row so ``top_candidate_id`` still reflects today's top card.
    Does NOT commit — caller owns the transaction.
    """
    from datetime import date as _date

    from sqlalchemy import select as _select

    from app.models.ml_insights import MLRanking
    from ml.ranking.candidates import generate_candidates
    from ml.ranking.heuristic import materialize_daily_ranking, RANKER_VERSION

    # Late import of the settings so env changes pick up without restart.
    from ml.config import get_ml_settings

    surface_date = _date.today()
    candidates = await generate_candidates(db, user_id, surface_date)
    rankings = await materialize_daily_ranking(
        db, user_id, candidates, surface_date
    )

    # Read-through: when materialize no-ops (rerun same day), look up
    # today's persisted top-1 so the report still answers "what IS the
    # top candidate for today?".
    top_candidate_id: str | None = None
    if rankings:
        top_candidate_id = rankings[0].candidate.id
    else:
        existing = await db.execute(
            _select(MLRanking).where(
                MLRanking.user_id == user_id,
                MLRanking.surface_date == surface_date.isoformat(),
                MLRanking.rank == 1,
                MLRanking.ranker_version == RANKER_VERSION,
            )
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            top_candidate_id = row.candidate_id

    return DailyInsightReport(
        user_id=user_id,
        surface_date=surface_date.isoformat(),
        candidates_generated=len(candidates),
        rankings_written=len(rankings),
        top_candidate_id=top_candidate_id,
        shadow_mode=get_ml_settings().ml_shadow_insight_card,
    )


async def explain_insight(
    db: "AsyncSession",
    user_id: str,
    insight_id: str,
) -> InsightExplanation:
    """Build a SHAP-backed explanation for a given insight.

    Called by ``/api/coach/explain-finding`` in Phase 5. Dispatches by
    candidate kind via ``ml.narrate.shap_explainer.explain``. Correlation
    candidates return a synthesized attribution (the r value IS the
    explanation); anomaly + forecast_warning candidates run SHAP on a
    lightweight XGBoost surrogate over the user's last 90 days.

    Returns an ``InsightExplanation`` with the top-3 contributions.
    Raises ``LookupError`` when the insight id is unknown or does not
    belong to the caller.
    """
    # Lazy import keeps cold boot clean.
    from ml.narrate.shap_explainer import explain as _explain

    internal = await _explain(db, user_id, insight_id)
    if internal is None:
        raise LookupError(f"insight {insight_id} not found for user {user_id}")

    return InsightExplanation(
        insight_id=internal.insight_id,
        user_id=internal.user_id,
        explanation_kind=internal.kind,
        top_contributing_features=[
            (c.feature, float(c.contribution)) for c in internal.contributions
        ],
        historical_examples=list(internal.historical_examples),
        shap_values=[
            (c.feature, float(c.contribution)) for c in internal.contributions
        ],
    )


@dataclass
class InsightNarration:
    """Narration result + flag for whether the fallback template was used."""

    insight_id: str
    kind: str
    text: str
    used_fallback: bool
    fallback_reason: str | None = None


async def narrate_insight(
    db: "AsyncSession",
    user_id: str,
    insight_id: str,
) -> InsightNarration:
    """Run Opus narration on a single insight candidate.

    Thin wrapper so the coach router does not need to import
    ``ml.narrate.translator`` directly (boundary rule). Emits the
    post-voice-compliance text; on failure returns the templated fallback.
    """
    from app.models.ml_insights import MLInsightCandidate
    from ml.narrate.translator import NarrationRequest, generate_narration
    import json as _json

    candidate = await db.get(MLInsightCandidate, insight_id)
    if candidate is None or candidate.user_id != user_id:
        raise LookupError(f"insight {insight_id} not found for user {user_id}")

    payload = _json.loads(candidate.payload_json) if candidate.payload_json else {}
    subjects = _json.loads(candidate.subject_metrics_json)

    result = await generate_narration(
        NarrationRequest(
            kind=candidate.kind,
            subject_metrics=list(subjects),
            payload=payload,
        )
    )
    return InsightNarration(
        insight_id=insight_id,
        kind=candidate.kind,
        text=result.text,
        used_fallback=result.used_fallback,
        fallback_reason=result.fallback_reason,
    )


async def forecast_metric(
    db: "AsyncSession",
    user_id: str,
    metric: str,
    horizon_days: int = 7,
) -> Forecast:
    """Return the most recent ensembled forecast for ``metric``.

    Reads from ``ml_forecasts`` populated by the nightly ``baselines_job``.
    Returns a Forecast with empty ``points`` if no forecast has been
    computed yet. On-demand computation is intentionally not supported:
    Prophet is too expensive to run synchronously in the request path.
    """
    from datetime import date as _date
    from datetime import timedelta

    from sqlalchemy import select

    # Lazy import to keep api.py light.
    from app.models.ml_baselines import MLForecast

    today = _date.today()
    start = today + timedelta(days=1)
    end = today + timedelta(days=horizon_days)

    result = await db.execute(
        select(MLForecast)
        .where(
            MLForecast.user_id == user_id,
            MLForecast.metric_key == metric,
            MLForecast.target_date >= start.isoformat(),
            MLForecast.target_date <= end.isoformat(),
        )
        .order_by(MLForecast.target_date.asc(), MLForecast.made_on.desc())
    )
    rows = list(result.scalars().all())

    # Keep the most recent made_on per target_date.
    seen_targets: set[str] = set()
    points: list[dict[str, object]] = []
    made_on_latest: str | None = None
    model_version: str | None = None
    for row in rows:
        if row.target_date in seen_targets:
            continue
        seen_targets.add(row.target_date)
        points.append(
            {
                "date": row.target_date,
                "y_hat": row.y_hat,
                "y_hat_low": row.y_hat_low,
                "y_hat_high": row.y_hat_high,
            }
        )
        if made_on_latest is None or row.made_on > made_on_latest:
            made_on_latest = row.made_on
        model_version = row.model_version

    return Forecast(
        user_id=user_id,
        metric=metric,
        horizon_days=horizon_days,
        points=points,
        model_version=model_version or "",
        made_on=made_on_latest or "",
    )


async def load_active_patterns(
    db: "AsyncSession",
    user_id: str,
    limit: int = 5,
) -> list[ActivePattern]:
    """Top-N developing+ patterns for this user, sorted by strength x confidence.

    Replaces the hardcoded ``KnowledgeGraph`` deleted from ``coach_engine.py``
    in Phase 5. Reads ``UserCorrelation`` rows at tiers developing, established,
    causal_candidate, or literature_supported. Returns a plain list of
    ``ActivePattern`` the coach prompt template can consume.
    """
    from sqlalchemy import select as _select

    from app.models.correlation import UserCorrelation

    result = await db.execute(
        _select(UserCorrelation).where(
            UserCorrelation.user_id == user_id,
            UserCorrelation.confidence_tier.in_(
                (
                    "developing",
                    "established",
                    "causal_candidate",
                    "literature_supported",
                )
            ),
        )
    )
    rows = list(result.scalars().all())
    if not rows:
        return []

    # Rank by strength * confidence-tier-weight. Same weights as Phase 4
    # candidate confidence so the coach and the insight card see consistent
    # prioritization.
    tier_weight = {
        "developing": 0.60,
        "established": 0.80,
        "causal_candidate": 0.90,
        "literature_supported": 0.95,
    }
    rows.sort(
        key=lambda r: (r.strength or 0) * tier_weight.get(r.confidence_tier, 0.3),
        reverse=True,
    )

    return [
        ActivePattern(
            source_metric=r.source_metric,
            target_metric=r.target_metric,
            direction=r.direction,
            strength=float(r.strength or 0.0),
            confidence_tier=r.confidence_tier or "emerging",
            sample_size=int(r.sample_size or 0),
            effect_description=r.effect_size_description or "",
            literature_ref=r.literature_ref,
        )
        for r in rows[:limit]
    ]


async def load_recent_anomalies(
    db: "AsyncSession",
    user_id: str,
    lookback_days: int = 7,
    confirmed_only: bool = True,
) -> list[RecentAnomaly]:
    """Recent ml_anomalies rows for coach prompt inclusion.

    ``confirmed_only=True`` (default) applies the plan's two-signal gate:
    only surface anomalies that BOCPD confirmed within the 48h window. An
    unconfirmed z-score spike stays in the shadow log but does not steer
    the coach's conversation.
    """
    from datetime import date as _date, timedelta

    from sqlalchemy import select as _select

    from app.models.ml_baselines import MLAnomaly

    start = (_date.today() - timedelta(days=lookback_days - 1)).isoformat()
    query = _select(MLAnomaly).where(
        MLAnomaly.user_id == user_id,
        MLAnomaly.observation_date >= start,
    )
    if confirmed_only:
        query = query.where(MLAnomaly.confirmed_by_bocpd.is_(True))

    result = await db.execute(query)
    rows = list(result.scalars().all())
    return [
        RecentAnomaly(
            metric_key=r.metric_key,
            observation_date=r.observation_date,
            direction=r.direction,
            z_score=float(r.z_score),
            observed_value=r.observed_value,
            forecasted_value=r.forecasted_value,
        )
        for r in rows
    ]


async def load_personal_forecasts(
    db: "AsyncSession",
    user_id: str,
    metrics: list[str] | None = None,
    horizon_days: int = 2,
) -> list[PersonalForecast]:
    """Today + tomorrow forecasts for the coach prompt.

    Reads the most recent forecast per (user, metric, target_date). Short
    horizon keeps the prompt tight. Default metrics are the five headline
    biometrics the coach is most often asked about.
    """
    from datetime import date as _date, timedelta

    from sqlalchemy import select as _select

    from app.models.ml_baselines import MLForecast

    default_metrics = ("hrv", "resting_hr", "sleep_efficiency", "readiness_score", "steps")
    target_metrics = metrics if metrics is not None else list(default_metrics)

    today = _date.today()
    end = today + timedelta(days=horizon_days - 1)
    result = await db.execute(
        _select(MLForecast)
        .where(
            MLForecast.user_id == user_id,
            MLForecast.metric_key.in_(target_metrics),
            MLForecast.target_date >= today.isoformat(),
            MLForecast.target_date <= end.isoformat(),
        )
        .order_by(MLForecast.target_date.asc(), MLForecast.made_on.desc())
    )
    rows = list(result.scalars().all())

    # Keep the most recent made_on per (metric, target_date).
    seen: set[tuple[str, str]] = set()
    out: list[PersonalForecast] = []
    for row in rows:
        key = (row.metric_key, row.target_date)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            PersonalForecast(
                metric_key=row.metric_key,
                target_date=row.target_date,
                y_hat=row.y_hat,
                y_hat_low=row.y_hat_low,
                y_hat_high=row.y_hat_high,
            )
        )
    return out


async def load_coach_signal_context(
    db: "AsyncSession",
    user_id: str,
) -> SignalContext:
    """One call that assembles the full Signal Engine context for the coach.

    Used by ``backend/app/routers/coach.py`` in the async request path.
    All three loaders are cheap (small indexed queries against indexed ml_
    tables); sequential is fine. If any single loader fails, returns an
    empty SignalContext so the coach still responds with health_data only.
    """
    import asyncio
    import logging as _logging

    _log = _logging.getLogger("meld.ml.api")

    try:
        patterns = await load_active_patterns(db, user_id)
    except Exception as e:  # noqa: BLE001 — defensive at the public boundary
        _log.warning("load_active_patterns failed for %s: %s", user_id, e)
        patterns = []

    try:
        anomalies = await load_recent_anomalies(db, user_id)
    except Exception as e:  # noqa: BLE001
        _log.warning("load_recent_anomalies failed for %s: %s", user_id, e)
        anomalies = []

    try:
        forecasts = await load_personal_forecasts(db, user_id)
    except Exception as e:  # noqa: BLE001
        _log.warning("load_personal_forecasts failed for %s: %s", user_id, e)
        forecasts = []

    # Suppress unused-import warning for asyncio — retained for potential
    # future parallelization via asyncio.gather if latency ever matters.
    _ = asyncio

    return SignalContext(
        active_patterns=patterns,
        recent_anomalies=anomalies,
        personal_forecasts=forecasts,
    )


async def ranker_model_metadata(db: "AsyncSession") -> RankerMetadata | None:
    """Current CoreML ranker metadata for the iOS sync endpoint.

    iOS uses the returned hash to decide whether to re-download the
    ``.mlmodel`` on wifi. Returns None if no active model exists.
    """
    from sqlalchemy import select
    from app.models.ml_models import MLModel

    result = await db.execute(
        select(MLModel).where(
            MLModel.model_type == "ranker",
            MLModel.is_active.is_(True),
        )
    )
    model = result.scalar_one_or_none()
    if model is None:
        return None

    return RankerMetadata(
        model_version=model.model_version,
        file_hash=model.file_hash or "",
        file_size_bytes=model.file_size_bytes or 0,
        download_url=model.download_url or "",
        expires_at="",
        min_ios_version="17.0",
    )


async def train_and_export_ranker(
    db: "AsyncSession",
    coldstart_threshold: int = 20,
) -> dict:
    """Full Phase 7 pipeline: train XGBoost, export CoreML, upload R2, register.

    Returns a summary dict for logging/telemetry. Does NOT commit.
    """
    from ml.ranking.trainer import train_ranker_pipeline
    from ml.ranking.coreml_export import (
        export_to_coreml,
        upload_to_r2,
        register_model,
    )

    summary: dict = {"trained": False}

    trained = await train_ranker_pipeline(db, coldstart_threshold=coldstart_threshold)
    if trained is None:
        summary["error"] = "Not enough data for training"
        return summary

    summary["trained"] = True
    summary["model_version"] = trained.model_version
    summary["train_samples"] = trained.train_samples
    summary["val_ndcg"] = trained.val_ndcg
    summary["feature_importances"] = trained.feature_importances

    # CoreML export (graceful if coremltools not installed).
    export = export_to_coreml(
        trained.model,
        trained.feature_names,
        model_version=trained.model_version,
    )
    summary["coreml_exported"] = export.success
    if export.error:
        summary["coreml_error"] = export.error

    # R2 upload (graceful if credentials not configured).
    r2_key = None
    download_url = None
    if export.success and export.mlmodel_path:
        r2_result = upload_to_r2(
            export.mlmodel_path,
            r2_key=f"coreml/{trained.model_version}.mlmodel",
        )
        summary["r2_uploaded"] = r2_result.success
        if r2_result.success:
            r2_key = r2_result.r2_key
            download_url = r2_result.download_url
        if r2_result.error:
            summary["r2_error"] = r2_result.error

    # Register in DB.
    model_id = await register_model(
        db,
        model_version=trained.model_version,
        feature_names=trained.feature_names,
        hyperparams=trained.hyperparams,
        train_samples=trained.train_samples,
        val_ndcg=trained.val_ndcg,
        file_hash=export.file_hash,
        file_size_bytes=export.file_size_bytes,
        r2_key=r2_key,
        download_url=download_url,
    )
    summary["model_id"] = model_id

    return summary


@dataclass
class AssociationsReport:
    """Phase 3 L2 output summary. Shape mirrors the internal report from
    ``ml.discovery.associations`` so callers (scheduler, tests) see a stable
    shape across the public boundary.
    """

    user_id: str
    window_days: int
    pairs_tested: int
    pairs_with_enough_data: int
    significant_results: int
    dynamic_pairs_generated: int
    rows_written: int


async def run_associations(
    db: "AsyncSession",
    user_id: str,
    window_days: int = 30,
) -> AssociationsReport:
    """Run L2 associations: dual-method correlations + BH-FDR + persistence.

    Replaces the direct call to
    ``backend/app/services/correlation_engine.compute_correlations`` in the
    scheduler. Reads the Phase 1 feature store (features must be fresh;
    scheduler orders ``feature_refresh_job`` at 03:30 UTC before
    ``correlation_engine_job``).

    Writes to ``UserCorrelation`` with legacy-compatible ``source_metric``
    and ``target_metric`` strings so downstream code that reads those fields
    keeps working without a rename migration.

    Does NOT commit — caller owns the transaction.
    """
    # Lazy import per boundary rules + cold-boot budget.
    from ml.discovery.associations import run_associations_for_user

    report = await run_associations_for_user(db, user_id, window_days=window_days)
    return AssociationsReport(
        user_id=report.user_id,
        window_days=report.window_days,
        pairs_tested=report.pairs_tested,
        pairs_with_enough_data=report.pairs_with_enough_data,
        significant_results=report.significant_results,
        dynamic_pairs_generated=report.dynamic_pairs_generated,
        rows_written=report.rows_written,
    )


@dataclass
class GrangerReport:
    """Phase 6 L3 output summary. Shape mirrors the internal report from
    ``ml.discovery.granger`` so callers (scheduler, tests) see a stable
    shape across the public boundary.
    """

    user_id: str
    pairs_tested: int
    pairs_stationary: int
    pairs_significant: int
    pairs_skipped_non_stationary: int
    pairs_skipped_insufficient_data: int
    rows_written: int
    correlations_updated: int


@dataclass
class CausalReport:
    """Phase 6 L4 output summary. Shape mirrors the internal report from
    ``ml.discovery.causal``.
    """

    user_id: str
    pairs_tested: int
    pairs_passed: int
    pairs_skipped_insufficient_data: int
    pairs_estimation_failed: int
    rows_written: int
    correlations_updated: int


async def run_granger(
    db: "AsyncSession",
    user_id: str,
    window_days: int = 90,
) -> GrangerReport:
    """Run L3 Granger causality on developing+ pairs.

    Sets ``directional_support=True`` on UserCorrelation rows that pass
    the Granger F-test at p < 0.05 with a stationarity gate.

    Does NOT commit. Caller owns the transaction.
    """
    from ml.discovery.granger import run_granger_for_user

    report = await run_granger_for_user(db, user_id, window_days=window_days)
    return GrangerReport(
        user_id=report.user_id,
        pairs_tested=report.pairs_tested,
        pairs_stationary=report.pairs_stationary,
        pairs_significant=report.pairs_significant,
        pairs_skipped_non_stationary=report.pairs_skipped_non_stationary,
        pairs_skipped_insufficient_data=report.pairs_skipped_insufficient_data,
        rows_written=report.rows_written,
        correlations_updated=report.correlations_updated,
    )


async def run_causal(
    db: "AsyncSession",
    user_id: str,
    window_days: int = 90,
    max_pairs: int = 10,
) -> CausalReport:
    """Run L4 quasi-causal estimation on directional-supported or literature-matched pairs.

    Uses DoWhy CausalModel with econml DML estimator and three refutation
    tests. Promotes passing pairs to ``causal_candidate`` confidence tier.

    Does NOT commit. Caller owns the transaction.
    """
    from ml.discovery.causal import run_causal_for_user

    report = await run_causal_for_user(
        db, user_id, window_days=window_days, max_pairs=max_pairs
    )
    return CausalReport(
        user_id=report.user_id,
        pairs_tested=report.pairs_tested,
        pairs_passed=report.pairs_passed,
        pairs_skipped_insufficient_data=report.pairs_skipped_insufficient_data,
        pairs_estimation_failed=report.pairs_estimation_failed,
        rows_written=report.rows_written,
        correlations_updated=report.correlations_updated,
    )


@dataclass
class CohortManifest:
    """Phase 4.5 output summary. Returned by ``generate_synth_cohort``.

    Describes one synthesis run: which users were created, over what date
    range, with what generator. Persisted to ``ml_synth_runs`` for audit
    and re-run reproducibility.
    """

    run_id: str  # uuid4
    seed: int | None
    generator: str  # "parametric" | "gan"
    n_users: int
    user_ids: list[str]
    days: int
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    created_at: str  # ISO-8601 UTC
    adversarial_fraction: float  # share of conversation personas that are adversarial


async def generate_synth_cohort(
    db: "AsyncSession",
    n_users: int,
    days: int | None = None,
    seed: int | None = None,
    generator: str | None = None,
) -> CohortManifest:
    """Generate N synthetic users with ``days`` of history each.

    Phase 4.5 infrastructure for ML test coverage. Writes to raw tables only
    (HealthMetricRecord, SleepRecord, ActivityRecord, MealRecord,
    FoodItemRecord). The existing nightly ``feature_refresh_job`` materializes
    features from those rows; synth does NOT write ``ml_feature_values``
    directly. Every row is tagged ``is_synthetic=True`` so production
    aggregates and crisis eval buckets filter unconditionally.

    Persists a ``CohortManifest`` to ``ml_synth_runs`` for audit trail.
    Does NOT commit. Caller owns the transaction.

    Args:
        n_users: number of synthetic users to generate.
        days: cohort length in days. Defaults to ``synth_default_days`` from
            ``MLSettings`` (120 days, covers L1 >=28, Prophet >=90, L3
            Granger >=120).
        seed: optional integer for reproducibility. ``None`` means a random
            manifest each run.
        generator: ``"parametric"`` (scipy + numpy, always available) or
            ``"gan"`` (DoppelGANger, gated behind the ``meld-backend[synth-gan]``
            extras install). Defaults to ``synth_default_generator``.
    """
    # Lazy import per boundary rules + cold-boot budget. ml.synth.factory
    # pulls numpy and ml.synth.demographics/wearables at module load,
    # none of which should touch the FastAPI request path.
    from ml.synth.factory import generate_cohort as _generate_cohort

    return await _generate_cohort(db, n_users, days, seed, generator)


@dataclass
class DriftReportSummary:
    """Phase 4.5 Commit 5 output. Summary of one drift-monitoring run.

    ``html_path`` is ``None`` when either (a) the dataset was too small
    to run a meaningful KS test on either partition, or (b) Evidently
    failed to render the HTML (import failure on Python 3.14, or a
    runtime exception during rendering). The KS-based ``p_values`` and
    ``drifted_metrics`` populate in every other case.

    ``ks_statistics`` and ``sample_sizes`` are the D statistic and the
    post-NaN-drop sample sizes per metric, carried here so
    ``persist_drift_results`` can write ``ml_drift_results`` rows
    without recomputing the KS test.
    """

    run_id: str
    created_at: str  # ISO-8601 UTC
    html_path: str | None
    html_backend: str  # "evidently" | "none"
    n_reference_rows: int
    n_current_rows: int
    metrics_tested: list[str] = field(default_factory=list)
    drifted_metrics: list[str] = field(default_factory=list)
    p_values: dict[str, float] = field(default_factory=dict)
    ks_statistics: dict[str, float] = field(default_factory=dict)
    sample_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    dataset_too_small: bool = False
    threshold: float = 0.05


# KS-statistic cutoff for the ``drifted`` flag persisted in
# ``ml_drift_results``. Specified by the Phase 4.5 plan: a feature is
# considered drifted when its two-sample KS D statistic strictly
# exceeds this value. This is a separate, more durable rule than the
# p-value-based ``drifted_metrics`` returned by the inline KS test
# (which is a standard alpha 0.05). Both are surfaced so the scheduler
# can log both and the endpoint can filter on the durable one.
DRIFT_KS_THRESHOLD: float = 0.15


async def build_synth_drift_report(
    db: "AsyncSession",
    output_dir: "str | None" = None,
    run_id: str | None = None,
    threshold: float = 0.05,
) -> DriftReportSummary:
    """Compare synth biometrics against real biometrics and return a summary.

    Reads canonical rows from ``HealthMetricRecord`` partitioned by the
    ``is_synthetic`` column added in Phase 4.5 Commit 3. Per-metric
    drift is decided by a two-sample Kolmogorov-Smirnov test; when
    Evidently is importable (Railway runs Python 3.12; Brock's local
    dev is 3.14 where Evidently's pydantic-v1 shim crashes), the call
    also writes a polished HTML report to ``output_dir`` or
    ``/tmp/evidently/``. When Evidently is unavailable, the KS summary
    still returns cleanly and ``html_path`` is ``None``.

    Does NOT commit. The caller (typically a scheduler job, or a
    dashboard endpoint) owns the transaction.
    """
    from pathlib import Path as _Path

    from ml.mlops.evidently_reports import build_drift_report as _build_drift_report

    resolved_output_dir = _Path(output_dir) if output_dir else None
    report = await _build_drift_report(
        db,
        output_dir=resolved_output_dir,
        run_id=run_id,
        threshold=threshold,
    )
    return DriftReportSummary(
        run_id=report.run_id,
        created_at=report.created_at,
        html_path=report.html_path,
        html_backend=report.html_backend,
        n_reference_rows=report.n_reference_rows,
        n_current_rows=report.n_current_rows,
        metrics_tested=list(report.metrics_tested),
        drifted_metrics=list(report.drifted_metrics),
        p_values=dict(report.p_values),
        ks_statistics=dict(report.ks_statistics),
        sample_sizes=dict(report.sample_sizes),
        dataset_too_small=report.dataset_too_small,
        threshold=report.threshold,
    )


async def persist_drift_results(
    db: "AsyncSession",
    summary: DriftReportSummary,
    ks_threshold: float = DRIFT_KS_THRESHOLD,
) -> int:
    """Write one ``ml_drift_results`` row per metric in ``summary``.

    The scheduler's ``synth_drift_job`` calls this after
    ``build_synth_drift_report``. Each metric in
    ``summary.ks_statistics`` becomes a row; the ``drifted`` flag is
    ``ks_statistic > ks_threshold`` (default 0.15, from the Phase 4.5
    plan). Skips the write entirely when ``dataset_too_small=True``
    or no metrics were tested, so a drift-skipped run leaves the
    table untouched rather than writing placeholders.

    Does NOT commit -- caller owns the transaction. Returns the number
    of rows staged.
    """
    from app.core.time import utcnow_naive
    from app.models.ml_drift import MLDriftResult

    if summary.dataset_too_small or not summary.ks_statistics:
        return 0

    now = utcnow_naive()
    count = 0
    for feature_key, ks_stat in summary.ks_statistics.items():
        ks_pvalue = float(summary.p_values.get(feature_key, float("nan")))
        n_ref, n_syn = summary.sample_sizes.get(feature_key, (0, 0))
        db.add(
            MLDriftResult(
                synth_run_id=summary.run_id,
                feature_key=feature_key,
                ks_statistic=float(ks_stat),
                ks_pvalue=ks_pvalue,
                threshold=float(ks_threshold),
                drifted=bool(ks_stat > ks_threshold),
                sample_size_real=int(n_ref),
                sample_size_synth=int(n_syn),
                computed_at=now,
            )
        )
        count += 1
    await db.flush()
    return count


# ─────────────────────────────────────────────────────────────────────────
# Phase 8: Cross-user cohort clustering
# ─────────────────────────────────────────────────────────────────────────


async def opt_in_to_cohorts(db: "AsyncSession", user_id: str) -> None:
    """Record user opt-in to cross-user cohort clustering. Idempotent."""
    from sqlalchemy import select

    from app.core.time import utcnow_naive
    from app.models.ml_cohorts import MLCohortConsent

    existing = await db.execute(
        select(MLCohortConsent).where(MLCohortConsent.user_id == user_id)
    )
    consent = existing.scalar_one_or_none()
    now = utcnow_naive()

    if consent is None:
        db.add(MLCohortConsent(user_id=user_id, opted_in=True, opted_in_at=now))
    else:
        consent.opted_in = True
        consent.opted_in_at = now
    await db.flush()


async def opt_out_of_cohorts(db: "AsyncSession", user_id: str) -> None:
    """Record user opt-out. Queue vector deletion (30-day SLA)."""
    from sqlalchemy import select

    from app.core.time import utcnow_naive
    from app.models.ml_cohorts import MLCohortConsent

    existing = await db.execute(
        select(MLCohortConsent).where(MLCohortConsent.user_id == user_id)
    )
    consent = existing.scalar_one_or_none()
    now = utcnow_naive()

    if consent is None:
        db.add(
            MLCohortConsent(
                user_id=user_id,
                opted_in=False,
                opted_out_at=now,
                deletion_requested_at=now,
            )
        )
    else:
        consent.opted_in = False
        consent.opted_out_at = now
        consent.deletion_requested_at = now
    await db.flush()


async def delete_cohort_vectors(db: "AsyncSession", user_id: str) -> bool:
    """Hard delete anonymized vectors for a user. Idempotent.

    Returns True if vectors were found and deleted.
    """
    from sqlalchemy import delete, select

    from app.core.time import utcnow_naive
    from app.models.ml_cohorts import MLAnonymizedVector, MLCohortConsent
    from ml.cohorts.anonymize import encrypt_user_id, get_rotating_key

    rotating_key = get_rotating_key()
    uid_enc = encrypt_user_id(user_id, rotating_key)

    result = await db.execute(
        delete(MLAnonymizedVector).where(
            MLAnonymizedVector.user_id_encrypted == uid_enc
        )
    )
    deleted = result.rowcount > 0

    # Mark deletion completed.
    existing = await db.execute(
        select(MLCohortConsent).where(MLCohortConsent.user_id == user_id)
    )
    consent = existing.scalar_one_or_none()
    if consent is not None:
        consent.deletion_completed_at = utcnow_naive()

    await db.flush()
    return deleted


async def get_cohort_status(db: "AsyncSession", user_id: str) -> dict:
    """Get opt-in status + cluster membership for a user."""
    from sqlalchemy import select

    from app.models.ml_cohorts import MLCohortConsent

    existing = await db.execute(
        select(MLCohortConsent).where(MLCohortConsent.user_id == user_id)
    )
    consent = existing.scalar_one_or_none()

    if consent is None:
        return {"opted_in": False}

    result = {
        "opted_in": consent.opted_in,
        "opted_in_at": consent.opted_in_at.isoformat() if consent.opted_in_at else None,
        "opted_out_at": consent.opted_out_at.isoformat() if consent.opted_out_at else None,
        "deletion_requested_at": (
            consent.deletion_requested_at.isoformat()
            if consent.deletion_requested_at
            else None
        ),
        "deletion_completed_at": (
            consent.deletion_completed_at.isoformat()
            if consent.deletion_completed_at
            else None
        ),
    }

    # TODO: Phase 8B will add cluster membership lookup here.
    return result


async def run_cohort_clustering(db: "AsyncSession") -> dict:
    """Full Phase 8 pipeline: anonymize opted-in users, cluster, persist.

    Does NOT commit. Caller owns the transaction.
    """
    from ml.cohorts.anonymize import build_anonymized_vectors
    from ml.cohorts.cluster import run_clustering_pipeline

    anon_report = await build_anonymized_vectors(db)
    cluster_report = await run_clustering_pipeline(db)

    return {
        "users_processed": anon_report.users_processed,
        "vectors_created": anon_report.vectors_created,
        "n_clusters": cluster_report.n_clusters,
        "n_noise_points": cluster_report.n_noise_points,
        "largest_cluster": cluster_report.largest_cluster,
        "run_id": cluster_report.run_id,
    }


# ─────────────────────────────────────────────────────────────────────────
# Phase 10: MLOps alerting (public entry points for boundary compliance)
# ─────────────────────────────────────────────────────────────────────────


async def send_drift_alert(report: object) -> None:
    """Send drift detection alert to Discord + Telegram."""
    from ml.mlops.alerts import alert_drift
    await alert_drift(report)


async def send_training_alert(summary: dict) -> None:
    """Send training completion alert to Discord."""
    from ml.mlops.alerts import alert_training_complete
    await alert_training_complete(summary)


async def send_rollback_alert(model_type: str, from_version: str, to_version: str) -> None:
    """Send rollback alert to Discord + Telegram."""
    from ml.mlops.alerts import alert_rollback
    await alert_rollback(model_type, from_version, to_version)


# ─────────────────────────────────────────────────────────────────────────
# Phase 9: L5 APTE n-of-1 experiments
# ─────────────────────────────────────────────────────────────────────────


async def create_experiment(
    db: "AsyncSession",
    user_id: str,
    experiment_name: str,
    treatment_metric: str,
    outcome_metric: str,
    hypothesis: str | None = None,
    design: str = "ab",
    baseline_days: int = 14,
    treatment_days: int = 14,
    washout_days: int = 3,
) -> "object":
    """Create a new personal experiment. Returns the MLExperiment row.

    Computes phase dates from today. Does NOT commit.
    """
    from datetime import date, timedelta

    from app.core.time import utcnow_naive
    from app.models.ml_experiments import MLExperiment

    today = date.today()
    now = utcnow_naive()

    baseline_end = today + timedelta(days=baseline_days - 1)
    treatment_start = baseline_end + timedelta(days=washout_days + 1)
    treatment_end = treatment_start + timedelta(days=treatment_days - 1)

    experiment = MLExperiment(
        user_id=user_id,
        experiment_name=experiment_name,
        hypothesis=hypothesis,
        treatment_metric=treatment_metric,
        outcome_metric=outcome_metric,
        design=design,
        baseline_days=baseline_days,
        treatment_days=treatment_days,
        washout_days=washout_days,
        min_compliance=10,
        status="baseline",
        started_at=now,
        baseline_end=baseline_end.isoformat(),
        treatment_start=treatment_start.isoformat(),
        treatment_end=treatment_end.isoformat(),
        created_at=now,
    )
    db.add(experiment)
    await db.flush()
    return experiment


async def log_experiment_adherence(
    db: "AsyncSession",
    experiment_id: int,
    adherence_date: str,
    compliant: bool,
) -> None:
    """Record daily compliance for an experiment phase. Does NOT commit."""
    from datetime import date

    from app.models.ml_experiments import MLExperiment

    experiment = await db.get(MLExperiment, experiment_id)
    if experiment is None or not compliant:
        return

    log_date = date.fromisoformat(adherence_date)
    baseline_end = date.fromisoformat(experiment.baseline_end)
    treatment_start = date.fromisoformat(experiment.treatment_start)
    treatment_end = date.fromisoformat(experiment.treatment_end)

    if log_date <= baseline_end:
        experiment.compliant_days_baseline += 1
    elif treatment_start <= log_date <= treatment_end:
        experiment.compliant_days_treatment += 1

    await db.flush()


async def check_and_complete_experiments(db: "AsyncSession") -> dict:
    """Scan active experiments, advance phases, run APTE on completable ones.

    Called by the daily experiment_check_job. Does NOT commit.
    """
    from datetime import date

    from sqlalchemy import select

    from app.models.ml_experiments import MLExperiment

    today = date.today()
    summary = {"checked": 0, "advanced": 0, "completed": 0, "failed": 0}

    # Find experiments that might be ready for phase transition.
    stmt = select(MLExperiment).where(
        MLExperiment.status.in_(("baseline", "washout_1", "treatment"))
    )
    result = await db.execute(stmt)
    experiments = result.scalars().all()

    for exp in experiments:
        summary["checked"] += 1
        treatment_end = date.fromisoformat(exp.treatment_end)

        if exp.status == "baseline":
            baseline_end = date.fromisoformat(exp.baseline_end)
            if today > baseline_end:
                exp.status = "washout_1"
                summary["advanced"] += 1

        elif exp.status == "washout_1":
            treatment_start = date.fromisoformat(exp.treatment_start)
            if today >= treatment_start:
                exp.status = "treatment"
                summary["advanced"] += 1

        elif exp.status == "treatment":
            if today > treatment_end:
                exp.status = "analyzing"
                summary["advanced"] += 1

                # Try to run APTE.
                from ml.discovery.apte import run_apte_for_experiment
                apte_result = await run_apte_for_experiment(db, exp.id)
                if apte_result is not None:
                    summary["completed"] += 1
                else:
                    summary["failed"] += 1

    await db.flush()
    return summary


async def get_experiment_result(
    db: "AsyncSession",
    experiment_id: int,
) -> dict | None:
    """Get the APTE result for a completed experiment. Returns None if not found."""
    from sqlalchemy import select

    from app.models.ml_experiments import MLNof1Result

    result = await db.execute(
        select(MLNof1Result).where(MLNof1Result.experiment_id == experiment_id)
    )
    nof1 = result.scalar_one_or_none()
    if nof1 is None:
        return None

    return {
        "apte": nof1.apte,
        "ci_lower": nof1.ci_lower,
        "ci_upper": nof1.ci_upper,
        "p_value": nof1.p_value,
        "effect_size_d": nof1.effect_size_d,
        "baseline_mean": nof1.baseline_mean,
        "treatment_mean": nof1.treatment_mean,
        "baseline_n": nof1.baseline_n,
        "treatment_n": nof1.treatment_n,
        "method": nof1.method,
    }
