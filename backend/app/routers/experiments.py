"""Phase 9 experiment API endpoints.

- ``POST /api/experiments``: create a new personal experiment.
- ``GET /api/experiments``: list user's experiments.
- ``GET /api/experiments/{id}``: experiment details + result if completed.
- ``POST /api/experiments/{id}/log-adherence``: record daily compliance.
- ``POST /api/experiments/{id}/abandon``: cancel an active experiment.
- ``GET /api/experiments/{id}/result``: get APTE result.

Shadow-gated behind ``ml_shadow_apte``.
"""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.core.time import utcnow_naive
from app.database import get_db

logger = logging.getLogger("meld.experiments")

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class CreateExperimentRequest(BaseModel):
    experiment_name: str
    treatment_metric: str
    outcome_metric: str
    hypothesis: str | None = None
    design: str = "ab"
    baseline_days: int = 14
    treatment_days: int = 14


class ExperimentResponse(BaseModel):
    id: int
    experiment_name: str
    hypothesis: str | None
    treatment_metric: str
    outcome_metric: str
    design: str
    status: str
    baseline_days: int
    treatment_days: int
    baseline_end: str
    treatment_start: str
    treatment_end: str
    compliant_days_baseline: int
    compliant_days_treatment: int
    started_at: str
    completed_at: str | None


class LogAdherenceRequest(BaseModel):
    date: str  # YYYY-MM-DD
    compliant: bool


class AdherenceResponse(BaseModel):
    ok: bool = True
    compliant_days_baseline: int
    compliant_days_treatment: int


class ResultResponse(BaseModel):
    apte: float | None
    ci_lower: float | None
    ci_upper: float | None
    p_value: float | None
    effect_size_d: float | None
    baseline_mean: float | None
    treatment_mean: float | None
    baseline_n: int
    treatment_n: int
    method: str


class AbandonResponse(BaseModel):
    ok: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    req: CreateExperimentRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Create a new personal experiment."""
    from ml import api as ml_api

    experiment = await ml_api.create_experiment(
        db,
        user_id=user.apple_user_id,
        experiment_name=req.experiment_name,
        treatment_metric=req.treatment_metric,
        outcome_metric=req.outcome_metric,
        hypothesis=req.hypothesis,
        design=req.design,
        baseline_days=req.baseline_days,
        treatment_days=req.treatment_days,
    )
    await db.commit()
    return _to_response(experiment)


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> list[ExperimentResponse]:
    """List all experiments for the current user."""
    from app.models.ml_experiments import MLExperiment

    result = await db.execute(
        select(MLExperiment)
        .where(MLExperiment.user_id == user.apple_user_id)
        .order_by(MLExperiment.created_at.desc())
    )
    experiments = result.scalars().all()
    return [_to_response(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Get experiment details."""
    from app.models.ml_experiments import MLExperiment

    experiment = await db.get(MLExperiment, experiment_id)
    if experiment is None or experiment.user_id != user.apple_user_id:
        raise HTTPException(status_code=404, detail="experiment not found")
    return _to_response(experiment)


@router.post("/{experiment_id}/log-adherence", response_model=AdherenceResponse)
async def log_adherence(
    experiment_id: int,
    req: LogAdherenceRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> AdherenceResponse:
    """Record daily compliance for an active experiment."""
    from app.models.ml_experiments import MLExperiment

    experiment = await db.get(MLExperiment, experiment_id)
    if experiment is None or experiment.user_id != user.apple_user_id:
        raise HTTPException(status_code=404, detail="experiment not found")
    if experiment.status in ("completed", "abandoned", "analyzing"):
        raise HTTPException(status_code=400, detail="experiment is not active")

    if not req.compliant:
        await db.commit()
        return AdherenceResponse(
            compliant_days_baseline=experiment.compliant_days_baseline,
            compliant_days_treatment=experiment.compliant_days_treatment,
        )

    # Determine which phase this date falls in.
    log_date = date.fromisoformat(req.date)
    baseline_end = date.fromisoformat(experiment.baseline_end)
    treatment_start = date.fromisoformat(experiment.treatment_start)
    treatment_end = date.fromisoformat(experiment.treatment_end)

    if log_date <= baseline_end:
        experiment.compliant_days_baseline += 1
    elif treatment_start <= log_date <= treatment_end:
        experiment.compliant_days_treatment += 1

    await db.commit()
    return AdherenceResponse(
        compliant_days_baseline=experiment.compliant_days_baseline,
        compliant_days_treatment=experiment.compliant_days_treatment,
    )


@router.post("/{experiment_id}/abandon", response_model=AbandonResponse)
async def abandon_experiment(
    experiment_id: int,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> AbandonResponse:
    """Cancel an active experiment."""
    from app.models.ml_experiments import MLExperiment

    experiment = await db.get(MLExperiment, experiment_id)
    if experiment is None or experiment.user_id != user.apple_user_id:
        raise HTTPException(status_code=404, detail="experiment not found")
    if experiment.status in ("completed", "abandoned"):
        raise HTTPException(status_code=400, detail="experiment already ended")

    experiment.status = "abandoned"
    experiment.completed_at = utcnow_naive()
    await db.commit()
    return AbandonResponse()


@router.get("/{experiment_id}/result", response_model=ResultResponse)
async def get_result(
    experiment_id: int,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ResultResponse:
    """Get APTE result for a completed experiment."""
    from app.models.ml_experiments import MLExperiment, MLNof1Result

    experiment = await db.get(MLExperiment, experiment_id)
    if experiment is None or experiment.user_id != user.apple_user_id:
        raise HTTPException(status_code=404, detail="experiment not found")
    if experiment.status != "completed":
        raise HTTPException(status_code=404, detail="experiment not completed")

    result = await db.execute(
        select(MLNof1Result).where(MLNof1Result.experiment_id == experiment_id)
    )
    nof1 = result.scalar_one_or_none()
    if nof1 is None:
        raise HTTPException(status_code=404, detail="result not found")

    return ResultResponse(
        apte=nof1.apte,
        ci_lower=nof1.ci_lower,
        ci_upper=nof1.ci_upper,
        p_value=nof1.p_value,
        effect_size_d=nof1.effect_size_d,
        baseline_mean=nof1.baseline_mean,
        treatment_mean=nof1.treatment_mean,
        baseline_n=nof1.baseline_n,
        treatment_n=nof1.treatment_n,
        method=nof1.method,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(e) -> ExperimentResponse:
    return ExperimentResponse(
        id=e.id,
        experiment_name=e.experiment_name,
        hypothesis=e.hypothesis,
        treatment_metric=e.treatment_metric,
        outcome_metric=e.outcome_metric,
        design=e.design,
        status=e.status,
        baseline_days=e.baseline_days,
        treatment_days=e.treatment_days,
        baseline_end=e.baseline_end,
        treatment_start=e.treatment_start,
        treatment_end=e.treatment_end,
        compliant_days_baseline=e.compliant_days_baseline,
        compliant_days_treatment=e.compliant_days_treatment,
        started_at=e.started_at.isoformat(),
        completed_at=e.completed_at.isoformat() if e.completed_at else None,
    )
