"""Phase 8 privacy endpoints for cross-user cohort opt-in/opt-out.

- ``POST /api/privacy/cohort-opt-in``: opt in to cross-user clustering.
- ``POST /api/privacy/cohort-opt-out``: opt out + queue vector deletion.
- ``GET /api/privacy/cohort-status``: current opt-in status + cohort info.
- ``DELETE /api/privacy/cohort-contribution``: hard delete anonymized vectors.

All endpoints require authentication. The cohort feature itself is
shadow-gated behind ``ml_shadow_cohorts``; these endpoints work
regardless of the shadow flag (a user can opt in before the feature
surfaces insights).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.database import get_db

logger = logging.getLogger("meld.privacy")

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CohortOptInResponse(BaseModel):
    opted_in: bool
    message: str


class CohortStatusResponse(BaseModel):
    opted_in: bool
    opted_in_at: str | None = None
    opted_out_at: str | None = None
    deletion_requested_at: str | None = None
    deletion_completed_at: str | None = None
    cluster_label: int | None = None
    cluster_name: str | None = None
    cluster_size: int | None = None


class DeletionResponse(BaseModel):
    deleted: bool
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/cohort-opt-in", response_model=CohortOptInResponse)
async def cohort_opt_in(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CohortOptInResponse:
    """Opt in to cross-user cohort clustering. Idempotent."""
    from ml import api as ml_api

    await ml_api.opt_in_to_cohorts(db, user.apple_user_id)
    await db.commit()
    return CohortOptInResponse(
        opted_in=True,
        message="You are now sharing your health patterns with the community.",
    )


@router.post("/cohort-opt-out", response_model=CohortOptInResponse)
async def cohort_opt_out(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CohortOptInResponse:
    """Opt out of cross-user cohort clustering. Queues vector deletion."""
    from ml import api as ml_api

    await ml_api.opt_out_of_cohorts(db, user.apple_user_id)
    await db.commit()
    return CohortOptInResponse(
        opted_in=False,
        message="Your shared data will be deleted within 30 days.",
    )


@router.get("/cohort-status", response_model=CohortStatusResponse)
async def cohort_status(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> CohortStatusResponse:
    """Get current cohort opt-in status and cluster membership."""
    from ml import api as ml_api

    status = await ml_api.get_cohort_status(db, user.apple_user_id)
    return CohortStatusResponse(**status)


@router.delete("/cohort-contribution", response_model=DeletionResponse)
async def delete_cohort_contribution(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> DeletionResponse:
    """Hard delete anonymized vectors for this user. Idempotent."""
    from ml import api as ml_api

    deleted = await ml_api.delete_cohort_vectors(db, user.apple_user_id)
    await db.commit()
    return DeletionResponse(
        deleted=deleted,
        message="Your anonymized data has been deleted." if deleted else "No data found to delete.",
    )
