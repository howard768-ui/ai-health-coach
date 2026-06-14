"""Waitlist signup endpoint for heymeld.com.

Public, unauthenticated endpoint. Accepts an email + optional campaign metadata,
normalizes the email (lowercase, trimmed), and either creates a new
`WaitlistSignup` row or increments the submission counter on an existing one.

Hardened against abuse with:
- Rate limiting (via slowapi, 10/minute per IP)
- Email format validation (EmailStr)
- Dedupe on lowercased email
- IP hashed (never stored raw)
- User-Agent truncated to 512 chars
"""

import hashlib
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ratelimit import limiter
from app.database import get_db
from app.models.waitlist import WaitlistSignup
from app.services.waitlist_notifier import send_new_signup_alert

logger = logging.getLogger("meld.waitlist")

router = APIRouter(prefix="/api/waitlist", tags=["waitlist"])


# ── Request / Response ──────────────────────────────────────


class WaitlistSubscribeRequest(BaseModel):
    email: EmailStr
    source: str | None = Field(default=None, max_length=64)
    utm_source: str | None = Field(default=None, max_length=128)
    utm_medium: str | None = Field(default=None, max_length=128)
    utm_campaign: str | None = Field(default=None, max_length=128)
    utm_term: str | None = Field(default=None, max_length=128)
    utm_content: str | None = Field(default=None, max_length=128)


class WaitlistSubscribeResponse(BaseModel):
    status: str
    message: str
    # True on a brand-new signup, False on a duplicate (idempotent re-submission).
    new: bool


# ── Helpers ─────────────────────────────────────────────────


def _hash_ip(ip: str | None) -> str | None:
    """One-way SHA-256 of the IP. We never store raw IPs."""
    if not ip:
        return None
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:32]


def _client_ip(request: Request) -> str | None:
    """Best-effort client IP — respects X-Forwarded-For for Cloudflare + Railway."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


# ── Endpoint ────────────────────────────────────────────────


@router.post("/subscribe", response_model=WaitlistSubscribeResponse)
@limiter.limit("10/minute")
async def subscribe(
    payload: WaitlistSubscribeRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user_agent: str | None = Header(default=None, alias="User-Agent"),
    referer: str | None = Header(default=None, alias="Referer"),
    db: AsyncSession = Depends(get_db),
) -> WaitlistSubscribeResponse:
    """Subscribe an email to the Meld launch waitlist.

    Idempotent: a duplicate submission returns 200 with `new=False` and bumps
    the row's `submissions` counter + `updated_at`. This keeps the API UX clean
    for users re-submitting the form (e.g. after a "did it work?" moment).
    """
    # Rate limited to 10/minute per real client IP via the @limiter.limit
    # decorator above (audit C2 re-gate: the documented limit was previously
    # not actually applied; only the global default protected this endpoint).

    normalized_email = payload.email.lower().strip()

    # Look up existing row
    result = await db.execute(
        select(WaitlistSignup).where(WaitlistSignup.email == normalized_email)
    )
    existing = result.scalar_one_or_none()

    if existing is not None:
        existing.submissions += 1
        # Refresh metadata to the latest submission's values if provided.
        if payload.source:
            existing.source = payload.source[:64]
        if payload.utm_source:
            existing.utm_source = payload.utm_source[:128]
        if payload.utm_medium:
            existing.utm_medium = payload.utm_medium[:128]
        if payload.utm_campaign:
            existing.utm_campaign = payload.utm_campaign[:128]
        if payload.utm_term:
            existing.utm_term = payload.utm_term[:128]
        if payload.utm_content:
            existing.utm_content = payload.utm_content[:128]
        await db.commit()
        # Log a redacted form: full emails in info logs end up in the log
        # aggregator (PII hygiene, same class as apple_user_id truncation).
        logger.info(
            "waitlist re-submission email=%s*** submissions=%d",
            normalized_email[:3],
            existing.submissions,
        )
        return WaitlistSubscribeResponse(
            status="ok",
            message="You're already on the list. We'll be in touch.",
            new=False,
        )

    # Create new row
    signup = WaitlistSignup(
        email=normalized_email,
        source=(payload.source or "")[:64] or None,
        utm_source=(payload.utm_source or "")[:128] or None,
        utm_medium=(payload.utm_medium or "")[:128] or None,
        utm_campaign=(payload.utm_campaign or "")[:128] or None,
        utm_term=(payload.utm_term or "")[:128] or None,
        utm_content=(payload.utm_content or "")[:128] or None,
        referer=(referer or "")[:512] or None,
        ip_hash=_hash_ip(_client_ip(request)),
        user_agent=(user_agent or "")[:512] or None,
    )
    db.add(signup)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        # If a race created the row between our SELECT and INSERT, treat as success.
        logger.warning("waitlist insert race or error: %s", exc)
        raise HTTPException(
            status_code=500, detail="Could not save your email. Please try again."
        ) from exc

    # Log a redacted form (PII hygiene, same as the re-submission path).
    logger.info(
        "waitlist new signup email=%s*** source=%s",
        normalized_email[:3],
        payload.source,
    )

    # Fire-and-forget admin alert via Resend. Runs after the response is sent
    # so signup latency is unaffected. Swallows its own errors; if Resend is
    # down or RESEND_API_KEY is unset, the signup still succeeds.
    background_tasks.add_task(send_new_signup_alert, signup.id)

    return WaitlistSubscribeResponse(
        status="ok",
        message="You're on the list. We'll be in touch.",
        new=True,
    )
