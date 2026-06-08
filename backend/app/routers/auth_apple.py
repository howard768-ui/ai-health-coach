"""Sign in with Apple authentication endpoints.

Flow overview:

1. POST /auth/apple         — verify Apple identity token, upsert user, issue tokens
2. POST /auth/refresh       — rotate refresh token, issue new access token
3. POST /auth/logout        — revoke the caller's refresh token
4. POST /auth/delete        — call Apple /auth/revoke, delete user + all data
5. POST /auth/apple/revoked — Apple server-to-server notification endpoint

All endpoints are unauthenticated EXCEPT /auth/logout and /auth/delete, which
require a valid access token via `CurrentUser`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ratelimit import limiter

from app.api.deps import CurrentUser
from app.core.apple import (
    is_private_relay_email,
    revoke_apple_token,
    verify_apple_identity_token,
    verify_apple_server_notification,
)
from app.core.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
)
from app.database import get_db
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.core.time import utcnow_naive

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("meld.auth")

# Per-IP rate limiter shared across auth endpoints. Tighter than the
# global default — auth endpoints are the highest-value attack surface
# (brute force, token enumeration, cost exhaustion via Apple JWT verify).
# Uses the single shared limiter (app.core.ratelimit), keyed on the real client
# IP, not the Cloudflare edge IP. Audit C2.


# ── Request / Response Schemas ───────────────────────────────────────────────


class AppleSignInRequest(BaseModel):
    identity_token: str = Field(..., description="JWT from ASAuthorizationAppleIDCredential")
    raw_nonce: str | None = Field(None, description="Raw nonce whose SHA256 was sent to Apple")
    # Full name & email only arrive on FIRST sign-in. Persist if present.
    full_name: str | None = None
    email: str | None = None
    device_id: str | None = Field(None, description="iOS identifierForVendor for session tracking")
    # If Apple returned an authorizationCode, we can exchange it for an Apple
    # refresh token via /auth/token (required for /auth/revoke on deletion).
    # MVP skip: we'll capture this in a later iteration.
    authorization_code: str | None = None


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until access_token expiry
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class DeleteRequest(BaseModel):
    # Optional confirmation string — iOS client can send the user's email or
    # a UI-generated challenge to prevent accidental deletions.
    confirmation: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _user_to_dict(user: User) -> dict:
    """Serialize a User into the minimal shape the iOS client needs post-auth."""
    return {
        "id": user.apple_user_id,
        "name": user.name,
        "email": user.email,
        "is_private_email": user.is_private_email,
    }


async def _issue_token_pair(
    db: AsyncSession, user: User, device_id: str | None
) -> TokenPair:
    """Issue a new access + refresh pair, persist the refresh hash, return."""
    access_token, access_expires = create_access_token(user.apple_user_id)
    raw_refresh, refresh_hash, refresh_expires = create_refresh_token()

    token_row = RefreshToken(
        id=refresh_hash,
        user_id=user.apple_user_id,
        device_id=device_id,
        expires_at=refresh_expires,
    )
    db.add(token_row)

    user.last_login_at = utcnow_naive()
    await db.commit()

    expires_in = int((access_expires - datetime.now(timezone.utc)).total_seconds())
    return TokenPair(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=expires_in,
        user=_user_to_dict(user),
    )


async def _revoke_chain(db: AsyncSession, start_hash: str) -> None:
    """Walk the `replaced_by` chain starting at `start_hash` and revoke every
    token. Used when a revoked token is presented for refresh — indicates the
    refresh token was stolen and replayed after rotation.
    """
    now = utcnow_naive()
    current = start_hash
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        row = await db.get(RefreshToken, current)
        if row is None:
            break
        if row.revoked_at is None:
            row.revoked_at = now
        current = row.replaced_by or ""
    await db.flush()


# ── POST /auth/apple ─────────────────────────────────────────────────────────


@router.post("/apple", response_model=TokenPair)
@limiter.limit("10/minute")
async def sign_in_with_apple(
    request: Request,
    body: AppleSignInRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenPair:
    """Verify an Apple identity token, upsert the user, and issue our tokens.

    Rate-limited to 10/min per IP — protects against Apple JWKS verification
    cost exhaustion and brute-force enumeration.
    """
    try:
        claims = verify_apple_identity_token(body.identity_token, body.raw_nonce)
    except jwt.InvalidTokenError as e:
        logger.warning("Apple identity token verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Apple identity token: {e}",
        )

    apple_sub = claims["sub"]  # source of truth — NOT request.user_identifier
    claim_email = claims.get("email")  # Apple puts email in claims on first sign-in
    # The iOS client may also provide full_name/email directly (first time only).

    # Find or create user
    result = await db.execute(select(User).where(User.apple_user_id == apple_sub))
    user = result.scalar_one_or_none()

    if user is None:
        logger.info("Creating new user for Apple sub=%s", apple_sub[:12] + "...")
        user = User(
            apple_user_id=apple_sub,
            email=body.email or claim_email,
            name=body.full_name,
            is_active=True,
            is_private_email=is_private_relay_email(body.email or claim_email),
        )
        db.add(user)
        await db.flush()
    else:
        # Subsequent sign-ins: update name/email ONLY if we have new info.
        # Apple only returns these on the first sign-in, so if iOS sent them
        # again it's probably a fresh re-auth after credential deletion.
        if body.full_name and not user.name:
            user.name = body.full_name
        if (body.email or claim_email) and not user.email:
            email_to_set = body.email or claim_email
            user.email = email_to_set
            user.is_private_email = is_private_relay_email(email_to_set)

    return await _issue_token_pair(db, user, body.device_id)


# ── POST /auth/refresh ───────────────────────────────────────────────────────


@router.post("/refresh", response_model=TokenPair)
@limiter.limit("30/minute")
async def refresh_token(
    request: Request,
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenPair:
    """Rotate a refresh token and issue a new access + refresh pair.

    Reuse detection: if a revoked token is presented, we walk the replacement
    chain and revoke every descendant for the user. This is the primary
    defense against refresh token theft — once the attacker uses a stolen
    token, the legitimate client's next refresh will be rejected.

    Rate-limited to 30/min per IP — generous enough for legitimate token
    refresh on app launches, tight enough to flag enumeration attempts.
    """
    token_hash = hash_refresh_token(body.refresh_token)
    row = await db.get(RefreshToken, token_hash)

    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unknown refresh token")

    # Reuse detection: revoked token presented again means the chain is compromised.
    if row.revoked_at is not None:
        # Logs internal user_id only, no token material.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        logger.warning(
            "Refresh token reuse detected for user=%s; revoking chain",
            row.user_id,
        )
        await _revoke_chain(db, token_hash)
        # Also revoke every other active refresh token for this user as a precaution.
        result = await db.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == row.user_id,
                RefreshToken.revoked_at.is_(None),
            )
        )
        for t in result.scalars():
            t.revoked_at = utcnow_naive()
        await db.commit()
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Refresh token reuse detected — please sign in again",
        )

    # Expiry check (accept both naive UTC and aware for legacy rows)
    now_naive = utcnow_naive()
    if row.expires_at < now_naive:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token expired")

    # Load user and verify active
    user = await db.execute(select(User).where(User.apple_user_id == row.user_id))
    user = user.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")

    # Mark old token revoked, issue new pair, link via replaced_by
    row.revoked_at = now_naive
    access_token, access_expires = create_access_token(user.apple_user_id)
    new_raw, new_hash, new_expires = create_refresh_token()
    row.replaced_by = new_hash

    new_row = RefreshToken(
        id=new_hash,
        user_id=user.apple_user_id,
        device_id=row.device_id,
        expires_at=new_expires,
    )
    db.add(new_row)
    await db.commit()

    expires_in = int((access_expires - datetime.now(timezone.utc)).total_seconds())
    return TokenPair(
        access_token=access_token,
        refresh_token=new_raw,
        expires_in=expires_in,
        user=_user_to_dict(user),
    )


# ── POST /auth/logout ────────────────────────────────────────────────────────


@router.post("/logout")
@limiter.limit("20/minute")
async def logout(
    request: Request,
    user: CurrentUser,
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Revoke the caller's refresh token. Access token remains valid until
    its 15-minute expiry, but without a refresh token the client can't
    extend the session.
    """
    token_hash = hash_refresh_token(body.refresh_token)
    row = await db.get(RefreshToken, token_hash)
    if row is not None and row.user_id == user.apple_user_id and row.revoked_at is None:
        row.revoked_at = utcnow_naive()
        await db.commit()
    return {"status": "logged_out"}


# ── POST /auth/delete ────────────────────────────────────────────────────────


# Per-user PHI tables keyed by user_id that have NO foreign key to users, so a
# CASCADE delete of the user row does not remove them. These must be purged
# explicitly on account deletion (audit A3 / round-1 critical right-to-erasure).
# Cohort data (ml_anonymized_vectors / ml_cohort_consent) is handled separately
# via ml.api.delete_cohort_vectors because it is keyed by an encrypted pseudonym.
# Raw table names (no ORM import) keep the ML import boundary clean, matching
# the ml_ops.py pattern.
_PER_USER_PHI_TABLES_NO_FK = (
    "ml_baselines",
    "ml_change_points",
    "ml_forecasts",
    "ml_anomalies",
    "ml_directional_tests",
    "ml_causal_estimates",
    "ml_feature_values",
    "ml_insight_candidates",
    "ml_rankings",
    # ml_n_of_1_results has a RESTRICT FK to ml_experiments, so with SQLite FK
    # enforcement now on (database.py) the child MUST be deleted before the
    # parent or the DELETE FROM ml_experiments raises and blocks the whole
    # account deletion. Order matters here.
    "ml_n_of_1_results",
    "ml_experiments",
    "user_mascot_state",
    "notification_records",
    "device_tokens",
    "notification_preferences",
)


@router.post("/delete")
@limiter.limit("3/minute")
async def delete_account(
    request: Request,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Permanently delete the user's account.

    Required by App Store guideline 5.1.1(v) since June 30, 2022.

    Steps:
    1. Call Apple's /auth/revoke with the user's Apple refresh token (if we
       have one captured from the original sign-in).
    2. Purge cohort vectors + every per-user PHI table that has no FK to users
       (the ml_*/notification/mascot tables CASCADE does not reach).
    3. DELETE the user row — CASCADE removes the FK-linked tenant tables (the
       c0518b5194eb set), now that SQLite FK enforcement is on (see database.py).
    """
    apple_id = user.apple_user_id
    apple_refresh = user.apple_refresh_token

    # Best-effort: call Apple's revoke endpoint. If this fails we still
    # delete local data — the user can manually revoke in Apple ID Settings.
    if apple_refresh:
        try:
            await revoke_apple_token(apple_refresh)
            # Logs first 12 chars of apple_id (truncated identifier), no token.
            # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
            logger.info("Apple token revoked for user=%s", apple_id[:12] + "...")
        except (httpx.HTTPError, jwt.PyJWTError, ValueError) as e:
            logger.error("Apple token revocation failed for user=%s: %s", apple_id[:12], e)
            try:
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("apple_action", "revoke_token")
                    scope.set_tag("user_id_prefix", apple_id[:12])
                    sentry_sdk.capture_exception(e)
            except Exception:  # noqa: BLE001 -- delete continues regardless
                logger.debug("Sentry capture failed (non-fatal)", exc_info=True)

    # Purge cohort vectors (keyed by encrypted pseudonym) via the ML boundary.
    # Wrap in a SAVEPOINT so a cohort-purge failure rolls back only this step
    # and cannot poison the session for the rest of the deletion (the local
    # delete must complete per App Store 5.1.1(v)).
    try:
        from ml import api as ml_api

        async with db.begin_nested():
            await ml_api.delete_cohort_vectors(db, apple_id)
    except Exception as e:  # noqa: BLE001 -- cohort purge must not block deletion
        logger.error("Cohort vector purge failed for user=%s: %s", apple_id[:12], e)

    # Purge every per-user PHI table CASCADE does not reach. Table names are
    # fixed constants, not user input, so the f-string is not an injection site.
    for table in _PER_USER_PHI_TABLES_NO_FK:
        await db.execute(
            text(f"DELETE FROM {table} WHERE user_id = :uid"),  # noqa: S608
            {"uid": apple_id},
        )

    # Delete the user — CASCADE handles the FK-linked tenant tables + refresh
    # tokens (FK enforcement is on for SQLite via database.py; native on PG).
    await db.delete(user)
    await db.commit()
    logger.info("Deleted user account: %s", apple_id[:12] + "...")

    return {"status": "deleted", "user_id": apple_id}


# ── POST /auth/apple/revoked ─────────────────────────────────────────────────


@router.post("/apple/revoked")
async def apple_server_notification(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Apple's server-to-server consent-revoked notifications.

    When a user revokes Sign in with Apple from Settings (or deletes their
    Apple ID, or toggles email relay), Apple POSTs ``{"payload": "<JWT>"}``
    here. We verify both the outer JWT and the inner ``events`` JWT against
    Apple's JWKS, then act on the event type:

      - consent-revoked / account-delete → mark user inactive
      - email-disabled / email-enabled   → log only (no action yet)

    Always returns 200 to Apple regardless of whether a matching user was
    found, because Apple's retry budget is scarce and a missing user is a
    legitimate state (we may have already deleted them locally).

    Reference: https://developer.apple.com/documentation/technotes/tn3194-handling-account-deletions-and-revoking-tokens-for-sign-in-with-apple
    """
    try:
        body_json = await request.json()
    except (ValueError, UnicodeDecodeError):
        logger.warning("Apple server notification with malformed body")
        return {"status": "invalid_body"}

    signed_payload = body_json.get("payload") if isinstance(body_json, dict) else None
    if not isinstance(signed_payload, str) or not signed_payload:
        logger.warning("Apple server notification missing payload field")
        return {"status": "invalid_payload"}

    try:
        event = verify_apple_server_notification(signed_payload)
    except jwt.InvalidTokenError as e:
        # Forged or malformed notification. Return 400 so attackers can
        # distinguish "we don't know who you are" from "we processed it";
        # legitimate Apple retries would never produce this.
        logger.warning("Apple server notification verification failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid Apple JWT")

    apple_user_id = event.get("sub")
    event_type = event.get("type")
    logger.info(
        "Apple server notification: type=%s sub=%s event_time=%s",
        event_type, apple_user_id, event.get("event_time"),
    )

    # consent-revoked + account-delete: mark user inactive. Full deletion
    # of associated rows is owned by /auth/delete (which runs the same
    # cleanup synchronously when the user requests it from inside the app);
    # here we only flip is_active so the user can't sign back in until they
    # explicitly re-consent.
    if event_type in ("consent-revoked", "account-delete") and apple_user_id:
        result = await db.execute(
            select(User).where(User.apple_user_id == apple_user_id)
        )
        user = result.scalar_one_or_none()
        if user is not None and user.is_active:
            user.is_active = False
            await db.commit()
            logger.info("Deactivated user %s on %s notification", apple_user_id, event_type)
    return {"status": "ok", "type": event_type}
