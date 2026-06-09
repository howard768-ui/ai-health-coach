"""Legacy Oura OAuth callback endpoints.

Oura uses a browser-redirect OAuth flow, so we can't use the normal bearer
auth dependency. Instead we pass the Meld user's apple_user_id through the
OAuth `state` parameter — Oura echoes it back on the callback, and we use
it to attach the resulting token to the right user.

The state is validated against an existing user row on return. This means
an attacker can't use the endpoint to attach Oura tokens to users they
don't control, because the user must have already completed Sign in with
Apple to exist in the users table.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timedelta

import httpx
from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.config import settings
from app.database import get_db
from app.models.health import OuraToken
from app.models.user import User
from app.services.oura import OuraClient
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.auth_oura")

router = APIRouter(prefix="/auth", tags=["auth"])

# How long a signed OAuth state token stays valid. The user has to finish the
# Oura consent screen within this window. 10 minutes is generous for a human.
_OAUTH_STATE_TTL_SECONDS = 600


def _sign_oauth_state(user_id: str) -> str:
    """Mint a signed, time-limited OAuth `state` token bound to `user_id`.

    The previous scheme used the raw apple_user_id as `state`, supplied by an
    unauthenticated caller. An attacker who knew (or guessed) a victim's
    apple_user_id could forge the callback URL and attach their OWN Oura
    account to the victim's Meld account (OAuth CSRF / account injection).

    A signed token can only be produced by `/auth/oura/start`, which requires
    a valid bearer token, so an attacker cannot mint a state for a user they
    do not control. The HMAC uses `app_secret_key` (NOT the PHI encryption
    key) so this is independent of at-rest encryption.

    Format: ``<base64url(payload)>.<base64url(hmac_sha256(payload))>``.
    """
    payload = {
        "uid": user_id,
        "exp": int(time.time()) + _OAUTH_STATE_TTL_SECONDS,
        "nonce": secrets.token_urlsafe(8),
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).rstrip(b"=")
    sig = hmac.new(settings.app_secret_key.encode(), body, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=")
    return f"{body.decode()}.{sig_b64.decode()}"


def _verify_oauth_state(state: str) -> str | None:
    """Return the `user_id` from a valid signed state, else None.

    None means the token is malformed, tampered, or expired. The callback
    treats None as "not a signed token" and falls back to the legacy
    apple_user_id path so older app builds keep working during the transition.
    """
    try:
        body_str, sig_str = state.rsplit(".", 1)
    except ValueError:
        return None
    expected = hmac.new(
        settings.app_secret_key.encode(), body_str.encode(), hashlib.sha256
    ).digest()
    expected_b64 = base64.urlsafe_b64encode(expected).rstrip(b"=").decode()
    if not hmac.compare_digest(expected_b64, sig_str):
        return None
    try:
        padded = body_str + "=" * (-len(body_str) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("uid"):
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return str(payload["uid"])


@router.post("/oura/start")
async def oura_start(current_user: CurrentUser):
    """Begin the Oura OAuth flow for the authenticated user (CSRF-safe).

    Returns the Oura authorize URL carrying a signed, time-limited `state`
    bound to this user. The iOS app opens the returned URL in Safari. This
    replaces opening ``/auth/oura`` with a client-supplied apple_user_id,
    which let an attacker attach their Oura account to another user.
    """
    state = _sign_oauth_state(current_user.apple_user_id)
    auth_url = OuraClient().get_auth_url()
    separator = "&" if "?" in auth_url else "?"
    return {"authorize_url": f"{auth_url}{separator}state={state}"}


@router.get("/oura")
async def oura_auth(
    state: str = Query(..., description="apple_user_id of the authenticated user initiating OAuth"),
):
    """Legacy redirect entry point for the Oura OAuth flow.

    DEPRECATED in favor of `POST /auth/oura/start`, which binds the state to
    the authenticated user. Kept so older app builds still function; the
    callback accepts both signed states and raw apple_user_id states. Remove
    once the CSRF-safe iOS build has fully rolled out.
    """
    client = OuraClient()
    # Append state to Oura's authorize URL
    auth_url = client.get_auth_url()
    separator = "&" if "?" in auth_url else "?"
    return RedirectResponse(url=f"{auth_url}{separator}state={state}")


# Deep-link scheme used by the iOS app (see `Meld/App/MeldApp.swift` onOpenURL).
# Safari follows these on iPhone, which closes the browser tab and pops back
# into the app. Without this, the user lands on the JSON response and has to
# manually swipe back — bad UX.
_OURA_SUCCESS_DEEPLINK = "meld://oura/connected"


def _oura_error_deeplink(reason: str) -> str:
    """Build a meld://oura/error deep link with a stable reason code."""
    return f"meld://oura/error?reason={reason}"


@router.get("/oura/callback")
async def oura_callback(
    code: str | None = Query(None),
    state: str = Query(..., description="apple_user_id echoed back from /auth/oura"),
    error: str | None = Query(None, description="OAuth error code per RFC 6749 §4.1.2.1 (e.g. access_denied)"),
    db: AsyncSession = Depends(get_db),
):
    """Handle Oura OAuth callback — exchange code for tokens, attach to user.

    On success, redirects Safari to ``meld://oura/connected`` so the iOS app
    re-opens automatically. On failure (bad state, exchange error, user
    cancellation), redirects to ``meld://oura/error?reason=<code>`` so the
    iOS handler can show an appropriate alert. We never return a JSON body
    here — Safari would just render it as text and strand the user on the web.

    `code` is optional because Oura sends `?error=access_denied&state=...` (no
    code) when the user taps Cancel on the consent screen. FastAPI's default
    422 for a missing required param would strand the user; we handle the
    cancel path explicitly instead.
    """
    # User cancelled on Oura's consent screen, or Oura returned an explicit
    # OAuth error. Bounce back with the reason so iOS can show "you cancelled".
    if error:
        logger.info("Oura callback received error param: %s", error)
        return RedirectResponse(url=_oura_error_deeplink(error))

    # No code AND no error means a malformed redirect (shouldn't happen, but
    # don't 422 the user out either — same deep-link failure path).
    if not code:
        logger.warning("Oura callback: missing both code and error params")
        return RedirectResponse(url=_oura_error_deeplink("missing_code"))

    # Resolve which Meld user this callback belongs to. Prefer the signed
    # state minted by /auth/oura/start (CSRF-safe); fall back to treating
    # `state` as a raw apple_user_id for older app builds still in the wild.
    # Remove the fallback once the CSRF-safe iOS build has fully rolled out.
    resolved_user_id = _verify_oauth_state(state) or state

    # Validate state points to a real user. Don't 400 — Safari renders 4xx
    # bodies as plain text on iPhone, stranding the user. Deep-link out so
    # the iOS app gets the failure signal and can show its own error UI.
    result = await db.execute(select(User).where(User.apple_user_id == resolved_user_id))
    user = result.scalar_one_or_none()
    if user is None:
        logger.warning("Oura callback: state did not resolve to any User row")
        return RedirectResponse(url=_oura_error_deeplink("invalid_state"))

    client = OuraClient()
    try:
        token_data = await client.exchange_code(code)
    except httpx.HTTPError as e:
        logger.error("Oura exchange_code failed for apple_user=%s: %s", resolved_user_id[:12] + "...", e)
        return RedirectResponse(url=_oura_error_deeplink("exchange_failed"))

    # MEL-45 part 2: capture Oura's user ID so the webhook receiver can route
    # incoming events to the correct Meld user. Best-effort — if personal_info
    # fails (network, Oura down), store NULL and let sync_user_data backfill
    # on the next sync. Webhooks will fall back to the single-user path until
    # the column is populated.
    oura_user_id: str | None = None
    try:
        info_client = OuraClient(access_token=token_data["access_token"])
        info = await info_client.get_personal_info()
        oura_user_id = info.get("id")
        if not oura_user_id:
            logger.warning(
                "Oura personal_info returned no `id` field for apple_user=%s; "
                "webhook routing will use single-user fallback until next sync backfills",
                resolved_user_id[:12] + "...",
            )
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning(
            "Oura personal_info fetch failed during connect for apple_user=%s: %s. "
            "Storing token without oura_user_id; sync will backfill later.",
            resolved_user_id[:12] + "...", e,
        )

    # Delete any existing Oura token for this user before inserting
    existing = await db.execute(select(OuraToken).where(OuraToken.user_id == resolved_user_id))
    for t in existing.scalars():
        await db.delete(t)

    oura_token = OuraToken(
        user_id=resolved_user_id,
        oura_user_id=oura_user_id,
        access_token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token", ""),
        expires_at=utcnow_naive() + timedelta(seconds=token_data.get("expires_in", 86400)),
    )
    db.add(oura_token)
    await db.commit()

    return RedirectResponse(url=_OURA_SUCCESS_DEEPLINK)
