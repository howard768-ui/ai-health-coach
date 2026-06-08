"""Webhook receiver endpoints.

Handles incoming webhooks from Oura (and future data sources).
When Oura sends a webhook, we trigger a sync + coaching notification.
"""

import asyncio
import json as _json
import logging
from datetime import datetime

import httpx
from fastapi import APIRouter, Request, Query, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db
from app.services.oura_sync import sync_user_data
from app.services.oura_webhooks import (
    _verification_token,
    register_all_webhooks,
    list_subscriptions,
)

from app.api.deps import CurrentAdminUser
from app.core.constants import TEST_DEVICE_TOKEN
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.webhooks")

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


@router.get("/oura")
async def oura_webhook_verification(
    verification_token: str = Query(None),
    challenge: str = Query(None),
):
    """Oura webhook verification handshake.

    Oura sends GET with verification_token + challenge. We verify the token
    matches our env-configured value, then echo back the challenge.

    Both sides must be non-empty. If our configured token is empty (dev
    default), the handshake is rejected so an attacker can't replay an
    empty-token request and get our challenge endpoint to echo arbitrary
    bytes back. Use timing-safe compare to avoid token-length leaks.
    """
    import hmac

    expected = _verification_token()
    if (
        not expected
        or not verification_token
        or not challenge
        or not hmac.compare_digest(verification_token, expected)
    ):
        # Logs presence booleans only, never the token or challenge value.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        logger.warning(
            "Oura webhook verification failed: token_present=%s challenge_present=%s expected_configured=%s",
            bool(verification_token),
            bool(challenge),
            bool(expected),
        )
        return {"error": "Invalid verification"}
    logger.info("Oura webhook verification successful, challenge=%s", challenge)
    return {"challenge": challenge}


@router.post("/oura")
async def oura_webhook_receiver(request: Request, db: AsyncSession = Depends(get_db)):
    """Receive Oura webhook events.

    Oura POSTs here when new data is available (sleep, readiness, activity, etc.).
    Oura does not send bearer auth, so we validate via:
      1. Body shape — must have `event_type`, `data_type`, and `user_id`
      2. The handler does NOT trust webhook body content for health data — it
         re-fetches everything from Oura's API using our OAuth token. A forged
         webhook can only cause an unnecessary sync, not poison data.
      3. Per-user-per-minute throttle — caps the damage of a webhook flood.

    TODO (multi-user): add `oura_user_id` column to OuraToken and match on
    `body["user_id"]` to route events to the correct Meld user.
    """
    try:
        body = await request.json()
    except (_json.JSONDecodeError, ValueError, UnicodeDecodeError):
        logger.warning("Oura webhook received invalid JSON")
        return {"status": "invalid_payload"}

    event_type = body.get("event_type")
    data_type = body.get("data_type")
    user_oura_id = body.get("user_id")
    timestamp = body.get("timestamp")

    # Body shape validation — Oura always sends these fields
    if not event_type or not data_type or not user_oura_id:
        logger.warning(
            "Oura webhook missing required fields: event=%s data=%s user=%s",
            event_type, data_type, user_oura_id,
        )
        return {"status": "invalid_payload"}

    logger.info(
        "Oura webhook: event=%s data=%s user=%s time=%s",
        event_type, data_type, user_oura_id, timestamp,
    )

    # Per-user throttle: at most 1 sync per 60s regardless of webhook count.
    # Prevents a webhook flood from exhausting Oura API quota or Claude budget.
    # In-memory store is fine for single-instance Railway deploy.
    from datetime import datetime, timedelta
    if not hasattr(oura_webhook_receiver, "_last_sync_at"):
        oura_webhook_receiver._last_sync_at = {}  # type: ignore[attr-defined]
    last_sync_map = oura_webhook_receiver._last_sync_at  # type: ignore[attr-defined]
    now = utcnow_naive()
    window = timedelta(seconds=60)
    # Evict entries older than the throttle window. The key is the
    # attacker-supplied Oura user_id; without eviction this map grows unbounded
    # (memory-growth DoS). Entries past the window carry no throttle meaning, so
    # pruning them bounds the map to users seen within the last 60s.
    cutoff = now - window
    stale = [k for k, ts in last_sync_map.items() if ts < cutoff]
    for k in stale:
        del last_sync_map[k]
    last = last_sync_map.get(user_oura_id)
    if last and (now - last) < window:
        logger.info("Oura webhook throttled (last sync %ds ago)", (now - last).seconds)
        return {"status": "throttled"}
    last_sync_map[user_oura_id] = now

    # MEL-45 part 2: route by oura_user_id match. The column was added in
    # MEL-45 part 1 (PR #102) and is populated on connect via personal_info
    # (or by the next sync after deploy if connect happened before the column
    # existed). Routing logic:
    #
    # 1. Look up `OuraToken WHERE oura_user_id == body.user_id`. If found,
    #    route to that token's apple_user_id. This is the correct multi-user
    #    behavior.
    # 2. If no match AND only one OuraToken row exists in the entire system,
    #    fall back to that row. Single-user transition window: the legacy
    #    user's token has NULL `oura_user_id` until backfilled, so the lookup
    #    won't match. The fallback keeps webhooks working during the gap.
    # 3. If no match AND multiple tokens exist, return 200 + Sentry capture.
    #    Don't 4xx — Apple's retry budget is scarce, and a missing user is a
    #    legitimate state (we may have just deleted them locally).
    from app.models.health import OuraToken
    from sqlalchemy import select, func as sql_func

    # Step 1: exact match on oura_user_id
    matched = await db.execute(
        select(OuraToken).where(OuraToken.oura_user_id == user_oura_id).limit(1)
    )
    token = matched.scalar_one_or_none()

    # Step 2: single-user fallback (transition window)
    if token is None:
        count_result = await db.execute(select(sql_func.count(OuraToken.id)))
        total_tokens = count_result.scalar() or 0
        if total_tokens == 0:
            logger.warning("Oura webhook received but no users have connected Oura — ignoring")
            return {"status": "no_user"}
        if total_tokens == 1:
            single = await db.execute(select(OuraToken).limit(1))
            token = single.scalar_one_or_none()
            if token is not None:
                logger.info(
                    "Oura webhook routed via single-user fallback for body.user_id=%s "
                    "(token has NULL oura_user_id; will backfill on next sync)",
                    user_oura_id,
                )

    # Step 3: still no match in multi-user mode — log + Sentry, return 200
    if token is None:
        logger.warning(
            "Oura webhook user_id=%s did not match any OuraToken.oura_user_id; "
            "no single-user fallback available (multi-user mode). Ignoring.",
            user_oura_id,
        )
        try:
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("oura_action", "webhook_unrouted")
                scope.set_extra("oura_user_id_received", user_oura_id)
                sentry_sdk.capture_message(
                    "Oura webhook arrived for unmatched oura_user_id",
                    level="warning",
                )
        except Exception:  # noqa: BLE001 -- never let Sentry crash the webhook handler
            logger.debug("Sentry capture failed (non-fatal)", exc_info=True)
        return {"status": "no_match"}

    meld_user_id = token.user_id

    # Trigger sync to pull latest data. Return 200 even on sync failure so
    # Oura doesn't retry — their retry budget is scarce and a failed sync
    # will be picked up by the scheduled job within 6 hours. P2-16.
    try:
        result = await sync_user_data(db, meld_user_id)
        logger.info("Webhook-triggered sync: %s", result)
    except (httpx.HTTPError, SQLAlchemyError, ValueError, KeyError) as e:
        logger.error("Webhook sync failed (returning 200 to Oura): %s", e)
        return {"status": "sync_failed", "error": str(e)}

    # If this is a readiness update, trigger the morning brief notification
    if data_type == "daily_readiness" and event_type in ("create", "update"):
        try:
            from app.services.notification_engine import notification_engine
            from app.services.apns import apns_client
            from app.models.notification import DeviceToken
            from app.models.health import SleepRecord
            from app.models.user import User
            from sqlalchemy import desc

            # Load user for personalized greeting
            user_result = await db.execute(
                select(User).where(User.apple_user_id == meld_user_id)
            )
            user = user_result.scalar_one_or_none()
            user_name = (user.name.split()[0] if user and user.name else "there")

            # Get latest health data
            sleep_result = await db.execute(
                select(SleepRecord)
                .where(SleepRecord.user_id == meld_user_id)
                .order_by(desc(SleepRecord.date))
                .limit(1)
            )
            sr = sleep_result.scalar_one_or_none()
            if sr:
                health_data = {
                    "sleep_efficiency": sr.efficiency,
                    "hrv_average": sr.hrv_average,
                    "resting_hr": sr.resting_hr,
                    "readiness_score": sr.readiness_score,
                    "total_sleep_hours": (sr.total_sleep_seconds or 0) / 3600,
                }

                # Generate morning brief. Offload the synchronous Claude SDK
                # call so it does not block the event loop for the 10-30s of the
                # call while the webhook handler is running. Audit P2a.
                content = await asyncio.to_thread(
                    notification_engine.generate_morning_brief,
                    health_data,
                    user_name=user_name,
                )

                # Send push notification
                token_result = await db.execute(
                    select(DeviceToken).where(
                        DeviceToken.user_id == meld_user_id,
                        DeviceToken.is_active == True,
                        DeviceToken.token != TEST_DEVICE_TOKEN,
                    )
                )
                for token_row in token_result.scalars().all():
                    await apns_client.send_push(
                        device_token=token_row.token,
                        title=content["title"],
                        body=content["body"],
                        category=content["apns"]["category"],
                        thread_id=content["apns"]["thread_id"],
                        interruption_level="time-sensitive",
                        relevance_score=1.0,
                        data=content["data"],
                        media_url=content.get("media_url"),
                    )
                    logger.info("Webhook-triggered morning brief sent")

        except (httpx.HTTPError, SQLAlchemyError, KeyError, ValueError) as e:
            logger.error("Failed to send webhook-triggered notification: %s", e)

    return {"status": "ok"}


# ── Admin Endpoints ─────────────────────────────────────────

@router.post("/oura/register")
async def register_webhooks(
    current_user: CurrentAdminUser,
    base_url: str = Query(...),
):
    """Register all Oura webhook subscriptions.

    Call once with your public backend URL:
    POST /api/webhooks/oura/register?base_url=https://your-domain.com

    Auth-gated to prevent attackers redirecting your webhooks to their servers (P1-3).
    """
    # Validate base_url's host is one of our known hosts. Prevents SSRF —
    # the prior `base_url.startswith("http://localhost")` accepted
    # `http://localhost.attacker.com/` and let an attacker register webhooks
    # pointing at their server. Use proper URL parsing + exact host match.
    from urllib.parse import urlparse
    from app.config import settings as _settings
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="base_url malformed")
    allowed_hosts = {"localhost"}
    pub = urlparse(_settings.public_base_url)
    if pub.hostname:
        allowed_hosts.add(pub.hostname)
    host = parsed.hostname.lower()
    is_lan = host.startswith("192.168.") or host.startswith("10.") or host == "127.0.0.1"
    if host not in allowed_hosts and not is_lan:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"base_url host must be one of {sorted(allowed_hosts)} or LAN",
        )

    results = await register_all_webhooks(base_url)
    return {"status": "ok", "subscriptions": len(results), "details": results}


@router.get("/oura/subscriptions")
async def get_subscriptions(current_user: CurrentAdminUser):
    """List all active Oura webhook subscriptions (admin endpoint)."""
    try:
        subs = await list_subscriptions()
        return {"subscriptions": subs}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"error": str(e)}
