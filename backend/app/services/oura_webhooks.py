"""Oura Webhook Management.

Registers and handles webhook subscriptions for near-real-time data delivery.
Oura pushes data to our callback URL when new sleep/readiness/activity data
is available — eliminating the need for frequent polling.

API: https://api.ouraring.com/v2/webhook/subscription
Auth: x-client-id + x-client-secret headers (app-level credentials)

Event types: create, update, delete
Data types: daily_sleep, daily_readiness, daily_activity, daily_spo2,
            daily_stress, workout, sleep, session, vo2_max, etc.
"""

import logging

import httpx

from app.config import settings
from app.core.http import DEFAULT_TIMEOUT

logger = logging.getLogger("meld.oura_webhooks")

OURA_WEBHOOK_URL = "https://api.ouraring.com/v2/webhook/subscription"

# Verification token — Oura sends this back during the GET handshake to
# prove the registrar owns this endpoint. Source-of-truth is the
# OURA_WEBHOOK_VERIFICATION_TOKEN env var (Settings.oura_webhook_verification_token).
# Empty = handshake rejected (dev default). The 2026-04-29 audit found a
# hardcoded "meld-oura-verify-2026" value in source; that string is now
# revoked and re-registration is required to install a fresh per-env token.
def _verification_token() -> str:
    return settings.oura_webhook_verification_token

# Data types we want to subscribe to
SUBSCRIBE_DATA_TYPES = [
    "daily_sleep",
    "daily_readiness",
    "daily_activity",
    "daily_spo2",
    "daily_stress",
    "workout",
]


def _webhook_headers() -> dict:
    """Build auth headers for Oura webhook API."""
    return {
        "x-client-id": settings.oura_client_id,
        "x-client-secret": settings.oura_client_secret,
        "Content-Type": "application/json",
    }


async def list_subscriptions() -> list[dict]:
    """List all active webhook subscriptions."""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.get(
            OURA_WEBHOOK_URL,
            headers=_webhook_headers(),
        )
        response.raise_for_status()
        return response.json()


async def create_subscription(
    callback_url: str,
    data_type: str,
    event_type: str = "create",
) -> dict:
    """Create a webhook subscription for a data type.

    Args:
        callback_url: URL where Oura will POST events
        data_type: One of SUBSCRIBE_DATA_TYPES
        event_type: "create", "update", or "delete"

    Returns:
        Subscription details including id and expiration_time
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            OURA_WEBHOOK_URL,
            headers=_webhook_headers(),
            json={
                "callback_url": callback_url,
                "verification_token": _verification_token(),
                "event_type": event_type,
                "data_type": data_type,
            },
        )
        response.raise_for_status()
        result = response.json()
        logger.info("Created Oura webhook: %s for %s", result.get("id"), data_type)
        return result


async def delete_subscription(subscription_id: str) -> bool:
    """Delete a webhook subscription."""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.delete(
            f"{OURA_WEBHOOK_URL}/{subscription_id}",
            headers=_webhook_headers(),
        )
        return response.status_code == 204


async def renew_subscription(subscription_id: str) -> dict:
    """Renew a subscription before it expires."""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.put(
            f"{OURA_WEBHOOK_URL}/renew/{subscription_id}",
            headers=_webhook_headers(),
        )
        response.raise_for_status()
        return response.json()


async def register_all_webhooks(callback_base_url: str) -> list[dict]:
    """Register webhook subscriptions for all data types we care about.

    Should be called once during app setup or deployment.
    """
    callback_url = f"{callback_base_url}/api/webhooks/oura"
    results = []

    for data_type in SUBSCRIBE_DATA_TYPES:
        for event_type in ["create", "update"]:
            try:
                result = await create_subscription(
                    callback_url=callback_url,
                    data_type=data_type,
                    event_type=event_type,
                )
                results.append(result)
            except (httpx.HTTPError, ValueError, KeyError) as e:
                logger.error(
                    "Failed to register webhook for %s/%s: %s",
                    data_type, event_type, e,
                )

    logger.info("Registered %d Oura webhooks", len(results))
    return results
