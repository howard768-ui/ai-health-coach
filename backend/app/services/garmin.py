"""Garmin Connect API client.

Uses the garminconnect Python library (SSO-based authentication).
The library is synchronous — wrap calls in asyncio.to_thread() for async backend.
"""

import logging
import asyncio
from datetime import date

logger = logging.getLogger("meld.garmin")

# Errors that should be treated as recoverable data-fetch failures:
# library-specific auth/connection errors + generic network errors.
# Anything else bubbles up so we can see programming bugs.
try:
    from garminconnect import (
        GarminConnectAuthenticationError,
        GarminConnectConnectionError,
        GarminConnectTooManyRequestsError,
    )
    _GARMIN_FETCH_ERRORS: tuple[type[BaseException], ...] = (
        GarminConnectAuthenticationError,
        GarminConnectConnectionError,
        GarminConnectTooManyRequestsError,
        ConnectionError,
        TimeoutError,
        OSError,
    )
except ImportError:
    # Library not installed — import errors handled at call sites.
    _GARMIN_FETCH_ERRORS = (ConnectionError, TimeoutError, OSError)


class GarminClient:
    """Wraps the garminconnect library for async usage."""

    def __init__(self):
        self._client = None

    async def login(self, username: str, password: str) -> dict:
        """Authenticate with Garmin Connect via SSO.

        Returns a dict including `session_data` — a serialized garth session
        (OAuth1 + OAuth2 tokens). Store THIS, not the password. Restore via
        `login_from_session()` for subsequent API calls.
        """
        try:
            from garminconnect import Garmin
        except ImportError:
            logger.error("garminconnect library not installed")
            raise RuntimeError("garminconnect package not installed. Run: uv add garminconnect")

        def _login():
            client = Garmin(username, password)
            client.login()
            return client

        self._client = await asyncio.to_thread(_login)

        # Serialize the garth session so we don't need the password on refresh.
        # garth.dumps() returns a JSON string containing OAuth1 + OAuth2 tokens.
        session_data: str | None = None
        try:
            session_data = await asyncio.to_thread(self._client.garth.dumps)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("Failed to serialize garth session: %s", e)
            try:
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("garmin_action", "session_serialize")
                    sentry_sdk.capture_exception(e)
            except Exception:  # noqa: BLE001 -- login flow continues
                logger.debug("Sentry capture failed (non-fatal)", exc_info=True)

        return {
            "status": "connected",
            "display_name": self._client.display_name,
            "session_data": session_data,
        }

    async def login_from_session(self, session_data: str) -> bool:
        """Restore a previously-serialized garth session."""
        try:
            from garminconnect import Garmin
        except ImportError:
            raise RuntimeError("garminconnect package not installed")

        def _restore():
            # Empty-string credentials — garth.loads() supplies the tokens.
            client = Garmin()
            client.garth.loads(session_data)
            return client

        try:
            self._client = await asyncio.to_thread(_restore)
            return True
        except _GARMIN_FETCH_ERRORS as e:
            logger.error("Garmin session restore failed: %s", e)
            return False
        except (ValueError, TypeError, KeyError) as e:
            # Bad session_data payload (corrupted or wrong format).
            logger.error("Garmin session data invalid: %s", e)
            return False

    async def get_steps(self, target_date: date) -> dict | None:
        """Get daily step count."""
        if not self._client:
            return None
        try:
            return await asyncio.to_thread(
                self._client.get_steps_data, target_date.isoformat()
            )
        except _GARMIN_FETCH_ERRORS as e:
            logger.error("Garmin steps error: %s", e)
            return None

    async def get_heart_rate(self, target_date: date) -> dict | None:
        """Get daily heart rate data."""
        if not self._client:
            return None
        try:
            return await asyncio.to_thread(
                self._client.get_heart_rates, target_date.isoformat()
            )
        except _GARMIN_FETCH_ERRORS as e:
            logger.error("Garmin heart rate error: %s", e)
            return None

    async def get_sleep(self, target_date: date) -> dict | None:
        """Get sleep data."""
        if not self._client:
            return None
        try:
            return await asyncio.to_thread(
                self._client.get_sleep_data, target_date.isoformat()
            )
        except _GARMIN_FETCH_ERRORS as e:
            logger.error("Garmin sleep error: %s", e)
            return None

    async def get_stress(self, target_date: date) -> dict | None:
        """Get stress data."""
        if not self._client:
            return None
        try:
            return await asyncio.to_thread(
                self._client.get_stress_data, target_date.isoformat()
            )
        except _GARMIN_FETCH_ERRORS as e:
            logger.error("Garmin stress error: %s", e)
            return None

    async def get_body_battery(self, target_date: date) -> dict | None:
        """Get body battery data."""
        if not self._client:
            return None
        try:
            return await asyncio.to_thread(
                self._client.get_body_battery, target_date.isoformat()
            )
        except _GARMIN_FETCH_ERRORS as e:
            logger.error("Garmin body battery error: %s", e)
            return None

    async def get_activities(self, start: int = 0, limit: int = 20) -> list:
        """Get recent activities/workouts."""
        if not self._client:
            return []
        try:
            return await asyncio.to_thread(
                self._client.get_activities, start, limit
            )
        except _GARMIN_FETCH_ERRORS as e:
            logger.error("Garmin activities error: %s", e)
            return []
