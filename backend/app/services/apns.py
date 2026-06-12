"""APNs client for sending push notifications via Apple's HTTP/2 API.

Uses JWT token-based authentication with a .p8 key.
"""

import json
import time
import logging
from pathlib import Path

import httpx
import jwt  # pyjwt — supports ES256 with the cryptography backend

from app.config import settings
from app.core.http import DEFAULT_TIMEOUT
from app.core.pem import PemConfigError, normalize_pem, validate_pem_loads

logger = logging.getLogger(__name__)

APNS_SANDBOX = "https://api.sandbox.push.apple.com"
APNS_PRODUCTION = "https://api.push.apple.com"


class APNsClient:
    """Sends push notifications to iOS devices via Apple Push Notification service."""

    def __init__(self):
        self._jwt_token: str | None = None
        self._jwt_issued_at: float = 0
        self._private_key: str | None = None

    def _reset_cache(self) -> None:
        """Drop all cached state so the next call re-reads from env.

        Both the parsed PEM and the JWT signed with it must be cleared
        together — clearing only the PEM (as `verify_apns_configured`
        previously did) left the cached JWT pointing at the now-evicted
        key, so push sends used a stale token until the 50-minute JWT
        rotation. After-this-PR, callers always invalidate as a pair.
        """
        self._private_key = None
        self._jwt_token = None
        self._jwt_issued_at = 0

    def _load_private_key(self) -> str:
        """Load the APNs .p8 private key.

        Priority:
        1. `APNS_KEY_CONTENT` env var — raw .p8 contents (production via Railway)
        2. `APNS_KEY_PATH` env var — on-disk .p8 file (local development)

        The image must NOT bake the .p8 file in. `.dockerignore` excludes `keys/`.

        Normalizes CRLF and literal ``\\n`` sequences, ensures trailing
        newline. The 2026-04-29 audit (scheduler_audit.md) traced six
        recurring ``JWSError: MalformedFraming`` Sentry issues to env-var
        mangling that survived the old single ``.replace("\\\\n", "\\n")``
        call and got cached forever.
        """
        if self._private_key:
            return self._private_key

        # Production path: key injected via env var
        if settings.apns_key_content:
            self._private_key = normalize_pem(settings.apns_key_content)
            return self._private_key

        # Local dev path: read from on-disk file
        if settings.apns_key_path:
            key_path = Path(settings.apns_key_path)
            if not key_path.exists():
                raise FileNotFoundError(f"APNs .p8 key not found at {key_path}")
            self._private_key = normalize_pem(key_path.read_text())
            return self._private_key

        raise ValueError(
            "APNs key not configured — set APNS_KEY_CONTENT (production) "
            "or APNS_KEY_PATH (local dev)"
        )

    def _get_jwt_token(self) -> str:
        """Generate or return cached JWT. Refresh hourly per Apple requirement."""
        now = time.time()
        # Refresh if older than 50 minutes (buffer before 60-min expiry)
        if self._jwt_token and (now - self._jwt_issued_at) < 3000:
            return self._jwt_token

        private_key = self._load_private_key()
        headers = {
            "alg": "ES256",
            "kid": settings.apns_key_id,
        }
        payload = {
            "iss": settings.apns_team_id,
            "iat": int(now),
        }
        self._jwt_token = jwt.encode(payload, private_key, algorithm="ES256", headers=headers)
        self._jwt_issued_at = now
        return self._jwt_token

    @property
    def _base_url(self) -> str:
        if settings.apns_environment == "production":
            return APNS_PRODUCTION
        return APNS_SANDBOX

    async def send_push(
        self,
        device_token: str,
        title: str,
        body: str,
        *,
        subtitle: str | None = None,
        category: str | None = None,
        thread_id: str | None = None,
        interruption_level: str = "active",
        relevance_score: float = 0.5,
        collapse_id: str | None = None,
        badge: int | None = None,
        sound: str = "default",
        data: dict | None = None,
        media_url: str | None = None,
    ) -> dict:
        """Send a push notification via APNs.

        Returns dict with success status and APNs response.
        """
        # Build APNs payload
        alert = {"title": title, "body": body}
        if subtitle:
            alert["subtitle"] = subtitle

        aps = {
            "alert": alert,
            "sound": sound,
            "interruption-level": interruption_level,
            "relevance-score": relevance_score,
        }
        if category:
            aps["category"] = category
        if thread_id:
            aps["thread-id"] = thread_id
        if badge is not None:
            aps["badge"] = badge
        if media_url:
            aps["mutable-content"] = 1

        payload = {"aps": aps}
        if data:
            payload.update(data)
        if media_url:
            payload["media_url"] = media_url

        # Build request
        url = f"{self._base_url}/3/device/{device_token}"
        token = self._get_jwt_token()
        headers = {
            "authorization": f"bearer {token}",
            "apns-topic": settings.apns_bundle_id,
            "apns-push-type": "alert",
            "apns-priority": "10" if interruption_level == "time-sensitive" else "5",
        }
        if collapse_id:
            headers["apns-collapse-id"] = collapse_id
        if interruption_level == "time-sensitive":
            headers["apns-priority"] = "10"

        payload_bytes = json.dumps(payload).encode("utf-8")
        if len(payload_bytes) > 4096:
            logger.warning("APNs payload exceeds 4KB limit: %d bytes", len(payload_bytes))

        try:
            async with httpx.AsyncClient(http2=True, timeout=DEFAULT_TIMEOUT) as client:
                response = await client.post(url, content=payload_bytes, headers=headers)

            if response.status_code == 200:
                logger.info("Push sent to %s...%s", device_token[:8], device_token[-4:])
                return {"success": True, "status": 200}
            else:
                error_body = response.text
                logger.error(
                    "APNs error %d for %s...%s: %s",
                    response.status_code,
                    device_token[:8],
                    device_token[-4:],
                    error_body,
                )
                return {
                    "success": False,
                    "status": response.status_code,
                    "error": error_body,
                }

        except httpx.HTTPError as e:
            # APNs returns structured errors as response bodies (handled above).
            # This catch is for transport-level failures: DNS, TLS, network,
            # timeout, and HTTP/2 frame errors.
            logger.error("APNs request failed: %s", str(e))
            return {"success": False, "status": 0, "error": str(e)}


# Singleton
apns_client = APNsClient()


def verify_apns_configured() -> None:
    """Validate APNs key at startup; log + disable push if malformed.

    Loads the key (which normalizes line endings) and asks `cryptography`
    to actually parse it. Three outcomes:

      - Key not configured -> info log, return. Push disabled.
      - Key configured + parses -> info log, return. Push enabled.
      - Key configured + fails to parse -> ERROR log + drop cached
        singleton + return WITHOUT raising. App boots; push will fail
        loudly on every send_push() with the same error, and Sentry
        will see exactly one ERROR log per startup attribution.

    Why warn-and-continue instead of fail-closed:
      The 2026-04-30 incident demonstrated that fail-closed PEM verify
      blocks every deploy until APNS_KEY_CONTENT is fixed. Apple's
      .p8 file is downloadable only once at creation, so a corrupt env
      var means a multi-day Apple-Developer dance to revoke + recreate
      the key. Coupling deploy success to that timeline is wrong.
      With warn-and-continue: the app keeps shipping non-APNs fixes
      while push notifications stay disabled in a known + visible way.

    Re-strict to fail-closed in a future PR once the env-var hygiene
    has a separate startup-friendly automated check (or Pro-plan
    secret rotation tooling).
    """
    if not (settings.apns_key_content or settings.apns_key_path):
        logger.info("APNs not configured (push disabled) — skipping verify")
        return
    try:
        # Inside the try: _load_private_key raises FileNotFoundError when
        # APNS_KEY_PATH points at a missing file (any fresh checkout without
        # the .p8, every CI runner). Outside the try it crashed the whole
        # app at startup, violating this function's own warn-and-continue
        # contract. SIWA's twin (core/apple.py verify_siwa_configured) had
        # it right; this now matches.
        pem = apns_client._load_private_key()
        validate_pem_loads(pem, label="APNs")
    except (PemConfigError, ValueError, FileNotFoundError) as e:
        # Drop the cached singleton so a subsequent fix + retry actually
        # re-reads the env. Both the PEM AND the JWT signed with it must
        # be cleared together (PR E hygiene fix). Do NOT raise — let the
        # app boot.
        apns_client._reset_cache()
        logger.error(
            "APNs key parse FAILED at startup: %s. Push notifications will "
            "fail for every send. Set APNS_KEY_CONTENT to the full .p8 file "
            "contents (including BEGIN/END markers) to enable.",
            e,
        )
