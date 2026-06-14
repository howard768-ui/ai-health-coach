"""Sign in with Apple — token verification + client secret generation.

Design decisions (see ultraplan):
- Verify Apple's identity token using `PyJWKClient` which fetches and caches
  JWKS from https://appleid.apple.com/auth/keys with automatic `kid` rotation.
- Explicit `algorithms=["RS256"]` — Apple always signs with RS256.
- Audience must equal our iOS bundle ID (NOT Team ID; NOT a web Services ID).
- Issuer must equal `https://appleid.apple.com` exactly.
- Use `claims["sub"]` as the source of truth for user identity. The
  `user` field the iOS client sends is untrusted input.
- Verify nonce: `sha256(raw_nonce) == claims["nonce"]`. The client sent us
  the raw nonce; Apple only sees the SHA256 hash.
- Detect Apple's private relay emails by suffix.

References:
- Apple: https://developer.apple.com/documentation/signinwithapplerestapi/generate-and-validate-tokens
- TN3194 (account deletion): https://developer.apple.com/documentation/technotes/tn3194-handling-account-deletions-and-revoking-tokens-for-sign-in-with-apple
- Audience must be bundle ID: https://developer.apple.com/forums/thread/663461
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import httpx
import jwt
from jwt import PyJWKClient

from app.config import settings
from app.core.http import DEFAULT_TIMEOUT

logger = logging.getLogger("meld.apple")

# ── Constants ────────────────────────────────────────────────────────────────

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"
APPLE_PRIVATE_RELAY_SUFFIX = "@privaterelay.appleid.com"

# PyJWKClient caches internally; instantiate once at module import.
_jwks_client = PyJWKClient(APPLE_JWKS_URL, cache_keys=True, max_cached_keys=16)


# ── Identity Token Verification ──────────────────────────────────────────────


def verify_apple_identity_token(identity_token: str, raw_nonce: str | None) -> dict:
    """Verify an Apple-issued identity token and return the validated claims.

    Args:
        identity_token: The JWT string from `ASAuthorizationAppleIDCredential.identityToken`.
        raw_nonce: The raw nonce the iOS client generated (the one whose SHA256
            was sent to Apple in the `request.nonce` field). May be None if the
            client did not send a nonce (not recommended but supported for
            backwards compatibility during dev).

    Raises:
        jwt.InvalidTokenError: if signature, audience, issuer, expiry, or nonce
            verification fails.

    Returns:
        dict of verified JWT claims. `claims["sub"]` is the stable Apple user
        identifier — use this as the source of truth for user identity.
    """
    signing_key = _jwks_client.get_signing_key_from_jwt(identity_token)
    claims = jwt.decode(
        identity_token,
        signing_key.key,
        algorithms=["RS256"],  # explicit — blocks algorithm confusion
        audience=settings.apple_bundle_id,
        issuer=APPLE_ISSUER,
        options={"require": ["exp", "iat", "sub", "aud", "iss"]},
    )

    # Nonce verification (manual — PyJWT doesn't validate custom claims).
    # Apple embeds the SHA256 hash of whatever we sent as `request.nonce`.
    if raw_nonce is not None:
        expected = hashlib.sha256(raw_nonce.encode("utf-8")).hexdigest()
        if claims.get("nonce") != expected:
            raise jwt.InvalidTokenError("Apple identity token nonce mismatch")

    return claims


def verify_apple_server_notification(signed_payload: str) -> dict:
    """Verify an Apple server-to-server notification and return its event payload.

    Apple POSTs ``{"payload": "<JWT>"}`` to /auth/apple/revoked when a user
    revokes Sign in with Apple, deletes their account, or toggles email-relay
    forwarding. The outer JWT is RS256-signed by Apple (same JWKS as identity
    tokens) and contains an ``events`` claim that is *itself* a JWT, also
    RS256-signed by Apple. Both must be verified before any action is taken.

    Reference:
    https://developer.apple.com/documentation/technotes/tn3194-handling-account-deletions-and-revoking-tokens-for-sign-in-with-apple

    Args:
        signed_payload: The JWT string from the request body's ``payload`` field.

    Raises:
        jwt.InvalidTokenError: if signature, audience, issuer, or expiry fails
            on either the outer or inner token.

    Returns:
        Dict of the inner events JWT claims:
            - ``type``: "consent-revoked" | "email-disabled" | "email-enabled"
              | "account-delete"
            - ``sub``: Apple user ID (stable, use as source of truth)
            - ``event_time``: int milliseconds since epoch
            - ``email`` (optional): for email-* events
    """
    # Outer JWT — same signing infrastructure as identity tokens.
    outer_signing_key = _jwks_client.get_signing_key_from_jwt(signed_payload)
    outer_claims = jwt.decode(
        signed_payload,
        outer_signing_key.key,
        algorithms=["RS256"],  # explicit — blocks algorithm confusion
        audience=settings.apple_bundle_id,
        issuer=APPLE_ISSUER,
        options={"require": ["exp", "iat", "aud", "iss", "events"]},
    )

    events_jwt = outer_claims.get("events")
    if not isinstance(events_jwt, str) or not events_jwt:
        raise jwt.InvalidTokenError("Apple server notification missing events JWT")

    # Inner events JWT — same JWKS, same algorithm, same audience/issuer.
    inner_signing_key = _jwks_client.get_signing_key_from_jwt(events_jwt)
    inner_claims = jwt.decode(
        events_jwt,
        inner_signing_key.key,
        algorithms=["RS256"],
        audience=settings.apple_bundle_id,
        issuer=APPLE_ISSUER,
        options={"require": ["aud", "iss", "type", "sub", "event_time"]},
    )
    return inner_claims


def is_private_relay_email(email: str | None) -> bool:
    """Return True if the email is an Apple private relay address.

    These are real and routable via Apple's relay — don't reject them.
    Just flag so we know this is a private email for compliance/display.
    """
    return bool(email) and email.lower().endswith(APPLE_PRIVATE_RELAY_SUFFIX)


# ── Client Secret (ES256 JWT) ────────────────────────────────────────────────


def _load_siwa_private_key() -> str:
    """Load the Sign in with Apple private key for client secret signing.

    Priority:
    1. `SIWA_KEY_CONTENT` env var — raw .p8 contents (production via Railway)
    2. `SIWA_KEY_PATH` env var — on-disk .p8 file (local development)

    Normalizes CRLF and literal ``\\n`` sequences via ``normalize_pem`` —
    same env-var-mangling failure modes as APNs (see core/pem.py).
    """
    from app.core.pem import normalize_pem

    if settings.siwa_key_content:
        return normalize_pem(settings.siwa_key_content)
    if settings.siwa_key_path:
        key_path = Path(settings.siwa_key_path)
        if not key_path.is_absolute():
            # Resolve relative to the backend/ directory
            key_path = Path(__file__).resolve().parent.parent.parent / key_path
        if not key_path.exists():
            raise FileNotFoundError(f"SIWA key not found at {key_path}")
        return normalize_pem(key_path.read_text())
    raise ValueError(
        "Sign in with Apple key not configured — set SIWA_KEY_CONTENT (prod) "
        "or SIWA_KEY_PATH (local)"
    )


def verify_siwa_configured() -> None:
    """Validate SIWA configuration at app startup. Warn-and-continue.

    SIWA is opt-in (only the account-deletion webhook needs it). To distinguish
    "SIWA disabled" from "SIWA partially configured", look at SIWA-specific env
    vars only. APPLE_TEAM_ID and APPLE_BUNDLE_ID are shared with APNs and will
    typically be set even when SIWA is not — counting them as SIWA indicators
    is the bug fixed in PR #88.

    States:
      1. SIWA-specific vars unset (no SIWA_KEY_CONTENT/SIWA_KEY_PATH AND no
         SIWA_KEY_ID): SIWA disabled. Return cleanly, no logs. This is the
         correct production state when only APNs is configured.
      2. Exactly one of SIWA_KEY_CONTENT/SIWA_KEY_PATH/SIWA_KEY_ID set:
         partial within SIWA itself. Log error.
      3. Both SIWA-specific vars set, but APPLE_TEAM_ID or APPLE_BUNDLE_ID
         missing: SIWA can't sign client secrets. Log error.
      4. All four set, PEM malformed: log error.
      5. All four set, PEM valid: clean.

    All error states log instead of raising — coupling deploy success to a
    corrupt env var means a multi-day Apple Developer revoke + recreate
    cycle for any single-shot rotation mistake (Apple .p8 files are
    downloadable exactly once at creation). Account-deletion endpoint
    still raises its own error at first use if config is wrong.
    """
    from app.core.pem import PemConfigError, validate_pem_loads

    has_siwa_key = bool(settings.siwa_key_content or settings.siwa_key_path)
    has_siwa_key_id = bool(settings.siwa_key_id)

    if not has_siwa_key and not has_siwa_key_id:
        return  # SIWA disabled in this env

    if not (has_siwa_key and has_siwa_key_id):
        missing = []
        if not has_siwa_key:
            missing.append("SIWA_KEY_CONTENT or SIWA_KEY_PATH")
        if not has_siwa_key_id:
            missing.append("SIWA_KEY_ID")
        logger.error(
            "SIWA partially configured: missing %s. Set both SIWA_KEY_CONTENT/"
            "SIWA_KEY_PATH and SIWA_KEY_ID together. The account-deletion "
            "webhook will fail until both are set.",
            " and ".join(missing),
        )
        return

    # SIWA-specific vars set: also require shared Apple IDs.
    missing_shared = []
    if not settings.apple_team_id:
        missing_shared.append("APPLE_TEAM_ID")
    if not settings.apple_bundle_id:
        missing_shared.append("APPLE_BUNDLE_ID")
    if missing_shared:
        logger.error(
            "SIWA enabled (key + key_id set) but missing shared Apple config: "
            "%s. The account-deletion webhook will fail until these are set.",
            ", ".join(missing_shared),
        )
        return

    # All four set: verify the PEM actually parses.
    try:
        pem = _load_siwa_private_key()
        validate_pem_loads(pem, label="SIWA")
    except (PemConfigError, ValueError, FileNotFoundError) as e:
        logger.error(
            "SIWA key parse FAILED at startup: %s. "
            "Account deletion flow will fail until SIWA_KEY_CONTENT is fixed.",
            e,
        )


def generate_apple_client_secret() -> str:
    """Generate an ES256 client secret JWT for calls to Apple REST endpoints.

    Per https://developer.apple.com/documentation/signinwithapplerestapi/generate-and-validate-tokens:
      - `iss` = Apple Team ID (10 chars)
      - `iat` = now
      - `exp` = now + up to 6 months (we use 1 hour for defense in depth)
      - `aud` = https://appleid.apple.com
      - `sub` = Apple bundle ID (our client_id)
      - alg = ES256
      - kid = our Sign in with Apple key ID
    """
    if not settings.apple_team_id or not settings.siwa_key_id or not settings.apple_bundle_id:
        raise ValueError(
            "Apple team_id, siwa_key_id, and bundle_id must all be configured"
        )

    private_key = _load_siwa_private_key()
    now = int(time.time())
    headers = {"alg": "ES256", "kid": settings.siwa_key_id}
    payload = {
        "iss": settings.apple_team_id,
        "iat": now,
        "exp": now + 3600,
        "aud": APPLE_ISSUER,
        "sub": settings.apple_bundle_id,
    }
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


# ── Token Revocation (account deletion) ──────────────────────────────────────


async def revoke_apple_token(refresh_token: str) -> None:
    """Call Apple's /auth/revoke endpoint to revoke a user's SIWA session.

    Called during account deletion — required by App Store guideline 5.1.1(v)
    per Apple TN3194.

    Args:
        refresh_token: An Apple-issued refresh token from the original /auth/apple
            exchange. If we never captured one (older accounts), this becomes a
            no-op — the user can manually revoke in Apple ID Settings.
    """
    if not refresh_token:
        return  # No Apple refresh token captured — nothing to revoke server-side

    client_secret = generate_apple_client_secret()
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        response = await client.post(
            APPLE_REVOKE_URL,
            data={
                "client_id": settings.apple_bundle_id,
                "client_secret": client_secret,
                "token": refresh_token,
                "token_type_hint": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10.0,
        )
        # Apple returns 200 on success, 400 on already-revoked. Either is fine.
        if response.status_code not in (200, 400):
            raise RuntimeError(
                f"Apple token revocation failed: {response.status_code} {response.text}"
            )
