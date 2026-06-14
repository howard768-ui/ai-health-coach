"""Tests for Apple JWT verifiers (`verify_apple_identity_token` +
`verify_apple_server_notification`).

These are CRITICAL security paths flagged in the 2026-04-30 audit (MEL-43)
as zero-test:
  - `verify_apple_identity_token` is the sole gate on Sign-in-with-Apple
    login. An algorithm-confusion or audience drift bug = anyone signs in
    as anyone.
  - `verify_apple_server_notification` validates the outer + inner JWT on
    Apple's webhook. A forged Apple event accepted = a malicious actor can
    deactivate any user.

Strategy: generate an RSA keypair locally; sign tokens with the private
key; monkeypatch `_jwks_client.get_signing_key_from_jwt` to return the
public key. Apple's actual JWKS is never hit.

Run: cd backend && uv run pytest tests/test_apple_jwt_verify.py -v
"""

import hashlib
import os
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-apple-jwt-tests")

from app.config import settings
from app.core import apple as apple_mod
from app.core.apple import (
    APPLE_ISSUER,
    verify_apple_identity_token,
    verify_apple_server_notification,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rsa_keypair():
    """Generate one RSA keypair for the whole module — keypair generation is
    expensive (~100ms). Tests don't share state otherwise."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return SimpleNamespace(private=private_pem, public=public_key)


@pytest.fixture(autouse=True)
def patch_jwks(monkeypatch, rsa_keypair):
    """Replace the module-level `_jwks_client` with a fake that returns our
    test public key for any JWT. Bypasses the real Apple JWKS fetch."""
    fake_signing_key = SimpleNamespace(key=rsa_keypair.public)

    class FakeJWKS:
        def get_signing_key_from_jwt(self, _token: str):
            return fake_signing_key

    monkeypatch.setattr(apple_mod, "_jwks_client", FakeJWKS())


@pytest.fixture(autouse=True)
def configured_apple_bundle_id(monkeypatch):
    """All tests assume the bundle ID is set; configure it deterministically."""
    monkeypatch.setattr(settings, "apple_bundle_id", "com.heymeld.app")


def _make_identity_token(
    private_pem: str,
    *,
    sub: str = "001234.deadbeef00000000000000000000.0001",
    aud: str = "com.heymeld.app",
    iss: str = APPLE_ISSUER,
    nonce: str | None = None,
    expires_in_seconds: int = 600,
    iat_offset: int = 0,
    extra: dict | None = None,
) -> str:
    """Build an RS256-signed identity token with arbitrary claims.

    Use `extra={"aud": ...}` etc. to override defaults. None means: drop the
    claim entirely. Use kwargs for clarity in test bodies."""
    now = int(time.time()) + iat_offset
    claims: dict = {
        "iss": iss,
        "sub": sub,
        "aud": aud,
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    if nonce is not None:
        claims["nonce"] = nonce
    if extra:
        claims.update(extra)
    return jwt.encode(claims, private_pem, algorithm="RS256")


# ── verify_apple_identity_token: happy path ───────────────────────────────


def test_identity_token_happy_path(rsa_keypair):
    """Valid token with all required claims and correct nonce → returns claims."""
    raw_nonce = "client-side-random-string"
    nonce_hash = hashlib.sha256(raw_nonce.encode()).hexdigest()
    token = _make_identity_token(rsa_keypair.private, nonce=nonce_hash)
    claims = verify_apple_identity_token(token, raw_nonce)
    assert claims["sub"] == "001234.deadbeef00000000000000000000.0001"
    assert claims["aud"] == "com.heymeld.app"
    assert claims["iss"] == APPLE_ISSUER


def test_identity_token_no_nonce_accepted_when_not_provided(rsa_keypair):
    """Backwards-compat: `raw_nonce=None` skips nonce verification entirely."""
    token = _make_identity_token(rsa_keypair.private)
    claims = verify_apple_identity_token(token, None)
    assert claims["sub"]


# ── verify_apple_identity_token: rejection cases ──────────────────────────


def test_identity_token_wrong_audience_rejected(rsa_keypair):
    """Token signed for a different audience → InvalidTokenError."""
    token = _make_identity_token(rsa_keypair.private, aud="com.attacker.app")
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_identity_token(token, None)


def test_identity_token_wrong_issuer_rejected(rsa_keypair):
    token = _make_identity_token(rsa_keypair.private, iss="https://attacker.example")
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_identity_token(token, None)


def test_identity_token_expired_rejected(rsa_keypair):
    """Token with exp in the past → InvalidTokenError."""
    token = _make_identity_token(rsa_keypair.private, expires_in_seconds=-60, iat_offset=-3600)
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_identity_token(token, None)


def test_identity_token_missing_sub_rejected(rsa_keypair):
    """Token without `sub` claim → InvalidTokenError (require list includes it)."""
    now = int(time.time())
    claims = {
        "iss": APPLE_ISSUER,
        "aud": "com.heymeld.app",
        "iat": now,
        "exp": now + 600,
    }  # no `sub`
    token = jwt.encode(claims, rsa_keypair.private, algorithm="RS256")
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_identity_token(token, None)


def test_identity_token_nonce_mismatch_rejected(rsa_keypair):
    """Nonce hash in token doesn't match SHA256(raw_nonce) → reject."""
    token = _make_identity_token(rsa_keypair.private, nonce="not-the-hash-of-anything")
    with pytest.raises(jwt.InvalidTokenError, match="nonce"):
        verify_apple_identity_token(token, "different-raw-nonce")


def test_identity_token_blocks_algorithm_confusion():
    """A token signed with `alg=HS256` must be rejected because our
    verifier passes `algorithms=["RS256"]`. This is the classic
    algorithm-confusion defense — without the `algorithms` kwarg pyjwt
    would allow any algorithm. Use a plain HMAC secret to forge the
    token (pyjwt now blocks using PEM strings as HMAC secrets at the
    encode side, but a real attacker would construct the token raw)."""
    now = int(time.time())
    claims = {
        "iss": APPLE_ISSUER,
        "sub": "attacker",
        "aud": "com.heymeld.app",
        "iat": now,
        "exp": now + 600,
    }
    forged = jwt.encode(claims, "any-shared-secret-string", algorithm="HS256")
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_identity_token(forged, None)


# ── verify_apple_server_notification: happy path ──────────────────────────


def _make_server_notification(
    private_pem: str,
    *,
    event_type: str = "consent-revoked",
    sub: str = "001234.deadbeef00000000000000000000.0001",
    event_time: int | None = None,
    aud: str = "com.heymeld.app",
    iss: str = APPLE_ISSUER,
) -> str:
    """Build the outer + inner nested JWT structure Apple uses for server-to-server
    notifications. Outer JWT contains `events` claim that is itself a JWT."""
    if event_time is None:
        event_time = int(time.time() * 1000)
    now = int(time.time())

    inner_claims = {
        "iss": iss,
        "aud": aud,
        "iat": now,
        "exp": now + 600,
        "type": event_type,
        "sub": sub,
        "event_time": event_time,
    }
    inner_jwt = jwt.encode(inner_claims, private_pem, algorithm="RS256")

    outer_claims = {
        "iss": iss,
        "aud": aud,
        "iat": now,
        "exp": now + 600,
        "events": inner_jwt,
    }
    return jwt.encode(outer_claims, private_pem, algorithm="RS256")


def test_server_notification_happy_path_consent_revoked(rsa_keypair):
    payload = _make_server_notification(rsa_keypair.private, event_type="consent-revoked")
    inner = verify_apple_server_notification(payload)
    assert inner["type"] == "consent-revoked"
    assert inner["sub"]
    assert "event_time" in inner


def test_server_notification_happy_path_account_delete(rsa_keypair):
    payload = _make_server_notification(rsa_keypair.private, event_type="account-delete")
    inner = verify_apple_server_notification(payload)
    assert inner["type"] == "account-delete"


# ── verify_apple_server_notification: rejection cases ─────────────────────


def test_server_notification_outer_wrong_audience_rejected(rsa_keypair):
    payload = _make_server_notification(rsa_keypair.private, aud="com.attacker.app")
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_server_notification(payload)


def test_server_notification_outer_wrong_issuer_rejected(rsa_keypair):
    payload = _make_server_notification(rsa_keypair.private, iss="https://forgery.example")
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_server_notification(payload)


def test_server_notification_missing_events_claim_rejected(rsa_keypair):
    """If outer JWT verifies but lacks `events` claim → reject."""
    now = int(time.time())
    outer = jwt.encode(
        {
            "iss": APPLE_ISSUER,
            "aud": "com.heymeld.app",
            "iat": now,
            "exp": now + 600,
        },  # no `events`
        rsa_keypair.private,
        algorithm="RS256",
    )
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_server_notification(outer)


def test_server_notification_inner_jwt_bad_audience_rejected(rsa_keypair):
    """Outer JWT verifies but inner JWT has wrong audience → reject."""
    now = int(time.time())
    inner = jwt.encode(
        {
            "iss": APPLE_ISSUER,
            "aud": "com.attacker.app",  # wrong
            "iat": now,
            "exp": now + 600,
            "type": "consent-revoked",
            "sub": "001234.deadbeef00000000000000000000.0001",
            "event_time": int(time.time() * 1000),
        },
        rsa_keypair.private,
        algorithm="RS256",
    )
    outer = jwt.encode(
        {
            "iss": APPLE_ISSUER,
            "aud": "com.heymeld.app",
            "iat": now,
            "exp": now + 600,
            "events": inner,
        },
        rsa_keypair.private,
        algorithm="RS256",
    )
    with pytest.raises(jwt.InvalidTokenError):
        verify_apple_server_notification(outer)
