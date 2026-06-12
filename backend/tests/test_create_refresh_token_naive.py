"""Regression test for MELD-BACKEND-F.

Production Sentry showed `DBAPIError <DataError>: invalid input for query
argument $4 ... can't subtract offset-naive and offset-aware datetimes` on
INSERT INTO refresh_tokens. Root cause: `create_refresh_token()` returned a
TZ-aware UTC datetime for `expires_at`, but `RefreshToken.expires_at` is a
naive `DateTime` column (project convention; SQLite doesn't support
TZ-aware, and we keep Postgres consistent for parity).

Postgres-strict env rejected the bind. SQLite-permissive dev hid it — same
class as the 2026-04-29 boolean-default `sa.text("0/1")` Postgres bug
(see feedback_postgres_dialect_parity.md).

This test pins the naive invariant so a future refactor can't regress.

Run: cd backend && uv run pytest tests/test_create_refresh_token_naive.py -v
"""

import os
from datetime import datetime, timedelta, timezone


os.environ.setdefault("JWT_SECRET_KEY", "test-secret-for-create-refresh-token-tests")

from app.core.security import REFRESH_TOKEN_EXPIRE_DAYS, create_refresh_token


def test_create_refresh_token_returns_tuple_of_three():
    raw, hashed_id, expires_at = create_refresh_token()
    assert isinstance(raw, str) and raw
    assert isinstance(hashed_id, str) and hashed_id
    assert isinstance(expires_at, datetime)


def test_create_refresh_token_expires_at_is_naive():
    """The headline regression: expires_at MUST be naive (tzinfo is None).

    `RefreshToken.expires_at` is a `TIMESTAMP WITHOUT TIME ZONE` column in
    Postgres. Binding a TZ-aware datetime here is what caused
    MELD-BACKEND-F."""
    _, _, expires_at = create_refresh_token()
    assert expires_at.tzinfo is None, (
        f"create_refresh_token returned tz-aware expires_at ({expires_at.tzinfo}); "
        "RefreshToken.expires_at is a naive DateTime column. "
        "Use utcnow_naive() not datetime.now(timezone.utc)."
    )


def test_create_refresh_token_expires_at_is_30_days_out():
    """Sanity: the expiry is the configured number of days from now."""
    _, _, expires_at = create_refresh_token()
    # Compare naive-to-naive (use modern API, not deprecated datetime.utcnow)
    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    delta = expires_at - now_naive
    # Allow 60s of test/clock slack
    expected = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    assert abs(delta - expected) < timedelta(seconds=60)


def test_create_refresh_token_hashed_id_is_sha256_hex():
    """The hashed_id is a deterministic 64-char SHA256 hex of the raw token."""
    import hashlib
    raw, hashed_id, _ = create_refresh_token()
    assert hashed_id == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert len(hashed_id) == 64


def test_create_refresh_token_returns_unique_tokens_each_call():
    """No accidental shared seed."""
    tokens = {create_refresh_token()[0] for _ in range(20)}
    assert len(tokens) == 20
