"""Integration tests for the two CRITICAL auth routes flagged as zero-test
in the 2026-04-30 audit (MEL-43):

  - POST /auth/apple/revoked   — Apple server-to-server webhook
  - POST /auth/delete          — App Store guideline 5.1.1(v) account deletion

Both lift the in-memory SQLite + AsyncClient pattern from test_ops.py with
explicit StaticPool so every connection shares the same in-memory database.

Mocks:
  - `verify_apple_server_notification` (the JWT verifier itself is unit-tested
    in test_apple_jwt_verify.py) — these tests focus on the route handler's
    behavior given a known-verified or known-bad outcome.
  - `revoke_apple_token` (the HTTP call to Apple) — these tests don't hit the
    real Apple API; the unit-level shape is exercised in test_apple_jwt_verify.

Run: cd backend && uv run pytest tests/test_auth_routes.py -v
"""

import os

import jwt
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Required env BEFORE app modules are imported (app/config reads at import).
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-auth-route-tests")
os.environ.setdefault(
    "ENCRYPTION_KEY",
    "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=",
)

from app.core.security import create_access_token
from app.database import Base, get_db
from app.main import app
from app.models.user import User
from app.routers import auth_apple as auth_apple_module

# Import the per-user PHI model modules so their tables are registered in
# Base.metadata and created by the test engine. These are the tables that have
# no FK to users (so they are NOT covered by cascade) and must be explicitly
# purged on account deletion (audit A3 / round-1 critical).
from app.models import (  # noqa: E402,F401
    ml_baselines,
    ml_discovery,
    ml_features,
    ml_insights,
    ml_experiments,
    ml_cohorts,
    mascot,
    notification,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def test_engine():
    """Shared in-memory engine. StaticPool ensures every connection sees the
    same in-memory DB (the default behavior would give each connection its
    own fresh memory file, which breaks across the override + verify pair)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    """Session for test setup + post-call verification. The app uses its own
    session per request via the `get_db` override below."""
    SessionMaker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with SessionMaker() as session:
        yield session


@pytest_asyncio.fixture
async def client(test_engine):
    """AsyncClient hitting the FastAPI app with `get_db` rewired to the test
    engine. Cleared after each test so `app.dependency_overrides` doesn't leak
    across tests."""
    SessionMaker = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with SessionMaker() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


def _make_user_kwargs(apple_user_id: str = "001234.deadbeef.0001") -> dict:
    """Minimum-viable kwargs to insert a User in test DB."""
    return {
        "apple_user_id": apple_user_id,
        "email": "test@example.com",
        "name": "Test User",
        "is_active": True,
    }


async def _seed_user(db_session, **kwargs) -> User:
    """Insert a User row + return the persisted instance."""
    fields = _make_user_kwargs() | kwargs
    user = User(**fields)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


# ── POST /auth/apple/revoked ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoked_consent_revoked_marks_user_inactive(client, db_session, monkeypatch):
    """Apple webhook: consent-revoked event → user.is_active = False."""
    user = await _seed_user(db_session, apple_user_id="001234.consent.0001")

    def fake_verify(_payload):
        return {"sub": "001234.consent.0001", "type": "consent-revoked", "event_time": 0}

    monkeypatch.setattr(auth_apple_module, "verify_apple_server_notification", fake_verify)

    resp = await client.post("/auth/apple/revoked", json={"payload": "fake.jwt.string"})
    assert resp.status_code == 200

    # Force re-fetch — the route used a separate session, so db_session's
    # identity map still holds the pre-commit User object cached.
    db_session.expire_all()
    refreshed = (
        await db_session.execute(select(User).where(User.apple_user_id == "001234.consent.0001"))
    ).scalar_one()
    assert refreshed.is_active is False


@pytest.mark.asyncio
async def test_revoked_account_delete_marks_user_inactive(client, db_session, monkeypatch):
    """Apple webhook: account-delete event → user.is_active = False."""
    user = await _seed_user(db_session, apple_user_id="001234.delete.0001")

    def fake_verify(_payload):
        return {"sub": "001234.delete.0001", "type": "account-delete", "event_time": 0}

    monkeypatch.setattr(auth_apple_module, "verify_apple_server_notification", fake_verify)

    resp = await client.post("/auth/apple/revoked", json={"payload": "fake.jwt.string"})
    assert resp.status_code == 200
    db_session.expire_all()
    refreshed = (
        await db_session.execute(select(User).where(User.apple_user_id == "001234.delete.0001"))
    ).scalar_one()
    assert refreshed.is_active is False


@pytest.mark.asyncio
async def test_revoked_email_disabled_does_not_deactivate_user(client, db_session, monkeypatch):
    """email-disabled is informational; user must remain active."""
    user = await _seed_user(db_session, apple_user_id="001234.email.0001")

    def fake_verify(_payload):
        return {"sub": "001234.email.0001", "type": "email-disabled", "event_time": 0}

    monkeypatch.setattr(auth_apple_module, "verify_apple_server_notification", fake_verify)

    resp = await client.post("/auth/apple/revoked", json={"payload": "fake.jwt.string"})
    assert resp.status_code == 200
    db_session.expire_all()
    refreshed = (
        await db_session.execute(select(User).where(User.apple_user_id == "001234.email.0001"))
    ).scalar_one()
    assert refreshed.is_active is True, "email-disabled must NOT deactivate the user"


@pytest.mark.asyncio
async def test_revoked_unknown_user_returns_200_no_op(client, monkeypatch):
    """Apple may notify about a user we've already deleted locally. Always
    return 200 so Apple doesn't retry indefinitely; just no-op."""

    def fake_verify(_payload):
        return {"sub": "001234.never_existed.0000", "type": "consent-revoked", "event_time": 0}

    monkeypatch.setattr(auth_apple_module, "verify_apple_server_notification", fake_verify)

    resp = await client.post("/auth/apple/revoked", json={"payload": "fake.jwt.string"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_revoked_invalid_jwt_returns_400(client, monkeypatch):
    """Forged or malformed JWT → 400 (legitimate Apple retries never produce
    this; it's an attack signal)."""

    def fake_verify(_payload):
        raise jwt.InvalidTokenError("forged signature")

    monkeypatch.setattr(auth_apple_module, "verify_apple_server_notification", fake_verify)

    resp = await client.post("/auth/apple/revoked", json={"payload": "forged.jwt.string"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_revoked_missing_payload_returns_200_invalid(client):
    """Missing `payload` field → 200 with invalid_payload status (Apple
    retry budget is scarce; never 4xx for shape errors)."""
    resp = await client.post("/auth/apple/revoked", json={"not_payload": "x"})
    assert resp.status_code == 200
    assert resp.json().get("status") == "invalid_payload"


@pytest.mark.asyncio
async def test_revoked_malformed_body_returns_200_invalid(client):
    """Non-JSON body → 200 with invalid_body status."""
    resp = await client.post(
        "/auth/apple/revoked",
        content=b"not json at all",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json().get("status") == "invalid_body"


# ── POST /auth/delete ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_account_unauthed_returns_401(client):
    """No Authorization header → 401 (CurrentUser guard fires first)."""
    resp = await client.post("/auth/delete", json={})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_delete_account_authed_user_deletes_self(client, db_session, monkeypatch):
    """Authed user posts to /auth/delete → user row gone from DB,
    revoke_apple_token called once."""
    apple_id = "001234.deletes_self.0001"
    user = await _seed_user(db_session, apple_user_id=apple_id)
    user.apple_refresh_token = "fake-refresh-token"
    await db_session.commit()

    revoke_calls: list = []

    async def fake_revoke(refresh_token: str):
        revoke_calls.append(refresh_token)

    monkeypatch.setattr(auth_apple_module, "revoke_apple_token", fake_revoke)

    token, _ = create_access_token(apple_id)
    resp = await client.post(
        "/auth/delete",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert revoke_calls == ["fake-refresh-token"], "revoke_apple_token must run with the user's refresh token"

    # Verify the user row is gone — expire identity map to force re-fetch.
    db_session.expire_all()
    gone = (
        await db_session.execute(select(User).where(User.apple_user_id == apple_id))
    ).scalar_one_or_none()
    assert gone is None, "user row must be deleted from DB"


@pytest.mark.asyncio
async def test_delete_account_without_apple_refresh_token_skips_revoke(client, db_session, monkeypatch):
    """User who never captured an Apple refresh token (older account) → don't
    call revoke; just delete locally."""
    apple_id = "001234.no_refresh.0001"
    user = await _seed_user(db_session, apple_user_id=apple_id)
    # Don't set apple_refresh_token; default is None
    await db_session.commit()

    revoke_calls: list = []

    async def fake_revoke(refresh_token: str):
        revoke_calls.append(refresh_token)

    monkeypatch.setattr(auth_apple_module, "revoke_apple_token", fake_revoke)

    token, _ = create_access_token(apple_id)
    resp = await client.post(
        "/auth/delete",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert revoke_calls == [], "no refresh token → no revoke call"


@pytest.mark.asyncio
async def test_delete_account_revoke_failure_still_deletes_locally(client, db_session, monkeypatch):
    """If Apple's /auth/revoke endpoint is down or returns an error, we still
    delete the user locally (best-effort revoke, per App Store guideline)."""
    import httpx

    apple_id = "001234.revoke_fails.0001"
    user = await _seed_user(db_session, apple_user_id=apple_id)
    user.apple_refresh_token = "fake-refresh-token"
    await db_session.commit()

    async def fake_revoke(refresh_token: str):
        raise httpx.HTTPError("Apple is down")

    monkeypatch.setattr(auth_apple_module, "revoke_apple_token", fake_revoke)

    token, _ = create_access_token(apple_id)
    resp = await client.post(
        "/auth/delete",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )
    # User STILL deleted locally — App Store guideline 5.1.1(v) requires the
    # local delete to complete even when Apple's revoke endpoint fails.
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    db_session.expire_all()
    gone = (
        await db_session.execute(select(User).where(User.apple_user_id == apple_id))
    ).scalar_one_or_none()
    assert gone is None, "local delete must run even when Apple revoke fails"


# Per-user PHI tables that have no FK to users and so survive a plain
# CASCADE delete. Each is (table, user_column). Account deletion must purge
# every one of them (audit A3 / round-1 critical right-to-erasure failure).
_UNCASCADED_PHI_TABLES = [
    ("ml_baselines", "user_id"),
    ("ml_change_points", "user_id"),
    ("ml_forecasts", "user_id"),
    ("ml_anomalies", "user_id"),
    ("ml_directional_tests", "user_id"),
    ("ml_causal_estimates", "user_id"),
    ("ml_feature_values", "user_id"),
    ("ml_insight_candidates", "user_id"),
    ("ml_rankings", "user_id"),
    ("ml_experiments", "user_id"),
    ("ml_n_of_1_results", "user_id"),
    ("user_mascot_state", "user_id"),
    ("notification_records", "user_id"),
]


@pytest.mark.asyncio
async def test_delete_account_purges_all_user_keyed_phi(client, db_session, monkeypatch):
    """Right-to-erasure (audit A3): account deletion must remove the user's
    rows from every per-user PHI table, not just the user row + cascaded
    tenant tables. These ml_*/notification/mascot tables have no FK to users.
    """
    apple_id = "001234.purge_phi.0001"
    await _seed_user(db_session, apple_user_id=apple_id)

    # Seed a representative uncascaded row across each category: an ML table, the
    # mascot table, and the notification table. All three have no FK to users.
    # Use the ORM so Python-side column defaults (created_at, etc.) apply.
    db_session.add_all(
        [
            ml_insights.MLRanking(
                user_id=apple_id,
                surface_date="2026-06-06",
                candidate_id="cand-1",
                rank=1,
                score=0.9,
                ranker_version="v1",
                was_shown=True,
            ),
            mascot.UserMascotState(user_id=apple_id, accessory_id="hat"),
            notification.NotificationRecord(
                user_id=apple_id,
                category="morning_brief",
                title="hi",
                body="your HRV is 42",
            ),
        ]
    )
    await db_session.commit()

    async def fake_revoke(refresh_token: str):
        return None

    monkeypatch.setattr(auth_apple_module, "revoke_apple_token", fake_revoke)

    # /auth/delete is rate-limited by auth_apple's own limiter. The file's other
    # delete tests consume its shared 3/minute bucket, so clear that limiter's
    # in-memory storage to keep this 4th delete from being rejected 429.
    for _attr in ("storage", "events", "expirations"):
        _d = getattr(auth_apple_module.limiter._storage, _attr, None)
        if hasattr(_d, "clear"):
            _d.clear()

    token, _ = create_access_token(apple_id)
    resp = await client.post(
        "/auth/delete", json={}, headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 200

    db_session.expire_all()
    for table in ("ml_rankings", "user_mascot_state", "notification_records"):
        count = (
            await db_session.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE user_id = :u"),  # noqa: S608
                {"u": apple_id},
            )
        ).scalar_one()
        assert count == 0, f"{table} still has rows for the deleted user (PHI not erased)"
