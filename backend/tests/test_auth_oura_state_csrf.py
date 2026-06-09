"""Tests for the CSRF-safe Oura OAuth state flow.

Pre-fix, the OAuth `state` was the raw apple_user_id supplied by an
unauthenticated caller, so an attacker who knew a victim's apple_user_id
could forge the callback and attach their own Oura account to the victim
(OAuth CSRF / account injection).

Post-fix, `POST /auth/oura/start` (bearer-authenticated) mints a signed,
time-limited state bound to the caller. The callback prefers that signed
state and only falls back to the legacy apple_user_id path for older app
builds. These tests pin the signer and the authenticated start endpoint.

Run: cd backend && ./.venv/bin/python -m pytest tests/test_auth_oura_state_csrf.py -v
"""

import os

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-oura-csrf")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

import app.routers.auth as auth_mod
from app.api import deps
from app.database import Base, get_db
from app.main import app
from app.models.health import OuraToken
from app.models.user import User
from app.routers.auth import _sign_oauth_state, _verify_oauth_state


# ── Signer unit tests ────────────────────────────────────────────────────


def test_sign_verify_roundtrip():
    state = _sign_oauth_state("001234.signed.0001")
    assert _verify_oauth_state(state) == "001234.signed.0001"


def test_verify_rejects_tampered_signature():
    state = _sign_oauth_state("001234.signed.0001")
    flipped = state[:-1] + ("A" if state[-1] != "A" else "B")
    assert _verify_oauth_state(flipped) is None


def test_verify_rejects_tampered_payload():
    # Swap the payload for a different user but keep the original signature.
    state = _sign_oauth_state("001234.victim.0001")
    _, sig = state.rsplit(".", 1)
    forged_body = _sign_oauth_state("001234.attacker.0002").split(".", 1)[0]
    assert _verify_oauth_state(f"{forged_body}.{sig}") is None


def test_verify_rejects_garbage():
    assert _verify_oauth_state("not-a-token") is None
    assert _verify_oauth_state("only.onedot") is None
    assert _verify_oauth_state("") is None


def test_verify_rejects_expired(monkeypatch):
    state = _sign_oauth_state("001234.expires.0001")
    now = auth_mod.time.time()
    monkeypatch.setattr(
        auth_mod.time, "time", lambda: now + auth_mod._OAUTH_STATE_TTL_SECONDS + 10
    )
    assert _verify_oauth_state(state) is None


# ── Fixtures for the API ─────────────────────────────────────────────────


@pytest_asyncio.fixture
async def test_engine():
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
    SessionMaker = async_sessionmaker(test_engine, expire_on_commit=False)
    async with SessionMaker() as session:
        yield session


@pytest_asyncio.fixture
async def client(test_engine):
    SessionMaker = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with SessionMaker() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


# ── /auth/oura/start ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_requires_authentication(client):
    """No bearer token -> 401/403, so an attacker can't mint a state for
    someone else."""
    resp = await client.post("/auth/oura/start")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_start_returns_authorize_url_with_signed_state(client):
    """Authenticated caller gets an Oura authorize URL whose state resolves
    back to their own apple_user_id."""
    user = User(apple_user_id="001234.start.0001", is_active=True)

    async def override_user():
        return user

    app.dependency_overrides[deps.get_current_user] = override_user
    try:
        resp = await client.post("/auth/oura/start")
    finally:
        app.dependency_overrides.pop(deps.get_current_user, None)

    assert resp.status_code == 200
    url = resp.json()["authorize_url"]
    assert "state=" in url
    state = url.split("state=", 1)[1]
    assert _verify_oauth_state(state) == "001234.start.0001"


# ── Callback with a signed state (CSRF-safe happy path) ──────────────────


@pytest.mark.asyncio
async def test_callback_accepts_signed_state(client, db_session):
    """A signed state minted for a user attaches the Oura token to THAT user."""
    db_session.add(User(apple_user_id="001234.signed-cb.0001", is_active=True))
    await db_session.commit()

    signed = _sign_oauth_state("001234.signed-cb.0001")
    token_data = {
        "access_token": "access-signed",
        "refresh_token": "refresh-signed",
        "expires_in": 86400,
    }

    with patch("app.routers.auth.OuraClient") as fake_cls:
        instance = fake_cls.return_value
        instance.exchange_code = AsyncMock(return_value=token_data)
        instance.get_personal_info = AsyncMock(return_value={"id": "oura-signed-1"})

        resp = await client.get(
            "/auth/oura/callback",
            params={"code": "fake-code", "state": signed},
        )

    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "meld://oura/connected"

    db_session.expire_all()
    row = (
        await db_session.execute(
            select(OuraToken).where(OuraToken.user_id == "001234.signed-cb.0001")
        )
    ).scalar_one()
    assert row.access_token == "access-signed"
    assert row.oura_user_id == "oura-signed-1"


@pytest.mark.asyncio
async def test_callback_signed_state_for_unknown_user_errors(client):
    """A validly signed state whose uid has no User row -> invalid_state.
    (Signature is fine, but the user doesn't exist.)"""
    signed = _sign_oauth_state("001234.ghost.9999")
    resp = await client.get(
        "/auth/oura/callback",
        params={"code": "fake-code", "state": signed},
    )
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "meld://oura/error?reason=invalid_state"
