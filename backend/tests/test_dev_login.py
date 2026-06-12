"""Tests for POST /auth/dev-login (CI/E2E-only token mint).

The route exists solely for Maestro E2E runs: the iOS client fires it
under -uitesting-skip-auth and expects the standard TokenPair shape.
Security property under test: anything that is not explicitly
development or test gets a 404 (fail closed on misspelled APP_ENV).

Fixtures lift the in-memory SQLite + AsyncClient pattern from
test_auth_routes.py.

Run: cd backend && uv run pytest tests/test_dev_login.py -v
"""

import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Required env BEFORE app modules are imported (app/config reads at import).
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-dev-login-tests")
os.environ.setdefault(
    "ENCRYPTION_KEY",
    "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=",
)

from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.routers.auth_apple import DEV_LOGIN_USER_ID


# ── Fixtures ─────────────────────────────────────────────────────────────


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
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dev_login_mints_token_pair_and_creates_user(client, db_session):
    """Empty POST returns the exact TokenPair shape the iOS decodeTokenPair
    requires, creates the fixed dev user, and persists a refresh hash."""
    resp = await client.post("/auth/dev-login")
    assert resp.status_code == 200

    body = resp.json()
    assert isinstance(body["access_token"], str) and body["access_token"]
    assert isinstance(body["refresh_token"], str) and body["refresh_token"]
    assert isinstance(body["expires_in"], int) and body["expires_in"] > 0
    assert body["user"]["id"] == DEV_LOGIN_USER_ID

    user = (
        await db_session.execute(
            select(User).where(User.apple_user_id == DEV_LOGIN_USER_ID)
        )
    ).scalar_one()
    assert user.is_active is True
    assert user.onboarding_complete is True

    token_count = (
        await db_session.execute(
            select(func.count())
            .select_from(RefreshToken)
            .where(RefreshToken.user_id == DEV_LOGIN_USER_ID)
        )
    ).scalar_one()
    assert token_count == 1


@pytest.mark.asyncio
async def test_dev_login_is_idempotent_on_user(client, db_session):
    """Second call reuses the dev user (no duplicate row) and mints a fresh
    refresh token (one row per call, like real per-flow relaunches)."""
    assert (await client.post("/auth/dev-login")).status_code == 200
    assert (await client.post("/auth/dev-login")).status_code == 200

    user_count = (
        await db_session.execute(
            select(func.count())
            .select_from(User)
            .where(User.apple_user_id == DEV_LOGIN_USER_ID)
        )
    ).scalar_one()
    assert user_count == 1

    token_count = (
        await db_session.execute(
            select(func.count())
            .select_from(RefreshToken)
            .where(RefreshToken.user_id == DEV_LOGIN_USER_ID)
        )
    ).scalar_one()
    assert token_count == 2


@pytest.mark.asyncio
async def test_dev_login_minted_token_authenticates(client):
    """The minted access token must satisfy the real auth dependency."""
    pair = (await client.post("/auth/dev-login")).json()
    resp = await client.get(
        "/api/user/profile",
        headers={"Authorization": f"Bearer {pair['access_token']}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["production", "staging", "PRODUCTION", "prod "])
async def test_dev_login_404s_outside_dev_and_test(client, monkeypatch, env):
    """Fail closed: anything not explicitly development/test is invisible."""
    monkeypatch.setattr(settings, "app_env", env)
    resp = await client.post("/auth/dev-login")
    assert resp.status_code == 404
