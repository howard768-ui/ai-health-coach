"""Tests for the MEL-45 part 3 per-user fan-out helper.

`iter_active_users` is the replacement for `_get_primary_user_id` across all
multi-user-aware scheduler jobs. These tests pin its contract:

- Returns active users in created_at order
- Excludes inactive users
- Returns empty list on a fresh DB
- One user's failure does not abort the rest of the tick

The 'default' placeholder fallback was removed in MEL-45 part 4 (the
placeholder row was dropped in migration e9c3f7b285a1).

Run: cd backend && uv run python -m pytest tests/test_scheduler_iter_active_users.py -v
"""

import os
from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-iter-active-users")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

from app.core.time import utcnow_naive
from app.database import Base
from app.models.user import User
from app.tasks.scheduler import iter_active_users


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


# ── iter_active_users contract ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_db_returns_empty_list(db_session):
    users = await iter_active_users(db_session)
    assert users == []


@pytest.mark.asyncio
async def test_returns_users_in_created_at_order(db_session):
    """Stable iteration order so logging + per-user retries are deterministic."""
    now = utcnow_naive()
    db_session.add(User(apple_user_id="apple-A", is_active=True, created_at=now - timedelta(days=2)))
    db_session.add(User(apple_user_id="apple-B", is_active=True, created_at=now - timedelta(days=1)))
    db_session.add(User(apple_user_id="apple-C", is_active=True, created_at=now))
    await db_session.commit()

    users = await iter_active_users(db_session)
    assert [u.apple_user_id for u in users] == ["apple-A", "apple-B", "apple-C"]


@pytest.mark.asyncio
async def test_inactive_users_excluded(db_session):
    """Soft-deleted accounts (is_active=False) must not appear."""
    db_session.add(User(apple_user_id="apple-active", is_active=True))
    db_session.add(User(apple_user_id="apple-deactivated", is_active=False))
    await db_session.commit()

    users = await iter_active_users(db_session)
    ids = [u.apple_user_id for u in users]
    assert ids == ["apple-active"]


# ── Fan-out smoke test on oura_sync_job ──────────────────────────────────


@pytest.mark.asyncio
async def test_oura_sync_job_visits_each_active_user(db_session, monkeypatch):
    """Two active users -> oura_sync called twice (once per user), reconcile twice."""
    db_session.add(User(apple_user_id="apple-A", is_active=True))
    db_session.add(User(apple_user_id="apple-B", is_active=True))
    await db_session.commit()

    sync_calls: list[str] = []
    reconcile_calls: list[tuple[str, str]] = []

    async def fake_sync(_db, user_id):
        sync_calls.append(user_id)
        return {"status": "ok"}

    async def fake_reconcile(_db, user_id, day):
        reconcile_calls.append((user_id, day))

    # Stub async_session so the job uses our test session
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_async_session():
        yield db_session

    from app.tasks import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "async_session", fake_async_session)
    monkeypatch.setattr(scheduler_module, "oura_sync", fake_sync)
    monkeypatch.setattr(scheduler_module, "reconcile_day", fake_reconcile)

    await scheduler_module.oura_sync_job()

    assert sorted(sync_calls) == ["apple-A", "apple-B"]
    assert {c[0] for c in reconcile_calls} == {"apple-A", "apple-B"}


@pytest.mark.asyncio
async def test_oura_sync_job_isolates_one_users_failure(db_session, monkeypatch):
    """If oura_sync raises for user A, user B's sync must still run."""
    db_session.add(User(apple_user_id="apple-A", is_active=True))
    db_session.add(User(apple_user_id="apple-B", is_active=True))
    await db_session.commit()

    visited: list[str] = []

    async def fake_sync(_db, user_id):
        visited.append(user_id)
        if user_id == "apple-A":
            raise RuntimeError("simulated failure for A")
        return {"status": "ok"}

    async def fake_reconcile(_db, user_id, day):
        pass

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_async_session():
        yield db_session

    from app.tasks import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "async_session", fake_async_session)
    monkeypatch.setattr(scheduler_module, "oura_sync", fake_sync)
    monkeypatch.setattr(scheduler_module, "reconcile_day", fake_reconcile)

    # Should NOT raise
    await scheduler_module.oura_sync_job()

    # Both users were visited despite A failing
    assert sorted(visited) == ["apple-A", "apple-B"]
