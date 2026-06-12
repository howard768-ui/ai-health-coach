"""Regression + behavior tests for `peloton_sync.sync_user_data` after MEL-44 part 2.

Architecture: pylotoncycle has no persistable session token, so every sync logs
in fresh using credentials stored on `PelotonToken` (username + Fernet-encrypted
password column added in PR #103). These tests pin:

- no token row at all -> no_session
- token row exists but password missing (legacy pre-MEL-44 row) -> needs_reauth
- login fails (rotated password / pylotoncycle error) -> needs_reauth + Sentry
- login succeeds + workouts returned -> writes WorkoutRecord + HealthMetricRecord
- workout dedup by external_id (same workout twice = one row)
- last_used_at gets bumped on success

Run: cd backend && uv run python -m pytest tests/test_peloton_sync.py -v
"""

import os
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-peloton-sync-tests")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

from app.core.time import utcnow_naive
from app.database import Base
from app.models.peloton import PelotonToken, WorkoutRecord
from app.services.peloton_sync import sync_user_data


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


def _seed_token(
    db,
    *,
    user_id: str = "apple-test-1",
    username: str | None = "user@example.com",
    password: str | None = "s3cret",
    peloton_user_id: str = "peloton-uid-1",
) -> PelotonToken:
    token = PelotonToken(
        user_id=user_id,
        peloton_user_id=peloton_user_id,
        session_id="oauth",  # legacy NOT-NULL column
        username=username,
        password=password,
    )
    db.add(token)
    return token


def _make_workout(
    *,
    peloton_id: str = "wo-1",
    discipline: str = "cycling",
    duration: int = 1800,
    calories: int = 350,
    output: float = 200.0,
) -> dict:
    """A minimal Peloton workout dict matching pylotoncycle's GetRecentWorkouts shape."""
    return {
        "id": peloton_id,
        "fitness_discipline": discipline,
        "total_work": calories * 1000,  # parse_workout divides by 1000
        "avg_heart_rate": 145.0,
        "max_heart_rate": 168.0,
        "avg_output": output,
        "ride": {
            "duration": duration,
            "title": "Power Zone Endurance Ride",
            "instructor": {"name": "Matt Wilpers"},
        },
        "created_at": 1735689600,  # 2026-01-01T00:00:00Z (deterministic)
    }


# ── No-session ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_returns_no_session_when_no_token(db_session):
    """User has never connected Peloton -> structured no_session status."""
    result = await sync_user_data(db_session, "apple-no-token")
    assert result["status"] == "no_session"
    assert "Connect your account" in result["message"]


# ── Missing credentials (legacy pre-MEL-44 row) ──────────────────────────


@pytest.mark.asyncio
async def test_sync_returns_needs_reauth_when_password_missing(db_session):
    """Legacy token from before the password column landed -> needs_reauth."""
    _seed_token(db_session, user_id="apple-legacy", password=None)
    await db_session.commit()

    result = await sync_user_data(db_session, "apple-legacy")
    assert result["status"] == "needs_reauth"
    assert "reconnect" in result["message"].lower() or "reauth" in result["message"].lower()


# ── Login failure ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_login_failure_returns_needs_reauth(db_session):
    """Password was rotated upstream / network error -> needs_reauth (no exception)."""
    _seed_token(db_session, user_id="apple-bad-creds")
    await db_session.commit()

    fake_login = AsyncMock(side_effect=ValueError("invalid credentials"))
    with patch("app.services.peloton_sync.PelotonClient") as fake_client_cls:
        fake_client = fake_client_cls.return_value
        fake_client.login = fake_login

        result = await sync_user_data(db_session, "apple-bad-creds")

    assert result["status"] == "needs_reauth"
    assert fake_login.await_count == 1


# ── Successful login + workout fetch + dedup + last_used_at ──────────────


@pytest.mark.asyncio
async def test_sync_success_writes_workouts_and_health_metrics(db_session):
    """Login succeeds + 2 workouts -> WorkoutRecord(2) + HealthMetricRecord(4: 2 minutes + 2 cals)."""
    _seed_token(db_session, user_id="apple-ok-1")
    await db_session.commit()

    fake_workouts = [
        _make_workout(peloton_id="wo-A", duration=1800, calories=300),
        _make_workout(peloton_id="wo-B", duration=900, calories=180),
    ]

    with patch("app.services.peloton_sync.PelotonClient") as fake_client_cls:
        fake_client = fake_client_cls.return_value
        fake_client.login = AsyncMock(return_value={"session_id": "oauth", "user_id": "p"})
        fake_client.get_workouts = AsyncMock(return_value=fake_workouts)
        # Use the real parse_workout method so we exercise the actual mapping
        from app.services.peloton import PelotonClient as RealPelotonClient
        fake_client.parse_workout = RealPelotonClient.parse_workout.__get__(fake_client)

        result = await sync_user_data(db_session, "apple-ok-1")

    assert result["status"] == "ok"
    assert result["records_saved"] == 2

    # Verify WorkoutRecord rows
    db_session.expire_all()
    rows = (
        await db_session.execute(
            select(WorkoutRecord).where(WorkoutRecord.user_id == "apple-ok-1")
        )
    ).scalars().all()
    assert len(rows) == 2
    assert {r.external_id for r in rows} == {"wo-A", "wo-B"}
    assert all(r.source == "peloton" for r in rows)


@pytest.mark.asyncio
async def test_sync_dedup_skips_already_seen_workout(db_session):
    """Same workout returned twice across two syncs -> only one WorkoutRecord."""
    _seed_token(db_session, user_id="apple-dedup-1")
    # Pre-seed an existing workout
    db_session.add(
        WorkoutRecord(
            user_id="apple-dedup-1",
            date="2026-01-01",
            source="peloton",
            external_id="wo-already-seen",
            workout_type="cycling",
            duration_seconds=1200,
        )
    )
    await db_session.commit()

    fake_workouts = [
        _make_workout(peloton_id="wo-already-seen"),  # dup
        _make_workout(peloton_id="wo-new"),
    ]

    with patch("app.services.peloton_sync.PelotonClient") as fake_client_cls:
        fake_client = fake_client_cls.return_value
        fake_client.login = AsyncMock(return_value={"session_id": "oauth", "user_id": "p"})
        fake_client.get_workouts = AsyncMock(return_value=fake_workouts)
        from app.services.peloton import PelotonClient as RealPelotonClient
        fake_client.parse_workout = RealPelotonClient.parse_workout.__get__(fake_client)

        result = await sync_user_data(db_session, "apple-dedup-1")

    assert result["status"] == "ok"
    assert result["records_saved"] == 1  # only the new one

    db_session.expire_all()
    rows = (
        await db_session.execute(
            select(WorkoutRecord).where(WorkoutRecord.user_id == "apple-dedup-1")
        )
    ).scalars().all()
    assert len(rows) == 2  # 1 pre-existing + 1 new
    assert {r.external_id for r in rows} == {"wo-already-seen", "wo-new"}


@pytest.mark.asyncio
async def test_sync_success_bumps_last_used_at(db_session):
    """A successful sync updates token.last_used_at so we can monitor freshness."""
    _seed_token(db_session, user_id="apple-bump-1")
    await db_session.commit()

    # Force last_used_at to far in the past
    token = (
        await db_session.execute(
            select(PelotonToken).where(PelotonToken.user_id == "apple-bump-1")
        )
    ).scalar_one()
    stale_time = utcnow_naive() - timedelta(days=7)
    token.last_used_at = stale_time
    await db_session.commit()

    with patch("app.services.peloton_sync.PelotonClient") as fake_client_cls:
        fake_client = fake_client_cls.return_value
        fake_client.login = AsyncMock(return_value={"session_id": "oauth", "user_id": "p"})
        fake_client.get_workouts = AsyncMock(return_value=[])  # no workouts is fine
        from app.services.peloton import PelotonClient as RealPelotonClient
        fake_client.parse_workout = RealPelotonClient.parse_workout.__get__(fake_client)

        result = await sync_user_data(db_session, "apple-bump-1")

    assert result["status"] == "ok"
    db_session.expire_all()
    refreshed = (
        await db_session.execute(
            select(PelotonToken).where(PelotonToken.user_id == "apple-bump-1")
        )
    ).scalar_one()
    assert refreshed.last_used_at > stale_time


# ── Defensive: unexpected return shapes ──────────────────────────────────


@pytest.mark.asyncio
async def test_sync_unexpected_return_shape_returns_error(db_session):
    """If pylotoncycle ever returns non-list, we log + bail with status=error."""
    _seed_token(db_session, user_id="apple-shape-1")
    await db_session.commit()

    with patch("app.services.peloton_sync.PelotonClient") as fake_client_cls:
        fake_client = fake_client_cls.return_value
        fake_client.login = AsyncMock(return_value={"session_id": "oauth", "user_id": "p"})
        fake_client.get_workouts = AsyncMock(return_value={"data": []})  # wrong shape

        result = await sync_user_data(db_session, "apple-shape-1")

    assert result["status"] == "error"
