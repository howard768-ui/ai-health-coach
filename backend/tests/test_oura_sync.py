"""Tests for the Oura background sync service.

Covers `sync_user_data`: token gating, error paths, dedup, and the
critical session-vs-daily_sleep field-source rule (the audit calls out
that daily_sleep.contributors.efficiency is a 0-100 score, not a
percentage; the real percentage comes from sleep_sessions).

Audit follow-up: docs/comprehensive-scan-2026-04-30.md section 5
flagged this as zero-test despite owning the "Sync_user_data ->
Refresh_access_token" 3-step process.

Run: cd backend && uv run python -m pytest tests/test_oura_sync.py -v
"""

from __future__ import annotations

import os
import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from datetime import date, timedelta
from unittest.mock import MagicMock

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.database import Base
from app.models.health import HealthMetricRecord, OuraToken, SleepRecord
from app.services import oura_sync


USER = "test-user-1"


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


def _add_valid_token(db: AsyncSession) -> OuraToken:
    """Add a token that won't trigger refresh (expires in 1 day)."""
    from app.core.time import utcnow_naive
    tok = OuraToken(
        user_id=USER,
        access_token="fake-access",
        refresh_token="fake-refresh",
        expires_at=utcnow_naive() + timedelta(days=1),
    )
    db.add(tok)
    return tok


def _make_client_mock(
    sleep_data=None,
    readiness_data=None,
    sleep_sessions=None,
    heartrate=None,
    raises=None,
):
    """Build a fake OuraClient whose async methods return canned data."""
    client = MagicMock()

    async def _async_return(value):
        return value

    async def _async_raise(exc):
        raise exc

    if raises is not None:
        client.get_daily_sleep = lambda *a, **kw: _async_raise(raises)
    else:
        client.get_daily_sleep = lambda *a, **kw: _async_return({"data": sleep_data or []})
    client.get_daily_readiness = lambda *a, **kw: _async_return({"data": readiness_data or []})
    client.get_sleep_sessions = lambda *a, **kw: _async_return({"data": sleep_sessions or []})
    client.get_heartrate = lambda *a, **kw: _async_return({"data": heartrate or []})
    return client


# ensure_valid_token + sync_user_data error paths -------------------------


@pytest.mark.asyncio
async def test_returns_error_when_no_token_in_db(db):
    """sync_user_data short-circuits with a clean error message."""
    result = await oura_sync.sync_user_data(db, USER)
    assert result == {"status": "error", "message": "No valid Oura token"}


@pytest.mark.asyncio
async def test_returns_revoked_message_on_oura_401(db, monkeypatch):
    """A 401 from Oura means the user revoked access; surface a friendly string."""
    await db.run_sync(lambda _: None)  # no-op to satisfy linters
    _add_valid_token(db)
    await db.commit()

    response = httpx.Response(401, request=httpx.Request("GET", "https://api.ouraring.com/v2/usercollection/daily_sleep"))
    err = httpx.HTTPStatusError("401", request=response.request, response=response)
    client = _make_client_mock(raises=err)

    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)
    result = await oura_sync.sync_user_data(db, USER)
    assert result["status"] == "error"
    assert "Oura access revoked" in result["message"]


@pytest.mark.asyncio
async def test_returns_error_on_oura_network_failure(db, monkeypatch):
    _add_valid_token(db)
    await db.commit()

    err = httpx.ConnectError("connection refused")
    client = _make_client_mock(raises=err)
    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)

    result = await oura_sync.sync_user_data(db, USER)
    assert result["status"] == "error"
    assert "Oura API error" in result["message"]


@pytest.mark.asyncio
async def test_returns_error_on_oura_response_parse_error(db, monkeypatch):
    _add_valid_token(db)
    await db.commit()

    err = ValueError("malformed json")
    client = _make_client_mock(raises=err)
    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)

    result = await oura_sync.sync_user_data(db, USER)
    assert result["status"] == "error"
    assert "unexpected data" in result["message"]


# Successful sync paths ---------------------------------------------------


@pytest.mark.asyncio
async def test_writes_sleep_record_with_session_efficiency_not_contributor_score(db, monkeypatch):
    """The audit's data-fidelity rule: efficiency comes from sleep_sessions
    (true 0-100 percentage), NOT daily_sleep.contributors.efficiency
    (which is a 0-100 score input to the daily sleep score).
    """
    _add_valid_token(db)
    await db.commit()

    today = date.today().isoformat()
    sleep_data = [{
        "day": today,
        "contributors": {"efficiency": 60},  # Score input, intentionally different
    }]
    sleep_sessions = [{
        "day": today,
        "efficiency": 92,  # True percentage
        "total_sleep_duration": 7 * 3600,
        "deep_sleep_duration": 90 * 60,
        "rem_sleep_duration": 100 * 60,
        "light_sleep_duration": 230 * 60,
        "lowest_heart_rate": 52,
        "bedtime_start": f"{today}T23:30:00+00:00",
        "bedtime_end": f"{today}T07:00:00+00:00",
    }]
    client = _make_client_mock(sleep_data=sleep_data, sleep_sessions=sleep_sessions)
    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)

    result = await oura_sync.sync_user_data(db, USER)
    assert result["status"] == "ok"
    assert result["records_saved"] == 1

    rec = (await db.execute(select(SleepRecord).where(SleepRecord.user_id == USER))).scalar_one()
    assert rec.efficiency == 92, "must use session efficiency, not contributor score"
    assert rec.total_sleep_seconds == 7 * 3600
    assert rec.deep_sleep_seconds == 90 * 60
    assert rec.resting_hr == 52


@pytest.mark.asyncio
async def test_picks_longest_sleep_session_per_day(db, monkeypatch):
    """When Oura returns multiple sessions for one day (nap + main sleep),
    we use the longest as the main sleep, not the first."""
    _add_valid_token(db)
    await db.commit()

    today = date.today().isoformat()
    sleep_data = [{"day": today}]
    sleep_sessions = [
        # Nap, listed first
        {"day": today, "efficiency": 50, "total_sleep_duration": 30 * 60},
        # Main sleep, longer
        {"day": today, "efficiency": 90, "total_sleep_duration": 7 * 3600},
    ]
    client = _make_client_mock(sleep_data=sleep_data, sleep_sessions=sleep_sessions)
    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)

    await oura_sync.sync_user_data(db, USER)
    rec = (await db.execute(select(SleepRecord).where(SleepRecord.user_id == USER))).scalar_one()
    assert rec.efficiency == 90
    assert rec.total_sleep_seconds == 7 * 3600


@pytest.mark.asyncio
async def test_dedup_skips_existing_sleep_record_for_same_day(db, monkeypatch):
    _add_valid_token(db)
    today = date.today().isoformat()
    db.add(SleepRecord(user_id=USER, date=today, efficiency=80))
    await db.commit()

    sleep_data = [{"day": today}]
    sleep_sessions = [{"day": today, "efficiency": 95, "total_sleep_duration": 7 * 3600}]
    client = _make_client_mock(sleep_data=sleep_data, sleep_sessions=sleep_sessions)
    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)

    result = await oura_sync.sync_user_data(db, USER)
    assert result["records_saved"] == 0

    rows = (await db.execute(select(SleepRecord).where(SleepRecord.user_id == USER))).scalars().all()
    assert len(rows) == 1
    assert rows[0].efficiency == 80, "existing record should be untouched"


@pytest.mark.asyncio
async def test_writes_health_metric_records_for_reconciliation(db, monkeypatch):
    """Successful sync also writes unified HealthMetricRecord rows."""
    _add_valid_token(db)
    await db.commit()

    today = date.today().isoformat()
    sleep_data = [{"day": today}]
    sleep_sessions = [{
        "day": today,
        "efficiency": 92,
        "total_sleep_duration": 7 * 3600,
        "lowest_heart_rate": 52,
    }]
    readiness = [{"day": today, "score": 88}]
    client = _make_client_mock(
        sleep_data=sleep_data, sleep_sessions=sleep_sessions, readiness_data=readiness
    )
    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)

    await oura_sync.sync_user_data(db, USER)

    metrics = {
        m.metric_type: m.value
        for m in (await db.execute(select(HealthMetricRecord).where(HealthMetricRecord.user_id == USER))).scalars().all()
    }
    assert metrics["sleep_efficiency"] == 92
    assert metrics["sleep_duration"] == pytest.approx(7.0, rel=1e-3)
    assert metrics["resting_hr"] == 52
    assert metrics["readiness"] == 88


@pytest.mark.asyncio
async def test_sync_bumps_last_synced_at(db, monkeypatch):
    tok = _add_valid_token(db)
    await db.commit()
    original = tok.last_synced_at  # may be None

    client = _make_client_mock()
    monkeypatch.setattr(oura_sync, "OuraClient", lambda *a, **kw: client)

    await oura_sync.sync_user_data(db, USER)
    refreshed = (await db.execute(select(OuraToken).where(OuraToken.user_id == USER))).scalar_one()
    assert refreshed.last_synced_at is not None
    assert refreshed.last_synced_at != original


# _parse_time helper ------------------------------------------------------


def test_parse_time_returns_none_for_empty():
    assert oura_sync._parse_time(None) is None
    assert oura_sync._parse_time("") is None


def test_parse_time_extracts_hh_mm():
    assert oura_sync._parse_time("2026-05-02T23:30:15+00:00") == "23:30"
    assert oura_sync._parse_time("2026-05-02T07:00:00Z") == "07:00"


def test_parse_time_returns_none_on_garbage():
    assert oura_sync._parse_time("not-a-timestamp") is None
