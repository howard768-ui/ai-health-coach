"""Tests for the waitlist signup admin alerter (Resend).

The notifier is wired as a FastAPI BackgroundTask on POST /api/waitlist/subscribe
so its failure modes must never break the signup endpoint. These tests pin:

- Empty RESEND_API_KEY = silent skip, notified stays False
- Successful send = notified flips to True, Resend called with the expected payload
- Send exception = swallowed, notified stays False
- Re-entry on an already-notified row = no second send

The Resend SDK is monkeypatched at the import site (`resend.Emails.send`) so
no network calls happen and no API key is needed.

Run: cd backend && uv run pytest tests/test_waitlist_notifier.py -v
"""

import os
from datetime import datetime

import pytest
import pytest_asyncio
import resend

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-waitlist-tests")

from app.config import settings
from app.database import Base, async_session, engine
from app.models.waitlist import WaitlistSignup
from app.services.waitlist_notifier import _render, send_new_signup_alert


@pytest_asyncio.fixture
async def _schema():
    """Create the waitlist table on the (sqlite in-memory) test DB.

    Scoped per-test to keep state isolation simple; the table is tiny so the
    teardown cost is negligible.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def _make_signup(**overrides) -> int:
    """Insert a waitlist row and return its id."""
    fields = {
        "email": "alice@example.com",
        "source": "hero",
        "utm_source": "reddit",
        "utm_medium": "social",
        "utm_campaign": "launch",
        "referer": "https://reddit.com/r/quantifiedself",
        "notified": False,
    }
    fields.update(overrides)
    async with async_session() as db:
        signup = WaitlistSignup(**fields)
        db.add(signup)
        await db.commit()
        await db.refresh(signup)
        return signup.id


# ── _render ───────────────────────────────────────────────────────────


def test_render_includes_email_and_utm():
    signup = WaitlistSignup(
        email="alice@example.com",
        source="hero",
        utm_source="reddit",
        utm_medium="social",
        utm_campaign="launch",
        referer="https://reddit.com/r/quantifiedself",
    )
    signup.created_at = datetime(2026, 5, 14, 14, 32)

    subject, html = _render(signup)

    assert "alice@example.com" in subject
    assert "alice@example.com" in html
    assert "utm_source=reddit" in html
    assert "utm_medium=social" in html
    assert "utm_campaign=launch" in html
    assert "source=hero" in html
    assert "2026-05-14 14:32 UTC" in html


def test_render_handles_no_attribution():
    signup = WaitlistSignup(email="bob@example.com")
    signup.created_at = datetime(2026, 5, 14, 14, 32)

    _subject, html = _render(signup)

    assert "direct (no attribution)" in html
    assert "direct" in html  # referer fallback


# ── send_new_signup_alert ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skip_when_resend_api_key_empty(monkeypatch, _schema):
    """Empty RESEND_API_KEY is the dev/preview default. Skip + log, do not
    flip notified, do not crash."""
    monkeypatch.setattr(settings, "resend_api_key", "")
    signup_id = await _make_signup()

    sends: list[dict] = []
    monkeypatch.setattr(resend.Emails, "send", lambda payload: sends.append(payload))

    result = await send_new_signup_alert(signup_id)

    assert result is False
    assert sends == []
    async with async_session() as db:
        row = await db.get(WaitlistSignup, signup_id)
        assert row.notified is False


@pytest.mark.asyncio
async def test_successful_send_flips_notified(monkeypatch, _schema):
    """Happy path: Resend accepts, notified flips to True, payload contains
    the signer's email."""
    monkeypatch.setattr(settings, "resend_api_key", "re_test_key")
    monkeypatch.setattr(settings, "resend_admin_to", "hello@heymeld.com")
    monkeypatch.setattr(settings, "resend_from", "Meld <noreply@heymeld.com>")
    signup_id = await _make_signup(email="alice@example.com", utm_source="reddit")

    sends: list[dict] = []

    def _fake_send(payload):
        sends.append(payload)
        return {"id": "re_msg_123"}

    monkeypatch.setattr(resend.Emails, "send", _fake_send)

    result = await send_new_signup_alert(signup_id)

    assert result is True
    assert len(sends) == 1
    payload = sends[0]
    assert payload["from"] == "Meld <noreply@heymeld.com>"
    assert payload["to"] == ["hello@heymeld.com"]
    assert "alice@example.com" in payload["subject"]
    assert "alice@example.com" in payload["html"]
    assert "utm_source=reddit" in payload["html"]

    async with async_session() as db:
        row = await db.get(WaitlistSignup, signup_id)
        assert row.notified is True


@pytest.mark.asyncio
async def test_send_exception_is_swallowed(monkeypatch, _schema):
    """Resend down / network blip / bad domain = swallow, log, do not flip
    notified so a future retry can pick the row up."""
    monkeypatch.setattr(settings, "resend_api_key", "re_test_key")
    signup_id = await _make_signup()

    def _boom(_payload):
        raise RuntimeError("resend 503")

    monkeypatch.setattr(resend.Emails, "send", _boom)

    result = await send_new_signup_alert(signup_id)

    assert result is False
    async with async_session() as db:
        row = await db.get(WaitlistSignup, signup_id)
        assert row.notified is False


@pytest.mark.asyncio
async def test_already_notified_is_noop(monkeypatch, _schema):
    """Re-entry on an already-flipped row must not double-send."""
    monkeypatch.setattr(settings, "resend_api_key", "re_test_key")
    signup_id = await _make_signup(notified=True)

    sends: list[dict] = []
    monkeypatch.setattr(resend.Emails, "send", lambda payload: sends.append(payload))

    result = await send_new_signup_alert(signup_id)

    assert result is False
    assert sends == []


@pytest.mark.asyncio
async def test_missing_signup_id_does_not_crash(monkeypatch, _schema):
    """Passing a stale id (row was deleted between enqueue and run) is logged
    but does not propagate."""
    monkeypatch.setattr(settings, "resend_api_key", "re_test_key")

    sends: list[dict] = []
    monkeypatch.setattr(resend.Emails, "send", lambda payload: sends.append(payload))

    result = await send_new_signup_alert(signup_id=999_999)

    assert result is False
    assert sends == []
