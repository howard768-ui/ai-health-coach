"""Tests for `get_current_admin_user` dependency.

Flagged in the 2026-04-30 audit (MEL-43) as a CRITICAL zero-test path
that gates admin endpoints (`/api/insights/admin/rollback` and friends).
A bug here means either:
  - admins locked out of admin endpoints (annoying)
  - non-admins permitted to call them (catastrophic — any beta tester
    could swap the active CoreML ranker model)

Pure unit tests on the dep function — it takes a `User` and a settings
read, returns the User on allow or raises 403. No TestClient needed.

Run: cd backend && uv run pytest tests/test_admin_gate.py -v
"""

import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-admin-gate-tests")

from app.api.deps import get_current_admin_user
from app.config import settings
from app.models.user import User


def _make_user(apple_user_id: str = "001234.deadbeef.0001") -> User:
    """Build an in-memory User instance. Not added to any session — the
    admin gate only reads `user.apple_user_id` and never queries the DB."""
    return User(
        apple_user_id=apple_user_id,
        email=None,
        name=None,
        is_active=True,
    )


# ── empty admin_user_ids = nobody is admin ─────────────────────────────


@pytest.mark.asyncio
async def test_empty_admin_list_denies_everyone(monkeypatch):
    """`admin_user_ids=""` (default in dev) means no one passes the gate."""
    monkeypatch.setattr(settings, "admin_user_ids", "")
    with pytest.raises(HTTPException) as exc:
        await get_current_admin_user(user=_make_user("brock-id"))
    assert exc.value.status_code == 403
    assert "Admin access required" in exc.value.detail


@pytest.mark.asyncio
async def test_none_admin_list_denies_everyone(monkeypatch):
    """`admin_user_ids=None` (env unset) is treated the same as empty string."""
    monkeypatch.setattr(settings, "admin_user_ids", None)
    with pytest.raises(HTTPException) as exc:
        await get_current_admin_user(user=_make_user("brock-id"))
    assert exc.value.status_code == 403


# ── single admin allowed ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_admin_passes(monkeypatch):
    """User in the allow-list returns their User row."""
    monkeypatch.setattr(settings, "admin_user_ids", "brock-id")
    user = _make_user("brock-id")
    result = await get_current_admin_user(user=user)
    assert result is user


@pytest.mark.asyncio
async def test_non_admin_denied_when_admin_configured(monkeypatch):
    """Other users are still denied even when an admin is configured."""
    monkeypatch.setattr(settings, "admin_user_ids", "brock-id")
    with pytest.raises(HTTPException) as exc:
        await get_current_admin_user(user=_make_user("attacker-id"))
    assert exc.value.status_code == 403


# ── comma-separated parsing + whitespace handling ────────────────────


@pytest.mark.asyncio
async def test_comma_separated_admin_list_all_pass(monkeypatch):
    """Multiple admins all pass via the same gate."""
    monkeypatch.setattr(settings, "admin_user_ids", "brock-id,alice-id,bob-id")
    for uid in ("brock-id", "alice-id", "bob-id"):
        user = await get_current_admin_user(user=_make_user(uid))
        assert user.apple_user_id == uid


@pytest.mark.asyncio
async def test_whitespace_around_admins_is_stripped(monkeypatch):
    """Pasting an env var with stray whitespace shouldn't lock admins out."""
    monkeypatch.setattr(settings, "admin_user_ids", "  brock-id ,  alice-id  ")
    for uid in ("brock-id", "alice-id"):
        user = await get_current_admin_user(user=_make_user(uid))
        assert user.apple_user_id == uid


@pytest.mark.asyncio
async def test_empty_segments_in_csv_ignored(monkeypatch):
    """Trailing comma or double-comma shouldn't admit an empty-string user."""
    monkeypatch.setattr(settings, "admin_user_ids", "brock-id,,alice-id,")
    # Empty-string user ID must NOT slip through
    with pytest.raises(HTTPException):
        await get_current_admin_user(user=_make_user(""))
    # Real admins still pass
    user = await get_current_admin_user(user=_make_user("brock-id"))
    assert user.apple_user_id == "brock-id"


# ── case sensitivity is intentional ──────────────────────────────────


@pytest.mark.asyncio
async def test_apple_user_id_matching_is_case_sensitive(monkeypatch):
    """Apple user IDs are case-significant. Don't accidentally match a
    near-miss case variant."""
    monkeypatch.setattr(settings, "admin_user_ids", "Brock-ID")
    with pytest.raises(HTTPException) as exc:
        await get_current_admin_user(user=_make_user("brock-id"))
    assert exc.value.status_code == 403


# ── exact-match invariant: substring shouldn't trip ──────────────────


@pytest.mark.asyncio
async def test_substring_user_id_does_not_match(monkeypatch):
    """If admin_user_ids='001234.abcd.0001', a user with apple_user_id
    of just '001234' should NOT be granted admin."""
    monkeypatch.setattr(settings, "admin_user_ids", "001234.abcd.0001")
    with pytest.raises(HTTPException) as exc:
        await get_current_admin_user(user=_make_user("001234"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_user_id_containing_admin_id_does_not_match(monkeypatch):
    """And vice versa: an attacker user_id that contains the admin id as
    a substring shouldn't get in."""
    monkeypatch.setattr(settings, "admin_user_ids", "brock-id")
    with pytest.raises(HTTPException) as exc:
        await get_current_admin_user(user=_make_user("evil-brock-id-attacker"))
    assert exc.value.status_code == 403


# ── webhook admin endpoints must be gated by CurrentAdminUser ──────────


def _dependency_callables(dependant) -> set:
    """Flatten a FastAPI route dependant tree into the set of callables."""
    calls: set = set()
    stack = [dependant]
    while stack:
        d = stack.pop()
        if d.call is not None:
            calls.add(d.call)
        stack.extend(d.dependencies)
    return calls


def _find_route(app, suffix: str, method: str):
    for r in app.routes:
        if getattr(r, "path", "").endswith(suffix) and method in getattr(
            r, "methods", set()
        ):
            return r
    return None


@pytest.mark.parametrize(
    "suffix,method",
    [("/oura/register", "POST"), ("/oura/subscriptions", "GET")],
)
def test_webhook_admin_endpoints_require_admin(suffix, method):
    """Audit A5: the Oura webhook register/list endpoints are admin actions
    (they can repoint webhooks). They must resolve get_current_admin_user, not
    just get_current_user, so a plain beta tester cannot reach them.
    """
    from app.main import app
    from app.api.deps import get_current_admin_user

    route = _find_route(app, suffix, method)
    assert route is not None, f"route {method} ...{suffix} not found"
    calls = _dependency_callables(route.dependant)
    assert get_current_admin_user in calls, f"...{suffix} is not admin-gated"
