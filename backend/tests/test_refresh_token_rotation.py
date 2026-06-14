"""Tests for the refresh-token rotation chain (`_revoke_chain` helper).

The chain-walk is the primary defense against refresh-token theft: when a
revoked token is presented to `/auth/refresh`, every token downstream of it
is also revoked, terminating the attacker's session in addition to the
victim's. The 2026-04-30 audit (MEL-43) flagged this as a CRITICAL zero-test
path — a regression in chain-walk logic could silently disable the
reuse-detection security property.

Run: cd backend && uv run pytest tests/test_refresh_token_rotation.py -v
"""

import os
from datetime import timedelta

import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-refresh-rotation-tests")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.routers.auth_apple import _revoke_chain
from app.core.time import utcnow_naive


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_session():
    """In-memory SQLite session, fresh schema per test.

    Keeps tests hermetic — a side-effect on one chain doesn't bleed into
    another. The session is closed (and thus the engine disposed) when the
    test completes.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


async def _make_user(session, apple_user_id: str = "user-001") -> User:
    user = User(
        apple_user_id=apple_user_id,
        is_active=True,
    )
    session.add(user)
    await session.flush()
    return user


def _make_token_row(
    *,
    token_hash: str,
    user_id: str,
    replaced_by: str | None = None,
    revoked: bool = False,
) -> RefreshToken:
    return RefreshToken(
        id=token_hash,
        user_id=user_id,
        device_id="test-device",
        expires_at=utcnow_naive() + timedelta(days=30),
        revoked_at=utcnow_naive() if revoked else None,
        replaced_by=replaced_by,
    )


# ── Single-token revocation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoke_chain_single_token(db_session):
    """A token with no replaced_by chain is revoked in place."""
    user = await _make_user(db_session)
    token = _make_token_row(token_hash="hash-A", user_id=user.apple_user_id)
    db_session.add(token)
    await db_session.flush()

    await _revoke_chain(db_session, "hash-A")

    refreshed = await db_session.get(RefreshToken, "hash-A")
    assert refreshed.revoked_at is not None


# ── Chain walking ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoke_chain_walks_full_chain(db_session):
    """A → B → C: revoking from A should revoke A, B, AND C."""
    user = await _make_user(db_session)
    db_session.add_all([
        _make_token_row(token_hash="hash-A", user_id=user.apple_user_id, replaced_by="hash-B"),
        _make_token_row(token_hash="hash-B", user_id=user.apple_user_id, replaced_by="hash-C"),
        _make_token_row(token_hash="hash-C", user_id=user.apple_user_id),
    ])
    await db_session.flush()

    await _revoke_chain(db_session, "hash-A")

    for h in ("hash-A", "hash-B", "hash-C"):
        row = await db_session.get(RefreshToken, h)
        assert row.revoked_at is not None, f"{h} should be revoked"


@pytest.mark.asyncio
async def test_revoke_chain_starts_mid_chain(db_session):
    """A → B → C: revoking from B revokes B + C, not A.

    Real-world case: attacker uses stolen token (hash-B), legitimate user's
    next refresh detects the chain and revokes B + everything downstream.
    A is the original token (already revoked when B was issued)."""
    user = await _make_user(db_session)
    db_session.add_all([
        _make_token_row(
            token_hash="hash-A",
            user_id=user.apple_user_id,
            replaced_by="hash-B",
            revoked=True,
        ),
        _make_token_row(token_hash="hash-B", user_id=user.apple_user_id, replaced_by="hash-C"),
        _make_token_row(token_hash="hash-C", user_id=user.apple_user_id),
    ])
    await db_session.flush()

    pre_a_revoked_at = (await db_session.get(RefreshToken, "hash-A")).revoked_at

    await _revoke_chain(db_session, "hash-B")

    a = await db_session.get(RefreshToken, "hash-A")
    b = await db_session.get(RefreshToken, "hash-B")
    c = await db_session.get(RefreshToken, "hash-C")
    # A was already revoked; revoked_at NOT overwritten with the new walk's now()
    assert a.revoked_at == pre_a_revoked_at
    # B and C are now revoked
    assert b.revoked_at is not None
    assert c.revoked_at is not None


@pytest.mark.asyncio
async def test_revoke_chain_terminates_on_missing_descendant(db_session):
    """A → B (deleted) → C: chain walk stops at the gap; we don't crash."""
    user = await _make_user(db_session)
    db_session.add_all([
        _make_token_row(
            token_hash="hash-A",
            user_id=user.apple_user_id,
            replaced_by="hash-MISSING",  # B was deleted
        ),
    ])
    await db_session.flush()

    # Should not raise
    await _revoke_chain(db_session, "hash-A")
    a = await db_session.get(RefreshToken, "hash-A")
    assert a.revoked_at is not None


@pytest.mark.asyncio
async def test_revoke_chain_handles_empty_chain(db_session):
    """Calling _revoke_chain with a non-existent hash is a no-op (no crash)."""
    await _make_user(db_session)
    await db_session.flush()

    # Should not raise even though no token with this hash exists
    await _revoke_chain(db_session, "hash-DOES-NOT-EXIST")


@pytest.mark.asyncio
async def test_revoke_chain_breaks_on_self_loop(db_session):
    """Defensive: a malformed chain pointing at itself shouldn't infinite-loop.

    The _revoke_chain implementation tracks visited hashes for exactly this
    reason. Pin the behavior so a future "optimization" doesn't drop the
    visited-set."""
    user = await _make_user(db_session)
    db_session.add(
        _make_token_row(
            token_hash="hash-LOOP",
            user_id=user.apple_user_id,
            replaced_by="hash-LOOP",  # points at itself
        ),
    )
    await db_session.flush()

    # Should terminate, not hang
    await _revoke_chain(db_session, "hash-LOOP")
    row = await db_session.get(RefreshToken, "hash-LOOP")
    assert row.revoked_at is not None


# ── Idempotency ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_revoke_chain_does_not_overwrite_existing_revocation(db_session):
    """If a token was already revoked at time T1, calling _revoke_chain at
    T2 must NOT overwrite the revoked_at timestamp.

    Why this matters: the revoked_at timestamp is used for forensic timeline
    reconstruction (when did we first detect the breach?). Overwriting it on
    every chain walk would erase the original detection moment."""
    user = await _make_user(db_session)
    earlier = utcnow_naive() - timedelta(hours=1)
    token = RefreshToken(
        id="hash-A",
        user_id=user.apple_user_id,
        device_id="test-device",
        expires_at=utcnow_naive() + timedelta(days=30),
        revoked_at=earlier,
    )
    db_session.add(token)
    await db_session.flush()

    await _revoke_chain(db_session, "hash-A")

    refreshed = await db_session.get(RefreshToken, "hash-A")
    assert refreshed.revoked_at == earlier
