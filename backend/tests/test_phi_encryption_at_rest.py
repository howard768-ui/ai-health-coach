"""Audit C2: PHI free-text / raw_json columns must be encrypted at rest.

Proves end to end that ChatMessageRecord.content, .health_context and the
raw_json payload columns are stored as Fernet ciphertext (not plaintext) and
still round-trip through the ORM. A stolen DB file must not yield readable
chat or health data.

Run: cd backend && uv run pytest tests/test_phi_encryption_at_rest.py -v
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-phi-enc-tests")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.core import encryption
from app.database import Base
from app.models.chat import ChatMessageRecord
from app.models.health import SleepRecord


@pytest_asyncio.fixture
async def session():
    # Guarantee the cipher is active regardless of import order.
    settings.encryption_key = "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
    encryption._reset_for_tests()
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionMaker = async_sessionmaker(engine, expire_on_commit=False)
    async with SessionMaker() as s:
        yield s
    await engine.dispose()
    encryption._reset_for_tests()


@pytest.mark.asyncio
async def test_chat_content_encrypted_at_rest(session):
    secret = "I have chest pain at 2am and feel hopeless"
    msg = ChatMessageRecord(
        conversation_id=1, user_id="u1", role="user", content=secret
    )
    session.add(msg)
    await session.commit()
    mid = msg.id

    # ORM read decrypts transparently.
    got = (
        await session.execute(
            select(ChatMessageRecord).where(ChatMessageRecord.id == mid)
        )
    ).scalar_one()
    assert got.content == secret

    # Raw column read bypasses EncryptedString: must be ciphertext.
    raw = (
        await session.execute(
            text("SELECT content FROM chat_messages WHERE id = :i"), {"i": mid}
        )
    ).scalar_one()
    assert raw != secret
    assert "chest pain" not in raw
    assert encryption.decrypt(raw) == secret  # it is the ciphertext of secret


@pytest.mark.asyncio
async def test_health_context_and_raw_json_encrypted_at_rest(session):
    ctx = "HRV 28ms RHR 71bpm readiness 41"
    msg = ChatMessageRecord(
        conversation_id=1, user_id="u1", role="coach", content="ok", health_context=ctx
    )
    session.add(msg)
    await session.commit()
    raw_ctx = (
        await session.execute(
            text("SELECT health_context FROM chat_messages WHERE id = :i"),
            {"i": msg.id},
        )
    ).scalar_one()
    assert raw_ctx != ctx
    assert "HRV" not in raw_ctx
    assert encryption.decrypt(raw_ctx) == ctx

    payload = '{"hrv": 28, "rhr": 71, "readiness": 41}'
    sr = SleepRecord(user_id="u1", date="2026-06-06", raw_json=payload)
    session.add(sr)
    await session.commit()
    raw_json = (
        await session.execute(
            text("SELECT raw_json FROM sleep_records WHERE id = :i"), {"i": sr.id}
        )
    ).scalar_one()
    assert raw_json != payload
    assert "hrv" not in raw_json
    assert encryption.decrypt(raw_json) == payload
