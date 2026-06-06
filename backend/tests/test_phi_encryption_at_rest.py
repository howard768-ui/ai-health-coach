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
from app.models.garmin import GarminDailyRecord
from app.models.health import SleepRecord
from app.models.peloton import WorkoutRecord
from app.models.user import User


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


@pytest.mark.asyncio
async def test_garmin_peloton_rawjson_and_custom_goal_encrypted_at_rest(session):
    g_payload = '{"stress": 42, "body_battery": 77}'
    g = GarminDailyRecord(user_id="u1", date="2026-06-06", raw_json=g_payload)
    p_payload = '{"output": 210, "instructor": "Ally"}'
    p = WorkoutRecord(
        user_id="u1",
        date="2026-06-06",
        source="peloton",
        workout_type="cycling",
        duration_seconds=1800,
        raw_json=p_payload,
    )
    goal = "manage my afib and lose 15 lbs before the wedding"
    u = User(apple_user_id="001234.deadbeef.0001", custom_goal_text=goal)
    session.add_all([g, p, u])
    await session.commit()

    for table, col, plaintext, rid in [
        ("garmin_daily_records", "raw_json", g_payload, g.id),
        ("workout_records", "raw_json", p_payload, p.id),
        ("users", "custom_goal_text", goal, u.id),
    ]:
        raw = (
            await session.execute(
                text(f"SELECT {col} FROM {table} WHERE id = :i"), {"i": rid}
            )
        ).scalar_one()
        assert raw != plaintext, f"{table}.{col} stored plaintext"
        assert encryption.decrypt(raw) == plaintext, f"{table}.{col} did not round-trip"
