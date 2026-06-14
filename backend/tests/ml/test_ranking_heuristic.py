"""Phase 4 heuristic ranker tests.

Pure-math tests for the scorer and DB-backed tests for cap enforcement +
idempotent rerun of ``materialize_daily_ranking``.

Run: ``cd backend && uv run python -m pytest tests/ml/test_ranking_heuristic.py -v``
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
# Register ORM models with Base.metadata before fixture create_all.
from app.models import ml_baselines as _ml_baselines_models  # noqa: F401
from app.models import ml_features as _ml_features_models  # noqa: F401
from app.models import ml_insights as _ml_insights_models  # noqa: F401
from app.models.ml_insights import MLRanking
from ml.ranking.candidates import InsightCandidate
from ml.ranking import heuristic


USER = "u-heuristic"


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session
    await engine.dispose()


def _fake_candidate(
    cid: str,
    effect: float = 0.5,
    conf: float = 0.6,
    actionability: float = 0.5,
    novelty: float = 1.0,
    literature: bool = False,
    kind: str = "correlation",
) -> InsightCandidate:
    return InsightCandidate(
        id=cid,
        user_id=USER,
        kind=kind,
        subject_metrics=("x", "y"),
        effect_size=effect,
        confidence=conf,
        novelty=novelty,
        recency_days=0,
        actionability_score=actionability,
        literature_support=literature,
    )


# ─────────────────────────────────────────────────────────────────────────
# heuristic_score (pure math)
# ─────────────────────────────────────────────────────────────────────────


def test_scorer_weights_sum_to_one():
    """Sanity: the five plan weights add up to 1 so a max candidate scores 1.0."""
    assert (
        heuristic._W_EFFECT
        + heuristic._W_CONFIDENCE
        + heuristic._W_ACTIONABILITY
        + heuristic._W_NOVELTY
        + heuristic._W_LITERATURE
    ) == pytest.approx(1.0)


def test_scorer_max_inputs_saturate_at_one():
    c = _fake_candidate("x", effect=1, conf=1, actionability=1, novelty=1, literature=True)
    assert heuristic.heuristic_score(c) == pytest.approx(1.0)


def test_scorer_zero_inputs_return_zero():
    c = _fake_candidate("x", effect=0, conf=0, actionability=0, novelty=0, literature=False)
    assert heuristic.heuristic_score(c) == pytest.approx(0.0)


def test_scorer_honors_plan_weights_by_component():
    """Each weight should dominate when only that input is non-zero."""
    e = _fake_candidate("e", effect=1.0, conf=0, actionability=0, novelty=0)
    c = _fake_candidate("c", effect=0, conf=1.0, actionability=0, novelty=0)
    a = _fake_candidate("a", effect=0, conf=0, actionability=1.0, novelty=0)
    n = _fake_candidate("n", effect=0, conf=0, actionability=0, novelty=1.0)
    lit = _fake_candidate("l", effect=0, conf=0, actionability=0, novelty=0, literature=True)

    assert heuristic.heuristic_score(e) == pytest.approx(heuristic._W_EFFECT)
    assert heuristic.heuristic_score(c) == pytest.approx(heuristic._W_CONFIDENCE)
    assert heuristic.heuristic_score(a) == pytest.approx(heuristic._W_ACTIONABILITY)
    assert heuristic.heuristic_score(n) == pytest.approx(heuristic._W_NOVELTY)
    assert heuristic.heuristic_score(lit) == pytest.approx(heuristic._W_LITERATURE)


def test_rank_sorts_by_score_desc():
    high = _fake_candidate("hi", effect=0.9, conf=0.9, actionability=0.9, novelty=0.9, literature=True)
    mid = _fake_candidate("mid", effect=0.5, conf=0.5, actionability=0.5, novelty=0.5)
    low = _fake_candidate("lo", effect=0.1, conf=0.1, actionability=0.1, novelty=0.1)

    ranked = heuristic.rank_candidates([low, high, mid])
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].candidate.id == "hi"
    assert ranked[-1].candidate.id == "lo"


def test_rank_stable_tiebreak_by_id():
    """Equal scores tie-break by id (lexicographic)."""
    a = _fake_candidate("aaa", effect=0.5)
    b = _fake_candidate("bbb", effect=0.5)
    ranked = heuristic.rank_candidates([b, a])
    assert ranked[0].candidate.id == "aaa"
    assert ranked[1].candidate.id == "bbb"


# ─────────────────────────────────────────────────────────────────────────
# Cap enforcement
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_can_surface_today_allows_fresh_user(db):
    today = date.today()
    allowed, reason = await heuristic.can_surface_today(db, USER, today)
    assert allowed
    assert reason == "ok"


@pytest.mark.asyncio
async def test_can_surface_today_blocks_after_one_card_today(db):
    today = date.today()
    db.add(
        MLRanking(
            user_id=USER,
            surface_date=today.isoformat(),
            candidate_id="c1",
            rank=1,
            score=0.8,
            ranker_version="heuristic-1.0.0",
            was_shown=True,
        )
    )
    await db.flush()
    allowed, reason = await heuristic.can_surface_today(db, USER, today)
    assert not allowed
    assert "daily cap" in reason


@pytest.mark.asyncio
async def test_can_surface_today_blocks_after_three_cards_in_a_week(db):
    today = date.today()
    # Three surfaced cards across the last 4 days, none today.
    for i, cid in enumerate(["c1", "c2", "c3"]):
        db.add(
            MLRanking(
                user_id=USER,
                surface_date=(today - timedelta(days=i + 1)).isoformat(),
                candidate_id=cid,
                rank=1,
                score=0.8,
                ranker_version="heuristic-1.0.0",
                was_shown=True,
            )
        )
    await db.flush()

    allowed, reason = await heuristic.can_surface_today(db, USER, today)
    assert not allowed
    assert "weekly cap" in reason


@pytest.mark.asyncio
async def test_can_surface_today_ignores_shadow_rows(db):
    """Rows with ``was_shown=False`` should not count toward caps."""
    today = date.today()
    for i, cid in enumerate(["c1", "c2", "c3"]):
        db.add(
            MLRanking(
                user_id=USER,
                surface_date=(today - timedelta(days=i + 1)).isoformat(),
                candidate_id=cid,
                rank=1,
                score=0.8,
                ranker_version="heuristic-1.0.0",
                was_shown=False,
            )
        )
    await db.flush()
    allowed, _ = await heuristic.can_surface_today(db, USER, today)
    assert allowed


# ─────────────────────────────────────────────────────────────────────────
# materialize_daily_ranking
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_materialize_daily_ranking_persists_top_n(db):
    today = date.today()
    cands = [
        _fake_candidate("c1", effect=0.9),
        _fake_candidate("c2", effect=0.5),
        _fake_candidate("c3", effect=0.2),
    ]
    ranked = await heuristic.materialize_daily_ranking(db, USER, cands, today, keep_top_n=3)
    assert [r.rank for r in ranked] == [1, 2, 3]

    rows = (await db.execute(select(MLRanking).where(MLRanking.user_id == USER))).scalars().all()
    # 3 ranks written.
    assert len(rows) == 3
    # None shown yet — shadow log.
    assert all(r.was_shown is False for r in rows)


@pytest.mark.asyncio
async def test_materialize_daily_ranking_is_idempotent(db):
    today = date.today()
    cands = [_fake_candidate("c1", effect=0.9), _fake_candidate("c2", effect=0.5)]

    await heuristic.materialize_daily_ranking(db, USER, cands, today)
    first = (await db.execute(select(MLRanking).where(MLRanking.user_id == USER))).scalars().all()

    # Rerun same day -> no-op.
    result = await heuristic.materialize_daily_ranking(db, USER, cands, today)
    second = (await db.execute(select(MLRanking).where(MLRanking.user_id == USER))).scalars().all()

    assert result == []  # second call returns empty
    assert len(first) == len(second)


@pytest.mark.asyncio
async def test_materialize_with_no_candidates_writes_nothing(db):
    today = date.today()
    result = await heuristic.materialize_daily_ranking(db, USER, [], today)
    assert result == []
    rows = (await db.execute(select(MLRanking).where(MLRanking.user_id == USER))).scalars().all()
    assert rows == []
