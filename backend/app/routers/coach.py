import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select, desc

from app.api.deps import CurrentUser
from app.core.ratelimit import limiter
from app.database import get_db
from app.models.chat import Conversation, ChatMessageRecord
from app.models.user import User
from app.services.coach_engine import CoachEngine
from app.services.content_blocks import (
    ContentBlock,
    TextBlock,
    flatten_to_markdown,
    parse_content_blocks,
    sanitize_output,
)
from app.services.health_data import get_latest_health_data
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.coach")

# Rate limit AI endpoints to prevent Claude API budget exhaustion (P1-4).
# Uses the single shared limiter (app.core.ratelimit), keyed on the real client
# IP, not the Cloudflare edge IP. Audit C2.

router = APIRouter(prefix="/api/coach", tags=["coach"])

engine = CoachEngine()


class ChatRequest(BaseModel):
    message: str


# Phase 5: /api/coach/explain-finding shapes. Used by the iOS "Why?"
# button on a SignalInsightCard. Runs SHAP-like explanation through
# ml.api.explain_insight and narrates the result with Opus.

class ExplainFindingRequest(BaseModel):
    insight_id: str


class ExplainFindingContribution(BaseModel):
    feature: str
    contribution: float
    observed_value: float | None = None
    baseline_value: float | None = None


class ExplainFindingResponse(BaseModel):
    insight_id: str
    kind: str
    narration: str
    narration_used_fallback: bool
    contributions: list[ExplainFindingContribution]
    historical_examples: list[dict]


class ChatResponse(BaseModel):
    role: str
    content: str  # Plain markdown with [[data:...]] tags flattened to bold inline.
    blocks: list[ContentBlock] = []  # Structured rendering for rich clients.
    message_id: int | None = None
    routing: dict | None = None
    safety: dict | None = None
    model_used: str | None = None


class HistoryMessage(BaseModel):
    id: int
    role: str
    content: str
    blocks: list[ContentBlock] = []
    model_used: str | None = None
    routing_tier: str | None = None
    created_at: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]
    conversation_id: int


# ============================================================
# CHAT — Send message, persist both user + coach messages
# ============================================================

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Send a message, get AI response, persist both to DB.

    Rate-limited to 30/min per IP — about one message every 2s. Generous
    enough for normal back-and-forth but caps Claude budget exhaustion.
    """

    user_id = current_user.apple_user_id

    # Get or create conversation
    conv = await _get_or_create_conversation(db, user_id)

    # Load recent history from DB for context (last 20 messages)
    history_records = await _get_recent_messages(db, conv.id, limit=20)
    # Map DB roles to Claude API roles (coach → assistant)
    history = [{"role": "assistant" if m.role == "coach" else "user", "content": m.content} for m in history_records]

    # Save user message
    user_msg = ChatMessageRecord(
        conversation_id=conv.id,
        user_id=user_id,
        role="user",
        content=body.message,
    )
    db.add(user_msg)
    await db.flush()

    # Load reconciled health data (multi-source: Oura + Apple Health + Garmin)
    health_data = await get_latest_health_data(db, user_id)

    # Load user profile for personalized name + goals + free-form goal text
    user_name, user_goals, custom_goal_text = await _load_user_context(db, user_id)

    # Phase 5: load Signal Engine context (active patterns, recent anomalies,
    # personal forecasts). Pre-loaded here in the async request path so the
    # sync CoachEngine.process_query can stay sync. Defensive: load_coach_
    # signal_context swallows per-loader exceptions and returns an empty
    # SignalContext, so a broken ML table never degrades the coach.
    from ml import api as ml_api

    signal_context = await ml_api.load_coach_signal_context(db, user_id)

    # Process through coach engine (with conversation history)
    # Wrap in to_thread because process_query() calls synchronous Anthropic SDK
    import asyncio
    start_time = time.monotonic()
    result = await asyncio.to_thread(
        engine.process_query,
        body.message,
        health_data,
        user_name,
        user_goals,
        custom_goal_text,
        history,
        signal_context,
    )
    latency_ms = int((time.monotonic() - start_time) * 1000)

    # Scrub em dashes (prompt forbids but model sometimes slips); preserve
    # everything else (markdown, data tags) so the client can render richly.
    raw_response = sanitize_output(result["response"])
    blocks = parse_content_blocks(raw_response)
    flat_content = flatten_to_markdown(raw_response)
    result["response"] = flat_content

    # Serialize health context for replay (offline eval)
    import json
    health_context_str = json.dumps(health_data) if health_data else None

    # Save coach response with production monitoring fields. Store the RAW
    # (sanitized) output so history can re-parse blocks with updated logic.
    coach_msg = ChatMessageRecord(
        conversation_id=conv.id,
        user_id=user_id,
        role="coach",
        content=raw_response,
        routing_tier=result.get("routing", {}).get("tier"),
        routing_reason=result.get("routing", {}).get("reason"),
        model_used=result.get("model_used"),
        safety_flagged=result.get("safety", {}).get("is_concerning", False),
        latency_ms=latency_ms,
        input_tokens=result.get("tokens", {}).get("input"),
        output_tokens=result.get("tokens", {}).get("output"),
        health_context=health_context_str,
        prompt_version="v2",  # v2: markdown allowed, data tags, BLUF, no em dashes
    )
    db.add(coach_msg)

    # Update conversation timestamp BEFORE commit so both writes land in
    # the same transaction. This was previously two separate commits, which
    # could leave conv.updated_at stale if the second commit failed.
    conv.updated_at = utcnow_naive()

    try:
        await db.commit()
        await db.refresh(coach_msg)
    except SQLAlchemyError as e:
        # The Claude call already succeeded. Roll back the failed write but
        # still return the AI response so the user sees something.
        await db.rollback()
        logger.error("Failed to persist coach response: %s", e)
        return ChatResponse(
            role="coach",
            content=flat_content,
            blocks=blocks,
            message_id=None,
            routing=result.get("routing"),
            safety=result.get("safety"),
            model_used=result.get("model_used"),
        )

    return ChatResponse(
        role="coach",
        content=flat_content,
        blocks=blocks,
        message_id=coach_msg.id,
        routing=result.get("routing"),
        safety=result.get("safety"),
        model_used=result.get("model_used"),
    )


# ============================================================
# FEEDBACK — Thumbs up/down on coach responses
# ============================================================

class FeedbackRequest(BaseModel):
    message_id: int
    feedback: str  # "up" or "down"


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Record thumbs up/down on a coach message owned by the caller."""
    if request.feedback not in ("up", "down"):
        raise HTTPException(status_code=400, detail="feedback must be 'up' or 'down'")

    result = await db.execute(
        select(ChatMessageRecord).where(
            ChatMessageRecord.id == request.message_id,
            ChatMessageRecord.role == "coach",
            ChatMessageRecord.user_id == current_user.apple_user_id,
        )
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="message not found")

    msg.feedback = request.feedback
    msg.feedback_at = utcnow_naive()
    await db.commit()

    return {"status": "ok", "message_id": request.message_id, "feedback": request.feedback}


# ============================================================
# EXPLAIN-FINDING (Phase 5) — why did this insight surface?
# ============================================================
#
# POST /api/coach/explain-finding
#
# The iOS "Why?" button on a SignalInsightCard hits this. We look up the
# candidate by id, dispatch to ml.api.explain_insight (SHAP-style
# attribution), and narrate the result with Opus via ml.api.narrate_insight.
# Owner check enforces that the insight belongs to the authenticated user.
#
# Shadow mode: this endpoint is always live. Shadow_insight_card gates
# whether iOS SHOWS the Why button, not whether we respond to the call.


@router.post("/explain-finding", response_model=ExplainFindingResponse)
@limiter.limit("30/minute")
async def explain_finding(
    request: Request,
    body: ExplainFindingRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> ExplainFindingResponse:
    """Return a SHAP-backed attribution + Opus narration for one insight."""
    from ml import api as ml_api

    try:
        explanation = await ml_api.explain_insight(
            db, current_user.apple_user_id, body.insight_id
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="insight not found")

    narration = await ml_api.narrate_insight(
        db, current_user.apple_user_id, body.insight_id
    )

    contributions = [
        ExplainFindingContribution(
            feature=feature,
            contribution=value,
            observed_value=None,
            baseline_value=None,
        )
        for feature, value in explanation.top_contributing_features
    ]

    return ExplainFindingResponse(
        insight_id=explanation.insight_id,
        kind=explanation.explanation_kind,
        narration=narration.text,
        narration_used_fallback=narration.used_fallback,
        contributions=contributions,
        historical_examples=explanation.historical_examples,
    )


# ============================================================
# HISTORY — Retrieve persisted conversation
# ============================================================

@router.get("/history", response_model=HistoryResponse)
async def get_history(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve full conversation history for the current user."""

    user_id = current_user.apple_user_id

    conv = await _get_or_create_conversation(db, user_id)
    records = await _get_recent_messages(db, conv.id, limit=100)

    # Parse blocks and flatten content on read so history renders richly
    # whether the message was stored pre-v2 (plain text) or post-v2 (with
    # [[data:...]] tags). User messages have no tags and just become a
    # single text block.
    messages = []
    for m in records:
        if m.role == "coach":
            blocks = parse_content_blocks(m.content)
            flat = flatten_to_markdown(m.content)
        else:
            blocks = [TextBlock(value=m.content)]
            flat = m.content
        messages.append(
            HistoryMessage(
                id=m.id,
                role=m.role,
                content=flat,
                blocks=blocks,
                model_used=m.model_used,
                routing_tier=m.routing_tier,
                created_at=m.created_at.isoformat(),
            )
        )

    return HistoryResponse(messages=messages, conversation_id=conv.id)


# ============================================================
# INSIGHT — Daily dashboard insight
# ============================================================

@router.post("/insight")
async def generate_insight(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Generate daily dashboard insight via the engine pipeline."""
    import asyncio
    user_id = current_user.apple_user_id
    health_data = await get_latest_health_data(db, user_id)
    _, user_goals, _ = await _load_user_context(db, user_id)
    # generate_daily_insight calls the sync Anthropic SDK — must offload
    result = await asyncio.to_thread(
        engine.generate_daily_insight,
        health_data,
        user_goals,
    )
    return result


# ============================================================
# ANALYTICS — Production monitoring dashboard
# ============================================================

@router.get("/analytics")
async def get_analytics(
    current_user: CurrentUser,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """Production monitoring: feedback rates, latency, model usage, safety flags.

    Scoped to the current user's own coach messages.
    """
    from datetime import timedelta
    cutoff = utcnow_naive() - timedelta(days=days)

    # Only the current user's coach messages in the window
    result = await db.execute(
        select(ChatMessageRecord).where(
            ChatMessageRecord.role == "coach",
            ChatMessageRecord.user_id == current_user.apple_user_id,
            ChatMessageRecord.created_at >= cutoff,
        )
    )
    messages = list(result.scalars().all())

    if not messages:
        return {"period_days": days, "total_responses": 0}

    thumbs_up = sum(1 for m in messages if m.feedback == "up")
    thumbs_down = sum(1 for m in messages if m.feedback == "down")
    no_feedback = sum(1 for m in messages if m.feedback is None)
    safety_flagged = sum(1 for m in messages if m.safety_flagged)

    latencies = [m.latency_ms for m in messages if m.latency_ms]
    avg_latency = round(sum(latencies) / len(latencies)) if latencies else None
    # Nearest-rank p95 — clamp index to valid range so it works for any n >= 1
    if latencies:
        sorted_latencies = sorted(latencies)
        import math
        p95_index = max(0, min(len(sorted_latencies) - 1, math.ceil(0.95 * len(sorted_latencies)) - 1))
        p95_latency = sorted_latencies[p95_index]
    else:
        p95_latency = None

    tokens_in = [m.input_tokens for m in messages if m.input_tokens]
    tokens_out = [m.output_tokens for m in messages if m.output_tokens]

    # Model usage breakdown
    model_counts = {}
    for m in messages:
        key = m.model_used or "rules"
        model_counts[key] = model_counts.get(key, 0) + 1

    # Routing tier breakdown
    tier_counts = {}
    for m in messages:
        key = m.routing_tier or "unknown"
        tier_counts[key] = tier_counts.get(key, 0) + 1

    # Prompt version breakdown (for A/B testing)
    version_feedback = {}
    for m in messages:
        ver = m.prompt_version or "unknown"
        if ver not in version_feedback:
            version_feedback[ver] = {"up": 0, "down": 0, "none": 0, "total": 0}
        version_feedback[ver]["total"] += 1
        if m.feedback == "up":
            version_feedback[ver]["up"] += 1
        elif m.feedback == "down":
            version_feedback[ver]["down"] += 1
        else:
            version_feedback[ver]["none"] += 1

    return {
        "period_days": days,
        "total_responses": len(messages),
        "feedback": {
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "no_feedback": no_feedback,
            "satisfaction_rate": round(thumbs_up / max(thumbs_up + thumbs_down, 1) * 100, 1),
        },
        "safety": {
            "flagged_count": safety_flagged,
            "flagged_rate": round(safety_flagged / len(messages) * 100, 1),
        },
        "latency": {
            "avg_ms": avg_latency,
            "p95_ms": p95_latency,
        },
        "tokens": {
            "avg_input": round(sum(tokens_in) / len(tokens_in)) if tokens_in else None,
            "avg_output": round(sum(tokens_out) / len(tokens_out)) if tokens_out else None,
            "total_input": sum(tokens_in),
            "total_output": sum(tokens_out),
        },
        "model_usage": model_counts,
        "routing_tiers": tier_counts,
        "prompt_versions": version_feedback,
    }


# ============================================================
# HELPERS
# ============================================================

async def _get_or_create_conversation(db: AsyncSession, user_id: str) -> Conversation:
    """Get the user's active conversation or create one."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(desc(Conversation.updated_at))
        .limit(1)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        conv = Conversation(user_id=user_id)
        db.add(conv)
        await db.flush()
    return conv


async def _load_user_context(
    db: AsyncSession, user_id: str
) -> tuple[str, list[str], str | None]:
    """Load the user's first name, goals, and free-form goal text.

    Falls back to sensible defaults if no profile exists (e.g. after DB reset).
    The free-form text is the "Want to share more?" answer from onboarding;
    it flows into the coach system prompt as additional context.
    """
    import json
    result = await db.execute(select(User).where(User.apple_user_id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        return "there", [], None  # Neutral greeting, no assumed goals

    first_name = (user.name.split()[0] if user.name else "there")

    goals: list[str] = []
    if user.goals:
        try:
            parsed = json.loads(user.goals) if isinstance(user.goals, str) else user.goals
            if isinstance(parsed, list):
                goals = [str(g) for g in parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    return first_name, goals, user.custom_goal_text


async def _get_recent_messages(db: AsyncSession, conversation_id: int, limit: int = 20) -> list[ChatMessageRecord]:
    """Get recent messages for a conversation, oldest first."""
    result = await db.execute(
        select(ChatMessageRecord)
        .where(ChatMessageRecord.conversation_id == conversation_id)
        .order_by(desc(ChatMessageRecord.created_at))
        .limit(limit)
    )
    messages = list(result.scalars().all())
    messages.reverse()  # Return oldest first for context
    return messages
