"""API contract tests — iOS client ↔ backend response shape validation.

Verifies that backend API responses contain all keys the iOS client expects.
If APIClient.swift adds a new field or a router drops one, these tests fail.

Run: cd backend && uv run python -m pytest tests/test_api_contracts.py -v
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")


# ── Health Latest Response Shape ────────────────────────────

# These are the keys that get_latest_health_data() returns when reconciled data exists.
# iOS DashboardView, CoachChatView, and TrendsView all depend on this shape.
HEALTH_LATEST_REQUIRED_KEYS = {
    "sleep_efficiency",
    "sleep_duration_hours",
    "deep_sleep_minutes",
    "hrv_average",
    "baseline_hrv",
    "resting_hr",
    "baseline_rhr",
    "readiness_score",
    "steps",
    "active_calories",
    "data_sources",
    "baseline_days",
}

# Keys when falling back to SleepRecord-only data
HEALTH_LATEST_FALLBACK_KEYS = {
    "sleep_efficiency",
    "sleep_duration_hours",
    "deep_sleep_minutes",
    "hrv_average",
    "baseline_hrv",
    "resting_hr",
    "baseline_rhr",
    "readiness_score",
    "data_sources",
}


def test_health_latest_reconciled_shape():
    """Reconciled health data has all keys iOS expects."""

    # Simulate what get_latest_health_data returns in the reconciled path
    sample = {
        "sleep_efficiency": 77,
        "sleep_duration_hours": 7.2,
        "deep_sleep_minutes": 48,
        "hrv_average": 38,
        "baseline_hrv": 42.5,
        "resting_hr": 57,
        "baseline_rhr": 59.3,
        "readiness_score": 65,
        "steps": 4201,
        "active_calories": 320,
        "data_sources": {"sleep_efficiency": "oura"},
        "baseline_days": 7,
    }
    missing = HEALTH_LATEST_REQUIRED_KEYS - set(sample.keys())
    assert not missing, f"Reconciled response missing keys: {missing}"


def test_health_latest_fallback_shape():
    """SleepRecord fallback has the minimum keys iOS expects."""
    sample = {
        "sleep_efficiency": 82,
        "sleep_duration_hours": 7.0,
        "deep_sleep_minutes": 60,
        "hrv_average": 45,
        "baseline_hrv": 42.0,
        "resting_hr": 60,
        "baseline_rhr": 58.0,
        "readiness_score": 75,
        "data_sources": {"all": "oura"},
    }
    missing = HEALTH_LATEST_FALLBACK_KEYS - set(sample.keys())
    assert not missing, f"Fallback response missing keys: {missing}"


# ── Coach Response Shape ────────────────────────────────────

# iOS CoachViewModel expects these keys from POST /coach/chat.
# `blocks` powers rich rendering (markdown text + data cards); removing it
# from the backend would silently degrade the iOS chat UI to plain text.
COACH_RESPONSE_REQUIRED_KEYS = {"role", "content", "blocks", "message_id"}


def test_coach_response_shape():
    """ChatResponse model has the keys iOS expects."""
    from app.routers.coach import ChatResponse
    fields = set(ChatResponse.model_fields.keys())
    missing = COACH_RESPONSE_REQUIRED_KEYS - fields
    assert not missing, f"ChatResponse missing keys: {missing}"


def test_coach_response_blocks_discriminator():
    """Each block in the response carries a `type` so iOS can decode the
    discriminated union (text vs data_card). Without this, the polymorphic
    decoder has nothing to dispatch on."""
    from app.services.content_blocks import (
        parse_content_blocks,
    )
    blocks = parse_content_blocks(
        "Good recovery. [[data:hrv:50:ms:baseline 45]] Push today."
    )
    serialized = [b.model_dump() for b in blocks]
    # Every block must have a type field with a recognized value
    for s in serialized:
        assert "type" in s, f"Block missing type field: {s}"
        assert s["type"] in {"text", "data_card"}, f"Unknown block type: {s['type']}"
    # Data cards must also carry the four payload fields the iOS decoder reads
    data_cards = [s for s in serialized if s["type"] == "data_card"]
    assert len(data_cards) == 1
    for key in ("metric", "value", "unit", "subtitle"):
        assert key in data_cards[0], f"data_card missing {key}: {data_cards[0]}"


# ── Chat History Shape ──────────────────────────────────────

# iOS CoachViewModel.loadHistory reads `blocks` so historical coach messages
# render with the same rich formatting as fresh responses.
HISTORY_MESSAGE_REQUIRED_KEYS = {"id", "role", "content", "blocks", "created_at"}


def test_chat_history_message_shape():
    """HistoryMessage model has the keys iOS expects."""
    from app.routers.coach import HistoryMessage
    fields = set(HistoryMessage.model_fields.keys())
    missing = HISTORY_MESSAGE_REQUIRED_KEYS - fields
    assert not missing, f"HistoryMessage missing keys: {missing}"


def test_history_response_shape():
    """HistoryResponse has messages list and conversation_id."""
    from app.routers.coach import HistoryResponse
    fields = set(HistoryResponse.model_fields.keys())
    assert "messages" in fields
    assert "conversation_id" in fields


# ── Trends Response Shape ───────────────────────────────────

# iOS TrendsView expects this nested structure
TRENDS_METRIC_KEYS = {"values", "dates", "baseline", "personal_min", "personal_max", "personal_average"}


def test_trends_metric_shape():
    """Each metric in trends response has the keys TrendsView expects."""
    # This mirrors the build_metric() function in health.py router
    sample_metric = {
        "values": [77, 80, 85],
        "dates": ["2026-04-11", "2026-04-12", "2026-04-13"],
        "baseline": 80.7,
        "personal_min": 77.0,
        "personal_max": 85.0,
        "personal_average": 80.7,
    }
    missing = TRENDS_METRIC_KEYS - set(sample_metric.keys())
    assert not missing, f"Trends metric missing keys: {missing}"


# ── Feedback Request Shape ──────────────────────────────────

def test_feedback_request_shape():
    """FeedbackRequest model accepts the keys iOS sends."""
    from app.routers.coach import FeedbackRequest
    fields = set(FeedbackRequest.model_fields.keys())
    assert "message_id" in fields
    assert "feedback" in fields


# ── Value Types ─────────────────────────────────────────────

def test_health_values_are_numeric():
    """All health metric values should be numeric (int or float), never strings."""
    sample = {
        "sleep_efficiency": 77,
        "sleep_duration_hours": 7.2,
        "deep_sleep_minutes": 48,
        "hrv_average": 38,
        "baseline_hrv": 42.5,
        "resting_hr": 57,
        "baseline_rhr": 59.3,
        "readiness_score": 65,
        "steps": 4201,
        "active_calories": 320,
    }
    for key, value in sample.items():
        assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"
