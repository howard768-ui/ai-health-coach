"""Audit P3/C3: notification generation must pin its model tier explicitly.

Notification prompts are static templates, not user text. Fed through
``Deliberator.route`` they match the keyword router's cross-domain regex
("Connect TWO health domains", "PATTERN" in COACHING_NUDGE_PROMPT), so every
coaching nudge billed at Opus rates, and morning/bedtime tier choice varied
with whichever metrics happened to be present (nondeterministic cost).

The fix: ``process_query(model_tier=...)`` bypasses keyword routing; safety
can still escalate the explicit tier up to Opus, never down. These tests pin:
 - the leak mechanism itself (template text trips the cross-domain regex)
 - nudge + morning brief now bill Sonnet
 - concerning health data still escalates an explicit Sonnet call to Opus

Run: cd backend && ./.venv/bin/python -m pytest tests/test_notification_model_tier.py -v
"""

import os
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-long-enough-aaaaaaaaaaaa")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-used")

from app.config import settings
from app.services.coach_engine import (
    CoachEngine,
    Deliberator,
    ModelTier,
    SafetyCheck,
)
from app.services.notification_content import (
    COACHING_NUDGE_PROMPT,
    SHARED_RULES,
    content_generator,
)
from app.services.notification_engine import notification_engine


def _fake_create(captured: dict):
    """Stand-in for anthropic messages.create that records its kwargs."""

    def fake(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text='{"title": "Hi", "body": "All good."}')],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )

    return fake


def _benign_health_data() -> dict:
    return {"readiness_score": 80, "sleep_efficiency": 90.0}


# ── The leak mechanism (pins WHY explicit tiers are required) ────────────


def test_nudge_template_text_trips_cross_domain_regex_to_opus():
    """The static nudge template routes to Opus through keyword routing.

    This is the cost leak: if a future refactor drops the explicit
    model_tier and feeds template text back through Deliberator.route,
    this documents that the router WILL bill Opus for it.
    """
    prompt = COACHING_NUDGE_PROMPT.format(
        shared_rules=SHARED_RULES,
        health_data="Recovery: good",
        connections="x",
    )
    safety = SafetyCheck(
        is_concerning=False, reasons=[], requires_disclaimer=False, requires_opus=False
    )
    routing = Deliberator.route(prompt, _benign_health_data(), safety)
    assert routing.tier is ModelTier.OPUS, (
        "expected the static template to trip the cross-domain regex; if this "
        "fails the router changed and the explicit-tier rationale should be revisited"
    )


# ── Nudge + morning brief bill Sonnet now ────────────────────────────────


def test_coaching_nudge_bills_sonnet_not_opus():
    captured: dict = {}
    content_generator.coach.client.messages.create = _fake_create(captured)

    result = content_generator.generate_coaching_nudge(_benign_health_data())

    assert captured.get("model") == settings.anthropic_model_sonnet, (
        f"nudge billed {captured.get('model')!r}, expected Sonnet"
    )
    assert result["category"] == "coaching_nudge"


def test_morning_brief_bills_sonnet_not_opus():
    captured: dict = {}
    notification_engine.coach.client.messages.create = _fake_create(captured)

    result = notification_engine.generate_morning_brief(_benign_health_data())

    assert captured.get("model") == settings.anthropic_model_sonnet, (
        f"morning brief billed {captured.get('model')!r}, expected Sonnet"
    )
    assert result["category"] == "morning_brief"


def test_bedtime_coaching_bills_sonnet_not_opus():
    captured: dict = {}
    content_generator.coach.client.messages.create = _fake_create(captured)

    content_generator.generate_bedtime_coaching(_benign_health_data())

    assert captured.get("model") == settings.anthropic_model_sonnet


# ── Safety still escalates an explicit tier ──────────────────────────────


def test_concerning_health_data_escalates_explicit_sonnet_to_opus():
    """Safety-critical -> Opus is non-negotiable, even with an explicit tier."""
    coach = CoachEngine()
    captured: dict = {}
    coach.client.messages.create = _fake_create(captured)

    # HRV < 20ms trips the deterministic safety gate (requires_opus=True).
    result = coach.process_query(
        query="static template text",
        health_data={"hrv_average": 12, "readiness_score": 80},
        model_tier=ModelTier.SONNET,
    )

    assert captured.get("model") == settings.anthropic_model_opus, (
        "concerning health data must escalate an explicit Sonnet tier to Opus"
    )
    assert result["routing"]["tier"] == "opus"
    assert result["routing"]["safety_flag"] is True


def test_explicit_tier_bypasses_rules_branch():
    """An explicit tier never returns a canned rules answer; the query is a
    template, not a user question the rules engine should intercept."""
    coach = CoachEngine()
    captured: dict = {}
    coach.client.messages.create = _fake_create(captured)

    result = coach.process_query(
        # Without model_tier, a greeting like this can short-circuit to RULES.
        query="hello",
        health_data=_benign_health_data(),
        model_tier=ModelTier.SONNET,
    )

    assert result["model_used"] != "rules"
    assert captured.get("model") == settings.anthropic_model_sonnet
