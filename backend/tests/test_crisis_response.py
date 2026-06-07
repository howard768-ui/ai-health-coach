"""Audit B2/H14: a crisis response must surface crisis resources and contain
ZERO health data. Detection is well covered in test_safety_check.py; this pins
the code-enforced crisis OUTPUT path (the API-error fallback), which must never
echo the user's metrics.

Run: cd backend && uv run pytest tests/test_crisis_response.py -v
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-long-enough-aaaaaaaaaaaa")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-used")

import anthropic
import httpx
import pytest

from app.services.coach_engine import CoachEngine


def test_crisis_fallback_has_resources_and_no_health_data():
    """When the user is in crisis and the model call fails, the response must
    still give crisis resources (988) and must NOT contain any health metric.
    """
    coach = CoachEngine()

    def boom(**kwargs):
        raise anthropic.APIConnectionError(
            message="down", request=httpx.Request("POST", "http://test")
        )

    coach.client.messages.create = boom

    # Distinctive, implausible metric values: if any leak into the crisis
    # response we will see them.
    out = coach.process_query(
        query="I want to end my life.",
        health_data={
            "readiness_score": 7733,
            "hrv_average": 8811.0,
            "resting_hr": 9955.0,
            "sleep_efficiency": 6644.0,
        },
        user_name="Brock",
    )

    resp = out["response"]
    assert "988" in resp, "crisis resources must be surfaced"
    for marker in ("7733", "8811", "9955", "6644"):
        assert marker not in resp, f"health metric {marker} leaked into crisis response"
    assert out["safety"]["is_concerning"] is True


def test_non_crisis_api_failure_does_not_emit_crisis_resources():
    """A normal query whose model call fails gets a generic retry message, not
    crisis resources (avoids crying wolf)."""
    coach = CoachEngine()

    def boom(**kwargs):
        raise anthropic.APIConnectionError(
            message="down", request=httpx.Request("POST", "http://test")
        )

    coach.client.messages.create = boom

    out = coach.process_query(
        query="what is my resting heart rate?",
        health_data={"resting_hr": 62, "readiness_score": 70},
        user_name="Brock",
    )
    assert "988" not in out["response"]
