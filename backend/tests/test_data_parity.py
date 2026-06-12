"""Data-to-prompt parity tests.

Verifies that every metric returned by health_data.py actually appears
in the system prompt built by coach_engine.py. This prevents the class of
bugs where data exists in the DB but never reaches the coach.

Run: cd backend && uv run python -m pytest tests/test_data_parity.py -v
"""

import json
import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from app.services.coach_engine import EVIDENCE_BOUND_SYSTEM_PROMPT


def _build_prompt(health_data: dict) -> str:
    """Build the system prompt the same way CoachEngine.process_query does.

    Phase 5 replaced the single ``knowledge_graph_context`` slot with three
    dynamic sections (``active_patterns``, ``recent_anomalies``,
    ``personal_forecast``). These data-parity tests exercise the health_data
    block only, so the Signal Engine sections are passed as empty strings.
    Dedicated Phase 5 tests cover prompt assembly with non-empty SignalContext.
    """
    return EVIDENCE_BOUND_SYSTEM_PROMPT.format(
        user_name="TestUser",
        health_data=json.dumps(health_data, indent=2),
        goals="general wellness",
        custom_goal_context="",
        memory_context="",
        active_patterns="",
        recent_anomalies="",
        personal_forecast="",
        safety_disclaimer="",
    )


# A realistic health data dict as returned by get_latest_health_data()
SAMPLE_HEALTH_DATA = {
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
    "data_sources": {"sleep_efficiency": "oura", "steps": "apple_health"},
    "baseline_days": 7,
}


def test_all_health_keys_in_prompt():
    """Every key from get_latest_health_data() output appears in the prompt JSON."""
    prompt = _build_prompt(SAMPLE_HEALTH_DATA)
    for key in SAMPLE_HEALTH_DATA:
        assert f'"{key}"' in prompt, f"Key '{key}' not found in system prompt"


def test_deep_sleep_in_prompt():
    """deep_sleep_minutes value from DB appears verbatim in prompt."""
    prompt = _build_prompt(SAMPLE_HEALTH_DATA)
    assert '"deep_sleep_minutes": 48' in prompt


def test_steps_in_prompt():
    """steps value appears in prompt."""
    prompt = _build_prompt(SAMPLE_HEALTH_DATA)
    assert '"steps": 4201' in prompt


def test_baselines_in_prompt():
    """baseline_hrv and baseline_rhr appear in prompt."""
    prompt = _build_prompt(SAMPLE_HEALTH_DATA)
    assert '"baseline_hrv": 42.5' in prompt
    assert '"baseline_rhr": 59.3' in prompt


def test_sources_in_prompt():
    """data_sources dict included in prompt."""
    prompt = _build_prompt(SAMPLE_HEALTH_DATA)
    assert '"data_sources"' in prompt
    assert '"apple_health"' in prompt
    assert '"oura"' in prompt


def test_zero_values_in_prompt():
    """When a metric is 0, it appears as 0 in prompt (not omitted)."""
    data = {**SAMPLE_HEALTH_DATA, "hrv_average": 0, "steps": 0, "deep_sleep_minutes": 0}
    prompt = _build_prompt(data)
    assert '"hrv_average": 0' in prompt
    assert '"steps": 0' in prompt
    assert '"deep_sleep_minutes": 0' in prompt


def test_prompt_json_is_parseable():
    """The health data JSON block in the prompt is valid JSON."""
    prompt = _build_prompt(SAMPLE_HEALTH_DATA)
    # Extract the JSON block between the markers
    marker = "USER'S HEALTH DATA (cite these values):"
    idx = prompt.index(marker) + len(marker)
    # Find the end of the JSON (next section marker)
    end_marker = "USER'S GOALS:"
    end_idx = prompt.index(end_marker)
    json_str = prompt[idx:end_idx].strip()
    parsed = json.loads(json_str)
    assert parsed["sleep_efficiency"] == 77
    assert parsed["deep_sleep_minutes"] == 48
    assert parsed["steps"] == 4201


def test_empty_data_in_prompt():
    """Empty health data produces valid empty JSON in prompt."""
    prompt = _build_prompt({})
    assert '{}' in prompt
