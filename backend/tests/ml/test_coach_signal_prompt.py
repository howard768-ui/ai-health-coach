"""Tests for Phase 5 prompt rendering — SignalContext into EVIDENCE_BOUND_SYSTEM_PROMPT.

Pure string-assembly tests (no DB, no LLM). Verifies that the three new
prompt sections render correctly when SignalContext is populated, and
stay silent when it is empty.

Run: ``cd backend && uv run python -m pytest tests/ml/test_coach_signal_prompt.py -v``
"""

from __future__ import annotations

import os


os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from app.services.coach_engine import (
    _render_active_patterns,
    _render_personal_forecast,
    _render_recent_anomalies,
)
from ml.api import (
    ActivePattern,
    PersonalForecast,
    RecentAnomaly,
    SignalContext,
)


# ─────────────────────────────────────────────────────────────────────────
# _render_active_patterns
# ─────────────────────────────────────────────────────────────────────────


def test_render_active_patterns_empty_context_returns_empty_string():
    assert _render_active_patterns(None) == ""
    assert _render_active_patterns(SignalContext()) == ""


def test_render_active_patterns_with_description_uses_description_verbatim():
    ctx = SignalContext(
        active_patterns=[
            ActivePattern(
                source_metric="protein_intake",
                target_metric="deep_sleep_seconds",
                direction="positive",
                strength=0.55,
                confidence_tier="literature_supported",
                sample_size=60,
                effect_description="Higher protein tends to mean longer deep sleep.",
                literature_ref="10.1007/s40279-014-0260-0",
            )
        ]
    )
    out = _render_active_patterns(ctx)
    assert "Higher protein tends to mean longer deep sleep." in out
    assert "n=60" in out
    assert "10.1007/s40279-014-0260-0" in out


def test_render_active_patterns_without_description_composes_fallback():
    ctx = SignalContext(
        active_patterns=[
            ActivePattern(
                source_metric="steps",
                target_metric="sleep_efficiency",
                direction="positive",
                strength=0.6,
                confidence_tier="developing",
                sample_size=40,
                effect_description="",
            )
        ]
    )
    out = _render_active_patterns(ctx)
    assert "steps" in out
    assert "sleep efficiency" in out
    assert "higher too" in out or "tends to be higher" in out


def test_render_active_patterns_negative_direction():
    ctx = SignalContext(
        active_patterns=[
            ActivePattern(
                source_metric="resting_hr",
                target_metric="readiness_score",
                direction="negative",
                strength=0.5,
                confidence_tier="developing",
                sample_size=30,
                effect_description="",
            )
        ]
    )
    out = _render_active_patterns(ctx)
    assert "lower" in out


def test_render_active_patterns_emits_tier_and_sample_size():
    ctx = SignalContext(
        active_patterns=[
            ActivePattern(
                source_metric="s",
                target_metric="t",
                direction="positive",
                strength=0.8,
                confidence_tier="causal_candidate",
                sample_size=80,
                effect_description="desc",
            )
        ]
    )
    out = _render_active_patterns(ctx)
    assert "causal candidate" in out
    assert "n=80" in out


# ─────────────────────────────────────────────────────────────────────────
# _render_recent_anomalies
# ─────────────────────────────────────────────────────────────────────────


def test_render_recent_anomalies_empty_context_returns_empty_string():
    assert _render_recent_anomalies(None) == ""
    assert _render_recent_anomalies(SignalContext()) == ""


def test_render_recent_anomalies_includes_date_values_and_z():
    ctx = SignalContext(
        recent_anomalies=[
            RecentAnomaly(
                metric_key="hrv",
                observation_date="2026-04-13",
                direction="low",
                z_score=-4.2,
                observed_value=22.0,
                forecasted_value=42.0,
            )
        ]
    )
    out = _render_recent_anomalies(ctx)
    assert "2026-04-13" in out
    assert "22.0" in out
    assert "42.0" in out
    assert "z=-4.2" in out or "z=" in out


def test_render_recent_anomalies_handles_missing_values():
    ctx = SignalContext(
        recent_anomalies=[
            RecentAnomaly(
                metric_key="hrv",
                observation_date="2026-04-13",
                direction="low",
                z_score=-3.8,
                observed_value=None,
                forecasted_value=None,
            )
        ]
    )
    out = _render_recent_anomalies(ctx)
    assert "n/a" in out


# ─────────────────────────────────────────────────────────────────────────
# _render_personal_forecast
# ─────────────────────────────────────────────────────────────────────────


def test_render_personal_forecast_empty_context_returns_empty_string():
    assert _render_personal_forecast(None) == ""
    assert _render_personal_forecast(SignalContext()) == ""


def test_render_personal_forecast_includes_interval():
    ctx = SignalContext(
        personal_forecasts=[
            PersonalForecast(
                metric_key="hrv",
                target_date="2026-04-15",
                y_hat=42.5,
                y_hat_low=38.0,
                y_hat_high=47.0,
            )
        ]
    )
    out = _render_personal_forecast(ctx)
    assert "42.5" in out
    assert "38.0" in out
    assert "47.0" in out
    assert "2026-04-15" in out


def test_render_personal_forecast_without_interval():
    ctx = SignalContext(
        personal_forecasts=[
            PersonalForecast(
                metric_key="hrv",
                target_date="2026-04-15",
                y_hat=42.5,
                y_hat_low=None,
                y_hat_high=None,
            )
        ]
    )
    out = _render_personal_forecast(ctx)
    assert "42.5" in out
    assert "interval" not in out


def test_render_personal_forecast_skips_entries_with_none_y_hat():
    ctx = SignalContext(
        personal_forecasts=[
            PersonalForecast(
                metric_key="hrv",
                target_date="2026-04-15",
                y_hat=None,
                y_hat_low=None,
                y_hat_high=None,
            )
        ]
    )
    out = _render_personal_forecast(ctx)
    # Header alone is dropped when no forecasts produced y_hat.
    assert out == ""


# ─────────────────────────────────────────────────────────────────────────
# Full system prompt end-to-end
# ─────────────────────────────────────────────────────────────────────────


def test_full_prompt_includes_all_three_sections():
    """The prompt template binds ``active_patterns`` / ``recent_anomalies``
    / ``personal_forecast``. A populated SignalContext should produce a
    prompt that contains all three rendered sections."""
    from app.services.coach_engine import EVIDENCE_BOUND_SYSTEM_PROMPT

    ctx = SignalContext(
        active_patterns=[
            ActivePattern(
                source_metric="protein_intake",
                target_metric="deep_sleep_seconds",
                direction="positive",
                strength=0.55,
                confidence_tier="literature_supported",
                sample_size=60,
                effect_description="Higher protein tends to mean longer deep sleep.",
            )
        ],
        recent_anomalies=[
            RecentAnomaly(
                metric_key="hrv",
                observation_date="2026-04-13",
                direction="low",
                z_score=-4.2,
                observed_value=22.0,
                forecasted_value=42.0,
            )
        ],
        personal_forecasts=[
            PersonalForecast(
                metric_key="hrv",
                target_date="2026-04-15",
                y_hat=42.5,
                y_hat_low=38.0,
                y_hat_high=47.0,
            )
        ],
    )
    prompt = EVIDENCE_BOUND_SYSTEM_PROMPT.format(
        user_name="Brock",
        health_data="{}",
        goals="general wellness",
        custom_goal_context="",
        memory_context="",
        active_patterns=_render_active_patterns(ctx),
        recent_anomalies=_render_recent_anomalies(ctx),
        personal_forecast=_render_personal_forecast(ctx),
        safety_disclaimer="",
    )
    assert "ACTIVE PATTERNS" in prompt
    assert "RECENT ANOMALIES" in prompt
    assert "PERSONAL FORECAST" in prompt
    assert "Higher protein" in prompt
    assert "2026-04-13" in prompt
    assert "42.5" in prompt


def test_full_prompt_gracefully_omits_empty_sections():
    """No SignalContext -> no Phase 5 sections in the prompt. The template
    still renders (no KeyError) and the slots just become blank lines."""
    from app.services.coach_engine import EVIDENCE_BOUND_SYSTEM_PROMPT

    prompt = EVIDENCE_BOUND_SYSTEM_PROMPT.format(
        user_name="Brock",
        health_data="{}",
        goals="general wellness",
        custom_goal_context="",
        memory_context="",
        active_patterns="",
        recent_anomalies="",
        personal_forecast="",
        safety_disclaimer="",
    )
    assert "ACTIVE PATTERNS" not in prompt
    assert "RECENT ANOMALIES" not in prompt
    assert "PERSONAL FORECAST" not in prompt
    assert "WHAT THE USER TOLD US" not in prompt


def test_full_prompt_includes_custom_goal_text_when_provided():
    """When a user filled in the 'Want to share more?' field during
    onboarding, the raw text flows into the coach prompt as its own
    section so responses can speak to their actual situation, not just
    the canned chip goals."""
    from app.services.coach_engine import EVIDENCE_BOUND_SYSTEM_PROMPT

    prompt = EVIDENCE_BOUND_SYSTEM_PROMPT.format(
        user_name="Brock",
        health_data="{}",
        goals="Build muscle",
        custom_goal_context=(
            "WHAT THE USER TOLD US IN THEIR OWN WORDS:\n"
            "I want a custom workout plan. I use the Peloton tread primarily.\n"
        ),
        memory_context="",
        active_patterns="",
        recent_anomalies="",
        personal_forecast="",
        safety_disclaimer="",
    )
    assert "WHAT THE USER TOLD US IN THEIR OWN WORDS" in prompt
    assert "Peloton tread" in prompt
