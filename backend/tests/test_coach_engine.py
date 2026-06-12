"""Tests for the coach engine's deterministic logic — SafetyCheck and Deliberator.

These don't require the Anthropic API. They cover the safety gates and routing
logic that runs before any AI call. This is the most security-critical code in
the entire backend (P1-13).

Run: cd backend && uv run python -m pytest tests/test_coach_engine.py -v
"""

import os
import pytest

# Set test config before importing
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-for-coach-engine-tests-only")
os.environ.setdefault("ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ=")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-key-tests-dont-call-anthropic")

from app.services.coach_engine import (
    SafetyCheck,
    Deliberator,
    ModelTier,
)


# ── SafetyCheck — deterministic safety gates ─────────────────────────────────


def test_safety_check_normal_data_not_concerning():
    """Healthy data should not trigger any safety flags."""
    data = {
        "hrv_average": 65,
        "resting_hr": 58,
        "sleep_efficiency": 90,
        "readiness_score": 85,
    }
    check = SafetyCheck.check_health_data(data)
    assert not check.is_concerning
    assert check.reasons == []
    assert not check.requires_disclaimer
    assert not check.requires_opus


def test_safety_check_critically_low_hrv():
    """HRV under 20ms should trigger a safety flag."""
    data = {"hrv_average": 15}
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert any("HRV" in r for r in check.reasons)
    assert check.requires_opus  # Must escalate to the smartest model


def test_safety_check_elevated_resting_hr():
    """Resting HR above 100 bpm is tachycardia — safety flag."""
    data = {"resting_hr": 110}
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert any("Resting HR" in r for r in check.reasons)


def test_safety_check_critically_low_sleep_efficiency():
    """Sleep efficiency under 50% is concerning."""
    data = {"sleep_efficiency": 42}
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert any("Sleep efficiency" in r for r in check.reasons)


def test_safety_check_low_readiness():
    """Readiness under 20 is critically low."""
    data = {"readiness_score": 15}
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert any("Readiness" in r for r in check.reasons)


def test_safety_check_hrv_baseline_deviation():
    """HRV more than 30% off baseline triggers a flag (with enough history)."""
    data = {"hrv_average": 30, "baseline_hrv": 60, "baseline_days": 7}
    # 30 vs 60 = 50% deviation
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning


def test_safety_check_missing_data_doesnt_crash():
    """Missing keys should default to no flags, not crash."""
    check = SafetyCheck.check_health_data({})
    assert not check.is_concerning


def test_safety_check_multiple_concerns_collected():
    """Multiple concerning values should all be reported."""
    data = {
        "hrv_average": 12,
        "resting_hr": 110,
        "sleep_efficiency": 35,
    }
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert len(check.reasons) >= 3  # All three concerns flagged


# ── Deliberator — routing decisions ──────────────────────────────────────────


def test_deliberator_safety_flag_routes_to_opus():
    """Anything safety-flagged must go to Opus, even simple queries."""
    safety = SafetyCheck(
        is_concerning=True,
        reasons=["HRV critically low (12ms)"],
        requires_disclaimer=True,
        requires_opus=True,
    )
    decision = Deliberator.route("hello", {}, safety)
    assert decision.tier == ModelTier.OPUS
    assert decision.safety_flag is True


def test_deliberator_cross_domain_routes_to_opus():
    """Queries with 'why', 'cause', etc. need cross-domain reasoning."""
    safety = SafetyCheck(False, [], False, False)
    queries = [
        "why is my sleep so bad?",
        "what's causing my low HRV?",
        "is there a pattern in my recovery?",
        "how does my workout affect my sleep?",
    ]
    for q in queries:
        decision = Deliberator.route(q, {"readiness_score": 70}, safety)
        assert decision.tier == ModelTier.OPUS, f"Failed on query: {q}"


def test_deliberator_readiness_routes_to_rules():
    """Plain readiness questions can be answered deterministically."""
    safety = SafetyCheck(False, [], False, False)
    decision = Deliberator.route(
        "how is my readiness today?",
        {"readiness_score": 75},
        safety,
    )
    assert decision.tier == ModelTier.RULES


def test_deliberator_routine_query_routes_to_sonnet():
    """Generic queries that aren't rule-matchable go to Sonnet."""
    safety = SafetyCheck(False, [], False, False)
    decision = Deliberator.route(
        "what should I eat for breakfast?",
        {"readiness_score": 70},
        safety,
    )
    assert decision.tier == ModelTier.SONNET


def test_deliberator_resting_hr_query_routes_to_ai():
    """RHR queries should go to AI with full data context — NOT canned rules.

    Regression test: previously matched 'rest' in 'resting heart rate' and
    returned the readiness rule, which was a confusing UX bug."""
    safety = SafetyCheck(False, [], False, False)
    decision = Deliberator.route(
        "what is my resting heart rate?",
        {"resting_hr": 62, "readiness_score": 70},
        safety,
    )
    # Should NOT be RULES — must route to AI with real data
    assert decision.tier != ModelTier.RULES


def test_can_answer_from_rules_readiness_high():
    """Known readiness queries return canned rules with the right tone."""
    answer_tuple = Deliberator.can_answer_from_rules(
        "should i push hard today?",
        {"readiness_score": 85},
    )
    can, answer = answer_tuple
    assert can
    assert answer is not None
    assert "high" in answer.lower() or "hard" in answer.lower()


def test_can_answer_from_rules_readiness_low():
    """Low readiness gets a rest-day message."""
    can, answer = Deliberator.can_answer_from_rules(
        "should i rest today?",
        {"readiness_score": 18},
    )
    assert can
    assert "rest" in answer.lower()


def test_can_answer_from_rules_resting_hr_returns_none():
    """RHR queries must NOT match the rule pathway (they need AI)."""
    can, answer = Deliberator.can_answer_from_rules(
        "what is my resting heart rate?",
        {"readiness_score": 70, "resting_hr": 60},
    )
    assert not can
    assert answer is None


# ── Word-boundary cost leak prevention (P3-8 regression) ─────────────────────


def test_deliberator_because_does_not_match_cause():
    """'because' should NOT match the 'cause' keyword and route to Opus.
    Regression test for the cost-leak bug where substring matching of 'caus'
    was triggering Opus on every casual mention of 'because'."""
    safety = SafetyCheck(False, [], False, False)
    decision = Deliberator.route(
        "I'm tired because I had a long day",
        {"readiness_score": 70},
        safety,
    )
    # Should NOT escalate to Opus just for "because"
    assert decision.tier != ModelTier.OPUS, "'because' incorrectly matched 'cause' keyword"


def test_deliberator_causing_matches_cross_domain():
    """'causing' should still trigger cross-domain routing."""
    safety = SafetyCheck(False, [], False, False)
    decision = Deliberator.route(
        "what's causing my low HRV?",
        {"readiness_score": 70},
        safety,
    )
    assert decision.tier == ModelTier.OPUS


# ── Zero-value safety gate hardening ────────────────────────────────────────


@pytest.mark.parametrize("hrv_value", [0, 0.0])
def test_safety_check_fires_at_zero_hrv(hrv_value):
    """HRV=0 (sensor error or critical) must still trigger the safety gate.
    Regression test: falsy check `if hrv and hrv < 20` silently skipped at 0."""
    data = {"hrv_average": hrv_value}
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert any("HRV" in r for r in check.reasons)


def test_safety_check_fires_at_zero_readiness():
    """Readiness=0 must trigger the safety gate."""
    data = {"readiness_score": 0}
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert any("Readiness" in r for r in check.reasons)


def test_safety_check_fires_at_zero_sleep_efficiency():
    """Sleep efficiency=0 must trigger the safety gate."""
    data = {"sleep_efficiency": 0}
    check = SafetyCheck.check_health_data(data)
    assert check.is_concerning
    assert any("Sleep efficiency" in r for r in check.reasons)


def test_safety_check_rhr_zero_not_concerning():
    """RHR=0 should NOT trigger tachycardia flag (0 is not > 100)."""
    data = {"resting_hr": 0}
    check = SafetyCheck.check_health_data(data)
    assert not any("Resting HR" in r for r in check.reasons)


def test_safety_check_baseline_deviation_zero_baseline():
    """Zero baseline HRV should not cause division-by-zero crash."""
    data = {"hrv_average": 30, "baseline_hrv": 0, "baseline_days": 7}
    check = SafetyCheck.check_health_data(data)
    # Should not crash; 0 baseline means no meaningful comparison
    assert isinstance(check.is_concerning, bool)


def test_safety_check_baseline_suppressed_for_new_user():
    """Baseline deviation should NOT fire when user has < 3 days of data."""
    data = {"hrv_average": 30, "baseline_hrv": 60, "baseline_days": 1}
    # 50% deviation, but only 1 day of history — too noisy to flag
    check = SafetyCheck.check_health_data(data)
    assert not any("baseline" in r.lower() for r in check.reasons)


def test_deliberator_hrv_rules_with_zero_values():
    """HRV=0 with baseline should still evaluate rule path without crash."""
    can, answer = Deliberator.can_answer_from_rules(
        "how is my HRV?",
        {"hrv_average": 0, "baseline_hrv": 50, "baseline_days": 7},
    )
    # Should return a rule answer (HRV below baseline)
    assert can
    assert answer is not None


def test_deliberator_hrv_skips_baseline_for_new_user():
    """HRV baseline comparison should be skipped when < 3 days of data."""
    can, answer = Deliberator.can_answer_from_rules(
        "how is my HRV?",
        {"hrv_average": 30, "baseline_hrv": 60, "baseline_days": 2},
    )
    # Not enough data for baseline comparison — should fall through to AI
    assert not can


def test_deliberator_inflections_all_match():
    """All inflections of cross-domain keywords route to Opus."""
    safety = SafetyCheck(False, [], False, False)
    queries = [
        "what causes my poor sleep",
        "what caused this drop",
        "how does diet affect recovery",
        "explain my low readiness",
        "what's the relationship between sleep and HRV",
    ]
    for q in queries:
        decision = Deliberator.route(q, {"readiness_score": 70}, safety)
        assert decision.tier == ModelTier.OPUS, f"Failed on: {q}"
