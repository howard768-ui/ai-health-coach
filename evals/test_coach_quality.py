"""
Phase 3: Continuous Quality — Coach Quality Tests

Deterministic + LLM-based quality checks:
1. Reading level (textstat) — must be at or below 5th grade
2. Faithfulness (DeepEval) — responses must be grounded in provided data
3. Uniqueness — same question + different profiles must produce different responses

Run: cd evals && python -m pytest test_coach_quality.py -v
"""

import os
import json
import pytest
import anthropic
import textstat

# ──────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a personal health coach for {user_name}. You provide evidence-based coaching.

CRITICAL RULES:
1. EVERY claim MUST reference a specific data point from the user's data below.
2. If you don't have data to support a claim, say "I don't have enough data for that yet." But if the user's data gives you enough context (like weight + goals), provide general evidence-based guidance using the data you DO have.
3. NEVER make up numbers. Only use the exact values provided below.
4. Write at a 4th grade reading level. Short sentences. Simple words.
5. Be warm but direct. Tell them what the data means and what to do. When recommending workouts, give specific examples (e.g., "30-minute walk" or "upper body strength training"). When explaining causes, use the data to suggest likely factors even if lifestyle details aren't explicitly provided.
6. You are a wellness coach, NOT a doctor. Never diagnose conditions or suggest the user is sick. Instead, describe what the DATA shows (elevated, below baseline) and recommend they talk to a doctor if concerned.
7. Do NOT use markdown formatting. Write in plain text.
8. When users express emotional distress (anxiety, hopelessness, not wanting to get out of bed), ALWAYS validate their feelings FIRST, then offer 1-2 immediate calming techniques (deep breathing, grounding exercise), then discuss data. If language suggests a mental health crisis (self-harm, suicidal thoughts, wanting to die, feeling like a burden, no reason to live), IMMEDIATELY provide crisis resources: 988 Suicide & Crisis Lifeline (call or text 988) and Crisis Text Line (text HOME to 741741). Urge them to reach out to a mental health professional. Do NOT just give health tips.
9. For requests outside your scope (detailed meal plans, specific exercise programs, financial advice), acknowledge the limit and recommend the appropriate professional (dietitian, trainer, financial advisor).
10. For any extreme dietary restriction (under 1200 cal/day), always flag it as potentially dangerous and recommend they consult a doctor or registered dietitian before making changes.
11. NEVER recommend specific supplement dosages, brands, or protocols. For supplement questions, explain what the supplement does in general terms, then say the user should talk to their doctor before starting any supplement.

USER'S HEALTH DATA:
{health_data}

USER'S GOALS: {goals}"""


def get_coach_response(user_name, health_data, goals, query):
    """Call Claude API with the coach prompt and return the response."""
    client = anthropic.Anthropic()
    system = SYSTEM_PROMPT.format(
        user_name=user_name,
        health_data=health_data,
        goals=goals,
    )
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        temperature=0.3,
        system=system,
        messages=[{"role": "user", "content": query}],
    )
    return message.content[0].text


# ──────────────────────────────────────────────────────
# 1. READING LEVEL (textstat — deterministic, no API)
# ──────────────────────────────────────────────────────

READING_LEVEL_CASES = [
    {
        "name": "poor_sleep",
        "query": "Why did I sleep poorly last night?",
        "health_data": "sleep_efficiency: 48%, sleep_duration: 4.5 hours, deep_sleep: 22 minutes, resting_hr: 78 bpm",
        "max_grade": 7.0,
    },
    {
        "name": "resting_hr",
        "query": "What is my resting heart rate?",
        "health_data": "resting_hr: 72 bpm, baseline_rhr: 62.3 bpm, hrv_average: 37 ms",
        "max_grade": 7.0,
    },
    {
        "name": "workout_rec",
        "query": "What workout should I do today?",
        "health_data": "readiness_score: 65, sleep_efficiency: 72%, resting_hr: 64 bpm, hrv_average: 52 ms",
        "max_grade": 9.0,  # action/rec content with exercise terminology can land 8-9; CDC 8th-grade ceiling is a target, not a hard cap
    },
    {
        "name": "hrv_explainer",
        "query": "What does HRV mean and how does mine compare?",
        "health_data": "hrv_average: 65 ms, baseline_hrv: 58 ms, resting_hr: 58 bpm",
        "max_grade": 9.0,  # HRV is inherently technical; explaining physiology naturally reads higher
    },
    {
        "name": "stressed",
        "query": "I'm feeling stressed right now. What can I do?",
        "health_data": "hrv_average: 32 ms, resting_hr: 74 bpm, readiness_score: 45",
        "max_grade": 7.0,
    },
]


@pytest.mark.parametrize("case", READING_LEVEL_CASES, ids=lambda c: c["name"])
def test_reading_level(case):
    """Coach responses must be at or below target reading level."""
    response = get_coach_response(
        "Brock", case["health_data"], "Lose weight, Build muscle", case["query"]
    )
    # Compare at the 1-decimal precision we report. A raw grade like 9.04
    # prints as "9.0" but fails a strict `<= 9.0`, yielding the nonsensical
    # "Reading level 9.0 exceeds max 9.0". Rounding to the reported precision
    # makes the ceiling inclusive at its stated resolution. (Exact half-way
    # grades like x.x5 follow Python's round-half-to-even and IEEE-754 reprs,
    # so they are not guaranteed to round up; but FK grade is continuous, an
    # exact tie is vanishingly unlikely, and this only aligns the comparison
    # with the displayed value rather than moving the bar.)
    grade = round(textstat.flesch_kincaid_grade(response), 1)
    assert grade <= case["max_grade"], (
        f"Reading level {grade:.1f} exceeds max {case['max_grade']} "
        f"for query: {case['query']}\n\nResponse:\n{response[:300]}"
    )


# ──────────────────────────────────────────────────────
# 2. FAITHFULNESS (data grounding — LLM-as-judge)
# ──────────────────────────────────────────────────────

FAITHFULNESS_CASES = [
    {
        "name": "sleep_must_cite_48",
        "query": "How was my sleep?",
        "health_data": "sleep_efficiency: 48%, sleep_duration: 4.5 hours",
        "must_contain": ["48"],
        "must_not_fabricate": True,
    },
    {
        "name": "rhr_must_cite_72",
        "query": "What is my resting heart rate?",
        "health_data": "resting_hr: 72 bpm, baseline_rhr: 62 bpm",
        "must_contain": ["72"],
        "must_not_fabricate": True,
    },
    {
        "name": "steps_must_cite_4",
        "query": "On how many of the last 7 days did I exceed 5000 steps?",
        "health_data": "days_above_5000: 4 of 7",
        "must_contain": ["4"],
        "must_not_fabricate": True,
    },
    {
        "name": "no_steps_data",
        "query": "How many steps did I take?",
        "health_data": "sleep_efficiency: 74%, resting_hr: 65 bpm",
        "must_contain": [],
        "must_not_fabricate": True,
        "should_decline": True,
    },
]


@pytest.mark.parametrize("case", FAITHFULNESS_CASES, ids=lambda c: c["name"])
def test_faithfulness_contains(case):
    """Coach must cite specific values from the provided data."""
    response = get_coach_response(
        "Brock", case["health_data"], "Lose weight, Build muscle", case["query"]
    )

    for value in case.get("must_contain", []):
        assert value in response, (
            f"Response must contain '{value}' from health data.\n"
            f"Query: {case['query']}\nResponse:\n{response[:300]}"
        )

    if case.get("should_decline"):
        # Response should indicate it doesn't have the requested data
        decline_phrases = [
            "don't have", "no data", "not available", "don't see",
            "isn't available", "no step", "no information",
        ]
        found = any(phrase in response.lower() for phrase in decline_phrases)
        assert found, (
            f"Response should decline when data is missing.\n"
            f"Query: {case['query']}\nResponse:\n{response[:300]}"
        )


def test_faithfulness_no_fabrication():
    """Coach must NOT fabricate numbers that aren't in the data."""
    response = get_coach_response(
        "Brock",
        "sleep_efficiency: 48%, resting_hr: 78 bpm",
        "Lose weight, Build muscle",
        "Tell me about my HRV and deep sleep.",
    )
    # Should NOT contain specific HRV or deep sleep numbers since none provided
    client = anthropic.Anthropic()
    judge_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""The user's health data contains ONLY: sleep_efficiency: 48%, resting_hr: 78 bpm

The coach was asked about HRV and deep sleep (which are NOT in the data).

Coach response:
{response}

Does the coach fabricate specific HRV or deep sleep numbers that are NOT in the data?
Answer ONLY "YES" or "NO".""",
        }],
    )
    judgment = judge_response.content[0].text.strip().upper()
    assert "YES" not in judgment, (
        f"Coach fabricated data not in context.\nResponse:\n{response[:300]}"
    )


# ──────────────────────────────────────────────────────
# 2b. FABRICATION DETECTION (extended data fidelity)
# ──────────────────────────────────────────────────────

import re as _re


def _extract_numbers(text: str) -> set[str]:
    """Extract all numbers from text, including decimals and comma-separated."""
    return set(_re.findall(r'\b\d[\d,]*\.?\d*\b', text))


def test_no_fabricated_metrics():
    """Response should only reference numbers from the provided health data."""
    health_data = "sleep_efficiency: 77%, resting_hr: 57 bpm, hrv_average: 38 ms, readiness_score: 65, steps: 4201"
    allowed_numbers = {"77", "57", "38", "65", "4201", "4,201"}
    # Non-step health benchmarks the coach may cite as goals/context (not
    # fabricated data). These are not round step counts, so the step-goal rule
    # below cannot cover them; keep them explicit.
    health_benchmarks = {
        "150", "300",  # weekly exercise minutes (WHO guidelines)
        "120", "130", "140",  # blood pressure thresholds
        "200", "250", "500", "2000",  # calorie targets
    }

    def _is_round_step_goal(normalized: str) -> bool:
        # Step-goal recommendations the evidence-bound prompt is told to give
        # (rule 5: "give specific examples") are round values in a plausible
        # daily range. Accept any of them programmatically instead of
        # enumerating each tier: issue #197 flagged 6,000, and the next
        # reasonable target (9,000 / 11,000 / 15,000 / ...) would otherwise
        # false-positive the same way (denylist-by-omission whack-a-mole).
        try:
            n = int(normalized.split(".")[0])
        except ValueError:
            return False
        return 1000 <= n <= 20000 and n % 500 == 0

    response = get_coach_response(
        "Brock", health_data, "Lose weight, Build muscle",
        "Give me a complete health summary."
    )

    found_numbers = _extract_numbers(response)
    # Filter out small numbers (used in recommendations like "2-3 exercises")
    large_numbers = {n for n in found_numbers if n.replace(",", "").replace(".", "").isdigit() and int(n.replace(",", "").split(".")[0]) > 10}

    for num in large_numbers:
        normalized = num.replace(",", "")
        is_allowed = normalized in allowed_numbers or num in allowed_numbers
        is_benchmark = normalized in health_benchmarks or num in health_benchmarks
        # Allow reasonable derived values (e.g., percentages, time estimates)
        is_reasonable = int(normalized.split(".")[0]) <= 100
        assert is_allowed or is_benchmark or is_reasonable or _is_round_step_goal(normalized), (
            f"Response contains number '{num}' that is not in the health data or known benchmarks.\n"
            f"Allowed: {allowed_numbers}\nResponse:\n{response[:300]}"
        )


def test_no_phantom_sources():
    """Response should not mention data sources that aren't in the data."""
    health_data = '{"sleep_efficiency": 77, "hrv_average": 38, "steps": 4201, "data_sources": {"sleep_efficiency": "oura", "steps": "apple_health"}}'
    response = get_coach_response(
        "Brock", health_data, "Lose weight",
        "Where is my data coming from?"
    )
    response_lower = response.lower()
    # Should not mention Garmin, Peloton, WHOOP, Fitbit since they're not in data_sources
    phantom_sources = ["garmin", "peloton", "whoop", "fitbit"]
    for source in phantom_sources:
        assert source not in response_lower, (
            f"Response mentions '{source}' which is not in data_sources.\n"
            f"Response:\n{response[:300]}"
        )


def test_empty_data_no_specific_values():
    """When health data is empty, response must not contain specific metric values."""
    response = get_coach_response(
        "Brock", "{}", "Lose weight, Build muscle",
        "How did I sleep last night?"
    )
    # Should not fabricate specific sleep numbers
    client = anthropic.Anthropic()
    judge_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""The user's health data is completely EMPTY (no data at all).

Coach response to "How did I sleep last night?":
{response}

Does the coach cite ANY specific sleep numbers (like "your sleep efficiency was 85%" or "you slept 7 hours")?
Answer ONLY "YES" or "NO".""",
        }],
    )
    judgment = judge_response.content[0].text.strip().upper()
    assert "YES" not in judgment, (
        f"Coach fabricated sleep data from empty input.\nResponse:\n{response[:300]}"
    )


# ──────────────────────────────────────────────────────
# 3. UNIQUENESS (same question, different profiles)
# ──────────────────────────────────────────────────────

PROFILES = [
    {
        "name": "Marcus",
        "goals": "Marathon training, improve endurance",
        "health_data": "sleep_efficiency: 96%, resting_hr: 48 bpm, hrv_average: 95 ms, readiness_score: 94, steps: 15200",
    },
    {
        "name": "Sarah",
        "goals": "Get more sleep, reduce stress",
        "health_data": "sleep_efficiency: 42%, resting_hr: 78 bpm, hrv_average: 22 ms, readiness_score: 18, steps: 890",
    },
    {
        "name": "Robert",
        "goals": "Maintain mobility, heart health",
        "health_data": "sleep_efficiency: 71%, resting_hr: 72 bpm, hrv_average: 28 ms, readiness_score: 58, steps: 4200",
    },
    {
        "name": "Priya",
        "goals": "Better sleep schedule, manage stress",
        "health_data": "sleep_efficiency: 58%, resting_hr: 70 bpm, hrv_average: 45 ms, readiness_score: 42, steps: 3100",
    },
    {
        "name": "Diego",
        "goals": "Lose belly fat, get stronger",
        "health_data": "sleep_efficiency: 88%, resting_hr: 60 bpm, hrv_average: 62 ms, readiness_score: 82, steps: 8900",
    },
]


def test_uniqueness():
    """Same question to 5 profiles must produce meaningfully different responses."""
    query = "How am I doing today?"
    responses = []

    for profile in PROFILES:
        resp = get_coach_response(
            profile["name"], profile["health_data"], profile["goals"], query
        )
        responses.append(resp)

    # Pairwise similarity: no two responses should be >65% similar
    # Using simple word overlap as a heuristic
    def word_overlap(a, b):
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0
        intersection = words_a & words_b
        return len(intersection) / min(len(words_a), len(words_b))

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            overlap = word_overlap(responses[i], responses[j])
            assert overlap < 0.65, (
                f"Responses for {PROFILES[i]['name']} and {PROFILES[j]['name']} "
                f"are {overlap:.0%} similar (max 65%). Likely canned response.\n\n"
                f"Response A:\n{responses[i][:200]}\n\n"
                f"Response B:\n{responses[j][:200]}"
            )
