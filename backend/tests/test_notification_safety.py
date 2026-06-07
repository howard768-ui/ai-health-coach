"""Audit B1/H1: AI-generated notification copy must never put raw numbers
(metrics / PHI) on the lock screen, and a malformed model response must not be
dumped verbatim into the body.

Run: cd backend && uv run pytest tests/test_notification_safety.py -v
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-long-enough-aaaaaaaaaaaa")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

import pytest

from app.services.notification_safety import contains_numbers, safe_notification_text


class _StubCoach:
    def __init__(self, response: str):
        self._response = response

    def process_query(self, **kwargs):
        return {"response": self._response}


# ── unit: the deterministic guard ─────────────────────────────────────


def test_contains_numbers():
    assert contains_numbers("Your HRV was 68ms")
    assert contains_numbers("3 of 5 days")
    assert not contains_numbers("Recovery looks strong today")
    assert not contains_numbers("")
    assert not contains_numbers(None)


def test_safe_notification_text_swaps_on_digits():
    assert safe_notification_text("Title", "Your HRV was 68", "FT", "FB") == ("FT", "FB")
    assert safe_notification_text("All good", "Recovery strong", "FT", "FB") == (
        "All good",
        "Recovery strong",
    )


# ── notification_content generators ───────────────────────────────────


def _content_gen(response: str):
    from app.services.notification_content import NotificationContentGenerator

    gen = NotificationContentGenerator()
    gen.coach = _StubCoach(response)
    return gen


def test_coaching_nudge_rejects_numbers_in_ai_output():
    gen = _content_gen('{"title":"Recovery strong","body":"Your HRV was 68ms today"}')
    out = gen.generate_coaching_nudge(health_data={}, user_name="Brock")
    assert not contains_numbers(out["title"])
    assert not contains_numbers(out["body"])
    # Falls back to the static, metric-free copy.
    assert out["body"] == "A pattern in your health data is worth checking out. Tap to see."


def test_coaching_nudge_non_json_does_not_leak_raw_response():
    raw = "Sorry, your readiness is 41 and HRV 28ms, take it easy"
    gen = _content_gen(raw)
    out = gen.generate_coaching_nudge(health_data={}, user_name="Brock")
    assert raw not in out["body"]
    assert "28ms" not in out["body"]
    assert not contains_numbers(out["body"])


def test_coaching_nudge_clean_output_passes_through():
    gen = _content_gen(
        '{"title":"A pattern in your data","body":"Earlier dinners seem to help your sleep."}'
    )
    out = gen.generate_coaching_nudge(health_data={}, user_name="Brock")
    assert out["title"] == "A pattern in your data"
    assert out["body"] == "Earlier dinners seem to help your sleep."


# ── notification_engine morning brief ─────────────────────────────────


def test_morning_brief_non_json_uses_safe_fallback(monkeypatch):
    monkeypatch.setattr(
        "app.services.notification_engine.generate_recovery_badge",
        lambda *a, **k: "http://test/badge.png",
    )
    from app.services.notification_engine import NotificationEngine

    eng = NotificationEngine()
    eng.coach = _StubCoach("Readiness 41, HRV 28ms - take it easy")
    out = eng.generate_morning_brief(health_data={"readiness_score": 80}, user_name="Brock")
    assert "28ms" not in out["body"]
    assert not contains_numbers(out["body"])


def test_morning_brief_rejects_numbers_in_ai_output(monkeypatch):
    monkeypatch.setattr(
        "app.services.notification_engine.generate_recovery_badge",
        lambda *a, **k: "http://test/badge.png",
    )
    from app.services.notification_engine import NotificationEngine

    eng = NotificationEngine()
    eng.coach = _StubCoach('{"title":"Good morning","body":"Your readiness is 41 today"}')
    out = eng.generate_morning_brief(health_data={"readiness_score": 80}, user_name="Brock")
    assert not contains_numbers(out["body"])
