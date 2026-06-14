"""Audit B3: the coach must not send the user's real first name to Anthropic.

The privacy policy says the name is not shared with the LLM provider unless
necessary. It is not necessary for coaching, so process_query must format the
system prompt with a generic placeholder, never the real name.

Run: cd backend && uv run pytest tests/test_coach_privacy.py -v
"""

import os
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-long-enough-aaaaaaaaaaaa")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-not-used")


from app.services.coach_engine import CoachEngine


def test_real_name_not_sent_to_anthropic():
    coach = CoachEngine()

    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text="Your resting heart rate looks steady.")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )

    # Capture exactly what is sent to the model.
    coach.client.messages.create = fake_create

    secret_name = "ZZSECRETREALNAME"
    coach.process_query(
        # A resting-HR question is known to route to AI, not canned rules.
        query="what is my resting heart rate?",
        health_data={"resting_hr": 62, "readiness_score": 70},
        user_name=secret_name,
    )

    # The query must reach the model (else the test proves nothing).
    assert "system" in captured, "expected an AI call; query routed to rules?"
    system_prompt = captured["system"]
    assert secret_name not in system_prompt, "real first name leaked to Anthropic"
    # And the generic placeholder is used instead.
    assert "coach for the user" in system_prompt
