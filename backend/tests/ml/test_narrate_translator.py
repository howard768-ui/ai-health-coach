"""Tests for the Opus narrator — injected fake Anthropic client, no API calls.

Covers the happy path, voice-compliance rejection, em-dash mechanical scrub,
and fallback-to-template behavior across kinds.

Run: ``cd backend && uv run python -m pytest tests/ml/test_narrate_translator.py -v``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ml.narrate import translator


# ─────────────────────────────────────────────────────────────────────────
# Fake Anthropic client
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeBlock:
    text: str


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]


class FakeAnthropicClient:
    """Stand-in for anthropic.Anthropic.

    ``next_text`` controls what the next ``messages.create`` call returns.
    Tests configure this to verify the narrator handles each case.
    """

    def __init__(self, next_text: str | Exception = "Default narration sentence."):
        self.next_text = next_text
        self.calls: list[dict] = []
        self.messages = self  # expose .messages.create on the same object

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if isinstance(self.next_text, Exception):
            raise self.next_text
        return _FakeResponse(content=[_FakeBlock(text=self.next_text)])


def _correlation_request() -> translator.NarrationRequest:
    return translator.NarrationRequest(
        kind="correlation",
        subject_metrics=["protein_intake", "deep_sleep_seconds"],
        payload={
            "source_metric": "protein_intake",
            "target_metric": "deep_sleep_seconds",
            "direction": "positive",
            "pearson_r": 0.55,
            "sample_size": 60,
            "confidence_tier": "literature_supported",
            "literature_ref": "10.1007/s40279-014-0260-0",
        },
    )


def _anomaly_request() -> translator.NarrationRequest:
    return translator.NarrationRequest(
        kind="anomaly",
        subject_metrics=["hrv"],
        payload={
            "metric_key": "hrv",
            "observation_date": "2026-04-13",
            "direction": "low",
            "z_score": -3.8,
            "observed_value": 22.0,
            "forecasted_value": 42.0,
            "confirmed_by_bocpd": True,
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_narration_returns_model_text_when_voice_compliant():
    client = FakeAnthropicClient(
        next_text="Higher protein tends to show up alongside longer deep sleep in your data."
    )
    result = await translator.generate_narration(_correlation_request(), client=client)
    assert result.text.startswith("Higher protein")
    assert result.used_fallback is False
    assert result.fallback_reason is None


@pytest.mark.asyncio
async def test_narration_strips_outer_whitespace():
    client = FakeAnthropicClient(next_text="  Higher protein, longer deep sleep.  \n")
    result = await translator.generate_narration(_correlation_request(), client=client)
    assert result.text == "Higher protein, longer deep sleep."


# ─────────────────────────────────────────────────────────────────────────
# Em dash scrub path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_em_dash_in_model_output_is_mechanically_scrubbed():
    # Model slipped in an em dash despite the prompt. Translator should
    # scrub it and still return non-fallback text.
    client = FakeAnthropicClient(
        next_text="Higher protein \u2014 longer deep sleep in your data."
    )
    result = await translator.generate_narration(_correlation_request(), client=client)
    assert "\u2014" not in result.text
    assert result.used_fallback is False


@pytest.mark.asyncio
async def test_em_dash_plus_emoji_falls_back_to_template():
    # Multiple violations — em dash scrub alone won't rescue this.
    client = FakeAnthropicClient(
        next_text="Higher protein \u2014 longer deep sleep \U0001F44D in your data."
    )
    result = await translator.generate_narration(_correlation_request(), client=client)
    assert result.used_fallback is True
    assert result.fallback_reason == "voice_compliance"
    assert "\u2014" not in result.text
    # The correlation template is bland but compliant.
    assert result.text == translator._TEMPLATE_FALLBACK["correlation"]


# ─────────────────────────────────────────────────────────────────────────
# API error path
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_error_returns_fallback():
    client = FakeAnthropicClient(next_text=RuntimeError("simulated 500"))
    result = await translator.generate_narration(_correlation_request(), client=client)
    assert result.used_fallback is True
    assert result.fallback_reason == "api_error"
    assert result.text == translator._TEMPLATE_FALLBACK["correlation"]


@pytest.mark.asyncio
async def test_empty_model_response_returns_fallback():
    client = FakeAnthropicClient(next_text="")
    result = await translator.generate_narration(_correlation_request(), client=client)
    assert result.used_fallback is True
    assert result.fallback_reason == "empty_response"


# ─────────────────────────────────────────────────────────────────────────
# Kind-specific fallback templates
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anomaly_fallback_uses_anomaly_template():
    client = FakeAnthropicClient(next_text=RuntimeError("down"))
    result = await translator.generate_narration(_anomaly_request(), client=client)
    assert result.text == translator._TEMPLATE_FALLBACK["anomaly"]


@pytest.mark.asyncio
async def test_unknown_kind_falls_back_to_correlation_template():
    """Defensive: a forward-compat kind the translator doesn't know about
    should still return compliant fallback copy, never crash."""
    client = FakeAnthropicClient(next_text=RuntimeError("down"))
    request = translator.NarrationRequest(
        kind="brand_new_kind",
        subject_metrics=["mystery"],
        payload={},
    )
    result = await translator.generate_narration(request, client=client)
    assert result.used_fallback is True
    assert result.text  # non-empty


# ─────────────────────────────────────────────────────────────────────────
# Prompt composition
# ─────────────────────────────────────────────────────────────────────────


def test_compose_user_prompt_correlation_includes_key_fields():
    prompt = translator._compose_user_prompt(_correlation_request())
    assert "protein_intake".replace("_", " ") in prompt
    assert "60 days" in prompt
    assert "literature supported" in prompt or "literature_supported" in prompt


def test_compose_user_prompt_anomaly_includes_observation_date_and_z():
    prompt = translator._compose_user_prompt(_anomaly_request())
    assert "2026-04-13" in prompt
    assert "3.8" in prompt
    assert "two-signal" in prompt  # BOCPD confirmation mentioned


# ─────────────────────────────────────────────────────────────────────────
# Request is forwarded to the API
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_narrator_uses_opus_model_by_default():
    """Narration must bill the CENTRALIZED Opus model id (Settings, PR #95),
    not a hardcoded literal: a model deprecation is one env-var override."""
    from app.config import settings

    client = FakeAnthropicClient(next_text="Higher protein and longer deep sleep often show up together.")
    await translator.generate_narration(_correlation_request(), client=client)
    assert client.calls, "Translator should have hit the client"
    assert client.calls[0]["model"] == settings.anthropic_model_opus
    # System prompt includes the voice rules.
    assert "em dashes" in client.calls[0]["system"] or "em dash" in client.calls[0]["system"]
