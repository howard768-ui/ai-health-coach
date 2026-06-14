"""Opus narrator for Signal Engine user-facing copy.

Generates one-sentence explanations for insights that the card view
renders verbatim (``payload.effect_description``). Ship these through
Claude Opus per the plan: it's the one line the user sees daily, must
pass the voice rules, and quality > cost. Cap is 1 narration + 2
explanations per user per day per the plan — at current Opus pricing
that's ~$0.05/user/month, trivially cheap.

Voice compliance (no em dashes, no emoji, Flesch-Kincaid grade <= 5) is
enforced post-generation. On violation we try one mechanical scrub
(strip em dashes via ``voice_compliance.scrub_em_dashes``); if that does
not clear the checks we fall back to a templated string rather than ship
non-compliant copy.

See feedback_testing_rigor.md — every narration path has a unit test
that pins the voice invariant.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger("meld.ml.narrate")


# Narration uses the Opus tier; the concrete model ID comes from Settings
# (PR #95 centralization) so a model deprecation is one Railway env-var
# override, not a grep across services. The previous hardcoded ID here was
# the single holdout defeating that invariant. Audit P4/G. Resolved lazily
# in _call_narrator so importing this module never touches settings.
NARRATION_MAX_TOKENS = 280

# System prompt for the narrator. Deliberately short — we want Opus's
# default voice constrained to the voice rules and nothing else. The
# candidate payload provides all the context.
_NARRATION_SYSTEM_PROMPT = """You translate statistical findings into one short sentence a health-coaching app shows to a user.

Rules:
1. 4th grade reading level. Short words. Short sentences.
2. NEVER use em dashes. Use commas, colons, parentheses, or a new sentence.
3. NEVER use emoji.
4. Warm but direct. Cite the specific finding; do not generalize.
5. Output ONE sentence. Nothing else. No preamble, no quotes, no explanation.
6. NEVER claim causation. Use "tends to", "is associated with", "shows up alongside".
"""


# Templated fallbacks by kind. Used when Opus produces non-compliant copy
# two passes in a row, or when the API call fails. Deliberately bland so
# we never ship something wrong over something boring.
_TEMPLATE_FALLBACK = {
    "correlation": "When one of your daily habits moves, a related health measure tends to move with it.",
    "anomaly": "A recent number is further from your usual range than normal.",
    "forecast_warning": "Your short-term forecast is trending away from your usual range.",
    "experiment_result": "Your recent experiment produced a result worth reviewing.",
    "streak": "Your streak continues.",
    "regression": "A pattern you had built is slipping.",
}


@dataclass
class NarrationRequest:
    """Input to the narrator. Field names mirror InsightCandidate."""

    kind: str
    subject_metrics: list[str]
    payload: dict
    user_name: str = "you"


@dataclass
class NarrationResult:
    """What the narrator returns. ``used_fallback`` flags when we had to
    skip Opus output for voice compliance reasons, so telemetry can track
    how often the model needs reining in."""

    text: str
    used_fallback: bool
    fallback_reason: str | None


# ─────────────────────────────────────────────────────────────────────────
# Prompt composition
# ─────────────────────────────────────────────────────────────────────────


def _compose_user_prompt(req: NarrationRequest) -> str:
    """Kind-specific prompt. Keeps Opus focused on exactly the facts it has."""
    if req.kind == "correlation":
        src = req.payload.get("source_metric", "a behavior")
        tgt = req.payload.get("target_metric", "a health metric")
        direction = req.payload.get("direction", "positive")
        tier = req.payload.get("confidence_tier", "")
        sample_size = req.payload.get("sample_size", 0)
        literature = req.payload.get("literature_ref")

        dir_phrase = "higher together" if direction == "positive" else "moving opposite"
        parts = [
            f"Pattern: when {src.replace('_', ' ')} moves, {tgt.replace('_', ' ')} tends to move {dir_phrase}.",
            f"Based on {sample_size} days of the user's data.",
            f"Confidence tier: {tier.replace('_', ' ')}." if tier else "",
        ]
        if literature:
            parts.append(f"Research supports this ({literature}).")
        return "\n".join(p for p in parts if p)

    if req.kind == "anomaly":
        metric = req.payload.get("metric_key", req.subject_metrics[0] if req.subject_metrics else "a metric")
        observation_date = req.payload.get("observation_date", "recently")
        direction = req.payload.get("direction", "unusual")
        z = req.payload.get("z_score", 0.0)
        observed = req.payload.get("observed_value")
        forecasted = req.payload.get("forecasted_value")
        confirmed = req.payload.get("confirmed_by_bocpd", False)

        lines = [
            f"Anomaly: {metric.replace('_', ' ')} on {observation_date} was {direction} by {abs(z):.1f} standard deviations.",
        ]
        if observed is not None and forecasted is not None:
            lines.append(f"Observed {observed}, forecast was about {forecasted}.")
        if confirmed:
            lines.append("A change-point detector also fired, so this is a two-signal confirmed shift.")
        return "\n".join(lines)

    # Generic fallback — narrator will produce something generic or we fall
    # back to the template below.
    return f"Kind: {req.kind}. Subjects: {', '.join(req.subject_metrics) or 'unknown'}."


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────


async def generate_narration(
    request: NarrationRequest,
    client: object | None = None,
) -> NarrationResult:
    """Run Opus narration with voice compliance enforcement.

    ``client`` is injected for tests. In production it defaults to a fresh
    Anthropic client using the backend's configured API key. One retry:
    first call returns text, we run compliance, on em-dash failure we
    mechanically scrub; on any other failure or a second-pass failure we
    return the templated fallback for this kind.
    """
    from ml.narrate import voice_compliance

    user_prompt = _compose_user_prompt(request)

    # Resolve the Anthropic client lazily. Importing anthropic at module
    # load is fine (it is already pulled by the main app), but grabbing the
    # API key here lets tests swap in a stub without touching settings.
    if client is None:
        from app.config import settings
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Centralized model ID (PR #95): one env-var override at deploy time.
    from app.config import settings as _settings
    narration_model = _settings.anthropic_model_opus

    # Single Opus call; keep max_tokens small so the model does not meander.
    raw_text: str | None = None
    try:
        # Offload the synchronous Anthropic SDK call so generate_narration (an
        # async function reached from the /explain-finding request path) does
        # not block the event loop for the duration of the Opus call. Audit P2a.
        response = await asyncio.to_thread(
            client.messages.create,  # type: ignore[attr-defined]
            model=narration_model,
            system=_NARRATION_SYSTEM_PROMPT,
            max_tokens=NARRATION_MAX_TOKENS,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Anthropic SDK returns content blocks; take the first text block.
        blocks = getattr(response, "content", []) or []
        for block in blocks:
            text = getattr(block, "text", None)
            if text:
                raw_text = text.strip()
                break
    except Exception as e:  # noqa: BLE001 — narrator never propagates LLM errors
        logger.warning("Opus narration failed: %s", e)
        return _fallback(request.kind, "api_error")

    if not raw_text:
        return _fallback(request.kind, "empty_response")

    # First compliance pass.
    check = voice_compliance.check_all(raw_text)
    if check.passed:
        return NarrationResult(text=raw_text, used_fallback=False, fallback_reason=None)

    # Em dash is mechanically fixable; retry compliance after scrub.
    if check.em_dash:
        scrubbed = voice_compliance.scrub_em_dashes(raw_text)
        recheck = voice_compliance.check_all(scrubbed)
        if recheck.passed:
            return NarrationResult(
                text=scrubbed,
                used_fallback=False,
                fallback_reason=None,
            )

    logger.warning(
        "narration rejected for voice compliance: %s | raw=%r",
        ", ".join(check.details),
        raw_text,
    )
    return _fallback(request.kind, "voice_compliance")


def _fallback(kind: str, reason: str) -> NarrationResult:
    """Return the templated fallback for this kind."""
    template = _TEMPLATE_FALLBACK.get(kind, _TEMPLATE_FALLBACK["correlation"])
    return NarrationResult(text=template, used_fallback=True, fallback_reason=reason)
