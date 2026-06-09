"""Notification content generator.

Uses the CoachEngine to generate short-form notification content.
All notifications follow the wiki rules:
- No raw health metrics in the title (privacy on lock screen)
- Body max ~120 chars, 4th grade reading level
- Evidence-bound: cite specific data when possible
"""

import logging
from datetime import datetime

import anthropic

from app.services.coach_engine import CoachEngine, ModelTier
from app.services.notification_media import generate_recovery_badge
from app.services.notification_safety import safe_notification_text
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.notifications")

# Prompt templates for short-form notification content
MORNING_BRIEF_PROMPT = """Generate a push notification for a morning health check-in.

STRICT RULES — violating any of these is a failure:
1. Title: max 40 chars. Warm, simple greeting. Example: "Good morning, Brock"
2. Body: max 120 chars. Contextual language about recovery and ONE action for today.
3. NEVER include numbers, percentages, scores, or raw metrics (e.g. "48%", "63", "68ms"). These are private health data visible on the lock screen. Instead use words like "great", "solid", "rough", "low", "high".
4. NEVER use emoji. Not one. Zero emoji anywhere.
5. 4th grade reading level. Short sentences. Simple words.
6. Be warm and human, not robotic.

GOOD examples:
- "Recovery looks strong. Great day to push your workout."
- "Sleep was rough last night. Take it easy today."
- "Your body is well-rested. Time to challenge yourself."

BAD examples (never do these):
- "Recovery is moderate today (readiness: 63)." ← raw number exposed
- "Your sleep was 48%." ← percentage on lock screen
- "Good morning! 🌅" ← emoji

Health context (for your reference only, do NOT put these numbers in the output): {health_data}

Respond in EXACTLY this JSON format and nothing else:
{{"title": "...", "body": "..."}}"""


class NotificationEngine:
    """Generates notification content using the CoachEngine pipeline."""

    def __init__(self):
        self.coach = CoachEngine()

    def generate_morning_brief(
        self, health_data: dict, user_name: str = "there"
    ) -> dict:
        """Generate morning brief notification content.

        Returns dict with title, body, category, and APNs metadata.
        """
        # Determine recovery level for routing
        from app.core.constants import readiness_level
        readiness = health_data.get("readiness_score", 0)
        recovery_level = readiness_level(readiness) if readiness else "low"

        sleep_eff = health_data.get("sleep_efficiency")
        hrv = health_data.get("hrv_average")
        rhr = health_data.get("resting_hr")

        # Build a concise summary for the prompt
        data_summary = f"Recovery: {recovery_level}"
        if sleep_eff:
            data_summary += f", Sleep: {sleep_eff:.0f}%"
        if hrv:
            data_summary += f", HRV: {hrv:.0f}ms"
        if rhr:
            data_summary += f", RHR: {rhr:.0f}bpm"

        # Use CoachEngine for AI-generated content. Explicit Sonnet tier:
        # the query is a static template, and keyword routing on template
        # wording is nondeterministic and can bill Opus. Safety can still
        # escalate inside process_query. Audit P3/C3.
        try:
            result = self.coach.process_query(
                query=MORNING_BRIEF_PROMPT.format(health_data=data_summary),
                health_data=health_data,
                user_name=user_name,
                model_tier=ModelTier.SONNET,
            )
            response_text = result["response"]

            # Parse JSON response
            import json
            try:
                content = json.loads(response_text)
                title = content.get("title", f"Good morning, {user_name}")
                body = content.get("body", "Your coach has an update for you.")
            except (json.JSONDecodeError, KeyError):
                # Do NOT dump the raw model response to the lock screen (it can
                # contain raw metrics / PHI). Use the static safe copy. Audit B1.
                logger.warning("Failed to parse morning brief JSON; using safe fallback")
                title = f"Good morning, {user_name}"
                body = "Your coach has an update for you."

        except anthropic.APIError as e:
            logger.error("Failed to generate morning brief via AI: %s", e)
            # Deterministic fallback based on data
            title = f"Good morning, {user_name}"
            if recovery_level == "high":
                body = "Recovery looks great today. Good day to push your training."
            elif recovery_level == "moderate":
                body = "Recovery is moderate. A lighter session might be best today."
            else:
                body = "Your body needs some extra rest today. Easy does it."

        # Deterministic safety net: never let a digit (raw metric / PHI) reach
        # the lock screen even if the model ignored the prompt rule. Audit B1.
        title, body = safe_notification_text(
            title, body, f"Good morning, {user_name}", "Your coach has an update for you."
        )

        # Generate recovery badge for rich notification.
        # P3-3: URLs come from settings, not hardcoded.
        from app.config import settings
        base_url = settings.local_base_url if settings.app_env == "development" else settings.public_base_url
        media_url = generate_recovery_badge(recovery_level, base_url=base_url)

        return {
            "title": title[:50],
            "body": body[:150],
            "category": "morning_brief",
            "apns": {
                "category": "MORNING_BRIEF",
                "thread_id": "daily-coaching",
                "interruption_level": "active",
                "relevance_score": 0.8,
                "collapse_id": f"morning-brief-{utcnow_naive().strftime('%Y-%m-%d')}",
            },
            "data": {
                "deep_link": "meld://dashboard",
                "notification_type": "morning_brief",
            },
            "media_url": media_url,
        }


# Singleton
notification_engine = NotificationEngine()
