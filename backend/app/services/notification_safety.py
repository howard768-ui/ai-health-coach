"""Deterministic safety net for AI-generated notification copy.

The notification prompts instruct the model never to put raw numbers / metrics
on the lock screen, but a prompt rule is not a guarantee: a model can echo an
HRV/RHR/sleep value, and a malformed (non-JSON) response used to be dumped to
the body verbatim. Both are PHI on the lock screen.

This module enforces the rule in code, after generation, so a prompt slip or a
parse failure falls back to static safe copy instead of leaking. Audit B1/H1.
"""

import logging
import re

logger = logging.getLogger("meld.notifications.safety")

# Any digit is disallowed in notification title/body: the "no raw metrics on the
# lock screen" rule. Static engagement copy that legitimately needs a count
# should not flow through here (it does not call the AI path).
_DIGIT_RE = re.compile(r"\d")


def contains_numbers(text: str | None) -> bool:
    """True if the text contains any digit (a potential raw metric / PHI)."""
    return bool(text and _DIGIT_RE.search(text))


def safe_notification_text(
    title: str,
    body: str,
    fallback_title: str,
    fallback_body: str,
) -> tuple[str, str]:
    """Return (title, body), swapping in the static fallback if either field
    contains a digit. The fallback copy is author-controlled and metric-free.
    """
    if contains_numbers(title) or contains_numbers(body):
        logger.warning(
            "AI notification copy contained digits; replaced with safe fallback"
        )
        return fallback_title, fallback_body
    return title, body
