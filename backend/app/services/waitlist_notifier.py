"""Admin email alert for new waitlist signups, sent via Resend.

Wired into `POST /api/waitlist/subscribe` as a FastAPI BackgroundTask so it
runs after the user already has a 200 response. A flaky email provider
must never affect signup latency or success.

Operational contract:
- Empty RESEND_API_KEY = silent skip (logged at INFO). Lets dev / preview
  environments run with no email plumbing.
- Any exception during send = logged at WARNING, swallowed. The `notified`
  flag stays False so a future retry path (scheduled task, manual replay)
  can pick the row up.
- Success = flips `notified=True`. Idempotent on re-entry.

Privacy: this routes the signer's email through Resend. Resend is listed
as a sub-processor in privacy.astro section 5.
"""

import asyncio
import logging

import resend

from app.config import settings
from app.database import async_session
from app.models.waitlist import WaitlistSignup

logger = logging.getLogger("meld.waitlist.notifier")


def _render(signup: WaitlistSignup) -> tuple[str, str]:
    bits: list[str] = []
    if signup.utm_source:
        bits.append(f"utm_source={signup.utm_source}")
    if signup.utm_medium:
        bits.append(f"utm_medium={signup.utm_medium}")
    if signup.utm_campaign:
        bits.append(f"utm_campaign={signup.utm_campaign}")
    if signup.source:
        bits.append(f"source={signup.source}")
    attribution = ", ".join(bits) or "direct (no attribution)"

    subject = f"New Meld signup: {signup.email}"
    html = (
        f"<p>New waitlist signup at {signup.created_at:%Y-%m-%d %H:%M UTC}.</p>"
        f"<p><strong>Email:</strong> {signup.email}</p>"
        f"<p><strong>Attribution:</strong> {attribution}</p>"
        f"<p><strong>Referer:</strong> {signup.referer or 'direct'}</p>"
    )
    return subject, html


async def send_new_signup_alert(signup_id: int) -> bool:
    """Send the admin Resend alert for a freshly created waitlist row.

    Re-loads the row in its own DB session so it can run as a FastAPI
    BackgroundTask after the request's session has been closed.

    Returns True on a fresh successful send, False on skip or failure.
    """
    if not settings.resend_api_key:
        logger.info(
            "waitlist notifier: RESEND_API_KEY empty, skipping signup_id=%d",
            signup_id,
        )
        return False

    async with async_session() as db:
        signup = await db.get(WaitlistSignup, signup_id)
        if signup is None:
            logger.warning("waitlist notifier: signup_id=%d not found", signup_id)
            return False
        if signup.notified:
            return False

        subject, html = _render(signup)

        try:
            resend.api_key = settings.resend_api_key
            await asyncio.to_thread(
                resend.Emails.send,
                {
                    "from": settings.resend_from,
                    "to": [settings.resend_admin_to],
                    "subject": subject,
                    "html": html,
                },
            )
        except Exception as exc:
            logger.warning(
                "waitlist notifier: send failed for signup_id=%d: %s",
                signup_id,
                exc,
            )
            return False

        signup.notified = True
        await db.commit()
        logger.info("waitlist notifier: sent alert for signup_id=%d", signup_id)
        return True
