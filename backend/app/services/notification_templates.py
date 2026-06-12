"""Template-based notification content system.

DOVA principle: check templates before calling AI (40-60% cost savings).
Templates use {variable} interpolation for personalization.
Falls back to AI generation when no template matches the context.

STATUS (2026-04-30 audit): UNWIRED. The `notification_templates` DB
table exists (created by alembic 6edcb0a00c24) and `backend.yml` CI
verifies it post-deploy, but `seed_templates` is never invoked at
startup and `pick_template` is never queried by `notification_content`,
which still hits Anthropic for every notification. The 40-60% cost
savings haven't shipped.

To wire it: call `seed_templates` from `lifespan()` in `app/main.py`
once, then patch `notification_content` generators to try
`pick_template` before constructing an Anthropic prompt. Seed data
below is preserved so the wiring work doesn't have to re-author copy.
See comprehensive scan recommendation 16 for the rest of the orphan
triage.
"""

import logging
import random

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.notification import NotificationTemplate

logger = logging.getLogger("meld.templates")


# ── Seed Data ───────────────────────────────────────────────
# Pre-authored notification variants. No AI cost, instant, fully controlled.
# All follow the wiki rules: no raw metrics, no emoji, 4th grade reading level.

SEED_TEMPLATES = [
    # Morning Brief — recovery high
    ("morning_brief", "recovery_high", "Good morning, {user_name}", "Recovery looks great today. Perfect day to challenge yourself."),
    ("morning_brief", "recovery_high", "Good morning, {user_name}", "Your body is well-rested. Time to make the most of it."),
    ("morning_brief", "recovery_high", "Good morning, {user_name}", "Strong recovery today. A hard workout would go well."),

    # Morning Brief — recovery moderate
    ("morning_brief", "recovery_moderate", "Good morning, {user_name}", "Recovery is fair today. A moderate session would be smart."),
    ("morning_brief", "recovery_moderate", "Good morning, {user_name}", "Not your best recovery. Listen to your body and adjust today."),
    ("morning_brief", "recovery_moderate", "Good morning, {user_name}", "Middle-of-the-road recovery. A lighter workout is a good call."),

    # Morning Brief — recovery low
    ("morning_brief", "recovery_low", "Good morning, {user_name}", "Your body needs extra rest today. Easy does it."),
    ("morning_brief", "recovery_low", "Good morning, {user_name}", "Sleep was rough. Focus on recovery today, not pushing hard."),
    ("morning_brief", "recovery_low", "Good morning, {user_name}", "Take it easy today. A walk and early dinner will help tonight."),

    # Streak Saver — at risk
    ("streak_saver", "streak_at_risk", "Don't lose your streak", "You're at {streak_count} of {streak_goal} days this week. One more session keeps it alive."),
    ("streak_saver", "streak_at_risk", "Almost there", "Just one more day to hit your goal this week. You've got this."),
    ("streak_saver", "streak_at_risk", "Quick check-in", "Your streak is on the line. Even a short session counts."),

    # Streak Saver — about to miss daily goal
    ("streak_saver", "goal_at_risk", "Still time today", "You haven't logged anything today. A quick walk or stretch keeps the chain going."),
    ("streak_saver", "goal_at_risk", "End of day check", "Today's almost over. Even ten minutes counts toward your goal."),

    # Weekly Review
    ("weekly_review", "positive_week", "Your week in review", "Solid week. Sleep improved and you hit {week_workout_days} workout days. Keep it going."),
    ("weekly_review", "positive_week", "Week wrapped up", "Good consistency this week. Your body is responding well."),
    ("weekly_review", "neutral_week", "Your week in review", "Mixed week. Some good days, some tough ones. Review inside."),
    ("weekly_review", "neutral_week", "Week wrapped up", "An okay week overall. Small improvements add up over time."),
    ("weekly_review", "tough_week", "Your week in review", "Tough week, but you showed up. Recovery matters as much as effort."),
    ("weekly_review", "tough_week", "Week wrapped up", "Not your strongest week. That's okay. Next week is a fresh start."),

    # Bedtime Coaching — additional templates
    ("bedtime_coaching", "high_strain", "Time to wind down", "You pushed hard today. Your body earned some extra rest tonight."),
    ("bedtime_coaching", "high_strain", "Almost bedtime", "Big day behind you. Dim the lights and let your body recover."),
    ("bedtime_coaching", "normal_day", "Time to wind down", "Good day overall. A calm evening will set up tomorrow well."),
    ("bedtime_coaching", "normal_day", "Wind-down time", "Wrapping up the day. Try putting your phone down early tonight."),
    ("bedtime_coaching", "poor_recovery", "Rest up tonight", "Your recovery needs a boost. Prioritize sleep tonight."),
    ("bedtime_coaching", "poor_recovery", "Early night tonight", "Recovery was low today. An early bedtime could turn things around."),

    # Health Alert
    ("health_alert", "metric_deviation", "Something looks different", "Your recovery data shifted from your usual pattern. Worth a look."),
    ("health_alert", "metric_deviation", "Check your data", "We noticed something unusual in today's health data. Tap to review."),
    ("health_alert", "concerning", "Your coach flagged something", "Some of your health signals are outside your normal range. Take a look when you can."),
]


async def seed_templates(db: AsyncSession):
    """Seed the notification_templates table if empty."""
    result = await db.execute(select(NotificationTemplate).limit(1))
    if result.scalar_one_or_none():
        return  # Already seeded

    for category, context, title, body in SEED_TEMPLATES:
        db.add(NotificationTemplate(
            category=category, context=context, title=title, body=body,
        ))
    await db.commit()
    logger.info("Seeded %d notification templates", len(SEED_TEMPLATES))


async def pick_template(
    db: AsyncSession,
    category: str,
    context: str,
    variables: dict | None = None,
) -> dict | None:
    """Pick a random active template matching category + context.

    Interpolates {variables} into the title and body.
    Returns None if no matching template found (caller should fall back to AI).
    """
    result = await db.execute(
        select(NotificationTemplate).where(
            NotificationTemplate.category == category,
            NotificationTemplate.context == context,
            NotificationTemplate.is_active,
        )
    )
    templates = result.scalars().all()
    if not templates:
        return None

    template = random.choice(templates)
    variables = variables or {}

    try:
        title = template.title.format(**variables)
        body = template.body.format(**variables)
    except KeyError as e:
        logger.warning("Template variable missing: %s", e)
        title = template.title
        body = template.body

    return {"title": title[:50], "body": body[:150]}
