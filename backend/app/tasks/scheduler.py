"""Background task scheduler for push notifications.

Uses APScheduler's AsyncIOScheduler to run periodic jobs.
All jobs check anti-fatigue gates before sending.
"""

import asyncio
import logging
from datetime import datetime

import httpx
import sentry_sdk
from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, desc
from sqlalchemy.exc import SQLAlchemyError

from app.database import async_session
from app.models.health import SleepRecord
from app.models.notification import DeviceToken, NotificationRecord
from app.services.apns import apns_client
from app.services.notification_engine import notification_engine
from app.services.notification_content import content_generator
from app.services.anti_fatigue import can_send
from app.services.notification_templates import seed_templates, pick_template
from app.services.coach_engine import SafetyCheck
from app.services.health_data import get_latest_health_data
from app.services.oura_sync import sync_user_data as oura_sync
from app.services.peloton_sync import sync_user_data as peloton_sync
from app.services.garmin_sync import sync_user_data as garmin_sync
from app.services.data_reconciliation import reconcile_day
from app.services.correlation_engine import compute_correlations
from app.services.literature import literature_service
from app.services.oura_webhooks import list_subscriptions, renew_subscription
from app.services.offline_eval import run_offline_eval
from app.core.constants import ReadinessThreshold, StreakGoal, TEST_DEVICE_TOKEN, WeeklyReviewThreshold
from app.core.time import utcnow_naive

logger = logging.getLogger("meld.scheduler")

scheduler = AsyncIOScheduler()


# ── User discovery ──────────────────────────────────────────
#
# Scheduler jobs run outside request context — they can't use Depends(current_user).
# Multi-user fan-out runs through `iter_active_users`. The single-user helper
# below is retained only for `timing_refresh_job`, which reschedules a single
# global cron trigger and so cannot meaningfully fan out per-user without a
# per-user cron design (separate piece of work).

async def _get_primary_user_id(db) -> str | None:
    """Return the apple_user_id of the primary active user, or None.

    Used only by ``timing_refresh_job``. Multi-user notification timing is
    deferred until per-user cron design lands.
    """
    from app.models.user import User
    result = await db.execute(
        select(User)
        .where(User.is_active == True)  # noqa: E712 -- SQLAlchemy boolean
        .order_by(User.created_at)
        .limit(1)
    )
    user = result.scalar_one_or_none()
    return user.apple_user_id if user else None


async def _get_primary_user(db):
    """Return the primary User object (or None if no active user exists).

    Used only by ``timing_refresh_job``. See ``_get_primary_user_id`` note.
    """
    user_id = await _get_primary_user_id(db)
    if user_id is None:
        return None
    from app.models.user import User
    result = await db.execute(select(User).where(User.apple_user_id == user_id))
    return result.scalar_one_or_none()


async def iter_active_users(db) -> list:
    """Return every active User in created_at order.

    MEL-45 part 3: the per-user fan-out for scheduler jobs. Callers run their
    per-user work in a try/except so one bad user doesn't abort the rest of
    the tick.

    The 'default' placeholder fallback was removed in MEL-45 part 4; the
    placeholder row itself was dropped in migration e9c3f7b285a1, so this
    query no longer needs to filter it out.
    """
    from app.models.user import User
    result = await db.execute(
        select(User)
        .where(User.is_active == True)  # noqa: E712 -- SQLAlchemy boolean
        .order_by(User.created_at)
    )
    return list(result.scalars().all())


def _first_name_of(user) -> str:
    """Get first name for notification templates, or 'there' as fallback."""
    if user and user.name:
        return user.name.split()[0]
    return "there"


# ── Shared Helpers ──────────────────────────────────────────

async def _get_active_tokens(db, user_id: str) -> list:
    """Get all active device tokens for a user (excluding test tokens)."""
    result = await db.execute(
        select(DeviceToken).where(
            DeviceToken.user_id == user_id,
            DeviceToken.is_active == True,
            DeviceToken.token != TEST_DEVICE_TOKEN,
        )
    )
    return result.scalars().all()


async def _get_latest_health_data(db, user_id: str) -> dict:
    """Get latest reconciled health data for scheduler jobs.

    Delegates to the canonical health_data.get_latest_health_data() so that
    scheduler notifications use the same multi-source reconciled data as the
    coach and dashboard — not stale SleepRecord-only data.
    """
    return await get_latest_health_data(db, user_id)


async def _send_notification(db, user_id: str, tokens: list, content: dict):
    """Send a notification to all active device tokens for a user and log it."""
    for token_row in tokens:
        apns_result = await apns_client.send_push(
            device_token=token_row.token,
            title=content["title"],
            body=content["body"],
            category=content["apns"]["category"],
            thread_id=content["apns"]["thread_id"],
            interruption_level=content["apns"]["interruption_level"],
            relevance_score=content["apns"]["relevance_score"],
            collapse_id=content["apns"].get("collapse_id"),
            data=content["data"],
            media_url=content.get("media_url"),
        )

        record = NotificationRecord(
            user_id=user_id,
            device_token_id=token_row.id,
            category=content["category"],
            title=content["title"],
            body=content["body"],
            payload_json=content,
        )
        db.add(record)

        if apns_result.get("success"):
            logger.info("%s sent to device %d", content["category"], token_row.id)
        else:
            error_msg = apns_result.get("error")
            status = apns_result.get("status")
            logger.error(
                "%s failed for device %d: %s",
                content["category"], token_row.id, error_msg,
            )
            if status == 410:
                # Expected when a user uninstalls; mark inactive, don't alert.
                token_row.is_active = False
            else:
                # Other failures (5xx, JWT/PEM parse, network) signal systemic
                # issues we want visible in Sentry alongside the log line. The
                # 2026-04-30 APNs incident sat in logs for 1+ week before
                # PR #80's startup verifier surfaced it; this catches the same
                # class of issue at the runtime send path too.
                try:
                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("apns_status", str(status))
                        scope.set_extra("category", content["category"])
                        scope.set_extra("device_token_id", token_row.id)
                        scope.set_extra("error", error_msg)
                        sentry_sdk.capture_message(
                            f"APNs send failed: {error_msg}",
                            level="error",
                        )
                except Exception:  # noqa: BLE001 -- never let Sentry crash a send
                    logger.debug("Sentry capture failed (non-fatal)", exc_info=True)

    await db.commit()


# ── Jobs ────────────────────────────────────────────────────

# P2-2: Shared shell for all notification jobs.
#
# The 6 notification jobs (morning_brief, coaching_nudge, bedtime_coaching,
# streak_saver, weekly_review, health_alert) all share the same boilerplate:
#   1. Open a DB session
#   2. Load the primary user (skip if none)
#   3. Gate on can_send(category)
#   4. Load active device tokens (skip if none)
#   5. Build notification content (job-specific — provided via callback)
#   6. Send notification if content is non-None
#
# The old code duplicated steps 1-4 and 6 across all six jobs. Extracting
# the shell leaves each job as just the content-building callback.
async def _run_notification_job(
    job_name: str,
    category: str,
    content_fn,
) -> None:
    """Shared shell for scheduler notification jobs.

    Args:
        job_name: Human-readable name for logging (e.g. "morning_brief_job").
        category: Anti-fatigue category key — passed to `can_send()`.
        content_fn: async callable(db, user_id, user_name) -> dict | None.
            Return None to skip the notification (e.g. streak on track,
            no concerning health data, frequency preference says skip).
    """
    logger.info("Running %s", job_name)
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("No active users — skipping %s", job_name)
            return

        # Materialize ids/names while the ORM objects are fresh: the mid-loop
        # rollback() below expires them, and touching an expired attribute on
        # an async session raises MissingGreenlet, which would kill the rest
        # of the tick (the exact failure the rollback exists to prevent).
        user_rows = [(u.apple_user_id, _first_name_of(u)) for u in users]

        sent = 0
        for user_id, user_name in user_rows:
            try:
                if not await can_send(db, user_id, category):
                    continue

                tokens = await _get_active_tokens(db, user_id)
                if not tokens:
                    # Logs first 12 chars of user_id + job name only.
                    # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
                    logger.info("No active tokens for user %s, skipping %s", user_id[:12], job_name)
                    continue

                content = await content_fn(db, user_id, user_name)
                if not content:
                    continue

                await _send_notification(db, user_id, tokens, content)
                sent += 1
            except Exception:  # noqa: BLE001 -- one user's failure must not abort the rest
                logger.exception("%s failed for user %s", job_name, user_id[:12])
                # Roll back so a failed/pending transaction does not poison the
                # shared session for the remaining users in this run (audit P2a).
                await db.rollback()
    logger.info("%s complete (sent=%d/%d)", job_name, sent, len(users))


async def morning_brief_job():
    """Daily morning brief with recovery score and coaching line.
    Uses templates first (DOVA: 40-60% cost savings), falls back to AI.
    """
    async def build(db, user_id, user_name):
        health_data = await _get_latest_health_data(db, user_id)

        # Determine context for template selection
        readiness = health_data.get("readiness_score", 0)
        context = (
            "recovery_high" if readiness >= ReadinessThreshold.HIGH
            else "recovery_moderate" if readiness >= ReadinessThreshold.MODERATE
            else "recovery_low"
        )

        # Try template first (DOVA: no AI cost)
        template_content = await pick_template(
            db, "morning_brief", context, {"user_name": user_name}
        )
        if template_content:
            logger.info("Morning brief from template (context=%s)", context)
            return {
                **template_content,
                "category": "morning_brief",
                "apns": {
                    "category": "MORNING_BRIEF",
                    "thread_id": "daily-coaching",
                    "interruption_level": "active",
                    "relevance_score": 0.8,
                },
                "data": {"deep_link": "meld://dashboard", "notification_type": "morning_brief"},
            }
        # Fallback to AI generation
        logger.info("Morning brief from AI (no template for context=%s)", context)
        # Offload the synchronous Claude SDK call so it does not block the
        # AsyncIOScheduler event loop (and all API request handling) for the
        # 10-30s of the call. Audit P2a.
        return await asyncio.to_thread(
            notification_engine.generate_morning_brief, health_data, user_name=user_name
        )

    await _run_notification_job("morning_brief_job", "morning_brief", build)


async def coaching_nudge_job():
    """Cross-domain insight notification. Frequency set by user preference."""
    async def build(db, user_id, user_name):
        # Check frequency preference
        from app.models.notification import NotificationPreference
        pref_result = await db.execute(
            select(NotificationPreference).where(NotificationPreference.user_id == user_id)
        )
        pref = pref_result.scalar_one_or_none()
        frequency = pref.nudge_frequency if pref else "2x_week"

        today = utcnow_naive().weekday()  # 0=Mon, 6=Sun
        if frequency == "weekly" and today != 0:
            logger.info("Nudge frequency is weekly, today is not Monday — skipping")
            return None
        if frequency == "2x_week" and today not in (0, 3):  # Mon, Thu
            logger.info("Nudge frequency is 2x/week, today is not Mon/Thu — skipping")
            return None
        # "daily" sends every day — no skip

        health_data = await _get_latest_health_data(db, user_id)
        return await asyncio.to_thread(
            content_generator.generate_coaching_nudge, health_data, user_name=user_name
        )

    await _run_notification_job("coaching_nudge_job", "coaching_nudge", build)


async def bedtime_coaching_job():
    """Wind-down reminder timed to sleep window."""
    async def build(db, user_id, user_name):
        health_data = await _get_latest_health_data(db, user_id)
        return await asyncio.to_thread(
            content_generator.generate_bedtime_coaching, health_data, user_name=user_name
        )

    await _run_notification_job("bedtime_coaching_job", "bedtime_coaching", build)


async def streak_saver_job():
    """Evening check — only fires when user is about to miss their streak."""
    async def build(db, user_id, user_name):
        # Calculate active days this week using readiness as proxy for workout days.
        from sqlalchemy import func as sqlfunc
        from datetime import timedelta
        today = utcnow_naive().replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today - timedelta(days=today.weekday())
        result = await db.execute(
            select(sqlfunc.count(SleepRecord.id)).where(
                SleepRecord.user_id == user_id,
                SleepRecord.date >= week_start.strftime("%Y-%m-%d"),
                SleepRecord.readiness_score >= ReadinessThreshold.ACTIVE_DAY_MIN,
            )
        )
        active_days = result.scalar() or 0
        streak_goal = int(StreakGoal.DEFAULT_WEEKLY_TARGET)

        content = content_generator.generate_streak_saver(
            active_days, streak_goal, user_name=user_name
        )
        if not content:
            logger.info("Streak on track (%d/%d) — no alert needed", active_days, streak_goal)
            return None

        template = await pick_template(db, "streak_saver", "streak_at_risk", {
            "streak_count": str(active_days), "streak_goal": str(streak_goal),
        })
        if template:
            content["title"] = template["title"]
            content["body"] = template["body"]
        return content

    await _run_notification_job("streak_saver_job", "streak_alerts", build)


async def weekly_review_job():
    """Sunday evening weekly summary."""
    async def build(db, user_id, user_name):
        result = await db.execute(
            select(SleepRecord)
            .where(SleepRecord.user_id == user_id)
            .order_by(desc(SleepRecord.date))
            .limit(7)
        )
        records = result.scalars().all()

        workout_days = sum(
            1 for r in records
            if r.readiness_score and r.readiness_score >= ReadinessThreshold.ACTIVE_DAY_MIN
        )
        avg_sleep = sum(r.efficiency or 0 for r in records) / max(len(records), 1)
        if avg_sleep > WeeklyReviewThreshold.SLEEP_TREND_IMPROVING_MIN:
            sleep_trend = "improving"
        elif avg_sleep > WeeklyReviewThreshold.SLEEP_TREND_STABLE_MIN:
            sleep_trend = "stable"
        else:
            sleep_trend = "declining"

        if sleep_trend == "improving" and workout_days >= WeeklyReviewThreshold.POSITIVE_WEEK_WORKOUTS:
            week_context = "positive_week"
        elif workout_days < WeeklyReviewThreshold.TOUGH_WEEK_WORKOUTS:
            week_context = "tough_week"
        else:
            week_context = "neutral_week"

        template = await pick_template(db, "weekly_review", week_context, {
            "week_workout_days": str(workout_days),
        })

        content = content_generator.generate_weekly_review(
            {"workout_days": workout_days, "sleep_trend": sleep_trend},
            user_name=user_name,
        )
        if template:
            content["title"] = template["title"]
            content["body"] = template["body"]
        return content

    await _run_notification_job("weekly_review_job", "weekly_review", build)


async def health_alert_job():
    """Check health data for concerning deviations after sync."""
    async def build(db, user_id, user_name):
        health_data = await _get_latest_health_data(db, user_id)
        if not health_data:
            return None

        safety = SafetyCheck.check_health_data(health_data)
        if not safety.is_concerning:
            return None

        content = content_generator.generate_health_alert(health_data, safety.reasons)
        if not content:
            return None

        context = "concerning" if safety.requires_opus else "metric_deviation"
        template = await pick_template(db, "health_alert", context)
        if template:
            content["title"] = template["title"]
            content["body"] = template["body"]

        logger.info("Health alert sent: %s", safety.reasons)
        return content

    await _run_notification_job("health_alert_job", "health_alerts", build)


async def oura_sync_job():
    """Sync latest Oura data every 6 hours, fanning out across all active users."""
    logger.info("Running oura_sync_job")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("oura_sync_job: no active users yet")
            return
        # Ids materialized up front: rollback() in the except expires the ORM
        # rows, and expired-attribute access on an async session raises
        # MissingGreenlet (see _run_notification_job).
        for user_id in [u.apple_user_id for u in users]:
            try:
                result = await oura_sync(db, user_id)
                logger.info("oura_sync_job user=%s result=%s", user_id[:12], result)
                # Guard reconcile on a successful sync, matching the Peloton and
                # Garmin jobs below. Reconciling after a failed sync re-blesses
                # stale canonical rows from whatever partial state the failed
                # sync left behind. Audit P3 (round-2 low, scheduler.py:449).
                if result.get("status") == "ok":
                    today = utcnow_naive().strftime("%Y-%m-%d")
                    await reconcile_day(db, user_id, today)
            except Exception:  # noqa: BLE001 -- isolate one user's failure
                logger.exception("oura_sync_job failed for user %s", user_id[:12])
                # Clear any failed transaction so one user's DB error cannot
                # poison the shared session for every later user in the tick
                # (PendingRollbackError cascade). Same fix as the notification
                # jobs. Audit P3 (round-2 A46 class). The rollback itself must
                # never raise (that would defeat the isolation it exists for),
                # so swallow + log any failure.
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001
                    logger.debug("rollback after per-user failure also failed", exc_info=True)


async def peloton_sync_job():
    """Sync Peloton workouts every 6 hours, fanning out across all active users."""
    logger.info("Running peloton_sync_job")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("peloton_sync_job: no active users yet")
            return
        # Ids materialized up front; see oura_sync_job.
        for user_id in [u.apple_user_id for u in users]:
            try:
                result = await peloton_sync(db, user_id)
                logger.info("peloton_sync_job user=%s result=%s", user_id[:12], result)
                if result.get("status") == "ok":
                    today = utcnow_naive().strftime("%Y-%m-%d")
                    await reconcile_day(db, user_id, today)
            except Exception:  # noqa: BLE001
                logger.exception("peloton_sync_job failed for user %s", user_id[:12])
                # Clear any failed transaction; see oura_sync_job above.
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001
                    logger.debug("rollback after per-user failure also failed", exc_info=True)


async def garmin_sync_job():
    """Sync Garmin health data every 6 hours, fanning out across all active users."""
    logger.info("Running garmin_sync_job")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("garmin_sync_job: no active users yet")
            return
        # Ids materialized up front; see oura_sync_job.
        for user_id in [u.apple_user_id for u in users]:
            try:
                result = await garmin_sync(db, user_id)
                logger.info("garmin_sync_job user=%s result=%s", user_id[:12], result)
                if result.get("status") == "ok":
                    today = utcnow_naive().strftime("%Y-%m-%d")
                    await reconcile_day(db, user_id, today)
            except Exception:  # noqa: BLE001
                logger.exception("garmin_sync_job failed for user %s", user_id[:12])
                # Clear any failed transaction; see oura_sync_job above.
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001
                    logger.debug("rollback after per-user failure also failed", exc_info=True)


async def webhook_renewal_job():
    """Monthly: renew Oura webhook subscriptions before they expire (90-day TTL)."""
    logger.info("Running webhook_renewal_job")
    try:
        subs = await list_subscriptions()
        renewed = 0
        for sub in subs:
            try:
                await renew_subscription(sub["id"])
                renewed += 1
            except (httpx.HTTPError, KeyError, ValueError) as e:
                logger.error("Failed to renew webhook %s: %s", sub["id"], e)
        logger.info("Renewed %d/%d webhook subscriptions", renewed, len(subs))
    except (httpx.HTTPError, ValueError) as e:
        logger.error("Webhook renewal failed: %s", e)


async def feature_refresh_job():
    """Nightly: materialize the Signal Engine feature store for the primary user.

    Rebuilds the trailing 30 days of derived features so downstream ML jobs
    (correlation engine, forecasting, ranker) always read a fresh, consistent
    frame. Idempotent — rerunning the same day is a no-op against the data
    (values may change as new Oura webhook rows land, which is the point).

    Kept under the 60-second budget per the Phase 1 acceptance criteria in
    ``~/.claude/plans/golden-floating-creek.md``. Scheduled at 03:30 UTC so
    it completes before the correlation engine runs at 04:00 UTC on Sundays
    and before morning brief in any user timezone.
    """
    from datetime import date

    # Lazy import per boundary rules — the rest of app/ only reaches into
    # ``ml.api`` (see tests/ml/test_boundary.py).
    from ml import api as ml_api

    logger.info("Running feature_refresh_job")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("feature_refresh_job: no active users yet, skipping")
            return
        for user in users:
            user_id = user.apple_user_id
            try:
                rows = await ml_api.refresh_features_for_user(
                    db, user_id, through_date=date.today(), lookback_days=30
                )
                await db.commit()
                logger.info(
                    "feature_refresh_job: wrote %d rows for user %s",
                    rows,
                    user_id[:12],
                )
            except SQLAlchemyError as e:
                await db.rollback()
                logger.exception("feature_refresh_job DB error for user %s: %s", user_id[:12], e)
            except Exception:  # noqa: BLE001
                await db.rollback()
                logger.exception("feature_refresh_job failed for user %s", user_id[:12])


async def insight_candidate_job():
    """Daily: generate candidates, rank them, persist the top-N slate.

    Shadow mode: writes to ``ml_rankings`` with ``was_shown=False``. The
    ``/api/insights/daily`` endpoint flips ``was_shown=True`` on read when
    ``ml_shadow_insight_card`` is off. Until then the slate is a shadow
    log for A/B comparison and for future ranker training.

    Caps (1/day, 3/week) are enforced at read time via
    ``ml.ranking.heuristic.can_surface_today``. The job itself writes
    rankings regardless, so the shadow log captures full exposure history.
    """
    from ml import api as ml_api

    logger.info("Running insight_candidate_job")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("insight_candidate_job: no active users yet, skipping")
            return
        for user in users:
            user_id = user.apple_user_id
            try:
                report = await ml_api.run_daily_insights(db, user_id)
                await db.commit()
                logger.info(
                    "insight_candidate_job user=%s (shadow=%s): %d candidates, %d rankings, top=%s",
                    user_id[:12],
                    report.shadow_mode,
                    report.candidates_generated,
                    report.rankings_written,
                    report.top_candidate_id,
                )
            except SQLAlchemyError as e:
                await db.rollback()
                logger.exception("insight_candidate_job DB error for user %s: %s", user_id[:12], e)
            except Exception:  # noqa: BLE001
                await db.rollback()
                logger.exception("insight_candidate_job failed for user %s", user_id[:12])


async def baselines_job():
    """Nightly: run L1 baselines, forecasts, and residual anomaly detection.

    Shadow mode: populates ``ml_baselines``, ``ml_change_points``,
    ``ml_forecasts``, ``ml_anomalies`` but nothing user-facing reads them
    yet. Phase 3 wires L2 associations to use baselines; Phase 4 wires
    insight candidates to use anomalies.
    """
    from datetime import date as _date

    from ml import api as ml_api

    logger.info("Running baselines_job")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("baselines_job: no active users yet, skipping")
            return
        for user in users:
            user_id = user.apple_user_id
            try:
                report = await ml_api.run_discovery_pipeline(db, user_id)
                await db.commit()
                logger.info(
                    "baselines_job user=%s (shadow=%s): layers=%s counts=%s",
                    user_id[:12],
                    report.shadow_mode,
                    report.layers_run,
                    report.tier_counts,
                )
            except SQLAlchemyError as e:
                await db.rollback()
                logger.exception("baselines_job DB error for user %s: %s", user_id[:12], e)
            except Exception:  # noqa: BLE001
                await db.rollback()
                logger.exception("baselines_job failed for user %s", user_id[:12])


async def correlation_engine_job():
    """Weekly: discover cross-domain health correlations from user data.

    Routes through ``ml.api.run_associations`` (Phase 3). The legacy
    ``compute_correlations`` in ``app/services/correlation_engine.py`` is
    preserved for the parity test but no longer exercised in production.
    """
    from ml import api as ml_api

    logger.info("Running correlation_engine_job (via ml.api.run_associations)")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("correlation_engine_job: no active users yet, skipping")
            return
        for user in users:
            user_id = user.apple_user_id
            try:
                report = await ml_api.run_associations(db, user_id, window_days=30)
                await db.commit()
                logger.info(
                    "correlation_engine_job user=%s: tested=%d sig=%d dynamic=%d rows=%d",
                    user_id[:12],
                    report.pairs_tested,
                    report.significant_results,
                    report.dynamic_pairs_generated,
                    report.rows_written,
                )
            except SQLAlchemyError as e:
                await db.rollback()
                logger.exception(
                    "correlation_engine_job DB error for user %s: %s", user_id[:12], e
                )
            except Exception:  # noqa: BLE001
                await db.rollback()
                logger.exception("correlation_engine_job failed for user %s", user_id[:12])


async def granger_causal_job():
    """Weekly: L3 Granger causality + L4 DoWhy quasi-causal on developing+ pairs.

    Runs after ``correlation_engine_job`` (04:00) so fresh L2 associations
    are available. Shadow-gated behind ``ml_shadow_granger_causal``.
    Phase 6 of the Signal Engine.
    """
    from ml import api as ml_api

    if not ml_api.is_shadow_enabled("granger_causal"):
        logger.info("granger_causal_job: shadow flag off, skipping")
        return

    logger.info("Running granger_causal_job (L3 + L4)")
    async with async_session() as db:
        users = await iter_active_users(db)
        if not users:
            logger.info("granger_causal_job: no active users yet, skipping")
            return
        for user in users:
            user_id = user.apple_user_id
            try:
                granger_report = await ml_api.run_granger(db, user_id, window_days=90)
                logger.info(
                    "granger_causal_job L3 user=%s: tested=%d significant=%d updated=%d",
                    user_id[:12],
                    granger_report.pairs_tested,
                    granger_report.pairs_significant,
                    granger_report.correlations_updated,
                )

                causal_report = await ml_api.run_causal(
                    db, user_id, window_days=90, max_pairs=10
                )
                logger.info(
                    "granger_causal_job L4 user=%s: tested=%d passed=%d updated=%d",
                    user_id[:12],
                    causal_report.pairs_tested,
                    causal_report.pairs_passed,
                    causal_report.correlations_updated,
                )

                await db.commit()
            except SQLAlchemyError as e:
                await db.rollback()
                logger.exception(
                    "granger_causal_job DB error for user %s: %s", user_id[:12], e
                )
            except Exception:  # noqa: BLE001
                await db.rollback()
                logger.exception("granger_causal_job failed for user %s", user_id[:12])


async def ranker_training_job():
    """Weekly: retrain XGBoost LambdaMART ranker on feedback data.

    Runs after ``offline_eval_job`` (05:00) at 06:00 UTC Sunday. Trains the
    model, exports to CoreML (if coremltools installed), uploads to R2 (if
    configured), and registers in the model DB. Shadow-gated behind
    ``ml_shadow_coreml_ranker``. Phase 7A of the Signal Engine.
    """
    from ml import api as ml_api

    if not ml_api.is_shadow_enabled("coreml_ranker"):
        logger.info("ranker_training_job: shadow flag off, skipping")
        return

    logger.info("Running ranker_training_job")
    async with async_session() as db:
        try:
            summary = await ml_api.train_and_export_ranker(db)
            await db.commit()
            logger.info(
                "ranker_training_job done: trained=%s version=%s ndcg=%.4f "
                "coreml=%s r2=%s",
                summary.get("trained"),
                summary.get("model_version", "none"),
                summary.get("val_ndcg", 0.0),
                summary.get("coreml_exported", False),
                summary.get("r2_uploaded", False),
            )
            # Phase 10: alert on training completion.
            try:
                await ml_api.send_training_alert(summary)
            except Exception:
                logger.debug("Training alert failed (non-fatal)", exc_info=True)
        except Exception as e:
            await db.rollback()
            logger.exception("ranker_training_job error: %s", e)


async def cohort_clustering_job():
    """Monthly: anonymize opted-in users and cluster into archetypes.

    Runs on the 1st of each month at 05:30 UTC. Shadow-gated behind
    ``ml_shadow_cohorts``. Phase 8 of the Signal Engine.
    """
    from ml import api as ml_api

    if not ml_api.is_shadow_enabled("cohorts"):
        logger.info("cohort_clustering_job: shadow flag off, skipping")
        return

    logger.info("Running cohort_clustering_job")
    async with async_session() as db:
        try:
            summary = await ml_api.run_cohort_clustering(db)
            await db.commit()
            logger.info(
                "cohort_clustering_job done: users=%d vectors=%d clusters=%d "
                "noise=%d largest=%d run=%s",
                summary.get("users_processed", 0),
                summary.get("vectors_created", 0),
                summary.get("n_clusters", 0),
                summary.get("n_noise_points", 0),
                summary.get("largest_cluster", 0),
                summary.get("run_id", "none"),
            )
        except Exception as e:
            await db.rollback()
            logger.exception("cohort_clustering_job error: %s", e)


async def experiment_check_job():
    """Daily: check active experiments for phase transitions and completion.

    Advances experiment phases by date, runs APTE on completed ones.
    Shadow-gated behind ``ml_shadow_apte``. Phase 9 of the Signal Engine.
    """
    from ml import api as ml_api

    if not ml_api.is_shadow_enabled("apte"):
        logger.info("experiment_check_job: shadow flag off, skipping")
        return

    logger.info("Running experiment_check_job")
    async with async_session() as db:
        try:
            summary = await ml_api.check_and_complete_experiments(db)
            await db.commit()
            logger.info(
                "experiment_check_job done: checked=%d advanced=%d "
                "completed=%d failed=%d",
                summary.get("checked", 0),
                summary.get("advanced", 0),
                summary.get("completed", 0),
                summary.get("failed", 0),
            )
        except Exception as e:
            await db.rollback()
            logger.exception("experiment_check_job error: %s", e)


async def offline_eval_job():
    """Weekly: evaluate recent coach responses for quality regressions."""
    logger.info("Running offline_eval_job")
    async with async_session() as db:
        report = await run_offline_eval(db, days=7)
        logger.info(
            "Offline eval complete: %d evaluated, reading %.0f%%, grounding %.0f%%, %d flagged",
            report.get("total_evaluated", 0),
            report.get("reading_level", {}).get("pass_rate", 0),
            report.get("data_grounding", {}).get("pass_rate", 0),
            len(report.get("flagged_for_review", [])),
        )


async def synth_drift_job():
    """Daily: compare synth vs real biometrics, log drift, write HTML.

    Phase 4.5 Commit 5 wiring. The report reads from HealthMetricRecord
    partitioned on ``is_synthetic`` (Commit 3 column). When either
    partition is below the min-samples floor (e.g., synth hasn't been
    generated in this environment yet) the job short-circuits with
    ``dataset_too_small=True`` and logs. Cheap noise, no drama.

    The HTML path is best-effort: Evidently 0.7.21 fails to import on
    Python 3.14 (pydantic v1 incompat); Railway runs 3.12 so it works
    there. On failure ``html_path`` is None and only the KS summary
    is logged.

    Persists one ``ml_drift_results`` row per metric so
    ``/ops/ml/feature-drift`` can surface the latest snapshot without
    recomputing. Persistence is best-effort: if the write fails, the
    drift run still logs cleanly (the table might be missing in an
    environment that lagged the migration, and we degrade rather than
    crash the scheduler).
    """
    from ml import api as ml_api

    logger.info("Running synth_drift_job")
    persisted = 0
    async with async_session() as db:
        try:
            report = await ml_api.build_synth_drift_report(db)
        except SQLAlchemyError as e:
            logger.exception("synth_drift_job DB error: %s", e)
            return
        except Exception as e:  # noqa: BLE001 -- scheduler must not crash on job errors
            logger.exception("synth_drift_job unexpected error: %s", e)
            return

        # Persist per-feature KS results so ``/ops/ml/feature-drift``
        # can surface this run. No-op when the report was too small.
        if not report.dataset_too_small:
            try:
                persisted = await ml_api.persist_drift_results(db, report)
                if persisted > 0:
                    await db.commit()
            except SQLAlchemyError as e:
                logger.exception("synth_drift_job persist DB error: %s", e)
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001 -- rollback is best-effort
                    pass
            except Exception as e:  # noqa: BLE001 -- persist must not crash the job
                logger.exception("synth_drift_job persist error: %s", e)
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001
                    pass

    if report.dataset_too_small:
        logger.info(
            "synth_drift_job: dataset too small (ref=%d, cur=%d); nothing to compare yet",
            report.n_reference_rows,
            report.n_current_rows,
        )
        return

    logger.info(
        "synth_drift_job done: ref=%d cur=%d tested=%s drifted=%s persisted=%d html=%s backend=%s",
        report.n_reference_rows,
        report.n_current_rows,
        report.metrics_tested,
        report.drifted_metrics,
        persisted,
        report.html_path or "<none>",
        report.html_backend,
    )

    # Phase 10: alert on drift.
    try:
        from ml import api as ml_api_alerts
        await ml_api_alerts.send_drift_alert(report)
    except Exception:
        logger.debug("Drift alert failed (non-fatal)", exc_info=True)


async def get_personalized_timing(db, user_id: str) -> dict:
    """Calculate rolling 7-day average wake/sleep times from Oura data.

    Returns dict with wake_hour, wake_minute, bedtime_hour, bedtime_minute.
    Falls back to defaults if insufficient data.
    """
    result = await db.execute(
        select(SleepRecord)
        .where(
            SleepRecord.user_id == user_id,
            SleepRecord.bedtime_start.isnot(None),
            SleepRecord.bedtime_end.isnot(None),
        )
        .order_by(desc(SleepRecord.date))
        .limit(7)
    )
    records = result.scalars().all()

    if len(records) < 3:
        return {"wake_hour": 8, "wake_minute": 0, "bedtime_hour": 22, "bedtime_minute": 0}

    # Average bedtime_start (sleep time) and bedtime_end (wake time)
    sleep_minutes = []
    wake_minutes = []
    for r in records:
        if r.bedtime_start:
            h, m = map(int, r.bedtime_start.split(":"))
            # Handle after-midnight bedtimes (e.g., 01:30 = 25*60+30 for averaging)
            total = h * 60 + m
            if total < 12 * 60:  # Before noon = after midnight
                total += 24 * 60
            sleep_minutes.append(total)
        if r.bedtime_end:
            h, m = map(int, r.bedtime_end.split(":"))
            wake_minutes.append(h * 60 + m)

    if not sleep_minutes or not wake_minutes:
        return {"wake_hour": 8, "wake_minute": 0, "bedtime_hour": 22, "bedtime_minute": 0}

    avg_sleep = sum(sleep_minutes) / len(sleep_minutes)
    avg_wake = sum(wake_minutes) // len(wake_minutes)

    # Normalize back from 24+ hour range
    avg_sleep = avg_sleep % (24 * 60)

    return {
        "wake_hour": avg_wake // 60,
        "wake_minute": avg_wake % 60,
        "bedtime_hour": avg_sleep // 60,
        "bedtime_minute": avg_sleep % 60,
    }


async def timing_refresh_job():
    """Recalculate personalized notification timing from Oura sleep data.

    Runs daily at 03:00 UTC. Reschedules morning brief and bedtime coaching
    with updated times based on rolling 7-day sleep patterns.
    """
    logger.info("Running timing_refresh_job")
    async with async_session() as db:
        user_id = await _get_primary_user_id(db)
        if user_id is None:
            logger.info("No active user — skipping timing refresh")
            return
        timing = await get_personalized_timing(db, user_id)

    # Reschedule in the user's local timezone (P1-18 fix). Without this,
    # personalized timing fired in UTC — defeating the point of personalization.
    from app.config import settings as _settings
    user_tz = _settings.user_timezone

    # Reschedule morning brief: 30 min after average wake time
    wake_hour = timing["wake_hour"]
    wake_minute = timing["wake_minute"] + 30
    if wake_minute >= 60:
        wake_hour = (wake_hour + 1) % 24  # Modulo 24 to avoid hour=24 crash
        wake_minute -= 60

    try:
        scheduler.reschedule_job(
            "morning_brief",
            trigger=CronTrigger(hour=wake_hour, minute=wake_minute, timezone=user_tz),
        )
    except (JobLookupError, ValueError) as e:
        logger.warning("Failed to reschedule morning_brief: %s", e)

    # Reschedule bedtime coaching: 30 min before average bedtime
    bed_hour = timing["bedtime_hour"]
    bed_minute = timing["bedtime_minute"] - 30
    if bed_minute < 0:
        bed_hour = (bed_hour - 1) % 24
        bed_minute += 60

    try:
        scheduler.reschedule_job(
            "bedtime_coaching",
            trigger=CronTrigger(hour=bed_hour, minute=bed_minute, timezone=user_tz),
        )
    except (JobLookupError, ValueError) as e:
        logger.warning("Failed to reschedule bedtime_coaching: %s", e)

    logger.info(
        "Rescheduled: morning_brief=%02d:%02d, bedtime=%02d:%02d",
        wake_hour, wake_minute, bed_hour, bed_minute,
    )


# ── Scheduler Setup ─────────────────────────────────────────

async def _seed_on_start():
    """Seed notification templates on scheduler start."""
    async with async_session() as db:
        await seed_templates(db)


def start_scheduler():
    """Initialize and start the scheduler with all jobs."""
    # Seed templates
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_seed_on_start())
    except RuntimeError:
        asyncio.run(_seed_on_start())

    # User-facing notifications run in the user's local timezone (P1-18 fix).
    # Sync jobs and webhook renewal stay in UTC since they're internal.
    from app.config import settings as _settings
    user_tz = _settings.user_timezone

    # Morning brief: 08:00 user-local time
    scheduler.add_job(
        morning_brief_job,
        trigger=CronTrigger(hour=8, minute=0, timezone=user_tz),
        id="morning_brief",
        name="Morning Brief",
        replace_existing=True,
    )

    # Coaching nudge: daily at 12:00 user-local (frequency logic inside job checks preference)
    scheduler.add_job(
        coaching_nudge_job,
        trigger=CronTrigger(hour=12, minute=0, timezone=user_tz),
        id="coaching_nudge",
        name="Coaching Nudge",
        replace_existing=True,
    )

    # Bedtime coaching: daily at 22:00 user-local
    scheduler.add_job(
        bedtime_coaching_job,
        trigger=CronTrigger(hour=22, minute=0, timezone=user_tz),
        id="bedtime_coaching",
        name="Bedtime Coaching",
        replace_existing=True,
    )

    # Streak saver: daily at 18:00 user-local (evening check)
    scheduler.add_job(
        streak_saver_job,
        trigger=CronTrigger(hour=18, minute=0, timezone=user_tz),
        id="streak_saver",
        name="Streak Saver",
        replace_existing=True,
    )

    # Weekly review: Sundays at 18:00 user-local
    scheduler.add_job(
        weekly_review_job,
        trigger=CronTrigger(day_of_week="sun", hour=18, minute=0, timezone=user_tz),
        id="weekly_review",
        name="Weekly Review",
        replace_existing=True,
    )

    # Health alert: every 6 hours (checks for deviations)
    scheduler.add_job(
        health_alert_job,
        trigger=CronTrigger(hour="*/6"),
        id="health_alert",
        name="Health Alert Check",
        replace_existing=True,
    )

    # Peloton sync: every 6 hours
    scheduler.add_job(
        peloton_sync_job,
        trigger=CronTrigger(hour="*/6", minute=30),
        id="peloton_sync",
        name="Peloton Workout Sync",
        replace_existing=True,
    )

    # Garmin sync: every 6 hours
    scheduler.add_job(
        garmin_sync_job,
        trigger=CronTrigger(hour="*/6", minute=45),
        id="garmin_sync",
        name="Garmin Health Sync",
        replace_existing=True,
    )

    # Oura sync: every 30 minutes. Backs up the on-demand sync in
    # health.py's dashboard endpoint — if the user opens the app and has
    # stale data, dashboard sync kicks in; otherwise the scheduler keeps
    # things fresh in the background for webhook/notification jobs that
    # don't go through the dashboard path (e.g. morning_brief_job at 8am
    # needs last-night's sleep already in the DB). Timezone-aware so we
    # fire at consistent wall-clock times regardless of DST.
    from apscheduler.triggers.interval import IntervalTrigger
    scheduler.add_job(
        oura_sync_job,
        trigger=IntervalTrigger(minutes=30),
        id="oura_sync",
        name="Oura Background Sync",
        replace_existing=True,
    )

    # Feature refresh: nightly at 03:30 UTC, before correlation engine runs at
    # 04:00 UTC on Sundays and before user-local morning brief in any timezone.
    # Signal Engine Phase 1 (see ~/.claude/plans/golden-floating-creek.md).
    scheduler.add_job(
        feature_refresh_job,
        trigger=CronTrigger(hour=3, minute=30),
        id="feature_refresh",
        name="Signal Engine Feature Refresh",
        replace_existing=True,
    )

    # Baselines + forecasts + anomalies: nightly at 03:45 UTC, after features
    # land at 03:30. Shadow mode in Phase 2; nothing user-facing reads these
    # rows yet. Plan says Sunday-weekly but daily is cheap and gives fresher
    # forecasts; revisit when prod load demands it.
    scheduler.add_job(
        baselines_job,
        trigger=CronTrigger(hour=3, minute=45),
        id="baselines",
        name="Signal Engine L1 Baselines + Forecasts + Anomalies",
        replace_existing=True,
    )

    # Insight candidates + heuristic ranker: daily at 07:00 user-local, per
    # the Phase 4 plan. Runs after L2 associations (Sunday 04:00 UTC) and L1
    # forecasts (03:45 UTC daily) so candidate generation sees fresh data.
    # Shadow mode: ml_rankings gets populated with was_shown=False until
    # ml_shadow_insight_card is flipped off.
    scheduler.add_job(
        insight_candidate_job,
        trigger=CronTrigger(hour=7, minute=0, timezone=user_tz),
        id="insight_candidate",
        name="Signal Engine Phase 4 Insight Candidates + Ranker",
        replace_existing=True,
    )

    # Correlation engine: weekly Sunday 04:00
    scheduler.add_job(
        correlation_engine_job,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=0),
        id="correlation_engine",
        name="Cross-Domain Correlation Discovery",
        replace_existing=True,
    )

    # Granger + DoWhy causal: weekly Sunday 04:30 (after correlations finish)
    scheduler.add_job(
        granger_causal_job,
        trigger=CronTrigger(day_of_week="sun", hour=4, minute=30),
        id="granger_causal",
        name="Phase 6 L3 Granger + L4 DoWhy Causal",
        replace_existing=True,
    )

    # Cohort clustering: 1st of each month at 05:30 UTC
    scheduler.add_job(
        cohort_clustering_job,
        trigger=CronTrigger(day=1, hour=5, minute=30),
        id="cohort_clustering",
        name="Phase 8 Cross-User Cohort Clustering",
        replace_existing=True,
    )

    # Experiment lifecycle check: daily at 08:00 UTC
    scheduler.add_job(
        experiment_check_job,
        trigger=CronTrigger(hour=8, minute=0),
        id="experiment_check",
        name="Phase 9 Experiment Phase Transitions + APTE",
        replace_existing=True,
    )

    # Webhook renewal: 1st of each month at 02:00 UTC (90-day TTL, renew monthly)
    scheduler.add_job(
        webhook_renewal_job,
        trigger=CronTrigger(day=1, hour=2, minute=0),
        id="webhook_renewal",
        name="Oura Webhook Renewal",
        replace_existing=True,
    )

    # Timing refresh: daily at 03:00 UTC (recalculates personalized send times)
    scheduler.add_job(
        timing_refresh_job,
        trigger=CronTrigger(hour=3, minute=0),
        id="timing_refresh",
        name="Personalized Timing Refresh",
        replace_existing=True,
    )

    # Offline eval: Sundays at 05:00 UTC (after correlation engine at 04:00)
    scheduler.add_job(
        offline_eval_job,
        trigger=CronTrigger(day_of_week="sun", hour=5, minute=0),
        id="offline_eval",
        name="Weekly Offline Eval",
        replace_existing=True,
    )

    # Ranker training: Sundays at 06:00 UTC (after offline eval at 05:00)
    scheduler.add_job(
        ranker_training_job,
        trigger=CronTrigger(day_of_week="sun", hour=6, minute=0),
        id="ranker_training",
        name="Phase 7 XGBoost Ranker Training + CoreML Export",
        replace_existing=True,
    )

    # Synth drift monitor: daily at 04:15 UTC. Sits after feature refresh
    # (03:30), baselines (03:45), and the Sunday correlation_engine slot
    # at 04:00; well before offline eval at 05:00. Short-circuits cleanly
    # when no synth has been generated. Phase 4.5 Commit 5.
    scheduler.add_job(
        synth_drift_job,
        trigger=CronTrigger(hour=4, minute=15),
        id="synth_drift",
        name="Phase 4.5 Synth vs Real Biometric Drift",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Notification scheduler started with %d jobs", len(scheduler.get_jobs()))


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Notification scheduler stopped")
