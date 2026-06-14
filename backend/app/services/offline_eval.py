"""
Phase 4: Weekly offline evaluation of production coach responses.

Runs as a scheduled job: pulls recent coach responses from DB,
evaluates reading level and data grounding, flags regressions.

Entry point: `run_offline_eval` is registered as the
`offline_eval_job` APScheduler task in `app/tasks/scheduler.py` and
runs weekly. Static call-graph analyzers will report this module as
orphan because the registration is via a decorator, not a direct call.

Results are logged and stored for the analytics endpoint.
"""

import json
import logging
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# NOTE: `textstat` is imported LAZILY inside ``run_offline_eval`` below.
# When scipy and scikit-learn are present in the environment (they now are, as
# part of the Signal Engine ML stack), nltk eagerly imports them at load time
# via textstat's dependency chain. That cascade was adding ~2.5s to FastAPI
# cold boot and blowing the 4s Railway budget. The cold-boot test in
# backend/tests/ml/test_cold_boot.py guards this invariant — if you move the
# import back to module level, that test will fail.

from app.models.chat import ChatMessageRecord
from app.core.time import utcnow_naive

logger = logging.getLogger(__name__)


async def run_offline_eval(db: AsyncSession, days: int = 7) -> dict:
    """Evaluate recent coach responses for quality regressions.

    Checks:
    1. Reading level — Flesch-Kincaid grade should be <= 8.0
       (allows headroom for technical health terms; the eval suite uses
       7.0 as the strict bar for golden tests, but production responses
       sometimes hit 7.5 with words like "circadian" or "cardiovascular")
    2. Data grounding — responses with health_context should cite numbers from it
    3. Feedback correlation — thumbs-down responses flagged for review
    """
    import textstat  # lazy — see module header for why

    cutoff = utcnow_naive() - timedelta(days=days)

    result = await db.execute(
        select(ChatMessageRecord).where(
            ChatMessageRecord.role == "coach",
            ChatMessageRecord.created_at >= cutoff,
        )
    )
    messages = list(result.scalars().all())

    if not messages:
        logger.info("[OfflineEval] No coach messages in the last %d days", days)
        return {"status": "no_data", "period_days": days}

    reading_level_results = []
    grounding_results = []
    flagged_for_review = []

    for msg in messages:
        # 1. Reading level
        grade = textstat.flesch_kincaid_grade(msg.content)
        reading_level_results.append({
            "message_id": msg.id,
            "grade": round(grade, 1),
            "pass": grade <= 8.0,
        })

        # 2. Data grounding — check if response cites numbers from context
        if msg.health_context:
            try:
                context = json.loads(msg.health_context)
                # Extract numeric values from health context
                numbers_in_context = set()
                for v in context.values():
                    if isinstance(v, (int, float)) and v > 0:
                        numbers_in_context.add(str(int(v)))

                # Check how many context numbers appear in the response
                cited = sum(1 for n in numbers_in_context if n in msg.content)
                total = len(numbers_in_context)
                grounding_rate = cited / max(total, 1)

                grounding_results.append({
                    "message_id": msg.id,
                    "cited": cited,
                    "total": total,
                    "grounding_rate": round(grounding_rate, 2),
                    "pass": grounding_rate >= 0.3,  # At least 30% of numbers cited
                })
            except (json.JSONDecodeError, AttributeError):
                pass

        # 3. Flag thumbs-down responses for human review
        if msg.feedback == "down":
            flagged_for_review.append({
                "message_id": msg.id,
                "content": msg.content[:200],
                "routing_tier": msg.routing_tier,
                "model_used": msg.model_used,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            })

    # Aggregate results
    reading_pass_rate = (
        sum(1 for r in reading_level_results if r["pass"]) / len(reading_level_results)
        if reading_level_results else 0
    )
    avg_reading_grade = (
        sum(r["grade"] for r in reading_level_results) / len(reading_level_results)
        if reading_level_results else 0
    )
    grounding_pass_rate = (
        sum(1 for r in grounding_results if r["pass"]) / len(grounding_results)
        if grounding_results else 0
    )

    report = {
        "status": "complete",
        "period_days": days,
        "total_evaluated": len(messages),
        "reading_level": {
            "avg_grade": round(avg_reading_grade, 1),
            "pass_rate": round(reading_pass_rate * 100, 1),
            "failures": [r for r in reading_level_results if not r["pass"]],
        },
        "data_grounding": {
            "pass_rate": round(grounding_pass_rate * 100, 1),
            "avg_grounding_rate": round(
                sum(r["grounding_rate"] for r in grounding_results) / max(len(grounding_results), 1), 2
            ),
            "failures": [r for r in grounding_results if not r["pass"]],
        },
        "flagged_for_review": flagged_for_review,
        "run_at": utcnow_naive().isoformat(),
    }

    # Log summary
    logger.info(
        "[OfflineEval] %d messages evaluated | Reading: %.0f%% pass (avg grade %.1f) | "
        "Grounding: %.0f%% pass | %d flagged for review",
        len(messages),
        reading_pass_rate * 100,
        avg_reading_grade,
        grounding_pass_rate * 100,
        len(flagged_for_review),
    )

    return report
