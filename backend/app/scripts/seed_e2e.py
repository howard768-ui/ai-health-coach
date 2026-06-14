"""Seed deterministic fixture data for the Maestro E2E backend flows.

Creates the dev-login user (same fixed apple_user_id the POST /auth/dev-login
route mints tokens for) plus 7 daily SleepRecord rows ending yesterday. That
one dataset satisfies both data-dependent flows:

- /api/dashboard renders the "Sleep Efficiency" and "HRV Status" metric cards
  via the SleepRecord fallback path (flow 13). hrv_average MUST be non-zero
  or the HRV card is omitted and the flow's second assert fails.
- /api/trends returns non-empty metrics for the 7-day window (flow 14).

Idempotent: existing sleep rows for the dev user are deleted before insert,
so CI retries never duplicate. Dates are computed at runtime so the seed
never goes stale.

stdlib + app.* only. Do not import numpy/pandas here (ML-boundary and
cold-boot gates AST-scan everything under app/), and never import this
module from anything on the app boot path.

Usage:
    cd backend
    uv run python -m app.scripts.seed_e2e
"""

import asyncio
from datetime import date, timedelta

from sqlalchemy import delete, select

from app.database import async_session
from app.models.health import SleepRecord
from app.models.user import User
from app.routers.auth_apple import DEV_LOGIN_USER_ID

# Deterministic week of plausible values (no randomness: same CI run, same
# screen). Index 0 is yesterday.
WEEK = [
    # (efficiency %, hrv ms, resting hr, readiness, total sleep s, deep s)
    (91.0, 58.0, 53.0, 82, 27360, 5520),
    (88.0, 52.0, 55.0, 76, 25920, 5100),
    (92.0, 61.0, 52.0, 85, 28200, 5700),
    (85.0, 47.0, 57.0, 71, 24480, 4860),
    (89.0, 55.0, 54.0, 79, 26640, 5340),
    (90.0, 57.0, 53.0, 81, 27000, 5400),
    (87.0, 50.0, 56.0, 74, 25200, 5040),
]


async def main() -> int:
    async with async_session() as db:
        result = await db.execute(
            select(User).where(User.apple_user_id == DEV_LOGIN_USER_ID)
        )
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                apple_user_id=DEV_LOGIN_USER_ID,
                name="Dev",
                is_active=True,
                onboarding_complete=True,
            )
            db.add(user)

        await db.execute(
            delete(SleepRecord).where(SleepRecord.user_id == DEV_LOGIN_USER_ID)
        )

        for i, (eff, hrv, rhr, readiness, total_s, deep_s) in enumerate(WEEK):
            day = (date.today() - timedelta(days=i + 1)).isoformat()
            db.add(
                SleepRecord(
                    user_id=DEV_LOGIN_USER_ID,
                    date=day,
                    efficiency=eff,
                    hrv_average=hrv,
                    resting_hr=rhr,
                    readiness_score=readiness,
                    total_sleep_seconds=total_s,
                    deep_sleep_seconds=deep_s,
                )
            )

        await db.commit()

    print(f"Seeded {len(WEEK)} sleep records for {DEV_LOGIN_USER_ID[:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
