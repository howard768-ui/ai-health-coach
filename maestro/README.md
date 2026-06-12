# Maestro E2E flows

Flow suites for driving the built app on an iOS simulator. Diagnosis that led
to this layout: 2026-06-12 Maestro E2E repair (see PR and the vault Progress
Log entry of the same date).

## Layout

```
maestro/
  flows/
    smoke/     UI-local flows. Run on every PR and main push by the
               "Maestro E2E" job in .github/workflows/ios.yml.
    backend/   Flows that assert on live backend data (coach chat replies,
               food search results, dashboard/trends metric values).
               QUARANTINED: not run in CI yet, see below.
  helpers/
    auth_bypass.yaml   Shared launch + auth-skip preamble (15 of 16 flows).
```

## How backend/ runs (un-quarantined 2026-06-12)

Simulator builds hard-pin the API base URL to `http://127.0.0.1:8000/api`
(`Meld/Services/APIClient.swift`). The "Maestro E2E (backend)" job in
ios.yml runs these six flows against a real seeded local backend on main
pushes, the nightly schedule, and manual dispatch (not on PRs: ~25 macOS
runner minutes of live-endpoint testing is the wrong per-push spend). The
backend contract:

1. `uvicorn` on 127.0.0.1:8000 with a file SQLite DB, `alembic upgrade
   head` first (`/readyz` needs the alembic_version table).
2. `POST /auth/dev-login` (development/test only, 404 elsewhere) mints the
   token pair the shipped `APIClient.devLogin()` expects.
3. `app/scripts/seed_e2e.py` seeds 7 days of sleep/HRV records for the
   same fixed user; that one dataset satisfies flows 13 and 14.
4. `ANTHROPIC_API_KEY` must be a NON-EMPTY dummy: empty raises a TypeError
   that escapes the `except anthropic.APIError` clauses and 500s
   `/api/dashboard`. No real key is needed; coach replies degrade to the
   deterministic fallback.

Before these flows existed in CI they had NEVER passed (verified back to
the first run that executed flows, 2026-04-20): the old job had no backend
and `|| true` hid the permanent failures.

## Running locally

```
maestro test maestro/flows/smoke/            # what PR CI runs
maestro test maestro/flows/backend/          # needs seeded backend on :8000
maestro test maestro/flows/smoke/12_onboarding_welcome.yaml   # standalone only
```

Seeded backend for local backend-flow runs:

```
cd backend
export DATABASE_URL=sqlite+aiosqlite:////tmp/meld-e2e.db ANTHROPIC_API_KEY=ci-dummy-key
uv run python -m alembic upgrade head
uv run python -m app.scripts.seed_e2e
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Conventions

- Every `launchApp` MUST carry `permissions: {}`. Maestro's default is to
  set ALL permissions on every launch via applesimutils/simctl, which can
  hang for 5 minutes per subprocess against a wedged simulator TCC daemon
  (observed: an 11m11s single-flow failure on 2026-06-10).
- Keep flows assertion-cheap. Every `tapOn`/`assertVisible` costs one
  accessibility snapshot; on loaded macos-15 runners a snapshot of a
  non-quiescent screen took 30-90s (fixed app-side by disabling
  `repeatForever` animations under UI testing, but the budget discipline
  stays).
- New flows that need backend data go in `backend/`, not `smoke/`.
