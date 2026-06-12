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

## Why backend/ is quarantined

Simulator builds hard-pin the API base URL to `http://127.0.0.1:8000/api`
(`Meld/Services/APIClient.swift`) and CI starts no backend, so these flows
fail deterministically: `13`/`14` assert metric labels that only render with
data, and the coach/meals flows depend on live responses. They have NEVER
passed in CI (verified back to the first run that executed flows,
2026-04-20). Re-enable them once CI gets a seeded local backend, which needs:

1. A `uvicorn` step in the Maestro job with an ephemeral DB.
2. The `/auth/dev-login` route the app already calls in UI-test mode
   (`APIClient.devLogin()`); it does not exist in `backend/app` today.
3. A seed script inserting a test user plus ~7 days of sleep/HRV records.

Until then a nightly run of `backend/` would be guaranteed red, which is
alarm noise, not signal.

## Running locally

```
maestro test maestro/flows/smoke/            # what CI runs
maestro test maestro/flows/backend/          # needs local backend on :8000
maestro test maestro/flows/smoke/12_onboarding_welcome.yaml   # standalone only
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
