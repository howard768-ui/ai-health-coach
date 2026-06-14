# CLAUDE.md

You are working in the Meld repo. This file orients you to the conventions, the autonomous engineering system running alongside you, and the safety rails you must not violate. Read it once at session start.

## Who Brock is

Brock is the solo founder of Meld (iOS health coach app, FastAPI backend, ML signal engine, Astro marketing site). Former VP, Product at Lark. He is trained in product, not engineering. He does not click `git merge`. Treat him as a sharp product partner, not an engineer. Match that register: no hand-holding on CS fundamentals, do explain product risk before taking actions that could change user experience.

## HITL tier model (in force)

Full plan: `~/.claude/plans/hitl-tier-model.md`. Summary:

| Tier | Who decides | Who merges | Trigger |
| --- | --- | --- | --- |
| 0 Engineering | Classifier + Claude review agent | Classifier auto-merges on green CI | Code, infra, CI, tests, deps, refactors, internal bug fixes, ML behind shadow flags, docs |
| 2 Product | Brock reviews + comments approval | Claude (you) executes merge + verifies deploy | Coach output, user-facing copy, design token VALUES, UX structure, marketing copy, dependabot majors |
| 3 Safety | Refuse by default; Brock may override | Claude executes on Brock's explicit override | Safety gates, coach-prompt, encryption, PHI migrations, shadow flag FLIPS, golden thresholds, HealthKit scope, credentials |

**You never ask Brock to click merge.** He reviews, says "ship it" or equivalent, and you execute the merge via `gh pr merge --merge`. See `~/.claude/projects/-Users-brockhoward-ai-health-coach/memory/feedback_claude_does_merges.md`.

Kill switch: repo variable `AUTO_MERGE_ENABLED` (currently `true`). Flip to `false` to pause all auto-merges instantly.

## Tier 3 hard refuse (do not open PRs touching these)

- `backend/app/services/safety*`, crisis escalation
- Files containing the evidence-bound coach-prompt constant (see `.github/hitl-config.json` for the exact needle strings)
- `backend/app/core/encryption.py`, `EncryptedString` columns
- Alembic migrations touching PHI tables: `health_metric_records`, `sleep_records`, `chat_messages`, `meals`, `oura_*`, `garmin_*`, `peloton_*`, `user_correlations`, `ml_features`, `ml_baselines`, `ml_insights`
- Shadow flag FLIPS for any ML gate (adding new flag definitions is Tier 0 fine)
- `backend/tests/ml/test_*golden*.py` golden-data thresholds
- `Meld/Services/Sync/HealthKitManager*` scope requests
- `.env*`, `**/*.p8`, `**/*.pem`
- Meta-guard: `.github/workflows/code-review.yml`, `auto-fix*.yml`, `hitl-classifier.yml`, `migration-health.yml`, `ios-structural-checks.yml`, `dependabot-automerge.yml`, `.github/hitl-config.json`

Path-based refuse is enforced by `.github/hitl-config.json` + `.github/workflows/hitl-classifier.yml`. Content-based refuse (PRs adding lines that contain certain coach-prompt or shadow-flag identifiers) is also in the classifier; workflow YAML and hitl-config paths are excluded from that scan to avoid self-trip.

## What's running autonomously

**40+ scheduled tasks** (stored at `~/.claude/scheduled-tasks/`). See `~/Documents/Obsidian Vault/HealthCoach/Autonomous Ops.md` for the full inventory grouped by phase. Highlights:

- Production health: `meld-backend-health` every 4h, `meld-post-deploy-verify` every 30 min, `meld-pipeline-health` daily
- PR lifecycle: `meld-autofix-poller` + `meld-autofix-impl-poller` every 15 min (both gated on `ENABLE_AUTO_FIX_AGENT` repo variable)
- Morning digest: `meld-hitl-daily` weekdays 08:51 local, single Linear issue summarizing overnight auto-merges + open Tier 2 PRs
- Weekly: product-digest Mon, sprint-status Fri, eval-iteration Wed, signal-quality Mon, ...
- Monthly 1st: roadmap-sync, cycle-planning, shadow-flag-report, model-registry, secret-rotation-reminder, infra-audit

**6 GitHub Actions workflows**:
- `backend.yml`, `eval.yml`, `ios.yml`, `website.yml`: existing CI
- `code-review.yml`: Claude PR review agent on every PR
- `hitl-classifier.yml`: Tier 0/2/3 routing + Tier 0 auto-merge
- `dependabot-automerge.yml`: patch + minor auto-merge
- `auto-fix-dryrun.yml`: Linear issue → proposal comment
- `auto-fix-impl.yml`: approved proposal → draft PR with code
- `migration-health.yml`: Alembic pattern checks
- `ios-structural-checks.yml`: DS token file presence + DS component snapshot baseline coverage

**7 ops endpoints** on the FastAPI backend at `https://zippy-forgiveness-production-0704.up.railway.app`:
- `/ops/status`: scheduler + pipeline freshness + deploy SHA + sentry_enabled
- `/ops/ml/signal-quality`, `/data-quality`, `/feature-drift`, `/experiments`, `/retrain-readiness`, `/model-registry`

## Conventions you must follow

1. **No em dashes**. Anywhere. Not in code, comments, commit messages, PR bodies, Linear issues, chat. Use commas, colons, parens, or sentence breaks. See `~/.claude/projects/-Users-brockhoward-ai-health-coach/memory/feedback_no_em_dashes.md`.
2. **5-gate CI before every commit**. pytest, cold-boot (under 4s), ML boundary, alembic round-trip, migration patterns. See `feedback_testing_rigor.md`.
3. **Update Obsidian wiki inline** as work lands, not at end of session. See `feedback_obsidian_and_qa_sync.md`.
4. **Eyes and ears**: at session start, audit for uncommitted changes, stale worktrees, CI red on main, stale PRs, production drift. Surface anomalies proactively. See `feedback_eyes_and_ears.md`.
5. **ML boundary**: `backend/app/` may only import `backend.ml.api` or `ml.api`. No deeper ML imports.
6. **Cold-boot budget**: `from app.main import app` must stay under 4s. No top-level imports of pandas, scipy, statsmodels, xgboost, prophet, dowhy, econml, coremltools, mlflow, evidently, nannyml, shap, hdbscan, ruptures, pycatch22, or sklearn in `backend/app/`.
7. **Check-in cadence on background tasks**: never assume a task will notify you. Poll at 10 min mark on any polling task. See `feedback_eyes_and_ears.md`.
8. **Verify actual filenames** via `gh api /repos/.../contents/<dir>` before shipping policy files that reference specific paths. Caught once: DS tokens are `DSColor.swift` not `Colors.swift`.
9. **CI checks are honest or they don't exist.** A check that cannot go red (swallowed exit codes, `|| true`, `| tee` without pipefail, misnamed no-op steps, green-on-missing-artifact) is worse than no check: it reads as verified when nothing was. Never add a suppressor to a test or lint step without a written justification beside it, and never name a step after a check it does not perform. Context (2026-06-12 sweep): Maestro was green with zero flows executed; SwiftLint hid 411 violations; ruff hid 126.
10. **Two identical failures = diagnose, not rerun.** After a CI check fails twice the same way, read the logs and artifacts before any third attempt. Reruns are for nondeterministic flakes only (different step or error each time). The Maestro 40-minute timeout was rerun twice (90 wasted macOS-runner minutes) when 20 minutes of log forensics proved it could never pass.
11. **macOS runner minutes are the scarce CI resource.** The account cap is 5 concurrent macOS jobs and queueing has been observed from 1h15m to 18.5h; that queue is why green iOS PRs outrun the classifier's 30-minute poll. Before adding or extending a `macos-*` job: budget it, reuse build artifacts instead of rebuilding, and put anything that does not need Xcode or a simulator on ubuntu.

## Key files to know

- `~/.claude/plans/groovy-sprouting-valley.md`: the 12-loop autonomous ops plan
- `~/.claude/plans/hitl-tier-model.md`: tier classification rules
- `~/.claude/plans/autofix-agent-ultraplan.md`: 3-layer auto-fix rollout (PR 1 dry-run + PR 2 code-writing shipped, PR 3 metrics pending)
- `~/.claude/projects/-Users-brockhoward-ai-health-coach/memory/`: all feedback / project / reference memory files
- `~/Documents/Obsidian Vault/HealthCoach/`: project wiki, Progress Log, screen specs, Autonomous Ops page, Shape Up

## How Brock talks to you

He is fast, direct, product-minded. When he says "ship it" or "LGTM" on a PR, execute the merge. When he pastes a screenshot of a problem, investigate before proposing. When he asks "is there anything left", give a crisp tier-by-tier answer, not a listicle. When he grants aggressive authority (he has), use it but explain meaningful product risk before acting.

He does NOT want:
- Em dashes
- AI-writing tells ("I'm excited to...", "I'd be happy to...")
- Hand-holding
- Asking him to do engineering tasks

He DOES want:
- Product-risk flagging before you ship user-visible changes
- Proactive eyes-and-ears reports when you notice state drift
- Brevity over throat-clearing
- Honest reports when you break something, including what you did wrong

## When in doubt

Read memory files. They encode specific past lessons. If you're about to do something that might violate a tier, check `.github/hitl-config.json` and the memory files first. If genuinely uncertain, open a draft PR + Linear issue + tag Brock, don't merge.
