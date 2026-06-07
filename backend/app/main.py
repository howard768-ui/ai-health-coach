import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import settings
from app.core.apple import verify_siwa_configured
from app.core.ratelimit import limiter, _real_remote_address  # noqa: F401 (re-exported)
from app.core.secrets import verify_secrets_configured
from app.routers import auth, auth_apple, health, coach, notifications, meals, user, peloton_auth, garmin_auth, webhooks, waitlist, mascot, insights, privacy, experiments, ops, ml_ops
from app.services.apns import verify_apns_configured
from app.tasks.scheduler import start_scheduler, stop_scheduler


_PHI_SCRUB_PATTERNS = [
    # apple_user_id: opaque numeric.numeric.string format. We log it routinely
    # for forensics but it's still user-identifying — scrub before Sentry.
    (re.compile(r"\b\d{6}\.[0-9a-f]{32}\.\d{4}\b"), "[apple_user_id]"),
    # bare email — only common shapes; private-relay addresses caught here too.
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    # bearer token — opaque ~200-char base64-ish JWT string after `Bearer`.
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"), "Bearer [token]"),
]


def _scrub_phi(value):
    """Walk arbitrary nested data and apply scrub patterns to every str.

    Sentry's `before_send` receives an event dict with arbitrary nesting
    (extra, contexts, request body, breadcrumbs, exception values).
    Recurse into dicts and lists; replace strings via the patterns.
    """
    if isinstance(value, str):
        out = value
        for pat, repl in _PHI_SCRUB_PATTERNS:
            out = pat.sub(repl, out)
        return out
    if isinstance(value, dict):
        return {k: _scrub_phi(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_phi(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_phi(v) for v in value)
    return value


def _sentry_before_send(event, _hint):
    """Sentry `before_send` callback: scrub PHI/PII before transmission.

    Belt-and-suspenders against `send_default_pii=False`. The default
    setting blocks Sentry from auto-attaching headers/cookies, but our own
    code logs apple_user_ids and tokens routinely (auth flow, scheduler
    job context). Scrub here so a stray log line doesn't leak.
    """
    return _scrub_phi(event)


def _init_sentry():
    """Initialize Sentry if a DSN is configured. No-op otherwise."""
    if not settings.sentry_dsn:
        return
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration

    # Release tag: Railway sets RAILWAY_GIT_COMMIT_SHA per deploy. Sentry
    # uses this to attribute errors to specific releases — without it, every
    # error in the dashboard says "release: unknown" and we can't tell
    # which deploy regressed what. The 2026-04-29 audit (sentry-side) said
    # this was the single missing knob preventing release-correlated
    # alerting from working at all.
    release = os.environ.get("RAILWAY_GIT_COMMIT_SHA") or "dev"

    integrations = [FastApiIntegration()]
    # APScheduler integration: surfaces scheduled job exceptions to Sentry
    # with proper grouping and tags. Sentry adds it to `default_integrations`
    # in v2.x but only if APScheduler is importable at sdk init time, which
    # it always is for us (it's a runtime dep). Listing it explicitly so a
    # future dep change can't silently disable cron error reporting.
    try:
        from sentry_sdk.integrations.apscheduler import ApschedulerIntegration
        integrations.append(ApschedulerIntegration())
    except ImportError:
        # Older sentry-sdk without the integration — silently skip.
        pass

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        release=release,
        traces_sample_rate=0.1,  # 10% of requests traced (cost control)
        profiles_sample_rate=0.1,
        environment=settings.app_env,
        send_default_pii=False,  # Never send PHI/PII to Sentry
        # The before_send scrubber only matches a few structured patterns and
        # cannot catch free-form chat/meal/health text. Do not capture request
        # bodies or frame-local variables at all: on an exception in coach/meal
        # handlers those would otherwise ship raw PHI to Sentry. Audit A4.
        max_request_body_size="never",
        include_local_variables=False,
        before_send=_sentry_before_send,
        integrations=integrations,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # CRITICAL: invoked by FastAPI ASGI machinery, not by any direct caller.
    # Static call graphs do not model lifespan invocation as a CALLS edge,
    # so impact tools will report this as orphaned. It is the boot path:
    # any change here gates every deploy.
    # Startup: Sentry, secret validation, PEM verifiers, scheduler.
    # Migrations are NOT run here — Railway's preDeployCommand owns
    # `alembic upgrade head` (see backend/railway.toml). See
    # ~/.claude/projects/.../memory/feedback_alembic_in_lifespan.md for
    # why: the 5-minute Railway healthcheck window is hostile to migrations,
    # a hung migration here brings the whole app down, and a failed
    # preDeployCommand keeps the previous SUCCESS deploy serving instead.
    _init_sentry()
    # Production-grade secret validation. Fails fast (raises) in production
    # when JWT_SECRET_KEY or ENCRYPTION_KEY are missing — those have
    # seconds-to-recover profiles so coupling deploy success to them is
    # correct. Non-critical secrets (Anthropic API key, app_secret_key,
    # apns_environment) log + continue. See app/core/secrets.py.
    verify_secrets_configured()
    # PEM startup validation — APNs and SIWA private keys parse correctly,
    # so a corrupt/CRLF-mangled env value surfaces here instead of
    # surfacing as a JWSError on the next scheduled morning brief (see
    # scheduler_audit.md, the 6 recurring Sentry JWSError issues from
    # 2026-04-29). Warn-and-continue (PR #86 stance: Apple .p8 has
    # multi-day recovery, can't gate the deploy queue). No-op in
    # environments without the keys configured.
    verify_apns_configured()
    verify_siwa_configured()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(
    title="Meld Health Coach API",
    description="Backend for Meld — AI-powered health coaching",
    version="0.1.0",
    lifespan=lifespan,
)

# Wire up rate limiter — slowapi attaches `request.app.state.limiter` and
# auto-returns 429 with Retry-After when limits are exceeded.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Register the middleware so default_limits apply to EVERY endpoint, not just
# those with an explicit @limiter.limit decorator. Without this the global
# limit was dead config and undecorated endpoints (incl. Claude-hitting ones)
# were unlimited (audit C2). No-op when the limiter is disabled (dev/test).
app.add_middleware(SlowAPIMiddleware)

# CORS — restrict to known origins. P3-3: read public URL from settings
# so changing the deploy URL is one config change, not a grep job.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.local_base_url,
        settings.public_base_url,
        # Marketing site (heymeld.com); the Astro app posts to /api/waitlist/subscribe
        "https://heymeld.com",
        "https://www.heymeld.com",
        # Cloudflare Pages preview deployments
        "https://heymeld.pages.dev",
        # Local Astro dev server
        "http://localhost:4321",
        "http://localhost:4322",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve notification media (recovery badges, etc.)
media_dir = Path(__file__).resolve().parent.parent / "media"
media_dir.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")

# Mount routers
app.include_router(auth.router)          # Legacy Oura OAuth callback
app.include_router(auth_apple.router)    # Sign in with Apple + JWT refresh
app.include_router(health.router)
app.include_router(coach.router)
app.include_router(notifications.router)
app.include_router(meals.router)
app.include_router(user.router)
app.include_router(peloton_auth.router)
app.include_router(garmin_auth.router)
app.include_router(webhooks.router)
app.include_router(waitlist.router)    # Public waitlist signup (heymeld.com)
app.include_router(mascot.router)      # Mascot accessory wardrobe
app.include_router(insights.router)    # Signal Engine Phase 4: daily ranked insight + feedback
app.include_router(privacy.router)     # Phase 8: cohort opt-in/opt-out + deletion
app.include_router(experiments.router) # Phase 9: n-of-1 experiment CRUD + APTE results
app.include_router(ops.router)         # Ops status for autonomous monitoring loops
app.include_router(ml_ops.router)      # Read-only ML ops endpoints for Phase 5 monitors


@app.get("/")
async def health_check():
    """Public root endpoint — no DB ping. Marketing/curiosity check only."""
    return {
        "status": "healthy",
        "app": "Meld Health Coach API",
        "version": "0.1.0",
        "environment": settings.app_env,
    }


@app.get("/healthz")
async def healthz():
    """Liveness probe — process is up. No DB check (cheap, fast)."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """Readiness probe — verifies DB *and schema* are healthy.

    Querying alembic_version (rather than SELECT 1) ensures the schema
    chain has run. SELECT 1 succeeds against an empty Postgres, which
    masked total schema loss for 10 days during the 2026-04-29 incident.
    See ~/.claude/projects/.../memory/feedback_db_ok_must_query_real_table.md.
    """
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError, DBAPIError
    from app.database import async_session
    try:
        async with async_session() as db:
            await db.execute(text("SELECT 1 FROM alembic_version LIMIT 1"))
        return {"status": "ready", "db": "ok"}
    except (SQLAlchemyError, DBAPIError, OSError) as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Database not ready: {e}")
