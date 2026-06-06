"""Production-grade configuration validation at app startup.

Different from `verify_apns_configured` and `verify_siwa_configured`
(which are warn-and-continue because Apple .p8 files have multi-day
recovery profiles), this module fail-closes on missing or sentinel
secrets in production environments. The recovery profile for these
secrets is seconds (set the env var, redeploy), so coupling deploy
success to "secret is set" is correct: the app should refuse to serve
rather than serve insecurely.

Secrets validated:
  - `jwt_secret_key`: HS256 key for our own access tokens. Empty in
    production = trivially-forgeable JWTs.
  - `encryption_key`: Fernet key for OAuth tokens + PHI columns. Empty
    in production = the EncryptedString columns silently fail at write
    time and existing data becomes unreadable on rotate.
  - `anthropic_api_key`: chat won't work without it. Less critical
    (graceful 500 from Anthropic vs broken auth), so log + continue.
  - `app_secret_key`: literal "change-me" sentinel ships if unset.
    Currently unused but log + continue if seen.
  - `apns_environment`: must be "sandbox" or "production"; a typo
    silently sends pushes to the wrong APNs cluster. Log + continue.

In development, all checks log info but never raise — local dev with
a SQLite DB and no real secrets is the expected state.

This complements the PEM verifiers; lifespan calls this first so a
missing JWT secret fails the deploy before the .p8 verifier even runs.
"""

from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger("meld.secrets")

_VALID_APNS_ENVIRONMENTS = ("sandbox", "production")
_APP_SECRET_SENTINEL = "change-me"


class SecretConfigError(RuntimeError):
    """Raised when production starts with a missing or sentinel secret.

    Caller (lifespan startup) bubbles it up so the deploy fails healthcheck
    and Railway rolls back to the previous SUCCESS deploy. The user sees no
    behavior change; Brock sees the failure in Railway logs and Sentry.
    """


def verify_secrets_configured() -> None:
    """Run configuration validation at startup.

    Production environments fail-fast on critical secrets; non-production
    environments log info and continue. Designed to be called from the
    FastAPI lifespan before any other startup work, so a misconfigured
    deploy never reaches the scheduler/router init.

    CRITICAL: invoked from `lifespan()` in app/main.py. Static analyzers
    do not model FastAPI lifespan as a CALLS edge, so impact tools that
    only look at the call graph will report this as orphaned. It is not.
    """
    # Fail closed for every non-development environment, not just the exact
    # string "production" (staging/preview hold real PHI; a misspelled APP_ENV
    # must not skip the gate). Mirrors the same rule in core/encryption.py.
    is_prod = settings.app_env.strip().lower() not in ("development", "test")

    critical_missing: list[str] = []
    if not _is_set(settings.jwt_secret_key):
        critical_missing.append("JWT_SECRET_KEY")
    if not _is_set(settings.encryption_key):
        critical_missing.append("ENCRYPTION_KEY")

    if critical_missing and is_prod:
        raise SecretConfigError(
            f"Refusing to start in production with missing secrets: "
            f"{', '.join(critical_missing)}. Set these in Railway env and redeploy."
        )

    if critical_missing:
        # Dev: fine for local iteration, just note it.
        # Logs the list of unset env-var NAMES (e.g. "JWT_SECRET_KEY"),
        # never any secret value.
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
        logger.info(
            "Critical secrets unset (dev mode, OK for local): %s",
            ", ".join(critical_missing),
        )

    # Below: not severe enough to block deploy; surface in Sentry instead.
    if not _is_set(settings.anthropic_api_key):
        msg = (
            "ANTHROPIC_API_KEY is unset; coach + chat endpoints will return "
            "500 on every call. Set in Railway env to enable."
        )
        if is_prod:
            logger.error(msg)
        else:
            logger.info(msg)

    if settings.app_secret_key == _APP_SECRET_SENTINEL:
        msg = (
            "APP_SECRET_KEY is the literal 'change-me' sentinel. Currently "
            "unused but a footgun if a future feature relies on it."
        )
        if is_prod:
            logger.error(msg)
        else:
            logger.info(msg)

    if settings.apns_environment not in _VALID_APNS_ENVIRONMENTS:
        # Typo means push goes to the wrong APNs cluster (e.g. "prod" vs
        # "production"). Don't crash startup — push is already non-fatal —
        # but log loudly.
        logger.error(
            "APNS_ENVIRONMENT='%s' is invalid. Must be one of %s. "
            "Defaulting behavior is APNs sandbox; production pushes will silently fail.",
            settings.apns_environment,
            list(_VALID_APNS_ENVIRONMENTS),
        )

    # Logs env type + presence booleans only, never any secret value.
    # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
    logger.info(
        "Secrets validation complete (env=%s, jwt=%s, encryption=%s, "
        "anthropic=%s, apns_env=%s)",
        settings.app_env,
        "set" if _is_set(settings.jwt_secret_key) else "UNSET",
        "set" if _is_set(settings.encryption_key) else "UNSET",
        "set" if _is_set(settings.anthropic_api_key) else "UNSET",
        settings.apns_environment,
    )


def _is_set(value: str | None) -> bool:
    """A secret counts as set only if it has non-whitespace content.

    Whitespace-only values are a common Railway-UI footgun (the same
    family as the newline-stripping issue from PR #86), so treat them
    the same as empty.
    """
    return bool(value and value.strip())
