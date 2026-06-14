import os
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings

# Manually load .env since pydantic-settings has issues on Python 3.14
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite+aiosqlite:///./meld.db"

    # Anthropic
    anthropic_api_key: str = ""
    # Model IDs centralized so a model deprecation is one env-var override
    # rather than a grep-and-replace across services. Anthropic deprecates
    # 12-month-old SKUs; the next rotation should be a single Railway change.
    anthropic_model_haiku: str = "claude-haiku-4-5-20251001"
    anthropic_model_sonnet: str = "claude-sonnet-4-20250514"
    anthropic_model_opus: str = "claude-opus-4-20250514"

    # Oura
    oura_client_id: str = ""
    oura_client_secret: str = ""
    oura_redirect_uri: str = "http://localhost:8000/auth/oura/callback"
    # Token Oura sends back during the GET handshake to verify the registrar
    # owns our endpoint. Generate per-env via `openssl rand -hex 24`.
    # Empty = handshake rejected (default-deny in dev). 2026-04-29 audit found
    # a hardcoded value in source — that string is now revoked.
    oura_webhook_verification_token: str = ""

    # App
    app_env: str = "development"
    app_secret_key: str = "change-me"

    # USDA FoodData Central
    usda_api_key: str = "DEMO_KEY"

    # APNs (Push Notifications)
    # Production: set APNS_KEY_CONTENT to the raw .p8 contents via Railway env var.
    # Local dev: set APNS_KEY_PATH to the on-disk .p8 file.
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_key_path: str = ""  # Local dev: path to .p8 file on disk
    apns_key_content: str = ""  # Production: raw .p8 contents injected via env var
    apns_bundle_id: str = "com.heymeld.app"
    apns_environment: str = "sandbox"  # "sandbox" or "production"

    # Encryption — Fernet symmetric key for OAuth tokens at rest.
    # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    encryption_key: str = ""

    # User timezone for scheduled notifications (morning brief, bedtime, etc.).
    # Single-user MVP: one timezone for the whole instance. Multi-user TODO:
    # store per user in the User model and look up per job.
    user_timezone: str = "America/New_York"

    # P3-3: Public URLs for the backend, surfaced as config so we can change
    # the production hostname (or run staging) without grepping for the
    # Railway slug. Default is empty; required in any non-development APP_ENV
    # so a misconfigured deploy fails loud instead of silently using the
    # wrong host.
    public_base_url: str = ""
    local_base_url: str = "http://localhost:8000"

    # Auth — Sign in with Apple + backend JWTs
    jwt_secret_key: str = ""  # HS256 secret for our own access tokens; from env
    apple_team_id: str = ""
    apple_bundle_id: str = "com.heymeld.app"
    # Sign in with Apple key (separate from APNs). Used to sign the ES256 client
    # secret for calls to Apple's /auth/revoke endpoint. Same file/content pattern
    # as APNs: SIWA_KEY_PATH for local dev, SIWA_KEY_CONTENT for production.
    siwa_key_id: str = ""
    siwa_key_path: str = ""
    siwa_key_content: str = ""

    # Sentry — error tracking + AI debugging (Seer). Empty DSN disables.
    sentry_dsn: str = ""

    # Admin Apple user IDs allowed to call /api/insights/admin/rollback and
    # any other endpoints gated by `CurrentAdminUser`. Comma-separated list.
    # Empty in dev (admin endpoints return 403). Production: set to Brock's
    # apple_user_id via Railway env. Adding/removing requires a redeploy.
    admin_user_ids: str = ""

    # Resend — transactional email for new-signup admin alerts on the waitlist.
    # Empty API key disables sending (alerts are silently skipped + logged).
    # Production: set RESEND_API_KEY via Railway. The sender domain on
    # `resend_from` must be verified in Resend before any send succeeds.
    resend_api_key: str = ""
    resend_admin_to: str = "hello@heymeld.com"
    resend_from: str = "Meld Waitlist <noreply@heymeld.com>"

    @model_validator(mode="after")
    def _require_public_base_url_in_non_dev(self) -> "Settings":
        # APP_ENV is the same env var /ops/status reports as uptime_env.
        # Default "development" means a missing PUBLIC_BASE_URL is fine for
        # local work; any other value (production, staging) must override it.
        app_env = os.environ.get("APP_ENV", "development")
        if app_env != "development" and not self.public_base_url:
            raise ValueError(
                f"PUBLIC_BASE_URL must be set when APP_ENV={app_env!r}. "
                "Set it via Railway env vars before deploy."
            )
        return self


settings = Settings()
