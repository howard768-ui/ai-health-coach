"""Tests for verify_siwa_configured and verify_apns_configured startup checks.

These verifiers must:
  1. Stay non-fatal (warn-and-continue) so a corrupt env var can't gate the deploy.
     Apple .p8 files are downloadable only once at creation; coupling deploy
     success to fixing a corrupt env var would mean a multi-day Apple Developer
     revoke + recreate cycle. See `feedback_railway_ui_strips_newlines.md`.
  2. Distinguish "feature disabled" from "feature partially configured".
     The 2026-04-30 sequel issue (MELD-BACKEND-E) was a false alarm because
     the prior verifier counted shared APPLE_TEAM_ID/APPLE_BUNDLE_ID as SIWA
     indicators when they're really APNs+SIWA shared config.
  3. Log structured errors so misconfig surfaces in Sentry.

Run: cd backend && uv run pytest tests/test_apple_verify.py -v
"""

import logging
import os

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

# Settings need a JWT secret to import without errors (independent of these tests).
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-verify-tests")

from app.config import settings
from app.core.apple import verify_siwa_configured
from app.services.apns import apns_client, verify_apns_configured


@pytest.fixture
def valid_es256_pem() -> str:
    """Generate a valid ES256 (P-256) private key in PEM format.

    Apple uses ES256 for both APNs and SIWA. Generate at runtime so tests
    don't ship a checked-in fixture key — even a synthetic one — that could
    look real to a casual reader.
    """
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


@pytest.fixture(autouse=True)
def reset_apns_singleton_cache():
    """The APNs client caches the parsed PEM across calls. Reset before each
    test so settings monkeypatched in one test don't leak via the cache."""
    apns_client._private_key = None
    yield
    apns_client._private_key = None


# ── verify_siwa_configured ───────────────────────────────────────────────────


def test_siwa_disabled_logs_nothing(monkeypatch, caplog):
    """SIWA-specific vars unset = SIWA disabled. No error logs even when
    shared Apple IDs are set (production state when APNs is configured but
    SIWA is not yet — the case that produced false alarm MELD-BACKEND-E)."""
    monkeypatch.setattr(settings, "siwa_key_content", None)
    monkeypatch.setattr(settings, "siwa_key_path", None)
    monkeypatch.setattr(settings, "siwa_key_id", None)
    monkeypatch.setattr(settings, "apple_team_id", "8HA7S8AFF7")
    monkeypatch.setattr(settings, "apple_bundle_id", "com.heymeld.app")

    with caplog.at_level(logging.ERROR, logger="meld.apple"):
        verify_siwa_configured()

    assert not caplog.records, f"expected no errors, got: {[r.message for r in caplog.records]}"


def test_siwa_partial_only_key_content_logs_error(monkeypatch, caplog, valid_es256_pem):
    """SIWA_KEY_CONTENT set but SIWA_KEY_ID missing: log partial-config error."""
    monkeypatch.setattr(settings, "siwa_key_content", valid_es256_pem)
    monkeypatch.setattr(settings, "siwa_key_path", None)
    monkeypatch.setattr(settings, "siwa_key_id", None)
    monkeypatch.setattr(settings, "apple_team_id", "8HA7S8AFF7")
    monkeypatch.setattr(settings, "apple_bundle_id", "com.heymeld.app")

    with caplog.at_level(logging.ERROR, logger="meld.apple"):
        verify_siwa_configured()

    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "SIWA partially configured" in msg
    assert "SIWA_KEY_ID" in msg


def test_siwa_partial_only_key_id_logs_error(monkeypatch, caplog):
    """SIWA_KEY_ID set but SIWA_KEY_CONTENT/SIWA_KEY_PATH missing: log error."""
    monkeypatch.setattr(settings, "siwa_key_content", None)
    monkeypatch.setattr(settings, "siwa_key_path", None)
    monkeypatch.setattr(settings, "siwa_key_id", "ABC1234567")
    monkeypatch.setattr(settings, "apple_team_id", "8HA7S8AFF7")
    monkeypatch.setattr(settings, "apple_bundle_id", "com.heymeld.app")

    with caplog.at_level(logging.ERROR, logger="meld.apple"):
        verify_siwa_configured()

    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "SIWA partially configured" in msg
    assert "SIWA_KEY_CONTENT" in msg


def test_siwa_missing_apple_team_id_logs_error(monkeypatch, caplog, valid_es256_pem):
    """Both SIWA-specific set but APPLE_TEAM_ID missing: log shared-config error."""
    monkeypatch.setattr(settings, "siwa_key_content", valid_es256_pem)
    monkeypatch.setattr(settings, "siwa_key_path", None)
    monkeypatch.setattr(settings, "siwa_key_id", "ABC1234567")
    monkeypatch.setattr(settings, "apple_team_id", None)
    monkeypatch.setattr(settings, "apple_bundle_id", "com.heymeld.app")

    with caplog.at_level(logging.ERROR, logger="meld.apple"):
        verify_siwa_configured()

    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "APPLE_TEAM_ID" in msg


def test_siwa_invalid_pem_logs_error(monkeypatch, caplog):
    """All four set but PEM malformed: log parse error, do not raise."""
    bogus_pem = "-----BEGIN PRIVATE KEY-----\nNOTbase64==\n-----END PRIVATE KEY-----\n"
    monkeypatch.setattr(settings, "siwa_key_content", bogus_pem)
    monkeypatch.setattr(settings, "siwa_key_path", None)
    monkeypatch.setattr(settings, "siwa_key_id", "ABC1234567")
    monkeypatch.setattr(settings, "apple_team_id", "8HA7S8AFF7")
    monkeypatch.setattr(settings, "apple_bundle_id", "com.heymeld.app")

    with caplog.at_level(logging.ERROR, logger="meld.apple"):
        verify_siwa_configured()  # must not raise

    assert len(caplog.records) == 1
    assert "SIWA key parse FAILED" in caplog.records[0].message


def test_siwa_fully_configured_logs_nothing(monkeypatch, caplog, valid_es256_pem):
    """All four set + valid PEM: no error logs (the happy path)."""
    monkeypatch.setattr(settings, "siwa_key_content", valid_es256_pem)
    monkeypatch.setattr(settings, "siwa_key_path", None)
    monkeypatch.setattr(settings, "siwa_key_id", "ABC1234567")
    monkeypatch.setattr(settings, "apple_team_id", "8HA7S8AFF7")
    monkeypatch.setattr(settings, "apple_bundle_id", "com.heymeld.app")

    with caplog.at_level(logging.ERROR, logger="meld.apple"):
        verify_siwa_configured()

    assert not caplog.records, f"expected no errors, got: {[r.message for r in caplog.records]}"


# ── verify_apns_configured ───────────────────────────────────────────────────


def test_apns_disabled_no_error(monkeypatch, caplog):
    """APNs unset: skip-with-info, no error logs."""
    monkeypatch.setattr(settings, "apns_key_content", None)
    monkeypatch.setattr(settings, "apns_key_path", None)

    with caplog.at_level(logging.ERROR, logger="meld.apns"):
        verify_apns_configured()

    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert not error_records, f"expected no errors, got: {[r.message for r in error_records]}"


def test_apns_invalid_pem_logs_error_does_not_raise(monkeypatch, caplog):
    """Configured but malformed PEM: log error, do not raise (PR #86 stance)."""
    bogus_pem = "-----BEGIN PRIVATE KEY-----\nNOTbase64==\n-----END PRIVATE KEY-----\n"
    monkeypatch.setattr(settings, "apns_key_content", bogus_pem)
    monkeypatch.setattr(settings, "apns_key_path", None)

    with caplog.at_level(logging.ERROR, logger="meld.apns"):
        verify_apns_configured()  # must not raise

    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_records) >= 1
    assert "APNs key parse FAILED" in error_records[0].message


def test_apns_valid_pem_no_error(monkeypatch, caplog, valid_es256_pem):
    """Configured + valid PEM: no error logs (the happy path)."""
    monkeypatch.setattr(settings, "apns_key_content", valid_es256_pem)
    monkeypatch.setattr(settings, "apns_key_path", None)

    with caplog.at_level(logging.ERROR, logger="meld.apns"):
        verify_apns_configured()

    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert not error_records, f"expected no errors, got: {[r.message for r in error_records]}"


def test_apns_missing_key_file_logs_error_does_not_raise(monkeypatch, caplog):
    """Configured key PATH pointing at a missing file: log error, do not
    raise. Regression for the 2026-06-12 finding: _load_private_key ran
    OUTSIDE the try in verify_apns_configured, so any fresh checkout or CI
    runner without the .p8 crashed the whole app at startup, violating the
    function's own warn-and-continue contract."""
    from app.services.apns import apns_client

    apns_client._reset_cache()  # a prior test may have cached a valid key
    monkeypatch.setattr(settings, "apns_key_content", None)
    monkeypatch.setattr(settings, "apns_key_path", "/nonexistent/AuthKey_TEST.p8")

    with caplog.at_level(logging.ERROR, logger="meld.apns"):
        verify_apns_configured()  # must not raise

    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(error_records) >= 1
    assert "APNs key parse FAILED" in error_records[0].message
