"""Tests for the OAuth-token-at-rest encryption layer.

Run: cd backend && uv run python -m pytest tests/test_encryption.py -v
"""

import os

# A valid Fernet key (32 url-safe base64 bytes). Set before the module under
# test is imported so the Settings object picks it up.
_PRIMARY_KEY = "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
_RETIRED_KEY = "mM8gYBhG93eW-y_n3qKh_yjSbFA9xJKZlKCt3UXCH6s="
os.environ["ENCRYPTION_KEY"] = _PRIMARY_KEY

import pytest  # noqa: E402 (env var above must be set before app.config loads)
from cryptography.fernet import Fernet  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.encryption import (  # noqa: E402
    EncryptionConfigError,
    _get_cipher,
    _reset_for_tests,
    decrypt,
    encrypt,
)


@pytest.fixture(autouse=True)
def _reset_cipher_state():
    """Every test starts with a clean cipher cache and restored settings."""
    original_key = settings.encryption_key
    original_env = settings.app_env
    _reset_for_tests()
    yield
    settings.encryption_key = original_key
    settings.app_env = original_env
    _reset_for_tests()


def test_roundtrip():
    plaintext = "fake-oura-access-token-abc123-with-special!@#chars"
    ct = encrypt(plaintext)
    assert ct is not None
    assert ct != plaintext
    assert decrypt(ct) == plaintext


def test_none_passthrough():
    assert encrypt(None) is None
    assert decrypt(None) is None


def test_empty_string_roundtrip():
    ct = encrypt("")
    assert decrypt(ct) == ""


def test_long_token_roundtrip():
    """Garmin garth sessions can be 1000+ chars."""
    long_plaintext = "x" * 5000
    ct = encrypt(long_plaintext)
    assert decrypt(ct) == long_plaintext


def test_legacy_plaintext_passthrough():
    """Reading a legacy non-Fernet value should not crash; return it as-is."""
    legacy = "old-plaintext-token-from-before-encryption"
    assert decrypt(legacy) == legacy


def test_decrypt_miss_is_logged(caplog):
    """A decrypt miss must surface in logs so legacy-vs-tampering is visible."""
    import logging

    caplog.set_level(logging.WARNING, logger="meld.encryption")
    decrypt("not-a-valid-fernet-token-at-all")
    assert any("decrypt_miss" in r.message for r in caplog.records), (
        "decrypt miss should emit a warning record for monitoring"
    )


def test_each_encryption_unique():
    """Fernet uses random IVs, so same plaintext yields different ciphertexts."""
    plaintext = "same-input"
    ct1 = encrypt(plaintext)
    ct2 = encrypt(plaintext)
    assert ct1 != ct2
    assert decrypt(ct1) == plaintext
    assert decrypt(ct2) == plaintext


def test_cipher_loads_from_env():
    cipher = _get_cipher()
    assert cipher is not None


def test_multi_key_decrypts_retired_key():
    """After rotation, data written with the old key must still decrypt.

    Config order is ``active,retired[,retired...]``. Simulate: write with the
    retired key directly, then configure the module with active first and
    retired second. Decrypt should succeed.
    """
    plaintext = "token-written-before-rotation"
    retired_ct = Fernet(_RETIRED_KEY.encode()).encrypt(plaintext.encode()).decode()

    settings.encryption_key = f"{_PRIMARY_KEY},{_RETIRED_KEY}"
    _reset_for_tests()

    assert decrypt(retired_ct) == plaintext


def test_multi_key_writes_use_active_key():
    """New writes must use the first-listed key so rotation actually drains."""
    settings.encryption_key = f"{_PRIMARY_KEY},{_RETIRED_KEY}"
    _reset_for_tests()

    ct = encrypt("fresh-token")

    # The primary key must be able to decrypt the new ciphertext directly.
    primary = Fernet(_PRIMARY_KEY.encode())
    assert primary.decrypt(ct.encode()).decode() == "fresh-token"


def test_missing_key_falls_through(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="meld.encryption")
    settings.encryption_key = ""
    _reset_for_tests()

    # Should not raise and should write plaintext unchanged.
    assert encrypt("plain") == "plain"
    assert decrypt("plain") == "plain"
    assert any("ENCRYPTION_KEY not set" in r.message for r in caplog.records), (
        "missing-key path should warn"
    )


def test_missing_key_warns_only_once(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="meld.encryption")
    settings.encryption_key = ""
    _reset_for_tests()

    encrypt("a")
    encrypt("b")
    encrypt("c")

    warn_records = [r for r in caplog.records if "ENCRYPTION_KEY not set" in r.message]
    assert len(warn_records) == 1, "warning should dedupe, not spam per call"


# ── fail closed in production (audit H3) ──────────────────────────────


def test_prod_missing_key_raises():
    """Production must never silently store PHI in plaintext: a missing key
    raises instead of falling through."""
    settings.app_env = "production"
    settings.encryption_key = ""
    _reset_for_tests()
    with pytest.raises(EncryptionConfigError):
        encrypt("phi-value")


def test_prod_malformed_key_raises():
    """A non-empty but unparseable key (passes the non-whitespace startup
    check, fails Fernet) must also fail closed in production, not plaintext."""
    settings.app_env = "production"
    settings.encryption_key = "this-is-not-a-valid-fernet-key"
    _reset_for_tests()
    with pytest.raises(EncryptionConfigError):
        decrypt("anything")


def test_dev_missing_key_still_falls_back_to_plaintext():
    """Non-production keeps the legacy plaintext fallback so local dev with no
    secrets works."""
    settings.app_env = "development"
    settings.encryption_key = ""
    _reset_for_tests()
    assert encrypt("phi-value") == "phi-value"
    assert decrypt("phi-value") == "phi-value"
