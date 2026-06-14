"""Symmetric encryption for OAuth tokens and other secrets at rest.

Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` library: authenticated
encryption that prevents both tampering and snooping. Wrapped in ``MultiFernet``
so keys can be rotated without a single big migration window.

Threat model:
- If someone steals the SQLite file or a Postgres backup, OAuth tokens and
  other encrypted columns are ciphertext, not usable credentials.
- Decryption keys live in Railway env vars, never in the DB or codebase.
- A single compromise (env leak plus DB leak) defeats this. Defense in depth,
  not perfect security.

Config:
    ENCRYPTION_KEY      Comma-separated list of Fernet keys. The FIRST key is
                        the active key used for every new write. Remaining
                        keys are tried on decrypt so ciphertext written with a
                        retired key still reads while we drain the DB.
                        A single-key value (no commas) behaves identically to
                        a single-key Fernet config.
                        Generate one with:
                            python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Missing key behavior: in NON-production environments, when ``ENCRYPTION_KEY``
is unset or unparseable the module logs a warning once and falls through to
plaintext storage (local dev with a SQLite DB and no real secrets is the
expected state). In production this fails CLOSED: a missing, unparseable, or
empty-after-parse key raises ``EncryptionConfigError`` rather than silently
storing PHI in plaintext. The startup gate in ``core/secrets.py`` already
blocks a totally-unset key at deploy time; this closes the malformed-key path
and is defense in depth at every encrypt/decrypt call.

Legacy plaintext rows: a row whose ciphertext cannot be decoded by any
configured key is assumed to be a pre-encryption row and returned unchanged.
That read is logged at warning level so "legacy row" can be distinguished
from "wrong key" or "tampering" in Sentry.

SQLAlchemy integration: ``EncryptedString`` is a TypeDecorator that encrypts
on write and decrypts on read transparently:

    access_token: Mapped[str] = mapped_column(EncryptedString(2000))
"""

from __future__ import annotations

import logging
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from sqlalchemy.types import String, TypeDecorator

from app.config import settings

logger = logging.getLogger("meld.encryption")


class EncryptionConfigError(RuntimeError):
    """Raised in production when the encryption key is missing or unusable.

    Fail closed: rather than silently storing/returning PHI in plaintext (the
    legacy behavior, kept only for non-production dev), production raises so the
    request fails loudly and the misconfiguration is caught immediately.
    """


_cipher: Optional[MultiFernet] = None
_missing_key_warned: bool = False


def _parse_keys(raw: str) -> list[Fernet]:
    """Split a comma-separated key string into Fernet instances.

    Whitespace around each entry is stripped and empty entries are skipped so
    a trailing comma will not explode. Each entry is validated by the Fernet
    constructor, which raises ValueError if the key is not a 32-byte url-safe
    base64 value.
    """
    keys: list[Fernet] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        keys.append(Fernet(entry.encode()))
    return keys


def _get_cipher() -> Optional[MultiFernet]:
    """Return the active cipher, or None when unconfigured.

    A None return means the caller should fall through to plaintext. The
    missing-key warning is emitted once per process to avoid log spam.
    """
    global _cipher, _missing_key_warned
    if _cipher is not None:
        return _cipher

    raw = settings.encryption_key or ""
    # Fail closed for EVERY non-development environment, not just the exact
    # string "production". Staging/preview hold real or copied PHI, and a
    # misspelled APP_ENV ("prod", "Production") must not silently disable
    # encryption. Only development and test get the plaintext fallback. Matches
    # the "anything but development is a serious env" rule already in config.py.
    is_prod = settings.app_env.strip().lower() not in ("development", "test")

    if not raw:
        if is_prod:
            # Fail closed. The startup gate normally catches this, but never
            # silently store PHI in plaintext in production.
            raise EncryptionConfigError(
                "ENCRYPTION_KEY is not set in production; refusing to store or "
                "read PHI in plaintext."
            )
        if not _missing_key_warned:
            logger.warning(
                "ENCRYPTION_KEY not set (APP_ENV=%s). Values will be stored "
                "in plaintext. Set ENCRYPTION_KEY before shipping to prod.",
                settings.app_env,
            )
            _missing_key_warned = True
        return None

    try:
        keys = _parse_keys(raw)
    except (ValueError, TypeError) as e:
        if is_prod:
            raise EncryptionConfigError(
                f"ENCRYPTION_KEY failed to parse in production: {e}"
            ) from e
        logger.error("ENCRYPTION_KEY failed to parse, falling back to plaintext: %s", e)
        return None

    if not keys:
        if is_prod:
            raise EncryptionConfigError(
                "ENCRYPTION_KEY parsed to zero keys in production."
            )
        logger.warning("ENCRYPTION_KEY parsed to zero keys, falling back to plaintext.")
        return None

    _cipher = MultiFernet(keys)
    if len(keys) > 1:
        logger.info("Encryption initialized with %d keys (rotation mode).", len(keys))
    return _cipher


def _reset_for_tests() -> None:
    """Drop the cached cipher so tests can manipulate settings and re-init.

    Not part of the public API; imported only by the encryption test module.
    """
    global _cipher, _missing_key_warned
    _cipher = None
    _missing_key_warned = False


def encrypt(plaintext: str | None) -> str | None:
    """Encrypt a string with the active key. Returns None for None input."""
    if plaintext is None:
        return None
    cipher = _get_cipher()
    if cipher is None:
        return plaintext
    return cipher.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str | None) -> str | None:
    """Decrypt a string. Returns None for None input.

    Tries every configured key. If none match, the input is returned
    unchanged on the assumption it is a pre-encryption legacy row, and the
    miss is logged so legacy rows can be distinguished from rotated-away
    keys or tampering in aggregate metrics.
    """
    if ciphertext is None:
        return None
    cipher = _get_cipher()
    if cipher is None:
        return ciphertext
    try:
        return cipher.decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        # Do NOT log any slice of the value: once PHI columns route through
        # EncryptedString, a legacy plaintext row's first chars would be real
        # chat/health content. Log only non-content discriminators. A
        # Fernet-shaped value that misses (looks_fernet=True) is a drained-key
        # or tampered row, not legacy plaintext, and is worth alerting on.
        logger.warning(
            "encryption.decrypt_miss len=%d looks_fernet=%s",
            len(ciphertext),
            ciphertext.startswith("gAAAAA"),
        )
        return ciphertext
    except (ValueError, UnicodeDecodeError) as e:
        logger.error("encryption.decrypt_error: %s", e)
        return ciphertext


class EncryptedString(TypeDecorator):
    """SQLAlchemy column type that encrypts and decrypts transparently.

    Usage:
        access_token: Mapped[str] = mapped_column(EncryptedString(2000))

    CRITICAL: this TypeDecorator is referenced as a column type in every
    PHI table model. SQLAlchemy column-type usage is NOT a CALLS edge in
    static call graphs; impact tools that only look at calls will report
    `EncryptedString` as low-impact. In reality, any change here touches
    every encrypted PHI column. Tier 3 by blast radius.
    """

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return encrypt(value)

    def process_result_value(self, value, dialect):
        return decrypt(value)
