"""encrypt PHI free-text and raw_json columns at rest

Revision ID: c7d2e9f1a3b4
Revises: e9c3f7b285a1
Create Date: 2026-06-06 18:00:00.000000

Audit C2. The EncryptedString column type was applied only to OAuth-token
columns; every clinical free-text / raw payload column was stored in
plaintext. The models now mark these columns EncryptedString:

  - chat_messages.content        (the user<->coach conversation)
  - chat_messages.health_context (the health snapshot sent to the model)
  - sleep_records.raw_json       (full raw Oura sleep payload)
  - garmin_daily_records.raw_json
  - workout_records.raw_json     (Peloton)

This is a DATA-only migration: it encrypts existing plaintext values in place
so reads through EncryptedString return them correctly. No DDL change is made
(the underlying text column already holds the ciphertext string; the type
decorator only governs Python-side encrypt/decrypt). It is idempotent: a value
that already decrypts under the active key is left untouched, so re-running or
running after some rows were written post-deploy is safe.

No key configured (local dev) => no-op. In production the key is always set
(core/secrets.py fails the deploy otherwise), so prod rows get encrypted.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.core.encryption import _get_cipher


revision: str = "c7d2e9f1a3b4"
down_revision: Union[str, Sequence[str], None] = "e9c3f7b285a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (table, column) pairs whose plaintext PHI is being encrypted at rest. These
# are fixed code constants, never user input, so the f-string SQL below is not
# an injection surface.
_TARGETS: list[tuple[str, str]] = [
    ("chat_messages", "content"),
    ("chat_messages", "health_context"),
    ("users", "custom_goal_text"),
    ("sleep_records", "raw_json"),
    ("garmin_daily_records", "raw_json"),
    ("workout_records", "raw_json"),
]

# KNOWN CONSTRAINTS (tracked follow-ups, not gating this PR):
#  - Key rotation: these columns are append-only (never re-saved), so they do
#    NOT drain a retired key the way refreshing OAuth tokens do. Before removing
#    any retired key from ENCRYPTION_KEY, run a re-encrypt pass over _TARGETS
#    first, or historical rows become undecryptable. Document in the rotation
#    runbook.
#  - Idempotency holds for a single active key (already-encrypted rows are
#    skipped). Re-running after a key rotation that DROPS a key can re-encrypt
#    rows that no longer decrypt; gate that on the re-encrypt pass above.
#  - Batching: this does fetchall()+per-row UPDATE in one transaction. Prod is
#    single-user today so this is fine; chunk by id keyset before any large
#    backfill (e.g. post-launch growth).


def _is_ciphertext(cipher, value: str) -> bool:
    """True if value already decrypts under a configured key (already encrypted)."""
    try:
        cipher.decrypt(value.encode())
        return True
    except Exception:
        return False


def _transform(transform_fn, skip_fn) -> None:
    bind = op.get_bind()
    cipher = _get_cipher()
    if cipher is None:
        # No encryption key (local dev). Nothing to do; columns stay plaintext.
        return
    for table, col in _TARGETS:
        rows = bind.execute(
            sa.text(f"SELECT id, {col} FROM {table} WHERE {col} IS NOT NULL")  # noqa: S608
        ).fetchall()
        for row_id, value in rows:
            if value is None or skip_fn(cipher, value):
                continue
            new_value = transform_fn(cipher, value)
            if new_value is None:
                continue
            bind.execute(
                sa.text(f"UPDATE {table} SET {col} = :v WHERE id = :i"),  # noqa: S608
                {"v": new_value, "i": row_id},
            )


def upgrade() -> None:
    # Encrypt any plaintext value; skip values that already decrypt (idempotent).
    _transform(
        transform_fn=lambda cipher, v: cipher.encrypt(v.encode()).decode(),
        skip_fn=_is_ciphertext,
    )


def _decrypt_or_none(cipher, value: str):
    try:
        return cipher.decrypt(value.encode()).decode()
    except Exception:
        # Already plaintext / legacy row: leave it.
        return None


def downgrade() -> None:
    # Decrypt back to plaintext; skip values that are not ciphertext.
    _transform(
        transform_fn=_decrypt_or_none,
        skip_fn=lambda cipher, v: not _is_ciphertext(cipher, v),
    )
