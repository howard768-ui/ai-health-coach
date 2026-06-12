"""Tests for the additional CRITICAL audit-flagged paths:

  - `_scrub_phi` (PHI/PII scrub before Sentry transmit)
  - `_real_remote_address` (real client IP for rate limiter — bypass risk)
  - `normalize_pem` (PEM env-var-mangling normalizer)

The first two live in `app/main.py`; the third in `app/core/pem.py`. All
were flagged as zero-test in the 2026-04-30 audit (MEL-43) despite being
load-bearing for security/observability.

Run: cd backend && uv run pytest tests/test_audit_safety_paths.py -v
"""

import os
from types import SimpleNamespace

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-audit-safety-tests")

from app.core.pem import normalize_pem
from app.main import _real_remote_address, _scrub_phi


# ── _scrub_phi: never let PII reach Sentry ─────────────────────────────


def test_scrub_phi_strips_apple_user_id_in_strings():
    """apple_user_id format is 6-digit.32-hex.4-digit. Must be replaced with
    `[apple_user_id]` placeholder."""
    result = _scrub_phi("user 123456.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.0001 logged in")
    assert "[apple_user_id]" in result
    assert "123456" not in result


def test_scrub_phi_strips_email():
    result = _scrub_phi("notify alice@example.com about it")
    assert "[email]" in result
    assert "alice@example.com" not in result


def test_scrub_phi_strips_apple_private_relay_email():
    """Apple's private-relay addresses follow the same email shape and
    must also be scrubbed (they identify a real user)."""
    result = _scrub_phi("relay user xyz123abc@privaterelay.appleid.com signed in")
    assert "[email]" in result
    assert "privaterelay" not in result


def test_scrub_phi_strips_bearer_token():
    """Bearer tokens (JWTs after `Bearer `) must be replaced. Pattern is
    case-insensitive on `bearer`."""
    result = _scrub_phi("Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234567890.abc.xyz")
    assert "[token]" in result
    assert "abcdefghijklmnopqrstuvwxyz" not in result


def test_scrub_phi_recursive_into_dicts():
    """Sentry events have nested dicts (extra, contexts, etc.). Recurse."""
    event = {
        "message": "alice@example.com hit the endpoint",
        "extra": {
            "user_email": "bob@example.com",
            "nested": {"deeper": "carol@example.com"},
        },
    }
    result = _scrub_phi(event)
    assert "alice@example.com" not in str(result)
    assert "bob@example.com" not in str(result)
    assert "carol@example.com" not in str(result)


def test_scrub_phi_recursive_into_lists_and_tuples():
    event = {
        "breadcrumbs": ["user alice@example.com", ("nested-tuple-with-bob@example.com",)],
    }
    result = _scrub_phi(event)
    assert "alice@example.com" not in str(result)
    assert "bob@example.com" not in str(result)


def test_scrub_phi_handles_non_string_primitives():
    """Numbers, bools, None pass through unchanged."""
    assert _scrub_phi(42) == 42
    assert _scrub_phi(True) is True
    assert _scrub_phi(None) is None


def test_scrub_phi_preserves_safe_strings():
    """Strings without PII should pass through unchanged (no false positives)."""
    safe = "scheduler ran job morning_brief at 08:00"
    assert _scrub_phi(safe) == safe


# ── _init_sentry: never capture request bodies or frame locals ────────────


def test_init_sentry_disables_body_and_local_capture(monkeypatch):
    """Audit A4: the regex scrubber only matches a few structured patterns, so
    it cannot catch free-form chat/meal/health text. Sentry must therefore not
    capture request bodies or frame-local variables at all, since those can
    hold PHI on an exception.
    """
    import sentry_sdk
    from app import main as appmain

    captured: dict = {}
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        appmain.settings, "sentry_dsn", "https://public@o0.ingest.sentry.io/1"
    )

    appmain._init_sentry()

    assert captured["send_default_pii"] is False
    assert captured["max_request_body_size"] == "never"
    assert captured["include_local_variables"] is False
    assert captured["before_send"] is not None


# ── _real_remote_address: rate-limit-IP determination ─────────────────


def _fake_request(*, headers: dict | None = None, client_host: str | None = None):
    """Build a minimal fake Request with the headers + client.host shape
    that `_real_remote_address` consumes. Avoids pulling in TestClient."""
    h = headers or {}
    client = SimpleNamespace(host=client_host) if client_host else None
    return SimpleNamespace(
        headers={k.lower(): v for k, v in h.items()},
        client=client,
    )


def test_remote_address_prefers_cf_connecting_ip():
    """When Cloudflare's `cf-connecting-ip` is set (CF-injected, can't be
    spoofed by client), trust it over everything else."""
    req = _fake_request(
        headers={
            "cf-connecting-ip": "203.0.113.5",
            "x-forwarded-for": "10.0.0.1, 198.51.100.7",
        },
        client_host="172.16.0.1",
    )
    assert _real_remote_address(req) == "203.0.113.5"


def test_remote_address_falls_back_to_xff_first_hop():
    """No `cf-connecting-ip` → take the leftmost (original client) entry
    from `x-forwarded-for`."""
    req = _fake_request(
        headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1, 192.0.2.1"},
        client_host="172.16.0.1",
    )
    assert _real_remote_address(req) == "203.0.113.5"


def test_remote_address_xff_strips_whitespace():
    req = _fake_request(headers={"x-forwarded-for": "  203.0.113.5 , 10.0.0.1"})
    assert _real_remote_address(req) == "203.0.113.5"


def test_remote_address_falls_through_to_client_host():
    """No CF / XFF headers → raw socket peer."""
    req = _fake_request(client_host="172.16.0.1")
    assert _real_remote_address(req) == "172.16.0.1"


def test_remote_address_returns_unknown_when_nothing_available():
    req = _fake_request(headers={}, client_host=None)
    assert _real_remote_address(req) == "unknown"


def test_remote_address_empty_xff_falls_through_to_client_host():
    """Empty XFF (just whitespace) shouldn't be mistaken for a valid IP."""
    req = _fake_request(headers={"x-forwarded-for": "   "}, client_host="172.16.0.1")
    assert _real_remote_address(req) == "172.16.0.1"


# ── normalize_pem: env-var-mangling normalizer ─────────────────────────


_VALID_PEM_BODY = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg0dkKClZOFfBiR97M
HC+W4pDRhURczG1gsgO4YBPmmaihRANCAASwILEr7OJHqY+5G8MG30ArYlc1NTjQ
basRidkaJtVpC+sq7++KgCzgcIlBpmT6X75guSSZxbvgha8EHqbYmyEZ
-----END PRIVATE KEY-----"""


def test_normalize_pem_empty_passes_through():
    assert normalize_pem("") == ""
    assert normalize_pem(None) is None  # type: ignore[arg-type]


def test_normalize_pem_lf_input_unchanged_modulo_trailing_newline():
    """Plain LF-terminated PEM gets a trailing newline appended (cryptography
    requires one)."""
    result = normalize_pem(_VALID_PEM_BODY)
    assert result.endswith("\n")
    assert "-----BEGIN PRIVATE KEY-----" in result
    assert "-----END PRIVATE KEY-----" in result


def test_normalize_pem_crlf_collapses_to_lf():
    """CRLF (Windows clipboard) collapses to LF."""
    crlf_pem = _VALID_PEM_BODY.replace("\n", "\r\n")
    result = normalize_pem(crlf_pem)
    assert "\r" not in result, "CR character must not appear in normalized output"


def test_normalize_pem_literal_backslash_n_replaced():
    """JSON-style env strings often deliver literal `\\n` (two chars).
    Pre-PR-#80 audit: this was the bug class that produced `MalformedFraming`
    for 7+ Sentry issues."""
    literal = _VALID_PEM_BODY.replace("\n", "\\n")
    result = normalize_pem(literal)
    assert "\\n" not in result, "Literal backslash-n must be replaced with real LF"
    assert "\n" in result


def test_normalize_pem_strips_leading_trailing_whitespace():
    padded = f"   \n\t  {_VALID_PEM_BODY}   \n\t  "
    result = normalize_pem(padded)
    assert result.startswith("-----BEGIN PRIVATE KEY-----")


def test_normalize_pem_ensures_exactly_one_trailing_newline():
    """No matter how many trailing newlines come in, exactly one comes out."""
    result_a = normalize_pem(_VALID_PEM_BODY)
    result_b = normalize_pem(_VALID_PEM_BODY + "\n\n\n\n")
    result_c = normalize_pem(_VALID_PEM_BODY + "    ")
    assert result_a.endswith("-----END PRIVATE KEY-----\n")
    assert result_b.endswith("-----END PRIVATE KEY-----\n")
    assert result_c.endswith("-----END PRIVATE KEY-----\n")
    # Exactly one — none of the outputs end with two newlines
    for result in (result_a, result_b, result_c):
        assert not result.endswith("\n\n")
