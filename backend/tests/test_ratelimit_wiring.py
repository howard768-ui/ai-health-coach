"""Audit C2: the rate limiter must actually be wired up.

Previously SlowAPIMiddleware was never registered (so default_limits were dead
config and undecorated endpoints were unlimited), and three routers each built
their own Limiter keyed on get_remote_address (the Cloudflare edge IP, one
shared bucket). These tests pin the corrected wiring: one shared limiter, the
middleware registered, and the real-client-IP key function.

Run: cd backend && uv run pytest tests/test_ratelimit_wiring.py -v
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-long-enough-aaaaaaaaaaaa")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

from types import SimpleNamespace

from slowapi.middleware import SlowAPIMiddleware

from app.core.ratelimit import limiter as shared_limiter, _real_remote_address


def test_slowapi_middleware_registered():
    """Without the middleware, default_limits never apply and undecorated
    endpoints (including Claude-hitting ones) are unlimited."""
    from app.main import app

    assert any(
        m.cls is SlowAPIMiddleware for m in app.user_middleware
    ), "SlowAPIMiddleware must be registered"


def test_app_and_routers_share_one_limiter():
    from app.main import app
    from app.routers import auth_apple, coach, meals

    assert app.state.limiter is shared_limiter
    assert auth_apple.limiter is shared_limiter
    assert coach.limiter is shared_limiter
    assert meals.limiter is shared_limiter


def test_keyfunc_prefers_real_client_ip_over_edge():
    # Cloudflare-injected real client IP wins over the socket peer (edge IP).
    r = SimpleNamespace(
        headers={"cf-connecting-ip": "203.0.113.7"},
        client=SimpleNamespace(host="10.0.0.1"),
    )
    assert _real_remote_address(r) == "203.0.113.7"
    # Else the leftmost x-forwarded-for hop.
    r2 = SimpleNamespace(
        headers={"x-forwarded-for": "198.51.100.9, 10.0.0.1"},
        client=SimpleNamespace(host="10.0.0.1"),
    )
    assert _real_remote_address(r2) == "198.51.100.9"
    # Else the raw peer.
    r3 = SimpleNamespace(headers={}, client=SimpleNamespace(host="10.0.0.1"))
    assert _real_remote_address(r3) == "10.0.0.1"


def test_default_limit_configured():
    # A global baseline must exist so undecorated endpoints are not unlimited.
    assert shared_limiter._default_limits, "a global default limit must be set"


def test_infra_endpoints_exempt_from_default_limit():
    """Re-gate fix: health/readiness/ops endpoints must be exempt from the
    global default, or Railway's deploy probe and monitoring get 429'd."""
    from app.main import app  # noqa: F401 (import registers all routes/decorators)

    exempt = shared_limiter._exempt_routes
    assert any(n.endswith(".healthz") for n in exempt), exempt
    assert any(n.endswith(".readyz") for n in exempt), exempt
    # All ml_ops monitoring endpoints exempt.
    assert any(".ml_ops." in n for n in exempt), exempt
    # /ops/status exempt.
    assert any(n.endswith("ops.status") or ".ops." in n for n in exempt), exempt


def test_waitlist_has_explicit_limit():
    """Re-gate fix: the public waitlist endpoint now carries its documented
    10/minute limit instead of relying on the global default."""
    from app.main import app  # noqa: F401

    assert any("subscribe" in n for n in shared_limiter._route_limits), (
        shared_limiter._route_limits.keys()
    )
