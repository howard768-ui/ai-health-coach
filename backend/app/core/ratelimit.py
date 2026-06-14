"""Single shared application rate limiter.

Audit C2. Previously each router (auth_apple, coach, meals) constructed its OWN
``Limiter(key_func=get_remote_address)`` and the app-level limiter's
``default_limits`` were never enforced because ``SlowAPIMiddleware`` was never
registered. Net effect: per-route limits keyed on the Cloudflare edge IP
(get_remote_address returns request.client.host = the CF/Railway edge, one
shared bucket for everyone), and every endpoint WITHOUT an explicit
``@limiter.limit`` decorator (including Claude-hitting ones) had no limit at all.

This module defines ONE limiter that the whole app shares (main + every router
decorator + the middleware), keyed on the real client IP, with the default
applied globally via SlowAPIMiddleware in app.main.
"""

from fastapi import Request
from slowapi import Limiter

from app.config import settings


def _real_remote_address(request: Request) -> str:
    """Real client IP, preferring trusted forwarded headers over the edge IP.

    The default ``slowapi.util.get_remote_address`` returns
    ``request.client.host``, which behind Cloudflare/Railway is the edge IP,
    collapsing every rate-limit bucket into one. Prefer cf-connecting-ip, then
    the first x-forwarded-for hop, then the raw peer.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",", 1)[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


# Active in production/staging; disabled in development/test so the in-process
# test client (a single shared IP) does not trip the global limit and local
# iteration stays unthrottled. Mirrors the env rule used elsewhere.
_RATE_LIMIT_ENABLED = settings.app_env.strip().lower() not in ("development", "test")

limiter = Limiter(
    key_func=_real_remote_address,
    default_limits=["120/minute"],  # global baseline for every endpoint
    enabled=_RATE_LIMIT_ENABLED,
)
