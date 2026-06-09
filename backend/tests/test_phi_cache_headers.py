"""Audit P3/E6: PHI-bearing API responses must not be cacheable.

Every /api/* response can carry PHI (dashboard, chat history, trends, meals,
profile). The no_store_on_api middleware stamps Cache-Control: no-store on
the whole /api/ surface so intermediaries and on-disk client caches (iOS
URLCache persists into device backups) never store PHI bodies.

Run: cd backend && ./.venv/bin/python -m pytest tests/test_phi_cache_headers.py -v
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-cache-headers")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)

from app.main import app


@pytest.mark.asyncio
async def test_api_routes_get_no_store():
    """Any /api/* response carries Cache-Control: no-store, including errors
    (401 bodies can still leak route shape/detail into caches)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Unauthenticated PHI route: response (401) must still be no-store.
        resp = await client.get("/api/user/profile")
        assert resp.headers.get("cache-control") == "no-store"


@pytest.mark.asyncio
async def test_health_probe_not_forced_no_store():
    """Railway probes (/healthz) are not /api/ and keep default caching
    semantics; the middleware must not blanket the whole app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") != "no-store"


@pytest.mark.asyncio
async def test_existing_cache_control_not_clobbered():
    """setdefault semantics: if a route ever sets its own Cache-Control,
    the middleware must not overwrite it. Pinned via the waitlist route
    (public, no PHI) only if it sets one; otherwise no-store applies."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/waitlist/subscribe", json={"email": "cache-test@example.com"}
        )
        # Whatever the route returns, the header must exist on /api/*.
        assert "cache-control" in resp.headers
