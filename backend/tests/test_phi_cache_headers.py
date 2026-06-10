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
async def test_header_present_on_validation_errors_too():
    """422 validation responses are also /api/* bodies and must carry the
    header. Uses an invalid payload so the route never touches the DB
    (CI has no app tables; the middleware is what is under test)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/waitlist/subscribe", json={"email": "not-an-email"}
        )
        assert resp.status_code == 422
        assert resp.headers.get("cache-control") == "no-store"
