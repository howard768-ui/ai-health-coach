"""Audit P2b (round-1 H): POST /api/notifications/opened must read
notification_id from the JSON body (the iOS client sends it in the body via
APINotificationOpenedRequest). It used to declare it as a bare scalar, which
FastAPI treats as a required query param, so every open report 422'd and open
tracking was silently broken.

Run: cd backend && uv run pytest tests/test_notification_opened_contract.py -v
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-long-enough-aaaaaaaaaaaa")
os.environ.setdefault(
    "ENCRYPTION_KEY", "T0TXLkHFSeZRYGIIejSFVkhQrvRE-bWLkwXSkkdWiKQ="
)


def _opened_route():
    from app.main import app

    for r in app.routes:
        if getattr(r, "path", "").endswith("/notifications/opened") and "POST" in getattr(
            r, "methods", set()
        ):
            return r
    raise AssertionError("POST /notifications/opened route not found")


def test_opened_reads_notification_id_from_body():
    route = _opened_route()
    assert route.body_field is not None, (
        "/opened must accept a request body (notification_id), not a query param"
    )


def test_opened_does_not_require_notification_id_as_query_param():
    route = _opened_route()
    query_names = {p.name for p in route.dependant.query_params}
    assert "notification_id" not in query_names, (
        "notification_id must be in the body, not the query string"
    )
