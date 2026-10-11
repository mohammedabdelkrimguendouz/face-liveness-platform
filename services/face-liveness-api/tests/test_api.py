"""API surface tests: health, readiness, session lifecycle, limits."""

from __future__ import annotations

import time


def test_health_ok(fastapi_client):
    r = fastapi_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_reports_model(fastapi_client):
    r = fastapi_client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["num_frames"] == 150
    assert "active_sessions" in body


def test_root_metadata(fastapi_client):
    r = fastapi_client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "face-liveness-api"


def test_create_session_shape(fastapi_client):
    r = fastapi_client.post("/sessions")
    assert r.status_code == 201
    body = r.json()
    for key in ("session_id", "ws_url", "ws_token", "num_frames", "max_frame_bytes", "expires_in_seconds"):
        assert key in body, f"missing {key}"
    assert body["num_frames"] == 150
    assert len(body["session_id"]) > 20  # crypto-random, not guessable
    # The WS URL must be a path, never an absolute ws://localhost/... URL:
    # the SDK resolves it against its configured API base, which is what lets
    # the same server build work locally and behind Render's proxy.
    assert body["ws_url"] == f"/ws/sessions/{body['session_id']}"
    assert not body["ws_url"].startswith(("ws://", "wss://", "http://", "https://"))
    assert "localhost" not in body["ws_url"]


def test_session_status_endpoint(fastapi_client):
    sid = fastapi_client.post("/sessions").json()["session_id"]
    r = fastapi_client.get(f"/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "created"
    assert body["received_frames"] == 0
    assert body["required_frames"] == 150


def test_unknown_session_404(fastapi_client):
    r = fastapi_client.get("/sessions/does-not-exist")
    assert r.status_code == 404


def test_session_limit_returns_429(fastapi_client, monkeypatch):
    """Creating more than the configured maximum must fail cleanly."""
    from app import main
    # Start from a clean slate (earlier tests share the module-level manager).
    main.SESSIONS._sessions.clear()
    monkeypatch.setattr(main.SESSIONS, "_max_sessions", 2)
    codes = [fastapi_client.post("/sessions").status_code for _ in range(3)]
    assert codes[:2] == [201, 201]
    assert codes[2] == 429
    main.SESSIONS._sessions.clear()


def test_session_ttl_expiry(fastapi_client, monkeypatch):
    """Expired sessions are evicted and no longer retrievable."""
    from app import main
    sid = fastapi_client.post("/sessions").json()["session_id"]
    # Force the session past its TTL.
    sess = main.SESSIONS.get(sid)
    sess.created_at = time.time() - (sess.ttl_seconds + 10)
    main.SESSIONS.evict_expired()
    r = fastapi_client.get(f"/sessions/{sid}")
    assert r.status_code == 404
