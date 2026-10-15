"""WebSocket protocol tests.

These use a stub pipeline (monkeypatched into ``app.main.PIPELINE``) so the
protocol mechanics - ordering, duplicates, limits, isolation, cancellation - are
exercised quickly and deterministically. Real-model inference is covered
separately in ``test_inference.py`` (marked ``real``).
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

SEQ = struct.Struct(">I")
REQUIRED = 150


def _jpeg_bytes(color=(120, 80, 200), size=(64, 64)) -> bytes:
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] = color
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _frame(index: int, payload: bytes | None = None) -> bytes:
    return SEQ.pack(index) + (payload if payload is not None else _jpeg_bytes())


class StubPipeline:
    """Fast stand-in that records inputs and returns a fixed result."""

    def __init__(self, prediction="live", score=0.05):
        self.is_ready = True
        self.load_error = None
        self.device = "cpu"
        self.config = SimpleNamespace(num_frames=REQUIRED)
        self.prediction = prediction
        self.score = score
        self.calls = []

    def predict(self, frames):
        self.calls.append(list(frames))
        return SimpleNamespace(
            prediction=self.prediction,
            anomaly_score=self.score,
            reconstruction_error=0.01,
            threshold=0.5,
            frames_processed=len(frames),
            inference_ms=12.3,
            mean_frame_error=0.01,
        )


def _install_stub(client, prediction="live", score=0.05) -> StubPipeline:
    from app import main
    stub = StubPipeline(prediction=prediction, score=score)
    main.PIPELINE = stub  # routes read PIPELINE from module globals at call time
    return stub


def _session(client):
    body = client.post("/sessions").json()
    return body["session_id"], body["ws_token"]


def _recv_until(ws, predicate, limit=400):
    """Read messages until ``predicate(msg)`` is true (or ``limit`` reached).

    ``TestClient.receive_json()`` blocks when nothing is pending, so tests must
    read exactly the messages the server will send. This helper reads lazily and
    stops as soon as the message of interest arrives.
    """
    collected = []
    for _ in range(limit):
        msg = ws.receive_json()
        collected.append(msg)
        if predicate(msg):
            return msg, collected
    raise AssertionError(f"predicate not satisfied within {limit} messages; got: {collected[-5:]}")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_full_stream_returns_result(fastapi_client):
    stub = _install_stub(fastapi_client, prediction="spoof", score=0.97)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["required_frames"] == REQUIRED

        for i in range(REQUIRED):
            ws.send_bytes(_frame(i))

        # Drain messages; expect acks then processing then result.
        result = None
        for _ in range(REQUIRED + 10):
            msg = ws.receive_json()
            if msg["type"] == "result":
                result = msg
                break
        assert result is not None, "no result received"
        assert result["prediction"] == "spoof"
        assert result["anomaly_score"] == 0.97
        assert result["threshold"] == 0.5
        assert result["frames_processed"] == REQUIRED
        assert "score_semantics" in result

    # Pipeline received the frames in order.
    assert len(stub.calls) == 1
    assert len(stub.calls[0]) == REQUIRED


def test_progress_acks_sent(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()  # ready
        for i in range(20):
            ws.send_bytes(_frame(i))
        # Acks fire at received == 10 and 20.
        ack, _ = _recv_until(ws, lambda m: m.get("type") == "ack" and m["received"] == 20)
        assert ack["required"] == REQUIRED


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------
def test_duplicate_frame_rejected(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        ws.send_bytes(_frame(0))
        ws.send_bytes(_frame(0))  # duplicate
        msg, _ = _recv_until(ws, lambda m: m.get("type") == "error")
    assert msg["error"] == "duplicate_frame"


def test_out_of_range_index_rejected(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        ws.send_bytes(_frame(REQUIRED + 5))  # beyond 150
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["error"] == "bad_index"


def test_malformed_short_frame_rejected(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x01")  # shorter than the 4-byte header
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["error"] == "malformed_frame"


def test_oversized_frame_fails_session(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        big = _frame(0, payload=b"x" * (3 * 1024 * 1024))  # > 2 MiB default
        ws.send_bytes(big)
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["error"] == "frame_too_large"


def test_incomplete_sequence_on_complete(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        for i in range(10):
            ws.send_bytes(_frame(i))
        ws.send_text('{"type": "complete"}')
        msg, _ = _recv_until(ws, lambda m: m.get("type") == "error")
    assert msg["error"] == "incomplete_sequence"
    assert msg["missing"] == REQUIRED - 10


def test_inference_not_run_on_incomplete_sequence(fastapi_client):
    stub = _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        for i in range(5):
            ws.send_bytes(_frame(i))
        ws.send_text('{"type": "complete"}')
        ws.receive_json()
    assert stub.calls == [], "inference must not run on an incomplete sequence"


def test_cancel_clears_session(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        for i in range(5):
            ws.send_bytes(_frame(i))
        ws.send_text('{"type": "cancel"}')
        msg = ws.receive_json()
    assert msg["type"] == "cancelled"


def test_invalid_token_rejected(fastapi_client):
    sid, _ = _session(fastapi_client)
    try:
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token=wrong") as ws:
            ws.receive_json()
            raised = False
    except Exception:
        raised = True
    assert raised, "wrong token should be rejected"


def test_control_json_must_be_valid(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ws.receive_json()
        ws.send_text("not json {")
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg["error"] == "bad_message"


# ---------------------------------------------------------------------------
# Handshake acceptance / rejection (the production `403 Forbidden` regression)
#
# A close() sent before accept() makes the ASGI server answer the upgrade
# with HTTP 403, so every rejection below is indistinguishable from the
# outside - the route logs a structured reason so Render's logs can tell
# them apart.
#
# Origin is deliberately NOT one of the gates. Starlette's CORSMiddleware
# only inspects `http` scopes and never sees a WebSocket, and the handshake
# is authorised by the per-session capability token instead: a third-party
# page can mint its own session but can never read another origin's
# `ws_token`, because reading that response is precisely what CORS blocks
# on POST /sessions. The tests below pin that contract down from both
# directions.
# ---------------------------------------------------------------------------
PRODUCTION_ORIGIN = "https://face-liveness-platform.vercel.app"


def test_handshake_accepted_for_valid_session_from_production_origin(fastapi_client):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(
        f"/ws/sessions/{sid}?token={token}", headers={"origin": PRODUCTION_ORIGIN}
    ) as ws:
        ready = ws.receive_json()
    assert ready["type"] == "ready"
    assert ready["session_id"] == sid


def test_handshake_accepted_with_no_origin_header(fastapi_client):
    """Local development and non-browser clients send no Origin at all."""
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        ready = ws.receive_json()
    assert ready["type"] == "ready"


def test_handshake_accepted_from_arbitrary_origin_with_valid_token(fastapi_client):
    """Origin is not a gate; the capability token is."""
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)
    with fastapi_client.websocket_connect(
        f"/ws/sessions/{sid}?token={token}", headers={"origin": "https://evil.example"}
    ) as ws:
        ready = ws.receive_json()
    assert ready["type"] == "ready"


def test_handshake_rejected_when_token_query_is_absent(fastapi_client):
    """The exact request the SDK issued before it appended `?token=`.

    A required query parameter failed FastAPI's dependency validation before
    the route ever ran, which is why production only ever showed uvicorn's
    bare `403 Forbidden` with no diagnostic of its own.
    """
    sid, _ = _session(fastapi_client)
    with pytest.raises(WebSocketDisconnect):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}") as ws:
            ws.receive_json()


def test_handshake_rejected_for_empty_token(fastapi_client):
    sid, _ = _session(fastapi_client)
    with pytest.raises(WebSocketDisconnect):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token=") as ws:
            ws.receive_json()


def test_handshake_rejected_for_wrong_token(fastapi_client):
    sid, _ = _session(fastapi_client)
    with pytest.raises(WebSocketDisconnect):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token=wrong") as ws:
            ws.receive_json()


def test_handshake_rejected_for_unknown_session(fastapi_client):
    with pytest.raises(WebSocketDisconnect):
        with fastapi_client.websocket_connect(
            "/ws/sessions/definitely-not-a-session?token=x"
        ) as ws:
            ws.receive_json()


def test_handshake_rejected_for_unknown_session_from_production_origin(fastapi_client):
    """A well-formed Origin grants nothing on its own."""
    with pytest.raises(WebSocketDisconnect):
        with fastapi_client.websocket_connect(
            "/ws/sessions/definitely-not-a-session?token=x",
            headers={"origin": PRODUCTION_ORIGIN},
        ) as ws:
            ws.receive_json()


def test_handshake_rejection_logs_reason_without_secrets(
    fastapi_client, caplog
):
    """Rejections must be diagnosable in production logs, leak-free.

    The token must never appear - neither as a value nor as part of the
    message - and the session id must be truncated, matching the rest of the
    module's logging.
    """
    sid, token = _session(fastapi_client)

    with caplog.at_level("WARNING", logger="liveness"):
        with pytest.raises(WebSocketDisconnect):
            with fastapi_client.websocket_connect(
                f"/ws/sessions/{sid}?token={token[:-3]}"
            ) as ws:
                ws.receive_json()

    rejection = [r for r in caplog.records if "ws_rejected" in r.getMessage()]
    assert rejection, "handshake rejection must emit a structured ws_rejected record"
    message = rejection[-1].getMessage()
    assert "reason=token_invalid" in message
    assert f"close_code=4401" in message
    assert token not in message, "capability token leaked into logs"
    assert sid not in message, "full session id leaked into logs"
    assert sid[:8] in message, "session prefix missing, so logs cannot be correlated"


def test_handshake_acceptance_is_logged(fastapi_client, caplog):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)

    with caplog.at_level("INFO", logger="liveness"):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
            ws.receive_json()

    accepted = [r for r in caplog.records if "ws_accepted" in r.getMessage()]
    assert accepted, "accepted handshake must be logged"
    message = accepted[-1].getMessage()
    assert token not in message
    assert sid not in message
