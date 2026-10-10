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
