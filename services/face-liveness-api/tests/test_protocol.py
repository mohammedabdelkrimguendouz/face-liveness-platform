"""WebSocket protocol tests.

These use a stub pipeline (monkeypatched into ``app.main.PIPELINE``) so the
protocol mechanics - ordering, duplicates, limits, isolation, cancellation - are
exercised quickly and deterministically. Real-model inference is covered
separately in ``test_inference.py`` (marked ``real``).
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
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

    def __init__(self, prediction="live", score=0.05, delay=0.0):
        self.is_ready = True
        self.load_error = None
        self.device = "cpu"
        self.config = SimpleNamespace(num_frames=REQUIRED)
        self.prediction = prediction
        self.score = score
        # Wall-clock stall inside predict(), to exercise the window in which
        # production inference takes 40-50 s.
        self.delay = delay
        # Callable invoked while the model is "working", so a test can land a
        # terminal state on the session from inside the in-flight window.
        self.side_effect = None
        self.calls = []

    def predict(self, frames):
        if self.side_effect is not None:
            self.side_effect(list(frames))
        if self.delay:
            time.sleep(self.delay)
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


# ---------------------------------------------------------------------------
# Post-inference lifecycle: delivery, serialization, finalization
#
# Inference finishing is NOT the same as the client receiving the answer. A
# result that fails to leave the server used to be logged at DEBUG and then
# swallowed, so production showed only `Inference done` followed by the
# client's bare "Connection closed (code 1011)". These tests pin the two
# stages apart.
# ---------------------------------------------------------------------------
class _CapturingWS:
    """Transport stand-in recording exactly what the server tried to send.

    `fail_after=N` lets the first N frames through and then raises, which is
    how a real socket behaves when the client (or a proxy) disappears while
    inference is still running.
    """

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise RuntimeError("send on closed connection")
        self.sent.append(text)


def _populated_session(fastapi_client, prediction="live", score=0.05):
    """A real Session already holding `REQUIRED` frames, ready to infer."""
    _install_stub(fastapi_client, prediction=prediction, score=score)
    from app import main

    sid, _ = _session(fastapi_client)
    session = main.SESSIONS.get(sid)
    session.frames.clear()
    # The stub pipeline never decodes, so arbitrary bytes stand in for JPEGs.
    for i in range(REQUIRED):
        session.frames[i] = b"\xff\xd8\xff\xe0"
    return session


def _infer_with(ws, session):
    """Run _run_inference on its own loop with a loop-local semaphore."""
    from app import main

    original = main.INFER_SEMAPHORE

    async def run():
        # A fresh semaphore, because asyncio primitives are bound to the loop
        # that first awaits them and the app's own loop belongs to TestClient.
        main.INFER_SEMAPHORE = asyncio.Semaphore(1)
        await main._run_inference(ws, session, asyncio.Lock())

    try:
        asyncio.run(run())
    finally:
        main.INFER_SEMAPHORE = original


def test_result_payload_is_strict_json_with_every_sdk_field(fastapi_client):
    """The exact bytes sent must be strict JSON the browser can JSON.parse.

    `allow_nan=False` matters: json.dumps would otherwise emit a bare `NaN`
    literal, which Python accepts but `JSON.parse` rejects — the SDK would
    then silently drop the message and report only a socket close.
    """
    session = _populated_session(fastapi_client, prediction="spoof", score=0.9925)
    ws = _CapturingWS()
    _infer_with(ws, session)

    # `processing` is sent before inference, `result` after it.
    assert len(ws.sent) == 2, f"expected processing + result, got {ws.sent}"
    assert json.loads(ws.sent[0])["type"] == "processing"

    wire = ws.sent[1]
    json.dumps(wire, allow_nan=False)  # strict, no NaN/Infinity literals

    msg = json.loads(wire)
    assert msg["type"] == "result"
    assert msg["prediction"] == "spoof", "a spoof verdict must be reported, never hidden"
    for field in ("session_id", "anomaly_score", "reconstruction_error", "threshold",
                  "frames_processed", "inference_ms", "score_semantics"):
        assert field in msg, f"SDK reads msg.{field}"
    assert msg["frames_processed"] == REQUIRED
    assert isinstance(msg["anomaly_score"], float)
    assert isinstance(msg["threshold"], float)


def test_successful_inference_marks_session_delivered(fastapi_client):
    session = _populated_session(fastapi_client, prediction="spoof", score=0.99)
    ws = _CapturingWS()
    _infer_with(ws, session)

    assert session.state.value == "completed"
    assert session.result_delivered is True
    assert session.frames == {}, "frame buffer must be released after delivery"


def test_successful_inference_with_dead_transport_reports_delivery_failure(
    fastapi_client, caplog
):
    """Inference succeeded; only the send failed. The two must not be merged.

    This is the case that previously vanished into a DEBUG log: the model had
    already answered, but the operator could only see `Inference done` plus a
    client-side socket close.
    """
    session = _populated_session(fastapi_client, prediction="spoof", score=0.99)
    # `processing` goes out, then the client vanishes while the model works,
    # so only the result send fails.
    ws = _CapturingWS(fail_after=1)

    with caplog.at_level("INFO", logger="liveness"):
        _infer_with(ws, session)

    messages = [r.getMessage() for r in caplog.records]

    completed = [m for m in messages if m.startswith("inference_completed")]
    assert completed, "the model finished and must be reported as such"
    assert "prediction=spoof" in completed[0]

    assert any(m.startswith("result_send_started") for m in messages)
    assert any(m.startswith("result_send_succeeded") for m in messages) is False

    failed = [m for m in messages if m.startswith("result_send_failed")]
    assert failed, "a failed delivery must be visible at the default log level"
    assert "stage=after_inference" in failed[0]
    assert session.result_delivered is False, "state must record the undelivered result"
    assert session.state.value == "completed", "the model still answered; state stays completed"
    assert session.frames == {}, "frames are released either way"

    # The traceback that precedes it must be attributable to a session, or a
    # production log with several concurrent verifications is uninterpretable.
    send_failures = [m for m in messages if m.startswith("ws_send_failed")]
    assert send_failures, "the underlying transport error must still be logged"
    assert f"session={session.session_id[:8]}" in send_failures[0], send_failures[0]


def test_send_failure_logs_a_warning_with_traceback(fastapi_client):
    """Delivery failures must not be swallowed at DEBUG."""
    import logging

    from app import main

    class _ExplodingWS:
        async def send_text(self, text: str) -> None:
            raise RuntimeError("send on closed connection")

    records = []
    handler = type("H", (logging.Handler,), {"emit": lambda self, r: records.append(r)})()
    main.logger.addHandler(handler)
    main.logger.setLevel(logging.DEBUG)
    try:
        delivered = asyncio.run(main._send(_ExplodingWS(), {"type": "result", "session_id": "abc"}))
    finally:
        main.logger.removeHandler(handler)

    assert delivered is False
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert warnings, "a failed send must surface as WARNING, not DEBUG"
    assert warnings[-1].exc_info is not None, "the traceback is the only way to diagnose this"
    assert "ws_send_failed" in warnings[-1].getMessage()


def test_session_finalized_logged_with_delivery_flag(fastapi_client, caplog):
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)

    with caplog.at_level("INFO", logger="liveness"):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            for i in range(REQUIRED):
                ws.send_bytes(_frame(i))
            msg = ws.receive_json()
            while msg.get("type") != "result":
                msg = ws.receive_json()

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("result_send_started") for m in messages)
    assert any(m.startswith("result_send_succeeded") for m in messages)

    finalized = [m for m in messages if m.startswith("session_finalized")]
    assert finalized, "every terminal path must emit exactly one session_finalized"
    assert "delivered=True" in finalized[0]
    assert "state=completed" in finalized[0]
    assert token not in finalized[0]


def test_session_finalized_reported_when_client_leaves_early(fastapi_client, caplog):
    """An abandoned stream is still finalized, with delivered=False."""
    _install_stub(fastapi_client)
    sid, token = _session(fastapi_client)

    with caplog.at_level("INFO", logger="liveness"):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
            ws.receive_json()
            for i in range(5):
                ws.send_bytes(_frame(i))
            # Client walks away mid-stream; no `complete`, no result.

    messages = [r.getMessage() for r in caplog.records]
    finalized = [m for m in messages if m.startswith("session_finalized")]
    assert finalized, "an abandoned session must still be finalized"
    assert "delivered=False" in finalized[0]
    assert "state=cancelled" in finalized[0]
    assert token not in finalized[0]


# ---------------------------------------------------------------------------
# Long-running inference over a live socket
#
# Production inference takes 40-50 s. The old handler blocked in
# ws.receive() for that whole window and sent nothing at all, so the server
# could not notice a disconnect until the model finished, could not service
# `cancel`, and left an idle connection that an idle-timeout proxy reaps.
# These tests pin the window itself.
# ---------------------------------------------------------------------------
def test_slow_inference_keeps_socket_alive_and_still_delivers(fastapi_client, monkeypatch):
    """A client that stays connected for the whole inference must get an answer.

    The connection has to carry real frames during the stall — protocol-level
    pings alone are invisible to a proxy that only inspects data frames, and
    the user otherwise cannot tell "still thinking" from "link dead".
    """
    from app import main

    monkeypatch.setattr(main, "INFERENCE_KEEPALIVE_SECONDS", 0.05)

    stub = _install_stub(fastapi_client, prediction="spoof", score=0.9925)
    stub.delay = 0.30  # ~6 keepalive intervals, i.e. a long inference in miniature
    sid, token = _session(fastapi_client)

    with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
        assert ws.receive_json()["type"] == "ready"
        for i in range(REQUIRED):
            ws.send_bytes(_frame(i))
        _final, collected = _recv_until(ws, lambda m: m["type"] == "result")

    kinds = [m["type"] for m in collected]
    assert kinds.count("progress") >= 3, f"keepalives missing from {kinds}"
    assert "processing" in kinds, "the client must be told inference began"
    assert kinds[-1] == "result", "the verdict is the last frame on the wire"
    assert collected[-1]["prediction"] == "spoof"

    # Keepalives are exposed to the UI, so they must never carry a secret.
    wire = json.dumps(collected)
    assert token not in wire
    assert "token" not in wire


def test_client_disconnect_during_slow_inference_is_seen_immediately(
    fastapi_client, monkeypatch, caplog
):
    """Reproduces the production failure, with the decisive detail pinned.

    The server must notice the departure *while the model is still working*
    (`inference_running=True`) instead of only discovering it seconds later
    when the result fails to leave. That difference is what distinguishes
    "something closed the socket early" from "the model took too long".
    """
    from app import main

    monkeypatch.setattr(main, "INFERENCE_KEEPALIVE_SECONDS", 0.05)

    stub = _install_stub(fastapi_client, prediction="spoof", score=0.9925)
    stub.delay = 0.40
    sid, token = _session(fastapi_client)

    with caplog.at_level("INFO", logger="liveness"):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
            assert ws.receive_json()["type"] == "ready"
            for i in range(REQUIRED):
                ws.send_bytes(_frame(i))
            # Get to mid-inference, then hang up with the model still busy.
            _msg, _ = _recv_until(ws, lambda m: m["type"] == "processing")
        # `with` exits here: the socket closes during inference.

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if any(
                r.getMessage().startswith("session_finalized") for r in caplog.records
            ):
                break
            time.sleep(0.05)

    messages = [r.getMessage() for r in caplog.records]

    gone = [m for m in messages if m.startswith("ws_client_gone")]
    assert gone, "the server must observe the disconnect"
    assert "inference_running=True" in gone[0], gone[0]
    assert "via=message" in gone[0] or "via=exception" in gone[0], gone[0]

    # The model still ran to completion — it is not what failed.
    assert any(m.startswith("inference_completed") for m in messages)

    failed = [m for m in messages if m.startswith("result_send_failed")]
    assert failed and "stage=after_inference" in failed[0], failed
    assert any(m.startswith("result_send_succeeded") for m in messages) is False

    finalized = [m for m in messages if m.startswith("session_finalized")]
    assert finalized, "a detached inference must still finalize the session"
    assert "delivered=False" in finalized[0]

    # Diagnostic logs must never carry the capability token.
    assert token not in " ".join(messages)


def test_cancel_during_inference_supersedes_the_result(fastapi_client, monkeypatch, caplog):
    """A cancel that arrives mid-inference wins; no late `result` may follow.

    Cancellation was unreachable while inference blocked the receive loop, so
    this ordering did not have to be defined before. It does now.
    """
    from app import main

    monkeypatch.setattr(main, "INFERENCE_KEEPALIVE_SECONDS", 0.05)

    stub = _install_stub(fastapi_client, prediction="live", score=0.05)
    stub.delay = 0.30
    sid, token = _session(fastapi_client)

    with caplog.at_level("INFO", logger="liveness"):
        with fastapi_client.websocket_connect(f"/ws/sessions/{sid}?token={token}") as ws:
            assert ws.receive_json()["type"] == "ready"
            for i in range(REQUIRED):
                ws.send_bytes(_frame(i))
            _msg, _ = _recv_until(ws, lambda m: m["type"] == "processing")
            ws.send_text(json.dumps({"type": "cancel"}))
            _msg, collected = _recv_until(ws, lambda m: m["type"] == "cancelled")

        deadline = time.time() + 10.0
        while time.time() < deadline and not any(
            r.getMessage().startswith("session_finalized") for r in caplog.records
        ):
            time.sleep(0.05)

    assert "result" not in [m["type"] for m in collected], "cancelled session must not answer"
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("inference_discarded") for m in messages), messages
    finalized = [m for m in messages if m.startswith("session_finalized")]
    assert finalized and "delivered=False" in finalized[0], finalized
    assert token not in " ".join(messages)


def test_cancel_that_beats_the_inference_task_skips_inference(fastapi_client, caplog):
    """`create_task` only schedules; a `cancel` can be processed first.

    When that happens the frame buffer is already cleared, so running the
    model would hand it a truncated sequence and then emit an `inference_error`
    for a session that never asked to fail.
    """
    from app.sessions import SessionState

    session = _populated_session(fastapi_client, prediction="live", score=0.05)
    session.state = SessionState.CANCELLED
    session.frames.clear()
    ws = _CapturingWS()

    with caplog.at_level("INFO", logger="liveness"):
        _infer_with(ws, session)

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "inference_discarded" in m and "reason=cancelled_before_start" in m
        for m in messages
    ), messages
    assert session.state.value == "cancelled", "the cancel must not be overwritten"
    assert session.result_delivered is False
    assert ws.sent == [], "nothing may be sent to a session that already cancelled"
    assert not any(m.startswith("inference_started") for m in messages)
    assert not any(m.startswith("inference_error") for m in messages)


def test_terminal_failure_while_the_model_runs_suppresses_the_result(
    fastapi_client, monkeypatch, caplog
):
    """The receive loop stays open during inference, so errors can land mid-run.

    A session that has already failed must not then be answered with a
    `result`: the client has the terminal error, and the two would contradict
    each other. The state must stay `failed` rather than be overwritten.
    """
    from app import main
    from app.sessions import SessionError, SessionState

    monkeypatch.setattr(main, "INFERENCE_KEEPALIVE_SECONDS", 0.05)
    session = _populated_session(fastapi_client, prediction="live", score=0.05)
    stub = main.PIPELINE

    def fail_midway(_frames):
        session.state = SessionState.FAILED
        session.error = SessionError("frame_too_large", "Frame exceeds size limit")

    stub.side_effect = fail_midway
    ws = _CapturingWS()

    with caplog.at_level("INFO", logger="liveness"):
        _infer_with(ws, session)

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "inference_discarded" in m and "reason=session_failed_during_inference" in m
        for m in messages
    ), messages
    assert session.state.value == "failed", "the failure must not be overwritten"
    assert session.result_delivered is False
    assert not any(json.loads(s)["type"] == "result" for s in ws.sent), ws.sent
    assert not any(m.startswith("inference_completed") for m in messages)


def test_uvicorn_access_log_redacts_the_token_but_keeps_the_session_path():
    """uvicorn logs the raw upgrade path; a capability token must not survive.

    `"WebSocket /ws/sessions/<id>?token=<secret>" [accepted]` is written by
    uvicorn itself, before any application code runs, so the redaction has to
    be a logging filter installed on uvicorn's own loggers.
    """
    import logging

    from app.main import _RedactTokenFilter

    record = logging.LogRecord(
        name="uvicorn.error",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='127.0.0.1:53114 - "WebSocket /ws/sessions/%s?token=%s HTTP/1.1" [accepted]',
        args=("abc123def456", "SUPERSECRETTOKENVALUE"),
        exc_info=None,
    )

    assert _RedactTokenFilter().filter(record) is True

    text = record.getMessage()
    assert "SUPERSECRETTOKENVALUE" not in text, text
    assert "token=<redacted>" in text, text
    assert "abc123def456" in text, "the session path stays correlatable for debugging"

    # An ordinary line must pass through untouched.
    plain = logging.LogRecord(
        name="uvicorn.error", level=logging.INFO, pathname=__file__, lineno=1,
        msg="Uvicorn running on 0.0.0.0:8000", args=(), exc_info=None,
    )
    assert _RedactTokenFilter().filter(plain) is True
    assert plain.getMessage() == "Uvicorn running on 0.0.0.0:8000"
