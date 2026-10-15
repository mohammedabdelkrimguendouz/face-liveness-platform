"""Face Liveness API.

FastAPI service that receives sequential camera frames over WebSocket, assembles
the ordered sequence, runs the existing (verified) ML pipeline, and returns the
authoritative prediction.

Endpoints
---------
GET  /health                  -> process liveness (no model dependency)
GET  /ready                   -> model readiness / failure detail
GET  /                        -> service metadata
POST /sessions                -> create a verification session
GET  /sessions/{id}           -> session status (diagnostics)
WS   /ws/sessions/{id}?token= -> frame stream (binary frames + JSON control)
"""

from __future__ import annotations

import asyncio
import logging
import re
import struct
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .ml import LivenessPipeline, PipelineConfig
from .sessions import SessionError, SessionManager, SessionState

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("liveness")

_TOKEN_IN_PATH_RE = re.compile(r"(?i)([?&]token=)[^&\"'\s]+")


class _RedactTokenFilter(logging.Filter):
    """Strip capability tokens from uvicorn's own log lines.

    uvicorn logs the full path of a WebSocket upgrade —
    ``"WebSocket /ws/sessions/<id>?token=<secret>" [accepted]`` — which puts a
    live capability token into the platform's log stream. The application's
    records already use an 8-character session prefix; this brings the
    server's access log in line with that.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _TOKEN_IN_PATH_RE.sub(r"\1<redacted>", message)
        if redacted != message:
            # Re-format eagerly: leaving args in place would put the token
            # back when the record is finally rendered.
            record.msg = redacted
            record.args = ()
        return True


for _uvicorn_logger in ("uvicorn.error", "uvicorn.access"):
    logging.getLogger(_uvicorn_logger).addFilter(_RedactTokenFilter())

SESSIONS = SessionManager(max_sessions=settings.max_sessions, ttl_seconds=settings.session_ttl_seconds)
PIPELINE = LivenessPipeline(
    models_dir=settings.models_dir,
    device=settings.device,
    config=PipelineConfig(batch_size=settings.batch_size, num_frames=settings.num_frames),
)
INFER_SEMAPHORE = asyncio.Semaphore(settings.max_inferences)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Face Liveness API")
    logger.info("Models dir: %s", settings.models_dir)
    try:
        PIPELINE.load()
        logger.info("Model load complete")
    except Exception:  # noqa: BLE001 - /ready reports the failure
        logger.exception("Model load failed; /ready will report not-ready")

    stop = asyncio.Event()

    async def _janitor():
        while not stop.is_set():
            try:
                SESSIONS.evict_expired()
            except Exception:  # noqa: BLE001
                logger.exception("Session eviction error")
            try:
                await asyncio.wait_for(stop.wait(), timeout=30)
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(_janitor())
    yield
    stop.set()
    task.cancel()


app = FastAPI(title="Face Liveness API", version="1.0.0", lifespan=lifespan)

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SessionCreateResponse(BaseModel):
    session_id: str
    ws_url: str
    ws_token: str
    num_frames: int
    max_frame_bytes: int
    expires_in_seconds: int


class SessionStatusResponse(BaseModel):
    session_id: str
    state: str
    received_frames: int
    required_frames: int
    remaining_ttl_seconds: float


class ErrorResponse(BaseModel):
    error: str
    message: str


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> Dict[str, object]:
    ok = PIPELINE.is_ready
    body: Dict[str, object] = {
        "ready": ok,
        "device": str(PIPELINE.device),
        "num_frames": PIPELINE.config.num_frames,
        "active_sessions": SESSIONS.active_count(),
    }
    if not ok:
        body["error"] = PIPELINE.load_error or "model not loaded"
    if not ok:
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/")
async def root() -> Dict[str, object]:
    return {
        "service": "face-liveness-api",
        "version": "1.0.0",
        "num_frames": PIPELINE.config.num_frames,
        "endpoints": ["/health", "/ready", "/sessions", "/ws/sessions/{id}"],
    }


@app.post("/sessions", response_model=SessionCreateResponse, status_code=201)
async def create_session() -> SessionCreateResponse:
    try:
        session = SESSIONS.create(
            num_frames=PIPELINE.config.num_frames,
            max_frame_bytes=settings.max_frame_bytes,
            max_total_frames=settings.max_frames_per_session,
        )
    except SessionError as exc:
        raise HTTPException(status_code=429, detail={"error": exc.code, "message": exc.message})

    return SessionCreateResponse(
        session_id=session.session_id,
        # Relative on purpose — see _ws_path().
        ws_url=_ws_path(session.session_id),
        ws_token=session.ws_token,
        num_frames=session.num_frames,
        max_frame_bytes=session.max_frame_bytes,
        expires_in_seconds=session.ttl_seconds,
    )


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def session_status(session_id: str) -> SessionStatusResponse:
    try:
        session = SESSIONS.get(session_id)
    except SessionError as exc:
        raise HTTPException(status_code=404, detail={"error": exc.code, "message": exc.message})
    return SessionStatusResponse(
        session_id=session.session_id,
        state=session.state.value,
        received_frames=session.received_count,
        required_frames=session.num_frames,
        remaining_ttl_seconds=round(session.remaining_ttl, 1),
    )


# ---------------------------------------------------------------------------
# WebSocket protocol
# ---------------------------------------------------------------------------
SEQ_HEADER = struct.Struct(">I")  # 4-byte big-endian sequence index


def _ws_path(session_id: str) -> str:
    """WebSocket path for a session, relative to the API's own origin.

    Deliberately **relative**: the client resolves it against the ``apiUrl``
    it was configured with. That is what makes one server build work in every
    environment — ``ws://localhost:8000/ws/sessions/...`` in local
    development, ``wss://<your-api>.onrender.com/ws/sessions/...`` in
    production — without reading ``X-Forwarded-*`` headers from a proxy.

    Returning an absolute ``ws://localhost/...`` here instead hard-codes the
    host and makes every deployed browser try to open its own machine's
    port 80.
    """
    return f"/ws/sessions/{session_id}"


async def _send(
    ws: WebSocket,
    payload: Dict[str, object],
    lock: Optional[asyncio.Lock] = None,
    session_id: Optional[str] = None,
) -> bool:
    """Send one JSON control frame. Returns True if it reached the transport.

    Failures are logged at WARNING with a traceback. An earlier revision
    logged them at DEBUG, which made a dropped final result completely
    invisible at the default ``LOG_LEVEL=info`` — the client would then report
    only "Connection closed (code N)" and the server log would show nothing
    between the inference line and the disconnect.

    ``lock`` serializes writes once inference runs as a background task:
    Starlette's WebSocket offers no concurrency guarantee, and keepalives then
    race the receive loop's acks and the final result.

    ``session_id`` is the 8-character prefix, so a traceback about a socket
    failure can be tied to the session it belonged to. Only the message
    ``type`` and that prefix are logged: payloads may carry model scores, and
    never tokens or frame data.
    """
    import json
    message_type = payload.get("type", "?") if isinstance(payload, dict) else "?"
    try:
        body = json.dumps(payload)
        if lock is None:
            await ws.send_text(body)
        else:
            async with lock:
                await ws.send_text(body)
        return True
    except Exception:  # noqa: BLE001 - most often a client that already left
        logger.warning(
            "ws_send_failed type=%s session=%s",
            message_type,
            session_id[:8] if session_id else "-",
            exc_info=True,
        )
        return False


# Real data frames emitted while the model works. See _keepalive().
INFERENCE_KEEPALIVE_SECONDS = 5.0


async def _keepalive(
    ws: WebSocket,
    session,
    lock: asyncio.Lock,
    stop: asyncio.Event,
    started_at: float,
) -> None:
    """Keep the connection demonstrably alive for the whole inference.

    Uvicorn already sends protocol-level pings every 20 s, but those are
    invisible to a proxy that only counts *data* frames, so a connection that
    carries nothing for the 40-50 s a CPU inference takes is still reaped — and
    the client cannot otherwise tell "the model is still working" from "the
    link is dead". Real frames address both.

    Emission stops when inference is no longer running (cancel, error) or on
    the first failed send: a socket that cannot take a keepalive cannot take
    the result either, and a known-dead socket must not keep producing
    tracebacks at WARNING for the rest of the model's run.
    """
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=INFERENCE_KEEPALIVE_SECONDS)
        except asyncio.TimeoutError:
            pass
        else:
            return
        if session.client_gone_at is not None or session.state is not SessionState.INFERRING:
            # Either the handler already saw the peer leave, or inference is no
            # longer running (cancel/error). Sending into a known-dead socket
            # would only add a traceback a second later without telling anyone
            # anything the disconnect line has not already said.
            return
        delivered = await _send(ws, {
            "type": "progress",
            "stage": "inferring",
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 1),
        }, lock=lock, session_id=session.session_id)
        if not delivered:
            return


async def _reject(ws: WebSocket, session_id: str, reason: str, code: int) -> None:
    """Refuse a handshake before ``accept()`` and record why.

    Closing a WebSocket that was never accepted makes the ASGI server answer
    the upgrade request with ``HTTP 403 Forbidden``, which is the only thing a
    proxy/load-balancer can see for a failed handshake — so without this log
    the rejection reason is invisible in production.

    ``session_id`` is truncated to the same 8-character prefix used elsewhere
    so a line can be correlated with the uvicorn access log without emitting
    a full identifier. The capability token is never logged, by value or by
    length.
    """
    logger.warning(
        "ws_rejected session=%s reason=%s close_code=%d",
        session_id[:8],
        reason,
        code,
    )
    await ws.close(code=code)


def _note_gone(session, via: str, inference_running: bool, accepted_at: float) -> None:
    """Record that the peer is gone, whichever path first reveals it.

    The departure surfaces either as a ``websocket.disconnect`` message or as
    an exception thrown out of a receive/send. Both must leave the same
    ``ws_client_gone`` line, because ``inference_running`` plus ``elapsed_ms``
    is what separates "the client left early" from "the model was too slow" —
    the distinction the previous revision could not make, since it only
    discovered the loss when the result failed to leave.
    """
    now = time.perf_counter()
    session.client_gone_at = now
    logger.info(
        "ws_client_gone session=%s state=%s via=%s inference_running=%s "
        "elapsed_ms=%.0f frames_received=%d last_frame_index=%s",
        session.session_id[:8],
        session.state.value,
        via,
        inference_running,
        (now - accepted_at) * 1000.0,
        session.frames_received,
        session.last_frame_index,
    )


@app.websocket("/ws/sessions/{session_id}")
async def session_socket(
    ws: WebSocket,
    session_id: str,
    # Optional rather than Query(...): a required query parameter fails
    # FastAPI's dependency validation before this function runs, so the
    # rejection would bypass _reject() and leave no diagnostic. Validating
    # here instead keeps every refusal path (and its log) in one place while
    # still closing before accept().
    token: Optional[str] = Query(default=None),
) -> None:
    try:
        session = SESSIONS.get(session_id)
    except SessionError:
        await _reject(ws, session_id, "session_not_found_or_expired", 4404)
        return

    if not token:
        await _reject(ws, session_id, "token_missing", 4401)
        return

    if token != session.ws_token:
        await _reject(ws, session_id, "token_invalid", 4401)
        return

    if session.state not in (SessionState.CREATED, SessionState.STREAMING):
        await _reject(ws, session_id, f"invalid_state:{session.state.value}", 4409)
        return

    logger.info("ws_accepted session=%s", session_id[:8])
    accepted_at = time.perf_counter()
    session.accepted_at = accepted_at
    await ws.accept()
    session.state = SessionState.STREAMING

    # Every write for this connection goes through one lock. Inference now runs
    # as a background task, so its keepalives and the final result race the
    # receive loop's acks; Starlette's WebSocket offers no concurrency
    # guarantee of its own.
    send_lock = asyncio.Lock()
    await _send(ws, {
        "type": "ready",
        "session_id": session.session_id,
        "required_frames": session.num_frames,
        "max_frame_bytes": session.max_frame_bytes,
    }, lock=send_lock)

    inference_task: Optional[asyncio.Task] = None
    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                _note_gone(
                    session,
                    via="message",
                    inference_running=inference_task is not None and not inference_task.done(),
                    accepted_at=accepted_at,
                )
                break

            if message.get("text") is not None:
                wants_inference = await _handle_control(ws, session, message["text"], send_lock)
                if wants_inference and inference_task is None:
                    inference_task = asyncio.create_task(
                        _run_inference(ws, session, send_lock, accepted_at)
                    )
                if session.state in (SessionState.COMPLETED, SessionState.FAILED,
                                     SessionState.CANCELLED):
                    break
                continue

            data = message.get("bytes")
            if data is None:
                continue

            await _handle_frame(ws, session, data, send_lock)

            if session.is_complete and inference_task is None:
                logger.info(
                    "inference_scheduled session=%s frames_received=%d last_frame_index=%s",
                    session_id[:8],
                    session.frames_received,
                    session.last_frame_index,
                )
                inference_task = asyncio.create_task(
                    _run_inference(ws, session, send_lock, accepted_at)
                )
            # Deliberately NOT a `break`. The loop stays open for the whole of
            # the inference so that a disconnect is observed the instant it
            # happens rather than tens of seconds later, and so `cancel` still
            # reaches the server while the model is busy.

    except WebSocketDisconnect:
        _note_gone(
            session,
            via="exception",
            inference_running=inference_task is not None and not inference_task.done(),
            accepted_at=accepted_at,
        )
    except Exception:  # noqa: BLE001
        # The traceback is essential here. If an exception escapes this
        # handler, the websockets layer answers the client with close code
        # 1011 and logs only "connection handler failed" — without this
        # record the server log cannot explain the client's failure.
        logger.exception("websocket_error session=%s", session_id[:8])
        await _send(ws, {"type": "error", "error": "internal_error", "message": "Unexpected server error"},
                    lock=send_lock)
    finally:
        # Deliberately does NOT stamp `client_gone_at`: this fires whenever the
        # handler returns, including after a normal completion or a cancel, and
        # overwriting a real departure timestamp with the exit time would both
        # misreport the timeline and make a cancel look like a disconnect.
        if inference_task is not None and not inference_task.done():
            # Let the model finish and release its own semaphore rather than
            # cancelling: `to_thread` cannot be interrupted, so a cancel would
            # abandon the thread while still freeing the slot, letting a second
            # inference start on a CPU that is already saturated. The task
            # finds the socket gone and reports `result_send_failed` itself.
            reason = (
                "inference_finished_after_client_left"
                if session.client_gone_at is not None
                else "inference_finished_after_cancel"
            )
            inference_task.add_done_callback(
                lambda _task, _reason=reason: _finalize(session, _reason)
            )
        else:
            _finalize(session, "connection_closed")


def _finalize(session, reason: str) -> None:
    """Release buffered frames and report the terminal state, exactly once.

    Reached both from the connection teardown and from a still-running
    inference task, so it must be idempotent. Incomplete sessions are dropped
    immediately; completed and failed ones stay queryable via
    ``GET /sessions/{id}`` until their TTL, exactly as before.
    """
    if session.finalized:
        return
    session.finalized = True
    if session.state in (SessionState.CREATED, SessionState.STREAMING, SessionState.CANCELLED):
        session.frames.clear()
        if session.state is not SessionState.CANCELLED:
            session.state = SessionState.CANCELLED
        SESSIONS.remove(session.session_id)

    left_after_ms = -1.0
    if session.accepted_at is not None and session.client_gone_at is not None:
        left_after_ms = (session.client_gone_at - session.accepted_at) * 1000.0
    logger.info(
        "session_finalized session=%s state=%s delivered=%s reason=%s client_left_after_ms=%.0f",
        session.session_id[:8],
        session.state.value,
        session.result_delivered,
        reason,
        left_after_ms,
    )


async def _handle_control(ws: WebSocket, session, raw: str, lock: asyncio.Lock) -> bool:
    """Handle one JSON control frame.

    Returns True when the caller should schedule inference. Inference is not
    run here: the receive loop must stay responsive for the whole time the
    model is busy.
    """
    import json
    try:
        control = json.loads(raw)
    except ValueError:
        await _send(ws, {"type": "error", "error": "bad_message", "message": "Control message must be valid JSON"}, lock=lock)
        return False

    ctype = control.get("type")

    if ctype == "complete":
        with session.lock:
            session.client_complete = True
            missing = [i for i in range(session.num_frames) if i not in session.frames]
        if missing:
            session.state = SessionState.FAILED
            session.error = SessionError("incomplete_sequence", f"Missing {len(missing)} frame(s)")
            await _send(ws, {
                "type": "error",
                "error": "incomplete_sequence",
                "message": f"Sequence incomplete: {len(missing)} missing frame(s)",
                "missing": len(missing),
            }, lock=lock)
        elif session.received_count > session.num_frames:
            session.state = SessionState.FAILED
            session.error = SessionError("too_many_frames", "Received more frames than required")
            await _send(ws, {"type": "error", "error": "too_many_frames", "message": "Too many frames received"}, lock=lock)
        else:
            return not session.inference_started
        return False
    elif ctype == "cancel":
        session.state = SessionState.CANCELLED
        session.frames.clear()
        logger.info("ws_cancel_requested session=%s", session.session_id[:8])
        await _send(ws, {"type": "cancelled"}, lock=lock)
    else:
        await _send(ws, {"type": "error", "error": "bad_message", "message": f"Unknown control type: {ctype!r}"}, lock=lock)
    return False


async def _handle_frame(ws: WebSocket, session, data: bytes, lock: asyncio.Lock) -> None:
    if len(data) < SEQ_HEADER.size:
        await _send(ws, {"type": "error", "error": "malformed_frame", "message": "Frame too short"}, lock=lock)
        return

    (index,) = SEQ_HEADER.unpack_from(data, 0)
    payload = data[SEQ_HEADER.size:]

    if len(payload) == 0:
        await _send(ws, {"type": "error", "error": "malformed_frame", "message": "Empty image payload"}, lock=lock)
        return
    if len(payload) > session.max_frame_bytes:
        session.state = SessionState.FAILED
        session.error = SessionError("frame_too_large", "Frame exceeds size limit")
        await _send(ws, {"type": "error", "error": "frame_too_large", "message": "Frame exceeds size limit"}, lock=lock)
        return

    # All state mutation happens under the lock; sends happen after release
    # so we never hold the lock across an event-loop yield.
    error_payload = None
    ack_payload = None
    with session.lock:
        session.last_frame_index = index
        if index >= session.num_frames:
            error_payload = {"type": "error", "error": "bad_index", "message": f"Sequence index {index} out of range"}
        elif index in session.frames:
            # Duplicate: reject but do not corrupt state.
            error_payload = {"type": "error", "error": "duplicate_frame", "message": f"Frame {index} already received"}
        elif len(session.frames) >= session.max_total_frames:
            session.state = SessionState.FAILED
            session.error = SessionError("too_many_frames", "Session frame limit exceeded")
            error_payload = {"type": "error", "error": "too_many_frames", "message": "Session frame limit exceeded"}
        else:
            session.frames[index] = payload
            received = len(session.frames)
            session.frames_received = received
            # Progress ack (throttled to every 10 frames plus completion).
            if received % 10 == 0 or received == session.num_frames:
                ack_payload = {"type": "ack", "received": received, "required": session.num_frames}

    if error_payload is not None:
        await _send(ws, error_payload, lock=lock)
        return
    if ack_payload is not None:
        await _send(ws, ack_payload, lock=lock)


async def _run_inference(
    ws: WebSocket,
    session,
    lock: asyncio.Lock,
    accepted_at: Optional[float] = None,
) -> None:
    """Run the model and deliver the result.

    Scheduled as a background task instead of being awaited inline. A CPU
    inference takes tens of seconds, and blocking the receive loop for that
    long means the server cannot notice a disconnect until the model finishes,
    cannot service `cancel`, and sends nothing at all in between — which is
    exactly the silence an idle-timeout proxy reaps a connection for.
    """
    if session.inference_started:
        await _send(
            ws,
            {"type": "error", "error": "already_running", "message": "Inference already started"},
            lock=lock,
            session_id=session.session_id,
        )
        return
    session.inference_started = True
    if session.state is SessionState.CANCELLED:
        # `create_task` only schedules; the handler can reach a `cancel` on its
        # next iteration before this coroutine ever runs. The client has
        # already had its `cancelled` frame, so there is nothing to say and
        # certainly no verdict to compute.
        logger.info(
            "inference_discarded session=%s reason=cancelled_before_start",
            session.session_id[:8],
        )
        return
    session.state = SessionState.INFERRING
    started = time.perf_counter()
    logger.info(
        "inference_started session=%s frames_received=%d last_frame_index=%s queued_ms=%.0f",
        session.session_id[:8],
        session.frames_received,
        session.last_frame_index,
        (started - accepted_at) * 1000.0 if accepted_at is not None else -1.0,
    )
    # Capture the frames BEFORE the first await below: a `cancel` that lands
    # during that await clears `session.frames`, and the model must not be
    # handed a truncated sequence for it.
    frames = session.ordered_frames()
    await _send(
        ws,
        {"type": "processing", "received": session.received_count},
        lock=lock,
        session_id=session.session_id,
    )

    stop = asyncio.Event()
    keeper = asyncio.create_task(_keepalive(ws, session, lock, stop, started))
    try:
        async with INFER_SEMAPHORE:
            result = await asyncio.to_thread(PIPELINE.predict, frames)
    except ValueError as exc:
        session.state = SessionState.FAILED
        session.error = SessionError("bad_frames", str(exc))
        await _send(ws, {"type": "error", "error": "bad_frames", "message": str(exc)},
                    lock=lock, session_id=session.session_id)
        return
    except Exception:  # noqa: BLE001
        logger.exception("inference_error session=%s", session.session_id[:8])
        session.state = SessionState.FAILED
        session.error = SessionError("inference_failed", "Inference failed")
        await _send(ws, {"type": "error", "error": "inference_failed", "message": "Inference failed"},
                    lock=lock, session_id=session.session_id)
        return
    finally:
        stop.set()
        try:
            await keeper
        except asyncio.CancelledError:  # pragma: no cover - interpreter shutdown
            pass

    if session.state in (SessionState.CANCELLED, SessionState.FAILED):
        # The receive loop stays open while the model works, so a terminal
        # answer can now land mid-run — something the old blocking handler
        # could never observe. The client already has that terminal message
        # (`cancelled`, or a frame error); a late `result` would contradict it.
        logger.info(
            "inference_discarded session=%s reason=%s",
            session.session_id[:8],
            "cancelled_during_inference"
            if session.state is SessionState.CANCELLED
            else "session_failed_during_inference",
        )
        session.frames.clear()
        return

    session.state = SessionState.COMPLETED
    logger.info(
        "inference_completed session=%s prediction=%s score=%.6f frames=%d ms=%.0f",
        session.session_id[:8],
        result.prediction,
        result.anomaly_score,
        result.frames_processed,
        (time.perf_counter() - started) * 1000.0,
    )

    # A `spoof` prediction is a valid model answer, not a transport failure.
    # The two must be distinguishable in the logs, so the send is reported
    # separately from the inference that preceded it.
    logger.info("result_send_started session=%s", session.session_id[:8])
    delivered = await _send(ws, {
        "type": "result",
        "session_id": session.session_id,
        "prediction": result.prediction,
        "anomaly_score": round(result.anomaly_score, 6),
        "reconstruction_error": round(result.reconstruction_error, 6),
        "threshold": result.threshold,
        "frames_processed": result.frames_processed,
        "inference_ms": round(result.inference_ms, 1),
        "score_semantics": "anomaly_score = 1 - exp(-MSE); not a calibrated probability",
    }, lock=lock, session_id=session.session_id)

    if delivered:
        logger.info("result_send_succeeded session=%s", session.session_id[:8])
    else:
        # The model already answered; only the delivery failed. Reporting this
        # separately is what lets an operator tell "retry the request" apart
        # from "the model is broken".
        logger.warning(
            "result_send_failed session=%s stage=after_inference client_left_after_ms=%.0f",
            session.session_id[:8],
            (session.client_gone_at - session.accepted_at) * 1000.0
            if session.client_gone_at is not None and session.accepted_at is not None
            else -1.0,
        )

    # Release the buffer once the result has been handed to the transport.
    session.frames.clear()
    session.result_delivered = delivered
