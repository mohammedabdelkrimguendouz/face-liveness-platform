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


async def _send(ws: WebSocket, payload: Dict[str, object]) -> None:
    import json
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:  # noqa: BLE001 - client likely gone
        logger.debug("Failed to send WS payload", exc_info=True)


@app.websocket("/ws/sessions/{session_id}")
async def session_socket(ws: WebSocket, session_id: str, token: str = Query(...)) -> None:
    try:
        session = SESSIONS.get(session_id)
    except SessionError:
        await ws.close(code=4404)
        return

    if token != session.ws_token:
        await ws.close(code=4401)
        return

    if session.state not in (SessionState.CREATED, SessionState.STREAMING):
        await ws.close(code=4409)
        return

    await ws.accept()
    session.state = SessionState.STREAMING
    await _send(ws, {
        "type": "ready",
        "session_id": session.session_id,
        "required_frames": session.num_frames,
        "max_frame_bytes": session.max_frame_bytes,
    })

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                await _handle_control(ws, session, message["text"])
                if session.state in (SessionState.COMPLETED, SessionState.FAILED,
                                     SessionState.CANCELLED):
                    break
                continue

            data = message.get("bytes")
            if data is None:
                continue

            await _handle_frame(ws, session, data)

            if session.is_complete:
                await _run_inference(ws, session)
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id[:8])
    except Exception:  # noqa: BLE001
        logger.exception("WebSocket error for session %s", session_id[:8])
        await _send(ws, {"type": "error", "error": "internal_error", "message": "Unexpected server error"})
    finally:
        # Always release buffered frames unless we are still inferring.
        if session.state in (SessionState.CREATED, SessionState.STREAMING, SessionState.CANCELLED):
            session.frames.clear()
            if session.state != SessionState.CANCELLED:
                session.state = SessionState.CANCELLED
            SESSIONS.remove(session.session_id)


async def _handle_control(ws: WebSocket, session, raw: str) -> None:
    import json
    try:
        control = json.loads(raw)
    except ValueError:
        await _send(ws, {"type": "error", "error": "bad_message", "message": "Control message must be valid JSON"})
        return

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
            })
        elif session.received_count > session.num_frames:
            session.state = SessionState.FAILED
            session.error = SessionError("too_many_frames", "Received more frames than required")
            await _send(ws, {"type": "error", "error": "too_many_frames", "message": "Too many frames received"})
        else:
            await _run_inference(ws, session)
    elif ctype == "cancel":
        session.state = SessionState.CANCELLED
        session.frames.clear()
        await _send(ws, {"type": "cancelled"})
    else:
        await _send(ws, {"type": "error", "error": "bad_message", "message": f"Unknown control type: {ctype!r}"})


async def _handle_frame(ws: WebSocket, session, data: bytes) -> None:
    if len(data) < SEQ_HEADER.size:
        await _send(ws, {"type": "error", "error": "malformed_frame", "message": "Frame too short"})
        return

    (index,) = SEQ_HEADER.unpack_from(data, 0)
    payload = data[SEQ_HEADER.size:]

    if len(payload) == 0:
        await _send(ws, {"type": "error", "error": "malformed_frame", "message": "Empty image payload"})
        return
    if len(payload) > session.max_frame_bytes:
        session.state = SessionState.FAILED
        session.error = SessionError("frame_too_large", "Frame exceeds size limit")
        await _send(ws, {"type": "error", "error": "frame_too_large", "message": "Frame exceeds size limit"})
        return

    # All state mutation happens under the lock; sends happen after release
    # so we never hold the lock across an event-loop yield.
    error_payload = None
    ack_payload = None
    with session.lock:
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
            # Progress ack (throttled to every 10 frames plus completion).
            if received % 10 == 0 or received == session.num_frames:
                ack_payload = {"type": "ack", "received": received, "required": session.num_frames}

    if error_payload is not None:
        await _send(ws, error_payload)
        return
    if ack_payload is not None:
        await _send(ws, ack_payload)


async def _run_inference(ws: WebSocket, session) -> None:
    if session.inference_started:
        await _send(ws, {"type": "error", "error": "already_running", "message": "Inference already started"})
        return
    session.inference_started = True
    session.state = SessionState.INFERRING
    await _send(ws, {"type": "processing", "received": session.received_count})

    frames = session.ordered_frames()
    try:
        async with INFER_SEMAPHORE:
            result = await asyncio.to_thread(PIPELINE.predict, frames)
    except ValueError as exc:
        session.state = SessionState.FAILED
        session.error = SessionError("bad_frames", str(exc))
        await _send(ws, {"type": "error", "error": "bad_frames", "message": str(exc)})
        return
    except Exception:  # noqa: BLE001
        logger.exception("Inference failed for session %s", session.session_id[:8])
        session.state = SessionState.FAILED
        session.error = SessionError("inference_failed", "Inference failed")
        await _send(ws, {"type": "error", "error": "inference_failed", "message": "Inference failed"})
        return

    session.state = SessionState.COMPLETED
    await _send(ws, {
        "type": "result",
        "session_id": session.session_id,
        "prediction": result.prediction,
        "anomaly_score": round(result.anomaly_score, 6),
        "reconstruction_error": round(result.reconstruction_error, 6),
        "threshold": result.threshold,
        "frames_processed": result.frames_processed,
        "inference_ms": round(result.inference_ms, 1),
        "score_semantics": "anomaly_score = 1 - exp(-MSE); not a calibrated probability",
    })
    # Release the buffer immediately after the result is delivered.
    session.frames.clear()
