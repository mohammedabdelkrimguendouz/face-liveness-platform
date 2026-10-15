"""Real-transport WebSocket integration tests.

These run the actual FastAPI app under a real uvicorn server (the production
``websockets`` stack) and talk to it with real socket clients, because the
failure they pin down cannot exist under Starlette's ``TestClient``: it lives
in the ASGI server's own connection handling.

Production evidence being reproduced here
-----------------------------------------
Render served 150 frames, inference completed in ~43 s, then

    result_send_started
    ws_send_failed ... ClientDisconnected
    result_send_failed stage=after_inference
    session_finalized state=completed delivered=False

while the browser showed ``Connection closed (code 1011)``. Code 1011 is
never produced by the application: in this stack (uvicorn + ``websockets``)
the only component that can put a Close frame with 1011 on the wire is
``websockets`` itself, via ``fail_connection(1011)``, and the variant that
logs nothing at ``--log-level info`` is the keepalive ping timeout
(``fail_connection(1011, "keepalive ping timeout")`` — a DEBUG-only line).

The tests below isolate that mechanism on the production ping policy
(interval 20 s / timeout 20 s):

* a peer that answers pings (every real browser does) survives a 46 s
  inference and receives the result;
* a peer that does NOT answer pings — the situation production found itself
  in — is killed with ``Close(1011, "keepalive ping timeout")`` mid-inference
  and the server then reports the exact production log sequence;
* raising only ``ws_ping_timeout`` (the setting Render's start command now
  pins) keeps the silent peer alive through the same inference.

Deterministic mocked inference is used (``StubPipeline``); no model runs here.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import socket
import struct
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
import websockets
from uvicorn.config import Config
from uvicorn.server import Server

SEQ = struct.Struct(">I")
REQUIRED = 150
HOST = "127.0.0.1"

# The production keepalive policy: uvicorn's defaults, which the Render start
# command does not override today. Kept explicit here so a change to either
# side shows up as a failing test rather than as a production outage.
PRODUCTION_PING_INTERVAL = 20.0
PRODUCTION_PING_TIMEOUT = 20.0


# ---------------------------------------------------------------------------
# Controllable pipeline (deterministic, thread-safe, no ML)
# ---------------------------------------------------------------------------
class StubPipeline:
    def __init__(self) -> None:
        self.is_ready = True
        self.load_error = None
        self.device = "cpu"
        self.config = SimpleNamespace(num_frames=REQUIRED)
        self.delay = 0.0
        self.prediction = "spoof"
        self.score = 0.9226
        self.calls: list[list[bytes]] = []
        self._lock = threading.Lock()

    def load(self) -> None:  # called by the app lifespan if it ever runs
        return None

    def predict(self, frames):
        delay = self.delay
        if delay:
            time.sleep(delay)
        with self._lock:
            self.calls.append(list(frames))
        return SimpleNamespace(
            prediction=self.prediction,
            anomaly_score=self.score,
            reconstruction_error=0.01,
            threshold=0.5,
            frames_processed=len(frames),
            inference_ms=delay * 1000.0,
            mean_frame_error=0.01,
        )


class _ThreadSemaphore:
    """Loop-agnostic stand-in for asyncio.Semaphore.

    The tests start two uvicorn servers, each with its own event loop, while
    ``app.main.INFER_SEMAPHORE`` is a single module-level object; an asyncio
    Semaphore remembers the loop it was first awaited on and refuses to be
    used from another one. A threading semaphore keeps the tests honest
    without touching application code for their benefit.
    """

    def __init__(self, value: int = 4) -> None:
        self._sem = threading.Semaphore(value)

    async def __aenter__(self):
        self._sem.acquire()
        return self

    async def __aexit__(self, *exc):
        self._sem.release()
        return False


@contextlib.contextmanager
def real_transport_server(
    ping_interval: float = PRODUCTION_PING_INTERVAL,
    ping_timeout: float = PRODUCTION_PING_TIMEOUT,
):
    """Run the real app under a real uvicorn server on a free port."""
    from app import main

    previous_pipeline = main.PIPELINE
    previous_semaphore = main.INFER_SEMAPHORE
    stub = StubPipeline()
    main.PIPELINE = stub
    main.INFER_SEMAPHORE = _ThreadSemaphore(4)

    with socket.socket() as probe:
        probe.bind((HOST, 0))
        port = probe.getsockname()[1]

    config = Config(
        main.app,
        host=HOST,
        port=port,
        log_level="warning",
        lifespan="off",  # no model load, no janitor: the stub is the model
        ws_ping_interval=ping_interval,
        ws_ping_timeout=ping_timeout,
    )
    server = Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline or not thread.is_alive():
            raise RuntimeError("uvicorn test server failed to start")
        time.sleep(0.02)
    try:
        yield SimpleNamespace(host=HOST, port=port, stub=stub)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        main.PIPELINE = previous_pipeline
        main.INFER_SEMAPHORE = previous_semaphore


def _create_session(port: int) -> tuple[str, str]:
    res = httpx.post(f"http://{HOST}:{port}/sessions", timeout=10)
    assert res.status_code == 201, res.text
    body = res.json()
    return body["session_id"], body["ws_token"]


def _frame(index: int) -> bytes:
    # The stub pipeline never decodes, so a few bytes stand in for a JPEG.
    return SEQ.pack(index) + b"\xff\xd8\xff\xe0stub"


async def _wait_for_log(marker: str, caplog, timeout: float) -> str:
    """Block until a `liveness` record containing `marker` appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for record in caplog.records:
            message = record.getMessage()
            if marker in message:
                return message
        await asyncio.sleep(0.05)
    raise AssertionError(f"log line containing {marker!r} did not arrive within {timeout}s")


# ---------------------------------------------------------------------------
# Minimal raw WebSocket peer that deliberately never answers Pings
#
# Browsers answer Ping with Pong at the network layer, but the production
# failure only occurs when that answer does not arrive. To reproduce that
# deterministically the peer below performs the RFC 6455 handshake by hand,
# frames its own messages (client frames must be masked), and swallows Pings
# without replying — exactly the vantage point of a connection whose pongs are
# being lost upstream.
# ---------------------------------------------------------------------------
class SilentRawPeer:
    def __init__(self, reader, writer, leftover: bytes) -> None:
        self.reader = reader
        self.writer = writer
        self._buffer = leftover
        self.pings_seen = 0

    @classmethod
    async def connect(cls, host: str, port: int, path: str) -> "SilentRawPeer":
        reader, writer = await asyncio.open_connection(host, port)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        writer.write(request.encode("ascii"))
        await writer.drain()

        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await reader.read(1024)
            if not chunk:
                raise ConnectionError("peer closed during the WebSocket handshake")
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0]
        if b"101" not in status:
            raise ConnectionError(f"handshake rejected: {status!r}")
        return cls(reader, writer, rest)

    async def send_text(self, text: str) -> None:
        await self._send_frame(0x1, text.encode("utf-8"))

    async def send_bytes(self, payload: bytes) -> None:
        await self._send_frame(0x2, payload)

    async def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", length)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.writer.write(bytes(header) + mask + masked)
        await self.writer.drain()

    async def _read_exactly(self, n: int) -> bytes:
        while len(self._buffer) < n:
            chunk = await self.reader.read(65536)
            if not chunk:
                raise ConnectionError("transport closed")
            self._buffer += chunk
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    async def _read_frame(self) -> tuple[int, bytes]:
        head = await self._read_exactly(2)
        opcode = head[0] & 0x0F
        length = head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", await self._read_exactly(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", await self._read_exactly(8))[0]
        if head[1] & 0x80:  # servers never mask; tolerate it if one does
            mask = await self._read_exactly(4)
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(await self._read_exactly(length)))
        else:
            payload = await self._read_exactly(length)
        return opcode, payload

    async def recv(self, timeout: float) -> dict:
        """Next application event. Pings are counted and deliberately ignored."""
        while True:
            opcode, payload = await asyncio.wait_for(self._read_frame(), timeout)
            if opcode == 0x9:  # Ping: a real endpoint MUST answer. We do not.
                self.pings_seen += 1
                continue
            if opcode == 0xA:  # Pong: the server never sends one, but be lenient.
                continue
            if opcode == 0x8:  # Close
                code = struct.unpack(">H", payload[:2])[0] if len(payload) >= 2 else None
                reason = payload[2:].decode("utf-8", "replace") if len(payload) > 2 else ""
                return {"type": "close", "code": code, "reason": reason}
            if opcode == 0x1:
                return {"type": "message", "message": json.loads(payload.decode("utf-8"))}
            if opcode == 0x2:
                return {"type": "binary"}

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            self.writer.close()


# ---------------------------------------------------------------------------
# 1. Fast inference over a real socket
# ---------------------------------------------------------------------------
def test_real_transport_fast_inference_delivers_the_result(caplog):
    with real_transport_server() as srv:
        srv.stub.delay = 0.2
        sid, token = _create_session(srv.port)

        async def scenario():
            uri = f"ws://{srv.host}:{srv.port}/ws/sessions/{sid}?token={token}"
            received: list[dict] = []
            started = time.perf_counter()
            async with websockets.connect(uri, open_timeout=10) as ws:
                async def pump_until_result():
                    while True:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                        received.append(msg)
                        if msg["type"] in ("result", "error"):
                            return

                pump = asyncio.create_task(pump_until_result())
                ready = await asyncio.wait_for(ws.recv(), timeout=10)
                ready_msg = json.loads(ready)
                assert ready_msg["type"] == "ready"
                assert ready_msg["required_frames"] == REQUIRED

                for i in range(REQUIRED):
                    await ws.send(_frame(i))
                await ws.send(json.dumps({"type": "complete"}))

                await asyncio.wait_for(pump, timeout=20)
                return received, time.perf_counter() - started

        received, elapsed = asyncio.run(scenario())

    kinds = [m["type"] for m in received]
    assert "processing" in kinds
    result = [m for m in received if m["type"] == "result"]
    assert result, f"no result over a real socket; got {kinds}"
    assert result[0]["prediction"] == "spoof"
    assert result[0]["frames_processed"] == REQUIRED
    assert elapsed < 15, f"fast inference took {elapsed:.1f}s"

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("result_send_succeeded") for m in messages)


# ---------------------------------------------------------------------------
# 2. Long (production-length) inference: client stays connected and idle
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_transport_46s_inference_delivers_while_client_sends_nothing(caplog):
    """The production window, end to end.

    150 frames, then ~46 s in which the client sends nothing at all (this is
    what the deployed demo does while the model works). The socket must stay
    healthy on the production ping policy and the verdict must arrive.
    """
    with real_transport_server() as srv:
        srv.stub.delay = 46.0
        sid, token = _create_session(srv.port)

        async def scenario():
            uri = f"ws://{srv.host}:{srv.port}/ws/sessions/{sid}?token={token}"
            events: list[tuple[float, dict]] = []
            started = time.perf_counter()
            closed: dict | None = None
            async with websockets.connect(uri, open_timeout=10) as ws:
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                assert ready["type"] == "ready"

                for i in range(REQUIRED):
                    await ws.send(_frame(i))

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=70)
                    except websockets.ConnectionClosed as exc:
                        closed = {
                            "code": exc.rcvd.code if exc.rcvd else None,
                            "reason": exc.rcvd.reason if exc.rcvd else "",
                            "at": time.perf_counter() - started,
                        }
                        break
                    msg = json.loads(raw)
                    events.append((time.perf_counter() - started, msg))
                    if msg["type"] == "result":
                        break
            return events, closed, time.perf_counter() - started

        events, closed, elapsed = asyncio.run(scenario())

    assert closed is None, f"connection died mid-inference: {closed}"
    kinds = [m["type"] for _, m in events]
    assert "progress" in kinds, f"no keepalives during a 46 s inference: {kinds}"
    progress_times = [t for t, m in events if m["type"] == "progress"]
    assert len(progress_times) >= 5, f"keepalives too sparse: {progress_times}"
    # Every keepalive must arrive while inference is still running (before 46s).
    assert all(t < 46.0 for t in progress_times), progress_times

    result = [m for _, m in events if m["type"] == "result"]
    assert result, f"no result; events={kinds}"
    assert result[0]["prediction"] == "spoof"
    assert 44.0 <= elapsed <= 60.0, f"elapsed={elapsed:.1f}s"
    print(f"[2] long inference: delivered after {elapsed:.1f}s, "
          f"{len(progress_times)} keepalives, no close event")

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("result_send_succeeded") for m in messages)


# ---------------------------------------------------------------------------
# 3. Client hangs up mid-inference: observed, cleaned up, nothing leaks
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_transport_client_disconnect_during_inference_is_observed(caplog):
    from app import main

    with real_transport_server() as srv:
        srv.stub.delay = 8.0
        sid, token = _create_session(srv.port)

        async def scenario():
            uri = f"ws://{srv.host}:{srv.port}/ws/sessions/{sid}?token={token}"
            async with websockets.connect(uri, open_timeout=10) as ws:
                ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                assert ready["type"] == "ready"
                for i in range(REQUIRED):
                    await ws.send(_frame(i))
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                    if msg["type"] == "processing":
                        break
                # Hang up with the model still busy.
            return True

        assert asyncio.run(scenario()) is True

        gone = _wait_for_log("ws_client_gone", caplog, timeout=20)
        assert "inference_running=True" in gone, gone

        finalized = _wait_for_log("session_finalized", caplog, timeout=30)
        assert "delivered=False" in finalized, finalized
        assert any(
            r.getMessage().startswith("inference_completed") for r in caplog.records
        ), "the model must still finish; only delivery fails"
        assert any(
            r.getMessage().startswith("result_send_failed") for r in caplog.records
        )

        # No session (and therefore no frame buffer) may survive the teardown.
        deadline = time.time() + 10
        while time.time() < deadline and main.SESSIONS.active_count() > 0:
            main.SESSIONS.evict_expired()
            time.sleep(0.1)
        assert main.SESSIONS.active_count() == 0, "session/frame buffers leaked"


# ---------------------------------------------------------------------------
# 4. REPRODUCTION of the production failure on the production ping policy
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_reproduction_peer_that_never_answers_pings_is_killed_with_1011(caplog):
    """The production incident, with the initiating party now identifiable.

    websockets sends a keepalive Ping every 20 s and, if the matching Pong is
    not processed within `ping_timeout` (20 s), closes the connection itself
    with code 1011 and the reason "keepalive ping timeout". That close is
    logged nowhere at `--log-level info`, which is why production showed a
    successful inference followed by an unexplained client-side 1011.
    """
    with real_transport_server() as srv:
        # Longer than ping_interval + ping_timeout, so the kill lands while
        # the model is still working — exactly the production timeline.
        srv.stub.delay = 46.0
        sid, token = _create_session(srv.port)

        async def scenario():
            peer = await SilentRawPeer.connect(
                srv.host, srv.port, f"/ws/sessions/{sid}?token={token}"
            )
            try:
                started = time.perf_counter()
                ready = await peer.recv(timeout=10)
                assert ready["type"] == "message" and ready["message"]["type"] == "ready"
                for i in range(REQUIRED):
                    await peer.send_bytes(_frame(i))

                events: list[dict] = []
                closed: dict | None = None
                while True:
                    try:
                        events.append(await peer.recv(timeout=70))
                    except (asyncio.TimeoutError, ConnectionError):
                        break
                    last = events[-1]
                    if last["type"] == "close":
                        closed = {**last, "at": time.perf_counter() - started}
                        break
                    if last["type"] == "message" and last["message"]["type"] == "result":
                        break
                return closed, events, peer.pings_seen, time.perf_counter() - started
            finally:
                await peer.aclose()

        closed, events, pings_seen, elapsed = asyncio.run(scenario())

    assert closed is not None, (
        "expected the server to kill the silent peer with a close frame; "
        f"events={[e.get('message', {}).get('type', e['type']) for e in events]}"
    )
    assert closed["code"] == 1011, closed
    assert "keepalive ping timeout" in closed["reason"], closed
    assert pings_seen >= 1, "the server must have sent the ping that went unanswered"
    assert not any(
        e["type"] == "message" and e["message"]["type"] == "result" for e in events
    ), "the result must not arrive on a connection the server already failed"
    assert 25.0 <= closed["at"] <= 55.0, f"close at {closed['at']:.1f}s"
    print(f"[4] reproduced: Close(code={closed['code']}, reason={closed['reason']!r}) "
          f"at {closed['at']:.1f}s after accept, {pings_seen} unanswered ping(s), "
          f"elapsed={elapsed:.1f}s")

    # Server-side, the production log sequence must now be visible.
    completed = _wait_for_log("inference_completed", caplog, timeout=40)
    assert "prediction=spoof" in completed
    failed = _wait_for_log("result_send_failed", caplog, timeout=20)
    assert "stage=after_inference" in failed, failed
    finalized = _wait_for_log("session_finalized", caplog, timeout=20)
    assert "delivered=False" in finalized, finalized


# ---------------------------------------------------------------------------
# 5. The pinned ping policy keeps a silent peer alive through the inference
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_pinned_ping_timeout_keeps_a_silent_peer_alive_through_the_inference(caplog):
    """`--ws-ping-timeout 60` (render.yaml) must make the kill above impossible.

    Only the pong deadline is relaxed: the interval stays at 20 s, so a dead
    peer is still detected, while a single lost or delayed pong can no longer
    destroy a verification that is seconds from completing.
    """
    with real_transport_server(ping_timeout=60.0) as srv:
        srv.stub.delay = 46.0
        sid, token = _create_session(srv.port)

        async def scenario():
            peer = await SilentRawPeer.connect(
                srv.host, srv.port, f"/ws/sessions/{sid}?token={token}"
            )
            try:
                started = time.perf_counter()
                ready = await peer.recv(timeout=10)
                assert ready["message"]["type"] == "ready"
                for i in range(REQUIRED):
                    await peer.send_bytes(_frame(i))

                result = None
                closed = None
                progress = 0
                while True:
                    try:
                        event = await peer.recv(timeout=70)
                    except (asyncio.TimeoutError, ConnectionError):
                        break
                    if event["type"] == "close":
                        closed = event
                        break
                    if event["type"] == "message":
                        kind = event["message"]["type"]
                        if kind == "progress":
                            progress += 1
                        if kind == "result":
                            result = event["message"]
                            break
                return result, closed, progress, peer.pings_seen, time.perf_counter() - started
            finally:
                await peer.aclose()

        result, closed, progress, pings_seen, elapsed = asyncio.run(scenario())

    assert closed is None, f"connection still died: {closed}"
    assert result is not None, "no result on the connection that survived"
    assert result["prediction"] == "spoof"
    assert result["frames_processed"] == REQUIRED
    assert progress >= 1, "progress keepalives must still flow"
    assert elapsed <= 60.0, f"elapsed={elapsed:.1f}s"
    print(f"[5] survived: result after {elapsed:.1f}s with {progress} keepalives "
          f"and {pings_seen} unanswered ping(s)")

    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("result_send_succeeded") for m in messages)


# ---------------------------------------------------------------------------
# 6. Resource accounting for the whole module
# ---------------------------------------------------------------------------
def test_no_sessions_or_frame_buffers_survive_the_module():
    from app import main

    main.SESSIONS.evict_expired()
    assert main.SESSIONS.active_count() == 0, "sessions leaked across scenarios"
    for session in main.SESSIONS._sessions.values():  # defensive: table is empty
        assert session.frames == {}, "frame buffers leaked"
