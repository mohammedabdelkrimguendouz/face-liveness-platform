import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LivenessClient } from "../src/client";
import { LivenessError, isLivenessError } from "../src/errors";

// ---------------------------------------------------------------------------
// Browser API mocks
// ---------------------------------------------------------------------------
const REQUIRED = 150;

class MockWebSocket {
  static OPEN = 1;
  static instances: MockWebSocket[] = [];
  readyState = 0;
  bufferedAmount = 0;
  binaryType = "arraybuffer";
  sent: (string | ArrayBuffer)[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string | ArrayBuffer }) => void) | null = null;
  onclose:
    | ((ev: { code: number; reason?: string; wasClean?: boolean }) => void)
    | null = null;
  onerror: (() => void) | null = null;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
  send(data: string | ArrayBuffer) {
    this.sent.push(data);
  }
  // Local close, i.e. what the SDK's own teardown() calls.
  close(code = 1000, reason = "") {
    this.readyState = 3;
    this.onclose?.({ code, reason, wasClean: true });
  }
  // test helpers
  _open() {
    this.readyState = 1;
    this.onopen?.();
  }
  _msg(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  /**
   * Close initiated by the peer rather than by us. The reason string is what
   * makes the production incident identifiable: websockets labels its 1011
   * kill "keepalive ping timeout", while a proxy kill carries another (or no)
   * reason, and the SDK now reports both.
   */
  _serverClose(code: number, reason = "", wasClean = true) {
    this.readyState = 3;
    this.onclose?.({ code, reason, wasClean });
  }
}

function installMocks() {
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket);

  // Canvas 2D context + toBlob producing a tiny fake JPEG buffer.
  const ctx = { drawImage: vi.fn() };
  vi.stubGlobal("HTMLCanvasElement", class {
    width = 0;
    height = 0;
    getContext() { return ctx; }
    toBlob(cb: (b: Blob) => void, _type: string, _q: number) {
      cb(new Blob([new Uint8Array([1, 2, 3, 4, 5])], { type: "image/jpeg" }));
    }
  });
  // document.createElement("canvas") must return our mock.
  const originalCreate = document.createElement.bind(document);
  vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
    if (tag === "canvas") return new (globalThis as any).HTMLCanvasElement() as any;
    return originalCreate(tag as any);
  });

  // Canvas is "supported".
  (window as any).isSecureContext = true;

  // getUserMedia returns a fake stream; video element plays immediately.
  const track = { stop: vi.fn() };
  (navigator as any).mediaDevices = {
    getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [track] }),
  };
  vi.spyOn(window.HTMLMediaElement.prototype, "play").mockResolvedValue(undefined as any);
  // jsdom's HTMLMediaElement has read-only readyState/videoWidth/videoHeight,
  // so override via defineProperty. Fire loadedmetadata so openCamera resolves.
  vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
    if (tag === "video") {
      const el = originalCreate("video") as any;
      Object.defineProperty(el, "readyState", { value: 4, configurable: true });
      Object.defineProperty(el, "videoWidth", { value: 640, configurable: true });
      Object.defineProperty(el, "videoHeight", { value: 480, configurable: true });
      queueMicrotask(() => el.dispatchEvent(new Event("loadedmetadata")));
      return el;
    }
    if (tag === "canvas") return new (globalThis as any).HTMLCanvasElement() as any;
    return originalCreate(tag as any);
  });

  // fetch -> session
  const sessionBody = {
    session_id: "sess123",
    ws_url: "/ws/sessions/sess123",
    ws_token: "tok123",
    num_frames: REQUIRED,
    max_frame_bytes: 2 * 1024 * 1024,
  };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    status: 201,
    json: async () => sessionBody,
  }));

  return { sessionBody, ctx, track };
}

function binaryFrames(ws: MockWebSocket): ArrayBuffer[] {
  return ws.sent.filter((s): s is ArrayBuffer => s instanceof ArrayBuffer);
}

async function waitFor(cond: () => boolean, timeout = 2000): Promise<void> {
  const start = Date.now();
  while (!cond()) {
    if (Date.now() - start > timeout) throw new Error("waitFor timeout");
    await new Promise((r) => setTimeout(r, 5));
  }
}

beforeEach(() => {
  installMocks();
  // jsdom's Blob does not implement arrayBuffer(); real browsers do.
  if (typeof Blob.prototype.arrayBuffer !== "function") {
    Blob.prototype.arrayBuffer = function (this: Blob): Promise<ArrayBuffer> {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as ArrayBuffer);
        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(this);
      });
    };
  }
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("LivenessClient", () => {
  /**
   * Starts a verification, captures the URL the WebSocket was opened with, then
   * aborts so the pending promise never leaks into the next test.
   *
   * `session` overrides the `POST /sessions` body, which is the only input the
   * SDK controls the handshake URL from (it returns `ws_url` as a path).
   */
  async function captureWsUrl(
    apiUrl: string,
    session: Record<string, unknown> = {},
  ): Promise<string> {
    (globalThis as any).fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: async () => ({
        session_id: "sess123",
        ws_url: "/ws/sessions/sess123",
        ws_token: "tok123",
        num_frames: REQUIRED,
        max_frame_bytes: 2 * 1024 * 1024,
        ...session,
      }),
    });

    const client = new LivenessClient({ apiUrl, targetFps: 200 });
    const promise = client.start();

    await waitFor(() => MockWebSocket.instances.length > 0);
    const ws = MockWebSocket.instances[0];
    const { url } = ws;

    ws._open();
    ws._msg({ type: "ready", required_frames: REQUIRED });
    await waitFor(() => binaryFrames(ws).length > 0);

    client.cancel();
    await expect(promise).rejects.toMatchObject({ code: "cancelled" });
    return url;
  }

  it("resolves the relative ws_url against the configured API host and appends the session token", async () => {
    const url = await captureWsUrl("https://face-liveness-api-892k.onrender.com");

    // Regression: without `?token=` the backend refuses the upgrade before
    // accept() and the ASGI server answers `HTTP 403 Forbidden`, so the
    // connection never reaches the WebSocket handler.
    expect(url).toBe(
      "wss://face-liveness-api-892k.onrender.com/ws/sessions/sess123?token=tok123",
    );
  });

  it("keeps local development on ws://localhost:8000", async () => {
    const url = await captureWsUrl("http://localhost:8000");
    expect(url).toBe("ws://localhost:8000/ws/sessions/sess123?token=tok123");
  });

  it("percent-encodes the session token in the query string", async () => {
    const url = await captureWsUrl("https://api.example.com", {
      ws_token: "ab/cd+ef==",
    });
    expect(url).toBe("wss://api.example.com/ws/sessions/sess123?token=ab%2Fcd%2Bef%3D%3D");
  });

  it("never puts the session token in the URL's path or fragment", async () => {
    const url = await captureWsUrl("https://api.example.com", { ws_token: "sekrit" });
    const parsed = new URL(url);
    expect(parsed.pathname).toBe("/ws/sessions/sess123");
    expect(parsed.searchParams.get("token")).toBe("sekrit");
    expect(parsed.hash).toBe("");
  });

  it("creates a session and streams frames in order, then resolves the result", async () => {
    const onProgress = vi.fn();
    const client = new LivenessClient({ apiUrl: "http://localhost:8000", targetFps: 200 }, { onProgress });

    const promise = client.start();

    // Wait for WS creation + open + ready handshake to begin capture.
    await waitFor(() => MockWebSocket.instances.length > 0);
    const ws = MockWebSocket.instances[0];
    ws._open();
    ws._msg({ type: "ready", required_frames: REQUIRED });

    // Capture loop should now send frames with sequential big-endian indices.
    await waitFor(() => binaryFrames(ws).length === REQUIRED, 5000);

    const frames = binaryFrames(ws);
    expect(frames.length).toBe(REQUIRED);
    for (let i = 0; i < frames.length; i++) {
      const idx = new DataView(frames[i]).getUint32(0, false);
      expect(idx).toBe(i);
    }

    // Server responds to complete() with processing + result.
    ws._msg({ type: "processing", received: REQUIRED });
    ws._msg({
      type: "result",
      session_id: "sess123",
      prediction: "live",
      anomaly_score: 0.04,
      reconstruction_error: 0.041,
      threshold: 0.5,
      frames_processed: REQUIRED,
      inference_ms: 1200,
    });

    const result = await promise;
    expect(result.prediction).toBe("live");
    expect(result.anomalyScore).toBeCloseTo(0.04);
    expect(result.threshold).toBe(0.5);
    expect(result.framesProcessed).toBe(REQUIRED);
    expect(result.sessionId).toBe("sess123");
  });

  it("surfaces a server error as a LivenessError", async () => {
    const onError = vi.fn();
    const client = new LivenessClient({ apiUrl: "http://localhost:8000", targetFps: 200 }, { onError });
    const promise = client.start();
    await waitFor(() => MockWebSocket.instances.length > 0);
    const ws = MockWebSocket.instances[0];
    ws._open();
    ws._msg({ type: "ready", required_frames: 5 });
    ws._msg({ type: "error", error: "incomplete_sequence", message: "Sequence incomplete" });

    await expect(promise).rejects.toMatchObject({ code: "server-error" });
    expect(onError).toHaveBeenCalled();
  });

  it("rejects with connection-failed when session fetch is unreachable", async () => {
    (globalThis as any).fetch = vi.fn().mockRejectedValue(new TypeError("network down"));
    const client = new LivenessClient({ apiUrl: "http://localhost:8000" });
    await expect(client.start()).rejects.toMatchObject({ code: "connection-failed" });
  });

  it("rejects with session-failed on non-2xx session creation", async () => {
    (globalThis as any).fetch = vi.fn().mockResolvedValue({ ok: false, status: 503 });
    const client = new LivenessClient({ apiUrl: "http://localhost:8000" });
    await expect(client.start()).rejects.toMatchObject({ code: "session-failed" });
  });

  it("cancel() stops a running verification and emits onCancel", async () => {
    const onCancel = vi.fn();
    const client = new LivenessClient({ apiUrl: "http://localhost:8000", targetFps: 200 }, { onCancel });
    const promise = client.start();
    await waitFor(() => MockWebSocket.instances.length > 0);
    const ws = MockWebSocket.instances[0];
    ws._open();
    ws._msg({ type: "ready", required_frames: REQUIRED });
    await waitFor(() => binaryFrames(ws).length > 0);

    client.cancel();
    expect(onCancel).toHaveBeenCalled();
    expect(client.getPhase()).toBe("cancelled");
    // The promise should settle (reject with cancelled) rather than hang.
    await expect(promise).rejects.toMatchObject({ code: "cancelled" });
  });

  it("times out and reports a timeout error", async () => {
    vi.useFakeTimers();
    const onError = vi.fn();
    const client = new LivenessClient({ apiUrl: "http://localhost:8000", timeoutMs: 100 }, { onError });
    const promise = client.start();
    // Attach the rejection assertion BEFORE firing the timer so the rejection
    // is never momentarily unhandled.
    const assertion = expect(promise).rejects.toMatchObject({ code: "timeout" });
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(100);
    await assertion;
    expect(onError).toHaveBeenCalled();
    vi.useRealTimers();
  });
});

// ---------------------------------------------------------------------------
// Timeout contract + close/keepalive diagnostics.
//
// Production showed "Connection closed (code 1011)" at ~52 s while the SDK
// default timeout is 120 000 ms, so these pin two things that must stay
// true: the timeout only ever fires when it is configured to, and when the
// server fails the connection the error carries the code, the reason and the
// cleanliness flag needed to tell a server-side failure from a proxy kill.
// ---------------------------------------------------------------------------
describe("LivenessClient — timeout and connection diagnostics", () => {
  const RESULT_MSG = {
    type: "result",
    session_id: "sess123",
    prediction: "live",
    anomaly_score: 0.04,
    reconstruction_error: 0.041,
    threshold: 0.5,
    frames_processed: REQUIRED,
    inference_ms: 43_390,
  };

  /** Flush microtasks (fetch → camera → session → WebSocket) without time. */
  async function flush(n = 12): Promise<void> {
    for (let i = 0; i < n; i++) await vi.advanceTimersByTimeAsync(0);
  }

  /**
   * Pump the fake clock in small steps until `cond` holds. Fixed advances are
   * brittle here: every captured frame costs the sleep interval plus the
   * canvas/encode microtask chain, so the exact fake-ms cost per frame is not
   * a clean multiple of the interval.
   */
  async function advanceUntil(
    cond: () => boolean,
    budgetMs = 5000,
    stepMs = 50,
  ): Promise<void> {
    for (let spent = 0; !cond() && spent < budgetMs; spent += stepMs) {
      await vi.advanceTimersByTimeAsync(stepMs);
    }
  }

  it("does not fire the default result timeout during a 119-second verification", async () => {
    vi.useFakeTimers();
    try {
      const onError = vi.fn();
      let settled = false;
      const client = new LivenessClient(
        { apiUrl: "http://localhost:8000", targetFps: 200 },
        { onError },
      );
      const promise = client.start();
      promise.then(
        () => { settled = true; },
        () => { settled = true; },
      );

      await flush();
      expect(MockWebSocket.instances.length).toBe(1);
      const ws = MockWebSocket.instances[0];
      ws._open();
      ws._msg({ type: "ready", required_frames: REQUIRED });

      // Capture 150 frames at 200 fps (5 ms apart) — under 1 s of fake time.
      await advanceUntil(() => binaryFrames(ws).length === REQUIRED);
      expect(binaryFrames(ws).length).toBe(REQUIRED);

      // ~119 s of the 120 s default budget elapse with the socket silent —
      // exactly the shape of a long CPU inference. Nothing may fire. The
      // budget is deliberately kept short of 120 000 ms total so the default
      // deadline itself is not the thing that settles the promise.
      await vi.advanceTimersByTimeAsync(118_000);
      expect(settled).toBe(false);
      expect(onError).not.toHaveBeenCalled();
      expect(ws.readyState).toBe(MockWebSocket.OPEN);

      // The verdict still arrives and resolves normally.
      ws._msg({ type: "processing", received: REQUIRED });
      ws._msg(RESULT_MSG);
      const result = await promise;
      expect(result.prediction).toBe("live");
      expect(settled).toBe(true);

      // teardown() cleared the timer: crossing 120 s afterwards must not
      // produce a spurious timeout on a finished verification.
      await vi.advanceTimersByTimeAsync(60_000);
      expect(onError).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("fires only a configured timeout, closes the socket, and ignores a late result", async () => {
    vi.useFakeTimers();
    try {
      const onError = vi.fn();
      const onSuccess = vi.fn();
      const client = new LivenessClient(
        { apiUrl: "http://localhost:8000", targetFps: 200, timeoutMs: 5_000 },
        { onError, onSuccess },
      );
      const promise = client.start();
      const assertion = expect(promise).rejects.toMatchObject({ code: "timeout" });

      await flush();
      const ws = MockWebSocket.instances[0];
      ws._open();
      ws._msg({ type: "ready", required_frames: REQUIRED });
      // Capture must finish inside the 5 s deadline — pump rather than guess.
      await advanceUntil(() => binaryFrames(ws).length === REQUIRED);
      expect(binaryFrames(ws).length).toBe(REQUIRED);

      // The explicitly configured 5 s deadline expires and must fire.
      await vi.advanceTimersByTimeAsync(5_000);
      await assertion;
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError.mock.calls[0][0]).toMatchObject({ code: "timeout" });
      expect(ws.readyState).toBe(3); // teardown closed the socket

      // A verdict that arrives after the deadline must not resurrect the run.
      ws._msg(RESULT_MSG);
      expect(onSuccess).not.toHaveBeenCalled();
      expect(onError).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("reports code, reason and cleanliness when the server fails the connection", async () => {
    const onError = vi.fn();
    const client = new LivenessClient({ apiUrl: "http://localhost:8000" }, { onError });
    const promise = client.start();
    const assertion = expect(promise).rejects.toMatchObject({ code: "connection-failed" });

    await waitFor(() => MockWebSocket.instances.length > 0);
    const ws = MockWebSocket.instances[0];
    ws._open();
    ws._msg({ type: "ready", required_frames: REQUIRED });

    // Production, verbatim: websockets' own keepalive ping timeout.
    ws._serverClose(1011, "keepalive ping timeout", true);
    await assertion;

    const err = onError.mock.calls[0][0];
    expect(err.code).toBe("connection-failed");
    expect(err.message).toContain("code=1011");
    expect(err.message).toContain("reason=keepalive ping timeout");
    expect(err.message).toContain("wasClean=true");
    expect(err.message).toMatch(/elapsedMs=\d+/);
    expect(err.detail).toContain("code=1011");
    // The generic user-facing sentence must still lead the message.
    expect(err.message).toContain("Connection closed unexpectedly");
  });

  it("surfaces an abnormal network drop as clearly as a server-side close", async () => {
    const onError = vi.fn();
    const client = new LivenessClient({ apiUrl: "http://localhost:8000" }, { onError });
    const promise = client.start();
    const assertion = expect(promise).rejects.toMatchObject({ code: "connection-failed" });

    await waitFor(() => MockWebSocket.instances.length > 0);
    const ws = MockWebSocket.instances[0];
    ws._open();
    ws._msg({ type: "ready", required_frames: REQUIRED });

    ws._serverClose(1006, "", false); // TCP drop: no close frame, no reason
    await assertion;

    const err = onError.mock.calls[0][0];
    expect(err.message).toContain("code=1006");
    expect(err.message).toContain("reason=none");
    expect(err.message).toContain("wasClean=false");
  });

  it("stays in processing while server keepalive progress frames arrive", async () => {
    const onProgress = vi.fn();
    const client = new LivenessClient(
      { apiUrl: "http://localhost:8000", targetFps: 200 },
      { onProgress },
    );
    const promise = client.start();

    await waitFor(() => MockWebSocket.instances.length > 0);
    const ws = MockWebSocket.instances[0];
    ws._open();
    ws._msg({ type: "ready", required_frames: REQUIRED });
    await waitFor(() => binaryFrames(ws).length === REQUIRED, 5000);

    ws._msg({ type: "processing", received: REQUIRED });
    // Two keepalives, 5 s apart — what the server emits during inference.
    ws._msg({ type: "progress", stage: "inferring", elapsed_ms: 5_000 });
    expect(client.getPhase()).toBe("processing");
    ws._msg({ type: "progress", stage: "inferring", elapsed_ms: 10_000 });
    expect(client.getPhase()).toBe("processing");

    const keepalives = onProgress.mock.calls.filter(
      (call) => call[0].message === "Analyzing your video…",
    );
    expect(keepalives.length).toBeGreaterThanOrEqual(2);

    // Keepalives are not terminal: the run only settles on the verdict.
    ws._msg(RESULT_MSG);
    const result = await promise;
    expect(result.prediction).toBe("live");
  });
});

describe("LivenessError", () => {
  it("is identifiable via isLivenessError", () => {
    const err = new LivenessError("timeout", "too slow");
    expect(isLivenessError(err)).toBe(true);
    expect(isLivenessError(new Error("plain"))).toBe(false);
    expect(err.code).toBe("timeout");
  });
});
