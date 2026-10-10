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
  onclose: ((ev: { code: number }) => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
  send(data: string | ArrayBuffer) {
    this.sent.push(data);
  }
  close(code = 1000) {
    this.readyState = 3;
    this.onclose?.({ code });
  }
  // test helpers
  _open() {
    this.readyState = 1;
    this.onopen?.();
  }
  _msg(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
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

describe("LivenessError", () => {
  it("is identifiable via isLivenessError", () => {
    const err = new LivenessError("timeout", "too slow");
    expect(isLivenessError(err)).toBe(true);
    expect(isLivenessError(new Error("plain"))).toBe(false);
    expect(err.code).toBe("timeout");
  });
});
