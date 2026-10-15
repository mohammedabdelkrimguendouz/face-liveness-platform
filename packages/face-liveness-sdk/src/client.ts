import { assertBrowserSupported, openCamera, type CameraHandle } from "./camera";
import { LivenessError } from "./errors";
import type {
  LivenessCallbacks,
  LivenessOptions,
  LivenessPhase,
  LivenessProgress,
  LivenessResult,
} from "./types";

/** 4-byte big-endian sequence header, matching the backend protocol. */
const SEQ_HEADER_SIZE = 4;

/** Fallback frame count if the server does not specify one. */
const FALLBACK_REQUIRED_FRAMES = 150;

interface SessionInfo {
  sessionId: string;
  wsUrl: string;
  wsToken: string;
  numFrames: number;
  maxFrameBytes: number;
}

interface ServerMessage {
  type: string;
  [key: string]: unknown;
}

/**
 * High-level, framework-agnostic client.
 *
 * Flow: create session -> open WebSocket -> capture frames at a controlled rate
 * -> transmit in order -> await authoritative server result.
 */
export class LivenessClient {
  private readonly opts: {
    apiUrl: string;
    width: number;
    targetFps: number;
    jpegQuality: number;
    timeoutMs: number;
    deviceId?: string;
    signal?: AbortSignal;
  };
  private readonly cb: LivenessCallbacks;
  private camera: CameraHandle | null = null;
  private ws: WebSocket | null = null;
  /** When the socket was constructed; used to age close/error diagnostics. */
  private wsOpenedAt = 0;
  private phase: LivenessPhase = "idle";
  private captured = 0;
  private required = FALLBACK_REQUIRED_FRAMES;
  private acknowledged = 0;
  private cancelled = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  /** Rejects the in-flight start() promise; used by cancel() to settle it. */
  private pendingReject: ((e: LivenessError) => void) | null = null;
  private readonly encoder: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D | null;

  constructor(options: LivenessOptions, callbacks: LivenessCallbacks = {}) {
    this.opts = {
      apiUrl: options.apiUrl,
      width: options.width ?? 720,
      targetFps: options.targetFps ?? 20,
      jpegQuality: options.jpegQuality ?? 85,
      timeoutMs: options.timeoutMs ?? 120_000,
      deviceId: options.deviceId,
      signal: options.signal,
    };
    this.cb = callbacks;
    // Offscreen canvas used to encode frames without touching the DOM.
    this.encoder = document.createElement("canvas");
    this.ctx = this.encoder.getContext("2d");
  }

  getPhase(): LivenessPhase {
    return this.phase;
  }

  /** Stop everything and release resources. Safe to call multiple times. */
  cancel(): void {
    if (this.cancelled) return;
    this.cancelled = true;
    // Settle the in-flight start() promise with a cancelled error (if pending).
    const reject = this.pendingReject;
    this.pendingReject = null;
    this.teardown();
    this.setPhase("cancelled");
    this.cb.onCancel?.();
    reject?.(new LivenessError("cancelled", "Verification was cancelled."));
  }

  /**
   * Run a full verification. Resolves with the backend result, or rejects with
   * a {@link LivenessError}. Always cleans up camera + socket on completion.
   *
   * `run()` is raced against a single outer deferred that `cancel()` and the
   * timeout settle, so whichever finishes first wins and the loser never
   * surfaces as an unhandled rejection.
   */
  async start(): Promise<LivenessResult> {
    let rejectOuter!: (e: LivenessError) => void;
    const outer = new Promise<LivenessResult>((_, rej) => {
      rejectOuter = rej;
    });
    // Guard: whichever promise loses the race below must never surface as an
    // unhandled rejection. `outer` is settled only by cancel()/timeout; the
    // run() promise is settled by the normal flow.
    outer.catch(() => {});
    this.pendingReject = (e) => rejectOuter(e);

    const runPromise = this.run();
    // Consume run() rejections so a late settle (after timeout/cancel won the
    // race) cannot become an unhandled rejection.
    runPromise.catch(() => {});

    try {
      const result = await Promise.race([runPromise, outer]);
      this.pendingReject = null;
      this.setPhase("completed");
      this.cb.onSuccess?.(result);
      return result;
    } catch (err) {
      this.pendingReject = null;
      const error =
        err instanceof LivenessError
          ? err
          : new LivenessError("server-error", (err as Error)?.message ?? "Unknown error");
      if (error.code !== "cancelled" && error.code !== "timeout") {
        this.setPhase("failed");
        this.cb.onError?.(error);
      } else if (error.code === "cancelled") {
        this.setPhase("cancelled");
        this.cb.onCancel?.();
      }
      throw error;
    } finally {
      this.teardown();
    }
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------
  private async run(): Promise<LivenessResult> {
    assertBrowserSupported();
    this.throwIfCancelled();

    if (this.opts.signal) {
      if (this.opts.signal.aborted) this.cancel();
      this.opts.signal.addEventListener("abort", () => this.cancel(), { once: true });
    }

    this.startTimeout();

    // 1. Permission + camera
    this.setPhase("requesting-permission");
    this.camera = await openCamera(this.opts.width, this.opts.deviceId);
    // Expose the live stream so the host app can render a preview.
    this.cb.onStream?.(this.camera.stream);
    this.throwIfCancelled();

    // 2. Session
    this.setPhase("connecting");
    const session = await this.createSession();
    this.required = session.numFrames;

    // 3. Capture + stream
    this.setPhase("capturing");
    const result = await this.captureAndStream(session);

    // Phase completion + onSuccess are emitted by start(), which owns the
    // race against cancel/timeout, so we simply return the result here.
    return result;
  }

  private async createSession(): Promise<SessionInfo> {
    const url = joinUrl(this.opts.apiUrl, "/sessions");
    let res: Response;
    try {
      res = await fetch(url, { method: "POST", headers: { Accept: "application/json" } });
    } catch (err) {
      throw new LivenessError(
        "connection-failed",
        "Could not reach the liveness service. Check the connection and try again.",
        (err as Error)?.message
      );
    }
    if (!res.ok) {
      throw new LivenessError("session-failed", `Session creation failed (HTTP ${res.status}).`);
    }
    const body = await res.json();
    if (!body.session_id || !body.ws_token) {
      throw new LivenessError("session-failed", "Malformed session response from server.");
    }
    return {
      sessionId: body.session_id,
      wsUrl: absoluteWsUrl(this.opts.apiUrl, body.ws_url),
      wsToken: body.ws_token,
      numFrames: body.num_frames ?? FALLBACK_REQUIRED_FRAMES,
      maxFrameBytes: body.max_frame_bytes ?? 2 * 1024 * 1024,
    };
  }

  private captureAndStream(session: SessionInfo): Promise<LivenessResult> {
    return new Promise<LivenessResult>((resolve, reject) => {
      let settled = false;
      const fail = (err: LivenessError) => {
        if (settled) return;
        settled = true;
        this.pendingReject = null;
        reject(err);
      };
      const succeed = (r: LivenessResult) => {
        if (settled) return;
        settled = true;
        this.pendingReject = null;
        resolve(r);
      };
      // Let cancel() settle this promise.
      this.pendingReject = fail;

      const ws = new WebSocket(withWsToken(session.wsUrl, session.wsToken));
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      this.wsOpenedAt = performance.now();

      /**
       * Every close/error is described by the four facts a support ticket
       * actually needs. `wasClean=false` with a low code points at the network
       * or a proxy; a clean close with 1011 points at the server. Elapsed time
       * localises the event on the verification timeline.
       */
      const describe = (ev: { code?: number; reason?: string; wasClean?: boolean }): string => {
        const elapsedMs = Math.round(performance.now() - this.wsOpenedAt);
        return `code=${ev.code ?? "n/a"} reason=${ev.reason || "none"} ` +
          `wasClean=${ev.wasClean ?? "n/a"} elapsedMs=${elapsedMs}`;
      };

      ws.onerror = () =>
        fail(
          new LivenessError(
            "connection-failed",
            `WebSocket connection error (${describe({})}).`,
            "socket-error",
          )
        );

      ws.onclose = (ev) => {
        // After an intentional cancel/teardown, ignore the close event so it
        // does not surface as a spurious connection error.
        if (settled || this.cancelled) return;
        const detail = describe(ev);
        if (ev.code === 4401) fail(new LivenessError("session-failed", `Invalid session token (${detail}).`));
        else if (ev.code === 4404) fail(new LivenessError("session-failed", `Session not found or expired (${detail}).`));
        else if (ev.code === 4409) fail(new LivenessError("session-failed", `Session already used (${detail}).`));
        else fail(new LivenessError("connection-failed", `Connection closed unexpectedly (${detail}).`, detail));
      };

      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        let msg: ServerMessage;
        try {
          msg = JSON.parse(ev.data) as ServerMessage;
        } catch {
          return;
        }
        switch (msg.type) {
          case "ready":
            this.required = (msg.required_frames as number) ?? this.required;
            void this.beginCaptureLoop(session, fail);
            break;
          case "ack":
            this.acknowledged = (msg.received as number) ?? this.acknowledged;
            this.emit("transmitting", "Uploading frames to the verification service…");
            break;
          case "processing":
            this.setPhase("processing");
            this.emit("processing", "Analyzing your video…");
            break;
          case "progress":
            // Server keepalive emitted while the model runs. It proves the
            // connection is alive during the multi-second inference window
            // rather than leaving the user staring at a frozen spinner.
            this.setPhase("processing");
            this.emit("processing", "Analyzing your video…");
            break;
          case "result": {
            const prediction = msg.prediction as LivenessResult["prediction"];
            if (prediction !== "live" && prediction !== "spoof") {
              fail(new LivenessError("server-error", "Server returned an unknown prediction."));
              return;
            }
            succeed({
              prediction,
              anomalyScore: Number(msg.anomaly_score),
              reconstructionError: Number(msg.reconstruction_error),
              threshold: Number(msg.threshold),
              framesProcessed: Number(msg.frames_processed),
              inferenceMs: Number(msg.inference_ms),
              sessionId: String(msg.session_id ?? session.sessionId),
            });
            break;
          }
          case "error":
            fail(
              new LivenessError(
                "server-error",
                (msg.message as string) ?? "Verification failed on the server.",
                msg.error as string
              )
            );
            break;
          case "cancelled":
            break;
        }
      };
    });
  }

  /**
   * Capture frames at the target rate and send them in order. Applies bounded
   * backpressure: waits for the socket's bufferedAmount to drain before
   * capturing more, so a slow network cannot grow memory unboundedly.
   */
  private async beginCaptureLoop(
    session: SessionInfo,
    fail: (e: LivenessError) => void
  ): Promise<void> {
    const interval = 1000 / this.opts.targetFps;
    const maxBuffered = 4 * session.maxFrameBytes;

    try {
      while (this.captured < this.required) {
        if (this.cancelled) return;
        if (this.ws?.readyState !== WebSocket.OPEN) {
          fail(new LivenessError("connection-failed", "Connection lost during capture."));
          return;
        }
        // Backpressure: wait for the socket to drain before capturing more.
        while (this.ws && this.ws.bufferedAmount > maxBuffered) {
          await sleep(10);
          if (this.cancelled || this.ws?.readyState !== WebSocket.OPEN) return;
        }

        const started = performance.now();
        const payload = await this.captureFrame(session.maxFrameBytes);
        if (payload) {
          this.sendFrame(this.captured, payload);
          this.captured += 1;
          if (this.captured % 10 === 0 || this.captured === this.required) {
            this.emit("capturing", `Captured ${this.captured} of ${this.required} frames…`);
          }
        }
        const elapsed = performance.now() - started;
        const wait = interval - elapsed;
        if (wait > 0) await sleep(wait);
      }

      // Signal completion; server responds with `processing` then `result`.
      this.emit("transmitting", "Finalizing upload…");
      this.ws?.send(JSON.stringify({ type: "complete" }));
    } catch (err) {
      if (err instanceof LivenessError) {
        fail(err);
      } else {
        fail(new LivenessError("capture-failed", (err as Error)?.message ?? "Frame capture failed."));
      }
    }
  }

  private sendFrame(index: number, payload: ArrayBuffer): void {
    const buf = new ArrayBuffer(SEQ_HEADER_SIZE + payload.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, index, false); // big-endian
    new Uint8Array(buf, SEQ_HEADER_SIZE).set(new Uint8Array(payload));
    this.ws?.send(buf);
  }

  /**
   * Draw the current video frame to the offscreen canvas and encode as JPEG.
   * Returns null (and lets the loop retry) if the frame is not ready.
   */
  private async captureFrame(maxBytes: number): Promise<ArrayBuffer | null> {
    const video = this.camera?.video;
    if (!video || video.readyState < 2 || !this.ctx) return null;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return null;
    if (this.encoder.width !== w || this.encoder.height !== h) {
      this.encoder.width = w;
      this.encoder.height = h;
    }
    this.ctx.drawImage(video, 0, 0, w, h);

    const blob = await new Promise<Blob | null>((resolve) =>
      this.encoder.toBlob((b) => resolve(b), "image/jpeg", this.opts.jpegQuality / 100)
    );
    if (!blob) return null;
    // Guard against oversized frames (e.g. very high-res cameras).
    if (blob.size > maxBytes) {
      throw new LivenessError(
        "capture-failed",
        "Captured frame exceeds the server size limit. Use a lower resolution."
      );
    }
    return blob.arrayBuffer();
  }

  private startTimeout(): void {
    this.timer = setTimeout(() => {
      // Settle the pending promise with a timeout error (distinct from cancel).
      const reject = this.pendingReject;
      this.pendingReject = null;
      this.cancelled = true;
      this.teardown();
      this.setPhase("failed");
      const err = new LivenessError("timeout", "Verification timed out. Please try again.");
      this.cb.onError?.(err);
      reject?.(err);
    }, this.opts.timeoutMs);
  }

  private throwIfCancelled(): void {
    if (this.cancelled) throw new LivenessError("cancelled", "Verification was cancelled.");
  }

  private setPhase(phase: LivenessPhase): void {
    this.phase = phase;
    this.cb.onPhaseChange?.(phase);
  }

  private emit(phase: LivenessPhase, message: string): void {
    this.setPhase(phase);
    const progress: LivenessProgress = {
      phase,
      captured: this.captured,
      required: this.required,
      acknowledged: this.acknowledged,
      message,
    };
    this.cb.onProgress?.(progress);
  }

  private teardown(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    try {
      if (this.ws && this.ws.readyState <= WebSocket.OPEN) this.ws.close(1000, "done");
    } catch {
      /* ignore */
    }
    this.ws = null;
    this.camera?.stop();
    this.camera = null;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function joinUrl(base: string, path: string): string {
  return `${base.replace(/\/+$/, "")}${path}`;
}

function absoluteWsUrl(apiBase: string, wsUrl: string): string {
  if (wsUrl.startsWith("ws://") || wsUrl.startsWith("wss://")) return wsUrl;
  try {
    const u = new URL(apiBase);
    const scheme = u.protocol === "https:" ? "wss:" : "ws:";
    if (wsUrl.startsWith("/")) return `${scheme}//${u.host}${wsUrl}`;
    return `${scheme}//${u.host}/${wsUrl}`;
  } catch {
    return wsUrl;
  }
}

/**
 * Append the session capability token to the WebSocket URL.
 *
 * The backend requires it as a query parameter
 * (`GET /ws/sessions/{id}?token={ws_token}` — see docs/protocol.md). The
 * token is a required query parameter on that route, so omitting it makes
 * FastAPI reject the handshake before `accept()`, and the ASGI server then
 * answers the upgrade with `HTTP 403 Forbidden` — the connection never
 * reaches the WebSocket handler at all.
 *
 * The token is a `token_urlsafe()` capability string, so it is percent-encoded
 * to stay valid in a query value regardless of how it is generated.
 */
function withWsToken(wsUrl: string, token: string): string {
  if (!token) return wsUrl;
  const separator = wsUrl.includes("?") ? "&" : "?";
  return `${wsUrl}${separator}token=${encodeURIComponent(token)}`;
}
