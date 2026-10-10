/**
 * Public types for the face liveness SDK.
 */

/** Overall lifecycle phase, emitted via {@link LivenessClient.onPhaseChange}. */
export type LivenessPhase =
  | "idle"
  | "requesting-permission"
  | "connecting"
  | "capturing"
  | "transmitting"
  | "processing"
  | "completed"
  | "failed"
  | "cancelled";

/** Authoritative prediction returned by the backend. */
export type LivenessPrediction = "live" | "spoof";

/** Result of a successful verification. */
export interface LivenessResult {
  /** Authoritative decision computed server-side. */
  prediction: LivenessPrediction;
  /**
   * Anomaly score in [0, 1]. Higher = more anomalous (more likely spoof).
   * This is NOT a calibrated probability; it is `1 - exp(-reconstruction_error)`.
   */
  anomalyScore: number;
  /** Raw reconstruction error (MSE) the score derives from. */
  reconstructionError: number;
  /** Decision threshold used by the model. Score > threshold => spoof. */
  threshold: number;
  /** Number of frames the backend processed. */
  framesProcessed: number;
  /** Server-side inference duration in milliseconds. */
  inferenceMs: number;
  /** Session identifier (for support / audit). */
  sessionId: string;
}

/** Structured error surfaced by the SDK. */
export type LivenessErrorCode =
  | "unsupported"          // browser/camera API not available
  | "insecure-context"     // not https (or localhost)
  | "permission-denied"    // user or policy denied camera
  | "camera-error"         // camera failed to start / track ended
  | "connection-failed"    // could not reach backend / WS failure
  | "session-failed"       // backend rejected session creation
  | "timeout"              // operation exceeded its deadline
  | "cancelled"            // caller aborted
  | "capture-failed"       // frame capture error
  | "server-error"         // backend returned a protocol/processing error
  | "invalid-config";      // bad options supplied to the SDK

export interface LivenessProgress {
  phase: LivenessPhase;
  /** Frames captured so far. */
  captured: number;
  /** Frames the backend requires (e.g. 150). */
  required: number;
  /** Frames acknowledged as received by the backend (<= captured). */
  acknowledged: number;
  /** Human-readable status line. */
  message: string;
}

/** Options for {@link LivenessClient.start}. */
export interface LivenessOptions {
  /** Base URL of the liveness backend, e.g. `https://api.example.com`. */
  apiUrl: string;
  /** Preferred capture resolution width (height is derived). Default 720. */
  width?: number;
  /** Target capture frames-per-second. Default 20. */
  targetFps?: number;
  /** JPEG quality (1-100) for uploaded frames. Default 85. */
  jpegQuality?: number;
  /** Overall deadline for the whole verification, ms. Default 120000. */
  timeoutMs?: number;
  /** Device id to use for the camera (front camera by default). */
  deviceId?: string;
  /** Abort signal for external cancellation. */
  signal?: AbortSignal;
}

/** Callbacks accepted by {@link LivenessClient.start}. */
export interface LivenessCallbacks {
  onProgress?: (progress: LivenessProgress) => void;
  onPhaseChange?: (phase: LivenessPhase) => void;
  /**
   * Called once with the live camera stream so the host app can render a
   * preview (e.g. attach to a `<video>` element). The SDK stops the stream on
   * completion, cancellation, or error.
   */
  onStream?: (stream: MediaStream) => void;
  /** Called once when the backend returns an authoritative result. */
  onSuccess?: (result: LivenessResult) => void;
  /** Called on any terminal error. */
  onError?: (error: Error) => void;
  /** Called when verification is cancelled. */
  onCancel?: () => void;
}
