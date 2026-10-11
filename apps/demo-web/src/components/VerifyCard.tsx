import * as React from "react";
import {
  useLiveness,
  type LivenessResult,
  type LivenessPhase,
} from "@mohammedabdelkrimguendouz/face-liveness-sdk/react";
import {
  CameraIcon,
  CheckCircleIcon,
  ExclamationIcon,
  RefreshIcon,
  StopIcon,
  XCircleIcon,
} from "./icons";

const API_URL =
  (import.meta.env.VITE_LIVENESS_API_URL as string | undefined) ??
  "http://localhost:8000";

/** Phase -> step index (0=idle, 1=permission, 2=capture, 3=processing, 4=done). */
function phaseToStep(phase: LivenessPhase): number {
  switch (phase) {
    case "idle":
    case "requesting-permission":
      return 1;
    case "connecting":
    case "capturing":
    case "transmitting":
      return 2;
    case "processing":
      return 3;
    case "completed":
    case "failed":
    case "cancelled":
      return 4;
    default:
      return 0;
  }
}

function isActive(phase: LivenessPhase): boolean {
  return (
    phase === "requesting-permission" ||
    phase === "connecting" ||
    phase === "capturing" ||
    phase === "transmitting" ||
    phase === "processing"
  );
}

export function VerifyCard(): React.ReactElement {
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const {
    phase,
    progress,
    result,
    error,
    isRunning,
    start,
    cancel,
    reset,
  } = useLiveness({ apiUrl: API_URL, targetFps: 20, jpegQuality: 85 });

  const [stream, setStream] = React.useState<MediaStream | null>(null);

  // Attach the SDK's camera stream to our preview <video>.
  React.useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.srcObject = stream;
      if (stream) void video.play().catch(() => undefined);
    }
  }, [stream]);

  // Release the preview on unmount.
  React.useEffect(() => {
    return () => setStream(null);
  }, []);

  const handleStart = React.useCallback(() => {
    setStream(null);
    void start({
      onStream: (s: MediaStream) => setStream(s),
      onCancel: () => setStream(null),
      onError: () => setStream(null),
      onSuccess: () => setStream(null),
    });
  }, [start]);

  const handleReset = React.useCallback(() => {
    setStream(null);
    reset();
  }, [reset]);

  const step = phaseToStep(phase);
  const active = isActive(phase);
  const showPreview = active && phase !== "processing" && stream !== null;
  const pct = progress
    ? Math.min(100, Math.round((progress.captured / Math.max(1, progress.required)) * 100))
    : 0;

  return (
    <section className="card verify-card" aria-labelledby="verify-title">
      <div className="verify-head">
        <h2 id="verify-title">Liveness check</h2>
        <StatusPill phase={phase} result={result} />
      </div>

      <Steps current={step} />

      {/* ---- Result state ---- */}
      {phase === "completed" && result ? (
        <ResultView result={result} onReset={handleReset} />
      ) : (
        <>
          {/* ---- Camera preview / placeholder ---- */}
          <div className="preview">
            {showPreview ? (
              <video
                ref={videoRef}
                className="preview__video"
                playsInline
                muted
                autoPlay
                aria-label="Live camera preview"
              />
            ) : (
              <div className="preview__placeholder">
                {phase === "processing" ? (
                  <>
                    <div className="spinner" role="status" aria-label="Processing" />
                    <p>Analyzing your frames…</p>
                  </>
                ) : phase === "failed" || phase === "cancelled" ? (
                  <>
                    <CameraIcon width={36} height={36} />
                    <p>Camera released.</p>
                  </>
                ) : (
                  <>
                    <CameraIcon width={36} height={36} />
                    <p>
                      Position your face in the frame. The check runs passively —
                      no action required.
                    </p>
                  </>
                )}
              </div>
            )}
            {(active || phase === "processing") && (
              <span
                className={`preview__guide${active ? " preview__guide--active" : ""}`}
                aria-hidden="true"
              />
            )}
          </div>

          {/* ---- Progress ---- */}
          {(phase === "capturing" || phase === "transmitting") && progress ? (
            <div>
              <div className="progress-label">
                <span>{progress.message}</span>
                <span aria-hidden="true">{pct}%</span>
              </div>
              <div
                className="progress"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={progress.required}
                aria-valuenow={progress.captured}
                aria-label="Frames captured"
              >
                <div className="progress__fill" style={{ width: `${pct}%` }} />
              </div>
              <p className="sr-only" aria-live="polite">
                {progress.captured} of {progress.required} frames captured
              </p>
            </div>
          ) : null}

          {/* ---- Error ---- */}
          {phase === "failed" && error ? (
            <div className="alert alert--error" role="alert">
              <ExclamationIcon className="alert__icon" />
              <div>
                <strong>Verification failed.</strong>
                <div>{error.message}</div>
              </div>
            </div>
          ) : null}

          {/* ---- Actions ---- */}
          <div className="btn-row">
            {!isRunning ? (
              <button
                type="button"
                className="btn btn--primary"
                onClick={phase === "failed" || phase === "cancelled" ? handleReset : handleStart}
              >
                {phase === "failed" || phase === "cancelled" ? (
                  <>
                    <RefreshIcon width={18} height={18} /> Try again
                  </>
                ) : phase === "completed" ? (
                  <>
                    <RefreshIcon width={18} height={18} /> Verify another
                  </>
                ) : (
                  <>
                    <CameraIcon width={18} height={18} /> Start verification
                  </>
                )}
              </button>
            ) : (
              <button type="button" className="btn btn--ghost" onClick={cancel}>
                <StopIcon width={18} height={18} /> Cancel
              </button>
            )}
          </div>
        </>
      )}

      <p className="note">
        The model runs server-side. Frames are streamed over WebSocket, analyzed,
        and the authoritative <strong>live / spoof</strong> decision is made by the
        backend — never in your browser.
      </p>
    </section>
  );
}

function StatusPill({
  phase,
  result,
}: {
  phase: LivenessPhase;
  result: LivenessResult | null;
}): React.ReactElement {
  let cls = "pill";
  let label: string;

  if (phase === "completed" && result) {
    if (result.prediction === "live") {
      cls += " pill--live";
      label = "Live person";
    } else {
      cls += " pill--spoof";
      label = "Spoof detected";
    }
  } else if (phase === "failed") {
    cls += " pill--failed";
    label = "Failed";
  } else if (isActive(phase)) {
    cls += " pill--active";
    label = phase === "processing" ? "Analyzing" : "In progress";
  } else if (phase === "cancelled") {
    label = "Cancelled";
  } else {
    label = "Ready";
  }

  return (
    <span className={cls}>
      <span className="pill__dot" aria-hidden="true" />
      {label}
    </span>
  );
}

function Steps({ current }: { current: number }): React.ReactElement {
  const labels = ["Permission", "Capture", "Analyze", "Result"];
  return (
    <ol className="steps" aria-label={`Step ${current} of 4`}>
      {labels.map((label, i) => {
        const n = i + 1;
        const state =
          n < current ? "step step--done" : n === current ? "step step--current" : "step";
        return (
          <li key={label} className={state} aria-current={n === current ? "step" : undefined}>
            <span className="step__num" aria-hidden="true">
              {n < current ? "✓" : n}
            </span>
            {label}
          </li>
        );
      })}
    </ol>
  );
}

function ResultView({
  result,
  onReset,
}: {
  result: LivenessResult;
  onReset: () => void;
}): React.ReactElement {
  const live = result.prediction === "live";
  return (
    <div className="result">
      <div className={`result__badge result__badge--${live ? "live" : "spoof"}`}>
        {live ? (
          <CheckCircleIcon width={40} height={40} />
        ) : (
          <XCircleIcon width={40} height={40} />
        )}
      </div>
      <h3 className={`result__title ${live ? "result__live" : "result__spoof"}`}>
        {live ? "Live person verified" : "Spoof attempt detected"}
      </h3>
      <p>
        {live
          ? "The sequence reconstructed as expected for a real face."
          : "The sequence did not match a live face. A printed photo, screen replay, or mask may have been presented."}
      </p>

      <div className="metrics">
        <div className="metric">
          <div className="metric__label">Anomaly score</div>
          <div className="metric__value">{result.anomalyScore.toFixed(4)}</div>
        </div>
        <div className="metric">
          <div className="metric__label">Threshold</div>
          <div className="metric__value">{result.threshold.toFixed(2)}</div>
        </div>
        <div className="metric">
          <div className="metric__label">Frames</div>
          <div className="metric__value">{result.framesProcessed}</div>
        </div>
        <div className="metric">
          <div className="metric__label">Server inference</div>
          <div className="metric__value">{Math.round(result.inferenceMs)} ms</div>
        </div>
      </div>

      <p className="note">
        Score = 1 − exp(−reconstruction error). Higher means more anomalous; a
        score above the threshold is classified as a spoof. This is an anomaly
        score, not a calibrated probability.
      </p>

      <button type="button" className="btn btn--primary" onClick={onReset}>
        <RefreshIcon width={18} height={18} /> Verify another
      </button>
    </div>
  );
}
