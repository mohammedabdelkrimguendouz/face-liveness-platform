import * as React from "react";
import {
  useLiveness,
  isLivenessError,
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
import logoUrl from "../logo.jfif";

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

/** Copy for the placeholder shown whenever no live camera frame is visible. */
function placeholderFor(phase: LivenessPhase): {
  spinner: boolean;
  title: string;
  hint?: string;
} {
  switch (phase) {
    case "requesting-permission":
      return { spinner: true, title: "Requesting camera access…" };
    case "connecting":
      return { spinner: true, title: "Starting the camera…" };
    case "processing":
      return { spinner: true, title: "Analyzing your video…" };
    case "failed":
    case "cancelled":
      return { spinner: false, title: "Camera released." };
    default:
      return {
        spinner: false,
        title: "Position your face in the frame.",
        hint: "The check runs passively — no action required.",
      };
  }
}

/** `1234 ms` below a second, `43.4 s` above — keeps result chips compact. */
function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
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
  } = useLiveness({ apiUrl: API_URL, targetFps: 10, jpegQuality: 85 });

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

  const step = phaseToStep(phase);
  const active = isActive(phase);
  const showPreview = active && phase !== "processing" && stream !== null;
  const pct = progress
    ? Math.min(100, Math.round((progress.captured / Math.max(1, progress.required)) * 100))
    : 0;

  return (
    <section className="card verify-card" aria-labelledby="verify-title">
      <div className="verify-head">
        <div className="verify-head__title">
          <img
            className="verify-head__logo"
            src={logoUrl}
            alt=""
            width={22}
            height={22}
            aria-hidden="true"
          />
          <h2 id="verify-title">Face Verification</h2>
        </div>
        <StatusPill phase={phase} result={result} />
      </div>

      <Steps current={step} />

      {/* ---- Result state ---- */}
      {phase === "completed" && result ? (
        <ResultView result={result} onStart={handleStart} />
      ) : (
        <>
          {/* ---- Camera preview / state placeholder ---- */}
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
              <Placeholder phase={phase} />
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
                <strong>
                  {isLivenessError(error) && error.code === "no-face"
                    ? "Face not detected."
                    : "Verification failed."}
                </strong>
                <div>{error.message}</div>
              </div>
            </div>
          ) : null}

          {/* ---- Actions ---- */}
          <div className="btn-row">
            {!isRunning ? (
              <button type="button" className="btn btn--primary" onClick={handleStart}>
                {phase === "failed" || phase === "cancelled" ? (
                  <>
                    <RefreshIcon width={18} height={18} /> Try again
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
    </section>
  );
}

function Placeholder({ phase }: { phase: LivenessPhase }): React.ReactElement {
  const { spinner, title, hint } = placeholderFor(phase);
  return (
    <div className="preview__placeholder">
      {spinner ? (
        <div className="spinner" role="status" aria-label={title} />
      ) : (
        <span className="preview__placeholder-icon" aria-hidden="true">
          <CameraIcon width={32} height={32} />
        </span>
      )}
      <p className="preview__placeholder-title">{title}</p>
      {hint ? <p className="preview__placeholder-hint">{hint}</p> : null}
    </div>
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
  } else if (phase === "cancelled") {
    label = "Cancelled";
  } else if (phase === "requesting-permission" || phase === "connecting") {
    cls += " pill--active";
    label = "Starting";
  } else if (phase === "processing") {
    cls += " pill--active";
    label = "Analyzing";
  } else if (phase === "capturing") {
    cls += " pill--active";
    label = "Capturing";
  } else if (phase === "transmitting") {
    cls += " pill--active";
    label = "Uploading";
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
  onStart,
}: {
  result: LivenessResult;
  onStart: () => void;
}): React.ReactElement {
  const live = result.prediction === "live";
  return (
    <div className="result">
      <div className={`result__badge result__badge--${live ? "live" : "spoof"}`}>
        {live ? (
          <CheckCircleIcon width={38} height={38} />
        ) : (
          <XCircleIcon width={38} height={38} />
        )}
      </div>

      <div className="result__text">
        <h3 className={`result__title ${live ? "result__live" : "result__spoof"}`}>
          {live ? "Live person verified" : "Spoof attempt detected"}
        </h3>
        <p className="result__desc">
          {live
            ? "A real person is present. You can continue."
            : "The sequence did not match a live face. A photo, screen, or mask may have been presented."}
        </p>
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="metric__label">Frames analyzed</div>
          <div className="metric__value">{result.framesProcessed}</div>
        </div>
        <div className="metric">
          <div className="metric__label">Analysis time</div>
          <div className="metric__value">{formatDuration(result.inferenceMs)}</div>
        </div>
      </div>

      <p className="result__ref">Verification ID: {result.sessionId.slice(0, 8)}</p>

      <button type="button" className="btn btn--primary" onClick={onStart}>
        <RefreshIcon width={18} height={18} /> Verify another
      </button>
    </div>
  );
}
