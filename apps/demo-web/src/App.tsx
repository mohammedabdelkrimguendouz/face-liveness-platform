import * as React from "react";
import { VerifyCard } from "./components/VerifyCard";
import { BoltIcon, EyeIcon, LockClosedIcon, ShieldCheckIcon } from "./components/icons";

const API_URL =
  (import.meta.env.VITE_LIVENESS_API_URL as string | undefined) ??
  "http://localhost:8000";

export default function App(): React.ReactElement {
  const [apiUp, setApiUp] = React.useState<boolean | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL.replace(/\/+$/, "")}/health`)
      .then((r) => {
        if (!cancelled) setApiUp(r.ok);
      })
      .catch(() => {
        if (!cancelled) setApiUp(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="container topbar__inner">
          <div className="brand">
            <span className="brand__mark" aria-hidden="true">
              <ShieldCheckIcon width={16} height={16} />
            </span>
            Face Liveness
          </div>
          <span className="brand__badge">
            {apiUp === null ? "Checking API…" : apiUp ? "API connected" : "API offline"}
          </span>
        </div>
      </header>

      <main className="container">
        <div className="hero">
          <div>
            <div className="eyebrow">
              <EyeIcon width={14} height={14} />
              Passive anti-spoofing
            </div>
            <h1>Confirm a real person is in front of the camera.</h1>
            <p className="lede">
              A short RGB camera sequence is streamed to a server-side model that
              measures how well it reconstructs as a live face. No passwords, no
              action prompts, no biometric template stored.
            </p>

            <ul className="feature-list">
              <li>
                <ShieldCheckIcon className="icon" />
                <div>
                  <strong>Server-side decision</strong>
                  <span>
                    The live / spoof verdict is computed by the model on the
                    backend — the client only captures and streams frames.
                  </span>
                </div>
              </li>
              <li>
                <BoltIcon className="icon" />
                <div>
                  <strong>Passive by design</strong>
                  <span>
                    ~150 frames in sequence over a few seconds. No blink, nod, or
                    text prompt required.
                  </span>
                </div>
              </li>
              <li>
                <LockClosedIcon className="icon" />
                <div>
                  <strong>Minimal data</strong>
                  <span>
                    Frames are analyzed for reconstruction error and released.
                    Nothing is persisted as a biometric template.
                  </span>
                </div>
              </li>
            </ul>
          </div>

          <VerifyCard />
        </div>
      </main>

      <footer className="footer">
        <div className="container footer__inner">
          <span>
            Face Liveness Platform — EfficientNet-B2 features + BiLSTM
            undercomplete autoencoder.
          </span>
          <span>Anomaly score = 1 − exp(−MSE), threshold 0.5</span>
        </div>
      </footer>
    </div>
  );
}
