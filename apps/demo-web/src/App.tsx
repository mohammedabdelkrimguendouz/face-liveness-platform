import * as React from "react";
import { VerifyCard } from "./components/VerifyCard";
import { BoltIcon, EyeIcon, LockClosedIcon } from "./components/icons";
import logoUrl from "./logo.jfif";

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
            <img
              className="brand__logo"
              src={logoUrl}
              alt=""
              width={36}
              height={36}
              aria-hidden="true"
            />
            <span className="brand__name">LIVENOX</span>
          </div>

          <span
            className={
              apiUp === null
                ? "status"
                : apiUp
                  ? "status status--up"
                  : "status status--down"
            }
          >
            <span className="status__dot" aria-hidden="true" />
            {apiUp === null
              ? "Checking service…"
              : apiUp
                ? "Service online"
                : "Service offline"}
          </span>
        </div>
      </header>

      <main className="main">
        <div className="container main__inner">
          <div className="intro">
            <p className="eyebrow">Passive liveness detection</p>
            <h1>Confirm a real person is in front of the camera.</h1>
            <p className="lede">
              Keep your face in view for about 15 seconds — no passwords, no
              action prompts.
            </p>
          </div>

          <VerifyCard />

          <ul className="trust">
            <li className="trust__item">
              <EyeIcon className="trust__icon" />
              <div>
                <strong>Passive by design</strong>
                <span>No blink, nod, or prompts — just look at the camera.</span>
              </div>
            </li>
            <li className="trust__item">
              <LockClosedIcon className="trust__icon" />
              <div>
                <strong>Nothing is stored</strong>
                <span>
                  Frames are used for this check only; no template is kept.
                </span>
              </div>
            </li>
            <li className="trust__item">
              <BoltIcon className="trust__icon" />
              <div>
                <strong>Done in seconds</strong>
                <span>Your result appears as soon as the check completes.</span>
              </div>
            </li>
          </ul>
        </div>
      </main>
    </div>
  );
}
