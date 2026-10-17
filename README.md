# Face Liveness Platform

Platform for **passive face liveness detection** (anti-spoofing),
built on the existing research pipeline. The client captures a short RGB camera
sequence and streams it to a backend that runs the trained model and returns
the authoritative **live / spoof** verdict.

> **The model runs server-side, always.** The browser never computes the
> decision — it only captures, encodes, and streams frames.

---

## What's in here

```
face-liveness-platform/
├── services/face-liveness-api/     FastAPI + WebSocket + PyTorch inference
├── packages/face-liveness-sdk/     @mohammedabdelkrimguendouz/face-liveness-sdk (TypeScript)
├── apps/demo-web/                  Vite + React demo
├── docs/                           Protocol, deployment, troubleshooting
├── render.yaml                     Render blueprint (≥2 GB instance)
└── README.md                       (this file)
```

| Component | Stack | Entry point |
|-----------|-------|-------------|
| **API** | FastAPI, WebSocket, PyTorch | `services/face-liveness-api/app/main.py` |
| **SDK** | TypeScript, ESM + CJS, React bindings | `packages/face-liveness-sdk/src/` |
| **Demo** | Vite, React 18 | `apps/demo-web/src/App.tsx` |

---

## How it works

1. **Face gate** (on-device) — every frame is checked with MediaPipe Face
   Detector *before* upload; frames without a face are never sent, and a
   sustained absence aborts with a `no-face` error so the user can retry.
2. **Session** — client `POST /sessions` → gets `session_id`, `ws_token`,
   `num_frames` (150).
3. **Stream** — client opens `WS /ws/sessions/{id}?token=…` and sends 150
   frames captured at **10 fps (~15 s)**:
   `4-byte big-endian sequence index` + `JPEG payload`.
4. **Assemble** — server stores frames by index (order-independent), rejects
   duplicates/oversized/out-of-range, and auto-finalizes when all arrive.
5. **Infer** — EfficientNet-B2 extracts 1408-d features per frame → a BiLSTM
   undercomplete autoencoder reconstructs the sequence →
   `anomaly_score = 1 − exp(−MSE)`.
6. **Decide** — `score > 0.5` ⇒ `spoof`, else `live`. Returned as `result`.

Frames are then released; the score is an **anomaly score, not a
probability**.

Full wire spec: [`docs/protocol.md`](docs/protocol.md).

---

## The ML pipeline (unchanged)

This platform **reuses the existing trained model with no retraining and no
preprocessing changes**:

| Parameter | Value | Source |
|-----------|-------|--------|
| Input size | 288 × 288 | `config.IMAGE_SIZE` |
| Frames | 150 (uniform sampling) | `config.NUM_FRAMES` |
| Features | 1408 (EfficientNet-B2) | `config.FEATURE_SIZE` |
| Threshold | 0.5 | `config.THRESHOLD` |
| Score | `1 − exp(−MSE)` | `run_test.py` |
| Normalization | ImageNet mean/std, `/255`, NCHW, BGR→RGB | `feature_extractor.py` |

`app/ml/feature_extractor.py` (the EfficientNet-B2 extractor) is copied
**verbatim** from the research project; its unused video-file helpers were
removed in a cleanup. `bi_lstm_undercomplete.py` provides the single model
class used by the platform — it reconstructs the architecture that matches
the shipped checkpoint (`ABC_BiLSTM_Undercomplete_AE.pt`) — verified with a
`strict=True` state-dict load (0 missing / 0 unexpected keys).

Verified decisions on the bundled Oulu samples:

| Video | Score | Prediction |
|-------|-------|------------|
| `1_1_14_1_real_oulu.avi` | 0.038 | **live** ✓ |
| `1_1_14_3_printer2_oulu.avi` | 0.996 | **spoof** ✓ |

*(Full analysis and honest limitations: [`docs/IMPLEMENTATION_REPORT.md`](docs/IMPLEMENTATION_REPORT.md))*

---

## Quick start (local)

### 1. API

```bash
cd services/face-liveness-api
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # or pip3 on macOS/Linux
.venv/Scripts/python scripts/fetch_models.py          # verifies weights
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

By default the API looks for the weights committed to this repo at
`services/face-liveness-api/models/` (resolved by `app/paths.py`).
Override with `LIVENESS_MODELS_DIR` — a relative value is anchored to
`services/face-liveness-api/`, not to the working directory.

```bash
curl localhost:8000/ready   # {"ready":true,"device":"cpu","num_frames":150,...}
```

### 2. Demo

```bash
cd apps/demo-web
npm install
cp .env.example .env        # points at http://localhost:8000
npm run dev                 # http://localhost:5173
```

> The demo consumes the SDK from `packages/face-liveness-sdk` via a local
> `file:` path. `npm run dev` and `npm run build` first run `build:sdk`
> (`scripts/ensure-sdk.mjs`), which installs the SDK's dependencies and builds
> `dist/` if it is missing or stale — so a fresh clone works with no extra
> steps. The SDK's `dist/` is gitignored, so don't delete that hook.

Open in a browser, allow camera access, click **Start verification**.

> Camera access needs a secure context: `https://` or `http://localhost`.

### 3. SDK (standalone)

```bash
cd packages/face-liveness-sdk
npm install && npm run build
```

```ts
import { LivenessClient } from "@mohammedabdelkrimguendouz/face-liveness-sdk";
// or: import { useLiveness } from "@mohammedabdelkrimguendouz/face-liveness-sdk/react";

const result = await new LivenessClient({ apiUrl: "https://api.example.com" }).start();
console.log(result.prediction); // "live" | "spoof"
```

---

## Tests

```bash
# API — 71 tests (mocked protocol + 2 real-model integration tests)
cd services/face-liveness-api && .venv/Scripts/python -m pytest -v

# SDK — 21 tests (mocked camera/WebSocket/face detector)
cd packages/face-liveness-sdk && npm test

# Demo — typecheck + production build
cd apps/demo-web && npm run typecheck && npm run build
```

---

## Deployment

Deployment configs and exact commands for (re)deploying each component;
no credentials are stored here. The current verified status of each
component is listed under [Honest limitations](#honest-limitations) below.

| Target | Config | Guide |
|--------|--------|-------|
| API | `render.yaml` | [`docs/RENDER.md`](docs/RENDER.md) |
| Demo | `apps/demo-web/vercel.json` | [`docs/VERCEL.md`](docs/VERCEL.md) |
| SDK | — | [`docs/NPM.md`](docs/NPM.md) |

**Important:** Render needs a **≥2 GB instance** (measured peak RSS ≈ 960 MB;
512 MB free tier will be OOM-killed). Model weights (101.4 MB across the two
checkpoints) **are in git** at `services/face-liveness-api/models/` — there is
no download step and no external storage. Both files are under GitHub's
100 MB per-file limit and Git LFS is not used.

---

## Configuration reference

All via environment variables (see `app/config.py`):

| Variable | Default | Notes |
|----------|---------|-------|
| `LIVENESS_MODELS_DIR` | `models/` (i.e. `services/face-liveness-api/models/`) | Checkpoint directory. Relative values anchor to `services/face-liveness-api/`, never the cwd. |
| `LIVENESS_DEVICE` | `cpu` | |
| `LIVENESS_BATCH_SIZE` | `8` | Lower ⇒ less RAM, slower. |
| `LIVENESS_NUM_FRAMES` | `150` | Must match the checkpoint. |
| `LIVENESS_SESSION_TTL` | `300` | Idle session expiry (s). |
| `LIVENESS_MAX_SESSIONS` | `50` | Concurrent session cap. |
| `LIVENESS_MAX_CONCURRENT_INFERENCES` | `1` | Inference semaphore. |
| `LIVENESS_MAX_FRAME_BYTES` | `2097152` | Per-frame size cap. |
| `LIVENESS_ALLOWED_ORIGINS` | `http://localhost:5173` | CORS, comma-separated. |
| `VITE_LIVENESS_API_URL` | `http://localhost:8000` | Build-time, demo only. |

---

## Honest limitations

1. **Consumer-webcam domain gap.** The checkpoint was trained on Oulu
   cross-device data; a face from an ordinary laptop webcam can score as
   *spoof*. This also happened with the original research code (a real webcam
   video scored 0.9999 spoof) — it is a property of the model, not this
   platform. Retraining on your target camera would fix it; that was
   explicitly out of scope.
2. **`reverse_target` is inferred**, not recoverable from the checkpoint. Both
   values give near-identical scores (0.993 on a reference video either way),
   so the choice has negligible practical impact — but it is an inference, not
   a recovered fact.
3. **CPU-only inference** — ~8–10 s per verification on a typical vCPU
   (excluding upload).
4. **Camera capture is untested in CI** — it needs a physical webcam and user
   permission. The WS + model path *is* covered by the real-model tests.
5. **Deployment status.** The demo frontend is deployed on Vercel at
   <https://face-liveness-platform.vercel.app>. The API is deployed on
   Render at <https://face-liveness-api-892k.onrender.com> — `/health`,
   `/ready`, `POST /sessions`, and the WebSocket handshake were verified
   responding; browsers are only served cross-origin when their origin is
   listed in `LIVENESS_ALLOWED_ORIGINS` (see
   [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)). The SDK is
   published to npm as
   [`@mohammedabdelkrimguendouz/face-liveness-sdk`](https://www.npmjs.com/package/@mohammedabdelkrimguendouz/face-liveness-sdk)
   (latest `0.1.2`).

See [`docs/IMPLEMENTATION_REPORT.md`](docs/IMPLEMENTATION_REPORT.md) for
measured numbers and the full verification record.

---

## Team Contributions

**Mohammed Abdelkrim Guendouz — Project Developer**

Responsible for developing the LIVENOX platform, integrating the face liveness model with the backend API, implementing the TypeScript SDK and React demo integration, and preparing the project documentation and deployment.

---

## Further reading

- [`docs/protocol.md`](docs/protocol.md) — WebSocket wire protocol
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — failure modes & fixes
- [`docs/RENDER.md`](docs/RENDER.md) · [`docs/VERCEL.md`](docs/VERCEL.md) · [`docs/NPM.md`](docs/NPM.md) — deployment
- [`docs/IMPLEMENTATION_REPORT.md`](docs/IMPLEMENTATION_REPORT.md) — what was built, measured, verified
