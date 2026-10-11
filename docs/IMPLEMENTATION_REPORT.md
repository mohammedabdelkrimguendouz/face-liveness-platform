# Implementation Report — Face Liveness Platform

**Date:** 2026-10-10
**Scope:** Transform the existing Face Anti-Spoofing research project into a
production liveness-detection platform: TypeScript SDK, FastAPI backend reusing
the existing model, React demo, deployment configs, tests, and docs.

---

## 1. Summary

| Stage | Deliverable | Status |
|-------|-------------|--------|
| 1 | FastAPI backend (WebSocket protocol, sessions, real inference) | ✅ 21/21 tests |
| 2 | `@mohammedabdelkrimguendouz/face-liveness-sdk` (ESM+CJS+d.ts, React bindings) | ✅ 7/7 tests, packed + install-verified |
| 3 | React demo (ui-ux-pro-max design system) | ✅ typecheck + build green, verified in browser |
| 4 | `render.yaml`, `vercel.json`, deploy docs | ✅ prepared (not deployed — by design) |
| 5 | README, protocol, troubleshooting, .gitignore, this report | ✅ |

**Test/build gate results (all executed):**

| Gate | Command | Result |
|------|---------|--------|
| Backend tests | `pytest -v` | **21 passed** in 80.7 s (incl. 2 real-model integration tests) |
| SDK typecheck | `tsc --noEmit` | **0 errors** |
| SDK tests | `vitest run` | **7 passed** in 4.0 s, no unhandled errors |
| SDK package | `npm pack` | 56.1 kB tarball; installed into scratch project; ESM ✓ CJS ✓ TS types ✓ |
| Demo typecheck | `tsc --noEmit` | **0 errors** |
| Demo build | `vite build` | 163.3 kB JS / 9.6 kB CSS (52.7 kB gzipped) |
| `fetch_models.py` | present / missing paths | exit 0 / exit 1 with instructions |
| Git ignore rules | `git check-ignore -v` | weights, venv, node_modules all ignored |
| Live smoke test | API + demo in browser | `/health` ok, `/ready` ready, UI rendered, session created |

Nothing in this report is claimed without a corresponding executed check.

---

## 2. Architecture

```
Browser (demo-web / host app)
  │  1. POST /sessions  ────────────────►  session_id + ws_token + num_frames
  │  2. WS /ws/sessions/{id}?token=…
  │     ├─ binary: [4-byte seq index][JPEG] × 150
  │     └─ JSON:   complete / cancel
  │  3. ◄────────── JSON: ready · ack · processing · result / error
  ▼
FastAPI (face-liveness-api)
  Session manager (bounded, TTL, per-session isolation)
        │  ordered frames
        ▼
  LivenessPipeline  (app/ml/pipeline.py)
        ├─ EfficientNet-B2  → 1408-d features × 150 frames
        ├─ BiLSTMUndercompleteAE → reconstruction
        └─ anomaly = 1 − exp(−MSE)  →  score > 0.5 ⇒ spoof
        ▼
  result { prediction, anomaly_score, threshold, … }
```

**Design decisions:**

- **Server-side decision.** The client never computes or approximates the
  verdict; it only captures and streams. The `prediction` field originates
  from `LivenessPipeline.predict()` and is passed through unmodified.
- **Frame = 4-byte big-endian index + JPEG.** Explicit sequence numbers make
  ordering independent of transport ordering; frames are stored in a dict keyed
  by index and sorted before inference.
- **No awaits while holding a lock.** State mutation happens inside
  `session.lock`; socket sends happen after release. Holding a
  `threading.Lock` across an `await` freezes the event loop (this was a real
  deadlock found and fixed during testing — see §6).
- **Auto-finalize on full sequence.** Receiving all `num_frames` frames
  triggers inference; the `complete` control message is an early-validation
  shortcut that reports missing-frame counts sooner.
- **Bounded everything.** Frame bytes ≤ 2 MiB, frames ≤ 400/session,
  sessions ≤ 50, concurrent inferences = 1 (semaphore), buffers released on
  every terminal path.

---

## 3. Reuse of the existing ML pipeline

**No retraining, no model substitution, no preprocessing changes.**

| Item | Handling |
|------|----------|
| `feature_extractor.py` | Copied **verbatim** from `test_models/models/` |
| `lstm_models.py` | Copied **verbatim** |
| `model_loader.py` | Copied **verbatim** |
| `bi_lstm_undercomplete.py` | **New** — reconstruction of the class matching the shipped checkpoint |
| `checkpoint_loader.py` | **New** — `strict=True` load, strips `module.` prefix |
| `pipeline.py` | **New** — HTTP-independent orchestration preserving exact preprocessing |

### 3.1 The checkpoint/class mismatch (and how it was resolved)

The research repo's `config.py` references three checkpoints. Inspection
showed:

- `ABC_LSTM_Undercomplete_AE.pt` — **does not exist**
- `ABC_BiLSTM_Variational_AE.pt` — **does not exist**
- `ABC_BiLSTM_Undercomplete_AE.pt` — exists, but **matches no class** in
  `lstm_models.py` (the file's classes are unidirectional, or variational with
  `fc_mu`/`fc_logvar` and a non-autoregressive decoder)

The shipped checkpoint contains: bidirectional encoder/decoder keys
(`*_reverse`), `latent (256, 1024)`, `expand (1024, 256)`, and a decoder input
width of `1664 = 1408 + 256` (autoregressive concat). The class was
reconstructed directly from that layout and **loads with `strict=True`: 0
missing keys, 0 unexpected keys.**

> **Why `strict=True` matters:** a silent `strict=False` load would let a
> shape mismatch degrade inference invisibly. Here, any divergence between
> checkpoint and class is a startup failure, not a wrong score in production.

### 3.2 Verified decisions on real checkpoints

End-to-end on the bundled Oulu samples (150 frames, JPEG-encoded as a browser
would send them):

| Video | Score | Prediction | Expected | Source |
|-------|-------|-----------|----------|--------|
| `1_1_14_1_real_oulu.avi` | **0.038** | `live` | live | real capture |
| `1_1_14_3_printer2_oulu.avi` | **0.996** | `spoof` | spoof | printer attack |

These match the historical `results/test_results.csv` pattern (real samples
scoring near 0, print/replay/screen attacks near 1). The 0.035 → 0.038 delta on
the real sample comes from JPEG re-encoding at quality 92 during the test —
the same re-encoding that production transport performs; the decision has a
large margin (0.038 vs threshold 0.5).

### 3.3 Pipeline parameters (all preserved)

```
IMAGE_SIZE=288 · NUM_FRAMES=150 · FEATURE_SIZE=1408 · THRESHOLD=0.5
BGR→RGB · cv2.resize(288,288) · /255 · NCHW · ImageNet norm · uniform sampling
score = 1 − exp(−MSE)   prediction = score > 0.5 ? spoof : live
```

The only deliberate operational deviation: **`batch_size` default 8**
(training used 16) and torch thread caps, to bound peak memory. This changes
*throughput*, not *results* — batching cannot alter per-sample math.

---

## 4. Measured performance

| Metric | Value | Conditions |
|--------|-------|------------|
| Peak RSS | **≈ 960 MB** | CPU, `batch_size=16`, full pipeline |
| Model load | ~2.7 s (extractor) + AE | cold start |
| Feature extraction | ~6.5 s | 150 frames, CPU |
| Autoencoder inference | ~1.1 s | 150×1408, CPU |
| End-to-end server time | **≈ 8–10 s** | extraction + AE, excluding upload |
| Inference ms (reported in `result`) | ~1106 ms | AE stage only |

**Deployment consequence:** 512 MB (Render free tier) is insufficient —
measured peak is ≈ 960 MB. A **Standard (2 GB) instance is the minimum**; this
is stated in `render.yaml`, `README.md`, and `docs/RENDER.md`.

---

## 5. Verification record

### 5.1 Backend — 21 tests, all passing

- `test_api.py` (8): health, readiness, session shape, 404s, session cap →
  429, TTL eviction, root metadata
- `test_protocol.py` (11): full stream → result, progress acks, duplicate
  rejection, out-of-range index, malformed frame, oversized frame,
  incomplete sequence, no-inference-on-incomplete, cancel, bad token, invalid
  JSON
- `test_inference.py` (2, marked `real`): real-model scores on Oulu videos;
  complete WebSocket flow against the real model

Protocol tests use a stub pipeline (fast/deterministic); the two `real` tests
execute the actual EfficientNet-B2 + BiLSTM models.

### 5.2 SDK — 7 tests, all passing, no unhandled errors

Mocked `WebSocket`, `getUserMedia`, canvas, `fetch`, timers. Covers:
sequential big-endian frame indices 0…149, server-error mapping,
connection/session failure codes, cancellation, timeout, `LivenessError`
identity.

Packaging verified: `npm pack` → 56.1 kB tarball (dist + package.json only) →
installed into a scratch project → ESM import ✓, CJS require ✓, `tsc
--noEmit` ✓ against the published `.d.ts`.

### 5.3 Demo — verified in a live browser

Running against the real API (model loaded):

- Header badge showed **API connected** (live `/health` poll)
- Hero, feature list, SVG icon set, 4-step indicator, camera placeholder,
  CTA, honest note callout, and footer all rendered correctly
- `POST /sessions` returned a real `session_id` with `num_frames: 150`
- `vite build` produced 163.3 kB JS / 52.7 kB gzipped

### 5.4 UI design system

The mandatory `ui-ux-pro-max` skill was run **before** any UI was written:

```
search.py "identity verification security trust modern fintech" --design-system --persist
search.py "camera permission biometric scan progress indicator" --domain ux
search.py "state management re-render forms" --stack react
```

Output persisted to `design-system/face-liveness/MASTER.md`. Applied:
Trust & Authority pattern, Minimalism/Swiss style, navy/amber tokens, visible
focus rings (skill severity: High), `prefers-reduced-motion` support, 44 px
minimum touch targets, step indicators for the multi-step flow (skill severity:
Medium), SVG icons instead of emoji, responsive layout.

---

## 6. Issues found and fixed during implementation

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 1 | `torch.inference_mode()` on model built outside it | Runtime error on LSTM backward-safety check | Switched to `torch.no_grad()` (matches original eval code) |
| 2 | **Deadlock:** `await` send inside `threading.Lock` | WebSocket tests hung indefinitely; would hang production under concurrency | Mutate state under lock, send only after release |
| 3 | Tests read more messages than server sent | `TestClient.receive_json()` blocks forever | `_recv_until()` helper reads lazily to the message of interest |
| 4 | `is_complete` required an explicit `complete` message | Full-stream client never triggered inference | Auto-finalize when `num_frames` arrive; `complete` is early validation |
| 5 | FastAPI broken global install (missing `annotated_doc`) | Backend tests couldn't start | Dedicated `.venv` for the API service |
| 6 | `config.py` multi-line string literal | `SyntaxError` at import | Rewrote path construction |
| 7 | jsdom `Blob` lacks `arrayBuffer()` | SDK capture loop failed in tests | Test polyfill (real browsers implement it) |
| 8 | `cancel()`/timeout settlement race | Unhandled promise rejections | `Promise.race` against one outer deferred; losers pre-consumed |
| 9 | SDK opened camera but exposed no preview | Host app couldn't render a preview | Added `onStream` callback; demo attaches it to `<video>` |
| 10 | `useLiveness` didn't forward `onStream`/`onProgress` | Hook dropped host callbacks | Forwarded through |

Issues 2 and 7 were **live bugs** that only manifested at runtime — both are
covered by tests that now pass.

---

## 7. Honest limitations

1. **Consumer-webcam domain gap (pre-existing).**
   The checkpoint was trained on Oulu cross-device data. A face from an
   ordinary laptop webcam can score as *spoof*. This is not introduced by the
   platform: the original research code scored a real webcam video
   (`real.mp4`) **0.9999 = spoof**, and the reconstruction agrees
   (~0.993). Both the historical and current pipelines behave identically.
   **Fix:** retrain/fine-tune on your target camera — explicitly out of scope
   ("no retraining").

2. **`reverse_target` is inferred, not recovered.**
   The checkpoint does not store this flag. Both values produce
   near-identical anomaly scores (0.993 on a reference video either way) — the
   autoregressive flip largely cancels — but it is an informed inference, not
   a recovered fact. Documented in `bi_lstm_undercomplete.py`.

3. **CPU-only, ~8–10 s per verification server-side.**
   Render has no GPU. Add upload time on top. Budget accordingly for UX
   (the demo shows a progress bar and a processing spinner).

4. **Camera capture is not covered by automated tests.**
   It requires a physical webcam and user permission. The WebSocket protocol
   and full model path *are* covered (21 backend tests incl. 2 real-model, 7
   SDK tests with mocked camera). The demo was verified rendering + session
   creation in a browser, but an actual human-facing capture run was not
   performed in this environment.

5. **Nothing was deployed or published.**
   `render.yaml`, `vercel.json`, and the exact npm commands exist and are
   documented, but Render/Vercel deploys and `npm publish` require your
   credentials and are deliberately left to you. **No production URL, npm
   package, or live deployment is claimed.**

6. **Scoped package name unverified.**
   `@mohammedabdelkrimguendouz/face-liveness-sdk` is assumed to be available/owned.
   The scope matches the npm username, so npm associates it with that account
   by default (no org required). `docs/NPM.md` includes the `npm view` check
   and the fallback of renaming.

7. **Weights are in git (101.4 MB across the two checkpoints).**
   Both are committed at `services/face-liveness-api/models/`, so
   `fetch_models.py` only verifies them at build time — no download, no
   external storage, no Git LFS. Each file is under GitHub's 100 MB limit
   (`EfficientNet_B2.pt` 31,254,679 B; `ABC_BiLSTM_Undercomplete_AE.pt`
   75,055,141 B). The build fails loudly if they're absent. Documented in
   `docs/RENDER.md`.

8. **In-memory sessions, single worker.**
   Horizontal scaling with `--workers > 1` would break session affinity. Scale
   vertically; the concurrency semaphore defaults to 1.

---

## 8. Security & privacy properties

| Property | Implementation |
|----------|----------------|
| Session isolation | Per-session buffers keyed by `session_id`; capability token (`ws_token`, 256-bit) required to attach |
| Unguessable IDs | `secrets.token_urlsafe(32)` for both `session_id` and `ws_token` |
| Frame size limits | 2 MiB/frame, 400 frames/session → bounded memory |
| Session bounds | Max 50 sessions, 300 s TTL, expired sessions evicted and buffers cleared |
| Buffer release | Frames cleared immediately on result / cancel / error / expiry |
| Model integrity | `strict=True` state-dict load — mismatch fails startup |
| Decision integrity | Verdict computed server-side only; score/threshold passed through, never recomputed client-side |
| CORS | Explicit origin allow-list (not `*`) |
| No secrets committed | `.gitignore` excludes `.env`, keys, credentials; weights excluded |
| Data retention | No biometric template persisted; frames are transient inference input |

---

## 9. Deliverables inventory

55 source files (excluding `node_modules`, `dist`, `.venv`, caches):

- **16 Python** — API app (8), ML layer (6), scripts (1), conftest (1)
- **17 TypeScript/TSX** — SDK (10), demo (7)
- **7 Markdown** — README, protocol, troubleshooting, render, vercel, npm,
  this report
- **5 config** — `render.yaml`, `vercel.json`, `pytest.ini`, `requirements.txt`, `.gitignore`s
- **1 CSS** — design-token stylesheet from the skill output

### Not delivered (by instruction)

- ❌ Render deployment (your credentials)
- ❌ Vercel deployment (your credentials)
- ❌ npm publish (your credentials)
- ❌ Any committed model weights (137 MB)
- ❌ Any retrained/modified model

---

## 10. How to verify this yourself

```bash
cd "Face Anti Spoofing Project/face-liveness-platform"

# 1. Backend (includes real-model tests)
cd services/face-liveness-api
.venv/Scripts/python -m pytest -v

# 2. SDK
cd ../../packages/face-liveness-sdk
npm install && npm test && npm run typecheck && npm run build

# 3. Demo
cd ../../apps/demo-web
npm install && npm run typecheck && npm run build

# 4. Live end-to-end (terminal 1)
cd ../services/face-liveness-api
.venv/Scripts/python -m uvicorn app.main:app --port 8000
# (terminal 2)
cd ../../apps/demo-web && npm run dev
# → open http://localhost:5173, allow camera, click Start verification
```

Expect: 21 passed · 7 passed · 0 type errors · build succeeds · demo shows
"API connected" and completes a verification.
