# Troubleshooting

## Backend

### `/ready` returns 503

```bash
curl https://your-api/ready
# {"ready":false,"error":"..."}
```

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Model checkpoint not found: …` | Weights absent from `services/face-liveness-api/models/` | Restore `EfficientNet_B2.pt` and `ABC_BiLSTM_Undercomplete_AE.pt` there (they are committed to the repo), or point `LIVENESS_MODELS_DIR` at them. See `docs/RENDER.md`. |
| `size mismatch for …` | Wrong checkpoint / class mismatch | Use `ABC_BiLSTM_Undercomplete_AE.pt` with `bi_lstm_undercomplete.BiLSTMUndercompleteAE`. The loader uses `strict=True` by design. |
| Build fails in `fetch_models.py` | Checkpoint(s) missing on disk | The script verifies — it does not download. Both `EfficientNet_B2.pt` and `ABC_BiLSTM_Undercomplete_AE.pt` must exist in `services/face-liveness-api/models/`; the error prints the resolved path. |
| Process starts, then OOM-killed | Instance < 2 GB | Measured peak ≈ 960 MB. Use Render Standard (2 GB)+. |

`/health` only reports the process is alive — always check `/ready` for model
state.

### WebSocket closes immediately

| Close code | Meaning |
|------------|---------|
| `4401` | Wrong/expired `ws_token` (or missing `?token=`). |
| `4404` | Session not found — TTL expired (300 s) or already evicted. |
| `4409` | Session already consumed; create a new one. |
| `1006` (client) | Network drop or the API is behind a proxy without WebSocket support. |

**Render/proxy note:** Render's proxy supports WebSockets, but some
CDNs/proxies time out idle connections. The client sends frames every ~50 ms,
so this shouldn't trigger — but if you front the API with another proxy, ensure
WebSocket upgrade is enabled and the idle timeout exceeds 60 s.

### `Origin ... not allowed` (CORS)

The API's CORS middleware only permits origins in
`LIVENESS_ALLOWED_ORIGINS` (comma-separated). Set it to your exact Vercel
origin, no trailing slash:

```
LIVENESS_ALLOWED_ORIGINS=https://your-demo-app.vercel.app
```

Remember to **restart/re-deploy** — it's read at startup.

### Client gets `duplicate_frame` / `bad_index`

- `duplicate_frame`: the SDK sends each index once; a retry after a network
  hiccup can legitimately resend. Recoverable — resend a *different* index or
  ignore.
- `bad_index`: client/server disagree on `num_frames`. The client takes
  `num_frames` from the `ready` message — if you changed
  `LIVENESS_NUM_FRAMES` on the server, redeploy the client config too.

### `incomplete_sequence` on `complete`

The client declared completion before all `num_frames` frames arrived. The
`missing` field says how many. Usually the capture loop exited early — check
for camera errors on the client.

### Inference is slow

CPU-only by design (Render has no GPU). Typical: ~6.5 s feature extraction +
~1.1 s autoencoder = **~8–10 s** server-side per verification, plus upload.
Lower `LIVENESS_BATCH_SIZE` reduces RAM but *slows* inference; raise it (16) if
you have ≥2 GB and need speed.

---

## Frontend / SDK

### Camera errors

| Code | Message | Fix |
|------|---------|-----|
| `insecure-context` | Camera requires HTTPS | Use `https://` or `http://localhost`. |
| `permission-denied` | User/policy denied camera | Ask the user to allow access; check browser site settings. |
| `camera-error` (no camera) | `NotFoundError` | Device has no camera. |
| `camera-error` (in use) | `NotReadableError` | Another app holds the camera. |
| `unsupported` | `getUserMedia` missing | Old/insecure browser. |

### `connection-failed` / `session-failed`

- `session-failed` with HTTP `503` → API `/ready` failed; fix weights first.
- `connection-failed` → wrong `VITE_LIVENESS_API_URL`, API down, or CORS
  blocked the `POST /sessions`.
- The demo's header badge shows **API offline** when `/health` fails — start
  there.

### Verification times out

Default `timeoutMs` is 120 000 ms for the whole flow (permission + capture +
upload + inference). On a slow uplink, 150 JPEGs can take a while. Raise
`timeoutMs`, or lower `jpegQuality` / `targetFps` in the SDK options.

### Frames rejected as `frame_too_large`

Default `max_frame_bytes` is 2 MiB. High-resolution cameras at high JPEG
quality can exceed this. Lower `width` (default 720) or `jpegQuality`
(default 85).

### Score looks wrong for a real face

**Known model limitation (not a bug):** the checkpoint was trained on Oulu
cross-device data. Consumer-webcam faces can score as spoof — this was
observed with the *original* research code too (a real webcam video scored
0.9999 = spoof before any platform work). See `docs/IMPLEMENTATION_REPORT.md`
→ *Limitations*.

---

## Running tests

```bash
# Backend (21 tests; includes real-model integration)
cd services/face-liveness-api
.venv/Scripts/python -m pytest -v

# Skip slow real-model tests:
python -m pytest -m "not real" -v

# SDK (7 tests, mocked browser APIs)
cd packages/face-liveness-sdk
npm test

# Demo typecheck + build
cd apps/demo-web
npm run typecheck && npm run build
```

**Not covered by CI:** actual browser camera capture (requires a physical
webcam + user permission) and the live end-to-end demo over a real network.
