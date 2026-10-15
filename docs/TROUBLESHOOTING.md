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

### WebSocket fails as `403 Forbidden` (handshake rejected)

The browser console shows only:

```
WebSocket connection to 'wss://…/ws/sessions/<id>' failed:
```

with no code, and Render logs `INFO: "WebSocket /ws/sessions/…" 403` followed by
`INFO: connection rejected (403 Forbidden)`.

That 403 is produced **before** the socket is accepted — it is *not* an
application-level close, so none of the `44xx` codes below ever reach the
client. Look for the API's own diagnostic line in the same logs:

| Log line | Cause | Fix |
|----------|-------|-----|
| `reason=token_missing close_code=4401` | `?token=` absent from the URL | Client must send `GET /ws/sessions/{id}?token={ws_token}` (`docs/protocol.md` §2). |
| `reason=token_invalid close_code=4401` | Wrong `ws_token` | Use the token from the `POST /sessions` response for *that* session. |
| `reason=session_not_found_or_expired close_code=4404` | Unknown, expired (300 s) or already-consumed id | Create a new session; a session is destroyed when its socket disconnects. |
| `reason=invalid_state:… close_code=4409` | Session already consumed | Create a new session. |

If `ws_rejected` is **absent** and only uvicorn's bare `403` appears, the
request never reached the route — check the proxy path and that the URL is
`wss://` (not `ws://`) when the page is served over HTTPS.

### Verification fails after `Inference done` (close code 1011)

The model answered — `app.ml.pipeline Inference done: score=… pred=…` proves
it — but the browser reported `Verification failed / Connection closed (code
1011)`.

A successful inference and a delivered result are **different events**. Read
the stage lines for the same 8-character session prefix, in order:

| Log line | Meaning |
|----------|---------|
| `ws_accepted session=…` | Handshake accepted; `elapsed_ms` on later lines is counted from here. |
| `inference_scheduled session=… frames_received=… last_frame_index=…` | The 150th frame arrived and the model was queued. |
| `inference_started session=… queued_ms=…` | The model actually began. |
| `ws_client_gone session=… state=… via=… inference_running=… elapsed_ms=…` | **The decisive line when the client leaves early.** `inference_running=True` means the socket died *during* the model's run, and `elapsed_ms` says exactly when. `via=message` / `via=exception` records which path revealed it. Absent when the client stayed connected. |
| `inference_completed session=… prediction=… score=… ms=…` | The model answered. `prediction=spoof` here is a valid verdict, not a fault. |
| `result_send_started session=…` | The server began handing the result to the transport. |
| `result_send_succeeded session=…` | It reached the socket. If the client still failed, the problem is downstream of the API. |
| `result_send_failed session=… stage=after_inference client_left_after_ms=…` | **Inference was fine; delivery was not.** `client_left_after_ms` matches the `ws_client_gone` line. |
| `session_finalized session=… state=… delivered=… reason=… client_left_after_ms=…` | Exactly one per connection. `delivered=False` confirms the answer never left the server; `reason=` says why (`connection_closed`, `inference_finished_after_client_left`, `inference_finished_after_cancel`). |
| `ws_send_failed type=result session=…` | The underlying transport error, with traceback. This is what names the failure mode (`ClientDisconnected`, a closed ASGI channel, …). |
| `websocket_error session=…` | An uncaught exception, with traceback. One of the two ways the client can end up seeing close code **1011** — see “Where a 1011 close actually comes from” below. |

#### Where a 1011 close actually comes from

The application never sends `1011`; it either delivers a `result` or closes
before `accept()` with a `44xx` code. In this stack (uvicorn + the `websockets`
library) only `websockets`' own `fail_connection(1011)` can put a `1011` close
frame on the wire, and it has four call sites — three of them are effectively
unreachable from this handler, one is not:

| Source | Visible in the log? | Reachable here? |
|--------|---------------------|-----------------|
| **Keepalive ping timeout** (`reason="keepalive ping timeout"`) | **No — DEBUG only** | **Yes.** Fires when a Pong is not processed within `--ws-ping-timeout` (uvicorn default 20 s). |
| `connection handler failed` | ERROR + traceback | No: the handler only awaits `wait_closed()`, which does not raise. |
| `data transfer failed` | ERROR + traceback | Only from an unexpected exception while *reading* — i.e. while the client is still sending. |
| interrupted fragmented send | ERROR + traceback | No: server messages are single JSON frames. |

So the diagnosis starts from the two logs you can actually read:

* **No `websocket_error` and no websockets ERROR** alongside
  `result_send_failed … stage=after_inference` → the server itself failed the
  connection: the ping timeout. **Confirm it with the reason string** — the
  browser's close event carries it, and the SDK now prints it:
  `Connection closed unexpectedly (code=1011 reason=keepalive ping timeout
  wasClean=true elapsedMs=…)` in the UI/console, and the
  `ws_send_failed … ` traceback on the server names the same
  `ConnectionClosed` line. `reason="keepalive ping timeout"` ⇒ the Pong did
  not arrive within the deadline.
* **`websocket_error` present** → an exception escaped the handler; its
  traceback is the cause, and the 1011 is a symptom.

`result_send_failed` with **no** accompanying `websocket_error` and a close
reason *other than* `keepalive ping timeout` means something between the two
ends dropped the socket — go to `ws_client_gone`: `inference_running=True`
with a small `elapsed_ms` means it died while the model was still working
(proxy or client idle timeout), whereas an `elapsed_ms` at or past the
inference duration means the connection survived the whole run and the loss
happened afterwards.

Three things keep that ping timeout from firing during a long inference:

1. `--ws-ping-timeout 60` in `render.yaml` (only the deadline is relaxed; the
   20 s interval is unchanged), so one lost or delayed pong cannot end a
   verification that is seconds from finishing;
2. the receive loop stays open for the whole inference, so `websockets` keeps
   draining the inbound queue — when the application stops calling `receive`
   with 32+ messages queued, the library stops reading the socket entirely,
   **control frames included**, and the Pong can no longer be processed;
3. `progress` frames every 5 s, so the connection is never silent long enough
   for an idle timer to matter.

`tests/test_real_transport.py` pins all three against a real uvicorn server:
a ponging peer survives a 46 s inference, a peer that never answers pings is
reproduced dying with exactly `Close(1011, "keepalive ping timeout")` and the
production log sequence, and the pinned ping policy keeps that silent peer
alive.

While inference runs the server emits `progress` frames every 5 s, so a
connection that is genuinely healthy carries traffic in both directions for
the entire window rather than sitting silent for tens of seconds.

Uvicorn's own upgrade line logs the full request path, so it contains the
capability token. `?token=` is rewritten to `token=<redacted>` by a logging
filter on uvicorn's loggers; if you ever see a raw token there, the filter was
bypassed (for example by a custom `--log-config`).

### WebSocket closes immediately

| Close code | Meaning |
|------------|---------|
| `4401` | Wrong/expired `ws_token` (or missing `?token=`). |
| `4404` | Session not found — TTL expired (300 s) or already evicted. |
| `4409` | Session already consumed; create a new one. |
| `1006` (client) | Network drop or the API is behind a proxy without WebSocket support. |

**Render/proxy note:** Render's proxy supports WebSockets, but some
CDNs/proxies time out connections that carry no *data* frames for a configured
period, and they do not necessarily count protocol-level ping/pong frames. The
client stops sending after its 150th frame, and inference then runs for tens
of seconds — that window used to be completely silent. The server now emits a
`progress` frame every 5 s during inference; if you front the API with another
proxy, keep its idle timeout above 60 s as well. Note that a proxy timeout
would appear as `ws_client_gone … inference_running=True` in the server log,
not as a server-side error.

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
