# WebSocket Protocol — Face Liveness API

Version 1. A session is created over HTTP, then exactly one WebSocket carries
the frame sequence and the result.

## 1. Create a session

```http
POST /sessions
Accept: application/json
```

**201 response**

```json
{
  "session_id": "hR8…",
  "ws_url": "/ws/sessions/hR8…",
  "ws_token": "kQ2…",
  "num_frames": 150,
  "max_frame_bytes": 2097152,
  "expires_in_seconds": 300
}
```

Errors: `429` (server busy / session cap reached), `503` on `/ready` if the
model isn't loaded.

> `ws_url` is a **path**. The client must combine it with the API origin,
> upgrading the scheme: `https://api.example.com` + `/ws/sessions/…`
> → `wss://api.example.com/ws/sessions/…`.

## 2. Connect

```
GET /ws/sessions/{session_id}?token={ws_token}
```

- Wrong token → close `4401`
- Unknown/expired session → close `4404`
- Session already consumed → close `4409`

On accept the server sends exactly one greeting:

```json
{ "type": "ready", "session_id": "…", "required_frames": 150, "max_frame_bytes": 2097152 }
```

## 3. Stream frames (client → server, binary)

Each frame is a **binary** message:

```
┌────────────────────┬──────────────────────┐
│ 4-byte seq (u32 BE)│ JPEG payload         │
└────────────────────┴──────────────────────┘
```

- Sequence indices are `0 … num_frames-1`.
- Decoding is **order-independent**: frames are stored by index and sorted
  before inference, so transport reordering cannot corrupt a sequence.
- Duplicate index → rejected with `duplicate_frame`, state unchanged.
- Index ≥ `num_frames` → `bad_index`.
- Payload > `max_frame_bytes` → `frame_too_large`, session failed.
- Any decoded frame is resized to 288×288 and sampled uniformly to 150 frames
  (matching the training pipeline exactly).

The client must tolerate `close` at any time (the session TTL is 300 s).

## 4. Control messages (JSON text, both directions)

### Client → server

| Message | Meaning |
|---------|---------|
| `{"type":"complete"}` | "I'm done" — server validates the sequence. If frames are missing → `incomplete_sequence` error. |
| `{"type":"cancel"}` | Abandon the session → server replies `{"type":"cancelled"}`, releases buffers. |

Sending all 150 frames is sufficient on its own — the server auto-finalizes
when `num_frames` frames have arrived. `complete` is an early-validation
shortcut for detecting gaps sooner.

### Server → client

| `type` | Payload | Notes |
|--------|---------|-------|
| `ready` | `session_id`, `required_frames`, `max_frame_bytes` | Sent once on accept. |
| `ack` | `received`, `required` | Throttled: every 10 frames, and at completion. |
| `processing` | `received` | Inference started. |
| `result` | see below | Terminal success. |
| `error` | `error`, `message`, optional `missing` | Terminal or recoverable (see below). |
| `cancelled` | — | Acknowledges `{"type":"cancel"}`. |

#### `result`

```json
{
  "type": "result",
  "session_id": "…",
  "prediction": "live",
  "anomaly_score": 0.0382,
  "reconstruction_error": 0.039,
  "threshold": 0.5,
  "frames_processed": 150,
  "inference_ms": 1106,
  "score_semantics": "anomaly_score = 1 - exp(-MSE); not a calibrated probability"
}
```

**Semantics (authoritative):**

- `anomaly_score = 1 − exp(−MSE)` where MSE is over all 150 frames × 1408
  features.
- `prediction = "spoof"` iff `anomaly_score > threshold`, else `"live"`.
- The score is an **anomaly score, not a probability** — it is not calibrated
  and must not be displayed as a confidence percentage.

#### `error` codes

| `error` | Recoverable? | Meaning |
|---------|--------------|---------|
| `bad_message` | yes | Malformed JSON / unknown control type. |
| `malformed_frame` | yes | Binary message shorter than 4 bytes, or empty payload. |
| `duplicate_frame` | yes | Index already received. |
| `bad_index` | yes | Index ≥ `num_frames`. |
| `frame_too_large` | **no** | Payload exceeded `max_frame_bytes`; session failed. |
| `too_many_frames` | **no** | Session frame cap exceeded. |
| `incomplete_sequence` | **no** | `complete` sent with missing frames (`missing` = count). |
| `bad_frames` | **no** | Frames could not be decoded/processed. |
| `inference_failed` | **no** | Model threw during inference. |
| `internal_error` | **no** | Unexpected server fault. |

Recoverable errors do **not** terminate the session — the client may resend.
Terminal errors leave the session in `failed`; the client should create a new
session.

## 5. Session lifecycle

```
created ──accept──▶ streaming ──auto-finalize / complete──▶ inferring ──▶ completed
                        │                                        │
                        ├──── cancel ────────────────────────▶  cancelled
                        └──── terminal error ────────────────▶  failed
                        └──── TTL (300 s) ───────────────────▶  expired
```

- **Isolation:** a session is reachable only with its `session_id` +
  `ws_token` pair (both 256-bit `token_urlsafe`). Buffers are keyed per
  session; one session can never observe another's frames.
- **Bounded:** frame buffers are released immediately after `result`,
  `cancel`, terminal error, or TTL expiry.
- **Single inference:** a second concurrent inference on the same session
  is rejected (`already_running`).

## Ordering, integrity, backpressure

| Guarantee | Mechanism |
|-----------|-----------|
| Ordering | Frames keyed by explicit 4-byte sequence index; sorted before inference. |
| Integrity | Duplicate detection; size limits; `complete`-time gap check with an explicit missing count. |
| Backpressure | Client waits for `WebSocket.bufferedAmount` to drain below 4 × `max_frame_bytes` before capturing the next frame. Server caps `LIVENESS_MAX_SESSIONS` and queues inference behind a bounded semaphore. |
| Isolation | Per-session state + capability token; no shared buffers. |

## Reference client

The TypeScript SDK in `packages/face-liveness-sdk` implements this protocol
(`src/client.ts`). Frame packing is a big-endian `DataView.setUint32(0, i, false)`.
