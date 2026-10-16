# Deploying the Face Liveness API on Render

This service holds a PyTorch model in memory, so **instance size matters**.

## Instance size (read this first)

| Plan | RAM | Verdict |
|------|-----|---------|
| Free / Starter | 512 MB | **Will be OOM-killed.** Measured peak RSS ≈ 960 MB. |
| **Standard** | **2 GB** | **Required minimum.** |
| Pro | 4 GB | Recommended for headroom / concurrency. |

The measured ~960 MB peak is CPU inference with `batch_size=16`. Even with the
default `LIVENESS_BATCH_SIZE=8` the process stays well above 512 MB (torch
runtime baseline + 18.7 M-param autoencoder + 31 MB EfficientNet + frame
buffers). Do not attempt the free tier.

## Model weights ship with the repository

There is **no download step and no external storage**. The two checkpoints are
committed to this repo:

```
services/face-liveness-api/models/
├── EfficientNet_B2.pt            (31 MB)
└── ABC_BiLSTM_Undercomplete_AE.pt (75 MB)
```

`render.yaml` runs `python scripts/fetch_models.py` during the build. That
script **verifies** the two files exist — it does not download anything — and
fails the build with the exact expected path if they are missing, so a broken
deploy is reported as failed rather than starting an API that can never become
ready.

> Both files are under GitHub's 100 MB per-file limit (31 MB and 75 MB).
> `.gitignore` un-ignores exactly these two paths; every other `*.pt` / `*.pth`
> stays out of git. No Git LFS is used.

### Where the application looks

`app/paths.py` is the **single source of truth**. Both the build script and the
running API call `resolve_models_dir()`, so they cannot disagree:

1. `$LIVENESS_MODELS_DIR`, if set and non-empty. A **relative** value is
   anchored to `services/face-liveness-api/` (the package location), *never* to
   the process working directory.
2. Otherwise `services/face-liveness-api/models/`.

`render.yaml` sets `LIVENESS_MODELS_DIR=models`, which resolves to the same
absolute path under both the build (`cwd` = `rootDir`) and the start command.
You normally never need to change it.

## Environment variables

Set in `render.yaml` (no dashboard work needed):

| Variable | Value | Purpose |
|----------|-------|---------|
| `LIVENESS_DEVICE` | `cpu` | Render has no GPU. |
| `LIVENESS_BATCH_SIZE` | `8` | Feature-extraction batch. Lower = less RAM. |
| `LIVENESS_MODELS_DIR` | `models` | Checkpoint dir, relative to `services/face-liveness-api/`. |
| `LIVENESS_ALLOWED_ORIGINS` | `https://face-liveness-platform.vercel.app,https://face-liveness-platform-5iupvthd2.vercel.app` | Exact browser origins allowed to call the API (comma-separated, no trailing slash). |

**Keep in sync manually if you add origins:**

| Variable | When | Notes |
|----------|------|-------|
| `LIVENESS_ALLOWED_ORIGINS` | Whenever the frontend (or a preview domain) changes | Comma-separated exact origins (`scheme://host`, no path). A blueprint deploy syncs this file's value **over** any manual dashboard edit, so change `render.yaml` too. If it does not list the origin the browser sends, `GET /health` omits `Access-Control-Allow-Origin` and the preflight answers `400 Disallowed CORS origin` — the UI then shows "Service offline" / "Could not reach the liveness service" even though the API is healthy. |

**Optional** (defaults apply if unset): `LIVENESS_NUM_FRAMES` (150 — don't
change, it must match the checkpoint), `LIVENESS_SESSION_TTL` (300),
`LIVENESS_MAX_SESSIONS` (50), `LIVENESS_MAX_CONCURRENT_INFERENCES` (1),
`LIVENESS_MAX_FRAME_BYTES` (2 MiB), `LIVENESS_MAX_FRAMES` (400), `HOST`,
`PORT` (provided by Render).

> There is deliberately **no `LIVENESS_MODELS_URL`**. Weights are in the repo.

## Deploy steps

1. Push the repo to GitHub (the weights are committed — see the size note above).
2. Render → **New → Blueprint** → select the repo. `render.yaml` is at the repo
   root, so it is detected automatically; **Root Directory needs no setting**
   (`render.yaml` sets `rootDir: services/face-liveness-api` itself).
3. Choose the **Standard (2 GB)** plan.
4. Deploy, then replace `LIVENESS_ALLOWED_ORIGINS` with your Vercel origin and
   redeploy (or set it before the first deploy).

### Health check

`render.yaml` sets `healthCheckPath: /ready`.

| Endpoint | Behaviour |
|----------|-----------|
| `/health` | Process liveness only. Always `200 {"status":"ok"}`, even if the model failed to load. |
| `/ready` | `200` once both checkpoints are loaded; **`503` with the load error otherwise.** |

Using `/ready` means a deploy whose model never loads is reported as **failed**
instead of going green while unable to serve. This is safe:

- **Startup:** the model takes ~8–10 s to load. Render retries health checks and
  only cancels a new deploy after **15 minutes** of failures, so a normal
  startup passes comfortably. Each individual check must return 2xx/3xx within
  **5 seconds**, which `/ready` does once the model is loaded.
- **Trade-off:** during the first ~10 s of a cold start `/ready` returns 503,
  so traffic is not routed until the model is actually loaded — which is the
  correct behaviour for this service.

### Verify

```bash
curl https://your-api.onrender.com/health   # {"status":"ok"}
curl https://your-api.onrender.com/ready    # {"ready":true,...}
```

If `/ready` is 503, read the build log for `fetch_models.py` output first —
it prints the expected directory. Then check the runtime log for
`Models dir:`.

## Operational notes

- **Single worker only.** Sessions are in-memory; `--workers > 1` would break
  session affinity. Scale vertically.
- **Cold starts.** Loading both models takes ~8–10 s; `/ready` holds traffic
  until it finishes.
- **No GPU on Render.** Inference is CPU-only: ~6.5 s feature extraction +
  ~1.1 s autoencoder for 150 frames on a typical vCPU. Budget ~10 s of
  server-side time per verification on top of upload time.
- **Concurrency.** With `LIVENESS_MAX_CONCURRENT_INFERENCES=1` (default),
  concurrent verifications queue behind a semaphore. Raise the plan before
  raising this.
