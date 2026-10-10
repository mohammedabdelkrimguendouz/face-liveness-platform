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

## Model weights are not in git

The checkpoints total ~137 MB and are intentionally excluded (see `.gitignore`):

- `EfficientNet_B2.pt` (31 MB) — feature extractor
- `ABC_BiLSTM_Undercomplete_AE.pt` (75 MB) — liveness autoencoder

`scripts/fetch_models.py` runs during the Render build and resolves them in
this order:

1. **Already on disk** at `LIVENESS_MODELS_DIR` → verify, continue.
2. **Downloaded** from `LIVENESS_MODELS_URL` if set (must be a `zip` or `tar`
   containing both files at any depth).
3. Otherwise the build **fails with instructions** rather than deploying an API
   that can never become ready.

### Option A — private bucket (recommended)

Upload a zip containing both checkpoints somewhere private:

```bash
cd "My Code For Testing Local Vedio/test_models/saved_models"
zip ~/face-liveness-models.zip EfficientNet_B2.pt ABC_BiLSTM_Undercomplete_AE.pt
# Upload to S3 / R2 / GCS / a private GitHub release asset...
```

Then in the Render dashboard set:

```
LIVENESS_MODELS_URL = https://your-private-bucket/face-liveness-models.zip
```

> If the bucket is private, use a signed URL (it will be embedded in the Render
> env var — treat it as a secret). Note the URL is fetched at **build time**;
> signed URLs that expire before the next build will break re-deploys.

### Option B — Render disk

Create a persistent disk, mount it, and copy the checkpoints to it manually
(Render shell → `cp`), then set `LIVENESS_MODELS_DIR` to the mount path.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LIVENESS_MODELS_DIR` | `./models` | Where the checkpoints live. |
| `LIVENESS_MODELS_URL` | *(unset)* | Zip/tar URL to fetch them from. |
| `LIVENESS_DEVICE` | `cpu` | `cpu` (Render has no GPU). |
| `LIVENESS_BATCH_SIZE` | `8` | Feature-extraction batch. Lower = less RAM. |
| `LIVENESS_NUM_FRAMES` | `150` | Frames required per session. **Don't change** — it must match the checkpoint. |
| `LIVENESS_SESSION_TTL` | `300` | Seconds before an idle session is evicted. |
| `LIVENESS_MAX_SESSIONS` | `50` | Cap on concurrent sessions (429 beyond this). |
| `LIVENESS_ALLOWED_ORIGINS` | `http://localhost:5173` | CORS allow-list, comma-separated. **Set to your Vercel URL.** |
| `LOG_LEVEL` | `info` | Uvicorn/log level. |

## Deploy steps

1. Push the repo to GitHub/GitLab.
2. Render → **New → Blueprint** → select the repo → point it at
   `face-liveness-platform/render.yaml`
   *(or New → Web Service and set Root Directory to
   `Face Anti Spoofing Project/face-liveness-platform`)*.
3. Set `LIVENESS_MODELS_URL` (or `LIVENESS_MODELS_DIR`) and
   `LIVENESS_ALLOWED_ORIGINS`.
4. Deploy. Health check is `/health`; readiness is `/ready`.

### Verify

```bash
curl https://your-api.onrender.com/health   # {"status":"ok"}
curl https://your-api.onrender.com/ready    # {"ready":true,...}
```

`/ready` returns **503** with a message if the model failed to load — check the
build log for `fetch_models.py` output first.

## Operational notes

- **Single worker only.** Sessions are in-memory; `--workers > 1` would break
  session affinity. Scale vertically.
- **Cold starts.** Loading both models takes ~3–5 s; the process handles that
  via `/ready` (503 until loaded).
- **No GPU on Render.** Inference is CPU-only: ~6.5 s feature extraction +
  ~1.1 s autoencoder for 150 frames on a typical vCPU. Budget ~10 s of
  server-side time per verification on top of upload time.
- **Concurrency.** With `LIVENESS_MAX_CONCURRENT_INFERENCES=1` (default),
  concurrent verifications queue behind a semaphore. Raise the plan before
  raising this.
