# Deploying the Demo Web App on Vercel

The demo is a static Vite + React SPA. It contains **no model code** — the
decision is always made by the API.

## Configure

The app reads one environment variable at **build time**:

```
VITE_LIVENESS_API_URL = https://your-api.onrender.com
```

(no trailing slash required; `http://localhost:8000` for local dev — see
`apps/demo-web/.env.example`)

Because Vite inlines `import.meta.env.*`, this must be set as a **build-time**
env var, not a runtime one. Changing it requires a re-deploy.

## Deploy steps

1. Vercel → **Add New → Project** → import the repo.
2. Set:
   - **Root Directory:** `Face Anti Spoofing Project/face-liveness-platform/apps/demo-web`
   - **Framework Preset:** Vite (auto-detected from `vercel.json`)
   - **Environment Variable:** `VITE_LIVENESS_API_URL`
3. Deploy.

`apps/demo-web/vercel.json` supplies `buildCommand`, `outputDirectory`, and an
SPA rewrite so deep links resolve to `index.html`.

## After deploying

Update the **API's** CORS allow-list to include your new Vercel origin, or the
browser will block `POST /sessions`:

```
LIVENESS_ALLOWED_ORIGINS = https://your-demo-app.vercel.app
```

The header badge in the demo polls `/health`, so it will show **API connected**
only once CORS + reachability are both correct.

## Local development

```bash
# terminal 1 — API
cd services/face-liveness-api
.venv/Scripts/python -m uvicorn app.main:app --port 8000

# terminal 2 — demo
cd apps/demo-web
npm run dev        # http://localhost:5173
```

> Camera access requires a secure context. `http://localhost` qualifies;
> any other origin must be HTTPS.

## Camera caveat

Frame capture needs a real webcam and user permission. In headless/CI
environments the UI renders but verification cannot complete — that is
expected, not a bug. The full WebSocket + model path is covered by the
backend test suite (`tests/test_inference.py`, marked `real`).
