# Publishing `@mohammedabdelkrimguendouz/face-liveness-sdk`

The package is published (`0.1.0`–`0.1.2` on npm). These are the steps for
each new release.

## Pre-flight checks

```bash
cd packages/face-liveness-sdk

# 1. Confirm the registry entry (lists the already-published versions)
npm view @mohammedabdelkrimguendouz/face-liveness-sdk versions

# 2. Confirm the full gate passes (typecheck -> test -> build)
npm run prepublishOnly
```

`prepublishOnly` runs `typecheck && test && build`, so a green run means the
tarball is publishable.

> The scope `@mohammedabdelkrimguendouz` matches your npm username, so npm
> associates it with your account automatically — **no npm organization is
> needed** and you cannot publish to a scope you don't own. If you ever rename
> the package to a different scope, that scope must belong to you or to an org
> you belong to.

## Publish

```bash
# Login (opens a browser)
npm login

# Dry run - inspect exactly what ships
npm pack --dry-run

# Publish (public scope; --access public is required for new scoped packages)
npm publish --access public
```

Expected tarball (~56 kB): `dist/**` (ESM + CJS + `.d.ts`), `package.json`.
Source, tests, and configs are excluded by the `files` allow-list.

> `--access public` is mandatory for the first publish of a scoped package;
> scoped packages default to restricted (paid) visibility otherwise.

## Verify the published package

```bash
npm view @mohammedabdelkrimguendouz/face-liveness-sdk dist.tarball
mkdir /tmp/consumer && cd /tmp/consumer && npm init -y
npm install @mohammedabdelkrimguendouz/face-liveness-sdk

node -e "const {LivenessClient}=require('@mohammedabdelkrimguendouz/face-liveness-sdk'); console.log(typeof LivenessClient)"  # function
```

This local-install path (tarball → scratch project → ESM/CJS/TS checks) was
already validated during implementation; repeat it against the *published*
artifact before announcing.

## Versioning & re-publishing

```bash
npm version patch   # 0.1.0 -> 0.1.1
npm publish --access public
```

Use `npm version minor` for new features, `major` once the protocol/API is
stable. Publishes are immutable — bump and republish rather than overwriting.

## Consumer usage

```ts
// Headless
import { LivenessClient } from "@mohammedabdelkrimguendouz/face-liveness-sdk";

// React
import { useLiveness, FaceLivenessCheck } from "@mohammedabdelkrimguendouz/face-liveness-sdk/react";
```

React is a **peer dependency** (`>=17`, optional) so it isn't bundled.

### Options

```ts
const client = new LivenessClient({
  apiUrl: "https://api.example.com",
  targetFps: 10,        // default 10 — 150 frames ≈ 15 s of capture
  jpegQuality: 85,      // default
  timeoutMs: 120_000,   // default

  // On-device face gate (MediaPipe Face Detector) — enabled by default.
  // Frames without a face are never uploaded; a sustained absence aborts
  // with a `no-face` error. Pass `false` to disable.
  faceDetection: {
    minScore: 0.5,             // detection confidence [0..1]
    maxConsecutiveMisses: 10,  // ≈1 s at 10 fps; 0 = fail on the first miss
    // For offline / first-party delivery, point these at your own copies:
    // wasmPath: "https://cdn.example.com/tasks-vision/wasm",
    // modelUrl: "https://cdn.example.com/models/blaze_face_short_range.tflite",
  },
});
```

The gate downloads the tasks-vision WASM runtime (jsDelivr, pinned to the
installed package version) and the ~225 KB BlazeFace model (Google-hosted)
once per page; detection itself runs entirely in the browser. Error codes to
branch on: `no-face` (gate aborted the run) and `face-detect-failed` (WASM or
model could not be loaded — verification fails closed).

## Note on the demo app

`apps/demo-web` depends on the SDK via a local `file:` path, not from the
registry, so it keeps building before and after you publish. Publishing does
not affect the demo. If you later want the demo to consume the published
package instead, change the dependency in `apps/demo-web/package.json` to
`"@mohammedabdelkrimguendouz/face-liveness-sdk": "^0.1.0"`.
