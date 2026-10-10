# Publishing `@thirduni/face-liveness-sdk`

The package has **not** been published. These are the steps when you're ready.

## Pre-flight checks

```bash
cd packages/face-liveness-sdk

# 1. Verify the scoped name is free (404 = available)
npm view @thirduni/face-liveness-sdk

# 2. Confirm the full gate passes (typecheck -> test -> build)
npm run prepublishOnly
```

`prepublishOnly` runs `typecheck && test && build`, so a green run means the
tarball is publishable.

> If `@thirduni` isn't an existing npm org you control, either create it
> (`npm org create thirduni`) or change `name` in `package.json` to something
> you own (e.g. `@<your-scope>/face-liveness-sdk`). Publishing to a scope you
> don't own will fail.

## Publish

```bash
# Login (opens a browser)
npm login

# Dry run — inspect exactly what ships
npm pack --dry-run

# Publish (public scope; --access public is required for new scoped packages)
npm publish --access public
```

Expected tarball (~56 kB): `dist/**` (ESM + CJS + `.d.ts`), `package.json`.
Source, tests, and configs are excluded by the `files` allow-list.

## Verify the published package

```bash
npm view @thirduni/face-liveness-sdk dist.tarball
mkdir /tmp/consumer && cd /tmp/consumer && npm init -y
npm install @thirduni/face-liveness-sdk

node -e "const {LivenessClient}=require('@thirduni/face-liveness-sdk'); console.log(typeof LivenessClient)"  # function
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
import { LivenessClient } from "@thirduni/face-liveness-sdk";

// React
import { useLiveness, FaceLivenessCheck } from "@thirduni/face-liveness-sdk/react";
```

React is a **peer dependency** (`>=17`, optional) so it isn't bundled.
