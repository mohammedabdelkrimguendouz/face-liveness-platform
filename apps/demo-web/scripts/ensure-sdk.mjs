#!/usr/bin/env node
/**
 * Ensure the sibling SDK package is installed and built before the demo runs.
 *
 * Why this exists:
 *   apps/demo-web depends on the SDK through a local `file:` path, and npm
 *   symlinks that dependency rather than copying it. The SDK's build output
 *   (`packages/face-liveness-sdk/dist/`) is gitignored, so a fresh clone —
 *   a Vercel checkout, CI, or a new teammate — has the sources but no `dist/`.
 *   TypeScript then fails with TS2307 ("Cannot find module
 *   '@mohammedabdelkrimguendouz/face-liveness-sdk/react'") and Vite cannot
 *   resolve the import either.
 *
 *   An npm `prepare` script would not help: npm does not install a linked
 *   package's devDependencies, so `tsup` would be missing, and install-script
 *   policies may block `prepare` entirely.
 *
 * What it does (idempotent, skips work when already satisfied):
 *   1. `npm install` in the SDK, if its node_modules is absent.
 *   2. `npm run build` in the SDK, if dist is absent or older than any source file.
 *
 * Both commands run with cwd set to the SDK directory, so this works from any
 * working directory and on any platform (the SDK path never goes through the
 * shell command line, which matters because this repo lives under a path with
 * spaces in it).
 */

import { execSync } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const demoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sdkRoot = resolve(demoRoot, "..", "..", "packages", "face-liveness-sdk");
const sdkSrc = join(sdkRoot, "src");
const sdkEntry = join(sdkRoot, "dist", "index.js");
const sdkNodeModules = join(sdkRoot, "node_modules");

if (!existsSync(join(sdkRoot, "package.json"))) {
  console.error(
    `[ensure-sdk] SDK package not found at:\n  ${sdkRoot}\n` +
      `The demo expects the SDK at packages/face-liveness-sdk relative to the repository root.`,
  );
  process.exit(1);
}

/** Newest mtime under a directory tree, or 0 if it does not exist. */
function newestMtime(dir) {
  let newest = 0;
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return 0;
  }
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      newest = Math.max(newest, newestMtime(full));
    } else {
      try {
        newest = Math.max(newest, statSync(full).mtimeMs);
      } catch {
        // Ignore entries that disappear mid-scan.
      }
    }
  }
  return newest;
}

function run(command, label) {
  console.log(`[ensure-sdk] ${label}: ${command}`);

  // npm exports its own configuration to child processes as `npm_config_*`
  // env vars. The demo's package.json declares an `allowScripts` field, and
  // npm forwards that to us as `npm_config_allow_scripts`. Passing it on to a
  // nested project-scoped `npm install` makes npm abort with
  // "--allow-scripts is not allowed in project-scoped installs", so drop it
  // and let the SDK's own configuration decide.
  const env = { ...process.env };
  for (const key of Object.keys(env)) {
    if (key.toLowerCase() === "npm_config_allow_scripts") {
      delete env[key];
    }
  }

  execSync(command, { cwd: sdkRoot, stdio: "inherit", env });
}

// 1. Install the SDK's own dependencies (tsup, typescript, vitest, ...).
if (!existsSync(sdkNodeModules)) {
  run("npm install --no-audit --no-fund", "installing SDK dependencies");
} else {
  console.log("[ensure-sdk] SDK dependencies already installed");
}

// 2. Build the SDK if dist is missing or older than anything under src/.
const distMtime = existsSync(sdkEntry) ? statSync(sdkEntry).mtimeMs : 0;
const srcMtime = newestMtime(sdkSrc);

if (distMtime === 0) {
  run("npm run build", "building SDK (no dist found)");
} else if (srcMtime > distMtime) {
  run("npm run build", "building SDK (src changed since last build)");
} else {
  console.log("[ensure-sdk] SDK dist is up to date");
}
