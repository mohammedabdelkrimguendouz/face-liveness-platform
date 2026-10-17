import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "react/index": "src/react/index.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  // ESM entries share one copy of each module (one LivenessError class
  // identity), so `instanceof`/isLivenessError work no matter which entry the
  // consumer imported the error from. CJS stays per-entry (tsup limitation).
  splitting: true,
  treeshake: true,
  external: ["react"],
  target: "es2019",
});
