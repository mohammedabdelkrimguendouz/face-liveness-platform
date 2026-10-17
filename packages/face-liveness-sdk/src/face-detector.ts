import { LivenessError } from "./errors";
import type { FaceDetectionOptions } from "./types";

/**
 * On-device face gate backed by MediaPipe Face Detector
 * (`@mediapipe/tasks-vision`, BlazeFace short-range).
 *
 * The WASM runtime and the model weights are fetched at runtime from pinned
 * public CDNs by default (both serve `Access-Control-Allow-Origin: *`). Hosts
 * that need offline or first-party delivery can point `wasmPath` / `modelUrl`
 * at their own copies via {@link FaceDetectionOptions}. Detection itself runs
 * entirely in the browser — no frame ever leaves the device un-gated, and no
 * image data is sent to any third party.
 */

/** Pinned tasks-vision WASM assets (must match the installed package version). */
export const DEFAULT_WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";

/** Google-hosted BlazeFace short-range model (~225 KB, float16). */
export const DEFAULT_FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

/**
 * Structural view of a loaded MediaPipe FaceDetector: the SDK only needs
 * `detect()` on an image source (we pass the encoder canvas) and the list of
 * detections it returns. Keeping this local means the rest of the client never
 * imports MediaPipe types, and tests can inject a plain stub.
 */
export interface FaceDetectorLike {
  detect(image: CanvasImageSource): {
    detections: Array<{ categories?: Array<{ score?: number }> }>;
  };
}

/** Module-level cache: WASM + model load once per page, not per verification. */
let cached: Promise<FaceDetectorLike> | null = null;
let cachedKey = "";

/**
 * Load (or reuse) the face detector. Rejects with
 * {@link LivenessError} code `face-detect-failed` when the runtime or model
 * cannot be loaded — a failed gate must never silently pass frames through.
 */
export function loadFaceDetector(
  opts: FaceDetectionOptions = {},
): Promise<FaceDetectorLike> {
  const wasmPath = opts.wasmPath ?? DEFAULT_WASM_PATH;
  const modelUrl = opts.modelUrl ?? DEFAULT_FACE_MODEL_URL;
  const minScore = opts.minScore ?? 0.5;
  const key = `${wasmPath}|${modelUrl}|${minScore}`;
  if (!cached || cachedKey !== key) {
    cachedKey = key;
    cached = createDetector(wasmPath, modelUrl, minScore).catch((err) => {
      // Do not cache failures: a transient network error should not poison
      // every later verification on this page.
      cached = null;
      throw err;
    });
  }
  return cached;
}

async function createDetector(
  wasmPath: string,
  modelUrl: string,
  minScore: number,
): Promise<FaceDetectorLike> {
  let mod: typeof import("@mediapipe/tasks-vision");
  try {
    // Dynamic import: keeps tasks-vision out of the main bundle until a
    // verification actually starts.
    mod = await import("@mediapipe/tasks-vision");
  } catch (err) {
    throw new LivenessError(
      "face-detect-failed",
      "The face detection library failed to load. Check the connection and try again.",
      (err as Error)?.message,
    );
  }

  let vision: Awaited<ReturnType<typeof mod.FilesetResolver.forVisionTasks>>;
  try {
    vision = await mod.FilesetResolver.forVisionTasks(wasmPath);
  } catch (err) {
    throw new LivenessError(
      "face-detect-failed",
      "The face detection engine (WASM) failed to load. Check the connection and try again.",
      (err as Error)?.message,
    );
  }

  const make = (delegate: "GPU" | "CPU") =>
    mod.FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: modelUrl, delegate },
      runningMode: "IMAGE",
      minDetectionConfidence: minScore,
    });

  // GPU (WebGL) where available; fall back to CPU on devices without it.
  try {
    return (await make("GPU")) as unknown as FaceDetectorLike;
  } catch {
    try {
      return (await make("CPU")) as unknown as FaceDetectorLike;
    } catch (err) {
      throw new LivenessError(
        "face-detect-failed",
        "The face detector failed to initialize on this device.",
        (err as Error)?.message,
      );
    }
  }
}
