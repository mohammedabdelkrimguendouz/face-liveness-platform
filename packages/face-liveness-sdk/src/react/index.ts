export { FaceLivenessCheck } from "./FaceLivenessCheck";
export type { FaceLivenessCheckProps } from "./FaceLivenessCheck";
export { useLiveness } from "./useLiveness";
export type { UseLivenessApi, UseLivenessState } from "./useLiveness";

// Re-export the core types so consumers importing from
// "@mohammedabdelkrimguendouz/face-liveness-sdk/react" don't need a second import.
export type {
  LivenessPhase,
  LivenessPrediction,
  LivenessResult,
  LivenessErrorCode,
  LivenessProgress,
  LivenessOptions,
  LivenessCallbacks,
  FaceDetectionOptions,
} from "../types";

// Runtime helpers come from the same modules the hook throws from, so
// `error instanceof LivenessError` / `isLivenessError(error)` hold when the
// consumer imports everything from the /react entry.
export { LivenessError, isLivenessError } from "../errors";
