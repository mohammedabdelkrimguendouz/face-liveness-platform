export { FaceLivenessCheck } from "./FaceLivenessCheck";
export type { FaceLivenessCheckProps } from "./FaceLivenessCheck";
export { useLiveness } from "./useLiveness";
export type { UseLivenessApi, UseLivenessState } from "./useLiveness";

// Re-export the core types so consumers importing from
// "@thirduni/face-liveness-sdk/react" don't need a second import.
export type {
  LivenessPhase,
  LivenessPrediction,
  LivenessResult,
  LivenessErrorCode,
  LivenessProgress,
  LivenessOptions,
  LivenessCallbacks,
} from "../types";
