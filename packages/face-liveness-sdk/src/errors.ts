import type { LivenessErrorCode } from "./types";

/**
 * Error type thrown/returned by the SDK. Carries a stable machine-readable
 * {@link LivenessErrorCode} so host apps can branch on failure kind.
 */
export class LivenessError extends Error {
  readonly code: LivenessErrorCode;
  /** Optional backend-provided detail. */
  readonly detail?: string;

  constructor(code: LivenessErrorCode, message: string, detail?: string) {
    super(message);
    this.name = "LivenessError";
    this.code = code;
    this.detail = detail;
    // Maintain prototype chain for `instanceof` under older TS targets.
    Object.setPrototypeOf(this, LivenessError.prototype);
  }
}

/** Type guard for {@link LivenessError}. */
export function isLivenessError(err: unknown): err is LivenessError {
  return err instanceof LivenessError;
}
