import * as React from "react";
import { useLiveness } from "./useLiveness";
import type { UseLivenessApi } from "./useLiveness";
import type { LivenessOptions } from "../types";

export interface FaceLivenessCheckProps extends LivenessOptions {
  /** Called once with the authoritative backend result. */
  onSuccess?: (result: import("../types").LivenessResult) => void;
  /** Called on any terminal error. */
  onError?: (error: Error) => void;
  /**
   * Render prop receiving the full hook API. Use it to render your own UI.
   * If omitted, nothing is rendered (headless by default).
   */
  children?: (api: UseLivenessApi) => React.ReactNode;
}

/**
 * Headless React component for face liveness verification.
 *
 * It exposes the entire {@link useLiveness} state machine through a render
 * prop so the host app controls the visuals. Nothing is rendered unless you
 * pass `children`.
 *
 * ```tsx
 * <FaceLivenessCheck apiUrl={API_URL} onSuccess={(r) => console.log(r.prediction)}>
 *   {({ phase, progress, start, cancel }) => (
 *     <button onClick={() => start()}>Verify</button>
 *   )}
 * </FaceLivenessCheck>
 * ```
 */
export function FaceLivenessCheck(props: FaceLivenessCheckProps): React.ReactElement | null {
  const { onSuccess, onError, children, ...options } = props;
  const api = useLiveness(options);

  // Wire external callbacks through a ref so the identity stays stable.
  const cbRef = React.useRef({ onSuccess, onError });
  cbRef.current = { onSuccess, onError };

  React.useEffect(() => {
    if (api.result) cbRef.current.onSuccess?.(api.result);
    // Only fire when a fresh result arrives.
  }, [api.result]);

  React.useEffect(() => {
    if (api.error) cbRef.current.onError?.(api.error);
  }, [api.error]);

  return children ? <>{children(api)}</> : null;
}
