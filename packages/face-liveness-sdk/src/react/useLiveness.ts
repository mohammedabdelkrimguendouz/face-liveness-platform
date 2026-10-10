import { useCallback, useEffect, useRef, useState } from "react";
import { LivenessClient } from "../client";
import { LivenessError } from "../errors";
import type {
  LivenessCallbacks,
  LivenessOptions,
  LivenessPhase,
  LivenessProgress,
  LivenessResult,
} from "../types";

export interface UseLivenessState {
  phase: LivenessPhase;
  progress: LivenessProgress | null;
  result: LivenessResult | null;
  error: LivenessError | null;
  /** True while a verification is in flight. */
  isRunning: boolean;
}

export interface UseLivenessApi extends UseLivenessState {
  /** Start a verification. No-op while one is already running. */
  start: (callbacks?: LivenessCallbacks) => Promise<LivenessResult | null>;
  /** Cancel the in-flight verification. */
  cancel: () => void;
  /** Reset to the idle state (clears result + error). */
  reset: () => void;
}

/**
 * React hook wrapping {@link LivenessClient}.
 *
 * ```tsx
 * const { phase, progress, result, error, start, cancel } = useLiveness({ apiUrl });
 * ```
 */
export function useLiveness(options: LivenessOptions): UseLivenessApi {
  const [state, setState] = useState<UseLivenessState>({
    phase: "idle",
    progress: null,
    result: null,
    error: null,
    isRunning: false,
  });
  const clientRef = useRef<LivenessClient | null>(null);
  const optsRef = useRef(options);
  optsRef.current = options;

  const cancel = useCallback(() => {
    clientRef.current?.cancel();
  }, []);

  const reset = useCallback(() => {
    setState({ phase: "idle", progress: null, result: null, error: null, isRunning: false });
  }, []);

  const start = useCallback(async (extra?: LivenessCallbacks): Promise<LivenessResult | null> => {
    if (clientRef.current) return null; // already running
    setState((s) => ({ ...s, isRunning: true, error: null, result: null }));

    const client = new LivenessClient(optsRef.current, {
      onProgress: (p) => {
        setState((s) => ({ ...s, progress: p }));
        extra?.onProgress?.(p);
      },
      onPhaseChange: (phase) => {
        setState((s) => ({ ...s, phase }));
        extra?.onPhaseChange?.(phase);
      },
      onStream: (stream) => extra?.onStream?.(stream),
      onSuccess: (r) => {
        setState((s) => ({ ...s, result: r, phase: "completed", isRunning: false }));
        extra?.onSuccess?.(r);
      },
      onError: (e) => {
        const le = e instanceof LivenessError ? e : new LivenessError("server-error", e.message);
        setState((s) => ({ ...s, error: le, phase: "failed", isRunning: false }));
        extra?.onError?.(e);
      },
      onCancel: () => {
        setState((s) => ({ ...s, phase: "cancelled", isRunning: false }));
        extra?.onCancel?.();
      },
    });
    clientRef.current = client;

    try {
      const r = await client.start();
      return r;
    } catch {
      return null; // surfaced via state.error
    } finally {
      clientRef.current = null;
      setState((s) => (s.isRunning ? { ...s, isRunning: false } : s));
    }
  }, []);

  // Cancel + release on unmount.
  useEffect(() => {
    return () => {
      clientRef.current?.cancel();
      clientRef.current = null;
    };
  }, []);

  return { ...state, start, cancel, reset };
}
