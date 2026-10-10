import { LivenessError } from "./errors";

/**
 * Browser capability + secure-context checks.
 * Throws {@link LivenessError} with a specific code when unsupported.
 */
export function assertBrowserSupported(): void {
  if (typeof window === "undefined" || typeof document === "undefined") {
    throw new LivenessError("unsupported", "This SDK only runs in a browser environment.");
  }
  const secure =
    typeof window.isSecureContext === "boolean" ? window.isSecureContext : true;
  if (!secure) {
    throw new LivenessError(
      "insecure-context",
      "Camera access requires a secure context (HTTPS, or http://localhost for development)."
    );
  }
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    throw new LivenessError(
      "unsupported",
      "getUserMedia is not available in this browser. Use a modern browser over HTTPS."
    );
  }
  if (typeof HTMLCanvasElement === "undefined" || !HTMLCanvasElement.prototype.getContext) {
    throw new LivenessError("unsupported", "Canvas is not available in this browser.");
  }
}

export interface CameraHandle {
  stream: MediaStream;
  video: HTMLVideoElement;
  stop: () => void;
}

/**
 * Request camera permission and return a live, playing video element.
 *
 * @param width target capture width
 * @param deviceId optional exact device to use
 */
export async function openCamera(width = 720, deviceId?: string): Promise<CameraHandle> {
  assertBrowserSupported();

  const aspect = 3 / 4; // portrait-ish crop keeps the face large
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      width: { ideal: width },
      height: { ideal: Math.round(width * aspect) },
      facingMode: deviceId ? undefined : "user",
      deviceId: deviceId ? { exact: deviceId } : undefined,
      frameRate: { ideal: 30 },
    },
  };

  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    const name = (err as DOMException)?.name;
    if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
      throw new LivenessError(
        "permission-denied",
        "Camera permission was denied. Please allow camera access and try again."
      );
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      throw new LivenessError("camera-error", "No camera was found on this device.");
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      throw new LivenessError(
        "camera-error",
        "The camera is already in use by another application."
      );
    }
    throw new LivenessError(
      "camera-error",
      "Could not start the camera.",
      (err as Error)?.message
    );
  }

  const video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;
  video.srcObject = stream;

  // Wait for dimensions to be known before capturing.
  await new Promise<void>((resolve, reject) => {
    const onMeta = () => {
      cleanup();
      resolve();
    };
    const onErr = () => {
      cleanup();
      reject(new LivenessError("camera-error", "Video failed to load."));
    };
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onMeta);
      video.removeEventListener("error", onErr);
    };
    video.addEventListener("loadedmetadata", onMeta);
    video.addEventListener("error", onErr);
  });

  try {
    await video.play();
  } catch {
    /* autoplay with muted+playsInline should succeed; ignore if already playing */
  }

  const stop = () => {
    stream.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  };

  return { stream, video, stop };
}
