"""Face liveness inference pipeline.

Wraps the existing, verified research pipeline:

    frames -> EfficientNet-B2 features (1408-d) -> BiLSTM undercomplete AE
           -> reconstruction error -> anomaly score -> decision

This module is deliberately independent of FastAPI/HTTP so it can be unit
tested and reused on its own.

Preprocessing is preserved *exactly* from the original training/eval code
(``models/feature_extractor.py``):

    * channel order  : BGR (as decoded by OpenCV) -> RGB
    * resize         : ``cv2.resize(frame, (288, 288))``
    * normalization  : float32 in [0, 1]  (``/255.0``)
    * layout         : NCHW
    * backbone norm  : ImageNet mean/std applied inside the extractor
    * sampling       : uniform to exactly NUM_FRAMES (150); if fewer frames are
                       supplied, repeat the last frame (same as the research
                       project's ``uniform_sampling`` padding branch)
    * score          : ``1 - exp(-MSE)`` compared against the checkpoint threshold
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch

from .checkpoint_loader import DEFAULT_CHECKPOINT, load_bilstm_undercomplete_model
from .feature_extractor import load_feature_extractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Model hyper-parameters. Defaults match the shipped checkpoints."""

    image_size: int = 288
    num_frames: int = 150
    feature_size: int = 1408
    batch_size: int = 8  # lowered from training's 16 to bound CPU activation memory


@dataclass(frozen=True)
class LivenessResult:
    prediction: str            # "live" | "spoof"
    anomaly_score: float       # 1 - exp(-MSE); NOT a calibrated probability
    reconstruction_error: float
    threshold: float
    frames_processed: int
    inference_ms: float
    mean_frame_error: Optional[float] = None


class LivenessPipeline:
    """Loads models once and runs the full feature-extraction + AE pipeline."""

    def __init__(
        self,
        models_dir: str,
        device: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
    ):
        self.models_dir = models_dir
        self.config = config or PipelineConfig()
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.extractor = None
        self.model = None
        self.threshold = 0.5
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self.extractor is not None and self.model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load(self) -> None:
        """Load both models. Raises on failure so startup/readiness is explicit."""
        try:
            self.extractor = load_feature_extractor(self.device, _CfgShim(self.models_dir, self.config))
            self.extractor.eval()

            checkpoint_path = os.path.join(self.models_dir, DEFAULT_CHECKPOINT)
            self.model, self.threshold = load_bilstm_undercomplete_model(checkpoint_path, self.device)
            self.model.eval()
            self._load_error = None
            logger.info(
                "Liveness pipeline ready on %s (threshold=%.3f, num_frames=%d)",
                self.device, self.threshold, self.config.num_frames,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced through /ready
            self._load_error = str(exc)
            logger.exception("Failed to load liveness pipeline")
            raise

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decode_frames(self, encoded_frames: List[bytes]) -> Optional[torch.Tensor]:
        """Decode encoded (JPEG/PNG) frames into the model's NCHW float tensor.

        Returns None if no frame could be decoded. Mirrors the research
        project's ``read_video_frames`` preprocessing but takes the frames from
        memory instead of a file.
        """
        frames: List[np.ndarray] = []
        for buf in encoded_frames:
            arr = np.frombuffer(buf, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.config.image_size, self.config.image_size))
            frames.append(rgb)

        if not frames:
            return None

        # Uniform-sample / pad to exactly NUM_FRAMES (same contract as training).
        target = self.config.num_frames
        total = len(frames)
        if total < target:
            indices: List[int] = list(range(total))
            while len(indices) < target:
                indices.append(indices[-1])
        else:
            indices = np.linspace(0, total - 1, target, dtype=int).tolist()
        sampled = [frames[i] for i in indices]

        tensor = np.array(sampled, dtype=np.float32)
        tensor = torch.from_numpy(tensor).permute(0, 3, 1, 2) / 255.0
        return tensor

    @torch.no_grad()
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        features = []
        bs = self.config.batch_size
        for i in range(0, len(frames), bs):
            batch = frames[i:i + bs].to(self.device, non_blocking=True)
            features.append(self.extractor(batch).cpu())
        feats = torch.cat(features, dim=0)

        # Enforce exactly NUM_FRAMES (zero-pad / truncate), as in training.
        target = self.config.num_frames
        if len(feats) < target:
            pad = torch.zeros(target - len(feats), self.config.feature_size)
            feats = torch.cat([feats, pad], dim=0)
        elif len(feats) > target:
            feats = feats[:target]
        return feats.unsqueeze(0)

    @torch.no_grad()
    def predict(self, encoded_frames: List[bytes]) -> LivenessResult:
        """Run the full pipeline on an ordered list of encoded frames."""
        if not self.is_ready:
            raise RuntimeError("Pipeline not loaded")
        if not encoded_frames:
            raise ValueError("No frames supplied")

        start = time.perf_counter()

        frames = self.decode_frames(encoded_frames)
        if frames is None:
            raise ValueError("Could not decode any supplied frames")

        features = self.extract_features(frames).to(self.device)

        t0 = time.perf_counter()
        reconstruction = self.model(features)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        total_error = torch.mean((reconstruction - features) ** 2, dim=(1, 2))
        per_frame = torch.mean((reconstruction - features) ** 2, dim=2)[0]
        anomaly = 1.0 - torch.exp(-total_error)

        score = float(anomaly[0].item())
        prediction = "spoof" if score > self.threshold else "live"

        logger.info(
            "Inference done: score=%.4f pred=%s total=%.2fs",
            score, prediction, time.perf_counter() - start,
        )

        return LivenessResult(
            prediction=prediction,
            anomaly_score=score,
            reconstruction_error=float(total_error[0].item()),
            threshold=self.threshold,
            frames_processed=len(encoded_frames),
            inference_ms=inference_ms,
            mean_frame_error=float(torch.mean(per_frame).item()),
        )


class _CfgShim:
    """Minimal config object satisfying ``load_feature_extractor``'s interface."""

    def __init__(self, models_dir: str, cfg: PipelineConfig):
        self.MODELS_DIR = models_dir
        self.FEATURE_EXTRACTOR_CHECKPOINT = os.path.join(models_dir, "EfficientNet_B2.pt")
        self.FEATURE_EXTRACTOR_MODEL = os.path.join(models_dir, "EfficientNet_B2_FeatureExtractor.pth")
