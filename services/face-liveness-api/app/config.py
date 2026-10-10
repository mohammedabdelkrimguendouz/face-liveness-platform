"""Environment-based configuration for the face liveness API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_models_dir() -> str:
    # Default to the repo's saved_models, relative to this file, unless overridden.
    here = os.path.dirname(os.path.abspath(__file__))
    platform_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    project_root = os.path.abspath(os.path.join(platform_root, ".."))
    return os.path.join(project_root, "My Code For Testing Local Vedio", "test_models", "saved_models")


@dataclass
class Settings:
    # Model
    models_dir: str = field(default_factory=lambda: os.environ.get("LIVENESS_MODELS_DIR", _default_models_dir()))
    device: str = os.environ.get("LIVENESS_DEVICE", "cpu")
    batch_size: int = int(os.environ.get("LIVENESS_BATCH_SIZE", "8"))

    # Sessions
    num_frames: int = int(os.environ.get("LIVENESS_NUM_FRAMES", "150"))
    session_ttl_seconds: int = int(os.environ.get("LIVENESS_SESSION_TTL", "300"))
    max_sessions: int = int(os.environ.get("LIVENESS_MAX_SESSIONS", "50"))
    max_inferences: int = int(os.environ.get("LIVENESS_MAX_CONCURRENT_INFERENCES", "1"))

    # Frame limits
    max_frame_bytes: int = int(os.environ.get("LIVENESS_MAX_FRAME_BYTES", str(2 * 1024 * 1024)))
    max_frames_per_session: int = int(os.environ.get("LIVENESS_MAX_FRAMES", "400"))

    # CORS (comma-separated). Empty disables CORS middleware (e.g. same-origin).
    allowed_origins: List[str] = field(
        default_factory=lambda: _csv(os.environ.get("LIVENESS_ALLOWED_ORIGINS", "http://localhost:5173"))
    )

    # Server
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    log_level: str = os.environ.get("LOG_LEVEL", "info")


settings = Settings()
