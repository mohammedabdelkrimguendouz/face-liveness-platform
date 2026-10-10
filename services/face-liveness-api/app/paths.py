"""Canonical model-directory resolution.

The two required checkpoints are committed to this repository at::

    services/face-liveness-api/models/
    ├── EfficientNet_B2.pt
    └── ABC_BiLSTM_Undercomplete_AE.pt

Every component that needs them resolves the directory through this module:

* :mod:`app.config` (runtime, so the API loads the right files)
* ``scripts/fetch_models.py`` (build step, so it verifies the same files)
* ``tests/conftest.py``

Because they share one function, the build step can never verify a different
directory from the one the application later reads. There is no download step
and no external storage: the weights ship with the repo.

Relative overrides are anchored to :data:`SERVICE_ROOT` rather than to the
process working directory, so the value is identical whether it is resolved at
build time (``cwd = services/face-liveness-api``) or at runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

#: ``services/face-liveness-api/`` — derived from this file, never from cwd.
SERVICE_ROOT = Path(__file__).resolve().parent.parent

#: Canonical, in-repo checkpoint directory (committed to git).
DEFAULT_MODELS_DIR: Path = SERVICE_ROOT / "models"

#: Environment variable that overrides :data:`DEFAULT_MODELS_DIR`.
MODELS_DIR_ENV = "LIVENESS_MODELS_DIR"

#: Checkpoints the service cannot run without.
REQUIRED_CHECKPOINTS: Tuple[str, ...] = (
    "EfficientNet_B2.pt",
    "ABC_BiLSTM_Undercomplete_AE.pt",
)


def resolve_models_dir() -> Path:
    """Return the absolute directory holding the model checkpoints.

    Resolution order:

    1. ``$LIVENESS_MODELS_DIR``, if set and non-empty. A relative value is
       resolved against :data:`SERVICE_ROOT` (not the working directory) so
       build and runtime always agree.
    2. Otherwise :data:`DEFAULT_MODELS_DIR`, i.e. ``<service root>/models``.

    The result is always absolute, so callers may rely on it regardless of
    the working directory they were started from.
    """
    override = os.environ.get(MODELS_DIR_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = SERVICE_ROOT / candidate
        return candidate.resolve()
    return DEFAULT_MODELS_DIR.resolve()


def missing_checkpoints(directory: Path | None = None) -> list[str]:
    """Return the names of required checkpoints absent from ``directory``."""
    target = directory if directory is not None else resolve_models_dir()
    if not target.is_dir():
        return list(REQUIRED_CHECKPOINTS)
    return [name for name in REQUIRED_CHECKPOINTS if not (target / name).is_file()]
