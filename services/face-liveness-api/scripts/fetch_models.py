#!/usr/bin/env python3
"""Verify the model checkpoints are present before the service starts.

The two checkpoints (``EfficientNet_B2.pt`` and ``ABC_BiLSTM_Undercomplete_AE.pt``)
are committed to this repository under ``services/face-liveness-api/models/``.
There is no download step and no external storage: this script only verifies
that the files the application will load are actually there.

It resolves the directory through :mod:`app.paths` — the exact same function
the application uses at runtime — so it can never verify one directory while
the API reads another.

If a checkpoint is missing it prints the expected path and exits non-zero, so
the deploy fails loudly instead of starting an API that can never become ready.

Usage:
    python scripts/fetch_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the `app` package importable when this file is run directly
# (``python scripts/fetch_models.py``), regardless of the working directory.
SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.paths import REQUIRED_CHECKPOINTS, missing_checkpoints, resolve_models_dir  # noqa: E402

# Kept for backwards compatibility with anything importing this script.
REQUIRED = REQUIRED_CHECKPOINTS
DEFAULT_DIR = SERVICE_ROOT / "models"


def models_dir() -> Path:
    """Directory the application will read checkpoints from."""
    return resolve_models_dir()


def have_models(directory: Path) -> list[str]:
    """Return the names of required checkpoints that are missing."""
    return missing_checkpoints(directory)


def main() -> int:
    directory = models_dir()
    missing = have_models(directory)

    if missing:
        print(
            "ERROR: model checkpoints are missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + f"\n\nExpected directory: {directory}\n\n"
            "The checkpoints are committed to this repository at\n"
            "  services/face-liveness-api/models/\n"
            "and are required for the service to start.\n\n"
            "If they are absent (e.g. a fresh clone that did not fetch them),\n"
            "copy them in from the research project:\n"
            "  'Face Anti Spoofing Project/My Code For Testing Local Vedio/"
            "test_models/saved_models/'\n\n"
            "See docs/RENDER.md for details.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"Models ready in {directory}", flush=True)
    for name in REQUIRED:
        print(f"  found {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
