#!/usr/bin/env python3
"""Ensure the model weights are present before the service starts.

The checkpoints are ~137 MB and are deliberately NOT committed to git. This
script gives you three ways to supply them, in priority order:

1. **Already present** at ``LIVENESS_MODELS_DIR`` -> verify and exit 0.
2. **Downloaded** from ``LIVENESS_MODELS_URL`` (a zip/tar containing both
   checkpoints) if that env var is set.
3. Otherwise -> print exact instructions and exit non-zero so the deploy fails
   loudly instead of starting an API that can never become ready.

Usage:
    python scripts/fetch_models.py
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REQUIRED = (
    "EfficientNet_B2.pt",
    "ABC_BiLSTM_Undercomplete_AE.pt",
)
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"


def models_dir() -> Path:
    return Path(os.environ.get("LIVENESS_MODELS_DIR", str(DEFAULT_DIR)))


def have_models(directory: Path) -> list[str]:
    """Return the names of required checkpoints that are missing."""
    if not directory.is_dir():
        return list(REQUIRED)
    return [name for name in REQUIRED if not (directory / name).is_file()]


def download(url: str, dest: Path) -> None:
    print(f"Downloading models from {url} ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "models.archive"
        urllib.request.urlretrieve(url, archive)  # noqa: S310 - operator-supplied URL
        members: list[str] = []
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
                members = zf.namelist()
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(tmp)  # noqa: S202 - trusted operator-supplied archive
                members = tf.getnames()
        else:
            raise SystemExit("LIVENESS_MODELS_URL did not point to a zip or tar archive")

        # Place every required checkpoint at the top level of models_dir.
        dest.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED:
            if (dest / name).is_file():
                continue
            match = next((Path(tmp) / m for m in members if m.endswith(name)), None)
            if match is None or not match.is_file():
                raise SystemExit(f"Archive is missing required checkpoint: {name}")
            (dest / name).write_bytes(match.read_bytes())
            print(f"  extracted {name}", flush=True)


def main() -> int:
    directory = models_dir()
    missing = have_models(directory)

    if not missing and os.environ.get("LIVENESS_MODELS_URL"):
        # Already satisfied; ignore the URL.
        print(f"Models present in {directory}", flush=True)
        return 0

    if missing and os.environ.get("LIVENESS_MODELS_URL"):
        try:
            download(os.environ["LIVENESS_MODELS_URL"], directory)
            missing = have_models(directory)
        except Exception as exc:  # noqa: BLE001 - report and fail the deploy
            print(f"ERROR: failed to download models: {exc}", file=sys.stderr, flush=True)
            return 1

    if missing:
        print(
            "ERROR: model checkpoints are missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + f"\n\nExpected directory: {directory}\n\n"
            "Ways to provide them:\n"
            "  1. Set LIVENESS_MODELS_DIR to a directory containing the checkpoints.\n"
            "  2. Set LIVENESS_MODELS_URL to a zip/tar URL that contains them\n"
            "     (e.g. a private S3/R2/GitHub-release link).\n"
            "  3. Locally: copy them from\n"
            "     'Face Anti Spoofing Project/My Code For Testing Local Vedio/"
            "test_models/saved_models/'\n\n"
            "See docs/RENDER.md for details. Weights are intentionally not in git.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"Models ready in {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
