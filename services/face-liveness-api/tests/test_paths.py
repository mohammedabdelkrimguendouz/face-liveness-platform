"""Path-resolution tests.

These exist to prove the guarantee the deploy depends on: the build step
(``scripts/fetch_models.py``) and the running application (``app.config``)
resolve the *same* absolute checkpoint directory under every environment the
deployment can produce. If that ever regresses, Render would report a healthy
deploy while the API could not load its models.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

from app.paths import (
    DEFAULT_MODELS_DIR,
    REQUIRED_CHECKPOINTS,
    SERVICE_ROOT,
    missing_checkpoints,
    resolve_models_dir,
)

API_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Loading the build script the way Render runs it
# ---------------------------------------------------------------------------
def _load_fetch_models():
    """Import ``scripts/fetch_models.py`` as a module, as the build step does."""
    path = API_ROOT / "scripts" / "fetch_models.py"
    spec = importlib.util.spec_from_file_location("fetch_models_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fetch_models():
    return _load_fetch_models()


@pytest.fixture
def clean_env(monkeypatch):
    """A bare environment with no models-dir override set."""
    monkeypatch.delenv("LIVENESS_MODELS_DIR", raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# The core invariant: one directory for both consumers
# ---------------------------------------------------------------------------
def test_build_script_and_app_resolve_identically_by_default(clean_env, fetch_models):
    """No env override -> build step and app agree, exactly as on Render."""
    assert fetch_models.models_dir() == resolve_models_dir()
    assert fetch_models.models_dir().is_absolute()
    assert resolve_models_dir().is_absolute()


def test_build_script_and_app_resolve_identically_when_env_set(
    monkeypatch, fetch_models, tmp_path
):
    """With LIVENESS_MODELS_DIR set, both consumers still agree."""
    monkeypatch.setenv("LIVENESS_MODELS_DIR", str(tmp_path))
    assert fetch_models.models_dir() == resolve_models_dir() == tmp_path.resolve()


def test_build_script_and_app_resolve_identically_for_relative_value(
    monkeypatch, fetch_models
):
    """A relative override is anchored to SERVICE_ROOT, not to the cwd.

    Render runs the build step and the server with cwd = rootDir, but any
    other cwd (a local run from the repo root, a CI shell) must produce the
    same absolute path.
    """
    monkeypatch.setenv("LIVENESS_MODELS_DIR", "models")
    assert fetch_models.models_dir() == resolve_models_dir()
    assert resolve_models_dir() == SERVICE_ROOT / "models"
    assert resolve_models_dir().is_absolute()


def test_relative_override_is_cwd_independent(monkeypatch, tmp_path):
    """Changing the working directory must not change where models resolve."""
    monkeypatch.setenv("LIVENESS_MODELS_DIR", "models")
    baseline = resolve_models_dir()

    monkeypatch.chdir(tmp_path)
    assert resolve_models_dir() == baseline

    monkeypatch.chdir(Path(os.sep))  # a very different cwd
    assert resolve_models_dir() == baseline

    assert resolve_models_dir() == SERVICE_ROOT / "models"


def test_empty_override_falls_back_to_default(clean_env):
    """An empty/whitespace value is treated as unset, not as cwd."""
    clean_env.setenv("LIVENESS_MODELS_DIR", "   ")
    assert resolve_models_dir() == DEFAULT_MODELS_DIR.resolve()


def test_settings_uses_the_same_resolver(clean_env):
    """``app.config.Settings`` must not carry its own default."""
    from app.config import Settings

    assert Settings().models_dir == str(resolve_models_dir())


def test_config_module_has_no_independent_default(clean_env):
    """Guards against a second hardcoded models path creeping back in."""
    import app.config as config_module

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert "_default_models_dir" not in source
    assert "My Code For Testing Local Vedio" not in source


# ---------------------------------------------------------------------------
# The committed checkpoints
# ---------------------------------------------------------------------------
def test_required_checkpoint_names():
    assert REQUIRED_CHECKPOINTS == (
        "EfficientNet_B2.pt",
        "ABC_BiLSTM_Undercomplete_AE.pt",
    )


def test_build_script_required_matches_paths_required(fetch_models):
    assert fetch_models.REQUIRED == REQUIRED_CHECKPOINTS


def test_default_models_dir_is_inside_the_service():
    assert DEFAULT_MODELS_DIR == SERVICE_ROOT / "models"
    assert DEFAULT_MODELS_DIR.parent == API_ROOT


def test_committed_checkpoints_are_present(clean_env):
    """Both weights must exist in the repo's models directory."""
    missing = missing_checkpoints()
    assert missing == [], f"missing checkpoints in {resolve_models_dir()}: {missing}"


def test_fetch_models_reports_ready_when_present(clean_env, fetch_models, capsys):
    assert fetch_models.main() == 0
    out = capsys.readouterr().out
    for name in REQUIRED_CHECKPOINTS:
        assert name in out


def test_fetch_models_required_files_exist_on_disk(fetch_models):
    directory = fetch_models.models_dir()
    for name in REQUIRED_CHECKPOINTS:
        file = directory / name
        assert file.is_file(), f"expected {file}"
        assert file.stat().st_size > 1_000_000, f"{name} looks like a stub"


# ---------------------------------------------------------------------------
# Missing-model configuration (must fail loudly, not silently)
# ---------------------------------------------------------------------------
def test_missing_models_detected_when_dir_absent(tmp_path):
    ghost = tmp_path / "not-there"
    assert missing_checkpoints(ghost) == list(REQUIRED_CHECKPOINTS)


def test_missing_models_detected_when_dir_empty(tmp_path):
    assert missing_checkpoints(tmp_path) == list(REQUIRED_CHECKPOINTS)


def test_partial_models_detected(tmp_path):
    (tmp_path / "EfficientNet_B2.pt").write_bytes(b"x" * 10)
    assert missing_checkpoints(tmp_path) == ["ABC_BiLSTM_Undercomplete_AE.pt"]


def test_fetch_models_fails_loudly_when_missing(
    monkeypatch, fetch_models, tmp_path, capsys
):
    """Empty override dir + no weights -> non-zero exit and a clear message."""
    monkeypatch.setenv("LIVENESS_MODELS_DIR", str(tmp_path))
    assert fetch_models.models_dir() == tmp_path.resolve()  # app would look here too
    assert fetch_models.main() == 1

    captured = capsys.readouterr()
    assert "ABC_BiLSTM_Undercomplete_AE.pt" in captured.err
    assert "EfficientNet_B2.pt" in captured.err
    assert str(tmp_path.resolve()) in captured.err


def test_app_would_look_in_the_same_place_as_the_failing_build_step(
    monkeypatch, fetch_models, tmp_path
):
    """The failure above is consistent: app and build step point at one path."""
    monkeypatch.setenv("LIVENESS_MODELS_DIR", str(tmp_path))
    assert fetch_models.main() == 1
    assert missing_checkpoints(resolve_models_dir()) == list(REQUIRED_CHECKPOINTS)


# ---------------------------------------------------------------------------
# Runtime behaviour: readiness tracks the model, health does not
# ---------------------------------------------------------------------------
def test_ready_reports_not_ready_without_models(monkeypatch, tmp_path):
    """With a bogus models dir the API must report 503, not fake readiness."""
    from fastapi.testclient import TestClient

    import app.main as main

    def _fail_load() -> None:
        main.PIPELINE._load_error = f"Model checkpoint not found in {tmp_path}"
        raise RuntimeError(f"Model checkpoint not found in {tmp_path}")

    # Simulate a build whose checkpoints never arrived: load fails, and the
    # pipeline is left with no extractor/model.
    monkeypatch.setattr(main.PIPELINE, "load", _fail_load)
    monkeypatch.setattr(main.PIPELINE, "extractor", None)
    monkeypatch.setattr(main.PIPELINE, "model", None)
    monkeypatch.setattr(main.PIPELINE, "_load_error", None)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        assert health.status_code == 200  # process health is unaffected
        assert health.json() == {"status": "ok"}

        ready = client.get("/ready")
        assert ready.status_code == 503
        body = ready.json()
        assert body["detail"]["ready"] is False
        assert str(tmp_path) in body["detail"]["error"]

    # Restore a clean pipeline for whatever runs next (the session-scoped
    # fixture below reloads the real weights through its own lifespan).
    monkeypatch.setattr(main.PIPELINE, "extractor", None)
    monkeypatch.setattr(main.PIPELINE, "model", None)
    monkeypatch.setattr(main.PIPELINE, "_load_error", None)


def test_ready_reports_ready_with_models(fastapi_client):
    """Default configuration: committed weights -> readiness passes."""
    r = fastapi_client.get("/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_settings_models_dir_points_at_committed_weights(clean_env):
    from app.config import Settings

    directory = Path(Settings().models_dir)
    for name in REQUIRED_CHECKPOINTS:
        assert (directory / name).is_file()


def test_resolver_is_reimportable_and_stable(clean_env):
    """Reloading the module (as a fresh process would) gives the same answer."""
    first = resolve_models_dir()
    reloaded = importlib.reload(sys.modules["app.paths"])
    assert reloaded.resolve_models_dir() == first
