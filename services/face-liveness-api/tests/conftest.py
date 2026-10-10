"""Shared pytest fixtures for the face liveness API."""

from __future__ import annotations

import os
import sys

import pytest

# Make the `app` package importable and point at the real model weights.
API_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

REPO_MODELS = os.path.abspath(os.path.join(
    API_ROOT, "..", "..", "..",
    "My Code For Testing Local Vedio", "test_models", "saved_models",
))
REPO_VIDEOS = os.path.abspath(os.path.join(
    API_ROOT, "..", "..", "..",
    "My Code For Testing Local Vedio", "test_models", "test_videos",
))


def pytest_configure(config):
    config.addinivalue_line("markers", "real: tests that execute the real ML model")
    config.addinivalue_line("markers", "slow: long-running tests")


@pytest.fixture(scope="session")
def models_dir() -> str:
    if not os.path.isdir(REPO_MODELS):
        pytest.skip("Model weights not available")
    return REPO_MODELS


@pytest.fixture(scope="session")
def videos_dir() -> str:
    if not os.path.isdir(REPO_VIDEOS):
        pytest.skip("Test videos not available")
    return REPO_VIDEOS


@pytest.fixture(scope="session")
def real_pipeline(models_dir):
    """A loaded LivenessPipeline against the real checkpoints."""
    from app.ml import LivenessPipeline, PipelineConfig
    pipe = LivenessPipeline(models_dir=models_dir, device="cpu", config=PipelineConfig(batch_size=8))
    pipe.load()
    return pipe


@pytest.fixture(scope="session")
def fastapi_client(real_pipeline):
    """TestClient with the real model loaded via app lifespan."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        yield client
