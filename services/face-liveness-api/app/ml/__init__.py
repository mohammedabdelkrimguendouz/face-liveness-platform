"""ML layer for the face liveness API.

``feature_extractor`` / ``lstm_models`` / ``model_loader`` are copied verbatim
from the original research project. ``bi_lstm_undercomplete`` and
``checkpoint_loader`` reconstruct the model class required by the single
shipped checkpoint. ``pipeline`` is the HTTP-independent inference entrypoint.
"""

from .pipeline import LivenessPipeline, LivenessResult, PipelineConfig  # noqa: F401
