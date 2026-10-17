"""ML layer for the face liveness API.

``feature_extractor`` retains the original research project's EfficientNet-B2
feature extractor (verbatim). ``bi_lstm_undercomplete`` provides the single
model class used by this project — the architecture that matches the shipped
checkpoint — and ``checkpoint_loader`` loads it with ``strict=True``.
``pipeline`` is the HTTP-independent inference entrypoint. The unused research
model classes (``LSTMAutoencoder``, ``BiLSTMVariationalAutoencoder``) and their
loader were removed; neither matches the shipped checkpoint.
"""

from .pipeline import LivenessPipeline, LivenessResult, PipelineConfig  # noqa: F401
