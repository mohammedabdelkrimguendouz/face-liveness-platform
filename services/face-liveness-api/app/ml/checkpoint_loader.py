"""Checkpoint loading for the single shipped temporal model.

Reads ``ABC_BiLSTM_Undercomplete_AE.pt`` and reconstructs
:class:`BiLSTMUndercompleteAE` with the exact hyper-parameters stored in the
checkpoint (input/hidden/latent sizes, num_layers, threshold). Strips any
``module.`` DataParallel prefix, as the original ``model_loader`` did.
"""

from __future__ import annotations

import os

import torch

from .bi_lstm_undercomplete import BiLSTMUndercompleteAE

DEFAULT_CHECKPOINT = "ABC_BiLSTM_Undercomplete_AE.pt"


def load_bilstm_undercomplete_model(model_path: str, device):
    """Load the BiLSTM undercomplete autoencoder exactly as trained.

    Returns ``(model, threshold)``. Raises FileNotFoundError / RuntimeError with
    actionable messages; never swallows a load failure.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {os.path.basename(model_path)}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_size = checkpoint.get("input_size", 1408)
    hidden_size = checkpoint.get("hidden_size", 512)
    latent_size = checkpoint.get("latent_size", 256)
    num_layers = checkpoint.get("num_layers", 1)
    reverse_target = checkpoint.get("reverse_target", True)

    model = BiLSTMUndercompleteAE(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        reverse_target=reverse_target,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    # strict=True is deliberate: a shape mismatch must surface as a startup
    # failure, not silently degraded inference.
    model.load_state_dict(cleaned, strict=True)
    model = model.to(device)
    model.eval()

    threshold = float(checkpoint.get("threshold", 0.5))
    return model, threshold
