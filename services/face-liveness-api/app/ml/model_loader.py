import os
import torch
from models.lstm_models import LSTMAutoencoder, BiLSTMVariationalAutoencoder


def load_lstm_undercomplete_model(model_path, device):
    """Load LSTM Undercomplete Autoencoder - EXACTLY same as training"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get parameters from checkpoint - same as training
    input_size = checkpoint.get("input_size", 1408)
    hidden_size = checkpoint.get("hidden_size", 512)
    latent_size = checkpoint.get("latent_size", 256)
    num_layers = checkpoint.get("num_layers", 1)
    
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers,
        reverse_target=True,
        autoregressive_decode=True,
        teacher_forcing_ratio=0.5
    )
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    threshold = checkpoint.get("threshold", 0.5)
    
    return model, threshold


def load_bilstm_variational_model(model_path, device):
    """Load BiLSTM Variational Autoencoder - EXACTLY same as training"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get parameters from checkpoint - same as training
    input_size = checkpoint.get("input_size", 1408)
    hidden_size = checkpoint.get("hidden_size", 512)
    latent_size = checkpoint.get("latent_size", 256)
    num_layers = checkpoint.get("num_layers", 1)
    
    model = BiLSTMVariationalAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_layers=num_layers
    )
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    threshold = checkpoint.get("threshold", 0.5)
    
    return model, threshold