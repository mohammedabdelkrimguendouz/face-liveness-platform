import os
import torch
import torch.nn as nn


class EfficientNetFeatureExtractor(nn.Module):
    """EfficientNet-B2 - EXACTLY same as training, loads from saved_models"""
    
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        # Same as training
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
        
        weights = EfficientNet_B2_Weights.DEFAULT
        model = efficientnet_b2(weights=weights)
        
        self.features = model.features
        self.avgpool = model.avgpool
        
        # ImageNet normalization - same as training
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
        # Load saved weights if provided
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def _load_from_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.load_state_dict(new_state_dict, strict=False)
            
            # Load normalization if available
            if "normalization_mean" in checkpoint and "normalization_std" in checkpoint:
                mean = torch.tensor(checkpoint["normalization_mean"]).view(1, 3, 1, 1)
                std = torch.tensor(checkpoint["normalization_std"]).view(1, 3, 1, 1)
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)
                
        except Exception:
            pass
    
    def forward(self, x):
        # Normalize with ImageNet stats - same as training
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_feature_extractor(device, config):
    """Load feature extractor from saved_models - same as training"""
    
    if os.path.exists(config.FEATURE_EXTRACTOR_CHECKPOINT):
        try:
            extractor = EfficientNetFeatureExtractor(
                checkpoint_path=config.FEATURE_EXTRACTOR_CHECKPOINT
            )
            extractor = extractor.to(device)
            return extractor
        except Exception:
            pass
    
    if os.path.exists(config.FEATURE_EXTRACTOR_MODEL):
        try:
            state_dict = torch.load(config.FEATURE_EXTRACTOR_MODEL, map_location='cpu', weights_only=False)
            extractor = EfficientNetFeatureExtractor()
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            extractor.load_state_dict(new_state_dict, strict=False)
            extractor = extractor.to(device)
            return extractor
        except Exception:
            pass
    
    raise FileNotFoundError(
        f"No saved feature extractor found in {config.MODELS_DIR}"
    )
