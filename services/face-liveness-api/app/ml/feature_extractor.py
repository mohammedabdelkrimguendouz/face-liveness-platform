import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from pathlib import Path


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


def uniform_sampling(total_frames, num_frames):
    """Uniform sampling - EXACTLY same as training"""
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return indices
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def read_video_frames(video_path, num_frames=150, image_size=288):
    """Read and preprocess video frames - EXACTLY same as training"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total == 0:
        cap.release()
        return None
    
    # Uniform sampling - same as training
    indices = uniform_sampling(total, num_frames)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # BGR to RGB - same as training
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize - same as training
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Convert to tensor - same as training
    frames = np.array(frames, dtype=np.float32)
    frames = torch.tensor(frames)
    frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    frames /= 255.0  # Normalize to [0, 1] - same as training
    
    return frames


def extract_features_from_video(video_path, extractor, device, config):
    
    if not os.path.exists(video_path):
        return None, None, None

    # =====================================================
    # Stage 1: Video preprocessing
    # =====================================================

    start_read = time.perf_counter()

    frames = read_video_frames(
        video_path,
        num_frames=config.NUM_FRAMES,
        image_size=config.IMAGE_SIZE
    )

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

    read_time = time.perf_counter() - start_read

    if frames is None:
        return None, None, None

    # =====================================================
    # Stage 2: EfficientNet feature extraction
    # =====================================================

    start_extract = time.perf_counter()

    features = []

    with torch.no_grad():

        for i in range(
            0,
            len(frames),
            config.BATCH_SIZE
        ):

            batch = frames[
                i:i + config.BATCH_SIZE
            ].to(
                device,
                non_blocking=True
            )

            batch_features = extractor(batch)

            # Wait for GPU before copying to CPU
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)

            features.append(
                batch_features.cpu()
            )

    if len(features) == 0:
        return None, None, None

    features = torch.cat(
        features,
        dim=0
    )

    # =====================================================
    # Ensure exactly NUM_FRAMES
    # =====================================================

    if len(features) < config.NUM_FRAMES:

        pad_size = (
            config.NUM_FRAMES
            - len(features)
        )

        padding = torch.zeros(
            pad_size,
            config.FEATURE_SIZE
        )

        features = torch.cat(
            [features, padding],
            dim=0
        )

    elif len(features) > config.NUM_FRAMES:

        features = features[
            :config.NUM_FRAMES
        ]

    # =====================================================
    # Add batch dimension
    # =====================================================

    features = features.unsqueeze(0)

    # Make sure all GPU operations are completed
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

    extract_time = (
        time.perf_counter()
        - start_extract
    )

    return (
        features,
        read_time,
        extract_time
    )