"""Grad-CAM saliency for DenseNet121."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import IMAGE_SIZE


def _densenet_target_layer(model: torch.nn.Module):
    return model.features.denseblock4.denselayer16.conv2


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """
    input_tensor: (1, C, H, W) normalized
    Returns saliency map (H, W) in [0, 1].
    """
    model.eval()
    targets = [ClassifierOutputTarget(target_class)]
    cam = GradCAM(model=model, target_layers=[_densenet_target_layer(model)])
    grayscale = cam(input_tensor=input_tensor, targets=targets)
    saliency = grayscale[0]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    if saliency.shape != (IMAGE_SIZE, IMAGE_SIZE):
        saliency_t = torch.from_numpy(saliency).float().unsqueeze(0).unsqueeze(0)
        saliency_t = F.interpolate(
            saliency_t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False,
        )
        saliency = saliency_t.squeeze().numpy()
    return saliency.astype(np.float32)
