"""Integrated Gradients via Captum."""
from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.config import IMAGE_SIZE


def generate_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    steps: int = 32,
) -> np.ndarray:
    """
    input_tensor: (1, C, H, W) normalized, requires grad internally.
    Returns attribution map (H, W) in [0, 1].
    """
    model.eval()
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)
    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=steps,
        internal_batch_size=1,
    )
    attr = attributions.squeeze(0).detach().cpu().numpy()
    attr = np.abs(attr).sum(axis=0)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    if attr.shape != (IMAGE_SIZE, IMAGE_SIZE):
        attr_t = torch.from_numpy(attr).float().unsqueeze(0).unsqueeze(0)
        attr_t = torch.nn.functional.interpolate(
            attr_t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False,
        )
        attr = attr_t.squeeze().numpy()
    return attr.astype(np.float32)
