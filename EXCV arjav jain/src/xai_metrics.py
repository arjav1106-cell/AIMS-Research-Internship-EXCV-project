"""Saliency evaluation: entropy, AOPC, insertion, deletion."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from src.config import IMAGENET_MEAN, IMAGENET_STD


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """(1,C,H,W) normalized -> [0,1] RGB."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def saliency_entropy(saliency: np.ndarray) -> float:
    """Shannon entropy of normalized saliency (lower = more focused)."""
    flat = saliency.flatten().astype(np.float64)
    flat = flat / (flat.sum() + 1e-12)
    flat = flat[flat > 1e-12]
    return float(-(flat * np.log(flat + 1e-12)).sum())


def _pixel_order(saliency: np.ndarray) -> np.ndarray:
    return np.argsort(saliency.flatten())[::-1]


@torch.no_grad()
def _target_confidence(model, image_01: torch.Tensor, target_class: int) -> float:
    logits = model(_normalize(image_01))
    return torch.softmax(logits, dim=1)[0, target_class].item()


def _apply_mask(
    image_01: torch.Tensor,
    saliency: np.ndarray,
    fraction: float,
    mode: str,
) -> torch.Tensor:
    """mode: 'deletion' masks top salient pixels; 'insertion' reveals top salient on baseline."""
    h, w = saliency.shape
    n_pixels = h * w
    k = int(fraction * n_pixels)
    order = _pixel_order(saliency)
    mask_flat = np.ones(n_pixels, dtype=np.float32)
    if k > 0:
        if mode == "deletion":
            mask_flat[order[:k]] = 0.0
        else:
            mask_flat[:] = 0.0
            mask_flat[order[:k]] = 1.0
    mask = torch.from_numpy(mask_flat.reshape(h, w)).to(image_01.device)
    mask = mask.unsqueeze(0).unsqueeze(0).expand_as(image_01)

    blurred = image_01.clone()
    img_np = image_01.squeeze(0).permute(1, 2, 0).cpu().numpy()
    for c in range(3):
        img_np[..., c] = gaussian_filter(img_np[..., c], sigma=8)
    baseline = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(image_01.device)

    if mode == "deletion":
        return image_01 * mask + baseline * (1 - mask)
    return baseline * (1 - mask) + image_01 * mask


def perturbation_curve(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    target_class: int,
    mode: str,
    steps: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns fractions [0..1] and target-class confidence at each step."""
    image_01 = _denormalize(input_tensor)
    fracs = np.linspace(0, 1, steps)
    confs = []
    for f in fracs:
        perturbed = _apply_mask(image_01, saliency, float(f), mode=mode)
        confs.append(_target_confidence(model, perturbed, target_class))
    return fracs, np.array(confs, dtype=np.float64)


def deletion_curve_metrics(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    target_class: int,
    steps: int = 20,
) -> tuple[float, float]:
    """
    Compute AOPC and deletion AUC from one deletion perturbation curve.

    Deletion AUC: area under remaining target-class confidence f(X_k) vs. fraction
    masked (trapezoidal rule over confidences).

    AOPC: mean cumulative drop from the original score,
        (1/K) * sum_k (f(X) - f(X_k)).
    Higher AOPC => larger average drop when removing salient pixels.
    """
    fracs, confs = perturbation_curve(
        model, input_tensor, saliency, target_class, mode="deletion", steps=steps,
    )
    p0 = float(confs[0])
    drops = p0 - confs
    aopc = float(np.mean(drops))
    deletion_auc_val = float(np.trapz(confs, fracs))
    return aopc, deletion_auc_val


def aopc_score(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    target_class: int,
    steps: int = 20,
) -> float:
    """AOPC = (1/K) * sum_k (f(X) - f(X_k)) along the deletion curve."""
    aopc, _ = deletion_curve_metrics(
        model, input_tensor, saliency, target_class, steps=steps,
    )
    return aopc


def insertion_auc(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    target_class: int,
    steps: int = 20,
) -> float:
    """AUC of insertion curve (higher = better)."""
    fracs, confs = perturbation_curve(
        model, input_tensor, saliency, target_class, mode="insertion", steps=steps,
    )
    return float(np.trapz(confs, fracs))


def deletion_auc(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    target_class: int,
    steps: int = 20,
) -> float:
    """Deletion AUC = trapz(f(X_k), fraction masked) — area under remaining confidence."""
    _, deletion_auc_val = deletion_curve_metrics(
        model, input_tensor, saliency, target_class, steps=steps,
    )
    return deletion_auc_val
