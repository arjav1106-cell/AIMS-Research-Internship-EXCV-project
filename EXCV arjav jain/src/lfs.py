"""Lung Focus Score (LFS) and method concordance utilities."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import spearmanr


def estimate_lung_mask(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Lightweight lung mask via intensity thresholding + morphology.
    gray_uint8: (H, W) uint8 in [0, 255]
    Returns binary mask float32 in [0, 1].
    """
    blurred = cv2.GaussianBlur(gray_uint8, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh.mean() > 127:
        thresh = 255 - thresh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = (mask > 0).astype(np.float32)
    return mask


def lung_focus_score(saliency: np.ndarray, lung_mask: np.ndarray) -> float:
    """Fraction of total saliency mass falling inside lung mask."""
    sal = saliency.astype(np.float64)
    sal_sum = sal.sum() + 1e-12
    inside = (sal * lung_mask).sum()
    return float(inside / sal_sum)


def method_concordance(map_a: np.ndarray, map_b: np.ndarray) -> dict:
    """
    Compare two saliency methods: Spearman correlation and top-k overlap.
    Stronger research signal than LFS alone when no pathology masks exist.
    """
    a = map_a.flatten().astype(np.float64)
    b = map_b.flatten().astype(np.float64)
    rho, p = spearmanr(a, b)
    k = max(1, int(0.05 * len(a)))
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    overlap = len(top_a & top_b) / k
    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "spearman_p": float(p) if not np.isnan(p) else 1.0,
        "top5pct_overlap": float(overlap),
    }
