"""Saliency robustness under input perturbations (noise, brightness, rotation)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.stats import spearmanr

from src.config import IMAGENET_MEAN, IMAGENET_STD, METRICS_DIR

NOISE_STD = 0.06
BRIGHTNESS_FACTOR = 1.15
ROTATION_DEGREES = 7.0
TOPK_IOU_FRACTION = 0.05
CONSTANT_ATOL = 1e-8

WEIGHT_SPEARMAN = 0.4
WEIGHT_SSIM = 0.3
WEIGHT_IOU = 0.3

PERTURBATIONS = ("noise", "brightness", "rotation")
SIMILARITY_METRICS = ("spearman", "ssim", "iou")

ROBUSTNESS_CSV_PATH = METRICS_DIR / "robustness.csv"


@dataclass
class RobustnessEvalStats:
    """Debug counters for the robustness pipeline."""

    total_samples: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    valid_saliency_pairs: int = 0
    constant_base_maps: int = 0
    constant_perturbed_maps: int = 0
    spearman_undefined_count: int = 0
    discarded_samples: int = 0

    per_perturbation: dict = field(default_factory=dict)

    def __post_init__(self):
        for p in PERTURBATIONS:
            self.per_perturbation[p] = {
                "spearman_values": [],
                "ssim_values": [],
                "iou_values": [],
                "robustness_values": [],
            }


def _denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def is_constant_map(saliency: np.ndarray, atol: float = CONSTANT_ATOL) -> bool:
    flat = saliency.astype(np.float64).flatten()
    return bool(np.ptp(flat) <= atol or np.std(flat) <= atol)


def maps_identical(map_a: np.ndarray, map_b: np.ndarray, atol: float = 1e-6) -> bool:
    return bool(np.allclose(map_a.astype(np.float64), map_b.astype(np.float64), atol=atol, rtol=0))


def _normalize01(saliency: np.ndarray) -> np.ndarray:
    sal = saliency.astype(np.float64)
    lo, hi = sal.min(), sal.max()
    if hi - lo <= CONSTANT_ATOL:
        return np.zeros_like(sal)
    return (sal - lo) / (hi - lo)


def _raw_spearman(map_a: np.ndarray, map_b: np.ndarray) -> tuple[float | None, bool]:
    """
    Returns (score in [0,1] or None if undefined, was_constant_handled).
    """
    a_const = is_constant_map(map_a)
    b_const = is_constant_map(map_b)
    if a_const and b_const:
        return (1.0 if maps_identical(map_a, map_b) else 0.0), True
    if a_const or b_const:
        return 0.0, True

    rho, _ = spearmanr(map_a.flatten(), map_b.flatten())
    if rho is None or np.isnan(rho):
        return None, False
    return float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0)), False


def spearman_robustness(map_a: np.ndarray, map_b: np.ndarray) -> float | None:
    score, _ = _raw_spearman(map_a, map_b)
    return score


def ssim_robustness(map_a: np.ndarray, map_b: np.ndarray) -> float:
    if is_constant_map(map_a) and is_constant_map(map_b):
        return 1.0 if maps_identical(map_a, map_b) else 0.0
    if is_constant_map(map_a) or is_constant_map(map_b):
        return 0.0

    a = _normalize01(map_a)
    b = _normalize01(map_b)
    c1, c2 = (0.01 ** 2), (0.03 ** 2)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    denom = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    if denom <= 0:
        return 0.0
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denom
    return float(np.clip(score, 0.0, 1.0))


def _binary_topk_mask(saliency: np.ndarray, top_fraction: float = TOPK_IOU_FRACTION) -> np.ndarray:
    flat = saliency.astype(np.float64).flatten()
    k = max(1, int(len(flat) * top_fraction))
    if is_constant_map(saliency):
        return np.ones_like(saliency, dtype=bool)
    threshold = np.partition(flat, -k)[-k]
    return saliency >= threshold


def iou_robustness(map_a: np.ndarray, map_b: np.ndarray) -> float:
    if is_constant_map(map_a) and is_constant_map(map_b):
        return 1.0 if maps_identical(map_a, map_b) else 0.0
    if is_constant_map(map_a) or is_constant_map(map_b):
        return 0.0

    mask_a = _binary_topk_mask(map_a)
    mask_b = _binary_topk_mask(map_b)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def weighted_robustness_score(
    spearman: float | None,
    ssim: float,
    iou: float,
) -> float:
    """
    Robustness = 0.4×Spearman + 0.3×SSIM + 0.3×IoU (all in [0,1]).
    If Spearman is undefined, renormalize: 0.5×SSIM + 0.5×IoU.
    """
    if spearman is None or np.isnan(spearman):
        return float(0.5 * ssim + 0.5 * iou)
    return float(
        WEIGHT_SPEARMAN * spearman + WEIGHT_SSIM * ssim + WEIGHT_IOU * iou
    )


def compare_saliency_maps(
    map_a: np.ndarray,
    map_b: np.ndarray,
    stats: RobustnessEvalStats | None = None,
    perturbation: str | None = None,
) -> dict[str, float | None]:
    if stats is not None:
        if is_constant_map(map_a):
            stats.constant_base_maps += 1
        if is_constant_map(map_b):
            stats.constant_perturbed_maps += 1

    spearman, _ = _raw_spearman(map_a, map_b)
    if spearman is None and stats is not None:
        stats.spearman_undefined_count += 1

    ssim = ssim_robustness(map_a, map_b)
    iou = iou_robustness(map_a, map_b)
    robustness = weighted_robustness_score(spearman, ssim, iou)

    if stats is not None and perturbation is not None:
        bucket = stats.per_perturbation[perturbation]
        if spearman is not None:
            bucket["spearman_values"].append(spearman)
        bucket["ssim_values"].append(ssim)
        bucket["iou_values"].append(iou)
        bucket["robustness_values"].append(robustness)
        stats.valid_saliency_pairs += 1

    return {
        "spearman": spearman,
        "ssim": ssim,
        "iou": iou,
        "robustness": robustness,
        "mean": robustness,
    }


def perturb_noise(image_01: torch.Tensor, std: float = NOISE_STD) -> torch.Tensor:
    noise = torch.randn_like(image_01) * std
    return (image_01 + noise).clamp(0, 1)


def perturb_brightness(image_01: torch.Tensor, factor: float = BRIGHTNESS_FACTOR) -> torch.Tensor:
    return (image_01 * factor).clamp(0, 1)


def perturb_rotation(image_01: torch.Tensor, degrees: float = ROTATION_DEGREES) -> torch.Tensor:
    return TF.rotate(image_01, angle=degrees, fill=0.0)


def apply_perturbation(
    input_tensor: torch.Tensor,
    perturb_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    image_01 = _denormalize(input_tensor)
    perturbed = perturb_fn(image_01)
    return _normalize(perturbed)


def robustness_score(
    saliency_fn: Callable[[torch.Tensor], np.ndarray],
    input_tensor: torch.Tensor,
    perturb_fn: Callable[[torch.Tensor], torch.Tensor],
    stats: RobustnessEvalStats | None = None,
    perturbation: str | None = None,
) -> dict[str, float | None]:
    base_map = saliency_fn(input_tensor)
    perturbed_input = apply_perturbation(input_tensor, perturb_fn)
    perturbed_map = saliency_fn(perturbed_input)
    return compare_saliency_maps(base_map, perturbed_map, stats=stats, perturbation=perturbation)


def _flatten_perturbation_scores(perturb_name: str, scores: dict) -> dict[str, float]:
    sp = scores["spearman"]
    out = {
        f"{perturb_name}_spearman": float(sp) if sp is not None else None,
        f"{perturb_name}_ssim": float(scores["ssim"]),
        f"{perturb_name}_iou": float(scores["iou"]),
        f"{perturb_name}_robustness": float(scores["robustness"]),
        f"{perturb_name}_mean": float(scores["robustness"]),
        perturb_name: float(scores["robustness"]),
    }
    return out


def evaluate_robustness_triplet(
    saliency_fn: Callable[[torch.Tensor], np.ndarray],
    input_tensor: torch.Tensor,
    stats: RobustnessEvalStats | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    perturb_scores: list[float] = []

    for name, fn in [
        ("noise", perturb_noise),
        ("brightness", perturb_brightness),
        ("rotation", perturb_rotation),
    ]:
        scores = robustness_score(
            saliency_fn, input_tensor, fn, stats=stats, perturbation=name,
        )
        out.update(_flatten_perturbation_scores(name, scores))
        perturb_scores.append(float(scores["robustness"]))

    out["robustness_mean"] = float(np.mean(perturb_scores))
    return out


def mean_robustness(robustness: dict[str, float]) -> float:
    if "robustness_mean" in robustness:
        return float(robustness["robustness_mean"])
    return float(np.mean([
        robustness.get("noise_mean", robustness.get("noise", 0.0)),
        robustness.get("brightness_mean", robustness.get("brightness", 0.0)),
        robustness.get("rotation_mean", robustness.get("rotation", 0.0)),
    ]))


def _samples_with_robustness(per_sample: list[dict], method_key: str) -> list[dict]:
    return [
        s for s in per_sample
        if "noise_robustness" in s.get(method_key, {})
        or "noise_mean" in s.get(method_key, {})
    ]


def aggregate_robustness_summary(per_sample: list[dict], method_key: str) -> dict[str, float]:
    """Aggregate over samples that have robustness (not only index 0)."""
    samples = _samples_with_robustness(per_sample, method_key)
    if not samples:
        return {}

    def _get(block: dict, perturb: str, metric: str) -> float | None:
        if metric == "robustness":
            return block.get(f"{perturb}_robustness", block.get(f"{perturb}_mean"))
        return block.get(f"{perturb}_{metric}")

    summary: dict[str, float] = {}
    for perturb in PERTURBATIONS:
        for metric in (*SIMILARITY_METRICS, "robustness"):
            vals = []
            for s in samples:
                v = _get(s[method_key], perturb, metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            if vals:
                summary[f"robustness_{perturb}_{metric}"] = float(np.mean(vals))
        rvals = [
            float(_get(s[method_key], perturb, "robustness"))
            for s in samples
            if _get(s[method_key], perturb, "robustness") is not None
        ]
        if rvals:
            summary[f"robustness_{perturb}"] = float(np.mean(rvals))

    overall = [s[method_key].get("robustness_mean") for s in samples]
    overall = [float(v) for v in overall if v is not None]
    if overall:
        summary["robustness_mean"] = float(np.mean(overall))
    return summary


def print_robustness_debug(stats: RobustnessEvalStats) -> None:
    print("\n" + "=" * 48)
    print("ROBUSTNESS EVALUATION DEBUG")
    print("=" * 48)
    print(f"Total samples attempted:     {stats.total_samples}")
    print(f"Successful evaluations:      {stats.successful_evaluations}")
    print(f"Failed evaluations:          {stats.failed_evaluations}")
    print(f"Valid saliency map pairs:    {stats.valid_saliency_pairs}")
    print(f"Constant base saliency maps: {stats.constant_base_maps}")
    print(f"Constant perturbed maps:     {stats.constant_perturbed_maps}")
    print(f"Spearman undefined (used SSIM+IoU): {stats.spearman_undefined_count}")
    print(f"Discarded samples:           {stats.discarded_samples}")
    print()
    for perturb in PERTURBATIONS:
        b = stats.per_perturbation[perturb]
        print(f"--- {perturb.upper()} ---")
        if b["spearman_values"]:
            print(f"  Mean Spearman: {np.mean(b['spearman_values']):.4f}  (n={len(b['spearman_values'])})")
        else:
            print("  Mean Spearman: N/A (all used SSIM+IoU fallback)")
        print(f"  Mean SSIM:     {np.mean(b['ssim_values']):.4f}" if b["ssim_values"] else "  Mean SSIM: N/A")
        print(f"  Mean IoU:      {np.mean(b['iou_values']):.4f}" if b["iou_values"] else "  Mean IoU: N/A")
        if b["robustness_values"]:
            print(f"  Mean Robustness (0.4S+0.3SSIM+0.3IoU): {np.mean(b['robustness_values']):.4f}")
        print()


def build_method_summary_table(
    per_sample: list[dict],
    method_key: str,
    method_label: str,
) -> dict[str, float]:
    agg = aggregate_robustness_summary(per_sample, method_key)
    return {
        "method": method_label,
        "noise_robustness": agg.get("robustness_noise"),
        "brightness_robustness": agg.get("robustness_brightness"),
        "rotation_robustness": agg.get("robustness_rotation"),
        "overall_robustness": agg.get("robustness_mean"),
        "noise_spearman_mean": agg.get("robustness_noise_spearman"),
        "noise_ssim_mean": agg.get("robustness_noise_ssim"),
        "noise_iou_mean": agg.get("robustness_noise_iou"),
        "brightness_spearman_mean": agg.get("robustness_brightness_spearman"),
        "brightness_ssim_mean": agg.get("robustness_brightness_ssim"),
        "brightness_iou_mean": agg.get("robustness_brightness_iou"),
        "rotation_spearman_mean": agg.get("robustness_rotation_spearman"),
        "rotation_ssim_mean": agg.get("robustness_rotation_ssim"),
        "rotation_iou_mean": agg.get("robustness_rotation_iou"),
    }


def print_summary_table(per_sample: list[dict]) -> None:
    print("\n" + "=" * 48)
    print("ROBUSTNESS SUMMARY TABLE")
    print("=" * 48)
    rows = [
        build_method_summary_table(per_sample, "gradcam", "Grad-CAM"),
        build_method_summary_table(per_sample, "integrated_gradients", "Integrated Gradients"),
    ]
    header = f"{'Method':<22} {'Noise':>10} {'Brightness':>12} {'Rotation':>10} {'Overall':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:<22} "
            f"{_fmt_val(r.get('noise_robustness')):>10} "
            f"{_fmt_val(r.get('brightness_robustness')):>12} "
            f"{_fmt_val(r.get('rotation_robustness')):>10} "
            f"{_fmt_val(r.get('overall_robustness')):>10}"
        )
    print()


def _fmt_val(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def save_robustness_csv(
    per_sample: list[dict],
    out_path: Path | None = None,
    stats: RobustnessEvalStats | None = None,
) -> Path:
    """Save detailed per-sample robustness to outputs/metrics/robustness.csv."""
    out_path = out_path or ROBUSTNESS_CSV_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["path", "true_class", "pred_class", "correct", "method"]
    for perturb in PERTURBATIONS:
        for metric in (*SIMILARITY_METRICS, "robustness"):
            fieldnames.append(f"{perturb}_{metric}")
    fieldnames.append("overall_robustness")

    rows: list[dict] = []
    for rec in per_sample:
        for method_prefix, method_key in [
            ("Grad-CAM", "gradcam"),
            ("Integrated Gradients", "integrated_gradients"),
        ]:
            block = rec.get(method_key, {})
            if "noise_robustness" not in block and "noise_mean" not in block:
                continue
            row: dict = {
                "path": rec.get("path", ""),
                "true_class": rec.get("true_class", ""),
                "pred_class": rec.get("pred_class", ""),
                "correct": rec.get("correct", ""),
                "method": method_prefix,
            }
            for perturb in PERTURBATIONS:
                for metric in SIMILARITY_METRICS:
                    key = f"{perturb}_{metric}"
                    row[key] = block.get(key)
                row[f"{perturb}_robustness"] = block.get(
                    f"{perturb}_robustness", block.get(f"{perturb}_mean"),
                )
            row["overall_robustness"] = block.get("robustness_mean")
            rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_path.parent / "robustness_aggregate_summary.csv"
    summary_fields = [
        "method", "noise_robustness", "brightness_robustness",
        "rotation_robustness", "overall_robustness",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for method_key, label in [("gradcam", "Grad-CAM"), ("integrated_gradients", "Integrated Gradients")]:
            w.writerow(build_method_summary_table(per_sample, method_key, label))

    return out_path


def load_robustness_aggregates_from_csv(csv_path: Path | None = None) -> dict[str, dict[str, float]]:
    """Load method-level aggregates from robustness_aggregate_summary.csv."""
    csv_path = csv_path or (METRICS_DIR / "robustness_aggregate_summary.csv")
    if not csv_path.is_file():
        return {}
    out: dict[str, dict[str, float]] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            out[method] = {
                "robustness_noise": _float_or_none(row.get("noise_robustness")),
                "robustness_brightness": _float_or_none(row.get("brightness_robustness")),
                "robustness_rotation": _float_or_none(row.get("rotation_robustness")),
                "robustness_mean": _float_or_none(row.get("overall_robustness")),
            }
    return out


def _float_or_none(s: str | None) -> float | None:
    if s is None or s == "" or s == "N/A":
        return None
    return float(s)
