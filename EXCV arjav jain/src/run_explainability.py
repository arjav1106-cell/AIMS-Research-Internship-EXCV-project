"""Generate saliency maps and compute XAI metrics on test subset."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.config import (
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    METRICS_DIR,
    PLOTS_DIR,
    SALIENCY_DIR,
)
from src.dataset import get_eval_transforms
from src.evaluate import load_trained_model
from src.gradcam import generate_gradcam
from src.integrated_gradients import generate_integrated_gradients
from src.lfs import estimate_lung_mask, lung_focus_score, method_concordance
from src.utils import ensure_dirs, get_device, save_json, set_seed
from src.xai_metrics import (
    deletion_curve_metrics,
    insertion_auc,
    saliency_entropy,
)


def _load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    tfm = get_eval_transforms()
    img = Image.open(path).convert("RGB")
    t = tfm(img).unsqueeze(0).to(device)
    return t


def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    x = tensor.squeeze(0).cpu().numpy()
    x = (x * std + mean).transpose(1, 2, 0)
    return np.clip(x, 0, 1)


def _save_overlay(
    image: np.ndarray,
    saliency: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(saliency, cmap="jet")
    axes[1].set_title(title)
    axes[1].axis("off")
    axes[2].imshow(image)
    axes[2].imshow(saliency, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _collect_test_paths(max_per_class: int | None) -> list[tuple[Path, int, str]]:
    samples = []
    class_to_idx = {}
    for split_root in [DATA_ROOT / "test"]:
        for class_dir in sorted(split_root.iterdir()):
            if not class_dir.is_dir():
                continue
            idx = len(class_to_idx)
            class_to_idx[class_dir.name] = idx
            paths = sorted(class_dir.glob("*"))
            paths = [p for p in paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if max_per_class:
                paths = paths[:max_per_class]
            for p in paths:
                samples.append((p, class_to_idx[class_dir.name], class_dir.name))
    return samples


def run_explainability(
    max_per_class: int = 30,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    device = get_device()
    ensure_dirs(SALIENCY_DIR, METRICS_DIR, PLOTS_DIR)

    model, class_names = load_trained_model(device)
    samples = _collect_test_paths(max_per_class)
    if max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]

    records = []
    for path, label, class_name in tqdm(samples, desc="XAI"):
        x = _load_image_tensor(path, device)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            conf = torch.softmax(logits, dim=1)[0, pred].item()

        target = pred
        cam = generate_gradcam(model, x, target)
        ig = generate_integrated_gradients(model, x, target)

        gray = np.array(Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))
        lung_mask = estimate_lung_mask(gray)

        cam_aopc, cam_deletion_auc = deletion_curve_metrics(model, x, cam, target)
        ig_aopc, ig_deletion_auc = deletion_curve_metrics(model, x, ig, target)

        rec = {
            "path": str(path),
            "true_class": class_name,
            "true_idx": label,
            "pred_idx": pred,
            "pred_class": class_names[pred],
            "confidence": conf,
            "correct": pred == label,
            "gradcam": {
                "entropy": saliency_entropy(cam),
                "aopc": cam_aopc,
                "insertion_auc": insertion_auc(model, x, cam, target),
                "deletion_auc": cam_deletion_auc,
                "lfs": lung_focus_score(cam, lung_mask),
            },
            "integrated_gradients": {
                "entropy": saliency_entropy(ig),
                "aopc": ig_aopc,
                "insertion_auc": insertion_auc(model, x, ig, target),
                "deletion_auc": ig_deletion_auc,
                "lfs": lung_focus_score(ig, lung_mask),
            },
            "concordance": method_concordance(cam, ig),
        }
        records.append(rec)

        stem = path.stem
        display = _tensor_to_display(x)
        _save_overlay(display, cam, SALIENCY_DIR / "gradcam" / f"{stem}.png", "Grad-CAM")
        _save_overlay(display, ig, SALIENCY_DIR / "ig" / f"{stem}.png", "Integrated Gradients")

    summary = _aggregate(records, class_names)
    save_json({"per_sample": records, "summary": summary}, METRICS_DIR / "xai_metrics.json")
    _plot_summary(summary)
    print("\n=== XAI Summary (Grad-CAM vs IG) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def _aggregate(records: list[dict], class_names: list[str]) -> dict:
    if not records:
        return {}

    def mean_metric(method: str, key: str) -> float:
        vals = [r[method][key] for r in records]
        return float(np.mean(vals))

    summary = {
        "n_samples": len(records),
        "accuracy_subset": float(np.mean([r["correct"] for r in records])),
        "gradcam_entropy_mean": mean_metric("gradcam", "entropy"),
        "gradcam_aopc_mean": mean_metric("gradcam", "aopc"),
        "gradcam_insertion_auc_mean": mean_metric("gradcam", "insertion_auc"),
        "gradcam_deletion_auc_mean": mean_metric("gradcam", "deletion_auc"),
        "gradcam_lfs_mean": mean_metric("gradcam", "lfs"),
        "ig_entropy_mean": mean_metric("integrated_gradients", "entropy"),
        "ig_aopc_mean": mean_metric("integrated_gradients", "aopc"),
        "ig_insertion_auc_mean": mean_metric("integrated_gradients", "insertion_auc"),
        "ig_deletion_auc_mean": mean_metric("integrated_gradients", "deletion_auc"),
        "ig_lfs_mean": mean_metric("integrated_gradients", "lfs"),
        "concordance_spearman_mean": float(np.mean([r["concordance"]["spearman_rho"] for r in records])),
        "concordance_top5pct_overlap_mean": float(np.mean([r["concordance"]["top5pct_overlap"] for r in records])),
        "class_names": class_names,
    }
    return summary


def _plot_summary(summary: dict) -> None:
    if not summary:
        return
    metrics = ["entropy", "aopc", "insertion_auc", "deletion_auc", "lfs"]
    cam_vals = [summary[f"gradcam_{m}_mean"] for m in metrics]
    ig_vals = [summary[f"ig_{m}_mean"] for m in metrics]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, cam_vals, w, label="Grad-CAM")
    ax.bar(x + w / 2, ig_vals, w, label="Integrated Gradients")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_title("Mean XAI Metrics (test subset)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "xai_metrics_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_explainability(
        max_per_class=args.max_per_class,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
