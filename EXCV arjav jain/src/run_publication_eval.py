"""
Publication-quality evaluation report generator.

Loads classification + XAI results, optionally computes robustness metrics,
and writes JSON / CSV / human-readable tables to outputs/metrics/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import IMAGE_SIZE, METRICS_DIR
from src.dataset import get_eval_transforms
from src.evaluate import load_trained_model
from src.gradcam import generate_gradcam
from src.integrated_gradients import generate_integrated_gradients
from src.publication_report import (
    build_full_report,
    format_text_report,
    load_classification_metrics,
    load_xai_metrics,
    save_publication_outputs,
)
from src.robustness import ROBUSTNESS_CSV_PATH, evaluate_robustness_triplet, save_robustness_csv
from src.run_robustness_eval import run_robustness_evaluation, _resolve_xai_path
from src.utils import ensure_dirs, get_device, save_json, set_seed


def _backfill_robustness_keys(per_sample: list[dict]) -> None:
    """Map legacy noise_mean keys to noise_robustness for aggregation."""
    for rec in per_sample:
        for method_key in ("gradcam", "integrated_gradients"):
            block = rec.get(method_key, {})
            for perturb in ("noise", "brightness", "rotation"):
                if f"{perturb}_mean" in block and f"{perturb}_robustness" not in block:
                    block[f"{perturb}_robustness"] = block[f"{perturb}_mean"]


def _load_tensor(path: Path, device) -> "torch.Tensor":
    import torch

    tfm = get_eval_transforms()
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


def enrich_with_robustness(
    per_sample: list[dict],
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Add noise / brightness / rotation robustness to each per-sample record."""
    import torch

    set_seed(seed)
    device = get_device()
    model, _ = load_trained_model(device)
    model.eval()

    indices = list(range(len(per_sample)))
    if max_samples and max_samples < len(indices):
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, size=max_samples, replace=False).tolist())

    for i in tqdm(indices, desc="Robustness"):
        rec = per_sample[i]
        path = Path(rec["path"])
        if not path.is_file():
            continue
        x = _load_tensor(path, device)
        target = int(rec["pred_idx"])

        def cam_fn(t: torch.Tensor) -> np.ndarray:
            return generate_gradcam(model, t, target)

        def ig_fn(t: torch.Tensor) -> np.ndarray:
            return generate_integrated_gradients(model, t, target)

        rec["gradcam"].update(evaluate_robustness_triplet(cam_fn, x))
        rec["integrated_gradients"].update(evaluate_robustness_triplet(ig_fn, x))

    return per_sample


def _export_per_sample_csv(per_sample: list[dict], out_path: Path) -> None:
    if not per_sample:
        return
    fieldnames = [
        "path", "true_class", "pred_class", "correct", "confidence",
        "gradcam_aopc", "gradcam_insertion_auc", "gradcam_deletion_auc",
        "gradcam_entropy", "gradcam_lfs",
        "gradcam_robustness_mean",
        "ig_aopc", "ig_insertion_auc", "ig_deletion_auc",
        "ig_entropy", "ig_lfs",
        "ig_robustness_mean",
        "spearman_rho", "top5pct_overlap",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rec in per_sample:
            row = {
                "path": rec["path"],
                "true_class": rec["true_class"],
                "pred_class": rec["pred_class"],
                "correct": rec["correct"],
                "confidence": rec["confidence"],
                "gradcam_aopc": rec["gradcam"]["aopc"],
                "gradcam_insertion_auc": rec["gradcam"]["insertion_auc"],
                "gradcam_deletion_auc": rec["gradcam"]["deletion_auc"],
                "gradcam_entropy": rec["gradcam"]["entropy"],
                "gradcam_lfs": rec["gradcam"]["lfs"],
                "ig_aopc": rec["integrated_gradients"]["aopc"],
                "ig_insertion_auc": rec["integrated_gradients"]["insertion_auc"],
                "ig_deletion_auc": rec["integrated_gradients"]["deletion_auc"],
                "ig_entropy": rec["integrated_gradients"]["entropy"],
                "ig_lfs": rec["integrated_gradients"]["lfs"],
                "spearman_rho": rec["concordance"]["spearman_rho"],
                "top5pct_overlap": rec["concordance"]["top5pct_overlap"],
            }
            for prefix, key in [("gradcam", "gradcam"), ("ig", "integrated_gradients")]:
                block = rec[key]
                if "noise_mean" in block:
                    for perturb in ["noise", "brightness", "rotation"]:
                        for metric in ["spearman", "ssim", "iou", "mean"]:
                            col = f"{prefix}_{perturb}_{metric}"
                            row[col] = block.get(f"{perturb}_{metric}")
                    row[f"{prefix}_robustness_mean"] = block.get("robustness_mean")
            w.writerow(row)


def run_publication_eval(
    classification_path: Path = METRICS_DIR / "test_classification_metrics.json",
    xai_path: Path = METRICS_DIR / "xai_metrics.json",
    out_dir: Path = METRICS_DIR,
    compute_robustness: bool = False,
    robustness_max_samples: int | None = 30,
    seed: int = 42,
) -> dict:
    ensure_dirs(out_dir)

    if not classification_path.is_file():
        raise FileNotFoundError(
            f"Classification metrics not found: {classification_path}\n"
            "Run: python -m src.evaluate"
        )
    xai_path = _resolve_xai_path(xai_path if xai_path.is_file() else None)
    if not xai_path.is_file():
        raise FileNotFoundError(
            f"XAI metrics not found: {xai_path}\n"
            "Run: python -m src.run_explainability"
        )

    classification = load_classification_metrics(classification_path)
    xai_data = load_xai_metrics(xai_path)
    per_sample = list(xai_data.get("per_sample", []))
    _backfill_robustness_keys(per_sample)

    rob_count = sum(
        1 for r in per_sample
        if "noise_robustness" in r.get("gradcam", {}) or "noise_mean" in r.get("gradcam", {})
    )
    if compute_robustness:
        per_sample, _ = run_robustness_evaluation(
            xai_path=xai_path,
            out_dir=out_dir,
            max_samples=robustness_max_samples,
            force_recompute=True,
            seed=seed,
        )
        xai_data["per_sample"] = per_sample
    elif rob_count > 0:
        save_robustness_csv(per_sample, ROBUSTNESS_CSV_PATH)
        print(f"Exported robustness for {rob_count} samples -> {ROBUSTNESS_CSV_PATH}")
    elif ROBUSTNESS_CSV_PATH.is_file():
        print(f"Using robustness aggregates from {ROBUSTNESS_CSV_PATH}")
    else:
        print(
            "WARNING: No robustness data. Run:\n"
            "  python -m src.run_robustness_eval --force-recompute"
        )

    report = build_full_report(classification, {"per_sample": per_sample, "summary": xai_data.get("summary", {})})
    paths = save_publication_outputs(report, out_dir)
    _export_per_sample_csv(per_sample, out_dir / "publication_per_sample_metrics.csv")

    print(format_text_report(report))
    print("\nSaved outputs:")
    for name, p in paths.items():
        print(f"  {name}: {p}")
    print(f"  per_sample_csv: {out_dir / 'publication_per_sample_metrics.csv'}")
    print(f"  method_comparison: {out_dir / 'publication_method_comparison.csv'}")
    print(f"  method_ranking: {out_dir / 'publication_method_ranking.csv'}")
    print(f"  research_gaps: {out_dir / 'publication_research_gaps.csv'}")
    print(f"  metrics_summary: {out_dir / 'publication_metrics_summary.csv'}")
    print(f"  classification: {out_dir / 'publication_classification.csv'}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality evaluation report")
    parser.add_argument(
        "--classification-path",
        type=Path,
        default=METRICS_DIR / "test_classification_metrics.json",
    )
    parser.add_argument("--xai-path", type=Path, default=METRICS_DIR / "xai_metrics.json")
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    parser.add_argument(
        "--compute-robustness",
        action="store_true",
        help="Compute noise/brightness/rotation robustness (slow; uses GPU)",
    )
    parser.add_argument(
        "--robustness-max-samples",
        type=int,
        default=30,
        help="Max images for robustness evaluation (default 30)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_publication_eval(
        classification_path=args.classification_path,
        xai_path=args.xai_path,
        out_dir=args.out_dir,
        compute_robustness=args.compute_robustness,
        robustness_max_samples=args.robustness_max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
