"""
Dedicated robustness evaluation pipeline.

Computes noise / brightness / rotation robustness for Grad-CAM and IG,
prints debug statistics, saves outputs/metrics/robustness.csv, and updates
xai_metrics_with_robustness.json for publication reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import METRICS_DIR
from src.dataset import get_eval_transforms
from src.evaluate import load_trained_model
from src.gradcam import generate_gradcam
from src.integrated_gradients import generate_integrated_gradients
from src.robustness import (
    ROBUSTNESS_CSV_PATH,
    RobustnessEvalStats,
    evaluate_robustness_triplet,
    print_robustness_debug,
    print_summary_table,
    save_robustness_csv,
)
from src.utils import ensure_dirs, get_device, save_json, set_seed


def _load_tensor(path: Path, device):
    import torch

    tfm = get_eval_transforms()
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


def _resolve_xai_path(xai_path: Path | None) -> Path:
    if xai_path and xai_path.is_file():
        return xai_path
    with_rob = METRICS_DIR / "xai_metrics_with_robustness.json"
    default = METRICS_DIR / "xai_metrics.json"
    return with_rob if with_rob.is_file() else default


def run_robustness_evaluation(
    xai_path: Path | None = None,
    out_dir: Path | None = None,
    max_samples: int | None = None,
    force_recompute: bool = False,
    seed: int = 42,
) -> tuple[list[dict], RobustnessEvalStats]:
    set_seed(seed)
    device = get_device()
    out_dir = out_dir or METRICS_DIR
    ensure_dirs(out_dir)

    xai_path = _resolve_xai_path(xai_path)
    if not xai_path.is_file():
        raise FileNotFoundError(f"XAI metrics not found: {xai_path}\nRun: python -m src.run_explainability")

    with open(xai_path, encoding="utf-8") as f:
        xai_data = json.load(f)
    per_sample = list(xai_data.get("per_sample", []))

    rob_count = _samples_with_robustness_count(per_sample)
    has_rob = (
        per_sample
        and not force_recompute
        and rob_count >= len(per_sample) * 0.9
        and rob_count > 0
    )

    stats = RobustnessEvalStats(total_samples=len(per_sample))

    if has_rob:
        print(f"Using existing robustness from {xai_path} ({rob_count}/{len(per_sample)} samples)")
        stats.successful_evaluations = rob_count
        for rec in per_sample:
            block = rec.get("gradcam", {})
            if "noise_robustness" not in block and "noise_mean" in block:
                for p in ("noise", "brightness", "rotation"):
                    block[f"{p}_robustness"] = block.get(f"{p}_mean")
    else:
        print(f"Computing robustness on GPU ({device})...")
        model, _ = load_trained_model(device)
        model.eval()

        indices = list(range(len(per_sample)))
        if max_samples and max_samples < len(indices):
            rng = np.random.default_rng(seed)
            indices = sorted(rng.choice(indices, size=max_samples, replace=False).tolist())
            print(f"Subset: {len(indices)} / {len(per_sample)} samples")

        for i in tqdm(indices, desc="Robustness"):
            rec = per_sample[i]
            path = Path(rec["path"])
            if not path.is_file():
                stats.failed_evaluations += 1
                stats.discarded_samples += 1
                continue
            try:
                x = _load_tensor(path, device)
                target = int(rec["pred_idx"])

                def cam_fn(t):
                    return generate_gradcam(model, t, target)

                def ig_fn(t):
                    return generate_integrated_gradients(model, t, target)

                rec["gradcam"].update(
                    evaluate_robustness_triplet(cam_fn, x, stats=stats),
                )
                rec["integrated_gradients"].update(
                    evaluate_robustness_triplet(ig_fn, x, stats=stats),
                )
                stats.successful_evaluations += 1
            except Exception as exc:
                stats.failed_evaluations += 1
                stats.discarded_samples += 1
                print(f"  [FAIL] {path.name}: {exc}")

        xai_data["per_sample"] = per_sample
        save_json(xai_data, out_dir / "xai_metrics_with_robustness.json")

    csv_path = save_robustness_csv(per_sample, out_dir / "robustness.csv", stats=stats)
    print_robustness_debug(stats)
    print_summary_table(per_sample)
    print(f"Saved: {csv_path}")
    print(f"Saved: {out_dir / 'robustness_aggregate_summary.csv'}")

    return per_sample, stats


def _samples_with_robustness_count(per_sample: list[dict]) -> int:
    n = 0
    for rec in per_sample:
        if "noise_robustness" in rec.get("gradcam", {}) or "noise_mean" in rec.get("gradcam", {}):
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for XAI methods")
    parser.add_argument("--xai-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--max-samples", type=int, default=None, help="None = all samples")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_robustness_evaluation(
        xai_path=args.xai_path,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        force_recompute=args.force_recompute,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
