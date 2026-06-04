"""Publication-quality Grad-CAM vs Integrated Gradients comparison figure."""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image

from src.config import (
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PLOTS_DIR,
)
from src.dataset import get_eval_transforms
from src.evaluate import load_trained_model
from src.gradcam import generate_gradcam
from src.integrated_gradients import generate_integrated_gradients
from src.utils import ensure_dirs, get_device, set_seed

CLASS_FOLDERS = ["COVID", "Normal", "Viral Pneumonia"]
ROW_LABELS = ["COVID", "COVID", "Normal", "Normal", "Viral Pneumonia", "Viral Pneumonia"]
COLUMN_TITLES = ["Original", "Grad-CAM", "Integrated Gradients"]
OUT_PATH = PLOTS_DIR / "saliency_map_comparison.png"
SAMPLES_PER_CLASS = 2
SEED = 42
DPI = 300


def _collect_class_paths(class_name: str) -> list[Path]:
    root = DATA_ROOT / "test" / class_name
    paths = sorted(
        p for p in root.glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return paths


def _select_samples(seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    selected: list[tuple[Path, str]] = []
    for class_name in CLASS_FOLDERS:
        paths = _collect_class_paths(class_name)
        if len(paths) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Not enough images in test/{class_name}: need {SAMPLES_PER_CLASS}, found {len(paths)}"
            )
        picks = rng.sample(paths, SAMPLES_PER_CLASS)
        for p in picks:
            selected.append((p, class_name))
    return selected


def _load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    tfm = get_eval_transforms()
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    x = tensor.squeeze(0).cpu().numpy()
    x = (x * std + mean).transpose(1, 2, 0)
    return np.clip(x, 0, 1)


def _predict(model: torch.nn.Module, x: torch.Tensor, class_names: list[str]) -> tuple[int, str, float]:
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(dim=1).item())
        conf = float(probs[0, pred].item())
    return pred, class_names[pred], conf


def generate_figure(seed: int = SEED) -> Path:
    set_seed(seed)
    device = get_device()
    ensure_dirs(PLOTS_DIR)

    model, class_names = load_trained_model(device)
    samples = _select_samples(seed)

    print("Selected images for saliency comparison figure:")
    for i, (path, true_class) in enumerate(samples, 1):
        print(f"  {i}. [{true_class}] {path.name}")

    n_rows = len(samples)
    n_cols = 3
    fig_w = n_cols * 3.2
    fig_h = n_rows * 3.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = GridSpec(
        n_rows, n_cols, figure=fig,
        width_ratios=[1, 1, 1],
        wspace=0.08, hspace=0.12,
        left=0.10, right=0.98, top=0.92, bottom=0.04,
    )

    for row, ((path, true_class), row_label) in enumerate(zip(samples, ROW_LABELS)):
        x = _load_tensor(path, device)
        display = _tensor_to_display(x)
        pred_idx, pred_class, conf = _predict(model, x, class_names)
        target = pred_idx

        cam = generate_gradcam(model, x, target)
        ig = generate_integrated_gradients(model, x, target)

        pred_text = f"Pred: {pred_class} ({conf:.2f})"
        true_text = f"True: {true_class}"

        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(display, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 1:
                ax.imshow(cam, cmap="jet", alpha=0.50, vmin=0, vmax=1)
            elif col == 2:
                ax.imshow(ig, cmap="jet", alpha=0.50, vmin=0, vmax=1)

            ax.set_ylabel(
                row_label,
                fontsize=11,
                fontweight="bold",
                rotation=90,
                labelpad=12,
                va="center",
            )

            if row == 0:
                if col == 0:
                    ax.set_title(
                        f"{COLUMN_TITLES[col]}\n{path.stem}\n{true_text}",
                        fontsize=10,
                        fontweight="bold",
                        pad=10,
                    )
                else:
                    ax.set_title(
                        f"{COLUMN_TITLES[col]}\n{pred_text}",
                        fontsize=11,
                        fontweight="bold",
                        pad=10,
                    )
            elif col == 0:
                ax.set_title(f"{path.stem}\n{true_text}", fontsize=8, pad=4)
            else:
                ax.set_title(pred_text, fontsize=8, pad=4)

            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        "Saliency Map Comparison: Grad-CAM vs Integrated Gradients (DenseNet121)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)

    print(f"\nSaved figure -> {OUT_PATH.resolve()} ({DPI} dpi)")
    return OUT_PATH


def main():
    generate_figure()


if __name__ == "__main__":
    main()
