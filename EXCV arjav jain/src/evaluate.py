"""Test-set classification metrics."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from src.config import BEST_MODEL_PATH, CLASS_NAMES, METRICS_DIR, PLOTS_DIR
from src.dataset import get_dataloaders
from src.model import build_densenet121
from src.utils import ensure_dirs, get_device, plot_confusion_matrix, save_json, set_seed


def load_trained_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    model = build_densenet121(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, class_names


@torch.no_grad()
def evaluate_test(seed: int = 42) -> dict:
    set_seed(seed)
    device = get_device()
    ensure_dirs(METRICS_DIR, PLOTS_DIR)

    _, _, test_loader, _ = get_dataloaders()
    model, class_names = load_trained_model(device)

    all_preds, all_labels, all_probs = [], [], []
    for images, labels in tqdm(test_loader, desc="Test eval"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.vstack(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "class_names": class_names,
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, digits=4,
        ),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    save_json(metrics, METRICS_DIR / "test_classification_metrics.json")
    plot_confusion_matrix(cm, class_names, PLOTS_DIR / "confusion_matrix.png")

    print("\n=== Test Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f} (macro)")
    print(f"Recall:    {metrics['recall_macro']:.4f} (macro)")
    print(f"F1:        {metrics['f1_macro']:.4f} (macro)")
    print(metrics["classification_report"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_test(seed=args.seed)


if __name__ == "__main__":
    main()
