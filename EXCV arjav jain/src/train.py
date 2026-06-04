"""Train DenseNet121 with early stopping."""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config import (
    BEST_MODEL_PATH,
    CHECKPOINT_PATH,
    EARLY_STOP_PATIENCE,
    LR,
    MAX_EPOCHS,
    MODELS_DIR,
    PLOTS_DIR,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
)
from src.dataset import get_dataloaders
from src.model import build_densenet121
from src.utils import ensure_dirs, get_device, plot_training_history, save_json, set_seed


@torch.no_grad()
def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, targets in tqdm(loader, leave=False, desc="train" if train else "eval"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_acc += _accuracy(logits, targets) * bs
            n += bs
    return total_loss / n, total_acc / n


def train_model(seed: int = 42) -> dict:
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    ensure_dirs(MODELS_DIR, PLOTS_DIR)
    train_loader, val_loader, _, class_names = get_dataloaders()
    print(f"Classes: {class_names}")

    model = build_densenet121(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device, train=False,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{MAX_EPOCHS} ({elapsed:.1f}s) | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "class_names": class_names,
        }, CHECKPOINT_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_names": class_names,
            }, BEST_MODEL_PATH)
            print(f"  -> Saved best model (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    plot_training_history(history, PLOTS_DIR / "training_history.png")
    save_json({"history": history, "best_epoch": best_epoch, "class_names": class_names},
              MODELS_DIR / "training_history.json")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_model(seed=args.seed)


if __name__ == "__main__":
    main()
