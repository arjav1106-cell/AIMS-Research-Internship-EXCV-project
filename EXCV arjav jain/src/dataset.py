"""Chest X-ray dataset loaders and transforms."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from src.config import (
    BATCH_SIZE,
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    NUM_WORKERS,
)


def _normalize() -> transforms.Normalize:
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        _normalize(),
    ])


def get_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        _normalize(),
    ])


def _build_imagefolder(
    split: Literal["train", "val", "test"],
    transform: transforms.Compose,
) -> datasets.ImageFolder:
    root = DATA_ROOT / split
    if not root.is_dir():
        raise FileNotFoundError(f"Split not found: {root}")
    return datasets.ImageFolder(root=str(root), transform=transform)


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_ds = _build_imagefolder("train", get_train_transforms())
    val_ds = _build_imagefolder("val", get_eval_transforms())
    test_ds = _build_imagefolder("test", get_eval_transforms())

    class_names = train_ds.classes
    pin = torch.cuda.is_available() and num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, class_names


class RawImageDataset(Dataset):
    """Load images with eval transforms for XAI (single-image batches)."""

    def __init__(self, image_paths: list[Path], transform=None):
        self.paths = image_paths
        self.transform = transform or get_eval_transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)
