"""DenseNet121 classifier."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def build_densenet121(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
