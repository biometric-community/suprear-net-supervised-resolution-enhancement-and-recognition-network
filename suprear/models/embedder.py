"""HQL discriminative embedder (paper LightCNN29 → ResNet18 128-D, D4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceEmbedder(nn.Module):
    def __init__(self, emb_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pipeline tensors are Normalize → [-1,1]; map to [0,1] for ImageNet backbone.
        x = (x + 1.0) * 0.5
        h = torch.flatten(self.features(x), 1)
        h = self.bn(self.fc(h))
        return F.normalize(h, p=2, dim=1)
