"""SURE-GAN: generator + discriminator + identity classifier (paper Fig. 3 / Sec. II-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class Generator(nn.Module):
    """Encoder–decoder: 3×conv (64,128,256) + 3×deconv (512) + out conv + tanh."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 64, 4, 2, 1), nn.ELU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ELU(inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ELU(inplace=True))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 512, 4, 2, 1), nn.ELU(inplace=True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.ELU(inplace=True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.ELU(inplace=True))
        self.out = nn.Sequential(nn.Conv2d(512, out_ch, 3, 1, 1), nn.Tanh())

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        # Optional noise concatenated as extra channels (paper: G(p,z)); we add as residual channel noise
        if z is not None:
            x = x + 0.05 * z
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        return self.out(h)


class Discriminator(nn.Module):
    """4×conv (64,128,256,512) → scalar real/fake logit."""

    def __init__(self, in_ch: int = 6):
        super().__init__()
        # Conditional: concat LR input + HR (real or fake)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, lr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([lr, hr], dim=1)).squeeze(1)


class IdentityClassifier(nn.Module):
    """ResNet18 ID head (paper uses ResNet50 — deviation D3)."""

    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inputs in [-1,1] (tanh); map to ImageNet-ish
        x = (x + 1.0) * 0.5
        h = torch.flatten(self.features(x), 1)
        return self.fc(h)
