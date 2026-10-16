"""CelebA HR/LR pairs for SURE-GAN + HQL (public MultiPIE/FaceSurv substitute)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from suprear.data.celeba import resolve_celeba_root


def _tf_hr(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1,1]
        ]
    )


def make_lr(hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Bicubic downsample then upsample back to HR size (blurry LR probe)."""
    squeeze = hr.dim() == 3
    x = hr.unsqueeze(0) if squeeze else hr
    h, w = x.shape[-2:]
    small = F.interpolate(
        x, size=(max(1, h // scale), max(1, w // scale)), mode="bicubic", align_corners=False
    )
    lr = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    return lr.squeeze(0) if squeeze else lr


class CelebALRHR(Dataset):
    """Yield (lr, hr, label) with remapped contiguous IDs."""

    def __init__(
        self,
        root: Path,
        image_size: int = 112,
        lr_scale: int = 4,
        max_n: Optional[int] = None,
        seed: int = 42,
        train_ratio: float = 0.8,
        split: str = "train",
    ):
        self.root = Path(root)
        self.img_dir = self.root / "img_align_celeba"
        self.image_size = image_size
        self.lr_scale = lr_scale
        self.tf = _tf_hr(image_size)

        id_map: Dict[str, int] = {}
        with open(self.root / "identity_CelebA.txt") as f:
            for line in f:
                fn, sid = line.split()
                id_map[fn] = int(sid)

        import numpy as np

        subjects = sorted(set(id_map.values()))
        rng = np.random.RandomState(seed)
        rng.shuffle(subjects)
        n_train = int(round(len(subjects) * train_ratio))
        train_subs = set(subjects[:n_train])
        want_train = split == "train"
        rows = [(fn, sid) for fn, sid in id_map.items() if (sid in train_subs) == want_train]
        if max_n is not None:
            rows = rows[: int(max_n)]
        uniq = sorted({r[1] for r in rows})
        remap = {s: i for i, s in enumerate(uniq)}
        self.samples: List[Tuple[str, int]] = [(fn, remap[sid]) for fn, sid in rows]
        self.num_classes = len(uniq)
        # index by class for HQL sampling
        self.by_class: Dict[int, List[int]] = {}
        for i, (_, y) in enumerate(self.samples):
            self.by_class.setdefault(y, []).append(i)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fn, y = self.samples[idx]
        hr = self.tf(Image.open(self.img_dir / fn).convert("RGB"))
        lr = make_lr(hr, self.lr_scale)
        return lr, hr, y
