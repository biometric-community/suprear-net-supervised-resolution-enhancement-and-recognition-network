"""Enhance one LR face with trained SURE-GAN and print embedding summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from suprear.data.celeba_lrhr import make_lr
from suprear.models.embedder import FaceEmbedder
from suprear.models.sure_gan import Generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/predict_sr.png")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    hr = tf(Image.open(args.image).convert("RGB"))
    lr = make_lr(hr, int(cfg["model"].get("lr_scale", 4))).unsqueeze(0).to(device)
    G = Generator().to(device)
    G.load_state_dict(torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "sure_gan.pt", map_location=device, weights_only=False)["G"])
    G.eval()
    emb = FaceEmbedder(emb_dim=int(cfg["model"].get("emb_dim", 128)), pretrained=False).to(device)
    emb.load_state_dict(torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "hql_embedder.pt", map_location=device, weights_only=False)["emb"])
    emb.eval()
    with torch.no_grad():
        sr = G(lr, torch.randn_like(lr))
        e = emb(sr).cpu().numpy()[0]
    # save SR image
    out = ((sr[0].cpu().clamp(-1, 1) + 1) * 0.5 * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(out).save(args.out)
    print(json.dumps({"image": args.image, "sr": args.out, "emb_dim": int(e.shape[0]), "emb_head": e[:8].tolist()}, indent=2))


if __name__ == "__main__":
    main()
