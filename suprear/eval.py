"""Cross-resolution ID: LR / bicubic / SURE-GAN+HQL Rank-1 on CelebA test IDs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from suprear.data.celeba import resolve_celeba_root
from suprear.data.celeba_lrhr import CelebALRHR
from suprear.models.embedder import FaceEmbedder
from suprear.models.sure_gan import Generator


@torch.no_grad()
def evaluate(cfg: dict) -> dict:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda")
        else "cpu"
    )
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    test_ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 112)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="test",
    )
    max_ids = cfg["eval"].get("max_ids")
    if max_ids is not None:
        keep = set(sorted(test_ds.by_class.keys())[: int(max_ids)])
        keep_idx = [i for i, (_, y) in enumerate(test_ds.samples) if y in keep]
        test_ds = Subset(test_ds, keep_idx)
        # rebuild by_class on subset via one pass of labels later
    else:
        keep = None

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    G = Generator().to(device)
    G.load_state_dict(
        torch.load(ckpt_dir / "sure_gan.pt", map_location=device, weights_only=False)["G"]
    )
    G.eval()
    emb = FaceEmbedder(emb_dim=int(cfg["model"].get("emb_dim", 128)), pretrained=False).to(device)
    emb.load_state_dict(
        torch.load(ckpt_dir / "hql_embedder.pt", map_location=device, weights_only=False)["emb"]
    )
    emb.eval()

    nw = int(cfg["train"].get("num_workers", 4))
    bs = int(cfg["eval"].get("batch_size", 64))
    loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
    )

    gallery = {}
    probes = defaultdict(list)
    for lr, hr, y in tqdm(loader, desc="index"):
        y_np = y.numpy()
        hr = hr.to(device, non_blocking=True)
        # embed HR for gallery (first image per id)
        need = [i for i, yi in enumerate(y_np) if int(yi) not in gallery]
        if need:
            feats = emb(hr[need]).cpu().numpy()
            for j, i in enumerate(need):
                gallery[int(y_np[i])] = feats[j]
        for i in range(lr.size(0)):
            probes[int(y_np[i])].append(lr[i])

    g_labels = np.array(sorted(gallery.keys()))
    g_mat = np.stack([gallery[k] for k in g_labels], axis=0)

    def rank1(probe_embs, labels):
        correct = 0
        for e, y in zip(probe_embs, labels):
            sims = g_mat @ e
            pred = g_labels[int(np.argmax(sims))]
            if pred == y:
                correct += 1
        return correct / max(len(labels), 1)

    probes_per = int(cfg["eval"].get("probes_per_id", 2))
    labels, lr_embs, bic_embs, sr_embs = [], [], [], []
    # batch probe embeds
    lr_list, y_list = [], []
    for y, lrs in probes.items():
        for lr in lrs[:probes_per]:
            lr_list.append(lr)
            y_list.append(y)
    for start in tqdm(range(0, len(lr_list), bs), desc="probes"):
        batch = torch.stack(lr_list[start : start + bs]).to(device)
        ys = y_list[start : start + bs]
        lr_e = emb(batch).cpu().numpy()
        z = torch.randn_like(batch)
        sr_e = emb(G(batch, z)).cpu().numpy()
        for i, y in enumerate(ys):
            lr_embs.append(lr_e[i])
            bic_embs.append(lr_e[i])  # bicubic-upsampled LR already in loader
            sr_embs.append(sr_e[i])
            labels.append(y)

    labels = np.asarray(labels)
    results = {
        "n_gallery": len(g_labels),
        "n_probes": len(labels),
        "rank1_lr_direct": rank1(lr_embs, labels),
        "rank1_bicubic_lr": rank1(bic_embs, labels),
        "rank1_suregan_hql": rank1(sr_embs, labels),
    }
    print(json.dumps(results, indent=2))
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    evaluate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
