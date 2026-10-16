"""Train SURE-GAN then HQL embedder (SUPREAR-NET two-stage)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from suprear.data.celeba import resolve_celeba_root
from suprear.data.celeba_lrhr import CelebALRHR
from suprear.losses import hql_loss
from suprear.models.embedder import FaceEmbedder
from suprear.models.sure_gan import Discriminator, Generator, IdentityClassifier


def _device(cfg: dict) -> torch.device:
    name = cfg.get("device", "cuda")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train_sure_gan(cfg: dict, ds: CelebALRHR, device: torch.device) -> dict:
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    G = Generator().to(device)
    D = Discriminator().to(device)
    C = IdentityClassifier(ds.num_classes, pretrained=True).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=float(cfg["train"].get("lr_g", 1e-3)), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=float(cfg["train"].get("lr_d", 1e-3)), betas=(0.5, 0.999))
    opt_c = torch.optim.SGD(C.parameters(), lr=float(cfg["train"].get("lr_c", 1e-2)), momentum=0.9)

    # Pretrain classifier on HR (Alg. 1 n1)
    n1 = int(cfg["train"].get("epochs_c_pretrain", 1))
    for ep in range(1, n1 + 1):
        C.train()
        losses = []
        for _, hr, y in tqdm(loader, desc=f"C-pre ep{ep}/{n1}", leave=False):
            hr, y = hr.to(device), y.to(device)
            logits = C(hr)
            loss = F.cross_entropy(logits, y)
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            opt_c.step()
            losses.append(float(loss.detach()))
        print(f"[C-pre] epoch={ep} loss={np.mean(losses):.4f}")

    lam1 = float(cfg["train"].get("lambda_adv", 1.0))
    lam2 = float(cfg["train"].get("lambda_cls", 1.0))
    lam3 = float(cfg["train"].get("lambda_l1", 10.0))
    n2 = int(cfg["train"].get("epochs_gan", 5))
    hist = []
    t0 = time.time()
    bce = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, n2 + 1):
        G.train()
        D.train()
        C.train()
        g_losses, d_losses, c_losses = [], [], []
        for lr, hr, y in tqdm(loader, desc=f"GAN ep{ep}/{n2}", leave=False):
            lr, hr, y = lr.to(device), hr.to(device), y.to(device)
            bsz = lr.size(0)
            z = torch.randn_like(lr)
            fake = G(lr, z)

            # --- D ---
            opt_d.zero_grad(set_to_none=True)
            real_logit = D(lr, hr)
            fake_logit = D(lr, fake.detach())
            loss_d = bce(real_logit, torch.ones(bsz, device=device)) + bce(
                fake_logit, torch.zeros(bsz, device=device)
            )
            loss_d.backward()
            opt_d.step()

            # --- C (real + fake) ---
            opt_c.zero_grad(set_to_none=True)
            loss_c = F.cross_entropy(C(hr), y) + F.cross_entropy(C(fake.detach()), y)
            loss_c.backward()
            opt_c.step()

            # --- G ---
            opt_g.zero_grad(set_to_none=True)
            fake = G(lr, z)
            adv = bce(D(lr, fake), torch.ones(bsz, device=device))
            cls = F.cross_entropy(C(fake), y)
            l1 = F.l1_loss(fake, hr)
            loss_g = lam1 * adv + lam2 * cls + lam3 * l1
            loss_g.backward()
            opt_g.step()

            g_losses.append(float(loss_g.detach()))
            d_losses.append(float(loss_d.detach()))
            c_losses.append(float(loss_c.detach()))

        row = {
            "epoch": ep,
            "loss_g": float(np.mean(g_losses)),
            "loss_d": float(np.mean(d_losses)),
            "loss_c": float(np.mean(c_losses)),
        }
        hist.append(row)
        print(f"[GAN] epoch={ep} G={row['loss_g']:.4f} D={row['loss_d']:.4f} C={row['loss_c']:.4f}")

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "C": C.state_dict(), "num_classes": ds.num_classes}, ckpt_dir / "sure_gan.pt")
    return {"history": hist, "seconds": time.time() - t0, "checkpoint": str(ckpt_dir / "sure_gan.pt")}


def train_hql(cfg: dict, ds: CelebALRHR, device: torch.device, G: Generator) -> dict:
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)

    emb = FaceEmbedder(emb_dim=int(cfg["model"].get("emb_dim", 128)), pretrained=True).to(device)
    opt = torch.optim.Adam(emb.parameters(), lr=float(cfg["train"].get("lr_hql", 1e-4)))
    epochs = int(cfg["train"].get("epochs_hql", 5))
    alpha1 = float(cfg["train"].get("alpha1", 0.5))
    alpha2 = float(cfg["train"].get("alpha2", 0.5))
    batch_size = int(cfg["train"].get("batch_size_hql", 32))
    # Cap steps: full epoch over all images is unnecessary for HQL with random tuples
    steps = int(cfg["train"].get("hql_steps_per_epoch", max(50, min(400, len(ds) // batch_size))))
    hist = []
    t0 = time.time()
    rng = np.random.RandomState(int(cfg["train"].get("seed", 42)))

    # Cap + parallel cache: full 20k serial decode is ~1h; HQL only needs identity tuples.
    cache_n = int(cfg["train"].get("hql_cache_images", min(len(ds), 4096)))
    cache_idx = rng.choice(len(ds), size=cache_n, replace=False)
    cache_idx.sort()

    class _SubDS(torch.utils.data.Dataset):
        def __len__(self):
            return cache_n

        def __getitem__(self, j):
            i = int(cache_idx[j])
            lr_i, hr_i, y_i = ds[i]
            return lr_i, hr_i, int(y_i)

    nw = int(cfg["train"].get("num_workers", 4))
    print(f"[HQL] caching {cache_n}/{len(ds)} LR/HR via DataLoader(workers={nw})…", flush=True)
    lr_cache = torch.empty(cache_n, 3, ds.image_size, ds.image_size)
    hr_cache = torch.empty(cache_n, 3, ds.image_size, ds.image_size)
    y_cache = np.empty(cache_n, dtype=np.int64)
    loader = DataLoader(_SubDS(), batch_size=64, shuffle=False, num_workers=nw, pin_memory=False)
    pos = 0
    for lr_b, hr_b, y_b in tqdm(loader, desc="HQL cache", leave=False):
        n = lr_b.size(0)
        lr_cache[pos : pos + n] = lr_b
        hr_cache[pos : pos + n] = hr_b
        y_cache[pos : pos + n] = y_b.numpy()
        pos += n
    by_class_idx: dict[int, list[int]] = {}
    for j, y in enumerate(y_cache):
        by_class_idx.setdefault(int(y), []).append(j)
    by_class_arr = {k: np.asarray(v, dtype=np.int64) for k, v in by_class_idx.items()}
    classes = np.asarray(list(by_class_arr.keys()))
    print(f"[HQL] cache ready classes={len(classes)}", flush=True)

    def pick_batch(cids: np.ndarray):
        idxs = np.empty(len(cids), dtype=np.int64)
        for i, cid in enumerate(cids):
            pool = by_class_arr[int(cid)]
            idxs[i] = int(pool[rng.randint(0, len(pool))])
        return lr_cache[idxs], hr_cache[idxs]

    for ep in range(1, epochs + 1):
        emb.train()
        losses = []
        for _ in tqdm(range(steps), desc=f"HQL ep{ep}/{epochs}", leave=False):
            a = rng.choice(classes, size=batch_size, replace=True)
            b = rng.choice(classes, size=batch_size, replace=True)
            c = rng.choice(classes, size=batch_size, replace=True)
            b = np.where(b == a, rng.choice(classes, size=batch_size, replace=True), b)
            c = np.where((c == a) | (c == b), rng.choice(classes, size=batch_size, replace=True), c)
            pa, qa = pick_batch(a)
            pb, _ = pick_batch(b)
            pc, _ = pick_batch(c)
            qa = qa.to(device, non_blocking=True)
            lr_all = torch.cat([pa, pb, pc], dim=0).to(device, non_blocking=True)
            with torch.no_grad():
                z = torch.randn_like(lr_all)
                sr_all = G(lr_all, z)
                pa_sr, pb_sr, pc_sr = sr_all.split(batch_size, dim=0)
            feats = emb(torch.cat([qa, pa_sr, pb_sr, pc_sr], dim=0))
            fq, fpa, fpb, fpc = feats.split(batch_size, dim=0)
            loss = hql_loss(fq, fpa, fpb, fpc, alpha1, alpha2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        mean = float(np.mean(losses))
        hist.append({"epoch": ep, "loss": mean, "steps": steps})
        print(f"[HQL] epoch={ep} loss={mean:.4f} steps={steps}", flush=True)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    torch.save({"emb": emb.state_dict(), "emb_dim": emb.fc.out_features}, ckpt_dir / "hql_embedder.pt")
    return {"history": hist, "seconds": time.time() - t0, "checkpoint": str(ckpt_dir / "hql_embedder.pt")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--stage",
        choices=("all", "gan", "hql"),
        default="all",
        help="all=GAN+HQL; gan=SURE-GAN only; hql=HQL from existing sure_gan.pt",
    )
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = _device(cfg)
    root = resolve_celeba_root(cfg)
    max_n = cfg["data"].get("max_images")
    ds = CelebALRHR(
        root,
        image_size=int(cfg["model"].get("image_size", 112)),
        lr_scale=int(cfg["model"].get("lr_scale", 4)),
        max_n=int(max_n) if max_n is not None else None,
        seed=int(cfg["train"].get("seed", 42)),
        split="train",
    )
    print(f"[data] train n={len(ds)} classes={ds.num_classes} device={device} stage={args.stage}")
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    gan_sum = None
    if args.stage in ("all", "gan"):
        gan_sum = train_sure_gan(cfg, ds, device)
    else:
        gan_path = ckpt_dir / "sure_gan.pt"
        if not gan_path.is_file():
            raise FileNotFoundError(f"need {gan_path} for --stage hql")
        gan_sum = {"history": [], "seconds": 0.0, "checkpoint": str(gan_path), "resumed": True}
        print(f"[GAN] resumed from {gan_path}")

    hql_sum = None
    if args.stage in ("all", "hql"):
        G = Generator().to(device)
        G.load_state_dict(
            torch.load(ckpt_dir / "sure_gan.pt", map_location=device, weights_only=False)["G"]
        )
        hql_sum = train_hql(cfg, ds, device, G)
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "protocol": cfg["train"].get("protocol", "full"),
        "n_train": len(ds),
        "num_classes": ds.num_classes,
        "sure_gan": gan_sum,
        "hql": hql_sum,
        "device": str(device),
        "stage": args.stage,
    }
    (log_dir / "train_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "n_train": out["n_train"],
                "num_classes": out["num_classes"],
                "gan_s": (gan_sum or {}).get("seconds"),
                "hql_s": (hql_sum or {}).get("seconds"),
                "stage": args.stage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
