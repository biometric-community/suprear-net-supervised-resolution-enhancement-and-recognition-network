"""REPORT.md + SVG figures for SUPREAR-NET."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from suprear.plot_style import BAR_COLOR, FIGSIZE, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    eval_path = log_dir / "eval_results.json"
    train_path = log_dir / "train_summary.json"
    if not eval_path.is_file():
        Path("REPORT.md").write_text("# REPORT\n\n**BLOCKED:** run eval first.\n")
        return
    metrics = json.loads(eval_path.read_text())
    train = json.loads(train_path.read_text()) if train_path.is_file() else {}

    names = ["LR→emb", "Bicubic LR→emb", "SURE-GAN+HQL"]
    vals = [
        metrics["rank1_lr_direct"] * 100,
        metrics["rank1_bicubic_lr"] * 100,
        metrics["rank1_suregan_hql"] * 100,
    ]
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(names, vals, color=[MULTI_SERIES_COLORS[0], MULTI_SERIES_COLORS[1], BAR_COLOR], edgecolor="#002080")
    ax.set_ylabel("Rank-1 (%)")
    ax.set_title("Cross-resolution Rank-1 on CelebA test (ours)")
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "rank1.svg", format="svg")
    plt.close(fig)

    # GAN loss curve if available
    gan_hist = train.get("sure_gan", {}).get("history", [])
    if gan_hist:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot([h["epoch"] for h in gan_hist], [h["loss_g"] for h in gan_hist], color=MULTI_SERIES_COLORS[0], linewidth=3, label="G")
        ax.plot([h["epoch"] for h in gan_hist], [h["loss_d"] for h in gan_hist], color=MULTI_SERIES_COLORS[1], linewidth=3, label="D")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("SURE-GAN training (ours)")
        ax.legend()
        apply_style(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / "gan_loss.svg", format="svg")
        plt.close(fig)

    gan = train.get("sure_gan", {}) or {}
    hql = train.get("hql", {}) or {}
    gan_note = " (resumed checkpoint)" if gan.get("resumed") else ""
    gan_s = gan.get("seconds")
    gan_s_s = "n/a" if gan_s in (None, 0, 0.0) and gan.get("resumed") else str(gan_s)
    figs = [f"![Rank-1]({(fig_dir / 'rank1.svg').as_posix()})"]
    if (fig_dir / "gan_loss.svg").is_file():
        figs.append(f"![SURE-GAN loss]({(fig_dir / 'gan_loss.svg').as_posix()})")
    report = f"""# REPORT — SUPREAR-NET

Lab twin of Ghosh et al., *SUPREAR-NET*, IEEE TBIOM 2022. CelebA substitute for MultiPIE/FaceSurv/MEDS (see DEVIATIONS).

## Train

- CelebA train: n={train.get('n_train')}, classes={train.get('num_classes')}
- SURE-GAN wall_s={gan_s_s}{gan_note}
- HQL wall_s={hql.get('seconds')}

## Cross-resolution Rank-1 (ours)

| Protocol | Rank-1 % |
|----------|--------:|
| LR → embedder | {metrics['rank1_lr_direct']*100:.2f} |
| Bicubic LR → embedder | {metrics['rank1_bicubic_lr']*100:.2f} |
| SURE-GAN + HQL | {metrics['rank1_suregan_hql']*100:.2f} |

Gallery IDs={metrics['n_gallery']}, probes={metrics['n_probes']}.

## Figures

{chr(10).join(figs)}

## Regenerate

```bash
bash scripts/report.sh
```
"""
    Path("REPORT.md").write_text(report)
    print(f"Wrote REPORT.md + {fig_dir}")


if __name__ == "__main__":
    main()
