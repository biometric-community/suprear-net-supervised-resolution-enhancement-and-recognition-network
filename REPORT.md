# REPORT — SUPREAR-NET

Lab twin of Ghosh et al., *SUPREAR-NET*, IEEE TBIOM 2022. CelebA substitute for MultiPIE/FaceSurv/MEDS (see DEVIATIONS).

## Train

- CelebA train: n=20000, classes=5765
- SURE-GAN wall_s=n/a (resumed checkpoint)
- HQL wall_s=448.26719975471497

## Cross-resolution Rank-1 (ours)

| Protocol | Rank-1 % |
|----------|--------:|
| LR → embedder | 10.29 |
| Bicubic LR → embedder | 10.29 |
| SURE-GAN + HQL | 0.34 |

Gallery IDs=300, probes=593.

## Figures

![Rank-1](outputs/figures/full/rank1.svg)
![SURE-GAN loss](outputs/figures/full/gan_loss.svg)

## Regenerate

```bash
bash scripts/report.sh
```
