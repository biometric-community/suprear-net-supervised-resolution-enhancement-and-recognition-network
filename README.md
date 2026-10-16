# SUPREAR-NET: Supervised Resolution Enhancement and Recognition Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Ghosh, Vatsa, and Singh, *SUPREAR-NET: Supervised Resolution Enhancement and Recognition Network* (IEEE TBIOM 2022). Two-stage pipeline: **SURE-GAN** (LR→HR) then **HQL** (heterogeneous quadruplet) embedder for cross-resolution face recognition.

**Repository:** https://github.com/biometric-community/suprear-net-supervised-resolution-enhancement-and-recognition-network

## Setup

```bash
bash scripts/setup_env.sh
```

Uses the monorepo shared `.venv` when present.

## Data

Paper datasets (MultiPIE / FaceSurv / NIST MEDS) are not redistributed here. This rebuild trains and evaluates on **CelebA** with synthetic ×4 bicubic LR (see `DEVIATIONS.md`).

```bash
bash scripts/check_dataset_size.sh ../../datasets/celeba
```

Point `configs/*.yaml` → `../../datasets/celeba/extracted/celeba`.

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
# or resume HQL only after sure_gan.pt exists:
bash scripts/train.sh configs/default.yaml --stage hql
bash scripts/eval.sh configs/default.yaml
bash scripts/report.sh configs/default.yaml
```

Metrics and figures: [`REPORT.md`](REPORT.md).

## Layout

| Path | Role |
|------|------|
| `suprear/models/` | Generator, Discriminator, IdentityClassifier, FaceEmbedder |
| `suprear/data/` | CelebA LR/HR loader |
| `configs/` | `smoke.yaml`, `default.yaml` |
| `scripts/` | setup / train / eval / predict / report |
| `DEVIATIONS.md` | Paper vs lab twin |

## Citation

```bibtex
@article{ghosh2022suprear,
  title   = {SUPREAR-NET: Supervised Resolution Enhancement and Recognition Network},
  author  = {Ghosh, Soumyadeep and Vatsa, Mayank and Singh, Richa},
  journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year    = {2022}
}
```
