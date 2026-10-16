# Implementation plan — SUPREAR-NET

Paper: Ghosh, Vatsa, Singh, IEEE TBIOM 2022.

## Method

1. SURE-GAN (G+D+C): LR→HR with adv + CE identity + L1 (Eq. 7–10, Alg. 1)
2. HQL embedder: (qa HR, pa LR/SR, pb/pc LR/SR negatives) (Eq. 15 family)
3. Test: G(LR) → embed → Rank-1 vs HR gallery

## Data

Required: CelebA (size gate). Paper MultiPIE/FaceSurv/MEDS → D1.

## GitHub

`suprear-net-supervised-resolution-enhancement-and-recognition-network`
