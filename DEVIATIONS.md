# Deviations

| ID | Topic | Paper | Ours |
|----|-------|-------|------|
| D1 | Datasets | FaceSurv, MultiPIE, NIST MEDS | CelebA with synthetic LR (×4 bicubic) |
| D2 | Pretrain | SURE-GAN on CASIA NIR-VIS; HQL LightCNN on MS1M | ImageNet-pretrained ResNet backbones only |
| D3 | Classifier | ResNet50 | ResNet18 |
| D4 | Embedder | LightCNN29, 256-D | ResNet18, 128-D |
| D5 | HQL inputs | Raw LR for pa/pb/pc | SR(G(LR)) for positives/negatives at HQL train (matches test path) |
| D6 | Train scale | Full MultiPIE/FaceSurv protocols | CelebA capped at 20k images / 3 GAN+HQL epochs for wall-clock |
| D7 | HQL cache | Online sampling over full set | HQL samples from a 4096-image preload cache (parallel DataLoader) |
| D8 | Eval IDs | Full gallery protocols | CelebA test capped at 300 IDs for wall-clock Rank-1 |

