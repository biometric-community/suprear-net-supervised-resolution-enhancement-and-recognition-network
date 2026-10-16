"""Heterogeneous quadruplet loss (paper Eq. 15 family)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hql_loss(
    f_qa: torch.Tensor,
    f_pa: torch.Tensor,
    f_pb: torch.Tensor,
    f_pc: torch.Tensor,
    alpha1: float = 0.5,
    alpha2: float = 0.5,
) -> torch.Tensor:
    """HQL on (HR anchor qa, LR pos pa, LR neg pb, LR neg pc).

    L = [||qa-pa||^2 - ||qa-pb||^2 + α1]+ + [||qa-pa||^2 - ||qa-pc||^2 + α2]+
    """
    d_pos = (f_qa - f_pa).pow(2).sum(dim=1)
    d_neg_b = (f_qa - f_pb).pow(2).sum(dim=1)
    d_neg_c = (f_qa - f_pc).pow(2).sum(dim=1)
    t1 = F.relu(d_pos - d_neg_b + alpha1)
    t2 = F.relu(d_pos - d_neg_c + alpha2)
    return (t1 + t2).mean()
