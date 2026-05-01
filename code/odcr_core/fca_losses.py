"""FCA：scorer 表征与 explainer（Flan encoder pool）显式余弦对齐。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fca_cosine_alignment_loss(
    h_score: torch.Tensor,
    h_explain: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    1 - cosine(h_score, h_explain) 的 batch 均值；可选 ``mask`` (B,) 0/1 仅高置信样本。
    """
    hs = F.normalize(h_score, dim=-1, eps=eps)
    he = F.normalize(h_explain, dim=-1, eps=eps)
    cos = (hs * he).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    one_minus = 1.0 - cos
    if mask is None:
        return one_minus.mean()
    m = mask.to(dtype=one_minus.dtype).clamp(0.0, 1.0)
    denom = m.sum().clamp(min=1.0)
    return (one_minus * m).sum() / denom


__all__ = ["fca_cosine_alignment_loss"]
