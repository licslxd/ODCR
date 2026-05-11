from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def graph_tied_zero(ref: torch.Tensor) -> torch.Tensor:
    """Return a scalar zero that remains attached to ``ref``'s autograd graph."""
    return ref.sum() * 0.0


@dataclass(frozen=True)
class OrthogonalLosses:
    """shared/specific 正交约束的可审计标量 bundle（张量保持计算图）。"""

    loss_ortho_xcov: torch.Tensor
    loss_ortho_cos: torch.Tensor
    loss_ortho_total: torch.Tensor


@dataclass(frozen=True)
class VarianceFloorLosses:
    """shared/specific 防塌缩方差下界约束与诊断统计。"""

    loss_shared_var: torch.Tensor
    loss_specific_var: torch.Tensor
    loss_var_total: torch.Tensor
    shared_std_mean: torch.Tensor
    specific_std_mean: torch.Tensor
    shared_std_min: torch.Tensor
    specific_std_min: torch.Tensor


def build_orthogonal_losses(
    shared_latent: torch.Tensor,
    specific_latent: torch.Tensor,
    eps: float = 1e-8,
    *,
    w_xcov: float = 1.0,
    w_cos: float = 0.25,
) -> OrthogonalLosses:
    """
    Batch-level normalized cross-correlation Frobenius 惩罚 + instance-level cosine 去相关。

    - loss_ortho_xcov: 保留历史字段名；实际为 batch-centered 且按维标准化后的
      cross-correlation 矩阵 C = (S_z^T P_z) / B，返回 dim-aware 的 ||C||_F^2 / H^2。
    - loss_ortho_cos: mean_b ( cos(S_b, P_b)^2 )，cos 为最后一维余弦相似度。
    - loss_ortho_total: w_xcov * loss_ortho_xcov + w_cos * loss_ortho_cos（默认 1, 0.25）。

    这样既保留 batch-centered shared/specific 去耦语义，也避免 raw covariance 随 hidden size
    和激活尺度爆炸。bf16/fp16 下在 float32 中累计，最后 cast 回输入 dtype。
    """
    if shared_latent.ndim != 2 or specific_latent.ndim != 2:
        raise ValueError("build_orthogonal_losses 期望 shared_latent/specific_latent 形状 [B, H]")
    if shared_latent.shape != specific_latent.shape:
        raise ValueError("shared_latent 与 specific_latent 形状必须一致")
    dev, dt = shared_latent.device, shared_latent.dtype
    b = int(shared_latent.shape[0])
    if b == 0 or shared_latent.numel() == 0:
        z = shared_latent.sum() * 0.0
        return OrthogonalLosses(loss_ortho_xcov=z, loss_ortho_cos=z, loss_ortho_total=z)

    acc_dtype = torch.float32 if dt in (torch.bfloat16, torch.float16) else torch.float64
    s = shared_latent.to(dtype=acc_dtype)
    p = specific_latent.to(dtype=acc_dtype)
    s_c = s - s.mean(dim=0, keepdim=True)
    p_c = p - p.mean(dim=0, keepdim=True)
    s_std = torch.sqrt(s_c.pow(2).mean(dim=0, keepdim=True) + float(eps))
    p_std = torch.sqrt(p_c.pow(2).mean(dim=0, keepdim=True) + float(eps))
    s_z = s_c / s_std
    p_z = p_c / p_std
    c = (s_z.transpose(0, 1) @ p_z) / float(b)
    loss_xcov_acc = c.pow(2).mean()
    cos = F.cosine_similarity(shared_latent, specific_latent, dim=-1, eps=eps)
    loss_cos = cos.pow(2).mean()
    total_acc = float(w_xcov) * loss_xcov_acc.float() + float(w_cos) * loss_cos.float()
    return OrthogonalLosses(
        loss_ortho_xcov=loss_xcov_acc.to(dtype=dt),
        loss_ortho_cos=loss_cos.to(dtype=dt),
        loss_ortho_total=total_acc.to(dtype=dt),
    )


def orthogonal_regularizer(shared_proj: torch.Tensor, specific_proj: torch.Tensor) -> torch.Tensor:
    shared_n = F.normalize(shared_proj, dim=-1)
    specific_n = F.normalize(specific_proj, dim=-1)
    return (shared_n * specific_n).sum(dim=-1).pow(2).mean()


def shared_invariance_loss(shared_proj: torch.Tensor, domain_idx: torch.Tensor) -> torch.Tensor:
    aux = shared_proj[domain_idx == 0]
    tgt = shared_proj[domain_idx == 1]
    if aux.numel() == 0 or tgt.numel() == 0:
        return graph_tied_zero(shared_proj)
    aux_mean = F.normalize(aux.mean(dim=0), dim=0, eps=1e-8)
    tgt_mean = F.normalize(tgt.mean(dim=0), dim=0, eps=1e-8)
    return (aux_mean - tgt_mean).pow(2).mean()


def specific_separation_loss(specific_proj: torch.Tensor, domain_idx: torch.Tensor, margin: float = 0.6) -> torch.Tensor:
    aux = specific_proj[domain_idx == 0]
    tgt = specific_proj[domain_idx == 1]
    if aux.numel() == 0 or tgt.numel() == 0:
        return graph_tied_zero(specific_proj)
    aux_mean = F.normalize(aux.mean(dim=0), dim=0, eps=1e-8)
    tgt_mean = F.normalize(tgt.mean(dim=0), dim=0, eps=1e-8)
    dist = torch.norm(aux_mean - tgt_mean, p=2)
    return torch.relu(dist.new_tensor(float(margin)) - dist)


def variance_floor_loss(
    shared_latent: torch.Tensor,
    specific_latent: torch.Tensor,
    *,
    target_std: float = 0.7,
    eps: float = 1e-4,
) -> VarianceFloorLosses:
    """
    VICReg 风格 variance hinge：按 batch 居中后要求每维 std >= target_std。

    对 shared/specific 分开统计并返回可审计诊断值，避免 latent 退化成低方差向量。
    """
    if shared_latent.ndim != 2 or specific_latent.ndim != 2:
        raise ValueError("variance_floor_loss 期望 shared_latent/specific_latent 形状 [B, H]")
    if shared_latent.shape != specific_latent.shape:
        raise ValueError("variance_floor_loss 要求 shared/specific 形状一致")
    if shared_latent.shape[0] < 2:
        z = shared_latent.sum() * 0.0
        return VarianceFloorLosses(
            loss_shared_var=z,
            loss_specific_var=z,
            loss_var_total=z,
            shared_std_mean=z,
            specific_std_mean=z,
            shared_std_min=z,
            specific_std_min=z,
        )

    target = float(target_std)
    acc_dtype = torch.float32 if shared_latent.dtype in (torch.bfloat16, torch.float16) else torch.float64

    def _branch(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = latent.to(dtype=acc_dtype) - latent.to(dtype=acc_dtype).mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.pow(2).mean(dim=0) + float(eps))
        hinge = torch.relu(std.new_tensor(target) - std)
        return hinge.mean().to(dtype=latent.dtype), std.mean().to(dtype=latent.dtype), std.min().to(dtype=latent.dtype)

    l_shared, shared_std_mean, shared_std_min = _branch(shared_latent)
    l_specific, specific_std_mean, specific_std_min = _branch(specific_latent)
    return VarianceFloorLosses(
        loss_shared_var=l_shared,
        loss_specific_var=l_specific,
        loss_var_total=l_shared + l_specific,
        shared_std_mean=shared_std_mean,
        specific_std_mean=specific_std_mean,
        shared_std_min=shared_std_min,
        specific_std_min=specific_std_min,
    )


def _sample_weight_vector(
    ref: torch.Tensor,
    sample_weight: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    if sample_weight is None:
        return None
    w = sample_weight.reshape(-1).to(device=ref.device, dtype=ref.dtype)
    return w.clamp_min(float(eps))


def anchor_score_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    *,
    sample_weight_eps: float = 1e-6,
) -> torch.Tensor:
    """标量锚预测与 CSV 锚分（0~1）对齐；pred/target 形状 (B,) 或 (B,1)。"""
    p = pred.reshape(-1)
    t = target.reshape(-1).to(dtype=p.dtype)
    if sample_weight is None:
        return F.mse_loss(p, t, reduction="mean")
    w = _sample_weight_vector(p, sample_weight, eps=sample_weight_eps)
    return (((p - t) ** 2) * w).sum() / w.sum()


def residual_l2_penalty(residual_vec: torch.Tensor) -> torch.Tensor:
    # 用 mean 而不是 sum 做 dim-aware 标度，避免 hidden_size 放大 residual loss。
    return residual_vec.pow(2).mean(dim=-1).mean()


def cosine_pull_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
    sample_weight_eps: float = 1e-6,
) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError("cosine_pull_loss 需要 source/target 形状一致")
    loss = 1.0 - F.cosine_similarity(source, target, dim=-1, eps=eps)
    if sample_weight is None:
        return loss.mean()
    w = _sample_weight_vector(loss, sample_weight, eps=sample_weight_eps)
    return (loss * w).sum() / w.sum()


def shared_prototype_pull_loss(
    shared_latent: torch.Tensor,
    shared_prototype: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-8,
    sample_weight_eps: float = 1e-6,
) -> torch.Tensor:
    return cosine_pull_loss(
        shared_latent,
        shared_prototype,
        sample_weight=sample_weight,
        eps=eps,
        sample_weight_eps=sample_weight_eps,
    )


def domain_style_prototype_separation(domain_prototypes: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """domain 风格原型（num_domains, D）非对角 Gram 惩罚（轻量分离）。"""
    if domain_prototypes.shape[0] < 2:
        return graph_tied_zero(domain_prototypes)
    p = F.normalize(domain_prototypes, dim=-1, eps=float(eps))
    g = p @ p.T
    eye = torch.eye(p.shape[0], device=p.device, dtype=p.dtype)
    off = g * (1.0 - eye)
    return off.pow(2).mean()


def lci_score_consistency_loss(score: torch.Tensor, score_perturbed: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(score, score_perturbed, reduction="mean")
