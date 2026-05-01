"""
Step5 词表头损失辅助：repeat 检测（无 B×T×T 物化）、基于已算好的 log_softmax 的 CE/UL。

主训练路径禁止对同一 word_dist 重复做完整 log_softmax；repeat mask 与旧 B×T×T 语义一致。
"""
from __future__ import annotations

import torch


def route_weighted_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """统一 train/valid 风险口径：仅对 route=1 的样本聚合。"""
    if values.numel() == 0:
        return values.sum() * 0.0
    w = torch.nan_to_num(weights.to(device=values.device, dtype=values.dtype), nan=0.0, posinf=0.0, neginf=0.0)
    m = torch.nan_to_num(mask.to(device=values.device, dtype=values.dtype), nan=0.0, posinf=0.0, neginf=0.0)
    values_b, w_b, m_b = torch.broadcast_tensors(values, w, m)
    route_w = (w_b * m_b).clamp_min(0.0)
    denom = route_w.sum().clamp_min(values_b.new_tensor(1e-8))
    return (values_b * route_w).sum() / denom


def odcr_repeat_mask_from_tgt_batched(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    (B,T) int64/int → (B,T) bool：与历史 Step5 ``repeat_full_mask.any(dim=1)`` 语义一致。

    即：非 pad 且 **同一 token 在更晚时间步还会出现** 的位置为 True（重复段中的「较早」位置；
    与「仅第二次及以后为 True」不同，这是原 B×T×T 实现沿 dim=1 归约的精确行为）。

    实现：按 (token_id, position) 排序后，在同 token 连续段内「后面还有同 token」的位置标 True，
    scatter 回时间轴；无 B×T×T 中间张量。
    """
    if tgt.dim() != 2:
        raise ValueError(f"tgt must be (B,T), got shape {tuple(tgt.shape)}")
    B, T = int(tgt.shape[0]), int(tgt.shape[1])
    if T == 0:
        return torch.zeros(B, 0, dtype=torch.bool, device=tgt.device)

    tok = tgt.long()
    # int64，避免大词表乘 (T+1) 溢出
    t_idx = torch.arange(T, device=tgt.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    sort_key = tok * (T + 1) + t_idx
    _sorted, sort_idx = torch.sort(sort_key, dim=-1)
    sorted_tok = _sorted // (T + 1)

    # 与下一项同 token ⇔ 非该 token 在时间上最后一次出现 ⇔ 原 any(dim=1) 标记集
    same_as_next = sorted_tok[:, :-1] == sorted_tok[:, 1:]
    is_repeat_sorted = torch.zeros(B, T, dtype=torch.bool, device=tgt.device)
    is_repeat_sorted[:, :-1] = same_as_next

    out = torch.zeros(B, T, dtype=torch.bool, device=tgt.device)
    out.scatter_(1, sort_idx, is_repeat_sorted)
    return out & (tok != int(pad_id))


def per_sample_mean_ce_from_logp(
    logp_bt: torch.Tensor,
    tgt: torch.Tensor,
    *,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    """
    (B,T,V) log_probs 与 (B,T) target → 每样本对非 padding 位置的平均 CE。

    label_smoothing 语义与 F.cross_entropy(..., label_smoothing=ls) 一致：
    (1-ls)*(-log p_y) + ls * mean_v(-log p_v)；ignore_index 位置贡献为 0。
    """
    B, T, V = logp_bt.shape
    ls = float(label_smoothing)
    tg = tgt.long()
    nll = -logp_bt.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
    if ls > 0.0:
        smooth = -logp_bt.mean(dim=-1)
        ce = (1.0 - ls) * nll + ls * smooth
    else:
        ce = nll
    ce = torch.where(tg == int(ignore_index), torch.zeros_like(ce), ce)
    mask = (tg != int(ignore_index)).to(dtype=ce.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (ce * mask).sum(dim=1) / denom


def odcr_anti_repeat_unlikelihood_loss_from_logp(
    logp: torch.Tensor,
    tgt: torch.Tensor,
    *,
    pad_id: int = 0,
) -> torch.Tensor:
    """Teacher-forcing unlikelihood：对已重复位置压低当前 token 的预测概率（消费 log_softmax 输出）。"""
    dtype = logp.dtype
    repeat_mask = odcr_repeat_mask_from_tgt_batched(tgt, pad_id)
    tg = tgt.long()
    logp_at = logp.gather(dim=-1, index=tg.unsqueeze(-1)).squeeze(-1)
    p = torch.exp(logp_at).clamp(max=1.0 - 1e-6)
    per_pos_loss = -torch.log(1.0 - p)
    valid_loss = per_pos_loss * repeat_mask.to(dtype=per_pos_loss.dtype)
    total = valid_loss.sum()
    count = repeat_mask.sum()
    safe_count = count.clamp(min=1).to(dtype)
    scaled = total / safe_count
    return scaled * (count > 0).to(dtype)
