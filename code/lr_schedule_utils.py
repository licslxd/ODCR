# -*- coding: utf-8 -*-
"""warmup + cosine 学习率倍率（供 LambdaLR 使用，与 DDP 逐步 step 兼容）。"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple


def resolve_warmup_steps(
    total_steps: int,
    n_steps_per_epoch: int,
    *,
    explicit_steps: Optional[int],
    explicit_ratio: Optional[float],
    warmup_epochs_fallback: float,
) -> Tuple[int, float]:
    """
    返回 (warmup_steps, warmup_ratio)，其中 warmup_ratio = warmup_steps / total_steps（用于日志）。

    total_steps、warmup_steps 均按 **optimizer step**（梯度累积后的全局优化步）口径，与训练循环一致。

    优先级：显式 steps > 显式 ratio > warmup_epochs_fallback（按每 epoch 的 optimizer 步数换算）>
    默认 warmup 比例 0.05 × total_steps。
    """
    ts = max(1, int(total_steps))
    ne = max(1, int(n_steps_per_epoch))
    if explicit_steps is not None:
        ws = max(1, min(int(explicit_steps), ts))
        return ws, ws / float(ts)
    if explicit_ratio is not None:
        wr = max(0.0, min(1.0, float(explicit_ratio)))
        ws = max(1, min(int(wr * ts), ts))
        return ws, wr
    if warmup_epochs_fallback > 0:
        ws = max(1, min(int(warmup_epochs_fallback * ne), ts))
        return ws, ws / float(ts)
    wr = 0.05
    ws = max(1, min(int(wr * ts), ts))
    return ws, wr


def warmup_cosine_multiplier_lambda(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """
    LambdaLR 倍率 m(step)：warmup 内线性 0→1；之后 cosine 从 1 衰减至 min_lr_ratio。
    有效 lr = optimizer 初始 base_lr * m(step)。

    与训练循环的约定：每完成一次 optimizer.step() 后调用一次 sched.step()，则 LambdaLR
    内部的 last_epoch（传入本函数的 step）与「全局优化步」从 0 递增一致；total_steps 应设为
    epochs × 每 epoch 优化步数（与梯度累积后的步数一致，而非 micro-batch 数）。
    """
    ws = max(1, int(warmup_steps))
    ts = max(ws + 1, int(total_steps))
    m_ratio = max(0.0, float(min_lr_ratio))
    denom = max(1, ts - ws)

    def lr_lambda(step: int) -> float:
        if step < ws:
            return float(step + 1) / float(ws)
        p = min(1.0, (float(step) - float(ws)) / float(denom))
        cos = 0.5 * (1.0 + math.cos(math.pi * p))
        return m_ratio + (1.0 - m_ratio) * cos

    return lr_lambda
