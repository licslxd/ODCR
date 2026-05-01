# -*- coding: utf-8 -*-
"""
训练入口的 CLI 覆盖项：仅在内存中从 argparse 传入 ``build_resolved_training_config``，不写回 os.environ。
"""

from __future__ import annotations

from typing import Any, Dict


def collect_training_hardware_overrides_from_args(args: Any) -> Dict[str, Any]:
    """
    从训练 CLI namespace 收集「非 None」字段，键名稳定，便于日志与统一 resolve。

    注意：不包含 torchrun 的 LOCAL_RANK 等；仅训练超参/调度相关。
    """
    out: Dict[str, Any] = {}
    pairs = [
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("coef", "coef"),
        ("num_proc", "num_proc"),
        ("per_device_batch_size", "per_device_batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("scheduler_initial_lr", "scheduler_initial_lr"),
        ("learning_rate", "learning_rate"),
        ("warmup_steps", "warmup_steps"),
        ("warmup_ratio", "warmup_ratio"),
        ("warmup_epochs", "warmup_epochs"),
        ("min_lr_ratio", "min_lr_ratio"),
        ("lr_scheduler", "lr_scheduler"),
        ("eval_batch_size", "eval_batch_size"),
        ("quick_eval_max_samples", "quick_eval_max_samples"),
        ("early_stop_patience_full", "early_stop_patience_full"),
        ("early_stop_patience_loss", "early_stop_patience_loss"),
        ("min_epochs", "min_epochs"),
        ("early_stop_patience", "early_stop_patience"),
        ("bleu4_max_samples", "bleu4_max_samples"),
    ]
    for key, attr in pairs:
        if not hasattr(args, attr):
            continue
        v = getattr(args, attr)
        if v is not None:
            out[key] = v
    return out
