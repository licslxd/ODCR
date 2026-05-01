"""
Step5B：Flan-T5-XL 权重主链 + soft prompt 条件化（CCV）。

- 仅解释路径使用本模块构建的 ``inputs_embeds``；与 ODCR 前缀 Transformer 解耦。
- 离线 ``local_files_only`` 加载；禁止 Hub 隐式下载。
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from paths_config import require_step5_text_model_dir


def load_t5_conditional_for_odcr(*, torch_dtype: torch.dtype | None = None) -> nn.Module:
    from transformers import T5ForConditionalGeneration

    path = require_step5_text_model_dir()
    kwargs = {"local_files_only": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return T5ForConditionalGeneration.from_pretrained(path, **kwargs)


def discover_flan_explainer_lora_targets(model: nn.Module, *, parent: str = "flan_explainer") -> List[str]:
    """枚举 Flan 子树内 ``nn.Linear``（排除 ``lm_head`` 与 tied 共享层）供 native LoRA 注入。"""
    out: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not name.startswith(parent + "."):
            continue
        if ".lm_head." in name or name.endswith(".lm_head"):
            continue
        if ".shared." in name:
            continue
        out.append(name)
    return out


def flan_teacher_forcing_loss(
    t5: nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (mean_loss, logits)；**不**传 HF ``labels``，CE 仅由外层 ``per_sample_decoder_ce_from_logits`` 聚合。
    ``labels`` 中非监督位为 ``pad_token_id``（默认 0）；logits: (B, dec_T, V)。
    """
    out = t5(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        return_dict=True,
    )
    logits = out.logits
    ps = per_sample_decoder_ce_from_logits(
        logits.float(),
        labels,
        ignore_index=int(pad_token_id),
        label_smoothing=0.0,
    )
    return ps.mean(), logits


def per_sample_decoder_ce_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """(B,T,V) 与 labels (B,T) → 每样本对非 ignore 位点的平均 CE。"""
    b, t, v = logits.shape
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, v),
        labels.reshape(-1),
        ignore_index=int(ignore_index),
        label_smoothing=float(label_smoothing),
        reduction="none",
    ).view(b, t)
    mask = (labels != int(ignore_index)).to(dtype=ce.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (ce * mask).sum(dim=1) / denom


__all__ = [
    "discover_flan_explainer_lora_targets",
    "flan_teacher_forcing_loss",
    "load_t5_conditional_for_odcr",
    "per_sample_decoder_ce_from_logits",
]
