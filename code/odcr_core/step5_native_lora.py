"""
Step5 文本栈原生 LoRA（不依赖 HuggingFace peft 包，避免与旧版 transformers 的导入链冲突）。

对 ``executors.step5_engine.Model`` 中「解释/语言建模」相关 ``nn.Linear`` 注入低秩旁路；
显式排除 ``recommender``、``odcr_scorer`` 评分支路。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

_TEXT_LORA_PREFIXES: Tuple[str, ...] = (
    "transformer_encoder",
    "hidden2token",
    "domain_cross_attn",
    "domain_gate",
    "odcr_explainer_bridge",
    "flan_explainer",
    "flan_soft_prompt_stack",
    "ccv_numeric_adapter",
    "ccv_control_adapter",
    "fca_score_align",
    "fca_explain_align",
)

_SKIP_PREFIXES: Tuple[str, ...] = (
    "recommender",
    "odcr_scorer",
)


class LoRALinear(nn.Module):
    """在冻结的 ``nn.Linear`` 旁叠加 LoRA：output = base(x) + scaling * B @ A @ dropout(x)（与常见实现等价）。"""

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA r 须为正整数，当前为 {r}")
        self.base = base
        self.r = int(r)
        self.scaling = float(alpha) / float(self.r)
        in_f = int(base.in_features)
        out_f = int(base.out_features)
        self.in_features = in_f
        self.out_features = out_f
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f))
        self.lora_B = nn.Parameter(torch.empty(out_f, self.r))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        """兼容 ``nn.MultiheadAttention`` 等对 ``out_proj.weight`` 的直接读取。"""
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        xd = self.dropout(x)
        return y0 + (xd @ self.lora_A.T @ self.lora_B.T) * self.scaling


def discover_step5_text_linear_targets(model: nn.Module) -> List[str]:
    """按 ``Model`` 实际 ``named_modules`` 枚举可注入的 ``nn.Linear`` 全名（排除评分器）。"""
    out: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not name:
            continue
        if ".lm_head." in name or name.endswith(".lm_head"):
            continue
        if ".shared." in name:
            continue
        if any(name == p or name.startswith(p + ".") for p in _SKIP_PREFIXES):
            continue
        if any(name == p or name.startswith(p + ".") for p in _TEXT_LORA_PREFIXES):
            out.append(name)
    return out


def _parent_and_child(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_native_lora_to_step5_model(
    model: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_modules_override: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """
    将文本栈 ``nn.Linear`` 替换为 :class:`LoRALinear`；失败时抛 ``RuntimeError``（fail-fast）。

    若提供 ``target_modules_override``，则须为 ``named_modules()`` 中出现的 **完整** 子模块名列表，
    且每个名须对应 ``nn.Linear``；否则报错（避免静默跳过）。
    """
    discovered = discover_step5_text_linear_targets(model)
    if target_modules_override is not None:
        want = [str(x).strip() for x in target_modules_override if str(x).strip()]
        if not want:
            raise RuntimeError("LoRA 注入失败：lora_target_modules 覆盖为非空列表，但解析后为空。")
        disc_set = set(discovered)
        missing = [n for n in want if n not in disc_set]
        if missing:
            raise RuntimeError(
                "LoRA 注入失败：lora_target_modules 与当前 Step5 Model 线性层不匹配；以下名在模型中不存在或非 Linear：\n"
                + "\n".join(f"  - {m}" for m in missing[:50])
                + ("\n  ..." if len(missing) > 50 else "")
                + f"\n当前可注入候选（节选前 40）：{discovered[:40]}"
            )
        targets = list(want)
    else:
        targets = list(discovered)
    if not targets:
        raise RuntimeError(
            "LoRA 注入失败：未发现可注入的文本栈 Linear（target_modules 探测为空）；"
            "请检查 Model 结构是否与 ODCR Step5 主线一致。"
        )
    for dotted in targets:
        parent, child = _parent_and_child(model, dotted)
        cur = getattr(parent, child)
        if not isinstance(cur, nn.Linear):
            raise RuntimeError(f"LoRA 注入失败：{dotted!r} 不是 nn.Linear（实际为 {type(cur).__name__}）。")
        setattr(parent, child, LoRALinear(cur, r=int(r), alpha=float(alpha), dropout=float(dropout)))
    return {
        "enabled": True,
        "type": "lora",
        "implementation": "odcr_native_linear",
        "r": int(r),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": list(targets),
    }


__all__ = [
    "LoRALinear",
    "apply_native_lora_to_step5_model",
    "discover_step5_text_linear_targets",
]
