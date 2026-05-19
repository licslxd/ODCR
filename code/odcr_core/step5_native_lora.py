"""Head-aware native LoRA injection for Step5.

The active Step5 path is allowlist-only: an empty ``target_modules`` list no
longer means "scan the model".  The resolver should pass the policy sentinel,
or a non-empty explicit subset of the head allowlist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from odcr_core.step5_grad_contract import (
    DELETED_STEP5_LEGACY_MODULES,
    STEP5_LORA_TARGET_POLICY_ID,
    head_specific_lora_allowlist_id,
    normalize_step5_head_for_contract,
)


HEAD_AWARE_LORA_TARGET_SENTINEL = "__HEAD_AWARE_STEP5_DEFAULT__"

_STEP5A_PREFIXES: tuple[str, ...] = (
    "domain_gate",
    "transformer_encoder",
)

_STEP5B_PREFIXES: tuple[str, ...] = (
    "domain_gate",
    "transformer_encoder",
    "ccv_numeric_adapter",
    "ccv_control_adapter",
    "fca_score_align",
    "fca_explain_align",
    "flan_explainer",
)

_COMBINED_PREFIXES: tuple[str, ...] = tuple(dict.fromkeys(_STEP5A_PREFIXES + _STEP5B_PREFIXES))


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA r must be positive, got {r}")
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
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        xd = self.dropout(x)
        return y0 + (xd @ self.lora_A.T @ self.lora_B.T) * self.scaling


def _parent_and_child(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _prefixes_for_head(head: str) -> tuple[str, ...]:
    norm = normalize_step5_head_for_contract(head)
    if norm == "step5A":
        return _STEP5A_PREFIXES
    if norm == "step5B":
        return _STEP5B_PREFIXES
    return _COMBINED_PREFIXES


def _legacy_target_hit(name: str) -> str | None:
    for legacy in DELETED_STEP5_LEGACY_MODULES:
        if name == legacy or name.startswith(legacy + "."):
            return legacy
    return None


def _forbidden_reason(model: nn.Module, dotted: str) -> str | None:
    legacy = _legacy_target_hit(dotted)
    if legacy:
        return f"deleted legacy module {legacy}"
    try:
        parent, child = _parent_and_child(model, dotted)
    except AttributeError:
        return "module path does not exist"
    if isinstance(parent, nn.MultiheadAttention) and child == "out_proj":
        return "nn.MultiheadAttention.out_proj is read functionally by PyTorch and must not be LoRA-wrapped"
    return None


def forbidden_lora_targets_for_model(model: nn.Module) -> list[str]:
    out: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not name:
            continue
        reason = _forbidden_reason(model, name)
        if reason:
            out.append(name)
    return out


def _is_allowed_linear(model: nn.Module, name: str, *, head: str) -> bool:
    if _forbidden_reason(model, name):
        return False
    if ".lm_head." in name or name.endswith(".lm_head"):
        return False
    if ".shared." in name:
        return False
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _prefixes_for_head(head))


def head_aware_step5_lora_targets(model: nn.Module, *, head: str) -> list[str]:
    targets: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or not name:
            continue
        if _is_allowed_linear(model, name, head=head):
            targets.append(name)
    if not targets:
        raise RuntimeError(f"Step5 {head} LoRA allowlist resolved to no Linear targets.")
    return targets


def resolve_step5_lora_targets(
    model: nn.Module,
    *,
    head: str,
    configured_target_modules: Sequence[str] | None,
) -> dict[str, Any]:
    configured = [str(item).strip() for item in (configured_target_modules or ()) if str(item).strip()]
    allowlist = head_aware_step5_lora_targets(model, head=head)
    allowset = set(allowlist)
    if not configured:
        raise RuntimeError(
            "step5.ccv.native_lora.target_modules=[] is retired; use "
            f"[{HEAD_AWARE_LORA_TARGET_SENTINEL!r}] for the head-aware allowlist."
        )
    if configured == [HEAD_AWARE_LORA_TARGET_SENTINEL]:
        final_targets = allowlist
        requested_policy = "head_aware_default_sentinel"
    else:
        forbidden = []
        for target in configured:
            reason = _forbidden_reason(model, target)
            if reason:
                forbidden.append({"target": target, "reason": reason})
            elif target not in allowset:
                forbidden.append({"target": target, "reason": "target is outside the head-aware allowlist"})
        if forbidden:
            detail = "; ".join(f"{item['target']} ({item['reason']})" for item in forbidden[:20])
            raise RuntimeError(f"LoRA target policy violation for Step5 {head}: {detail}")
        final_targets = list(dict.fromkeys(configured))
        requested_policy = "explicit_subset_of_head_aware_allowlist"
    return {
        "target_policy_id": STEP5_LORA_TARGET_POLICY_ID,
        "head_specific_lora_allowlist_id": head_specific_lora_allowlist_id(head),
        "configured_target_modules": configured,
        "allowlist_target_modules": allowlist,
        "final_lora_target_modules": final_targets,
        "forbidden_lora_targets": forbidden_lora_targets_for_model(model),
        "requested_policy": requested_policy,
    }


def discover_step5_text_linear_targets(model: nn.Module, *, head: str = "combined") -> List[str]:
    """Compatibility helper returning the explicit head-aware allowlist."""
    return head_aware_step5_lora_targets(model, head=head)


def apply_native_lora_to_step5_model(
    model: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    head: str,
    target_modules_override: Sequence[str] | None = None,
) -> Dict[str, Any]:
    target_policy = resolve_step5_lora_targets(
        model,
        head=head,
        configured_target_modules=target_modules_override,
    )
    targets = list(target_policy["final_lora_target_modules"])
    for dotted in targets:
        parent, child = _parent_and_child(model, dotted)
        cur = getattr(parent, child)
        if not isinstance(cur, nn.Linear):
            raise RuntimeError(f"LoRA injection failed: {dotted!r} is {type(cur).__name__}, not nn.Linear.")
        reason = _forbidden_reason(model, dotted)
        if reason:
            raise RuntimeError(f"LoRA injection refused forbidden target {dotted!r}: {reason}.")
        setattr(parent, child, LoRALinear(cur, r=int(r), alpha=float(alpha), dropout=float(dropout)))
    return {
        "enabled": True,
        "type": "lora",
        "implementation": "odcr_native_linear_head_aware",
        "r": int(r),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": list(targets),
        "target_policy_id": target_policy["target_policy_id"],
        "head_specific_lora_allowlist_id": target_policy["head_specific_lora_allowlist_id"],
        "forbidden_lora_targets": list(target_policy["forbidden_lora_targets"]),
        "deleted_legacy_modules": list(DELETED_STEP5_LEGACY_MODULES),
        "configured_target_modules": list(target_policy["configured_target_modules"]),
        "allowlist_target_modules": list(target_policy["allowlist_target_modules"]),
        "requested_policy": target_policy["requested_policy"],
    }


__all__ = [
    "HEAD_AWARE_LORA_TARGET_SENTINEL",
    "LoRALinear",
    "apply_native_lora_to_step5_model",
    "discover_step5_text_linear_targets",
    "forbidden_lora_targets_for_model",
    "head_aware_step5_lora_targets",
    "resolve_step5_lora_targets",
]
