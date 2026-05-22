"""Step5 trainable graph contracts shared by formal preflight and E4 probes."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


DELETED_STEP5_LEGACY_MODULES: tuple[str, ...] = (
    "recommender",
    "flan_soft_prompt_stack",
    "hidden2token",
)

STEP5_LORA_TARGET_POLICY_ID = "step5_explanation_lora_allowlist/1"
STEP5_ALL_TRAINABLE_GRAD_POLICY_ID = "step5_all_trainable_grad/1"


def normalize_step5_head_for_contract(head: Any) -> str:
    raw = str(head or "explanation").strip()
    if raw.lower() == ("step5" + "b").lower():
        return "explanation"
    if raw != "explanation":
        raise RuntimeError("invalid Step5 head for trainable contract: Step5 is explanation-only")
    return "explanation"


def head_specific_lora_allowlist_id(head: Any) -> str:
    return f"{STEP5_LORA_TARGET_POLICY_ID}:{normalize_step5_head_for_contract(head)}"


def head_specific_trainable_policy_id(head: Any) -> str:
    return f"step5_head_specific_trainable_contract/1:{normalize_step5_head_for_contract(head)}"


def head_gated_loss_contract(head: Any) -> dict[str, Any]:
    norm = normalize_step5_head_for_contract(head)
    active = ["explainer_ce", "ccv", "fca", "orthogonal_keep"]
    graph_zero = ["retired_prediction_zero_guard", "repeat_ul", "terminal_clean", "batch_diversity"]
    return {
        "schema_version": "odcr_step5_explanation_loss_contract/1",
        "head": norm,
        "mode": "explanation_only",
        "active_losses": active,
        "graph_safe_zero_losses": graph_zero,
        "single_total_loss_insertion": True,
        "rating_training": False,
    }


def _param_grad_finite(param: nn.Parameter) -> bool:
    if param.grad is None:
        return False
    try:
        return bool(torch.isfinite(param.grad).all().item())
    except RuntimeError:
        return False


def build_all_trainable_grad_report(
    model: nn.Module,
    *,
    head: Any,
    evidence_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad_present = param.grad is not None
        rows.append(
            {
                "name": name,
                "requires_grad": True,
                "grad_present": bool(grad_present),
                "grad_finite": bool(_param_grad_finite(param)) if grad_present else False,
                "is_lora": "lora_" in name,
                "numel": int(param.numel()),
            }
        )
    missing = [row["name"] for row in rows if not row["grad_present"]]
    nonfinite = [row["name"] for row in rows if row["grad_present"] and not row["grad_finite"]]
    lora_rows = [row for row in rows if row["is_lora"]]
    status = "pass" if not missing and not nonfinite else "fail"
    return {
        "schema_version": STEP5_ALL_TRAINABLE_GRAD_POLICY_ID,
        "status": status,
        "head": normalize_step5_head_for_contract(head),
        "evidence_context": dict(evidence_context or {}),
        "trainable_param_count": len(rows),
        "grad_present_count": sum(1 for row in rows if row["grad_present"]),
        "grad_finite_count": sum(1 for row in rows if row["grad_finite"]),
        "lora_trainable_count": len(lora_rows),
        "lora_grad_present_count": sum(1 for row in lora_rows if row["grad_present"]),
        "lora_grad_finite_count": sum(1 for row in lora_rows if row["grad_finite"]),
        "missing_grad_params": missing,
        "nonfinite_grad_params": nonfinite,
        "all_trainable_grad_table": rows,
    }


def validate_all_trainable_params_receive_grad(
    model: nn.Module,
    loss: torch.Tensor | None = None,
    *,
    head: Any,
    evidence_context: Mapping[str, Any] | None = None,
    fail_on_missing: bool = True,
) -> dict[str, Any]:
    """Validate the current backward pass covered every trainable parameter."""
    del loss
    report = build_all_trainable_grad_report(model, head=head, evidence_context=evidence_context)
    if fail_on_missing and report["status"] != "pass":
        missing = list(report.get("missing_grad_params") or [])
        nonfinite = list(report.get("nonfinite_grad_params") or [])
        parts: list[str] = []
        if missing:
            preview = ", ".join(missing[:20])
            more = "" if len(missing) <= 20 else f" ... (+{len(missing) - 20} more)"
            parts.append(f"trainable params without grad: {preview}{more}")
        if nonfinite:
            preview = ", ".join(nonfinite[:20])
            more = "" if len(nonfinite) <= 20 else f" ... (+{len(nonfinite) - 20} more)"
            parts.append(f"trainable params with non-finite grad: {preview}{more}")
        raise RuntimeError("Step5 all-trainable-grad preflight failed; " + "; ".join(parts))
    return report


__all__ = [
    "DELETED_STEP5_LEGACY_MODULES",
    "STEP5_ALL_TRAINABLE_GRAD_POLICY_ID",
    "STEP5_LORA_TARGET_POLICY_ID",
    "build_all_trainable_grad_report",
    "head_gated_loss_contract",
    "head_specific_lora_allowlist_id",
    "head_specific_trainable_policy_id",
    "normalize_step5_head_for_contract",
    "validate_all_trainable_params_receive_grad",
]
