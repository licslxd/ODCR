"""Step5 task-decoupled policy helpers.

The policy is intentionally hard-edged: Step5A is a target-gold scorer path and
Step5B is the large explainer path.  Silent mixed-head behavior is retired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


STEP5_TASK_DECOUPLED_POLICY_SCHEMA_VERSION = "odcr_step5_task_decoupled_policy/1"
TARGET_GOLD_COMPONENT_ID = 0
AUX_GOLD_COMPONENT_ID = 1
CF_COMPONENT_ID = 2


class Step5TaskDecoupledPolicyError(RuntimeError):
    """Raised when retired Step5 mixed-head behavior is requested."""


@dataclass(frozen=True)
class Step5TaskDecoupledPolicy:
    enabled: bool
    step5a_branch: str
    step5a_target_gold: float
    step5a_aux_gold: float
    step5a_cf: float
    step5a_forbid_aux_cf_in_scorer_loss: bool
    step5a_forbid_generation: bool
    step5a_forbid_big_model: bool
    step5a_scorer_init_required: bool
    step5a_scorer_init_source: str
    step5a_distillation_enabled: bool
    step5a_distillation_weight: float
    step5a_teacher: Mapping[str, Any]
    step5a_residual_calibration: Mapping[str, Any]
    step5b_branch: str
    step5b_use_big_model: bool
    step5b_aux_gold: float
    step5b_cf: float
    step5b_target_gold: Any


def default_task_decoupled_policy() -> dict[str, Any]:
    return {
        "schema_version": STEP5_TASK_DECOUPLED_POLICY_SCHEMA_VERSION,
        "enabled": True,
        "step5A": {
            "branch": "scorer_clean",
            "train_components": {"target_gold": 1.0, "aux_gold": 0.0, "cf": 0.0},
            "forbid_aux_cf_in_scorer_loss": True,
            "forbid_generation": True,
            "forbid_big_model": True,
            "scorer_init_required": True,
            "scorer_init_source": "frozen_step3_teacher",
            "distillation_enabled": False,
            "distillation_weight": 0.0,
            "teacher": {
                "checkpoint_path": "runs/step3/task2/2/model/best_observed.pth",
                "checkpoint_sha256": "9089ac53b138c12ba1260370aed3d637b305f7f7f6a98a7bcbc7721eb5559017",
                "parity_required": True,
                "tokenized_evidence_source": "step3_tokenizer_cache",
            },
            "residual_calibration": {
                "enabled": True,
                "zero_init": True,
                "feature_source": "teacher_pred",
                "lambda_gt_initial": 0.0,
                "lambda_gt_final": 1.0,
                "lambda_distill_initial": 1.0,
                "lambda_distill_final": 0.25,
                "lambda_residual": 0.05,
                "regularizer": "huber",
                "huber_delta": 0.1,
            },
        },
        "step5B": {
            "branch": "explainer_rich",
            "train_components": {"target_gold": "optional_anchor", "aux_gold": ">0", "cf": ">0"},
            "use_big_model": True,
            "allow_target_anchor": True,
            "target_anchor_role": "optional_target_explanation_anchor_not_rating_supervision",
        },
    }


def parse_task_decoupled_policy(raw: Mapping[str, Any] | None) -> Step5TaskDecoupledPolicy:
    obj = dict(raw or default_task_decoupled_policy())
    a = obj.get("step5A") if isinstance(obj.get("step5A"), Mapping) else {}
    b = obj.get("step5B") if isinstance(obj.get("step5B"), Mapping) else {}
    ac = a.get("train_components") if isinstance(a.get("train_components"), Mapping) else {}
    bc = b.get("train_components") if isinstance(b.get("train_components"), Mapping) else {}
    return Step5TaskDecoupledPolicy(
        enabled=bool(obj.get("enabled", True)),
        step5a_branch=str(a.get("branch") or "scorer_clean"),
        step5a_target_gold=float(ac.get("target_gold", 1.0)),
        step5a_aux_gold=float(ac.get("aux_gold", 0.0)),
        step5a_cf=float(ac.get("cf", 0.0)),
        step5a_forbid_aux_cf_in_scorer_loss=bool(a.get("forbid_aux_cf_in_scorer_loss", True)),
        step5a_forbid_generation=bool(a.get("forbid_generation", True)),
        step5a_forbid_big_model=bool(a.get("forbid_big_model", True)),
        step5a_scorer_init_required=bool(a.get("scorer_init_required", True)),
        step5a_scorer_init_source=str(a.get("scorer_init_source") or "frozen_step3_teacher"),
        step5a_distillation_enabled=bool(a.get("distillation_enabled", False)),
        step5a_distillation_weight=float(a.get("distillation_weight", 0.0) or 0.0),
        step5a_teacher=dict(a.get("teacher") or {}),
        step5a_residual_calibration=dict(a.get("residual_calibration") or {}),
        step5b_branch=str(b.get("branch") or "explainer_rich"),
        step5b_use_big_model=bool(b.get("use_big_model", True)),
        step5b_aux_gold=float(bc.get("aux_gold", 0.0)) if not isinstance(bc.get("aux_gold"), str) else 1.0,
        step5b_cf=float(bc.get("cf", 0.0)) if not isinstance(bc.get("cf"), str) else 1.0,
        step5b_target_gold=bc.get("target_gold", "optional_anchor"),
    )


def assert_step5a_policy_clean(policy_raw: Mapping[str, Any] | None) -> None:
    policy = parse_task_decoupled_policy(policy_raw)
    if not policy.enabled:
        raise Step5TaskDecoupledPolicyError("step5.task_decoupled_policy.enabled must be true")
    if policy.step5a_branch != "scorer_clean":
        raise Step5TaskDecoupledPolicyError("Step5A branch must be scorer_clean")
    if abs(policy.step5a_target_gold - 1.0) > 1e-9 or policy.step5a_aux_gold != 0.0 or policy.step5a_cf != 0.0:
        raise Step5TaskDecoupledPolicyError(
            "Step5A scorer-clean policy requires target_gold=1.0, aux_gold=0.0, cf=0.0"
        )
    if not policy.step5a_forbid_aux_cf_in_scorer_loss:
        raise Step5TaskDecoupledPolicyError("Step5A must forbid aux/cf in scorer loss")
    if not policy.step5a_forbid_generation or not policy.step5a_forbid_big_model:
        raise Step5TaskDecoupledPolicyError("Step5A must forbid generation and big-model construction")
    if (
        policy.step5a_scorer_init_required
        and policy.step5a_scorer_init_source != "frozen_step3_teacher"
        and not policy.step5a_distillation_enabled
    ):
        raise Step5TaskDecoupledPolicyError(
            "Step5A scorer-clean requires frozen_step3_teacher; partial transplant/random init is forbidden"
        )
    if not bool(policy.step5a_teacher.get("parity_required", True)):
        raise Step5TaskDecoupledPolicyError("Step5A frozen Step3 teacher parity_required must be true")
    if not bool(policy.step5a_residual_calibration.get("zero_init", True)):
        raise Step5TaskDecoupledPolicyError("Step5A residual calibration must be zero-init")


def enforce_step5a_target_gold_counts(
    counts: Mapping[str, Any],
    *,
    context: str,
    require_positive_target: bool = True,
) -> None:
    target = int(counts.get("target_gold") or 0)
    aux = int(counts.get("aux_gold") or 0)
    cf = int(counts.get("cf") or counts.get("aux_cf") or 0)
    if require_positive_target and target <= 0:
        raise Step5TaskDecoupledPolicyError(f"{context}: Step5A target_gold count must be positive")
    if aux != 0 or cf != 0:
        raise Step5TaskDecoupledPolicyError(
            f"{context}: Step5A scorer-clean forbids aux_gold/cf rows; got target_gold={target} aux_gold={aux} cf={cf}"
        )


def normalized_actual_counts(counts: Mapping[str, Any] | None) -> dict[str, int]:
    raw = dict(counts or {})
    return {
        "target_gold": int(raw.get("target_gold") or 0),
        "aux_gold": int(raw.get("aux_gold") or 0),
        "cf": int(raw.get("cf") or raw.get("aux_cf") or 0),
    }


def assert_step5a_batch_target_gold_only(component_ids: torch.Tensor | None, *, stage: str) -> dict[str, int]:
    if component_ids is None:
        raise Step5TaskDecoupledPolicyError(
            f"{stage}: Step5A scorer-clean requires sampler_component_id metadata for target_gold-only enforcement"
        )
    flat = component_ids.detach().view(-1).long()
    if flat.numel() <= 0:
        return {"target_gold": 0, "aux_gold": 0, "cf": 0}
    target = int((flat == TARGET_GOLD_COMPONENT_ID).sum().item())
    aux = int((flat == AUX_GOLD_COMPONENT_ID).sum().item())
    cf = int((flat == CF_COMPONENT_ID).sum().item())
    unknown = int(((flat != TARGET_GOLD_COMPONENT_ID) & (flat != AUX_GOLD_COMPONENT_ID) & (flat != CF_COMPONENT_ID)).sum().item())
    if aux or cf or unknown:
        raise Step5TaskDecoupledPolicyError(
            f"{stage}: Step5A scorer-clean batch contains forbidden components: "
            f"target_gold={target} aux_gold={aux} cf={cf} unknown={unknown}"
        )
    return {"target_gold": target, "aux_gold": aux, "cf": cf}


__all__ = [
    "AUX_GOLD_COMPONENT_ID",
    "CF_COMPONENT_ID",
    "STEP5_TASK_DECOUPLED_POLICY_SCHEMA_VERSION",
    "Step5TaskDecoupledPolicyError",
    "TARGET_GOLD_COMPONENT_ID",
    "assert_step5a_batch_target_gold_only",
    "assert_step5a_policy_clean",
    "default_task_decoupled_policy",
    "enforce_step5a_target_gold_counts",
    "normalized_actual_counts",
    "parse_task_decoupled_policy",
]
