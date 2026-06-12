"""Step3 clean-path recovery and phase scheduling helpers.

The helpers in this module are deliberately pure and checkpoint-neutral: they
resolve policy decisions and build handoff metadata without starting training,
eval, or downstream stages.
"""
from __future__ import annotations

from typing import Any, Mapping


STEP3_V3_POLICY_SCHEMA_VERSION = "odcr_step3_v3_policy/1"
STEP3_OBJECTIVE_DRIFT_SCHEMA_VERSION = "odcr_step3_objective_drift/1"
STEP3_RECOVERY_PLAN_SCHEMA_VERSION = "odcr_step3_recovery_plan/1"
STEP3_PHASE_SCHEDULE_SCHEMA_VERSION = "odcr_step3_phase_loss_schedule/1"


DRIFT_COMPONENTS: tuple[str, ...] = (
    "L_rating_shared",
    "L_content_alignment",
    "L_specific_separation",
    "L_variance",
    "L_style_alignment",
    "L_domain_style_alignment",
)


def training_effectiveness_action(record: Mapping[str, Any]) -> str:
    status = str(record.get("effective_improvement_status") or "")
    action = str(record.get("recommended_action") or "")
    reasons = {str(x) for x in (record.get("reasons") or [])}
    if status == "low_lr_no_progress" or action == "review_scheduler":
        return "review_checkpoint_readiness"
    if status == "no_meaningful_improvement" or "validation_plateau" in reasons:
        return "evaluate_recovery_or_loss_rebalance"
    return "continue"


def detect_objective_drift(
    *,
    epoch: int,
    valid_loss: float,
    best_valid_loss: float | None,
    previous_valid_loss: float | None = None,
    component_deltas: Mapping[str, float] | None = None,
    config: Mapping[str, Any] | None = None,
    training_effectiveness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect objective drift from validation and component movement."""

    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", True))
    abs_threshold = float(cfg.get("valid_loss_abs_threshold", 0.25))
    ratio_threshold = float(cfg.get("valid_loss_ratio_threshold", 0.10))
    component_threshold = float(cfg.get("component_weighted_delta_threshold", 0.01))
    severe_component_count = int(cfg.get("severe_component_count", 3))
    severe_abs_threshold = float(cfg.get("severe_valid_loss_abs_threshold", max(abs_threshold * 2.0, abs_threshold)))
    severe_ratio_threshold = float(cfg.get("severe_valid_loss_ratio_threshold", max(ratio_threshold, 0.20)))

    best = float(best_valid_loss) if best_valid_loss is not None else float(valid_loss)
    delta_from_best = float(valid_loss) - best
    ratio_from_best = delta_from_best / max(abs(best), 1.0e-12)
    recent_delta = None if previous_valid_loss is None else float(valid_loss) - float(previous_valid_loss)
    components = {str(k): float(v) for k, v in (component_deltas or {}).items()}
    drift_components = sorted(
        key
        for key in DRIFT_COMPONENTS
        if float(components.get(key, 0.0)) >= component_threshold
    )
    validation_trigger = delta_from_best >= abs_threshold or ratio_from_best >= ratio_threshold
    severe_validation = delta_from_best >= severe_abs_threshold or ratio_from_best >= severe_ratio_threshold
    component_trigger = len(drift_components) >= max(1, severe_component_count - 1)
    effectiveness_action = training_effectiveness_action(training_effectiveness or {})
    if not enabled:
        status = "disabled"
        action = "continue"
    elif validation_trigger and (component_trigger or (recent_delta is not None and recent_delta > 0.0)):
        status = "severe_objective_drift" if (severe_validation or len(drift_components) >= severe_component_count) else "objective_drift"
        action = "start_recovery" if status == "severe_objective_drift" else "plan_recovery_or_loss_rebalance"
    elif validation_trigger:
        status = "warning"
        action = "monitor_recent_trend"
    else:
        status = "none"
        action = "continue"
    if effectiveness_action == "review_checkpoint_readiness" and status in {"objective_drift", "severe_objective_drift"}:
        action = "review_checkpoint_readiness"
    return {
        "schema_version": STEP3_OBJECTIVE_DRIFT_SCHEMA_VERSION,
        "enabled": enabled,
        "epoch": int(epoch),
        "status": status,
        "valid_loss": float(valid_loss),
        "best_valid_loss": best,
        "delta_from_best": float(delta_from_best),
        "ratio_from_best": float(ratio_from_best),
        "recent_delta": recent_delta,
        "validation_trigger": bool(validation_trigger),
        "component_trigger": bool(component_trigger),
        "drift_components": drift_components,
        "component_deltas": components,
        "training_effectiveness_action": effectiveness_action,
        "action": action,
    }


def build_recovery_plan(
    *,
    epoch: int,
    drift_record: Mapping[str, Any],
    config: Mapping[str, Any],
    best_observed_checkpoint: str,
    latest_checkpoint: str | None = None,
    recovery_index: int = 1,
) -> dict[str, Any]:
    cfg = dict(config or {})
    if "recovery_epochs" not in cfg:
        raise ValueError("Step3 recovery config must include explicit recovery_epochs.")
    restart_ratio = float(cfg.get("restart_lr_ratio", 0.25))
    recovery_epochs = int(cfg["recovery_epochs"])
    max_recoveries = int(cfg.get("max_recoveries", 1))
    source_scope = str(cfg.get("source_checkpoint_scope") or "best_observed")
    scheduler = str(cfg.get("recovery_scheduler") or "short_cosine")
    if source_scope != "best_observed":
        raise ValueError("Step3 recovery source_checkpoint_scope must be best_observed.")
    if latest_checkpoint and str(latest_checkpoint) == str(best_observed_checkpoint):
        latest_checkpoint = None
    return {
        "schema_version": STEP3_RECOVERY_PLAN_SCHEMA_VERSION,
        "enabled": bool(cfg.get("enabled", True)),
        "formal_allowed": bool(cfg.get("formal_allowed", True)),
        "epoch": int(epoch),
        "recovery_index": int(recovery_index),
        "max_recoveries": max_recoveries,
        "action": "rollback_best_observed_and_restart",
        "trigger_status": str(drift_record.get("status") or ""),
        "source_checkpoint_scope": source_scope,
        "source_checkpoint": str(best_observed_checkpoint),
        "forbidden_source_checkpoint": str(latest_checkpoint or ""),
        "save_drift_checkpoint": bool(cfg.get("save_drift_checkpoint", True)),
        "restart_lr_ratio": restart_ratio,
        "recovery_epochs": recovery_epochs,
        "recovery_scheduler": scheduler,
        "max_recoveries_prevents_infinite_loop": True,
    }


def resolve_phase_for_epoch(
    *,
    epoch: int,
    config: Mapping[str, Any],
    objective_drift_status: str = "none",
    recovery_active: bool = False,
) -> dict[str, Any]:
    cfg = dict(config or {})
    phases = list(cfg.get("phases") or [])
    if not phases:
        phases = [
            {"name": "alignment_warmup", "start_epoch": 1, "end_epoch": 2, "loss_multipliers": {}},
            {"name": "task_refinement", "start_epoch": 3, "end_epoch": 5, "loss_multipliers": {}},
        ]
    selected = phases[-1]
    if recovery_active:
        for phase in phases:
            if str(phase.get("name")) == "task_refinement":
                selected = phase
                break
    elif str(objective_drift_status) in {"objective_drift", "severe_objective_drift"}:
        for phase in phases:
            if str(phase.get("name")) == "task_refinement":
                selected = phase
                break
    else:
        for phase in phases:
            start = int(phase.get("start_epoch", 1) or 1)
            end_raw = phase.get("end_epoch")
            end = None if end_raw in (None, "") else int(end_raw)
            if int(epoch) >= start and (end is None or int(epoch) <= end):
                selected = phase
                break
    return {
        "schema_version": STEP3_PHASE_SCHEDULE_SCHEMA_VERSION,
        "enabled": bool(cfg.get("enabled", True)),
        "epoch": int(epoch),
        "phase": str(selected.get("name") or "task_refinement"),
        "transition": str(cfg.get("transition") or "epoch_or_objective_drift"),
        "objective_drift_status": str(objective_drift_status),
        "recovery_active": bool(recovery_active),
        "loss_multipliers": dict(selected.get("loss_multipliers") or {}),
    }


def apply_loss_multipliers(weights: Mapping[str, float], multipliers: Mapping[str, Any]) -> dict[str, float]:
    """Apply internal-only curriculum multipliers to existing loss components."""
    out = {str(k): float(v) for k, v in weights.items()}
    for key, value in multipliers.items():
        if key in out:
            out[key] = float(out[key]) * float(value)
    return out
