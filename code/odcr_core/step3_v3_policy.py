"""Step3 V3 recovery, conflict-audit, and paper-aware selection helpers.

The helpers in this module are deliberately pure and checkpoint-neutral: they
resolve policy decisions, validate schemas, and build handoff metadata without
starting training, eval, or downstream stages.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from odcr_core.csb_contract import CSB_ODCR_METHOD_NAME, CSB_PACKET_SCHEMA_VERSION


STEP3_V3_POLICY_SCHEMA_VERSION = "csb_odcr_step3_v3_policy/1"
STEP3_OBJECTIVE_DRIFT_SCHEMA_VERSION = "odcr_step3_objective_drift/1"
STEP3_RECOVERY_PLAN_SCHEMA_VERSION = "odcr_step3_recovery_plan/1"
STEP3_PHASE_SCHEDULE_SCHEMA_VERSION = "csb_odcr_step3_phase_loss_schedule/1"
STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION = "csb_odcr_step3_loss_gradient_conflict_audit/1"
STEP3_PAPER_CANDIDATE_SCHEMA_VERSION = "csb_odcr_step3_paper_candidate_selection/1"


STEP3_TOTAL_LOSS_COMPONENTS: tuple[str, ...] = (
    "L_rating_shared",
    "L_light_explainer",
    "L_orthogonal",
    "L_variance",
    "L_shared_invariance",
    "L_specific_separation",
    "L_anchor_content",
    "L_anchor_style",
    "L_content_alignment",
    "L_style_alignment",
    "L_shared_proto",
    "L_domain_style_alignment",
    "L_local_style_alignment",
    "L_polarity_alignment",
    "L_residual_specific",
    "L_prototype_separation",
)


LOSS_GROUPS: dict[str, tuple[str, ...]] = {
    "rating": ("L_rating_shared",),
    "explanation": ("L_light_explainer",),
    "easd_content": (
        "L_content_alignment",
        "L_polarity_alignment",
        "L_anchor_content",
        "L_shared_proto",
    ),
    "hss_style": (
        "L_style_alignment",
        "L_anchor_style",
        "L_domain_style_alignment",
        "L_local_style_alignment",
    ),
    "disentangle_geometry": (
        "L_specific_separation",
        "L_variance",
        "L_orthogonal",
        "L_residual_specific",
        "L_shared_invariance",
        "L_prototype_separation",
    ),
}


DRIFT_COMPONENTS: tuple[str, ...] = (
    "L_rating_shared",
    "L_content_alignment",
    "L_specific_separation",
    "L_variance",
    "L_style_alignment",
    "L_domain_style_alignment",
)


def validate_loss_group_mapping(active_components: Sequence[str] | None = None) -> dict[str, Any]:
    """Return an internal-only audit schema and fail if any active loss is unmapped."""

    active = tuple(active_components or STEP3_TOTAL_LOSS_COMPONENTS)
    mapped = {name for names in LOSS_GROUPS.values() for name in names}
    missing = sorted(set(active) - mapped)
    duplicate_components = sorted(
        name
        for name in mapped
        if sum(1 for names in LOSS_GROUPS.values() if name in names) > 1
    )
    status = "pass" if not missing and not duplicate_components else "fail"
    return {
        "schema_version": STEP3_GRADIENT_CONFLICT_SCHEMA_VERSION,
        "status": status,
        "loss_groups": {key: list(value) for key, value in LOSS_GROUPS.items()},
        "active_components": list(active),
        "unmapped_components": missing,
        "duplicate_components": duplicate_components,
        "real_data_only": True,
        "synthetic_benchmark_forbidden": True,
        "writes_formal_checkpoint": False,
    }


def training_effectiveness_action(record: Mapping[str, Any]) -> str:
    status = str(record.get("effective_improvement_status") or "")
    action = str(record.get("recommended_action") or "")
    reasons = {str(x) for x in (record.get("reasons") or [])}
    if status == "low_lr_no_progress" or action == "review_scheduler":
        return "stop_and_select_candidate"
    if status == "no_meaningful_improvement" or "validation_plateau" in reasons:
        return "evaluate_recovery_or_loss_rebalance"
    if action == "run_paper_target_only_eval":
        return "paper_candidate_probe"
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
    """Detect objective drift from validation and component movement.

    The function is intentionally conservative: tiny validation noise does not
    trigger drift unless the configured absolute/relative gap and component
    movement support the decision.
    """

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
    if effectiveness_action == "stop_and_select_candidate" and status in {"objective_drift", "severe_objective_drift"}:
        action = "stop_and_select_candidate"
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
    restart_ratio = float(cfg.get("restart_lr_ratio", 0.25))
    recovery_epochs = int(cfg.get("recovery_epochs", 8))
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
        "damping_enabled": False,
        "candidate_scope": "recovery",
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
            {"name": "primary_fit", "start_epoch": 1, "end_epoch": 2, "loss_multipliers": {}},
            {"name": "csb_alignment_controlled_injection", "start_epoch": 3, "end_epoch": 24, "loss_multipliers": {}},
            {"name": "recovery_pareto_candidate", "start_epoch": 25, "end_epoch": None, "loss_multipliers": {}},
        ]
    selected = phases[-1]
    if recovery_active:
        for phase in phases:
            if str(phase.get("name")) == "csb_alignment_controlled_injection":
                selected = phase
                break
    elif str(objective_drift_status) in {"objective_drift", "severe_objective_drift"}:
        for phase in phases:
            if str(phase.get("name")) == "csb_alignment_controlled_injection":
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
        "phase": str(selected.get("name") or "csb_alignment_controlled_injection"),
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


def safe_damping_v2_decision(
    *,
    epoch: int,
    valid_loss: float,
    best_valid_loss: float,
    previous_valid_loss: float | None,
    current_lr: float,
    base_min_lr: float,
    event_count: int,
    cooldown_remaining: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", False)):
        return {"apply": False, "reason": "disabled", "cooldown_remaining": int(cooldown_remaining)}
    max_events = int(cfg.get("max_damping_events", 2))
    if int(event_count) >= max_events:
        return {"apply": False, "reason": "max_damping_events_reached", "action_gate": "stop_and_select_candidate"}
    if int(cooldown_remaining) > 0:
        return {"apply": False, "reason": "cooldown_active", "cooldown_remaining": int(cooldown_remaining) - 1}
    start_epoch = int(cfg.get("start_epoch", 4))
    if int(epoch) < start_epoch:
        return {"apply": False, "reason": "warmup_or_peak_transition"}
    abs_gap = float(valid_loss) - float(best_valid_loss)
    ratio_gap = abs_gap / max(abs(float(best_valid_loss)), 1.0e-12)
    worsened = (
        abs_gap >= float(cfg.get("worsen_abs_threshold", 0.25))
        or ratio_gap >= float(cfg.get("worsen_ratio_threshold", 0.10))
    )
    recent_ok = previous_valid_loss is None or (float(valid_loss) - float(previous_valid_loss)) >= -float(
        cfg.get("recent_recovery_tolerance", 1.0e-3)
    )
    floor_ratio = float(cfg.get("effective_lr_floor_ratio", 0.25))
    floor_abs = float(cfg.get("effective_lr_floor_abs", 0.0))
    floor = max(floor_abs, float(base_min_lr) * floor_ratio)
    factor = float(cfg.get("lr_decay_factor", 0.5))
    after = max(float(current_lr) * factor, floor)
    if float(current_lr) <= floor * 1.01:
        return {"apply": False, "reason": "effective_lr_floor_reached", "effective_lr_floor": floor, "action_gate": "low_lr_no_progress"}
    if not (worsened and recent_ok):
        return {"apply": False, "reason": "recent_trend_not_worse", "effective_lr_floor": floor}
    return {
        "apply": True,
        "reason": "safe_damping_v2_recent_trend_worsened",
        "epoch": int(epoch),
        "worsen_abs": float(abs_gap),
        "worsen_ratio": float(ratio_gap),
        "lr_decay_factor": factor,
        "lr_after": after,
        "effective_lr_floor": floor,
        "cooldown_remaining": int(cfg.get("cooldown_epochs", 3)),
        "event_count_after": int(event_count) + 1,
    }


def flatten_paper_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    recommendation = metrics.get("recommendation") if isinstance(metrics.get("recommendation"), Mapping) else {}
    explanation = metrics.get("explanation") if isinstance(metrics.get("explanation"), Mapping) else {}
    rouge = explanation.get("rouge") if isinstance(explanation.get("rouge"), Mapping) else {}
    bleu = explanation.get("bleu") if isinstance(explanation.get("bleu"), Mapping) else {}
    dist = explanation.get("dist") if isinstance(explanation.get("dist"), Mapping) else {}
    mapping = {
        "MAE": recommendation.get("mae"),
        "RMSE": recommendation.get("rmse"),
        "ROUGE-1": rouge.get("1"),
        "ROUGE-L": rouge.get("l"),
        "BLEU-1": bleu.get("1"),
        "BLEU-2": bleu.get("2"),
        "BLEU-3": bleu.get("3"),
        "BLEU-4": bleu.get("4"),
        "DIST-1": dist.get("1"),
        "DIST-2": dist.get("2"),
        "METEOR": explanation.get("meteor"),
    }
    for key, value in mapping.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def candidate_paper_score(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config or {})
    metrics = {str(k): float(v) for k, v in (candidate.get("metrics") or {}).items()}
    rating_guard = cfg.get("rating_guard") if isinstance(cfg.get("rating_guard"), Mapping) else {}
    diversity_guard = cfg.get("diversity_guard") if isinstance(cfg.get("diversity_guard"), Mapping) else {}
    weights = cfg.get("composite_weights") if isinstance(cfg.get("composite_weights"), Mapping) else {}
    max_mae = float(rating_guard.get("max_mae", 10.0))
    max_rmse = float(rating_guard.get("max_rmse", 10.0))
    dist1_floor = float(diversity_guard.get("dist1_floor", 0.05))
    dist2_floor = float(diversity_guard.get("dist2_floor", 0.20))
    mae = metrics.get("MAE", float("inf"))
    rmse = metrics.get("RMSE", float("inf"))
    dist1 = metrics.get("DIST-1", 0.0)
    dist2 = metrics.get("DIST-2", 0.0)
    candidate_conflict_rate = float(candidate.get("candidate_conflict_rate", metrics.get("candidate_conflict_rate", 0.0)))
    rating_aux_conflict_rate = float(candidate.get("rating_aux_conflict_rate", metrics.get("rating_aux_conflict_rate", candidate_conflict_rate)))
    csb_stability_score = float(candidate.get("csb_stability_score", metrics.get("csb_stability_score", 0.0)))
    rating_ok = mae <= max_mae and rmse <= max_rmse
    diversity_ok = dist1 >= dist1_floor and dist2 >= dist2_floor
    text_score = (
        float(weights.get("rouge_l", 0.30)) * metrics.get("ROUGE-L", 0.0)
        + float(weights.get("bleu4", 0.25)) * metrics.get("BLEU-4", 0.0)
        + float(weights.get("meteor", 0.25)) * metrics.get("METEOR", 0.0)
        + float(weights.get("dist1", 0.10)) * metrics.get("DIST-1", 0.0)
        + float(weights.get("dist2", 0.10)) * metrics.get("DIST-2", 0.0)
    )
    rating_score = -(mae + rmse)
    collapse_penalty = float(diversity_guard.get("collapse_penalty", 100.0)) if not diversity_ok else 0.0
    conflict_penalty = max(0.0, candidate_conflict_rate) + max(0.0, rating_aux_conflict_rate)
    csb_bonus = max(0.0, min(1.0, csb_stability_score)) * 0.10
    return {
        "candidate_id": str(candidate.get("candidate_id") or candidate.get("checkpoint_scope") or "candidate"),
        "checkpoint": str(candidate.get("checkpoint") or ""),
        "checkpoint_hash": str(candidate.get("checkpoint_hash") or ""),
        "checkpoint_scope": str(candidate.get("checkpoint_scope") or ""),
        "metrics": metrics,
        "rating_guard_pass": bool(rating_ok),
        "diversity_guard_pass": bool(diversity_ok),
        "text_score": float(text_score),
        "rating_score": float(rating_score),
        "explainer_score": float(text_score - collapse_penalty - conflict_penalty + csb_bonus),
        "collapse_penalty": float(collapse_penalty),
        "candidate_conflict_rate": float(candidate_conflict_rate),
        "rating_aux_conflict_rate": float(rating_aux_conflict_rate),
        "csb_stability_score": float(csb_stability_score),
        "conflict_penalty": float(conflict_penalty),
        "csb_bonus": float(csb_bonus),
    }


def select_paper_aware_candidates(
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(config or {})
    scored = [candidate_paper_score(item, cfg) for item in candidates]
    scorer_pool = [item for item in scored if item["rating_guard_pass"]]
    explainer_pool = [item for item in scored if item["rating_guard_pass"] and item["diversity_guard_pass"]]
    scorer = min(scorer_pool, key=lambda item: (item["metrics"].get("MAE", float("inf")), item["metrics"].get("RMSE", float("inf"))), default=None)
    explainer = max(explainer_pool, key=lambda item: item["explainer_score"], default=None)
    return {
        "schema_version": STEP3_PAPER_CANDIDATE_SCHEMA_VERSION,
        "method_name": CSB_ODCR_METHOD_NAME,
        "csb_packet_schema_version": CSB_PACKET_SCHEMA_VERSION,
        "selection_available": bool(scored),
        "paper_eval_required": True,
        "candidate_count": len(scored),
        "candidate_scores": scored,
        "scorer_downstream_checkpoint": scorer,
        "explainer_downstream_checkpoint": explainer,
        "csb_control_checkpoint": explainer or scorer,
        "csb_packet_source": (explainer or scorer or {}).get("checkpoint") if (explainer or scorer) else "",
        "route_calibration_source": scorer,
        "scorer_explainer_can_differ": True,
        "no_paper_eval_no_selection": len(scored) == 0,
        "dist_guard_active": True,
        "low_dist_blocks_explainer": True,
        "csb_candidate_selection": True,
    }


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
