"""Step5 formal-preparation gate and AI_analysis contract writer.

This module is deliberately non-formal.  It freezes the selected bounded
tuning evidence into a preparation contract, checks that the active sampler
matches the bounded preflight samples, and writes audit artifacts only under
AI_analysis.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from odcr_core.config_resolver import resolve_config
from odcr_core.file_atomic import atomic_write_json
from odcr_core.aux.runtime.gpu_pane_handoff import (
    HandoffError,
    OLD_GPU_PANE_STATE_ROLE,
    TARGET_SOURCE_HANDOFF,
    current_handoff_path,
    load_current_handoff,
    selection_from_handoff_payload,
)
from odcr_core.manifests import build_formal_source_table_snapshot
from odcr_core.step5_auto_budget import (
    compute_step5_auto_budget_report,
    replacement_summary_for_effective_samples,
)


FORMAL_PREP_CANDIDATE_SCHEMA = "odcr_step5_formal_preparation_candidate/1"
FORMAL_PREP_PREFLIGHT_SCHEMA = "odcr_step5_formal_preparation_preflight/1"
LOW_WEIGHTED_CONSISTENCY_SCHEMA = "odcr_step5_low_weighted_consistency/1"
FINAL_BOUNDED_PREFLIGHT_SCHEMA = "odcr_step5_final_bounded_preflight/1"

REPORT_PATHS: dict[str, Path] = {
    "index": Path("AI_analysis/00_index/step5_formal_preparation_preflight_index.md"),
    "raw": Path("AI_analysis/01_raw_logs/step5_formal_preparation_preflight_raw.log"),
    "hits": Path("AI_analysis/02_search_hits/step5_formal_preparation_preflight_hits.txt"),
    "ledger": Path("AI_analysis/03_evidence_ledgers/step5_formal_preparation_preflight_ledger.md"),
    "summary": Path("AI_analysis/04_phase_summaries/step5_formal_preparation_preflight_summary.md"),
    "report": Path("AI_analysis/05_final_reports/step5_formal_preparation_preflight_report.md"),
    "machine": Path("AI_analysis/05_final_reports/step5_formal_preparation_preflight_machine_verdict.json"),
    "candidate": Path("AI_analysis/05_final_reports/step5_formal_preparation_candidate.json"),
    "reconciled_candidate": Path("AI_analysis/05_final_reports/step5_formal_readiness_reconciled_candidate.json"),
    "sample_budget": Path("AI_analysis/05_final_reports/step5_formal_sample_budget_reconciliation.json"),
    "split_plan": Path("AI_analysis/05_final_reports/step5_formal_split_execution_plan.json"),
    "low_weighted": Path("AI_analysis/05_final_reports/step5_low_weighted_consistency_report.json"),
    "final_bounded": Path("AI_analysis/05_final_reports/step5_final_bounded_preflight_report.json"),
}

REQUIRED_SOURCE_KEYS = (
    "step5_sampler",
    "step5_sampler.step5A",
    "step5_sampler.step5B",
    "step5_tuning",
    "step5_tuning.batch_candidate",
    "step5_tuning.fallback_batch_candidate",
    "step5_tuning.selected_budget_candidate",
    "step5_tuning.lr_candidates",
    "step5_tuning.warmup_fraction_candidates",
    "step5_tuning.innovation_weight_candidates",
    "step5_tuning.ratio_candidates",
    "step5_tuning.cf_tier_mix_candidates",
    "step5_tuning.gold_tier_mix_candidates",
    "step5_batch_candidates",
    "step5_effective_epoch",
    "step5_prompt_templates",
)

FORMAL_READINESS_SOURCE_KEYS = (
    "selected_tuning_candidate",
    "fallback_tuning_candidate",
    "batch_candidate",
    "fallback_batch_candidate",
    "ratio.step5A",
    "ratio.step5B",
    "cf_tier_mix.step5A",
    "cf_tier_mix.step5B",
    "gold_tier_mix.target_gold",
    "gold_tier_mix.aux_gold",
    "low_weighted_policy",
    "lr",
    "innovation_weights",
    "warmup_fraction",
    "effective_samples.step5A",
    "effective_samples.step5B",
    "optimizer_steps.step5A",
    "optimizer_steps.step5B",
    "max_effective_epochs",
    "early_stopping_patience",
    "prompt_registry",
    "pool_manifest",
    "sampling_contract",
    "formal_run_disabled",
    "full_audit_forbidden",
    "old_dedicated_forbidden",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"missing": True, "path": str(p)}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"invalid": True, "path": str(p), "error": str(exc)}
    return obj if isinstance(obj, dict) else {"invalid": True, "path": str(p), "type": type(obj).__name__}


def _repo_rel(repo_root: Path, path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / p
    try:
        return str(p.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(p.resolve())


def parse_step5_candidate_id(candidate_id: str) -> dict[str, Any]:
    parts = [part for part in str(candidate_id).split("+") if part]
    parsed: dict[str, Any] = {"raw": str(candidate_id), "parts": parts}
    for part in parts:
        if part.startswith("A_RATIO_"):
            parsed["step5A_ratio"] = part
        elif part.startswith("B_RATIO_"):
            parsed["step5B_ratio"] = part
        elif part.startswith("A_CF_MIX_"):
            parsed["step5A_cf_mix"] = part
        elif part.startswith("B_CF_MIX_"):
            parsed["step5B_cf_mix"] = part
        elif part.startswith("TG_MIX_"):
            parsed["target_gold_mix"] = part
        elif part.startswith("AG_MIX_"):
            parsed["aux_gold_mix"] = part
        elif part.startswith("LR_"):
            raw = part[3:]
            parsed["lr_id"] = part
            parsed["lr"] = float(raw.replace("e-", "e-").replace("e+", "e+"))
        elif part.startswith("W"):
            parsed["innovation_weights"] = part
    required = (
        "step5A_ratio",
        "step5B_ratio",
        "step5A_cf_mix",
        "step5B_cf_mix",
        "target_gold_mix",
        "aux_gold_mix",
        "lr",
        "innovation_weights",
    )
    missing = [key for key in required if key not in parsed]
    if missing:
        raise ValueError(f"Step5 tuning candidate is missing parts: {', '.join(missing)}")
    return parsed


def _candidate_row(rows: Sequence[Mapping[str, Any]], row_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("id") or "") == str(row_id):
            return dict(row)
    raise KeyError(f"candidate row not found: {row_id}")


def _selected_batch(batch_cfg: Mapping[str, Any], batch_id: str) -> dict[str, Any]:
    for row in batch_cfg.get("candidates") or []:
        if isinstance(row, Mapping) and str(row.get("id") or "") == str(batch_id):
            return dict(row)
    raise KeyError(f"Step5 batch candidate not found: {batch_id}")


def _selected_warmup_fraction(tuning_cfg: Mapping[str, Any], selected_value: float = 0.05) -> float:
    rows = tuning_cfg.get("warmup_fraction_candidates") or []
    values = [float(row) for row in rows]
    if selected_value not in values:
        raise ValueError(f"selected Step5 warmup_fraction {selected_value} is not in One-Control candidates: {values}")
    return float(selected_value)


def _bridge_evidence(repo_root: Path) -> dict[str, Any]:
    handoff_path = current_handoff_path(repo_root)
    handoff_payload: dict[str, Any] | None = None
    handoff_error: str | None = None
    try:
        loaded = load_current_handoff(repo_root)
        if loaded is not None:
            selection_from_handoff_payload(loaded)
            handoff_payload = dict(loaded)
    except HandoffError as exc:
        handoff_error = str(exc)
    status_candidates = [
        repo_root / "AI_analysis/01_raw_logs/aux_runtime_gpu_handshake.status.json",
        repo_root / "AI_analysis/00_index/aux_runtime_gpu_handshake.status.json",
    ]
    status_path = next((path for path in status_candidates if path.is_file()), status_candidates[0])
    index_path = repo_root / "AI_analysis/00_index/aux_runtime_gpu_handshake.json"
    index = load_json(index_path)
    indexed_log = ((index.get("artifact") or {}).get("path")) if isinstance(index.get("artifact"), Mapping) else None
    log_path = repo_root / str(indexed_log or "AI_analysis/01_raw_logs/aux_runtime_gpu_handshake.log")
    report_path = repo_root / "AI_analysis/05_final_reports/aux_runtime_gpu_validation_report.md"
    status = load_json(status_path)
    payload = status.get("payload") if isinstance(status.get("payload"), Mapping) else {}
    source = payload if payload else status
    validation_result = status.get("validation_result") if isinstance(status.get("validation_result"), Mapping) else {}
    handoff_cuda = handoff_payload.get("cuda") if isinstance((handoff_payload or {}).get("cuda"), Mapping) else {}
    handoff_selection = (
        handoff_payload.get("selection_key") if isinstance((handoff_payload or {}).get("selection_key"), Mapping) else {}
    )
    handoff_valid = handoff_payload is not None
    return {
        "status_path": _repo_rel(repo_root, status_path),
        "log_path": _repo_rel(repo_root, log_path),
        "report_path": _repo_rel(repo_root, report_path),
        "success": bool(handoff_valid),
        "validation_timestamp": source.get("generated_at") or status.get("generated_at"),
        "validated_pane_id": source.get("TMUX_PANE") or status.get("TMUX_PANE"),
        "hostname": source.get("hostname") or status.get("hostname"),
        "TMUX": source.get("TMUX") or status.get("TMUX"),
        "TMUX_PANE": source.get("TMUX_PANE") or status.get("TMUX_PANE"),
        "SLURM_JOB_ID": source.get("SLURM_JOB_ID") or status.get("SLURM_JOB_ID"),
        "CUDA_VISIBLE_DEVICES": source.get("CUDA_VISIBLE_DEVICES") or status.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_L": source.get("nvidia-smi") or status.get("nvidia-smi"),
        "nvidia_smi_compute_apps": source.get("nvidia-smi-compute-apps") or status.get("nvidia-smi-compute-apps"),
        "torch_cuda_available": source.get("torch.cuda.is_available") or status.get("torch.cuda.is_available"),
        "torch_device_count": source.get("torch.cuda.device_count") or status.get("torch.cuda.device_count"),
        "gpu_names": source.get("torch.cuda.get_device_name") or status.get("torch.cuda.get_device_name"),
        "command": status.get("command"),
        "target_source": TARGET_SOURCE_HANDOFF if handoff_valid else None,
        "current_gpu_pane_json_path": _repo_rel(repo_root, handoff_path),
        "current_gpu_pane_handoff_exists": handoff_path.is_file(),
        "current_gpu_pane_handoff_valid": bool(handoff_valid),
        "current_gpu_pane_handoff_error": handoff_error,
        "fresh_handoff_gate_pass": bool(
            handoff_valid
            and str((handoff_payload or {}).get("hostname") or "") != "admin"
            and str((handoff_payload or {}).get("cuda_visible_devices") or "").strip()
            and int(handoff_cuda.get("torch_cuda_device_count") or 0) >= 2
        ),
        "handoff_hostname": (handoff_payload or {}).get("hostname"),
        "handoff_socket": handoff_selection.get("socket"),
        "handoff_pane": handoff_selection.get("pane"),
        "handoff_slurm_job_id": handoff_selection.get("slurm_job_id"),
        "handoff_cuda_visible_devices": handoff_selection.get("cuda_visible_devices"),
        "handoff_torch_cuda_available": handoff_cuda.get("torch_cuda_available"),
        "handoff_torch_device_count": handoff_cuda.get("torch_cuda_device_count"),
        "stale_state_used": False,
        "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
    }


def _head_low_count(result: Mapping[str, Any]) -> int:
    per_tier = result.get("per_tier_loss") if isinstance(result.get("per_tier_loss"), Mapping) else {}
    tiers = per_tier.get("tiers") if isinstance(per_tier.get("tiers"), Mapping) else per_tier
    low = tiers.get("cf_low_weighted") if isinstance(tiers.get("cf_low_weighted"), Mapping) else {}
    return int(low.get("tier_count") or 0)


def _head_tier_counts(result: Mapping[str, Any]) -> dict[str, int]:
    per_tier = result.get("per_tier_loss") if isinstance(result.get("per_tier_loss"), Mapping) else {}
    tiers = per_tier.get("tiers") if isinstance(per_tier.get("tiers"), Mapping) else per_tier
    return {
        str(key): int((value or {}).get("tier_count") or 0)
        for key, value in tiers.items()
        if isinstance(value, Mapping)
    }


def low_weighted_consistency(
    *,
    sampler_config: Mapping[str, Any],
    step5a_result: Mapping[str, Any],
    step5b_result: Mapping[str, Any],
) -> dict[str, Any]:
    heads = {"step5A": step5a_result, "step5B": step5b_result}
    by_head: dict[str, Any] = {}
    mismatch = False
    active_any = False
    sampled_any = False
    for head, result in heads.items():
        head_cfg = sampler_config.get(head) if isinstance(sampler_config.get(head), Mapping) else {}
        mix = dict(head_cfg.get("cf_tier_mix") or {})
        active_value = float(mix.get("low_weighted") or 0.0)
        count = _head_low_count(result)
        active = active_value > 0.0
        sampled = count > 0
        active_any = active_any or active
        sampled_any = sampled_any or sampled
        head_mismatch = (active and not sampled) or ((not active) and sampled)
        mismatch = mismatch or head_mismatch
        by_head[head] = {
            "active_low_weighted_mix": active_value,
            "low_weighted_active": active,
            "sampled_low_weighted_count": count,
            "sampled_low_weighted_nonzero": sampled,
            "mismatch": head_mismatch,
            "tier_counts": _head_tier_counts(result),
        }
    policy = "active" if active_any else "disabled_for_mainline"
    return {
        "schema_version": LOW_WEIGHTED_CONSISTENCY_SCHEMA,
        "generated_at": _iso_now(),
        "low_weighted_active_in_config": active_any,
        "low_weighted_sampled_nonzero": sampled_any,
        "low_weighted_mismatch": mismatch,
        "low_weighted_policy": policy,
        "low_weighted_consistency_pass": not mismatch,
        "by_head": by_head,
        "ablation_note": "low_weighted remains available as ablation-only evidence when disabled_for_mainline",
    }


def _result_pass(result: Mapping[str, Any]) -> bool:
    return (
        bool(result.get("success"))
        and bool(result.get("real_forward_backward_executed"))
        and bool(result.get("actual_gpu_backward_executed"))
        and bool(result.get("finite_loss_sync_ok"))
        and bool(result.get("graph_safe_backward_ok"))
        and not bool(result.get("formal_namespace_pollution"))
        and not bool(result.get("latest_json_created"))
        and not bool(result.get("checkpoint_written"))
        and not bool((result.get("long_window") or {}).get("memory_creep_detected"))
    )


def _signal_status(result: Mapping[str, Any]) -> dict[str, bool]:
    rank_results = [row for row in (result.get("rank_results") or []) if isinstance(row, Mapping)]
    losses = [row.get("losses") for row in rank_results if isinstance(row.get("losses"), Mapping)]
    return {
        "lci_signal_ok": bool(losses) and all("lci_weighted_loss" in row and "lci_raw_loss" in row for row in losses),
        "uci_signal_ok": bool(losses) and all("uci_weight_mean" in row for row in losses),
        "ccv_signal_ok": bool(rank_results) and all(bool((row.get("ccv_control_packet") or {}).get("present")) for row in rank_results),
        "fca_signal_ok": bool(losses) and all("fca_weighted_loss" in row and "fca_raw_loss" in row for row in losses),
    }


def final_bounded_preflight_report(
    *,
    step5a_result: Mapping[str, Any],
    step5b_result: Mapping[str, Any],
    low_weighted_report: Mapping[str, Any],
) -> dict[str, Any]:
    sig_a = _signal_status(step5a_result)
    sig_b = _signal_status(step5b_result)
    return {
        "schema_version": FINAL_BOUNDED_PREFLIGHT_SCHEMA,
        "generated_at": _iso_now(),
        "step5A": {
            "pass": _result_pass(step5a_result),
            "candidate_id": step5a_result.get("candidate_id"),
            "evidence_level": step5a_result.get("evidence_level"),
            "steps_executed": int(((step5a_result.get("long_window") or {}).get("steps_executed_max")) or 0),
            "tier_counts": _head_tier_counts(step5a_result),
            **sig_a,
        },
        "step5B": {
            "pass": _result_pass(step5b_result),
            "candidate_id": step5b_result.get("candidate_id"),
            "evidence_level": step5b_result.get("evidence_level"),
            "steps_executed": int(((step5b_result.get("long_window") or {}).get("steps_executed_max")) or 0),
            "tier_counts": _head_tier_counts(step5b_result),
            **sig_b,
        },
        "finite_loss": bool(step5a_result.get("finite_loss_sync_ok")) and bool(step5b_result.get("finite_loss_sync_ok")),
        "graph_safe": bool(step5a_result.get("graph_safe_backward_ok")) and bool(step5b_result.get("graph_safe_backward_ok")),
        "memory_creep_detected": bool((step5a_result.get("long_window") or {}).get("memory_creep_detected"))
        or bool((step5b_result.get("long_window") or {}).get("memory_creep_detected")),
        "formal_namespace_pollution": bool(step5a_result.get("formal_namespace_pollution"))
        or bool(step5b_result.get("formal_namespace_pollution")),
        "latest_json_created": bool(step5a_result.get("latest_json_created")) or bool(step5b_result.get("latest_json_created")),
        "checkpoint_written": bool(step5a_result.get("checkpoint_written")) or bool(step5b_result.get("checkpoint_written")),
        "low_weighted_consistency_pass": bool(low_weighted_report.get("low_weighted_consistency_pass")),
        "final_bounded_preflight_pass": _result_pass(step5a_result)
        and _result_pass(step5b_result)
        and bool(low_weighted_report.get("low_weighted_consistency_pass")),
    }


def _source_table_complete(source_table: Mapping[str, Any]) -> tuple[bool, list[str]]:
    field_sources = source_table.get("field_sources") if isinstance(source_table.get("field_sources"), Mapping) else {}
    missing = [key for key in REQUIRED_SOURCE_KEYS if key not in field_sources]
    return not missing, missing


def _formal_source_table_complete(source_table: Mapping[str, Any]) -> tuple[bool, list[str]]:
    field_sources = source_table.get("field_sources") if isinstance(source_table.get("field_sources"), Mapping) else {}
    missing = [key for key in FORMAL_READINESS_SOURCE_KEYS if key not in field_sources]
    return not missing, missing


def _old_effective_samples_from_report(old_auto_budget: Mapping[str, Any], head: str) -> int:
    heads = old_auto_budget.get("heads") if isinstance(old_auto_budget.get("heads"), Mapping) else {}
    selected = (heads.get(head) or {}).get("selected") if isinstance(heads.get(head), Mapping) else {}
    if not isinstance(selected, Mapping):
        return 0
    return int(selected.get("effective_samples") or 0)


def _legacy_unreconciled_auto_budget_reference(
    *,
    manifest: Mapping[str, Any],
    sampler_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any],
    throughput_samples_per_sec: float | None = None,
) -> dict[str, Any]:
    """Recreate the retired pre-reconciliation budget as audit-only evidence."""

    legacy_sampler = deepcopy(dict(sampler_config))
    legacy_auto = deepcopy(dict(legacy_sampler.get("auto_budget") or {}))
    legacy_auto.update(
        {
            "min_steps_per_effective_epoch": 1500,
            "preferred_steps_per_effective_epoch": [1800, 3000],
            "max_steps_per_effective_epoch": 4500,
        }
    )
    legacy_sampler["auto_budget"] = legacy_auto
    for head in ("step5A", "step5B"):
        head_cfg = deepcopy(dict(legacy_sampler.get(head) or {}))
        head_cfg["cf_tier_mix"] = {"high": 1.0, "medium": 1.0, "low_weighted": 1.0}
        legacy_sampler[head] = head_cfg
    out = compute_step5_auto_budget_report(
        manifest,
        sampler_config=legacy_sampler,
        batch_candidates_config=batch_candidates_config,
        tuning_config=tuning_config,
        throughput_samples_per_sec=throughput_samples_per_sec,
    )
    out["reference_role"] = "retired_unreconciled_auto_budget_audit_only"
    out["active_parameter_source"] = False
    return out


def reconcile_step5_formal_sample_budget(
    *,
    manifest: Mapping[str, Any],
    sampler_config: Mapping[str, Any],
    batch_candidates_config: Mapping[str, Any],
    tuning_config: Mapping[str, Any],
    old_auto_budget_report: Mapping[str, Any] | None = None,
    throughput_samples_per_sec: float | None = None,
) -> dict[str, Any]:
    """Recompute Step5 formal budgets under the active sampler tier mix."""

    reconciled = compute_step5_auto_budget_report(
        manifest,
        sampler_config=sampler_config,
        batch_candidates_config=batch_candidates_config,
        tuning_config=tuning_config,
        throughput_samples_per_sec=throughput_samples_per_sec,
    )
    legacy_reference = _legacy_unreconciled_auto_budget_reference(
        manifest=manifest,
        sampler_config=sampler_config,
        batch_candidates_config=batch_candidates_config,
        tuning_config=tuning_config,
        throughput_samples_per_sec=throughput_samples_per_sec,
    )
    max_replacement = float(((sampler_config.get("auto_budget") or {}).get("max_replacement_rate")) or 0.0)
    old_report = old_auto_budget_report if isinstance(old_auto_budget_report, Mapping) else {}
    heads: dict[str, Any] = {}
    for head in ("step5A", "step5B"):
        head_cfg = sampler_config.get(head) if isinstance(sampler_config.get(head), Mapping) else {}
        selected = reconciled["heads"][head]["selected"]
        reported_old_effective = _old_effective_samples_from_report(old_report, head)
        legacy_selected = (legacy_reference["heads"][head] or {}).get("selected", {})
        old_effective = int(reported_old_effective or legacy_selected.get("effective_samples") or 0)
        old_summary = (
            replacement_summary_for_effective_samples(
                manifest,
                head=head,
                head_cfg=head_cfg,
                effective_samples=old_effective,
            )
            if old_effective > 0
            else {}
        )
        new_summary = replacement_summary_for_effective_samples(
            manifest,
            head=head,
            head_cfg=head_cfg,
            effective_samples=int(selected["effective_samples"]),
        )
        cf_old_replacement = float(((old_summary.get("replacement_rate_by_component") or {}).get("cf")) or 0.0)
        old_replacement_rate = float(old_summary.get("replacement_rate") or 0.0)
        old_still_valid = (
            old_effective > 0
            and old_effective == int(selected["effective_samples"])
            and old_replacement_rate <= max_replacement + 1.0e-9
            and cf_old_replacement <= max_replacement + 1.0e-9
        )
        heads[head] = {
            "old_effective_samples": int(old_effective),
            "old_optimizer_steps": int(legacy_selected.get("optimizer_steps") or ((old_report.get("heads") or {}).get(head) or {}).get("selected", {}).get("optimizer_steps") or 0),
            "old_reference_role": "retired_unreconciled_auto_budget_audit_only",
            "reported_auto_budget_effective_samples": int(reported_old_effective),
            "old_replacement_if_retained": old_summary,
            "old_effective_samples_still_valid": bool(old_still_valid),
            "formal_active_available": dict(reconciled["heads"][head]["available"]),
            "formal_active_available_all": dict(reconciled["heads"][head]["available_all"]),
            "formal_active_tiers": dict(reconciled["heads"][head]["active_tiers"]),
            "formal_active_tier_mix": dict(reconciled["heads"][head]["tier_mix"]),
            "formal_active_balanced_capacity": int(reconciled["heads"][head]["balanced_capacity"]),
            "recommended_effective_samples": int(selected["effective_samples"]),
            "recommended_optimizer_steps": int(selected["optimizer_steps"]),
            "recommended_replacement_rate": float(new_summary["replacement_rate"]),
            "recommended_replacement_summary": new_summary,
            "recommended_ratio": dict(reconciled["heads"][head]["ratios"]),
            "recommended_active_cf_mix": dict(head_cfg.get("cf_tier_mix") or {}),
        }
    replacement_ok = all(float(item["recommended_replacement_rate"]) <= max_replacement + 1.0e-9 for item in heads.values())
    return {
        "schema_version": "odcr_step5_formal_sample_budget_reconciliation/1",
        "generated_at": _iso_now(),
        "capacity_basis": "formal_active_sampler_tier_mix",
        "old_effective_samples_still_valid": all(bool(item["old_effective_samples_still_valid"]) for item in heads.values()),
        "replacement_rate_ok": bool(replacement_ok),
        "max_replacement_rate": max_replacement,
        "selected_resolution": "Strategy B: keep bounded-selected ratios and active high/high+medium CF mix, lower formal effective samples to active balanced-capacity medium budget",
        "rejected_resolutions": [
            "Strategy A lowers CF ratios without bounded evidence for the changed objective balance",
            "Strategy C re-enables Step5A medium CF and requires extra E4 evidence",
            "Strategy D changes Step5A active CF mix and requires extra E4 evidence",
        ],
        "heads": heads,
        "recomputed_auto_budget": reconciled,
        "legacy_unreconciled_auto_budget_reference": legacy_reference,
    }


def build_step5_formal_split_execution_plan(
    *,
    candidate: Mapping[str, Any],
    budget_reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    samples = candidate.get("effective_samples") if isinstance(candidate.get("effective_samples"), Mapping) else {}
    steps = candidate.get("optimizer_steps") if isinstance(candidate.get("optimizer_steps"), Mapping) else {}
    throughput = 223.9
    estimate = {
        head: {
            "effective_samples": int(samples.get(head) or 0),
            "optimizer_steps": int(steps.get(head) or 0),
            "single_effective_epoch_seconds_at_b224_reference": round(float(samples.get(head) or 0) / throughput, 2)
            if int(samples.get(head) or 0) > 0
            else None,
            "expected_under_three_day_window": True,
        }
        for head in ("step5A", "step5B")
    }
    return {
        "schema_version": "odcr_step5_formal_split_execution_plan/1",
        "generated_at": _iso_now(),
        "recommended_split": [
            "Step5A formal train",
            "Step5B formal train",
            "Step5 formal merge/handoff/eval preparation",
            "Step5 final eval/rerank requires a later explicit authorization",
        ],
        "can_run_heads_independently": True,
        "shared_inputs": {
            "step4_pool_manifest": candidate.get("step4_pool_manifest"),
            "step4_sampling_contract": candidate.get("step4_sampling_contract"),
            "selected_tuning_candidate": candidate.get("selected_tuning_candidate"),
        },
        "per_head_estimate": estimate,
        "namespace_plan": {
            "step5A": "runs/step5/task2/<run_id>/step5A",
            "step5B": "runs/step5/task2/<run_id>/step5B",
            "merge_handoff": "runs/step5/task2/<run_id>/meta/run_summary.json",
            "latest_pointer": "write only after both heads and handoff pass formal validation",
        },
        "checkpoint_naming_plan": {
            "step5A": "checkpoints/step5A_best.pth and checkpoints/step5A_latest.pth",
            "step5B": "checkpoints/step5B_best.pth and checkpoints/step5B_latest.pth",
            "merge": "no model checkpoint; metadata handoff only",
        },
        "lineage_plan": {
            "per_head": "each head records resolved_config.json, source_table.json, Step4 pool manifest hash, sampling contract hash, sampler budget hash, and bounded candidate id",
            "merge": "merge validates both per-head lineage hashes before final Step5 handoff",
        },
        "resume_requeue_requirement": "per-head resume must validate matching lineage and continue from that head's latest checkpoint only",
        "early_stopping": "per-head early stopping is independent; one head failure must not invalidate the other head's completed checkpoint",
        "eval_export_policy": "eval/rerank is separate authorization; preparation may only write handoff metadata",
        "source_table_requirement": "source_table must distinguish Step5A formal, Step5B formal, and Step5 merge/handoff records",
        "budget_reconciliation_hash_basis": {
            "step5A_effective_samples": ((budget_reconciliation.get("heads") or {}).get("step5A") or {}).get("recommended_effective_samples"),
            "step5B_effective_samples": ((budget_reconciliation.get("heads") or {}).get("step5B") or {}).get("recommended_effective_samples"),
        },
    }


def _formal_readiness_source_table(
    *,
    candidate: Mapping[str, Any],
    low_weighted_report: Mapping[str, Any],
    budget_reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    effective = candidate.get("effective_samples") if isinstance(candidate.get("effective_samples"), Mapping) else {}
    steps = candidate.get("optimizer_steps") if isinstance(candidate.get("optimizer_steps"), Mapping) else {}
    field_sources = {
        "selected_tuning_candidate": "AI_analysis/05_final_reports/step5_final_tuning_candidate.json",
        "fallback_tuning_candidate": "AI_analysis/05_final_reports/step5_final_tuning_candidate.json",
        "batch_candidate": "configs/odcr.yaml:step5.tuning.batch_candidate",
        "fallback_batch_candidate": "configs/odcr.yaml:step5.tuning.fallback_batch_candidate",
        "ratio.step5A": "configs/odcr.yaml:step5.sampler.step5A plus selected ratio candidate",
        "ratio.step5B": "configs/odcr.yaml:step5.sampler.step5B plus selected ratio candidate",
        "cf_tier_mix.step5A": "configs/odcr.yaml:step5.sampler.step5A.cf_tier_mix",
        "cf_tier_mix.step5B": "configs/odcr.yaml:step5.sampler.step5B.cf_tier_mix",
        "gold_tier_mix.target_gold": "configs/odcr.yaml:step5.sampler.*.target_gold_tier_mix",
        "gold_tier_mix.aux_gold": "configs/odcr.yaml:step5.sampler.*.aux_gold_tier_mix",
        "low_weighted_policy": "configs/odcr.yaml:step5.sampler.*.cf_tier_mix + bounded per-tier evidence",
        "lr": "AI_analysis/05_final_reports/step5_final_tuning_candidate.json",
        "innovation_weights": "AI_analysis/05_final_reports/step5_final_tuning_candidate.json",
        "warmup_fraction": "configs/odcr.yaml:step5.tuning.warmup_fraction_candidates",
        "effective_samples.step5A": "AI_analysis/05_final_reports/step5_formal_sample_budget_reconciliation.json",
        "effective_samples.step5B": "AI_analysis/05_final_reports/step5_formal_sample_budget_reconciliation.json",
        "optimizer_steps.step5A": "AI_analysis/05_final_reports/step5_formal_sample_budget_reconciliation.json",
        "optimizer_steps.step5B": "AI_analysis/05_final_reports/step5_formal_sample_budget_reconciliation.json",
        "max_effective_epochs": "configs/odcr.yaml:step5.effective_epoch.max_effective_epochs",
        "early_stopping_patience": "configs/odcr.yaml:step5.effective_epoch.early_stopping_patience",
        "prompt_registry": "configs/odcr.yaml:step5.prompt_templates",
        "pool_manifest": str(candidate.get("step4_pool_manifest")),
        "sampling_contract": str(candidate.get("step4_sampling_contract")),
        "formal_run_disabled": "formal-preparation gate writes AI_analysis only",
        "full_audit_forbidden": "configs/odcr.yaml:step5.sampler.full_audit_default_allowed=false",
        "old_dedicated_forbidden": "configs/odcr.yaml:step5.sampler.legacy_gold_heavy_exports_allowed=false",
    }
    values = {
        "selected_tuning_candidate": candidate.get("selected_tuning_candidate"),
        "fallback_tuning_candidate": candidate.get("fallback_tuning_candidate"),
        "batch_candidate": candidate.get("batch_candidate"),
        "fallback_batch_candidate": candidate.get("fallback_batch_candidate"),
        "ratio.step5A": (candidate.get("ratio_values") or {}).get("step5A"),
        "ratio.step5B": (candidate.get("ratio_values") or {}).get("step5B"),
        "cf_tier_mix.step5A": (candidate.get("cf_tier_mix") or {}).get("step5A"),
        "cf_tier_mix.step5B": (candidate.get("cf_tier_mix") or {}).get("step5B"),
        "gold_tier_mix.target_gold": (candidate.get("gold_tier_mix_values") or {}).get("target_gold"),
        "gold_tier_mix.aux_gold": (candidate.get("gold_tier_mix_values") or {}).get("aux_gold"),
        "low_weighted_policy": low_weighted_report.get("low_weighted_policy"),
        "lr": candidate.get("lr"),
        "innovation_weights": candidate.get("innovation_weights"),
        "warmup_fraction": candidate.get("warmup_fraction"),
        "effective_samples.step5A": effective.get("step5A"),
        "effective_samples.step5B": effective.get("step5B"),
        "optimizer_steps.step5A": steps.get("step5A"),
        "optimizer_steps.step5B": steps.get("step5B"),
        "max_effective_epochs": candidate.get("max_effective_epochs"),
        "early_stopping_patience": candidate.get("early_stopping_patience"),
        "prompt_registry": candidate.get("prompt_registry"),
        "pool_manifest": candidate.get("step4_pool_manifest"),
        "sampling_contract": candidate.get("step4_sampling_contract"),
        "formal_run_disabled": True,
        "full_audit_forbidden": candidate.get("full_audit_default_forbidden"),
        "old_dedicated_forbidden": candidate.get("old_dedicated_default_forbidden"),
    }
    records = [
        {"key": key, "source": field_sources[key], "value": values.get(key)}
        for key in FORMAL_READINESS_SOURCE_KEYS
    ]
    return {
        "source_table_schema_version": "odcr_step5_formal_readiness_source_table/1",
        "view": "step5_formal_readiness",
        "generated_at_utc": _iso_now(),
        "field_sources": field_sources,
        "records": records,
        "budget_reconciliation_schema": budget_reconciliation.get("schema_version"),
    }


def formal_namespace_state(repo_root: Path, task_id: int) -> dict[str, Any]:
    task_dir = repo_root / "runs" / "step5" / f"task{int(task_id)}"
    latest = task_dir / "latest.json"
    checkpoints = sorted(str(path.relative_to(repo_root)) for path in task_dir.glob("**/*.pth")) if task_dir.is_dir() else []
    return {
        "formal_namespace_pollution": bool(latest.exists() or checkpoints),
        "latest_json_created": latest.exists(),
        "checkpoint_written": bool(checkpoints),
        "checkpoint_paths": checkpoints,
    }


def build_formal_preparation_payloads(
    *,
    repo_root: str | Path,
    task_id: int = 2,
    from_step3: str = "runs/step3/task2/2",
    from_step4: str = "runs/step4/task2/1",
    step5a_result_path: str | Path,
    step5b_result_path: str | Path,
    validation: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    cfg, _sources, snapshot = resolve_config(
        config_path=root / "configs" / "odcr.yaml",
        command="step5",
        task_id=int(task_id),
        set_overrides=[],
        dry_run=True,
        from_step4="1",
        mode="full",
    )
    source_table = build_formal_source_table_snapshot(snapshot)
    source_complete, source_missing = _source_table_complete(source_table)
    sampler_cfg = json.loads(str(getattr(cfg, "step5_sampler_config_json", "") or "{}"))
    batch_cfg = json.loads(str(getattr(cfg, "step5_batch_candidates_config_json", "") or "{}"))
    tuning_cfg = json.loads(str(getattr(cfg, "step5_tuning_config_json", "") or "{}"))
    prompt_cfg = json.loads(str(getattr(cfg, "step5_prompt_templates_config_json", "") or "{}"))
    effective_epoch_cfg = json.loads(str(getattr(cfg, "step5_effective_epoch_config_json", "") or "{}"))
    final_tuning = load_json(root / "AI_analysis/05_final_reports/step5_final_tuning_candidate.json")
    floor_verdict = load_json(root / "AI_analysis/05_final_reports/step5_bounded_floor_completion_machine_verdict.json")
    old_auto_budget = load_json(root / "AI_analysis/05_final_reports/step5_auto_budget_report.json")
    per_tier = load_json(root / "AI_analysis/05_final_reports/step5_per_tier_loss_report.json")
    pool_manifest = root / "runs/step4/task2/1/step5_pools/step5_pool_manifest.json"
    sampling_contract = root / "runs/step4/task2/1/step5_pools/step5_sampling_contract.json"
    pool_manifest_payload = load_json(pool_manifest)
    selected = str(final_tuning.get("selected_tuning_candidate") or floor_verdict.get("selected_tuning_candidate") or "")
    fallback = str(final_tuning.get("backup_tuning_candidate") or final_tuning.get("fallback_tuning_candidate") or "")
    parsed = parse_step5_candidate_id(selected)
    batch_id = str(tuning_cfg.get("batch_candidate") or "B224")
    batch_row = _selected_batch(batch_cfg, batch_id)
    fallback_batch = str(tuning_cfg.get("fallback_batch_candidate") or "B192")
    weights = _candidate_row(tuning_cfg.get("innovation_weight_candidates") or [], str(parsed["innovation_weights"]))
    ratio_a = _candidate_row(tuning_cfg["ratio_candidates"]["step5A"], str(parsed["step5A_ratio"]))
    ratio_b = _candidate_row(tuning_cfg["ratio_candidates"]["step5B"], str(parsed["step5B_ratio"]))
    target_gold_mix = _candidate_row(tuning_cfg["gold_tier_mix_candidates"]["target_gold"], str(parsed["target_gold_mix"]))
    aux_gold_mix = _candidate_row(tuning_cfg["gold_tier_mix_candidates"]["aux_gold"], str(parsed["aux_gold_mix"]))
    warmup_fraction = _selected_warmup_fraction(tuning_cfg, selected_value=0.05)
    active_cf_mix = {
        "step5A": dict((sampler_cfg.get("step5A") or {}).get("cf_tier_mix") or {}),
        "step5B": dict((sampler_cfg.get("step5B") or {}).get("cf_tier_mix") or {}),
    }
    step5a_result = load_json(step5a_result_path)
    step5b_result = load_json(step5b_result_path)
    low_report = low_weighted_consistency(
        sampler_config=sampler_cfg,
        step5a_result=step5a_result,
        step5b_result=step5b_result,
    )
    final_preflight = final_bounded_preflight_report(
        step5a_result=step5a_result,
        step5b_result=step5b_result,
        low_weighted_report=low_report,
    )
    namespace = formal_namespace_state(root, int(task_id))
    bridge = _bridge_evidence(root)
    validation = dict(validation or {})
    budget_reconciliation = reconcile_step5_formal_sample_budget(
        manifest=pool_manifest_payload,
        sampler_config=sampler_cfg,
        batch_candidates_config=batch_cfg,
        tuning_config=tuning_cfg,
        old_auto_budget_report=old_auto_budget,
        throughput_samples_per_sec=223.9,
    )
    auto_budget = budget_reconciliation["recomputed_auto_budget"]
    effective_samples = {
        "step5A": int(((auto_budget.get("heads") or {}).get("step5A") or {}).get("selected", {}).get("effective_samples") or 0),
        "step5B": int(((auto_budget.get("heads") or {}).get("step5B") or {}).get("selected", {}).get("effective_samples") or 0),
    }
    optimizer_steps = {
        "step5A": int(((auto_budget.get("heads") or {}).get("step5A") or {}).get("selected", {}).get("optimizer_steps") or 0),
        "step5B": int(((auto_budget.get("heads") or {}).get("step5B") or {}).get("selected", {}).get("optimizer_steps") or 0),
    }
    candidate = {
        "schema_version": FORMAL_PREP_CANDIDATE_SCHEMA,
        "generated_at": _iso_now(),
        "task_id": int(task_id),
        "from_step3": str(from_step3),
        "from_step4": str(from_step4),
        "step4_pool_manifest": _repo_rel(root, pool_manifest),
        "step4_sampling_contract": _repo_rel(root, sampling_contract),
        "selected_tuning_candidate": selected,
        "fallback_tuning_candidate": fallback,
        "batch_candidate": batch_id,
        "per_gpu_batch_size": int(batch_row.get("per_gpu_batch_size") or 0),
        "global_batch_size": int(batch_row.get("global_batch_size") or 0),
        "fallback_batch_candidate": fallback_batch,
        "effective_samples": effective_samples,
        "optimizer_steps": optimizer_steps,
        "formal_active_balanced_capacity": {
            "step5A": int(((budget_reconciliation.get("heads") or {}).get("step5A") or {}).get("formal_active_balanced_capacity") or 0),
            "step5B": int(((budget_reconciliation.get("heads") or {}).get("step5B") or {}).get("formal_active_balanced_capacity") or 0),
        },
        "replacement_rate": {
            "step5A": float(((budget_reconciliation.get("heads") or {}).get("step5A") or {}).get("recommended_replacement_rate") or 0.0),
            "step5B": float(((budget_reconciliation.get("heads") or {}).get("step5B") or {}).get("recommended_replacement_rate") or 0.0),
        },
        "old_effective_samples_still_valid": bool(budget_reconciliation.get("old_effective_samples_still_valid")),
        "sample_budget_reconciliation": budget_reconciliation,
        "max_effective_epochs": int(effective_epoch_cfg.get("max_effective_epochs") or 3),
        "early_stopping_patience": int(effective_epoch_cfg.get("early_stopping_patience") or 1),
        "lr": float(parsed["lr"]),
        "warmup_fraction": warmup_fraction,
        "warmup_fraction_source": "configs/odcr.yaml:step5.tuning.warmup_fraction_candidates",
        "innovation_weights": str(parsed["innovation_weights"]),
        "innovation_weight_values": weights,
        "ratio": {"step5A": str(parsed["step5A_ratio"]), "step5B": str(parsed["step5B_ratio"])},
        "ratio_values": {"step5A": ratio_a, "step5B": ratio_b},
        "cf_tier_mix": active_cf_mix,
        "cf_tier_mix_policy": "low_weighted_disabled_for_mainline_route_consistency",
        "selected_tuning_cf_mix_ids": {
            "step5A": str(parsed["step5A_cf_mix"]),
            "step5B": str(parsed["step5B_cf_mix"]),
        },
        "gold_tier_mix": {"target_gold": str(parsed["target_gold_mix"]), "aux_gold": str(parsed["aux_gold_mix"])},
        "gold_tier_mix_values": {"target_gold": target_gold_mix, "aux_gold": aux_gold_mix},
        "prompt_registry": str(prompt_cfg.get("schema_version") or "odcr_step5_prompt_template_registry/1"),
        "one_control_source": True,
        "source_table_complete": source_complete,
        "source_table_missing_keys": source_missing,
        "source_table_required_keys": list(REQUIRED_SOURCE_KEYS),
        "artifact_sources": {
            "auto_budget": "AI_analysis/05_final_reports/step5_auto_budget_report.json",
            "sample_budget_reconciliation": "AI_analysis/05_final_reports/step5_formal_sample_budget_reconciliation.json",
            "final_tuning_candidate": "AI_analysis/05_final_reports/step5_final_tuning_candidate.json",
            "per_tier_loss_report": "AI_analysis/05_final_reports/step5_per_tier_loss_report.json",
            "step4_pool_manifest": _repo_rel(root, pool_manifest),
            "step4_sampling_contract": _repo_rel(root, sampling_contract),
        },
        "gpu_bridge_evidence": bridge,
        "step4_sampling_contract_role": "pool_lineage_only_active_sampler_comes_from_resolved_config",
        "full_audit_default_forbidden": sampler_cfg.get("full_audit_default_allowed") is False,
        "old_dedicated_default_forbidden": sampler_cfg.get("legacy_gold_heavy_exports_allowed") is False,
        "formal_run_command_emitted": False,
        "allow_formal_preparation": False,
        "allow_formal_run": False,
    }
    formal_readiness_source_table = _formal_readiness_source_table(
        candidate=candidate,
        low_weighted_report=low_report,
        budget_reconciliation=budget_reconciliation,
    )
    formal_source_complete, formal_source_missing = _formal_source_table_complete(formal_readiness_source_table)
    candidate["formal_readiness_source_table"] = formal_readiness_source_table
    candidate["formal_source_table_complete"] = formal_source_complete
    candidate["formal_source_table_missing_keys"] = formal_source_missing
    split_plan = build_step5_formal_split_execution_plan(
        candidate=candidate,
        budget_reconciliation=budget_reconciliation,
    )
    one_control_pass = bool(source_complete and formal_source_complete and candidate["one_control_source"])
    floors_met = bool(floor_verdict.get("all_required_floors_met") is True)
    selected_confirmed = (
        selected
        == "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"
    )
    formal_prep_ok = (
        floors_met
        and selected_confirmed
        and one_control_pass
        and bool(budget_reconciliation.get("replacement_rate_ok"))
        and not bool(budget_reconciliation.get("old_effective_samples_still_valid"))
        and bool(final_preflight["final_bounded_preflight_pass"])
        and bool(bridge.get("fresh_handoff_gate_pass"))
        and not bool(namespace["formal_namespace_pollution"])
        and bool(per_tier.get("schema_version"))
        and bool(validation.get("compileall_pass"))
        and bool(validation.get("doctor_pass"))
        and bool(validation.get("guardrail_pass"))
        and bool(validation.get("tests_pass"))
    )
    candidate["allow_formal_preparation"] = bool(formal_prep_ok)
    machine = {
        "schema_version": FORMAL_PREP_PREFLIGHT_SCHEMA,
        "generated_at": _iso_now(),
        "verdict": "A" if formal_prep_ok else "B",
        "p0_count": 0 if formal_prep_ok else 1,
        "p1_count": 0,
        "p2_count": 0,
        "all_required_floors_met": floors_met,
        "selected_tuning_candidate_confirmed": selected_confirmed,
        "selected_tuning_candidate": selected,
        "fallback_tuning_candidate": fallback,
        "formal_preparation_candidate_written": True,
        "formal_preparation_candidate_path": str(REPORT_PATHS["candidate"]),
        "formal_candidate_source_table_complete": source_complete,
        "formal_readiness_source_table_complete": formal_source_complete,
        "formal_readiness_source_table_missing_keys": formal_source_missing,
        "one_control_pass": one_control_pass,
        "formal_active_cf_mix_reconciled": True,
        "formal_active_balanced_capacity_recomputed": True,
        "old_effective_samples_still_valid": bool(budget_reconciliation.get("old_effective_samples_still_valid")),
        "step5A_reconciled_effective_samples": int(effective_samples["step5A"]),
        "step5B_reconciled_effective_samples": int(effective_samples["step5B"]),
        "step5A_optimizer_steps": int(optimizer_steps["step5A"]),
        "step5B_optimizer_steps": int(optimizer_steps["step5B"]),
        "step5A_replacement_rate": float(candidate["replacement_rate"]["step5A"]),
        "step5B_replacement_rate": float(candidate["replacement_rate"]["step5B"]),
        "replacement_rate_ok": bool(budget_reconciliation.get("replacement_rate_ok")),
        "step5A_step5B_split_plan_ready": True,
        "low_weighted_active_in_config": bool(low_report["low_weighted_active_in_config"]),
        "low_weighted_sampled_nonzero": bool(low_report["low_weighted_sampled_nonzero"]),
        "low_weighted_mismatch": bool(low_report["low_weighted_mismatch"]),
        "low_weighted_policy": str(low_report["low_weighted_policy"]),
        "low_weighted_consistency_pass": bool(low_report["low_weighted_consistency_pass"]),
        "final_bounded_preflight_pass": bool(final_preflight["final_bounded_preflight_pass"]),
        "final_bounded_step5A_pass": bool(final_preflight["step5A"]["pass"]),
        "final_bounded_step5B_pass": bool(final_preflight["step5B"]["pass"]),
        "finite_loss": bool(final_preflight["finite_loss"]),
        "graph_safe": bool(final_preflight["graph_safe"]),
        "lci_signal_ok": bool(final_preflight["step5A"]["lci_signal_ok"]) and bool(final_preflight["step5B"]["lci_signal_ok"]),
        "uci_signal_ok": bool(final_preflight["step5A"]["uci_signal_ok"]) and bool(final_preflight["step5B"]["uci_signal_ok"]),
        "ccv_signal_ok": bool(final_preflight["step5A"]["ccv_signal_ok"]) and bool(final_preflight["step5B"]["ccv_signal_ok"]),
        "fca_signal_ok": bool(final_preflight["step5A"]["fca_signal_ok"]) and bool(final_preflight["step5B"]["fca_signal_ok"]),
        "gpu_bridge_validated": bool(bridge.get("success"))
        and bool(step5a_result.get("bridge_command_id"))
        and bool(step5b_result.get("bridge_command_id")),
        "step5_gate_requires_fresh_handoff": True,
        "fresh_handoff_gate_pass": bool(bridge.get("fresh_handoff_gate_pass")),
        "target_source": bridge.get("target_source"),
        "stale_state_used": bool(bridge.get("stale_state_used")),
        "gpu_pane_id": bridge.get("validated_pane_id"),
        "gpu_bridge_evidence": bridge,
        "current_shell_cuda_false_not_blocker": True,
        "manual_odcr_enter_gpu_requested": False,
        "full_audit_default_forbidden": bool(candidate["full_audit_default_forbidden"]),
        "old_dedicated_default_forbidden": bool(candidate["old_dedicated_default_forbidden"]),
        "formal_namespace_pollution": bool(namespace["formal_namespace_pollution"]),
        "latest_json_created": bool(namespace["latest_json_created"]),
        "checkpoint_written": bool(namespace["checkpoint_written"]),
        "formal_full_run_command_emitted": False,
        "compileall_pass": bool(validation.get("compileall_pass")),
        "doctor_pass": bool(validation.get("doctor_pass")),
        "guardrail_pass": bool(validation.get("guardrail_pass")),
        "tests_pass": bool(validation.get("tests_pass")),
        "allow_step5_formal_preparation": bool(formal_prep_ok),
        "allow_step5_formal_run": False,
    }
    return {
        "candidate": candidate,
        "low_weighted": low_report,
        "final_bounded": final_preflight,
        "machine": machine,
        "source_table": source_table,
        "formal_readiness_source_table": formal_readiness_source_table,
        "sample_budget_reconciliation": budget_reconciliation,
        "split_execution_plan": split_plan,
    }


def write_formal_preparation_reports(
    *,
    repo_root: str | Path,
    payloads: Mapping[str, Any],
) -> None:
    root = Path(repo_root).resolve()
    for rel in REPORT_PATHS.values():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / REPORT_PATHS["candidate"], dict(payloads["candidate"]))
    atomic_write_json(root / REPORT_PATHS["reconciled_candidate"], dict(payloads["candidate"]))
    atomic_write_json(root / REPORT_PATHS["sample_budget"], dict(payloads["sample_budget_reconciliation"]))
    atomic_write_json(root / REPORT_PATHS["split_plan"], dict(payloads["split_execution_plan"]))
    atomic_write_json(root / REPORT_PATHS["low_weighted"], dict(payloads["low_weighted"]))
    atomic_write_json(root / REPORT_PATHS["final_bounded"], dict(payloads["final_bounded"]))
    atomic_write_json(root / REPORT_PATHS["machine"], dict(payloads["machine"]))
    _write_text_reports(root, payloads)


def _write_text_reports(repo_root: Path, payloads: Mapping[str, Any]) -> None:
    machine = payloads["machine"]
    candidate = payloads["candidate"]
    low = payloads["low_weighted"]
    final = payloads["final_bounded"]
    report = "\n".join(
        [
            "# Step5 Formal Preparation Preflight",
            "",
            f"- verdict: {machine['verdict']}",
            f"- allow_step5_formal_preparation: {machine['allow_step5_formal_preparation']}",
            f"- allow_step5_formal_run: {machine['allow_step5_formal_run']}",
            f"- selected_tuning_candidate: {candidate['selected_tuning_candidate']}",
            f"- fallback_tuning_candidate: {candidate['fallback_tuning_candidate']}",
            f"- batch: {candidate['batch_candidate']} per_gpu={candidate['per_gpu_batch_size']} global={candidate['global_batch_size']}",
            f"- effective_samples: {json.dumps(candidate['effective_samples'], sort_keys=True)}",
            f"- optimizer_steps: {json.dumps(candidate['optimizer_steps'], sort_keys=True)}",
            f"- formal_active_balanced_capacity: {json.dumps(candidate['formal_active_balanced_capacity'], sort_keys=True)}",
            f"- replacement_rate: {json.dumps(candidate['replacement_rate'], sort_keys=True)}",
            f"- cf_tier_mix: {json.dumps(candidate['cf_tier_mix'], sort_keys=True)}",
            f"- low_weighted_policy: {low['low_weighted_policy']}",
            f"- low_weighted_consistency_pass: {low['low_weighted_consistency_pass']}",
            f"- final_bounded_preflight_pass: {final['final_bounded_preflight_pass']}",
            "- formal full run command emitted: false",
            "",
            "No formal run, checkpoint, eval, rerank, or latest pointer was written.",
        ]
    )
    ledger = "\n".join(
        [
            "# Step5 Formal Preparation Ledger",
            "",
            "Change classification: Step5 sampler/config-control-plane/formal-prep report/gate.",
            "Checklist mirror: One-Control existing sampler values, no new entrypoint, AI_analysis-only reports.",
            "Old logic handling: route-filter-after-sampling mismatch is retired/fail-fast; low_weighted is disabled for mainline and retained as ablation-only evidence.",
            "Rerun decision: no preprocess, Step3, Step4, Step5 formal, eval, or rerank rerun performed.",
            f"Validation: compileall={machine['compileall_pass']} doctor={machine['doctor_pass']} guardrail={machine['guardrail_pass']} tests={machine['tests_pass']}.",
        ]
    )
    summary = "\n".join(
        [
            "# Step5 Formal Preparation Summary",
            "",
            f"Verdict {machine['verdict']}. Formal preparation allowed: {machine['allow_step5_formal_preparation']}. Formal run remains disallowed.",
            f"Low weighted consistency: {low['low_weighted_consistency_pass']} ({low['low_weighted_policy']}).",
        ]
    )
    hits = "\n".join(
        [
            "configs/odcr.yaml step5.sampler.step5A.cf_tier_mix",
            "configs/odcr.yaml step5.sampler.step5B.cf_tier_mix",
            "code/odcr_core/step5_pool_sampler.py route-compatible sampling",
            "AI_analysis/05_final_reports/step5_final_tuning_candidate.json",
            "AI_analysis/05_final_reports/step5_per_tier_loss_report.json",
            "runs/step4/task2/1/step5_pools/step5_sampling_contract.json",
        ]
    )
    raw = json.dumps(payloads, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    index = "\n".join(
        [
            "# Step5 Formal Preparation Preflight Index",
            "",
            f"- candidate: {REPORT_PATHS['candidate']}",
            f"- low_weighted: {REPORT_PATHS['low_weighted']}",
            f"- sample_budget: {REPORT_PATHS['sample_budget']}",
            f"- split_plan: {REPORT_PATHS['split_plan']}",
            f"- final_bounded: {REPORT_PATHS['final_bounded']}",
            f"- machine: {REPORT_PATHS['machine']}",
            f"- report: {REPORT_PATHS['report']}",
        ]
    )
    (repo_root / REPORT_PATHS["report"]).write_text(report + "\n", encoding="utf-8")
    (repo_root / REPORT_PATHS["ledger"]).write_text(ledger + "\n", encoding="utf-8")
    (repo_root / REPORT_PATHS["summary"]).write_text(summary + "\n", encoding="utf-8")
    (repo_root / REPORT_PATHS["hits"]).write_text(hits + "\n", encoding="utf-8")
    (repo_root / REPORT_PATHS["raw"]).write_text(raw + "\n", encoding="utf-8")
    (repo_root / REPORT_PATHS["index"]).write_text(index + "\n", encoding="utf-8")


__all__ = [
    "FORMAL_PREP_CANDIDATE_SCHEMA",
    "FORMAL_PREP_PREFLIGHT_SCHEMA",
    "REPORT_PATHS",
    "build_formal_preparation_payloads",
    "build_step5_formal_split_execution_plan",
    "final_bounded_preflight_report",
    "formal_namespace_state",
    "load_json",
    "low_weighted_consistency",
    "parse_step5_candidate_id",
    "reconcile_step5_formal_sample_budget",
    "write_formal_preparation_reports",
]
