"""Bounded runtime probe admission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .command_registry import require_command
from odcr_core.evidence_level import (
    E3_GPU_TRANSPORT,
    E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
)

from .runtime_report import write_runtime_report
from .stage_dispatch import probe_command_name


def build_probe_payload(
    stage: str,
    task: int,
    *,
    handshake: dict[str, Any] | None = None,
    probe_result: dict[str, Any] | None = None,
    target_source: str | None = None,
    stale_state_used: bool = False,
    handoff_admission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = probe_command_name(stage, bounded=True)
    spec = require_command(name)
    stage_name = str(stage)
    required_fresh_handoff = stage_name in {"step5", "step5A", "step5B"}
    target_source_ok = (not required_fresh_handoff) or str(target_source or "") in {
        "current_gpu_pane_handoff",
        "live_discovery_cuda_probe",
        "cli_explicit",
    }
    cuda_ok = bool(
        handshake
        and handshake.get("torch.cuda.is_available")
        and int(handshake.get("torch.cuda.device_count") or 0) >= (2 if required_fresh_handoff else 1)
        and (not required_fresh_handoff or not str(handshake.get("hostname") or "").strip().lower().startswith("admin"))
        and (not required_fresh_handoff or bool(str(handshake.get("CUDA_VISIBLE_DEVICES") or "").strip()))
        and not bool(stale_state_used)
        and target_source_ok
    )
    real_e4 = bool(
        probe_result
        and probe_result.get("success") is True
        and probe_result.get("evidence_level") == E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION
        and probe_result.get("formal_entry_lifecycle") is True
        and probe_result.get("forward_executed") is True
        and probe_result.get("loss_backward_executed") is True
        and probe_result.get("optimizer_step_executed") is True
        and probe_result.get("preflight_executed") is True
        and probe_result.get("scratch_cleanup_status") == "pass"
        and probe_result.get("graph_tensor_audit_status") == "pass"
        and not list(probe_result.get("graph_scratch_before_ema") or [])
        and probe_result.get("ema_init_pass") is True
        and probe_result.get("ema_init_executed_in_E4") is True
        and probe_result.get("ddp_wrap_pass") is True
        and probe_result.get("first_train_step_pass") is True
        and probe_result.get("validation_pass_executed") is True
        and probe_result.get("validation_forward_pass") is True
        and probe_result.get("validation_loss_finite") is True
        and probe_result.get("validation_oom") is False
        and (
            stage_name != "step5A"
            or (
                probe_result.get("step5A_validation_scorer_only") is True
                and probe_result.get("flan_explainer_called_in_step5A_validation") is False
                and probe_result.get("out_logits_materialized_in_step5A_validation") is False
            )
        )
        and int(probe_result.get("valid_forward_micro_batch_size") or 10**9)
        <= int(probe_result.get("train_per_gpu_batch_size") or -1)
        and probe_result.get("all_trainable_grad_status") == "pass"
        and int(probe_result.get("trainable_param_count") or -1) == int(probe_result.get("grad_present_count") or -2)
        and int(probe_result.get("lora_trainable_count") or -1) == int(probe_result.get("lora_grad_present_count") or -2)
        and not list(probe_result.get("missing_grad_params") or [])
        and probe_result.get("real_forward_backward_executed") is True
        and (probe_result.get("real_data_batch_used") is True or probe_result.get("real_task_data_used") is True)
        and probe_result.get("real_ccv_packet_used") is True
        and probe_result.get("synthetic_batch_used_for_formal_gate") is not True
        and isinstance(probe_result.get("memory_truth"), dict)
        and (probe_result.get("memory_truth") or {}).get("reserved_is_diagnostic_only") is True
        and (probe_result.get("candidate_decision") or {}).get("reserved_memory_used_for_rejection") is False
        and cuda_ok
    )
    artifact_build_preflight = bool(
        probe_result
        and probe_result.get("success") is True
        and probe_result.get("artifact_build_only") is True
        and probe_result.get("artifact_build_preflight_pass") is True
        and int(probe_result.get("ddp_world_size") or 0) == 2
        and probe_result.get("rank0_rank1_cache_fingerprint_match") is True
        and probe_result.get("index_contract_audit_pass") is True
        and probe_result.get("formal_namespace_pollution") is False
        and probe_result.get("latest_json_created") is False
        and probe_result.get("checkpoint_written") is False
    )
    success = real_e4 if stage_name in {"step5A", "step5B"} else cuda_ok
    payload = {
        "schema_version": "odcr_runtime_bounded_probe/1",
        "command": spec.name,
        "stage": stage,
        "task": int(task),
        "requires_gpu": spec.requires_gpu,
        "requires_tmux": spec.requires_tmux,
        "writes_formal_runs": spec.writes_formal_runs,
        "cuda_handshake_ok": cuda_ok,
        "evidence_level": E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION if real_e4 else E3_GPU_TRANSPORT,
        "handshake_only": not bool(probe_result),
        "forward_backward_required_for_e4": stage_name in {"step5A", "step5B"},
        "backward_validation_claimed": bool(real_e4),
        "artifact_build_only_does_not_validate_backward": bool(artifact_build_preflight),
        "artifact_build_preflight_admitted": artifact_build_preflight,
        "current_real_batch_backward_e4_required": stage_name in {"step5A", "step5B"},
        "formal_entry_lifecycle_e4_required": stage_name in {"step5A", "step5B"},
        "formal_entry_validation_e4_required": stage_name in {"step5A", "step5B"},
        "fresh_gpu_pane_handoff_required": required_fresh_handoff,
        "target_source": target_source,
        "target_source_allowed_for_step5_gate": target_source_ok,
        "stale_state_used": bool(stale_state_used),
        "validated_hostname_non_admin": bool(
            handshake and not str(handshake.get("hostname") or "").strip().lower().startswith("admin")
        ),
        "validated_cuda_visible_devices": bool(handshake and str(handshake.get("CUDA_VISIBLE_DEVICES") or "").strip()),
        "validated_device_count_gte_2": bool(handshake and int(handshake.get("torch.cuda.device_count") or 0) >= 2),
        "synthetic_batch_used_for_formal_gate": bool(probe_result and probe_result.get("synthetic_batch_used_for_formal_gate") is True),
        "success": success,
        "formal_training": "not run",
    }
    if handoff_admission is not None:
        payload["handoff_admission"] = dict(handoff_admission)
    if probe_result is not None:
        payload["probe_result"] = dict(probe_result)
    return payload


def write_probe_report(
    stage: str,
    task: int,
    *,
    handshake: dict[str, Any] | None,
    probe_result: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
    target_source: str | None = None,
    stale_state_used: bool = False,
    handoff_admission: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_probe_payload(
        stage,
        task,
        handshake=handshake,
        probe_result=probe_result,
        target_source=target_source,
        stale_state_used=stale_state_used,
        handoff_admission=handoff_admission,
    )
    path = write_runtime_report(
        "aux_runtime_gpu_validation_report.md",
        payload,
        repo_root=repo_root,
        stage=stage,
        task=task,
        source="bounded_probe",
    )
    payload["report_path"] = str(path)
    return payload
