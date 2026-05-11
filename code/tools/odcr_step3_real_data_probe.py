#!/usr/bin/env python3
"""Internal Step3 real-data DDP smoke probe.

This is a bridge-only diagnostic tool. It is not an ODCR user entrypoint and it
must not run formal Step3 training or write formal checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
AI_ANALYSIS = REPO_ROOT / "AI_analysis"
PROBE_ROOTS = {
    "step3-ddp-smoke": AI_ANALYSIS / "06_probe_evidence" / "step3_ddp_smoke",
    "step3-performance-probe": AI_ANALYSIS / "06_probe_evidence" / "step3_performance_probe",
    "step3-short-pilot": AI_ANALYSIS / "06_probe_evidence" / "step3_short_pilot",
}
SCHEMA_VERSION = "odcr_step3_real_data_probe/2"
BRIDGE_STATUS_SCHEMA = "odcr_tmux_gpu_bridge_status/1.0"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class ProbeError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ProbePaths:
    run_dir: Path
    json_path: Path
    md_path: Path
    rank_dir: Path
    cache_root: Path
    stage_run_dir: Path
    bridge_log_path: Path | None
    bridge_status_path: Path | None


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_run_id(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ProbeError("run_id must be non-empty")
    if len(value) > 96:
        raise ProbeError("run_id is too long")
    if value in {".", ".."} or ".." in value or "/" in value or "\\" in value:
        raise ProbeError("run_id must not contain path traversal")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ProbeError("run_id contains unsafe characters")
    return value


def _stable_hash(payload: Any, *, length: int = 32) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _ensure_under_ai_analysis(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = AI_ANALYSIS.resolve()
    if resolved != root and root not in resolved.parents:
        raise ProbeError(f"probe output must live under AI_analysis: {resolved}")
    return resolved


def make_paths(
    run_id: str,
    *,
    mode: str,
    bridge_log_path: str | None,
    bridge_status_path: str | None,
) -> ProbePaths:
    safe = _safe_run_id(run_id)
    if mode not in PROBE_ROOTS:
        raise ProbeError(f"unsupported probe mode: {mode}")
    run_dir = _ensure_under_ai_analysis(PROBE_ROOTS[mode] / safe)
    stem = mode.replace("-", "_")
    return ProbePaths(
        run_dir=run_dir,
        json_path=run_dir / f"{stem}.json",
        md_path=run_dir / f"{stem}.md",
        rank_dir=run_dir / "ranks",
        cache_root=run_dir / "cache",
        stage_run_dir=run_dir / "probe_stage_run",
        bridge_log_path=_ensure_under_ai_analysis(Path(bridge_log_path)) if bridge_log_path else None,
        bridge_status_path=_ensure_under_ai_analysis(Path(bridge_status_path)) if bridge_status_path else None,
    )


def _append_bridge_log(paths: ProbePaths, run_id: str, *parts: Any) -> None:
    if paths.bridge_log_path is None:
        return
    paths.bridge_log_path.parent.mkdir(parents=True, exist_ok=True)
    with paths.bridge_log_path.open("a", encoding="utf-8") as fh:
        fh.write(" ".join(str(part) for part in parts) + "\n")


def _bridge_heartbeat(paths: ProbePaths, run_id: str, event: str, **payload: Any) -> None:
    if payload:
        _append_bridge_log(paths, run_id, "heartbeat", event, json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _append_bridge_log(paths, run_id, "heartbeat", event)


def _bridge_status(
    *,
    run_id: str,
    mode: str,
    success: bool,
    exit_code: int,
    stop_reason: str,
    started: float,
    metrics: Mapping[str, Any],
    first_result_seen: bool,
) -> dict[str, Any]:
    return {
        "schema_version": BRIDGE_STATUS_SCHEMA,
        "run_id": run_id,
        "kind": mode,
        "success": bool(success),
        "exit_code": int(exit_code),
        "elapsed_s": round(time.monotonic() - started, 3),
        "startup_timeout_s": 20,
        "first_result_timeout_s": 900,
        "hard_timeout_s": 900,
        "first_result_seen": bool(first_result_seen),
        "success_condition": f"{mode}_completed",
        "stop_reason": stop_reason,
        "metrics": dict(metrics),
    }


def _write_bridge_status(paths: ProbePaths, status: Mapping[str, Any]) -> None:
    if paths.bridge_status_path is not None:
        _write_json(paths.bridge_status_path, status)


def collect_host_slurm_cuda(*, include_torch: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
        "SLURM_CPUS_ON_NODE": os.environ.get("SLURM_CPUS_ON_NODE"),
        "nproc": None,
        "os_cpu_count": os.cpu_count(),
        "torch_cuda_is_available": None,
        "device_count": None,
        "device_names": [],
    }
    try:
        proc = subprocess.run(["nproc"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        if proc.returncode == 0:
            out["nproc"] = int(proc.stdout.strip())
    except Exception:
        out["nproc"] = None
    if include_torch:
        try:
            import torch

            out["torch_cuda_is_available"] = bool(torch.cuda.is_available())
            out["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            out["device_names"] = [torch.cuda.get_device_name(i) for i in range(int(out["device_count"] or 0))]
        except Exception as exc:
            out["torch_error"] = repr(exc)
    return out


def _safe_selector(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", candidate):
        raise ProbeError(f"unsafe {label} name: {candidate!r}")
    return candidate


def _resolve_config_with_overrides(task_id: int, set_overrides: Sequence[str]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from odcr_core.config_resolver import resolve_config

    cfg, sources, snapshot = resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=int(task_id),
        set_overrides=list(set_overrides),
        dry_run=True,
        run_id="auto",
        mode="full",
    )
    source_rows = [
        {"key": record.key, "value": record.value, "source": record.source}
        for record in sources
    ]
    return cfg, source_rows, snapshot


def resolve_one_control(
    task_id: int,
    *,
    mode: str = "step3-ddp-smoke",
    smoke_candidate: str | None = None,
    candidate_name: str | None = None,
    worker_profile: str | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    candidate_name = _safe_selector(candidate_name or smoke_candidate, label="candidate")
    worker_profile_name = _safe_selector(worker_profile, label="worker profile")
    set_overrides: list[str] = []
    selection: dict[str, Any] | None = None
    if candidate_name is not None:
        _base_cfg, _base_sources, base_snapshot = _resolve_config_with_overrides(task_id, [])
        del _base_cfg, _base_sources
        if mode == "step3-ddp-smoke":
            candidate_source = "step3.backup_profiles"
            candidate_pool = dict(base_snapshot.get("step3_backup_profiles") or {})
        else:
            candidate_source = "step3.performance_candidates.batch_ladder"
            candidate_pool = dict(((base_snapshot.get("step3_performance_candidates") or {}).get("batch_ladder")) or {})
        candidate = candidate_pool.get(candidate_name)
        selected_source = candidate_source
        if not isinstance(candidate, Mapping):
            for key, item in candidate_pool.items():
                if str((item or {}).get("candidate") or "") == candidate_name:
                    candidate = item
                    selected_source = f"{candidate_source}.{key}"
                    break
        if not isinstance(candidate, Mapping) and mode in {"step3-performance-probe", "step3-short-pilot"}:
            exploration = dict(base_snapshot.get("step3_exploration_profiles") or {})
            for key, item in exploration.items():
                if str(key) == candidate_name or str((item or {}).get("candidate") or "") == candidate_name:
                    candidate = item
                    selected_source = f"step3.exploration_profiles.{key}"
                    break
        if not isinstance(candidate, Mapping):
            exploration_names: list[str] = []
            if mode in {"step3-performance-probe", "step3-short-pilot"}:
                exploration = dict(base_snapshot.get("step3_exploration_profiles") or {})
                exploration_names = [
                    f"{key}/{(item or {}).get('candidate')}"
                    for key, item in exploration.items()
                    if isinstance(item, Mapping)
                ]
            available = ", ".join(sorted(str(key) for key in candidate_pool) + sorted(exploration_names)) or "<none>"
            raise ProbeError(
                f"candidate {candidate_name!r} is not defined in {candidate_source}; available: {available}"
            )
        batch_size = int(candidate.get("batch_size"))
        per_gpu_batch_size = int(candidate.get("per_gpu_batch_size"))
        set_overrides.extend(
            [
                f"step3.train.batch_size={batch_size}",
                f"step3.train.per_gpu_batch_size={per_gpu_batch_size}",
            ]
        )
        if bool(candidate.get("cross_rank_structured_gather", False)):
            set_overrides.append("step3.cross_rank_structured_gather.enabled=true")
            set_overrides.append(f"step3.cross_rank_structured_gather.mode={candidate.get('gather_mode') or 'local_gradient_context'}")
        if str(candidate.get("activation_checkpointing") or "off") == "selective":
            set_overrides.append("step3.memory.activation_checkpointing.enabled=true")
            set_overrides.append("step3.memory.activation_checkpointing.policy=selective")
        if candidate.get("profile_buffer_policy"):
            set_overrides.append(f"step3.memory.profile_buffer_policy={candidate.get('profile_buffer_policy')}")
        selection = {
            "name": candidate_name,
            "mode": mode,
            "source": f"configs/odcr.yaml:{selected_source if selected_source.endswith(candidate_name) or selected_source.startswith('step3.exploration_profiles.') else selected_source + '.' + candidate_name}",
            "batch_size": batch_size,
            "per_gpu_batch_size": per_gpu_batch_size,
            "batch_semantics_version": "odcr_no_accum/1",
            "batch_formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
            "grad_accum_removed": True,
            "cross_rank_structured_gather": bool(candidate.get("cross_rank_structured_gather", False)),
            "activation_checkpointing": str(candidate.get("activation_checkpointing") or "off"),
            "profile_buffer_policy": str(candidate.get("profile_buffer_policy") or "gpu_resident"),
            "task_profile_id": str(candidate.get("task_profile_id") or ""),
            "effective_pool_expected": int(candidate.get("effective_pool_expected") or batch_size),
            "formal_allowed": bool(candidate.get("formal_allowed", True)),
            "probe_only": bool(candidate.get("probe_only", False)),
            "role": str(candidate.get("role") or ""),
            "resolver_injected_overrides": list(set_overrides),
            "verdict": "selected from One-Control profile/candidate matrix; not arbitrary numeric CLI",
        }
    worker_selection: dict[str, Any] | None = None
    if worker_profile_name is not None:
        _base_cfg, _base_sources, worker_snapshot = _resolve_config_with_overrides(task_id, [])
        del _base_cfg, _base_sources
        workers = dict(worker_snapshot.get("step3_worker_profiles") or {})
        profile = workers.get(worker_profile_name)
        if not isinstance(profile, Mapping):
            available = ", ".join(sorted(str(key) for key in workers)) or "<none>"
            raise ProbeError(f"worker profile {worker_profile_name!r} is not defined; available: {available}")
        worker_overrides = [
            f"hardware.profiles.default.dataloader_num_workers_train={int(profile['train_workers_per_rank'])}",
            f"hardware.profiles.default.dataloader_prefetch_factor_train={int(profile['prefetch_factor'])}",
        ]
        set_overrides.extend(worker_overrides)
        worker_selection = {
            "name": worker_profile_name,
            "source": f"configs/odcr.yaml:step3.worker_profiles.{worker_profile_name}",
            "resolver_injected_overrides": worker_overrides,
        }
    cfg, source_rows, snapshot = _resolve_config_with_overrides(task_id, set_overrides)
    if selection is not None:
        key = "step3_probe_candidate_selection"
        snapshot[key] = selection
        source_rows.append({"key": key, "value": selection, "source": str(selection["source"])})
    if worker_selection is not None:
        snapshot["step3_worker_profile_selection"] = worker_selection
        source_rows.append(
            {
                "key": "step3_worker_profile_selection",
                "value": worker_selection,
                "source": str(worker_selection["source"]),
            }
        )
    return cfg, source_rows, snapshot


def worker_formula(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    hw = dict(snapshot.get("hardware") or {})
    world = int(hw.get("ddp_world_size") or 1)
    max_cpu = int(hw.get("max_parallel_cpu") or 0)
    num_proc = int(hw.get("num_proc") or 0)
    train_workers = int(hw.get("dataloader_num_workers_train") or 0)
    valid_workers = int(hw.get("dataloader_num_workers_valid") or 0)
    test_workers = int(hw.get("dataloader_num_workers_test") or 0)
    reserved = 2
    return {
        "semantics": "dataloader_num_workers_* are per rank; num_proc is rank0 datasets.map/tokenizer processes",
        "ddp_world_size": world,
        "max_parallel_cpu": max_cpu,
        "reserved_cpu": reserved,
        "num_proc": num_proc,
        "dataloader_workers_per_rank": {
            "train": train_workers,
            "valid": valid_workers,
            "test": test_workers,
        },
        "train_total_with_reserved_cpu": train_workers * world + reserved,
        "valid_total_with_reserved_cpu": valid_workers * world + reserved,
        "test_total_with_reserved_cpu": test_workers * world + reserved,
        "tokenization_total_with_reserved_cpu": num_proc + reserved,
        "train_safe": train_workers * world + reserved <= max_cpu,
        "valid_safe": valid_workers * world + reserved <= max_cpu,
        "test_safe": test_workers * world + reserved <= max_cpu,
        "tokenization_safe": num_proc + reserved <= max_cpu,
    }


def validate_cpu_budget(snapshot: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    formula = worker_formula(snapshot)
    blockers: list[str] = []
    if int(formula["max_parallel_cpu"]) != 12:
        blockers.append(f"max_parallel_cpu is {formula['max_parallel_cpu']}, expected 12")
    for key in ("train_safe", "valid_safe", "test_safe", "tokenization_safe"):
        if not bool(formula[key]):
            blockers.append(f"CPU worker formula failed: {key}")
    if strict and blockers:
        raise ProbeError("; ".join(blockers))
    return {"formula": formula, "blockers": blockers}


def _markdown_report(evidence: Mapping[str, Any]) -> str:
    verdict = evidence.get("verdict") or {}
    oc = evidence.get("one_control") or {}
    ddp = evidence.get("ddp") or {}
    loss = evidence.get("loss") or {}
    outputs = evidence.get("outputs") or {}
    lines = [
        f"# Step3 DDP Smoke Probe {evidence.get('run_id')}",
        "",
        f"- verdict: {verdict.get('status')}",
        f"- mode: {evidence.get('mode')}",
        f"- task: {evidence.get('task_id')} {evidence.get('source_domain')} -> {evidence.get('target_domain')}",
        f"- smoke candidate: {oc.get('smoke_candidate')}",
        f"- max_parallel_cpu: {oc.get('max_parallel_cpu')}",
        f"- batch formula: {oc.get('batch_formula_proof')}",
        f"- worker formula: {oc.get('dataloader_worker_formula')}",
        f"- DDP ranks: {ddp.get('rank_count')}, init: {ddp.get('init_status')}",
        f"- loss finite all-reduce: {loss.get('finite_all_reduce_result')}",
        f"- JSON evidence: {outputs.get('json_evidence_path')}",
        f"- MD evidence: {outputs.get('md_evidence_path')}",
        "",
        "## Blockers",
    ]
    blockers = verdict.get("blockers") or []
    lines.extend(f"- {item}" for item in blockers) if blockers else lines.append("- none")
    warnings = verdict.get("warnings") or []
    lines.append("")
    lines.append("## Warnings")
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- none")
    return "\n".join(lines) + "\n"


def _base_evidence(
    *,
    run_id: str,
    mode: str,
    task_id: int,
    cfg: Any,
    source_rows: list[dict[str, Any]],
    snapshot: Mapping[str, Any],
    paths: ProbePaths,
    include_torch: bool,
) -> dict[str, Any]:
    train = snapshot.get("train") or {}
    hw = snapshot.get("hardware") or {}
    task = snapshot.get("task") or {}
    formula = worker_formula(snapshot)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "timestamp": _now(),
        "task_id": int(task_id),
        "source_domain": task.get("source"),
        "target_domain": task.get("target"),
        "host_slurm_cuda": collect_host_slurm_cuda(include_torch=include_torch),
        "one_control": {
            "resolved_config_hash": _stable_hash(snapshot),
            "source_table_hash": _stable_hash(source_rows),
            "max_parallel_cpu": hw.get("max_parallel_cpu"),
            "train_precision": train.get("precision"),
            "precision": snapshot.get("step3_precision"),
            "optimizer": snapshot.get("step3_optimizer"),
            "tokenizer": snapshot.get("step3_tokenizer"),
            "evidence": snapshot.get("step3_evidence"),
            "scheduler": snapshot.get("step3_scheduler"),
            "valid_batch": snapshot.get("step3_eval"),
            "scenario": task.get("scenario"),
            "direction": task.get("direction"),
            "task_profile_id": task.get("task_profile_id"),
            "task_profile_key": task.get("task_profile_key"),
            "profile_isolation_hash": task.get("profile_isolation_hash"),
            "task_profile": snapshot.get("step3_task_profile"),
            "max_grad_norm": train.get("max_grad_norm"),
            "batch_size": train.get("batch_size"),
            "per_gpu_batch_size": train.get("per_gpu_batch_size"),
            "batch_semantics_version": train.get("batch_semantics_version"),
            "batch_formula": train.get("batch_formula"),
            "grad_accum_removed": train.get("grad_accum_removed"),
            "ddp_world_size": train.get("ddp_world_size"),
            "ddp": snapshot.get("step3_ddp"),
            "performance_candidates": snapshot.get("step3_performance_candidates"),
            "backup_profiles": snapshot.get("step3_backup_profiles"),
            "exploration_profiles": snapshot.get("step3_exploration_profiles"),
            "worker_profiles": snapshot.get("step3_worker_profiles"),
            "prefetcher": snapshot.get("step3_prefetcher"),
            "cross_rank_structured_gather": snapshot.get("step3_cross_rank_structured_gather"),
            "memory": snapshot.get("step3_memory"),
            "timing": snapshot.get("step3_timing"),
            "performance_probe": snapshot.get("step3_performance_probe"),
            "short_pilot": snapshot.get("step3_short_pilot"),
            "loss_semantics": snapshot.get("step3_loss_semantics"),
            "batch_formula_proof": (
                f"{train.get('batch_size')} == {train.get('per_gpu_batch_size')} * "
                f"{train.get('ddp_world_size')}"
            ),
            "dataloader_worker_formula": formula,
            "hardware_profile_json": cfg.hardware_profile_json,
            "probe_candidate": snapshot.get("step3_probe_candidate_selection"),
            "worker_profile": snapshot.get("step3_worker_profile_selection"),
        },
        "preprocess_upstream_evidence": {
            "status": "not_run",
            "preprocess_latest": {},
            "upstream_gate_hash": None,
            "manifest_metrics_verify_fingerprints": {},
            "profile_domain_artifact_fingerprints": {},
        },
        "tokenizer_cache": {
            "manifest_schema": None,
            "compatibility_result": "not_run",
            "cache_namespace": str(paths.cache_root),
            "formal_cache_written": False,
        },
        "ddp": {
            "rank_count": hw.get("ddp_world_size"),
            "local_ranks": [],
            "init_status": "not_run",
            "sampler": {},
            "forward_success_per_rank": {},
            "backward_success_per_rank": {},
            "find_unused_parameters": None,
            "unused_parameter_summary": None,
        },
        "loss": {
            "total_loss": None,
            "component_losses": {},
            "component_weights": {},
            "weighted_components": {},
            "component_participation": {},
            "component_finite_status": {},
            "global_component_finite_status": {},
            "finite_sync_summary": {},
            "component_graph_tied_zero_status": {},
            "loss_logging_summary": {},
            "loss_semantics": {},
            "finite_per_rank": {},
            "finite_all_reduce_result": None,
            "nan_inf_fail_fast_path": "not_run",
            "duplicate_loss_check_summary": "not_run",
            "graph_safety_preflight": {},
        },
        "memory_performance": {
            "startup_vs_steady_state": {
                "warmup_optimizer_steps": None,
                "measured_optimizer_steps": None,
                "candidate_name": None,
                "worker_profile": None,
            },
            "timing_fields": [
                "dataloader_next_wait",
                "h2d_prefetch_time",
                "compute_wait_for_prefetch",
                "forward_time",
                "loss_time",
                "backward_time",
                "optimizer_time",
                "scheduler_time",
                "sync_time",
                "logging_time",
                "total_step_time",
            ],
            "per_rank_memory": {},
            "step_wall_time_s": None,
            "dataloader_wait_s": None,
            "h2d_wait_s": None,
            "oom_status": "not_run",
            "profile_domain_artifacts": {},
        },
        "outputs": {
            "json_evidence_path": str(paths.json_path),
            "md_evidence_path": str(paths.md_path),
            "probe_meta_dir": str(paths.run_dir / "meta"),
            "probe_resolved_config_path": str(paths.run_dir / "meta" / "resolved_config.json"),
            "probe_source_table_path": str(paths.run_dir / "meta" / "source_table.json"),
            "formal_namespace_pollution_check": "not_run",
        },
        "verdict": {
            "status": "NOT_VERIFIED",
            "blockers": [],
            "warnings": [],
        },
    }


def _install_probe_env(cfg: Any, paths: ProbePaths) -> None:
    os.environ.update(
        {
            "ODCR_ROOT": str(REPO_ROOT),
            "ODCR_STAGE_RUN_DIR": str(paths.stage_run_dir),
            "ODCR_ITERATION_META_DIR": str(paths.run_dir / "meta"),
            "ODCR_MANIFEST_DIR": str(paths.run_dir / "meta"),
            "ODCR_LOG_DIR": str(paths.run_dir / "meta"),
            "ODCR_RESOLVED_DATA_DIR": str(Path(cfg.data_dir).resolve()),
            "ODCR_RESOLVED_MERGED_DIR": str(Path(cfg.merged_dir).resolve()),
            "ODCR_RESOLVED_MODELS_DIR": str(Path(cfg.models_dir).resolve()),
            "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(Path(cfg.step5_text_model).resolve()),
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(Path(cfg.sentence_embed_model).resolve()),
            "ODCR_RESOLVED_EMBED_DIM": str(int(cfg.embed_dim)),
            "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON": cfg.effective_training_payload_json,
            "ODCR_CONFIG_FIELD_SOURCES_JSON": cfg.config_field_sources_json,
            "ODCR_HARDWARE_PROFILE_JSON": cfg.hardware_profile_json,
            "ODCR_HARDWARE_PRESET": str(cfg.hardware_preset_id),
            "ODCR_RUNTIME_PRECISION_MODE": str(cfg.train_precision),
            "ODCR_RUNTIME_ALLOW_TF32": "1" if bool(getattr(cfg, "allow_tf32", False)) else "0",
            "ODCR_RUNTIME_AMP_AUTOCAST": "1" if bool(getattr(cfg, "amp_autocast", True)) else "0",
            "ODCR_RUNTIME_GRAD_SCALER": "1" if bool(getattr(cfg, "grad_scaler", False)) else "0",
            "ODCR_STEP3_TOKENIZER_MAX_LENGTH": str(int(getattr(cfg, "tokenizer_max_length", 0) or 0)),
            "ODCR_STEP3_EVIDENCE_MAX_LENGTH": str(int(getattr(cfg, "evidence_max_length", 0) or 0)),
            "OMP_NUM_THREADS": str(int(cfg.omp_num_threads)),
            "MKL_NUM_THREADS": str(int(cfg.mkl_num_threads)),
            "TOKENIZERS_PARALLELISM": "true" if bool(cfg.tokenizers_parallelism) else "false",
            "ODCR_LOG_STEP_LOSS_PARTS": "1",
        }
    )
    try:
        launcher_env = json.loads(str(cfg.launcher_env_effective_json or "{}"))
    except json.JSONDecodeError:
        launcher_env = {}
    if isinstance(launcher_env, dict) and launcher_env.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(launcher_env["CUDA_VISIBLE_DEVICES"])


def _write_probe_meta(snapshot: Mapping[str, Any], paths: ProbePaths) -> dict[str, str]:
    from odcr_core.manifests import write_resolved_config_artifacts

    meta_dir = paths.run_dir / "meta"
    resolved_config_path, source_table_path = write_resolved_config_artifacts(meta_dir, snapshot)
    return {
        "probe_meta_dir": str(meta_dir),
        "probe_resolved_config_path": str(resolved_config_path),
        "probe_source_table_path": str(source_table_path),
    }


def _free_tcp_init_method() -> str:
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _empty_timing(fields: Sequence[str]) -> dict[str, float]:
    return {str(field): 0.0 for field in fields}


def _add_timing(dst: dict[str, float], src: Mapping[str, Any]) -> None:
    for key in dst:
        try:
            dst[key] += float(src.get(key, 0.0) or 0.0)
        except Exception:
            dst[key] += 0.0


def _sync_cuda(device: Any) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        return


def _cuda_memory_snapshot(label: str, device: Any, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"label": label, "cuda_available": False, **dict(extra or {})}
    try:
        idx = int(device)
    except Exception:
        idx = torch.cuda.current_device()
    snapshot = {
        "label": label,
        "cuda_available": True,
        "device": idx,
        "allocated": int(torch.cuda.memory_allocated(idx)),
        "reserved": int(torch.cuda.memory_reserved(idx)),
        "max_allocated": int(torch.cuda.max_memory_allocated(idx)),
        "max_reserved": int(torch.cuda.max_memory_reserved(idx)),
    }
    snapshot.update(dict(extra or {}))
    try:
        snapshot["memory_summary"] = torch.cuda.memory_summary(device=idx, abbreviated=True)
    except Exception as exc:
        snapshot["memory_summary_error"] = repr(exc)
    return snapshot


def _oom_request_size_from_text(text: str) -> str | None:
    match = re.search(r"Tried to allocate ([0-9.]+) ([KMGTP]i?B)", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


def _series_stats(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(clean)
    p90_index = min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)
    return {
        "count": len(clean),
        "mean": float(sum(clean) / len(clean)),
        "median": float(ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0),
        "p90": float(ordered[p90_index]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _summarize_optimizer_records(records: Sequence[Mapping[str, Any]], timing_fields: Sequence[str]) -> dict[str, Any]:
    if not records:
        return {
            "optimizer_step_count": 0,
            "step_time_stats": _series_stats([]),
            "samples_per_second": None,
            "timing_mean_s": {field: None for field in timing_fields},
            "timing_median_s": {field: None for field in timing_fields},
            "dataloader_wait_ratio": None,
            "h2d_wait_ratio": None,
        }
    total_time = sum(float(item.get("total_optimizer_step_time", 0.0) or 0.0) for item in records)
    total_samples = sum(int(item.get("global_samples", 0) or 0) for item in records)
    timing_mean: dict[str, Any] = {}
    timing_median: dict[str, Any] = {}
    for field in timing_fields:
        stats = _series_stats([float((item.get("timing") or {}).get(field, 0.0) or 0.0) for item in records])
        timing_mean[str(field)] = stats["mean"]
        timing_median[str(field)] = stats["median"]
    dataloader_total = sum(float((item.get("timing") or {}).get("dataloader_next_wait", 0.0) or 0.0) for item in records)
    h2d_total = sum(float((item.get("timing") or {}).get("h2d_prefetch_time", 0.0) or 0.0) for item in records)
    return {
        "optimizer_step_count": len(records),
        "step_time_stats": _series_stats([float(item.get("total_optimizer_step_time", 0.0) or 0.0) for item in records]),
        "samples_per_second": float(total_samples / total_time) if total_time > 0 else None,
        "timing_mean_s": timing_mean,
        "timing_median_s": timing_median,
        "dataloader_wait_ratio": float(dataloader_total / total_time) if total_time > 0 else None,
        "h2d_wait_ratio": float(h2d_total / total_time) if total_time > 0 else None,
    }


def _rank_worker(rank: int, context: Mapping[str, Any]) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    from odcr_core.gather_schema import require_gathered_batch
    from executors import step3_train_core as s3

    original_read_csv = s3.pd.read_csv
    train_row_limit = int(context.get("train_row_limit") or 0)
    valid_row_limit = int(context.get("valid_row_limit") or 0)

    def _probe_read_csv(path: Any, *args: Any, **kwargs: Any) -> Any:
        name = Path(str(path)).name
        if name == "aug_train.csv" and train_row_limit > 0 and "nrows" not in kwargs:
            kwargs["nrows"] = train_row_limit
        elif name == "aug_valid.csv" and valid_row_limit > 0 and "nrows" not in kwargs:
            kwargs["nrows"] = valid_row_limit
        return original_read_csv(path, *args, **kwargs)

    s3.pd.read_csv = _probe_read_csv

    world_size = int(context["world_size"])
    local_rank = rank
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    memory_phase_snapshots: list[dict[str, Any]] = [
        _cuda_memory_snapshot("before model/buffer load", device),
        _cuda_memory_snapshot("after cache/prefetch init", device),
    ]
    backend = "nccl"
    dist.init_process_group(
        backend=backend,
        init_method=str(context["init_method"]),
        rank=rank,
        world_size=world_size,
    )
    rank_path = Path(str(context["rank_dir"])) / f"rank_{rank}.json"
    try:
        cache_root = Path(str(context["cache_root"]))
        s3.get_hf_cache_root = lambda task_idx: str(cache_root / f"task{int(task_idx)}" / "hf")
        args = SimpleNamespace(
            auxiliary=str(context["source"]),
            target=str(context["target"]),
            save_file=str(Path(str(context["run_dir"])) / "probe_only_no_formal_checkpoint.pth"),
            log_file=str(Path(str(context["run_dir"])) / f"rank_{rank}.log"),
            epochs=None,
            learning_rate=None,
            coef=None,
            emsize=None,
            nlayers=int(context["nlayers"]),
            nhead=None,
            nhid=None,
            dropout=None,
        )
        final_cfg, train_dataloader, valid_dataloader, model, sampler = s3.build_config_and_data_ddp(
            args,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
        tokenizer_cache_manifest_summary = json.loads(
            str(getattr(final_cfg, "step3_tokenizer_cache_manifest_json", "") or "{}")
        )
        upstream_evidence_summary = json.loads(
            str(getattr(final_cfg, "step3_upstream_evidence_json", "") or "{}")
        )
        model.train()
        underlying = s3.get_underlying_model(model)
        structured_weights = s3.step3_structured_loss_weights_from_config(final_cfg)
        loss_semantics = s3.step3_loss_semantics_from_config(final_cfg)
        s3.apply_step3_precision_backend(final_cfg)
        optimizer = s3.build_step3_optimizer(model, final_cfg)
        optimizer.zero_grad(set_to_none=True)
        mode = str(context.get("mode") or "step3-ddp-smoke")
        timing_fields = tuple(getattr(s3, "_STEP3_PREFETCH_TIMING_FIELDS"))
        warmup_optimizer_steps = max(0, int(context.get("warmup_optimizer_steps") or 0))
        measured_optimizer_steps = max(1, int(context.get("measured_optimizer_steps") or 1))
        max_wall_seconds = max(1, int(context.get("max_wall_seconds") or 180))
        max_epochs = max(1, int(context.get("max_epochs") or 1))
        max_optimizer_steps = max(1, int(context.get("max_optimizer_steps") or 1))
        validate_every_steps = max(1, int(context.get("validate_every_steps") or 1))
        validate_every_epoch = bool(context.get("validate_every_epoch", True))
        if mode == "step3-performance-probe":
            target_optimizer_steps = warmup_optimizer_steps + measured_optimizer_steps
        elif mode == "step3-short-pilot":
            target_optimizer_steps = max_optimizer_steps
        else:
            target_optimizer_steps = 1
        n_micro = max(1, len(train_dataloader))
        steps_per_epoch = max(1, n_micro)
        if mode == "step3-short-pilot":
            target_optimizer_steps = min(target_optimizer_steps, steps_per_epoch * max_epochs)
        total_steps_plan = max(1, target_optimizer_steps)
        sched = None
        scheduler_name = str(getattr(final_cfg, "lr_scheduler", "") or "")
        if scheduler_name == "warmup_cosine":
            ws_resolved, _warmup_ratio_logged = s3.resolve_warmup_steps(
                total_steps_plan,
                max(1, steps_per_epoch),
                explicit_steps=getattr(final_cfg, "odcr_warmup_steps", None),
                explicit_ratio=getattr(final_cfg, "odcr_warmup_ratio", None),
                warmup_epochs_fallback=float(getattr(final_cfg, "warmup_epochs", 0.0)),
            )
            lr_lambda = s3.warmup_cosine_multiplier_lambda(
                ws_resolved,
                total_steps_plan,
                float(getattr(final_cfg, "min_lr_ratio", 0.05)),
            )
            sched = s3.lr_sched.LambdaLR(optimizer, lr_lambda)

        prefetch_cfg = json.loads(str(getattr(final_cfg, "prefetcher_config_json", "") or "{}"))
        prefetch_enabled = bool(prefetch_cfg.get("enabled", True))
        memory_phase_snapshots.append(
            _cuda_memory_snapshot("after model/buffer load", device, extra={
                "activation_checkpointing": json.loads(str(getattr(final_cfg, "memory_config_json", "") or "{}")).get("activation_checkpointing", {}),
                "profile_buffer_policy": json.loads(str(getattr(final_cfg, "memory_config_json", "") or "{}")).get("profile_buffer_policy", "gpu_resident"),
                "profile_domain_buffer_footprint": s3.summarize_step3_profile_buffers(model, optimizer),
            })
        )
        first_batch_wait_s = None
        first_h2d_wait_s = None
        graph_preflight = None
        finite_sync = {}
        loss_bundle = None
        loss = None
        forward_out = None
        local_finite = False
        global_finite = False
        backward_success = False
        optimizer_step = False
        optimizer_steps_done = 0
        batch_steps_done = 0
        epoch_index = 0
        stop_reason = "target_optimizer_steps_reached"
        nonfinite_count = 0
        measured_records: list[dict[str, Any]] = []
        warmup_records: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        validation_records: list[dict[str, Any]] = []
        current_window_timing = _empty_timing(timing_fields)
        current_window_loss_values: list[float] = []
        current_window_samples = 0
        loop_started = time.perf_counter()

        def _new_iterator(epoch: int):
            if sampler is not None:
                sampler.set_epoch(epoch)
            base_loader = train_dataloader
            if prefetch_enabled:
                prefetcher = s3.Step3CUDAPrefetcher(
                    train_dataloader,
                    device=device,
                    non_blocking=bool(getattr(final_cfg, "non_blocking_h2d", True)),
                    enabled=True,
                    diagnostic_cpu_mode=bool(prefetch_cfg.get("diagnostic_cpu_mode", False)),
                )
                return iter(prefetcher), prefetcher
            return iter(base_loader), None

        iterator, prefetcher = _new_iterator(epoch_index)
        optimizer.zero_grad(set_to_none=True)
        while optimizer_steps_done < target_optimizer_steps:
            if time.perf_counter() - loop_started > max_wall_seconds:
                stop_reason = "max_wall_seconds_reached_before_target"
                break
            try:
                batch = next(iterator)
            except StopIteration:
                epoch_index += 1
                if mode == "step3-short-pilot" and epoch_index >= max_epochs:
                    stop_reason = "max_epochs_reached"
                    break
                iterator, prefetcher = _new_iterator(epoch_index)
                continue

            step_timing = _empty_timing(timing_fields)
            if prefetcher is not None:
                _add_timing(step_timing, prefetcher.last_timing)
            gather_t0 = time.perf_counter()
            g = require_gathered_batch(underlying.gather(batch, device))
            _sync_cuda(device)
            gather_elapsed = time.perf_counter() - gather_t0
            memory_phase_snapshots.append(_cuda_memory_snapshot("after batch H2D", device))
            if prefetcher is None:
                step_timing["h2d_prefetch_time"] += gather_elapsed
                if first_batch_wait_s is None:
                    first_batch_wait_s = 0.0
            if first_batch_wait_s is None:
                first_batch_wait_s = float(step_timing.get("dataloader_next_wait", 0.0) or 0.0)
            if first_h2d_wait_s is None:
                first_h2d_wait_s = float(step_timing.get("h2d_prefetch_time", 0.0) or 0.0)
            c_a = g.content_anchor_score
            s_a = g.style_anchor_score
            ce = g.content_evidence_ids
            se = g.style_evidence_ids
            dsa = g.domain_style_anchor_ids
            lsh = g.local_style_hint_ids
            pol = g.polarity_ids
            eq = g.evidence_quality_prior
            if any(value is None for value in (c_a, s_a, ce, se, dsa, lsh, pol, eq)):
                raise RuntimeError("Step3 probe gather missing canonical evidence tensors.")
            sync_t0 = time.perf_counter()
            with s3.odcr_cuda_bf16_autocast():
                forward_t0 = time.perf_counter()
                forward_out = model(
                    g.user_idx,
                    g.item_idx,
                    g.tgt_input,
                    g.domain_idx,
                    content_anchor=c_a,
                    style_anchor=s_a,
                    content_evidence_ids=ce,
                    style_evidence_ids=se,
                    domain_style_anchor_ids=dsa,
                    local_style_hint_ids=lsh,
                    polarity_ids=pol,
                    evidence_quality_prior=eq,
                )
                _sync_cuda(device)
                memory_phase_snapshots.append(_cuda_memory_snapshot("after forward", device))
                step_timing["forward_time"] += time.perf_counter() - forward_t0
                loss_t0 = time.perf_counter()
                loss_bundle = s3.compose_step3_loss_from_forward_output(
                    forward_output=forward_out,
                    batch=g,
                    final_cfg=final_cfg,
                    weights=structured_weights,
                    semantics=loss_semantics,
                )
                loss = loss_bundle.total_loss
                finite_sync = s3.step3_sync_loss_bundle_finite_status(loss_bundle, world_size=world_size)
                local_finite = bool(finite_sync["local_total_finite"])
                global_finite = bool(finite_sync["global_total_finite"])
                _sync_cuda(device)
                memory_phase_snapshots.append(_cuda_memory_snapshot("after loss composition", device, extra={
                    "structured_loss_temporary_tensor_summary": (loss_bundle.logging_summary or {}).get("cross_rank_gather", {})
                }))
                step_timing["loss_time"] += time.perf_counter() - loss_t0
                if graph_preflight is None:
                    graph_preflight = s3.validate_step3_graph_safety_preflight(
                        forward_output=forward_out,
                        loss_bundle=loss_bundle,
                        underlying_model=underlying,
                        ctx=f"step3/{mode}",
                    )
                if global_finite:
                    backward_t0 = time.perf_counter()
                    loss.backward()
                    _sync_cuda(device)
                    memory_phase_snapshots.append(_cuda_memory_snapshot("after backward", device))
                    step_timing["backward_time"] += time.perf_counter() - backward_t0
                    backward_success = True
                else:
                    nonfinite_count += 1
            step_timing["sync_time"] += time.perf_counter() - sync_t0
            batch_steps_done += 1
            current_window_samples += int(g.user_idx.shape[0]) * world_size
            current_window_loss_values.append(float(loss.detach().item()) if loss is not None else float("nan"))
            _add_timing(current_window_timing, step_timing)
            if global_finite:
                optimizer_t0 = time.perf_counter()
                nn.utils.clip_grad_norm_(s3.step3_trainable_parameters(model), float(final_cfg.max_grad_norm))
                optimizer.step()
                _sync_cuda(device)
                memory_phase_snapshots.append(_cuda_memory_snapshot("after optimizer.step", device))
                optimizer.zero_grad(set_to_none=True)
                _sync_cuda(device)
                memory_phase_snapshots.append(_cuda_memory_snapshot("after zero_grad", device))
                current_window_timing["optimizer_time"] += time.perf_counter() - optimizer_t0
                optimizer_step = True
                if sched is not None:
                    scheduler_t0 = time.perf_counter()
                    sched.step()
                    current_window_timing["scheduler_time"] += time.perf_counter() - scheduler_t0
            else:
                optimizer.zero_grad(set_to_none=True)
                stop_reason = "nonfinite_loss_global_sync"
                break
            optimizer_steps_done += 1
            current_window_timing["total_step_time"] = sum(
                float(current_window_timing.get(field, 0.0) or 0.0)
                for field in timing_fields
                if field != "total_step_time"
            )
            record = {
                "optimizer_step": optimizer_steps_done,
                "epoch_index": epoch_index,
                "batch_steps": 1,
                "batch_semantics_version": "odcr_no_accum/1",
                "global_samples": int(current_window_samples),
                "loss": float(sum(current_window_loss_values) / max(1, len(current_window_loss_values))),
                "timing": dict(current_window_timing),
                "total_optimizer_step_time": float(current_window_timing["total_step_time"]),
            }
            all_records.append(record)
            if mode == "step3-performance-probe" and optimizer_steps_done <= warmup_optimizer_steps:
                warmup_records.append(record)
            else:
                measured_records.append(record)
            if mode == "step3-short-pilot" and optimizer_steps_done % validate_every_steps == 0:
                valid_t0 = time.perf_counter()
                valid_loss_sum, valid_n_samples = s3.validModel_sum_batches(model, valid_dataloader, device)
                v_stat = torch.tensor([valid_loss_sum, float(valid_n_samples)], dtype=torch.double, device=device)
                dist.all_reduce(v_stat, op=dist.ReduceOp.SUM)
                validation_records.append(
                    {
                        "optimizer_step": optimizer_steps_done,
                        "epoch_index": epoch_index,
                        "valid_loss": float(v_stat[0] / v_stat[1]) if v_stat[1] > 0 else None,
                        "valid_samples": int(v_stat[1].item()) if v_stat[1] > 0 else 0,
                        "elapsed_s": float(time.perf_counter() - valid_t0),
                        "trigger": "validate_every_steps",
                    }
                )
            current_window_timing = _empty_timing(timing_fields)
            current_window_loss_values = []
            current_window_samples = 0

        if mode == "step3-short-pilot" and validate_every_epoch and validation_records:
            validation_records[-1]["validate_every_epoch"] = bool(validate_every_epoch)
        completed_target = optimizer_steps_done >= target_optimizer_steps or (
            mode == "step3-short-pilot" and stop_reason == "max_epochs_reached"
        )
        if mode != "step3-short-pilot" and optimizer_steps_done < target_optimizer_steps and stop_reason == "target_optimizer_steps_reached":
            stop_reason = "dataloader_exhausted_before_target"
        allocated = torch.cuda.memory_allocated(local_rank)
        reserved = torch.cuda.memory_reserved(local_rank)
        peak = torch.cuda.max_memory_allocated(local_rank)
        torch.cuda.reset_peak_memory_stats(local_rank)
        result = {
            "rank": rank,
            "local_rank": local_rank,
            "probe_run_id": str(context.get("run_id") or ""),
            "candidate_name": str(context.get("candidate_name") or ""),
            "forward_success": True,
            "backward_success": backward_success,
            "optimizer_step": optimizer_step,
            "optimizer_steps_completed": optimizer_steps_done,
            "target_optimizer_steps": target_optimizer_steps,
            "warmup_optimizer_steps": warmup_optimizer_steps,
            "measured_optimizer_steps": measured_optimizer_steps,
            "measured_optimizer_steps_completed": len(measured_records),
            "batch_steps_completed": batch_steps_done,
            "batch_semantics_version": "odcr_no_accum/1",
            "mode_stop_reason": stop_reason,
            "completed_target": completed_target,
            "nonfinite_count": nonfinite_count,
            "local_finite": local_finite,
            "global_finite": global_finite,
            "total_loss": float(loss.detach().item()) if loss is not None else None,
            "component_losses": {
                name: float(value.detach().item()) for name, value in sorted((loss_bundle.components if loss_bundle else {}).items())
            },
            "component_weights": dict(loss_bundle.weights) if loss_bundle else {},
            "weighted_components": {
                name: float(value.detach().item())
                for name, value in sorted((loss_bundle.weighted_components if loss_bundle else {}).items())
            },
            "component_participation": dict(loss_bundle.participates_in_total) if loss_bundle else {},
            "component_finite_status": dict(loss_bundle.finite_status) if loss_bundle else {},
            "global_component_finite_status": dict((finite_sync or {}).get("global_component_finite_status", {})),
            "finite_sync_summary": dict(finite_sync or {}),
            "component_graph_tied_zero_status": dict(loss_bundle.graph_tied_zero_status) if loss_bundle else {},
            "duplicate_loss_check_summary": dict(loss_bundle.duplicate_loss_check_summary) if loss_bundle else {},
            "loss_logging_summary": dict(loss_bundle.logging_summary) if loss_bundle else {},
            "loss_semantics": dataclasses.asdict(loss_semantics),
            "graph_safety_preflight": graph_preflight or {},
            "optimizer_step_records": all_records,
            "warmup_summary": _summarize_optimizer_records(warmup_records, timing_fields),
            "measured_summary": _summarize_optimizer_records(measured_records, timing_fields),
            "validation_records": validation_records,
            "profile_domain_artifacts": s3.summarize_step3_profile_buffers(model, optimizer),
            "tokenizer_cache_manifest_summary": tokenizer_cache_manifest_summary,
            "step3_upstream_evidence_summary": upstream_evidence_summary,
            "find_unused_parameters": bool(final_cfg.ddp_find_unused_parameters),
            "static_graph": bool(getattr(final_cfg, "ddp_static_graph", False)),
            "graph_safety_preflight_enabled": bool(getattr(final_cfg, "ddp_graph_safety_preflight", True)),
            "batch_size": int(g.user_idx.shape[0]),
            "dataloader_wait_s": float(first_batch_wait_s or 0.0),
            "h2d_wait_s": float(first_h2d_wait_s or 0.0),
            "step_wall_time_s": float((measured_records[-1]["total_optimizer_step_time"] if measured_records else all_records[-1]["total_optimizer_step_time"]) if all_records else 0.0),
            "memory": {
                "allocated": int(allocated),
                "reserved": int(reserved),
                "peak_allocated": int(peak),
                "phase_snapshots": memory_phase_snapshots,
                "fragmentation_hint": "compare reserved_minus_allocated and max_reserved across phase_snapshots",
            },
            "sampler": {
                "drop_last": bool(getattr(train_dataloader, "drop_last", False)),
                "shuffle": True,
            },
        }
        _write_json(rank_path, result)
    except Exception as exc:
        error_text = repr(exc)
        is_oom = "out of memory" in error_text.lower()
        _write_json(
            rank_path,
            {
                "rank": rank,
                "probe_run_id": str(context.get("run_id") or ""),
                "candidate_name": str(context.get("candidate_name") or ""),
                "forward_success": False,
                "backward_success": False,
                "error": error_text,
                "oom": bool(is_oom),
                "oom_allocation_request": _oom_request_size_from_text(error_text),
                "memory": {
                    "phase_snapshots": memory_phase_snapshots if "memory_phase_snapshots" in locals() else [],
                    "fragmentation_hint": "OOM likely fragmentation if reserved is much larger than allocated before failure.",
                },
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_real_probe(args: argparse.Namespace, paths: ProbePaths, evidence: dict[str, Any], cfg: Any) -> dict[str, Any]:
    import torch
    import torch.multiprocessing as mp

    cuda = collect_host_slurm_cuda(include_torch=True)
    evidence["host_slurm_cuda"] = cuda
    if not bool(cuda.get("torch_cuda_is_available")) or int(cuda.get("device_count") or 0) <= 0:
        evidence["verdict"]["status"] = "NOT_VERIFIED"
        evidence["verdict"]["blockers"].append(
            "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` in this same tmux to enter the GPU node, then rerun the probe."
        )
        return evidence
    world_size = int(evidence["one_control"]["ddp_world_size"])
    if int(cuda.get("device_count") or 0) < world_size:
        evidence["verdict"]["status"] = "NOT_VERIFIED"
        evidence["verdict"]["blockers"].append(
            f"CUDA device_count={cuda.get('device_count')} is less than ddp_world_size={world_size}"
        )
        return evidence
    per_gpu_batch = int(evidence["one_control"].get("per_gpu_batch_size") or 1)
    max_batches = int(getattr(args, "max_batches", 1) or 1)
    if args.mode == "step3-performance-probe":
        perf_ladder = (((evidence.get("one_control") or {}).get("performance_candidates") or {}).get("batch_ladder") or {}).values()
        per_candidate_limits = []
        valid_candidate_limits = []
        for candidate in perf_ladder:
            try:
                cand_per_gpu = int(candidate.get("per_gpu_batch_size"))
            except Exception:
                continue
            cand_required_steps = int(args.warmup_optimizer_steps) + int(args.measured_optimizer_steps)
            per_candidate_limits.append(cand_per_gpu * world_size * cand_required_steps * 2)
            valid_candidate_limits.append(cand_per_gpu * world_size)
        required_batch_steps = int(args.warmup_optimizer_steps) + int(args.measured_optimizer_steps)
        train_row_limit = max(2048, *(int(value) for value in per_candidate_limits)) if per_candidate_limits else max(2048, per_gpu_batch * world_size * required_batch_steps * 2)
        valid_row_limit = max(256, *(int(value) for value in valid_candidate_limits)) if valid_candidate_limits else max(256, per_gpu_batch * world_size)
    elif args.mode == "step3-short-pilot":
        # Short pilots are still governed bridge probes.  They use a real-data
        # capped pilot epoch so the bridge does not become a formal long train.
        perf_ladder = (((evidence.get("one_control") or {}).get("performance_candidates") or {}).get("batch_ladder") or {}).values()
        pilot_epoch_steps = min(int(args.max_optimizer_steps), int(args.warmup_optimizer_steps) + int(args.measured_optimizer_steps))
        per_candidate_limits = []
        valid_candidate_limits = []
        for candidate in perf_ladder:
            try:
                cand_per_gpu = int(candidate.get("per_gpu_batch_size"))
            except Exception:
                continue
            cand_required_steps = max(1, pilot_epoch_steps * int(args.max_epochs))
            per_candidate_limits.append(cand_per_gpu * world_size * cand_required_steps * 2)
            valid_candidate_limits.append(cand_per_gpu * world_size)
        required_batch_steps = max(1, pilot_epoch_steps * int(args.max_epochs))
        train_row_limit = max(2048, *(int(value) for value in per_candidate_limits)) if per_candidate_limits else max(2048, per_gpu_batch * world_size * required_batch_steps * 2)
        valid_row_limit = max(256, *(int(value) for value in valid_candidate_limits)) if valid_candidate_limits else max(256, per_gpu_batch * world_size)
    else:
        required_batch_steps = max(1, max_batches)
        train_row_limit = max(2048, per_gpu_batch * world_size * required_batch_steps * 2)
        valid_row_limit = max(256, per_gpu_batch * world_size)
    evidence["outputs"]["short_window_read_csv"] = {
        "train_rows": int(train_row_limit),
        "valid_rows": int(valid_row_limit),
        "source": "real current merged CSV rows via probe-only nrows cap",
    }
    _install_probe_env(cfg, paths)
    common_probe_cache_root = paths.cache_root
    if args.mode in {"step3-performance-probe", "step3-short-pilot"}:
        common_probe_cache_root = _ensure_under_ai_analysis(
            AI_ANALYSIS / "06_probe_evidence" / "step3_tokenizer_cache_probe" / f"task{int(args.task_id)}_s2bp"
        )
    evidence["tokenizer_cache"]["cache_namespace"] = str(common_probe_cache_root)
    paths.rank_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "run_id": args.run_id,
        "candidate_name": getattr(args, "candidate_name", None) or getattr(args, "smoke_candidate", None),
        "world_size": world_size,
        "init_method": _free_tcp_init_method(),
        "rank_dir": str(paths.rank_dir),
        "cache_root": str(common_probe_cache_root),
        "run_dir": str(paths.run_dir),
        "source": evidence["source_domain"],
        "target": evidence["target_domain"],
        "nlayers": int(cfg.nlayers),
        "allow_optimizer_step": bool(args.mode == "step3-performance-probe" or args.mode == "step3-short-pilot" or args.allow_one_optimizer_step),
        "train_row_limit": int(train_row_limit),
        "valid_row_limit": int(valid_row_limit),
        "mode": args.mode,
        "warmup_optimizer_steps": int(args.warmup_optimizer_steps),
        "measured_optimizer_steps": int(args.measured_optimizer_steps),
        "max_wall_seconds": int(args.max_wall_seconds),
        "max_epochs": int(args.max_epochs),
        "max_optimizer_steps": int(args.max_optimizer_steps),
        "validate_every_steps": int(args.validate_every_steps),
        "validate_every_epoch": bool(args.validate_every_epoch),
    }
    evidence["ddp"]["init_status"] = "starting"
    _bridge_heartbeat(paths, args.run_id, "warmup_started", warmup_optimizer_steps=int(args.warmup_optimizer_steps))
    if int(args.measured_optimizer_steps) > 0:
        _bridge_heartbeat(paths, args.run_id, "measured_started", measured_optimizer_steps=int(args.measured_optimizer_steps))
    mp.spawn(_rank_worker, args=(context,), nprocs=world_size, join=True)
    rank_results = []
    for rank in range(world_size):
        rank_file = paths.rank_dir / f"rank_{rank}.json"
        if rank_file.is_file():
            _bridge_heartbeat(paths, args.run_id, "rank_json_written", rank=rank, path=str(rank_file))
            rank_results.append(json.loads(rank_file.read_text(encoding="utf-8")))
    rank0_records = next((item.get("optimizer_step_records") for item in rank_results if item.get("rank") == 0), []) or []
    for record in rank0_records:
        _bridge_heartbeat(
            paths,
            args.run_id,
            "measured_step_n",
            optimizer_step=record.get("optimizer_step"),
            phase=record.get("phase"),
        )
    evidence["ddp"]["rank_count"] = world_size
    evidence["ddp"]["local_ranks"] = [item.get("local_rank") for item in rank_results]
    evidence["ddp"]["init_status"] = "ok" if len(rank_results) == world_size else "incomplete"
    evidence["ddp"]["forward_success_per_rank"] = {str(item.get("rank")): bool(item.get("forward_success")) for item in rank_results}
    evidence["ddp"]["backward_success_per_rank"] = {str(item.get("rank")): bool(item.get("backward_success")) for item in rank_results}
    evidence["ddp"]["find_unused_parameters"] = next((item.get("find_unused_parameters") for item in rank_results if "find_unused_parameters" in item), None)
    evidence["ddp"]["sampler"] = next((item.get("sampler") for item in rank_results if "sampler" in item), {})
    evidence["loss"]["finite_per_rank"] = {str(item.get("rank")): bool(item.get("local_finite")) for item in rank_results}
    evidence["loss"]["finite_all_reduce_result"] = all(bool(item.get("global_finite")) for item in rank_results)
    first_ok = next((item for item in rank_results if "total_loss" in item), None)
    if first_ok:
        cache_summary = first_ok.get("tokenizer_cache_manifest_summary", {}) or {}
        upstream_summary = first_ok.get("step3_upstream_evidence_summary", {}) or {}
        evidence["loss"]["total_loss"] = first_ok.get("total_loss")
        evidence["loss"]["component_losses"] = first_ok.get("component_losses", {})
        evidence["loss"]["component_weights"] = first_ok.get("component_weights", {})
        evidence["loss"]["weighted_components"] = first_ok.get("weighted_components", {})
        evidence["loss"]["component_participation"] = first_ok.get("component_participation", {})
        evidence["loss"]["component_finite_status"] = first_ok.get("component_finite_status", {})
        evidence["loss"]["global_component_finite_status"] = first_ok.get("global_component_finite_status", {})
        evidence["loss"]["finite_sync_summary"] = first_ok.get("finite_sync_summary", {})
        evidence["loss"]["component_graph_tied_zero_status"] = first_ok.get("component_graph_tied_zero_status", {})
        evidence["loss"]["loss_logging_summary"] = first_ok.get("loss_logging_summary", {})
        evidence["loss"]["loss_semantics"] = first_ok.get("loss_semantics", {})
        evidence["loss"]["duplicate_loss_check_summary"] = first_ok.get("duplicate_loss_check_summary", {})
        evidence["loss"]["graph_safety_preflight"] = first_ok.get("graph_safety_preflight", {})
        evidence["loss"]["nan_inf_fail_fast_path"] = "global finite all-reduce before backward"
        evidence["memory_performance"]["profile_domain_artifacts"] = first_ok.get("profile_domain_artifacts", {})
        evidence["tokenizer_cache"].update(
            {
                "manifest_schema": cache_summary.get("schema_version"),
                "compatibility_result": "manifest_gated_cache_ready" if cache_summary else "missing_summary",
                "manifest_path": cache_summary.get("manifest_path"),
                "manifest_fingerprint": cache_summary.get("manifest_fingerprint"),
                "cache_dir": cache_summary.get("cache_dir"),
                "cache_content_hash": cache_summary.get("cache_content_hash"),
                "tokenizer_cache_compat_hash": cache_summary.get("tokenizer_cache_compat_hash"),
                "data_contract_hash": cache_summary.get("data_contract_hash"),
                "preprocessing_artifact_hash": cache_summary.get("preprocessing_artifact_hash"),
                "full_run_config_hash": cache_summary.get("full_run_config_hash"),
                "train_runtime_config_hash": cache_summary.get("train_runtime_config_hash"),
                "optimizer_config_hash": cache_summary.get("optimizer_config_hash"),
                "performance_profile_hash": cache_summary.get("performance_profile_hash"),
                "compatibility_key": cache_summary.get("compatibility_key"),
                "fingerprint_hash": cache_summary.get("fingerprint_hash"),
                "upstream_gate_hash": cache_summary.get("upstream_gate_hash"),
                "startup_timing": cache_summary.get("startup_timing", {}),
                "rank0_only_cache_build": cache_summary.get("rank0_only_cache_build"),
                "rank0_only_csv_tokenizer_build": cache_summary.get("rank0_only_csv_tokenizer_build"),
            }
        )
        if upstream_summary:
            preprocess = upstream_summary.get("preprocess", {})
            evidence["preprocess_upstream_evidence"].update(
                {
                    "status": "pass" if upstream_summary.get("status") == "ok" else upstream_summary.get("status"),
                    "preprocess_latest": {
                        unit: {
                            "run_id": (preprocess.get(unit) or {}).get("run_id"),
                            "latest_status": (preprocess.get(unit) or {}).get("latest_status"),
                            "validation_status": (preprocess.get(unit) or {}).get("validation_status"),
                            "paths": (preprocess.get(unit) or {}).get("paths"),
                        }
                        for unit in ("a", "b", "c")
                    },
                    "upstream_gate_hash": upstream_summary.get("fingerprint_hash"),
                    "manifest_metrics_verify_fingerprints": {
                        unit: {
                            "stage_manifest": (preprocess.get(unit) or {}).get("stage_manifest_fingerprint"),
                            "metrics": (preprocess.get(unit) or {}).get("metrics_fingerprint"),
                            "verify_report": (preprocess.get(unit) or {}).get("verify_report_fingerprint"),
                        }
                        for unit in ("a", "b", "c")
                    },
                    "profile_domain_artifact_fingerprints": {
                        "profile_artifact_fingerprints": upstream_summary.get("profile_artifact_fingerprints", {}),
                        "domain_artifact_fingerprints": upstream_summary.get("domain_artifact_fingerprints", {}),
                    },
                }
            )
    evidence["memory_performance"]["per_rank_memory"] = {str(item.get("rank")): item.get("memory") for item in rank_results}
    evidence["memory_performance"]["step_wall_time_s"] = max((float(item.get("step_wall_time_s", 0.0)) for item in rank_results), default=None)
    evidence["memory_performance"]["dataloader_wait_s"] = max((float(item.get("dataloader_wait_s", 0.0)) for item in rank_results), default=None)
    evidence["memory_performance"]["h2d_wait_s"] = max((float(item.get("h2d_wait_s", 0.0)) for item in rank_results), default=None)
    evidence["memory_performance"]["optimizer_step_probe"] = {
        "per_rank": {
            str(item.get("rank")): {
                "optimizer_steps_completed": item.get("optimizer_steps_completed"),
                "target_optimizer_steps": item.get("target_optimizer_steps"),
                "warmup_optimizer_steps": item.get("warmup_optimizer_steps"),
                "measured_optimizer_steps": item.get("measured_optimizer_steps"),
                "measured_optimizer_steps_completed": item.get("measured_optimizer_steps_completed"),
                "batch_steps_completed": item.get("batch_steps_completed"),
                "batch_semantics_version": item.get("batch_semantics_version"),
                "stop_reason": item.get("mode_stop_reason"),
                "completed_target": item.get("completed_target"),
                "nonfinite_count": item.get("nonfinite_count"),
                "warmup_summary": item.get("warmup_summary"),
                "measured_summary": item.get("measured_summary"),
            }
            for item in rank_results
        },
        "rank0_records": next((item.get("optimizer_step_records") for item in rank_results if item.get("rank") == 0), []),
    }
    evidence["short_pilot"] = {
        "validation_records_rank0": next((item.get("validation_records") for item in rank_results if item.get("rank") == 0), []),
        "non_downstream_consumable": True,
        "checkpoint_written": False,
        "pilot_namespace": str(paths.run_dir),
    }
    evidence["memory_performance"]["oom_status"] = "no_oom"
    oom_ranks = [item for item in rank_results if bool(item.get("oom"))]
    if oom_ranks:
        evidence["memory_performance"]["oom_status"] = "oom"
        evidence["memory_performance"]["oom_allocation_requests"] = {
            str(item.get("rank")): item.get("oom_allocation_request") for item in oom_ranks
        }
        evidence["memory_performance"]["oom_attribution_questions"] = {
            "activation_backward_graph": "inspect after forward/backward phase deltas",
            "loss_temporary_tensors": "inspect after loss composition delta and structured_loss_temporary_tensor_summary",
            "frozen_profile_domain_buffers": "inspect profile_domain_buffer_footprint and profile_buffer_policy",
            "optimizer_state": "inspect after optimizer.step delta",
            "fragmentation": "inspect reserved_minus_allocated and CUDA memory_summary",
            "profile_gpu_residency": "compare gpu_resident baseline with probe-only cpu_pinned_batch_gather exploration",
        }
    all_forward = all(bool(item.get("forward_success")) for item in rank_results) and len(rank_results) == world_size
    all_backward = all(bool(item.get("backward_success")) for item in rank_results) and len(rank_results) == world_size
    all_optimizer = all(bool(item.get("optimizer_step")) for item in rank_results) and len(rank_results) == world_size
    all_targets = all(bool(item.get("completed_target")) for item in rank_results) and len(rank_results) == world_size
    finite_global = bool(evidence["loss"].get("finite_all_reduce_result"))
    if all_forward and all_backward and all_optimizer and finite_global and all_targets:
        evidence["verdict"]["status"] = "PASS"
    else:
        evidence["verdict"]["status"] = "FAIL"
        evidence["verdict"]["blockers"].append(
            "DDP probe did not complete forward/backward/optimizer/finite-loss/target-step checks on every rank"
        )
    return evidence


def check_formal_namespace_pollution(task_id: int) -> dict[str, Any]:
    latest = REPO_ROOT / "runs" / "step3" / f"task{int(task_id)}" / "latest.json"
    return {
        "formal_step3_latest_path": str(latest),
        "formal_step3_latest_exists": latest.exists(),
        "probe_wrote_formal_latest": False,
        "probe_wrote_formal_checkpoint": False,
    }


def finalize_evidence(evidence: dict[str, Any], paths: ProbePaths, *, task_id: int) -> dict[str, Any]:
    evidence["outputs"]["json_evidence_path"] = str(paths.json_path)
    evidence["outputs"]["md_evidence_path"] = str(paths.md_path)
    evidence["outputs"]["formal_namespace_pollution_check"] = check_formal_namespace_pollution(task_id)
    if evidence["verdict"]["status"] == "PASS" and evidence["verdict"]["blockers"]:
        evidence["verdict"]["status"] = "FAIL"
    _write_json(paths.json_path, evidence)
    _write_text(paths.md_path, _markdown_report(evidence))
    return evidence


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    run_id = _safe_run_id(args.run_id)
    paths = make_paths(
        run_id,
        mode=args.mode,
        bridge_log_path=args.bridge_log_path,
        bridge_status_path=args.bridge_status_path,
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    _append_bridge_log(paths, run_id, f"ODCR_BRIDGE_BEGIN_{run_id}")
    try:
        if not args.no_formal_checkpoint:
            raise ProbeError("--no-formal-checkpoint is mandatory")
        if args.mode == "step3-ddp-smoke":
            if int(args.max_batches) < 1 or int(args.max_batches) > 2:
                raise ProbeError("--max-batches must be 1 or 2")
            if int(args.max_steps) != 1:
                raise ProbeError("--max-steps must be exactly 1")
            candidate_for_resolve = args.smoke_candidate
        else:
            if not args.candidate_name:
                raise ProbeError(f"--candidate-name is required for {args.mode}")
            if int(args.max_wall_seconds) > 900:
                raise ProbeError("--max-wall-seconds must be <= 900 for governed bridge performance modes")
            candidate_for_resolve = args.candidate_name
        _bridge_heartbeat(
            paths,
            run_id,
            "candidate_started",
            mode=args.mode,
            candidate=candidate_for_resolve,
            max_wall_seconds=int(args.max_wall_seconds),
        )
        cfg, source_rows, snapshot = resolve_one_control(
            int(args.task_id),
            mode=args.mode,
            smoke_candidate=args.smoke_candidate,
            candidate_name=candidate_for_resolve,
            worker_profile=args.worker_profile,
        )
        probe_meta = _write_probe_meta(snapshot, paths)
        cpu = validate_cpu_budget(snapshot, strict=bool(args.strict))
        evidence = _base_evidence(
            run_id=run_id,
            mode=args.mode,
            task_id=int(args.task_id),
            cfg=cfg,
            source_rows=source_rows,
            snapshot=snapshot,
            paths=paths,
            include_torch=not bool(args.schema_only),
        )
        evidence["outputs"].update(probe_meta)
        evidence["one_control"]["cpu_budget_validation"] = cpu
        evidence["memory_performance"]["startup_vs_steady_state"].update(
            {
                "warmup_optimizer_steps": int(args.warmup_optimizer_steps),
                "measured_optimizer_steps": int(args.measured_optimizer_steps),
                "candidate_name": args.candidate_name or args.smoke_candidate,
                "worker_profile": args.worker_profile,
                "max_wall_seconds": int(args.max_wall_seconds),
                "mode_contract": (
                    "correctness_only_no_performance_recommendation"
                    if args.mode == "step3-ddp-smoke"
                    else "full_forward_backward_optimizer_step_probe_no_formal_namespace"
                ),
            }
        )
        evidence["outputs"]["formal_writes_forbidden"] = {
            "formal_latest": True,
            "formal_checkpoint": True,
            "formal_cache": True,
            "formal_stage_run": True,
            "probe_namespace_only": str(paths.run_dir),
        }
        if args.mode == "step3-performance-probe":
            evidence["one_control"]["optimizer_step_required"] = True
        if args.mode == "step3-short-pilot":
            evidence["one_control"]["short_pilot_contract"] = {
                "max_epochs": int(args.max_epochs),
                "max_optimizer_steps": int(args.max_optimizer_steps),
                "validate_every_steps": int(args.validate_every_steps),
                "validate_every_epoch": bool(args.validate_every_epoch),
                "downstream_consumable_checkpoint": False,
            }
        if cpu["blockers"]:
            evidence["verdict"]["blockers"].extend(cpu["blockers"])
        if args.schema_only:
            evidence["verdict"]["status"] = "NOT_VERIFIED"
            evidence["verdict"]["blockers"].append("schema_only requested; real DDP probe not run")
        else:
            from odcr_core.step3_upstream_gate import validate_step3_preprocess_upstream_gate

            upstream = validate_step3_preprocess_upstream_gate(
                repo_root=REPO_ROOT,
                task_id=int(args.task_id),
                auxiliary_domain=str(evidence["source_domain"]),
                target_domain=str(evidence["target_domain"]),
                data_dir=Path(cfg.data_dir),
                merged_dir=Path(cfg.merged_dir),
                runs_dir=Path(cfg.runs_dir),
                embed_dim=int(cfg.embed_dim),
            )
            evidence["preprocess_upstream_evidence"] = {
                "status": "pass",
                "preprocess_latest": {
                    unit: ((upstream.get("preprocess_runs") or {}).get(unit) or {}).get("latest_run_id")
                    for unit in ("a", "b", "c")
                },
                "upstream_gate_hash": upstream.get("fingerprint_hash"),
                "manifest_metrics_verify_fingerprints": upstream.get("preprocess_fingerprints", {}),
                "profile_domain_artifact_fingerprints": {
                    "profile_artifact_fingerprints": upstream.get("profile_artifact_fingerprints", {}),
                    "domain_artifact_fingerprints": upstream.get("domain_artifact_fingerprints", {}),
                },
            }
            evidence = run_real_probe(args, paths, evidence, cfg)
        _bridge_heartbeat(paths, run_id, "aggregation_started")
        evidence = finalize_evidence(evidence, paths, task_id=int(args.task_id))
        _bridge_heartbeat(
            paths,
            run_id,
            "aggregation_completed",
            verdict=str((evidence.get("verdict") or {}).get("status")),
        )
        status_name = str((evidence.get("verdict") or {}).get("status"))
        success = status_name == "PASS"
        exit_code = 0 if success or args.schema_only else 1
        mode_success_reasons = {
            "step3-ddp-smoke": "step3_ddp_smoke_completed",
            "step3-performance-probe": "step3_performance_probe_completed",
            "step3-short-pilot": "step3_short_pilot_completed",
        }
        stop_reason = mode_success_reasons.get(args.mode, "step3_probe_completed") if success else status_name.lower()
        _bridge_heartbeat(paths, run_id, "candidate_finished", verdict=status_name, exit_code=exit_code)
        _append_bridge_log(paths, run_id, f"ODCR_BRIDGE_END_{run_id}")
        _write_bridge_status(
            paths,
            _bridge_status(
                run_id=run_id,
                mode=args.mode,
                success=success,
                exit_code=exit_code,
                stop_reason=stop_reason,
                started=started,
                metrics={"json_evidence_path": str(paths.json_path), "md_evidence_path": str(paths.md_path), "verdict": status_name},
                first_result_seen=True,
            ),
        )
        return exit_code
    except Exception as exc:
        cfg = None
        try:
            cfg, source_rows, snapshot = resolve_one_control(
                int(args.task_id),
                mode=args.mode,
                smoke_candidate=args.smoke_candidate,
                candidate_name=getattr(args, "candidate_name", None),
                worker_profile=getattr(args, "worker_profile", None),
            )
            evidence = _base_evidence(
                run_id=run_id,
                mode=args.mode,
                task_id=int(args.task_id),
                cfg=cfg,
                source_rows=source_rows,
                snapshot=snapshot,
                paths=paths,
                include_torch=False,
            )
        except Exception:
            evidence = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "mode": args.mode,
                "timestamp": _now(),
                "task_id": int(args.task_id),
                "outputs": {"json_evidence_path": str(paths.json_path), "md_evidence_path": str(paths.md_path)},
                "verdict": {"status": "FAIL", "blockers": [], "warnings": []},
            }
        evidence["verdict"]["status"] = "FAIL"
        evidence["verdict"]["blockers"].append(repr(exc))
        evidence["verdict"]["warnings"].append("traceback written into JSON evidence")
        evidence["error_traceback"] = traceback.format_exc()
        finalize_evidence(evidence, paths, task_id=int(args.task_id))
        _append_bridge_log(paths, run_id, "bridge_error", repr(exc))
        _append_bridge_log(paths, run_id, traceback.format_exc())
        _append_bridge_log(paths, run_id, f"ODCR_BRIDGE_END_{run_id}")
        _write_bridge_status(
            paths,
            _bridge_status(
                run_id=run_id,
                mode=args.mode,
                success=False,
                exit_code=1,
                stop_reason="step3_ddp_smoke_failed",
                started=started,
                metrics={"error": repr(exc), "json_evidence_path": str(paths.json_path), "md_evidence_path": str(paths.md_path)},
                first_result_seen=True,
            ),
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Internal bridge-only Step3 real-data DDP smoke/performance probe. Does not run formal training.",
    )
    parser.add_argument(
        "--mode",
        choices=("step3-ddp-smoke", "step3-performance-probe", "step3-short-pilot"),
        default="step3-ddp-smoke",
    )
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--smoke-candidate", help="Internal selector for a key/candidate from configs/odcr.yaml:step3.backup_profiles.")
    parser.add_argument("--candidate-name", help="Internal selector for step3.performance_candidates.batch_ladder or probe-only step3.exploration_profiles.")
    parser.add_argument("--worker-profile", help="Internal selector for configs/odcr.yaml:step3.worker_profiles.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--warmup-optimizer-steps", type=int, default=10)
    parser.add_argument("--measured-optimizer-steps", type=int, default=50)
    parser.add_argument("--max-wall-seconds", type=int, default=900)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--max-optimizer-steps", type=int, default=2000)
    parser.add_argument("--validate-every-steps", type=int, default=500)
    parser.add_argument("--validate-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--schema-only", action="store_true", help="Write NOT_VERIFIED schema evidence without CUDA/DDP.")
    parser.add_argument("--no-formal-checkpoint", action="store_true", required=True)
    parser.add_argument("--allow-one-optimizer-step", action="store_true", help="Optional smoke-only optimizer step; never saves a model.")
    parser.add_argument("--bridge-status-path")
    parser.add_argument("--bridge-log-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
