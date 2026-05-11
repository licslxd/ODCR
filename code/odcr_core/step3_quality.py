"""Step3 quality, checkpoint, and runtime-evidence helpers.

The helpers in this module deliberately separate Evidence Level claims:

- Level 1: code exists.
- Level 2: active path is wired.
- Level 3: controlled runtime behavior is verified.
- Level 4: formal run evidence is verified.
- Level 5: downstream eligibility is granted.

Unit tests may prove Level 1/2 contracts. They must not claim Level 3/4/5
without run-time artifacts from the controlled validation or formal run paths.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from odcr_core.file_atomic import atomic_write_json
from odcr_core.training_checkpoint import checkpoint_file_sha256, file_fingerprint, stable_hash
from odcr_core.step3_eval_handoff import Step3EvalHandoffError, quality_audit_from_eval_handoff


STEP3_QUALITY_GATE_VERSION = "odcr_step3_quality_gate/1"
STEP3_QUALITY_AUDIT_SCHEMA_VERSION = "odcr_step3_quality_audit/1"
STEP3_CHECKPOINT_POLICY_VERSION = "odcr_step3_checkpoint_policy/1"
STEP3_RUNTIME_EVIDENCE_SCHEMA_VERSION = "odcr_step3_runtime_evidence/1"
STEP3_DIAGNOSTIC_PROTOCOL_VERSION = "odcr_step3_diagnostic_protocol/1"
STEP3_PERFORMANCE_CANDIDATE_SCHEMA_VERSION = "odcr_step3_a100_candidate_matrix/1"

EVIDENCE_LEVELS: dict[str, dict[str, Any]] = {
    "level1_code_present": {
        "level": 1,
        "proves": "Class/function/config/schema exists.",
        "does_not_prove": "The formal loop uses it or that runtime behavior occurred.",
    },
    "level2_active_path": {
        "level": 2,
        "proves": "Formal train loop or resolver path is wired.",
        "does_not_prove": "Runtime behavior, performance, or quality passed.",
    },
    "level3_runtime_verified": {
        "level": 3,
        "proves": "Controlled validation/probe observed real behavior.",
        "does_not_prove": "A full formal run passed.",
    },
    "level4_formal_verified": {
        "level": 4,
        "proves": "Full formal run artifacts show behavior/quality/performance.",
        "does_not_prove": "Downstream may consume unless quality gate passes.",
    },
    "level5_downstream_eligible": {
        "level": 5,
        "proves": "quality_status=pass and downstream_ready=true with selected checkpoint hash.",
        "does_not_prove": "Future code/config changes remain compatible.",
    },
}

TIMING_REQUIRED_FIELDS: tuple[str, ...] = (
    "epoch",
    "global_step",
    "rank",
    "step_total_ms",
    "loader_next_wait_ms",
    "cpu_collate_ms",
    "h2d_submit_ms",
    "h2d_wait_ms",
    "prefetch_wait_ms",
    "forward_ms",
    "loss_compute_ms",
    "structured_gather_ms",
    "finite_sync_ms",
    "duplicate_loss_check_ms",
    "ddp_backward_sync_ms",
    "backward_compute_ms",
    "grad_check_ms",
    "grad_norm_compute_ms",
    "grad_clip_ms",
    "grad_monitor_ms",
    "nonfinite_detect_ms",
    "optimizer_ms",
    "ema_ms",
    "zero_grad_ms",
    "scheduler_ms",
    "metrics_io_ms",
    "logging_io_ms",
    "checkpoint_io_ms",
    "cuda_sync_ms",
    "unknown_ms",
    "timing_closed_ratio",
    "optimizer_step_executed",
    "scheduler_step_executed",
    "grad_finite",
    "skipped_step_reason",
)

MEMORY_PHASES: tuple[str, ...] = (
    "after_batch_cpu",
    "after_h2d",
    "after_forward",
    "after_loss_compute",
    "after_structured_gather",
    "after_backward",
    "after_grad_norm",
    "after_optimizer",
    "after_validation",
    "after_eval_decode",
    "after_checkpoint_save",
    "after_epoch_boundary",
)

MEMORY_REQUIRED_FIELDS: tuple[str, ...] = (
    "phase",
    "rank",
    "allocated_gib",
    "max_allocated_gib",
    "reserved_gib",
    "max_reserved_gib",
    "reserved_minus_allocated_gib",
    "inactive_split_gib",
    "non_releasable_gib",
    "cuda_malloc_retry_count",
    "cuda_oom_count",
    "largest_free_block",
    "memory_snapshot_path",
)

PREFETCH_EVIDENCE_FIELDS: tuple[str, ...] = (
    "prefetcher_code_present",
    "prefetcher_active_in_formal_loop",
    "h2d_stream_created",
    "double_buffer_configured",
    "double_buffer_active",
    "num_device_buffers",
    "record_stream_tensor_count",
    "compute_wait_stream_count",
    "h2d_event_elapsed_ms",
    "h2d_wait_ms",
    "prefetch_wait_ms",
    "h2d_hidden_by_compute_ratio",
    "overlap_verified",
    "fallback_used",
    "fallback_reason",
)

DIAGNOSTIC_PROTOCOLS: dict[str, dict[str, Any]] = {
    "odcr_step3_diagnostic": {
        "evaluator_protocol": "odcr_step3_diagnostic",
        "diagnostic_only": True,
        "not_final_paper_metric": True,
        "split_policy": "merged auxiliary + target valid",
    },
    "code1_target_only_comparable": {
        "evaluator_protocol": "code1_target_only_comparable",
        "diagnostic_only": True,
        "not_final_paper_metric": True,
        "split_policy": "target-only valid/test inside Step3; no Step4/5/eval/rerank",
    },
    "full_pipeline_final": {
        "evaluator_protocol": "full_pipeline_final",
        "diagnostic_only": False,
        "not_final_paper_metric": False,
        "split_policy": "available only after Step4/Step5/eval/rerank",
    },
}


class Step3QualityGateError(RuntimeError):
    """Raised before downstream consumers load a blocked Step3 checkpoint."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json_object(path: str | Path, *, required: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        if required:
            raise Step3QualityGateError(f"missing required JSON artifact: {p}")
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step3QualityGateError(f"invalid JSON artifact: {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step3QualityGateError(f"JSON artifact root must be an object: {p}")
    return data


def read_epoch_summary(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file() or p.stat().st_size <= 0:
        return []
    with p.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("epoch",):
            try:
                item[key] = int(item[key])
            except Exception:
                pass
        for key in ("train_loss", "valid_loss", "best_metric", "elapsed_s", "samples_per_sec"):
            try:
                item[key] = float(item[key])
            except Exception:
                pass
        out.append(item)
    return out


def metric_improved(value: float, best: float | None, *, direction: str = "min") -> bool:
    if not math.isfinite(float(value)):
        return False
    if best is None or not math.isfinite(float(best)):
        return True
    if direction == "min":
        return float(value) < float(best)
    if direction == "max":
        return float(value) > float(best)
    raise ValueError(f"unsupported checkpoint selection direction: {direction!r}")


def checkpoint_filename_for_metric(epoch: int, metric_name: str, metric_value: float) -> str:
    safe_metric = str(metric_name).replace("/", "_").replace(" ", "_")
    return f"epoch_{int(epoch):03d}_{safe_metric}_{float(metric_value):.4f}.pth"


def checkpoint_sidecar_payload(
    *,
    checkpoint_file: str | Path,
    checkpoint_epoch: int,
    selection_metric: str,
    selection_metric_value: float,
    selection_scope: str,
    global_best_epoch: int | None,
    global_best_metric: float | None,
    after_min_epochs_best_epoch: int | None,
    after_min_epochs_best_metric: float | None,
    epoch_summary_path: str | Path | None,
    metrics_jsonl_path: str | Path | None,
    resolved_config_hash: str,
    training_runtime_config_hash: str,
    quality_status_at_save: str,
    grad_inf_count_until_epoch: int,
    optimizer_state_path: str | Path | None = None,
) -> dict[str, Any]:
    ckpt = Path(checkpoint_file).expanduser().resolve()
    if not ckpt.is_file():
        raise Step3QualityGateError(f"checkpoint sidecar requires existing checkpoint file: {ckpt}")
    opt_hash = ""
    if optimizer_state_path:
        opt = Path(optimizer_state_path).expanduser().resolve()
        if opt.is_file():
            opt_hash = str(file_fingerprint(opt).get("sha256") or "")
    return {
        "checkpoint_policy_schema_version": STEP3_CHECKPOINT_POLICY_VERSION,
        "checkpoint_file": str(ckpt),
        "checkpoint_file_hash": checkpoint_file_sha256(ckpt),
        "checkpoint_epoch": int(checkpoint_epoch),
        "selection_metric": str(selection_metric),
        "selection_metric_value": float(selection_metric_value),
        "selection_direction": "min",
        "selection_scope": str(selection_scope),
        "global_best_epoch": None if global_best_epoch is None else int(global_best_epoch),
        "global_best_metric": None if global_best_metric is None else float(global_best_metric),
        "after_min_epochs_best_epoch": (
            None if after_min_epochs_best_epoch is None else int(after_min_epochs_best_epoch)
        ),
        "after_min_epochs_best_metric": (
            None if after_min_epochs_best_metric is None else float(after_min_epochs_best_metric)
        ),
        "epoch_summary_hash": _optional_file_hash(epoch_summary_path),
        "metrics_jsonl_hash": _optional_file_hash(metrics_jsonl_path),
        "resolved_config_hash": str(resolved_config_hash or ""),
        "training_runtime_config_hash": str(training_runtime_config_hash or ""),
        "quality_status_at_save": str(quality_status_at_save or "not_evaluated"),
        "grad_inf_count_until_epoch": int(grad_inf_count_until_epoch),
        "model_file_hash": checkpoint_file_sha256(ckpt),
        "optimizer_state_hash": opt_hash,
        "code_commit": _git_commit_or_empty(),
    }


def checkpoint_event_from_sidecar(sidecar: Mapping[str, Any], *, reason: str, replaced_previous: bool) -> dict[str, Any]:
    created_at = utc_now()
    event_core = {
        "checkpoint_file": sidecar.get("checkpoint_path") or sidecar.get("checkpoint_file"),
        "checkpoint_file_hash": sidecar.get("checkpoint_file_hash"),
        "checkpoint_epoch": sidecar.get("checkpoint_epoch"),
        "selection_scope": sidecar.get("selection_scope"),
        "selection_metric": sidecar.get("selection_metric"),
        "selection_metric_value": sidecar.get("selection_metric_value"),
        "selection_direction": sidecar.get("selection_direction"),
        "reason": str(reason),
        "replaced_previous": bool(replaced_previous),
    }
    event_id = stable_hash(
        {
            "schema_version": STEP3_CHECKPOINT_POLICY_VERSION,
            **event_core,
            "lineage_hash": sidecar.get("lineage_hash"),
        },
        length=32,
    )
    return {
        "event_schema_version": STEP3_CHECKPOINT_POLICY_VERSION,
        "event_id": event_id,
        **event_core,
        "global_best_epoch": sidecar.get("global_best_epoch"),
        "global_best_metric": sidecar.get("global_best_metric"),
        "after_min_epochs_best_epoch": sidecar.get("after_min_epochs_best_epoch"),
        "after_min_epochs_best_metric": sidecar.get("after_min_epochs_best_metric"),
        "resolved_config_hash": sidecar.get("resolved_config_hash"),
        "training_runtime_config_hash": sidecar.get("training_runtime_config_hash"),
        "epoch_summary_hash": sidecar.get("epoch_summary_hash"),
        "metrics_jsonl_hash": sidecar.get("metrics_jsonl_hash"),
        "quality_status": sidecar.get("quality_status") or sidecar.get("quality_status_at_save"),
        "downstream_ready": bool(sidecar.get("downstream_ready", False)),
        "created_at": created_at,
        "created_at_utc": created_at,
        "epoch": sidecar.get("checkpoint_epoch"),
        "metric": sidecar.get("selection_metric_value"),
        "metric_name": sidecar.get("selection_metric"),
        "metric_source": "meta/epoch_summary.csv.valid_loss",
        "path": event_core["checkpoint_file"],
        "hash": event_core["checkpoint_file_hash"],
    }


def build_best_event_payload(
    *,
    best_observed_event: Mapping[str, Any] | None,
    best_after_min_epochs_event: Mapping[str, Any] | None,
    latest_event: Mapping[str, Any] | None,
    topk_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "odcr_step3_best_event/2",
        "best_observed_event": dict(best_observed_event or {}),
        "best_after_min_epochs_event": dict(best_after_min_epochs_event or {}),
        "latest_event": dict(latest_event or {}),
        "topk_events": [dict(item) for item in topk_events],
        "updated_at_utc": utc_now(),
    }


def _optional_file_hash(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path).expanduser()
    if not p.is_file():
        return ""
    fp = file_fingerprint(p)
    return str(fp.get("sha256") or stable_hash(fp))


def _git_commit_or_empty() -> str:
    head = Path(".git") / "HEAD"
    try:
        raw = head.read_text(encoding="utf-8").strip()
        if raw.startswith("ref:"):
            ref = Path(".git") / raw.split(" ", 1)[1].strip()
            return ref.read_text(encoding="utf-8").strip()
        return raw
    except OSError:
        return ""


def _count_jsonl(path: Path) -> int:
    if not path.is_file() or path.stat().st_size <= 0:
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def _best_epoch(rows: Sequence[Mapping[str, Any]]) -> tuple[int | None, float | None]:
    best_epoch: int | None = None
    best_metric: float | None = None
    for row in rows:
        try:
            epoch = int(row.get("epoch"))
            metric = float(row.get("valid_loss"))
        except Exception:
            continue
        if metric_improved(metric, best_metric, direction="min"):
            best_epoch = epoch
            best_metric = metric
    return best_epoch, best_metric


def build_step3_quality_audit(
    run_root: str | Path,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    meta = root / "meta"
    state = root / "state"
    model = root / "model"
    thresholds = dict(thresholds or {})
    grad_inf_threshold = int(thresholds.get("grad_inf_count_block_threshold", 0))
    severe_deterioration_ratio = float(thresholds.get("valid_loss_deterioration_ratio_block_threshold", 0.25))
    reasons: list[str] = []
    warnings: list[str] = []

    summary = load_json_object(meta / "run_summary.json", required=False)
    rows = read_epoch_summary(meta / "epoch_summary.csv")
    best_epoch, best_metric = _best_epoch(rows)
    best_event = load_json_object(state / "best_event.json", required=False)
    best_observed_event = best_event.get("best_observed_event") if isinstance(best_event.get("best_observed_event"), Mapping) else {}
    legacy_best_epoch = best_event.get("epoch")
    selected_epoch = best_observed_event.get("epoch") or legacy_best_epoch

    if best_epoch is None:
        reasons.append("best_observed_event_missing")
    if selected_epoch is not None and best_epoch is not None and int(selected_epoch) != int(best_epoch):
        reasons.append("best_checkpoint_not_global_best")
    if not (model / "best_observed.pth").is_file() and best_epoch is not None:
        if not (model / "best.pth").is_file() or int(selected_epoch or -1) != int(best_epoch):
            reasons.append("global_best_checkpoint_missing")

    metrics_path = meta / "metrics.jsonl"
    loss_path = meta / "loss_breakdown.jsonl"
    timing_path = meta / "timing_profile.jsonl"
    gpu_path = meta / "gpu_profile.jsonl"
    for path, reason in (
        (metrics_path, "metrics_jsonl_missing"),
        (loss_path, "loss_breakdown_jsonl_missing"),
        (timing_path, "timing_profile_jsonl_missing"),
        (gpu_path, "gpu_profile_jsonl_missing"),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            reasons.append(reason)

    grad_inf_count = int(_count_full_log_grad_inf(meta / "full.log"))
    if grad_inf_count > grad_inf_threshold:
        reasons.append("grad_inf_mass_events")

    samples_path = meta / "samples.jsonl"
    if not samples_path.is_file() or samples_path.stat().st_size <= 0:
        reasons.append("samples_missing_or_empty")

    collapse = load_json_object(meta / "collapse_stats.json", required=False)
    empty_rate = float(collapse.get("empty_rate", 0.0) or 0.0) if collapse else 0.0
    distinct1 = float(collapse.get("distinct1", 1.0) or 0.0) if collapse else 1.0
    distinct2 = float(collapse.get("distinct2", 1.0) or 0.0) if collapse else 1.0
    if collapse and (empty_rate >= float(thresholds.get("empty_pred_rate_block_threshold", 0.5)) or (distinct1 == 0.0 and distinct2 == 0.0)):
        reasons.append("diagnostic_eval_collapse")
    elif not collapse and "samples_missing_or_empty" in reasons:
        reasons.append("diagnostic_eval_collapse")

    if best_metric is not None and rows:
        last_metric = float(rows[-1].get("valid_loss", best_metric))
        if math.isfinite(last_metric) and last_metric > float(best_metric) * (1.0 + severe_deterioration_ratio):
            reasons.append("valid_loss_after_best_severe_deterioration")

    status = str(summary.get("status") or "").lower()
    if status and status not in {"ok", "completed", "success"}:
        reasons.append("run_status_not_ok")

    quality_status = "blocked" if reasons else ("warning" if warnings else "pass")
    downstream_ready = quality_status == "pass"
    selected_checkpoint_path = (model / "best_observed.pth") if (model / "best_observed.pth").is_file() else (model / "best.pth")
    selected_checkpoint = str(selected_checkpoint_path)
    selected_hash = checkpoint_file_sha256(selected_checkpoint_path) if selected_checkpoint_path.is_file() else ""
    return {
        "schema_version": STEP3_QUALITY_AUDIT_SCHEMA_VERSION,
        "quality_gate_version": STEP3_QUALITY_GATE_VERSION,
        "run_root": str(root),
        "status": status or "unknown",
        "quality_status": quality_status,
        "downstream_ready": downstream_ready,
        "quality_block_reasons": sorted(set(reasons)),
        "quality_warnings": sorted(set(warnings)),
        "quality_gate_inputs": {
            "best_epoch_from_epoch_summary": best_epoch,
            "best_metric_from_epoch_summary": best_metric,
            "selected_best_epoch": selected_epoch,
            "grad_inf_count": grad_inf_count,
            "samples_count": _count_jsonl(samples_path),
            "metrics_rows": _count_jsonl(metrics_path),
            "timing_rows": _count_jsonl(timing_path),
            "gpu_rows": _count_jsonl(gpu_path),
        },
        "selected_downstream_checkpoint": selected_checkpoint,
        "selected_downstream_checkpoint_hash": selected_hash,
        "selected_downstream_checkpoint_scope": "best_observed",
        "selected_downstream_checkpoint_epoch": best_epoch,
        "selected_downstream_checkpoint_metric": best_metric,
        "evidence_levels": {
            "code_present": True,
            "active_path": bool(best_event or summary),
            "runtime_verified": False,
            "formal_verified": False,
            "downstream_eligible": downstream_ready,
        },
        "generated_at_utc": utc_now(),
    }


def write_step3_quality_audit(run_root: str | Path, audit: Mapping[str, Any]) -> Path:
    path = Path(run_root).expanduser().resolve() / "meta" / "quality_audit.json"
    atomic_write_json(path, dict(audit))
    return path


def load_step3_quality_audit(run_root: str | Path) -> dict[str, Any]:
    return load_json_object(Path(run_root).expanduser().resolve() / "meta" / "quality_audit.json", required=False)


def validate_step3_downstream_quality_gate(
    run_root: str | Path,
    *,
    selected_checkpoint: str | Path | None = None,
    require_sidecar_hash: bool = True,
    missing_policy: str = "block",
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    audit = load_step3_quality_audit(root)
    if not audit:
        if missing_policy == "warning":
            raise Step3QualityGateError(f"Step3 quality audit sidecar missing: {root / 'meta' / 'quality_audit.json'}")
        try:
            audit = quality_audit_from_eval_handoff(root)
        except Step3EvalHandoffError as handoff_exc:
            raise Step3QualityGateError(f"Step3 quality audit sidecar missing; downstream is blocked: {root}") from handoff_exc
    if str(audit.get("quality_status")) != "pass" or audit.get("downstream_ready") is not True:
        try:
            audit = quality_audit_from_eval_handoff(root)
        except Step3EvalHandoffError as handoff_exc:
            reasons = audit.get("quality_block_reasons") or []
            raise Step3QualityGateError(
                "Step3 downstream quality gate blocked run: "
                f"quality_status={audit.get('quality_status')!r} downstream_ready={audit.get('downstream_ready')!r} "
                f"reasons={reasons}"
            ) from handoff_exc
    ckpt = Path(selected_checkpoint or audit.get("selected_downstream_checkpoint") or "").expanduser()
    if not ckpt.is_absolute():
        ckpt = (root / ckpt).resolve()
    if not ckpt.is_file():
        raise Step3QualityGateError(f"Step3 selected downstream checkpoint is missing: {ckpt}")
    if require_sidecar_hash:
        expected_hash = str(audit.get("selected_downstream_checkpoint_hash") or "")
        actual_hash = checkpoint_file_sha256(ckpt)
        if expected_hash and expected_hash != actual_hash:
            raise Step3QualityGateError(
                f"Step3 selected checkpoint hash mismatch: audit={expected_hash!r} actual={actual_hash!r}"
            )
    return audit


def _count_full_log_grad_inf(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "grad_norm_pre_clip=inf" in line or "grad_norm_pre_clip=Infinity" in line:
            count += 1
    return count


@dataclass(frozen=True)
class GradInspection:
    grad_finite: bool
    nonfinite_param_count: int
    nonfinite_param_topk: list[dict[str, Any]]
    nonfinite_param_group_topk: list[dict[str, Any]]
    grad_norm_pre_clip: float


def inspect_gradients(
    named_parameters: Iterable[tuple[str, Any]],
    *,
    topk: int = 5,
) -> GradInspection:
    """Inspect gradients without mutating them."""
    import torch

    nonfinite: list[tuple[str, int, float]] = []
    group_counter: Counter[str] = Counter()
    norms: list[Any] = []
    for name, param in named_parameters:
        grad = getattr(param, "grad", None)
        if grad is None or getattr(grad, "is_sparse", False):
            continue
        detached = grad.detach()
        finite_mask = torch.isfinite(detached)
        is_finite = bool(finite_mask.all().item())
        if not is_finite:
            bad_count = int((~finite_mask).sum().item())
            try:
                max_abs = float(torch.nan_to_num(detached.float().abs(), nan=math.inf, posinf=math.inf).max().item())
            except Exception:
                max_abs = math.inf
            nonfinite.append((str(name), bad_count, max_abs))
            group_counter[_param_group_name(str(name))] += bad_count
        try:
            norms.append(torch.linalg.vector_norm(detached.float(), ord=2))
        except Exception:
            pass
    if norms:
        try:
            total_norm = float(torch.linalg.vector_norm(torch.stack(norms), ord=2).item())
        except Exception:
            total_norm = math.inf
    else:
        total_norm = 0.0
    nonfinite.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return GradInspection(
        grad_finite=not nonfinite and math.isfinite(total_norm),
        nonfinite_param_count=sum(item[1] for item in nonfinite),
        nonfinite_param_topk=[
            {"name": name, "nonfinite_count": count, "max_abs": max_abs}
            for name, count, max_abs in nonfinite[: max(0, int(topk))]
        ],
        nonfinite_param_group_topk=[
            {"group": name, "nonfinite_count": count}
            for name, count in group_counter.most_common(max(0, int(topk)))
        ],
        grad_norm_pre_clip=total_norm,
    )


def _param_group_name(name: str) -> str:
    if "embedding" in name or "embeddings" in name:
        return "embedding"
    if "attention" in name or "attn" in name:
        return "attention"
    if "norm" in name:
        return "norm"
    if "recommender" in name:
        return "recommender"
    if "disentangler" in name:
        return "disentangler"
    return name.split(".", 1)[0] if "." in name else name


def sync_grad_finite_decision(local_finite: bool, *, device: Any, world_size: int) -> bool:
    import torch
    import torch.distributed as dist

    flag = torch.tensor([1 if local_finite else 0], device=device, dtype=torch.int32)
    if int(world_size) > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Step3 grad finite sync requested before torch.distributed init.")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(int(flag.item()))


def timing_row_with_closure(
    step_timing: Mapping[str, Any],
    *,
    base: Mapping[str, Any],
    unknown_warn_ratio: float = 0.05,
) -> dict[str, Any]:
    row = dict(base)
    aliases = {
        "step_total_ms": ("total_step_time",),
        "loader_next_wait_ms": ("dataloader_next_wait",),
        "h2d_submit_ms": ("h2d_prefetch_time",),
        "prefetch_wait_ms": ("compute_wait_for_prefetch",),
        "forward_ms": ("forward_time",),
        "loss_compute_ms": ("loss_time",),
        "backward_compute_ms": ("backward_time",),
        "optimizer_ms": ("optimizer_time",),
        "scheduler_ms": ("scheduler_time",),
        "logging_io_ms": ("logging_time",),
    }
    for field in TIMING_REQUIRED_FIELDS:
        if field in row:
            continue
        candidates = aliases.get(field, ())
        value = 0.0
        for key in candidates:
            if key in step_timing:
                value = _seconds_to_ms(step_timing.get(key))
                break
        row[field] = value
    for key, value in step_timing.items():
        if key.endswith("_ms"):
            row[key] = float(value or 0.0)
    total = float(row.get("step_total_ms") or 0.0)
    measured_fields = (
        "loader_next_wait_ms",
        "cpu_collate_ms",
        "h2d_submit_ms",
        "h2d_wait_ms",
        "prefetch_wait_ms",
        "forward_ms",
        "loss_compute_ms",
        "structured_gather_ms",
        "finite_sync_ms",
        "duplicate_loss_check_ms",
        "ddp_backward_sync_ms",
        "backward_compute_ms",
        "grad_check_ms",
        "grad_norm_compute_ms",
        "grad_clip_ms",
        "grad_monitor_ms",
        "nonfinite_detect_ms",
        "optimizer_ms",
        "ema_ms",
        "zero_grad_ms",
        "scheduler_ms",
        "metrics_io_ms",
        "logging_io_ms",
        "checkpoint_io_ms",
        "cuda_sync_ms",
    )
    measured = sum(float(row.get(field) or 0.0) for field in measured_fields)
    unknown = max(0.0, total - measured) if total > 0.0 else 0.0
    row["unknown_ms"] = unknown
    row["timing_closed_ratio"] = 1.0 - (unknown / total) if total > 0.0 else 0.0
    row["performance_timing_status"] = (
        "timing_closed" if total > 0.0 and unknown / total < float(unknown_warn_ratio) else "performance_not_closed"
    )
    return row


def _seconds_to_ms(value: Any) -> float:
    try:
        return float(value or 0.0) * 1000.0
    except Exception:
        return 0.0


def collapse_stats_from_predictions(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    total = len(predictions)
    tokenized = [str(text or "").split() for text in predictions]
    ref_tok = [str(text or "").split() for text in references]
    empty = sum(1 for toks in tokenized if not toks)
    unigram_total = sum(len(toks) for toks in tokenized)
    bigrams = [tuple(toks[i : i + 2]) for toks in tokenized for i in range(max(0, len(toks) - 1))]
    unigrams = [tok for toks in tokenized for tok in toks]
    pred_counter = Counter(str(text or "") for text in predictions)
    repeat_ngram = 0
    bigram_total = len(bigrams)
    if bigram_total:
        counts = Counter(bigrams)
        repeat_ngram = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "schema_version": "odcr_step3_collapse_stats/1",
        "empty_rate": float(empty / total) if total else 1.0,
        "avg_pred_len": float(sum(len(toks) for toks in tokenized) / total) if total else 0.0,
        "avg_target_len": float(sum(len(toks) for toks in ref_tok) / len(ref_tok)) if ref_tok else 0.0,
        "distinct1": float(len(set(unigrams)) / unigram_total) if unigram_total else 0.0,
        "distinct2": float(len(set(bigrams)) / bigram_total) if bigram_total else 0.0,
        "repeat_ngram_rate": float(repeat_ngram / bigram_total) if bigram_total else 0.0,
        "top_20_predictions": [
            {"text": text, "count": count} for text, count in pred_counter.most_common(20)
        ],
        "malformed_rate": 0.0,
        "decode_error_count": 0,
        "generated_at_utc": utc_now(),
    }


def diagnostic_sample_record(
    *,
    run_id: str,
    epoch: int | None,
    split: str,
    sample_id: Any,
    user_id: Any = None,
    item_id: Any = None,
    source_domain: str,
    target_domain: str,
    rating_gold: float,
    rating_pred: float,
    target_text: str,
    pred_text: str,
    evaluator_protocol: str = "odcr_step3_diagnostic",
) -> dict[str, Any]:
    pred_tokens = str(pred_text or "").split()
    target_tokens = str(target_text or "").split()
    bigrams = [tuple(pred_tokens[i : i + 2]) for i in range(max(0, len(pred_tokens) - 1))]
    return {
        "schema_version": "odcr_step3_diagnostic_sample/1",
        "run_id": str(run_id),
        "epoch": None if epoch is None else int(epoch),
        "split": str(split),
        "sample_id": _json_scalar(sample_id),
        "user_id": _json_scalar(user_id),
        "item_id": _json_scalar(item_id),
        "source_domain": str(source_domain),
        "target_domain": str(target_domain),
        "rating_gold": float(rating_gold),
        "rating_pred": float(rating_pred),
        "target_text": str(target_text or ""),
        "pred_text": str(pred_text or ""),
        "pred_len": len(pred_tokens),
        "target_len": len(target_tokens),
        "empty_pred": not bool(pred_tokens),
        "unique_unigrams": len(set(pred_tokens)),
        "unique_bigrams": len(set(bigrams)),
        "repeat_ngram_ratio": _repeat_ngram_ratio(bigrams),
        "decode_status": "empty_prediction" if not pred_tokens else "ok",
        "evaluator_protocol": str(evaluator_protocol),
        "diagnostic_only": True,
        "not_final_paper_metric": True,
    }


def _repeat_ngram_ratio(bigrams: Sequence[tuple[str, str]]) -> float:
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated / len(bigrams))


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def default_a100_candidate_matrix() -> dict[str, Any]:
    return {
        "schema_version": STEP3_PERFORMANCE_CANDIDATE_SCHEMA_VERSION,
        "adopt": [
            "zero_grad_set_to_none",
            "timing_closure_instrumentation",
            "memory_phase_attribution",
            "grad_finite_gate",
            "checkpoint_global_best_policy",
            "quality_gate",
            "samples_collapse_stats",
        ],
        "probe": [
            "fused_adamw",
            "foreach_adamw",
            "ddp_gradient_as_bucket_view",
            "ddp_bucket_cap_mb",
            "ddp_static_graph_feasibility",
            "torch_compile_model_forward",
            "torch_compile_loss_submodule",
            "torch_compile_reduce_overhead",
            "cuda_graphs_steady_state_capture",
            "allocator_expandable_segments",
            "allocator_max_split_size_mb",
            "chunked_structured_loss",
            "compact_gather_memory_optimization",
            "micro_batch_ladder",
            "worker_prefetch_factor_ladder",
        ],
        "reject": [
            "dali_without_data_bottleneck",
            "flashattention_without_compatible_attention_bottleneck",
            "two_to_four_sparsity_without_quality_plan",
            "activation_checkpointing_as_speed_optimization",
            "bare_torchrun",
            "g2_direct_formal",
            "retired_optimizer_delay_training",
        ],
    }


def batch_ladder_gate_summary(
    *,
    quality_pass: bool,
    grad_inf_count: int,
    timing_closed_ratio: float,
    peak_allocated_gib: float,
    peak_reserved_gib: float,
    safe_allocated_gib: float,
    safe_reserved_gib: float,
    cuda_malloc_retry_count: int = 0,
    throughput_improvement: float = 0.0,
    valid_loss_not_worse: bool = False,
    prefetch_overlap_verified: bool = False,
) -> dict[str, Any]:
    gates = {
        "quality_gate_pass": bool(quality_pass),
        "grad_inf_count_ok": int(grad_inf_count) == 0,
        "timing_closed_ratio_pass": float(timing_closed_ratio) >= 0.95,
        "peak_allocated_safe": float(peak_allocated_gib) < float(safe_allocated_gib),
        "peak_reserved_safe": float(peak_reserved_gib) < float(safe_reserved_gib),
        "cuda_malloc_retry_ok": int(cuda_malloc_retry_count) == 0,
        "throughput_improvement_ok": float(throughput_improvement) >= 0.10,
        "valid_loss_not_worse": bool(valid_loss_not_worse),
        "prefetch_overlap_verified": bool(prefetch_overlap_verified),
    }
    return {
        "schema_version": "odcr_step3_batch_ladder_gate/1",
        "formal_allowed": all(gates.values()),
        "gates": gates,
        "policy": "future_probe_only_until_all_gates_pass_and_user_confirms",
    }
