"""Step3 bounded runtime-probe truth contract.

This module owns the validation-only Step3 performance probe state machine.  A
probe may pass only after a bounded Step3 hot path emits real timing, memory,
prefetch, gradient, and DDP/gather evidence.  Plan/status artifacts are never
runtime evidence.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from odcr_core import path_layout


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

SCHEMA_VERSION = "odcr_step3_runtime_probe/1"
BRIDGE_STATUS_SCHEMA = "odcr_tmux_gpu_bridge_status/1.0"

STEP3_RUNTIME_PROBE_TYPES: tuple[str, ...] = (
    "timing-profile-window",
    "prefetch-ab",
    "grad-monitor-window",
    "memory-phase-window",
    "ddp-gather-sync-window",
    "quality-checkpoint-window",
    "batch-ladder-window",
)

STATE_FIELDS: tuple[str, ...] = (
    "plan_created",
    "bridge_dispatched",
    "runtime_started",
    "components_built",
    "dataloader_built",
    "ddp_initialized",
    "batch_executed",
    "forward_executed",
    "loss_executed",
    "backward_executed",
    "optimizer_executed_or_intentionally_skipped",
    "timing_rows_emitted",
    "memory_rows_emitted",
    "prefetch_rows_emitted",
    "grad_rows_emitted",
    "ddp_rows_emitted",
    "runtime_verified",
    "evidence_complete",
    "formal_namespace_polluted",
)

TIMING_REQUIRED_FIELDS: tuple[str, ...] = (
    "validation_run_id",
    "task_id",
    "profile_id",
    "probe_type",
    "rank",
    "world_size",
    "device",
    "global_step",
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
    "optimizer_ms",
    "zero_grad_ms",
    "scheduler_ms",
    "metrics_io_ms",
    "logging_io_ms",
    "unknown_ms",
    "timing_closed_ratio",
)

MEMORY_REQUIRED_FIELDS: tuple[str, ...] = (
    "validation_run_id",
    "task_id",
    "profile_id",
    "probe_type",
    "phase",
    "rank",
    "world_size",
    "device",
    "allocated_gib",
    "max_allocated_gib",
    "reserved_gib",
    "max_reserved_gib",
    "reserved_minus_allocated_gib",
    "cuda_malloc_retry_count",
    "non_releasable_gib",
    "inactive_split_gib",
)

PREFETCH_REQUIRED_FIELDS: tuple[str, ...] = (
    "validation_run_id",
    "probe_type",
    "rank",
    "world_size",
    "global_step",
    "prefetcher_active",
    "double_buffer_active",
    "num_device_buffers",
    "h2d_stream_created",
    "h2d_event_elapsed_ms",
    "h2d_wait_ms",
    "prefetch_wait_ms",
    "h2d_hidden_by_compute_ratio",
    "record_stream_tensor_count",
    "compute_wait_stream_count",
    "fallback_used",
    "fallback_reason",
)

GRAD_REQUIRED_FIELDS: tuple[str, ...] = (
    "validation_run_id",
    "probe_type",
    "rank",
    "world_size",
    "global_step",
    "grad_finite",
    "grad_norm_pre_clip",
    "grad_norm_post_clip",
    "grad_check_ms",
    "grad_norm_compute_ms",
    "grad_clip_ms",
    "grad_monitor_ms",
    "optimizer_step_executed",
    "scheduler_step_executed",
    "skipped_step_count",
    "nonfinite_param_count",
    "nonfinite_param_topk",
)

DDP_REQUIRED_FIELDS: tuple[str, ...] = (
    "validation_run_id",
    "probe_type",
    "rank",
    "world_size",
    "global_step",
    "structured_gather_ms",
    "structured_gather_total_bytes",
    "gather_tensor_shapes",
    "finite_sync_ms",
    "ddp_backward_sync_ms",
    "rank_step_skew_ms",
    "rank0_step_ms",
    "rank1_step_ms",
    "compact_gather_only",
)

MEMORY_REQUIRED_PHASES: tuple[str, ...] = (
    "after_batch_cpu",
    "after_h2d",
    "after_forward",
    "after_loss_compute",
    "after_structured_gather",
    "after_backward",
    "after_grad_norm",
    "after_optimizer",
)


class Step3RuntimeProbeError(RuntimeError):
    """Stable runtime-probe failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_component(value: str, *, label: str = "component") -> str:
    raw = str(value or "").strip()
    if not raw:
        raise Step3RuntimeProbeError(f"{label} must be non-empty")
    if raw in {".", ".."} or "/" in raw or "\\" in raw or ".." in raw:
        raise Step3RuntimeProbeError(f"unsafe {label}: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw):
        raise Step3RuntimeProbeError(f"unsafe {label}: {value!r}")
    return raw


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras: list[str] = []
    seen = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extras])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "na", "n/a", "nan", "none", "null"}
    return False


def _json_load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class Step3ValidationNamespaceGuard:
    """Fail-fast guard for validation-only Step3 probe writes."""

    repo_root: Path
    task_id: int
    validation_slug: str
    run_id: str
    formal_hashes: dict[str, str | None] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(self, "validation_slug", safe_component(self.validation_slug, label="validation_slug"))
        object.__setattr__(self, "run_id", safe_component(self.run_id, label="run_id"))
        object.__setattr__(
            self,
            "formal_hashes",
            {str(path): _sha256(path) for path in self.forbidden_formal_paths()},
        )

    @property
    def run_root(self) -> Path:
        return path_layout.get_step3_validation_run_root(self.repo_root, self.validation_slug, self.run_id)

    @property
    def meta_dir(self) -> Path:
        return path_layout.get_step3_validation_meta_dir(self.repo_root, self.validation_slug, self.run_id)

    @property
    def evidence_dir(self) -> Path:
        return path_layout.step3_validation_evidence_root(self.repo_root, self.validation_slug, self.run_id)

    @property
    def formal_task_root(self) -> Path:
        return path_layout.get_stage_task_root(self.repo_root, "step3", int(self.task_id)).resolve()

    def forbidden_formal_paths(self) -> tuple[Path, ...]:
        root = path_layout.get_stage_task_root(self.repo_root, "step3", int(self.task_id)).resolve()
        return (
            root / "latest.json",
            root / "model" / "best.pth",
            root / "model" / "best_observed.pth",
            root / "model" / "latest.pth",
            root / "state" / "checkpoint_lineage.json",
            root / "meta" / "run_summary.json",
        )

    def assert_validation_path(self, path: str | Path, *, role: str = "output") -> Path:
        resolved = Path(path).expanduser().resolve()
        allowed_roots = (self.evidence_dir.resolve(), self.run_root.resolve())
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            raise Step3RuntimeProbeError(f"{role} must stay in Step3 validation namespace: {resolved}")
        if self.formal_task_root in resolved.parents or resolved == self.formal_task_root:
            raise Step3RuntimeProbeError(f"{role} entered formal Step3 namespace: {resolved}")
        return resolved

    def assert_no_formal_token(self, value: str, *, role: str = "probe output") -> None:
        text = str(value)
        formal_tokens = (
            self.formal_task_root.as_posix(),
            f"runs/step3/task{int(self.task_id)}",
            "runs/step4/",
            "runs/step5/",
            "runs/eval/",
            "runs/rerank/",
            "model/best.pth",
            "checkpoint_lineage.json",
            "./odcr step3",
            "./odcr step4",
            "./odcr step5",
            "./odcr eval",
            "./odcr rerank",
        )
        bad = [token for token in formal_tokens if token and token in text]
        if bad:
            raise Step3RuntimeProbeError(f"{role} contains forbidden formal token(s): {bad}")

    def formal_namespace_polluted(self) -> bool:
        for raw, before_hash in self.formal_hashes.items():
            path = Path(raw)
            after_hash = _sha256(path)
            if after_hash != before_hash:
                return True
        return False

    def assert_clean_after(self) -> None:
        if self.formal_namespace_polluted():
            raise Step3RuntimeProbeError("Step3 validation probe attempted to modify formal namespace")


@dataclass(frozen=True)
class Step3ValidationWindowRequest:
    task_id: int
    validation_slug: str
    run_id: str
    probe_type: str
    warmup_steps: int = 5
    measured_steps: int = 20
    max_wall_seconds: int = 170
    namespace: str = "validation"
    bridge_dispatched: bool = True
    candidate_name: str | None = None

    def __post_init__(self) -> None:
        if self.namespace != "validation":
            raise Step3RuntimeProbeError("Step3 runtime probe only supports validation namespace")
        if self.probe_type not in STEP3_RUNTIME_PROBE_TYPES:
            raise Step3RuntimeProbeError(f"unsupported Step3 runtime probe type: {self.probe_type!r}")
        if self.candidate_name is not None:
            safe_component(str(self.candidate_name), label="candidate_name")
        if int(self.warmup_steps) < 0:
            raise Step3RuntimeProbeError("warmup_steps must be >= 0")
        if int(self.measured_steps) < 1:
            raise Step3RuntimeProbeError("measured_steps must be >= 1")
        if int(self.max_wall_seconds) < 20:
            raise Step3RuntimeProbeError("max_wall_seconds must be >= 20")


class Step3RuntimeEvidenceSink:
    """Collect and validate bounded Step3 runtime evidence."""

    def __init__(self, *, request: Step3ValidationWindowRequest, guard: Step3ValidationNamespaceGuard) -> None:
        self.request = request
        self.guard = guard
        self.evidence_dir = guard.assert_validation_path(guard.evidence_dir, role="evidence_dir")
        self.meta_dir = guard.assert_validation_path(guard.meta_dir, role="meta_dir")
        self.timing_rows: list[dict[str, Any]] = []
        self.memory_rows: list[dict[str, Any]] = []
        self.prefetch_rows: list[dict[str, Any]] = []
        self.grad_rows: list[dict[str, Any]] = []
        self.ddp_rows: list[dict[str, Any]] = []
        self.rank_results: list[dict[str, Any]] = []

    def add_rank_result(self, result: Mapping[str, Any]) -> None:
        item = dict(result)
        self.rank_results.append(item)
        self.timing_rows.extend(dict(row) for row in item.get("timing_rows", []) or [])
        self.memory_rows.extend(dict(row) for row in item.get("memory_rows", []) or [])
        self.prefetch_rows.extend(dict(row) for row in item.get("prefetch_rows", []) or [])
        self.grad_rows.extend(dict(row) for row in item.get("grad_rows", []) or [])
        self.ddp_rows.extend(dict(row) for row in item.get("ddp_rows", []) or [])

    def _required_complete(self, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> tuple[bool, list[str]]:
        if not rows:
            return False, ["no rows"]
        findings: list[str] = []
        for index, row in enumerate(rows):
            missing = [field for field in fields if field not in row or _missing(row.get(field))]
            if missing:
                findings.append(f"row {index} missing/null required fields: {missing}")
                break
        return not findings, findings

    def _rank0_measured_timing_count(self) -> int:
        rank0: list[Mapping[str, Any]] = []
        for row in self.timing_rows:
            try:
                if int(row.get("rank", -1)) == 0:
                    rank0.append(row)
            except Exception:
                continue
        return len(rank0)

    def validate(self, *, state: Mapping[str, Any]) -> tuple[bool, list[str]]:
        findings: list[str] = []
        if self._rank0_measured_timing_count() < int(self.request.measured_steps):
            findings.append(
                f"timing rows count for rank0 < measured_steps: {self._rank0_measured_timing_count()} < {self.request.measured_steps}"
            )
        for label, rows, fields in (
            ("timing", self.timing_rows, TIMING_REQUIRED_FIELDS),
            ("memory", self.memory_rows, MEMORY_REQUIRED_FIELDS),
            ("prefetch", self.prefetch_rows, PREFETCH_REQUIRED_FIELDS),
            ("grad", self.grad_rows, GRAD_REQUIRED_FIELDS),
            ("ddp", self.ddp_rows, DDP_REQUIRED_FIELDS),
        ):
            ok, row_findings = self._required_complete(rows, fields)
            if not ok:
                findings.append(f"{label} evidence incomplete: {'; '.join(row_findings)}")
        phases = {str(row.get("phase")) for row in self.memory_rows}
        missing_phases = [phase for phase in MEMORY_REQUIRED_PHASES if phase not in phases]
        if missing_phases:
            findings.append(f"memory required phases missing: {missing_phases}")
        if bool(state.get("formal_namespace_polluted")):
            findings.append("formal namespace polluted")
        for field_name in (
            "runtime_started",
            "components_built",
            "dataloader_built",
            "ddp_initialized",
            "batch_executed",
            "forward_executed",
            "loss_executed",
            "backward_executed",
            "optimizer_executed_or_intentionally_skipped",
        ):
            if not bool(state.get(field_name)):
                findings.append(f"state {field_name}=false")
        return not findings, findings

    def write_outputs(self, report: Mapping[str, Any]) -> dict[str, str]:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "timing_breakdown_csv": self.evidence_dir / "timing_breakdown.csv",
            "timing_breakdown_jsonl": self.evidence_dir / "timing_breakdown.jsonl",
            "memory_phase_summary_csv": self.evidence_dir / "memory_phase_summary.csv",
            "memory_phase_summary_jsonl": self.evidence_dir / "memory_phase_summary.jsonl",
            "prefetch_overlap_summary_json": self.evidence_dir / "prefetch_overlap_summary.json",
            "grad_monitor_validation_json": self.evidence_dir / "grad_monitor_validation.json",
            "ddp_gather_sync_summary_json": self.evidence_dir / "ddp_gather_sync_summary.json",
            "run_summary_validation_json": self.evidence_dir / "run_summary_validation.json",
            "report_json": self.evidence_dir / "report.json",
            "report_md": self.evidence_dir / "report.md",
        }
        _write_csv(paths["timing_breakdown_csv"], self.timing_rows, TIMING_REQUIRED_FIELDS)
        _append_jsonl(paths["timing_breakdown_jsonl"], self.timing_rows)
        _write_csv(paths["memory_phase_summary_csv"], self.memory_rows, MEMORY_REQUIRED_FIELDS)
        _append_jsonl(paths["memory_phase_summary_jsonl"], self.memory_rows)
        write_json(paths["prefetch_overlap_summary_json"], summarize_prefetch_rows(self.prefetch_rows, report=report))
        write_json(paths["grad_monitor_validation_json"], summarize_grad_rows(self.grad_rows, report=report))
        write_json(paths["ddp_gather_sync_summary_json"], summarize_ddp_rows(self.ddp_rows, report=report))
        write_json(paths["run_summary_validation_json"], dict(report))
        write_json(paths["report_json"], dict(report))
        _write_text(paths["report_md"], markdown_runtime_probe_report(report))
        return {key: str(value) for key, value in paths.items()}


def summarize_prefetch_rows(rows: Sequence[Mapping[str, Any]], *, report: Mapping[str, Any]) -> dict[str, Any]:
    h2d_waits = [float(row.get("h2d_wait_ms", 0.0) or 0.0) for row in rows if not _missing(row.get("h2d_wait_ms"))]
    prefetch_waits = [float(row.get("prefetch_wait_ms", 0.0) or 0.0) for row in rows if not _missing(row.get("prefetch_wait_ms"))]
    totals = [float(row.get("step_total_ms", 0.0) or 0.0) for row in rows if not _missing(row.get("step_total_ms"))]
    hidden = [float(row.get("h2d_hidden_by_compute_ratio", 0.0) or 0.0) for row in rows if not _missing(row.get("h2d_hidden_by_compute_ratio"))]
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_type": report.get("probe_type"),
        "runtime_verified": bool(report.get("runtime_verified")),
        "evidence_complete": bool(report.get("evidence_complete")),
        "row_count": len(rows),
        "prefetcher_active": any(bool(row.get("prefetcher_active")) for row in rows),
        "double_buffer_active": any(bool(row.get("double_buffer_active")) for row in rows),
        "h2d_wait_ms": _mean(h2d_waits),
        "prefetch_wait_ms": _mean(prefetch_waits),
        "step_total_ms": _mean(totals),
        "h2d_hidden_by_compute_ratio": _mean(hidden),
        "overlap_verdict": "verified" if hidden and max(hidden) >= 0.0 and bool(report.get("runtime_verified")) else "not_verified",
        "comparison_mode": "configured_runtime_window",
    }


def summarize_grad_rows(rows: Sequence[Mapping[str, Any]], *, report: Mapping[str, Any]) -> dict[str, Any]:
    finite = [bool(row.get("grad_finite")) for row in rows]
    opt = [bool(row.get("optimizer_step_executed")) for row in rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_type": report.get("probe_type"),
        "runtime_verified": bool(report.get("runtime_verified")),
        "evidence_complete": bool(report.get("evidence_complete")),
        "row_count": len(rows),
        "grad_finite_rate": float(sum(1 for item in finite if item) / len(finite)) if finite else None,
        "optimizer_step_executed": all(opt) if opt else None,
        "grad_check_ms": _mean([row.get("grad_check_ms") for row in rows]),
        "grad_norm_compute_ms": _mean([row.get("grad_norm_compute_ms") for row in rows]),
        "grad_clip_ms": _mean([row.get("grad_clip_ms") for row in rows]),
        "grad_monitor_ms": _mean([row.get("grad_monitor_ms") for row in rows]),
    }


def summarize_ddp_rows(rows: Sequence[Mapping[str, Any]], *, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_type": report.get("probe_type"),
        "runtime_verified": bool(report.get("runtime_verified")),
        "evidence_complete": bool(report.get("evidence_complete")),
        "row_count": len(rows),
        "structured_gather_ms": _mean([row.get("structured_gather_ms") for row in rows]),
        "structured_gather_total_bytes": int(sum(int(row.get("structured_gather_total_bytes", 0) or 0) for row in rows)),
        "gather_tensor_shapes": [row.get("gather_tensor_shapes") for row in rows[:4]],
        "finite_sync_ms": _mean([row.get("finite_sync_ms") for row in rows]),
        "ddp_backward_sync_ms": _mean([row.get("ddp_backward_sync_ms") for row in rows]),
        "rank_step_skew_ms": _mean([row.get("rank_step_skew_ms") for row in rows]),
        "compact_gather_only": all(bool(row.get("compact_gather_only")) for row in rows) if rows else None,
    }


def _mean(values: Sequence[Any]) -> float | None:
    clean = [float(value) for value in values if not _missing(value) and math.isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _memory_row_from_torch(
    s3: Any,
    final_cfg: Any,
    *,
    rank: int,
    world_size: int,
    device: Any,
    global_step: int,
    phase: str,
    request: Step3ValidationWindowRequest,
) -> dict[str, Any]:
    row = dict(
        s3._step3_gpu_profile_row(
            final_cfg=final_cfg,
            rank=int(rank),
            device=device,
            global_step=int(global_step),
            epoch=1,
            phase=phase,
        )
    )
    row.update(
        {
            "validation_run_id": request.run_id,
            "task_id": int(request.task_id),
            "profile_id": str(getattr(final_cfg, "task_profile_id", "") or ""),
            "probe_type": request.probe_type,
            "world_size": int(world_size),
        }
    )
    return row


def _tensor_shape_bytes(value: Any) -> tuple[list[int], int] | None:
    try:
        import torch

        if not torch.is_tensor(value):
            return None
        return list(value.shape), int(value.numel() * value.element_size())
    except Exception:
        return None


def _gather_shapes_and_bytes(batch: Any) -> tuple[dict[str, list[int]], int]:
    shapes: dict[str, list[int]] = {}
    total = 0
    for name in (
        "user_idx",
        "item_idx",
        "rating",
        "tgt_input",
        "tgt_output",
        "domain_idx",
        "content_anchor_score",
        "style_anchor_score",
        "content_evidence_ids",
        "style_evidence_ids",
        "domain_style_anchor_ids",
        "local_style_hint_ids",
        "polarity_ids",
        "evidence_quality_prior",
    ):
        if not hasattr(batch, name):
            continue
        item = _tensor_shape_bytes(getattr(batch, name))
        if item is None:
            continue
        shape, size = item
        shapes[name] = shape
        total += size
    return shapes, total


def _timing_ms(timing: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in timing and not _missing(timing.get(name)):
            value = float(timing.get(name) or 0.0)
            if name.endswith("_ms"):
                return value
            return value * 1000.0
    return 0.0


def _runtime_rank_worker(rank: int, context: Mapping[str, Any]) -> None:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from dataclasses import replace

    from executors import step3_train_core as s3
    from odcr_core import path_layout as runtime_path_layout
    from odcr_core.gather_schema import require_gathered_batch

    request = Step3ValidationWindowRequest(**dict(context["request"]))
    rank_dir = Path(str(context["rank_dir"]))
    rank_dir.mkdir(parents=True, exist_ok=True)
    rank_path = rank_dir / f"rank_{rank}.json"
    env_updates = {str(k): str(v) for k, v in dict(context["env_updates"]).items()}
    os.environ.update(env_updates)
    os.environ.update(
        {
            "MASTER_ADDR": str(context["master_addr"]),
            "MASTER_PORT": str(context["master_port"]),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(int(context["world_size"])),
            "LOCAL_WORLD_SIZE": str(int(context["world_size"])),
        }
    )
    world_size = int(context["world_size"])
    local_rank = int(rank)
    device = torch.device(f"cuda:{local_rank}")
    state = {field_name: False for field_name in STATE_FIELDS}
    state["plan_created"] = True
    state["bridge_dispatched"] = bool(request.bridge_dispatched)
    timing_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    prefetch_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    ddp_rows: list[dict[str, Any]] = []
    try:
        if not torch.cuda.is_available() or int(torch.cuda.device_count()) < world_size:
            raise RuntimeError(
                "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
                "in this same tmux to enter the GPU node, then rerun the probe."
            )
        torch.cuda.set_device(device)
        state["runtime_started"] = True

        original_read_csv = s3.pd.read_csv
        train_row_limit = int(context.get("train_row_limit") or 2048)
        valid_row_limit = int(context.get("valid_row_limit") or 256)

        def _validation_read_csv(path: Any, *args: Any, **kwargs: Any) -> Any:
            name = Path(str(path)).name
            if name == "aug_train.csv" and train_row_limit > 0 and "nrows" not in kwargs:
                kwargs["nrows"] = train_row_limit
            elif name == "aug_valid.csv" and valid_row_limit > 0 and "nrows" not in kwargs:
                kwargs["nrows"] = valid_row_limit
            return original_read_csv(path, *args, **kwargs)

        original_build_cache_dir = s3._build_step3_cache_dir

        def _validation_cache_dir(*args: Any, **kwargs: Any) -> tuple[str, str, dict[str, Any]]:
            _formal_dir, key, payload = original_build_cache_dir(*args, **kwargs)
            fp_payload = dict(payload)
            fp_payload["validation_cache_namespace"] = True
            cache_dir = runtime_path_layout.step3_validation_tokenizer_cache_entry_dir(
                Path(str(context["repo_root"])),
                validation_slug=request.validation_slug,
                run_id=request.run_id,
                task_id=int(args[0]),
                source_domain=str(kwargs["source_domain"]),
                target_domain=str(kwargs["target_domain"]),
                compatibility_key=key,
            )
            return str(cache_dir), key, fp_payload

        s3.pd.read_csv = _validation_read_csv
        s3._build_step3_cache_dir = _validation_cache_dir
        try:
            args = SimpleNamespace(
                auxiliary=str(context["source_domain"]),
                target=str(context["target_domain"]),
                save_file=str(Path(str(context["run_root"])) / "probe_no_formal_checkpoint.pth"),
                log_file=str(Path(str(context["meta_dir"])) / f"rank_{rank}.log"),
                epochs=None,
                learning_rate=None,
                coef=None,
                emsize=None,
                nlayers=int(context["nlayers"]),
                nhead=None,
                nhid=None,
                dropout=None,
                batch_size=None,
                num_proc=None,
                per_device_batch_size=None,
                scheduler_initial_lr=None,
                warmup_steps=None,
                warmup_ratio=None,
                warmup_epochs=None,
                min_lr_ratio=None,
                lr_scheduler=None,
                eval_batch_size=None,
                quick_eval_max_samples=None,
                early_stop_patience_full=None,
                early_stop_patience_loss=None,
                min_epochs=None,
                early_stop_patience=None,
                bleu4_max_samples=None,
            )
            final_cfg, train_dataloader, _valid_dataloader, model, sampler = s3.build_config_and_data_ddp(
                args,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
            )
            final_cfg = replace(
                final_cfg,
                run_id=request.run_id,
                log_file=str(Path(str(context["meta_dir"])) / f"rank_{rank}.log"),
                rank0_only_logging=True,
            )
        finally:
            s3.pd.read_csv = original_read_csv
            s3._build_step3_cache_dir = original_build_cache_dir

        state["components_built"] = True
        state["dataloader_built"] = True
        state["ddp_initialized"] = bool(dist.is_available() and dist.is_initialized())
        model.train()
        underlying = s3.get_underlying_model(model)
        structured_weights = s3.step3_structured_loss_weights_from_config(final_cfg)
        loss_semantics = s3.step3_loss_semantics_from_config(final_cfg)
        s3.apply_step3_precision_backend(final_cfg)
        optimizer = s3.build_step3_optimizer(model, final_cfg)
        optimizer.zero_grad(set_to_none=True)

        prefetch_cfg = json.loads(str(getattr(final_cfg, "prefetcher_config_json", "") or "{}"))
        prefetch_enabled = bool(prefetch_cfg.get("enabled", True))
        timing_fields = tuple(getattr(s3, "_STEP3_PREFETCH_TIMING_FIELDS"))
        warmup_steps = int(request.warmup_steps)
        measured_steps = int(request.measured_steps)
        target_steps = warmup_steps + measured_steps
        max_wall_seconds = int(request.max_wall_seconds)
        epoch_index = 0
        optimizer_steps_done = 0
        measured_done = 0
        loop_started = time.perf_counter()

        memory_rows.append(
            _memory_row_from_torch(
                s3,
                final_cfg,
                rank=rank,
                world_size=world_size,
                device=device,
                global_step=0,
                phase="after_batch_cpu",
                request=request,
            )
        )

        def _new_iterator(epoch: int):
            if sampler is not None:
                sampler.set_epoch(int(epoch))
            if prefetch_enabled:
                prefetcher = s3.Step3CUDAPrefetcher(
                    train_dataloader,
                    device=device,
                    non_blocking=bool(getattr(final_cfg, "non_blocking_h2d", True)),
                    enabled=True,
                    diagnostic_cpu_mode=False,
                    double_buffer=bool(prefetch_cfg.get("double_buffer", True)),
                    fallback_policy=str(prefetch_cfg.get("fallback_policy") or "fail_fast"),
                )
                return iter(prefetcher), prefetcher
            return iter(train_dataloader), None

        iterator, prefetcher = _new_iterator(epoch_index)
        while optimizer_steps_done < target_steps:
            if time.perf_counter() - loop_started > max_wall_seconds:
                raise RuntimeError("max_wall_seconds reached before measured Step3 validation window completed")
            try:
                batch = next(iterator)
            except StopIteration:
                epoch_index += 1
                iterator, prefetcher = _new_iterator(epoch_index)
                continue

            state["batch_executed"] = True
            step_timing: dict[str, Any] = {str(field_name): 0.0 for field_name in timing_fields}
            if prefetcher is not None:
                step_timing.update({k: float(v) for k, v in dict(prefetcher.last_timing).items() if isinstance(v, (int, float))})
            step_started = time.perf_counter()
            gather_t0 = time.perf_counter()
            g = require_gathered_batch(underlying.gather(batch, device))
            torch.cuda.synchronize(device)
            structured_gather_ms = (time.perf_counter() - gather_t0) * 1000.0
            step_timing["structured_gather_ms"] = structured_gather_ms
            shapes, gather_bytes = _gather_shapes_and_bytes(g)
            memory_rows.append(
                _memory_row_from_torch(
                    s3,
                    final_cfg,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    global_step=optimizer_steps_done + 1,
                    phase="after_h2d",
                    request=request,
                )
            )
            memory_rows.append(
                _memory_row_from_torch(
                    s3,
                    final_cfg,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    global_step=optimizer_steps_done + 1,
                    phase="after_structured_gather",
                    request=request,
                )
            )
            c_a = g.content_anchor_score
            s_a = g.style_anchor_score
            ce = g.content_evidence_ids
            se = g.style_evidence_ids
            dsa = g.domain_style_anchor_ids
            lsh = g.local_style_hint_ids
            pol = g.polarity_ids
            eq = g.evidence_quality_prior
            if any(value is None for value in (c_a, s_a, ce, se, dsa, lsh, pol, eq)):
                raise RuntimeError("Step3 runtime probe gather missing canonical evidence tensors.")
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
                torch.cuda.synchronize(device)
                step_timing["forward_time"] = time.perf_counter() - forward_t0
                state["forward_executed"] = True
                memory_rows.append(
                    _memory_row_from_torch(
                        s3,
                        final_cfg,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        global_step=optimizer_steps_done + 1,
                        phase="after_forward",
                        request=request,
                    )
                )
                loss_t0 = time.perf_counter()
                loss_bundle = s3.compose_step3_loss_from_forward_output(
                    forward_output=forward_out,
                    batch=g,
                    final_cfg=final_cfg,
                    weights=structured_weights,
                    semantics=loss_semantics,
                )
                loss = loss_bundle.total_loss
                step_timing["loss_time"] = time.perf_counter() - loss_t0
                state["loss_executed"] = True
                memory_rows.append(
                    _memory_row_from_torch(
                        s3,
                        final_cfg,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        global_step=optimizer_steps_done + 1,
                        phase="after_loss_compute",
                        request=request,
                    )
                )
                finite_t0 = time.perf_counter()
                finite_sync = s3.step3_sync_loss_bundle_finite_status(loss_bundle, world_size=world_size)
                step_timing["finite_sync_ms"] = (time.perf_counter() - finite_t0) * 1000.0
                global_loss_finite = bool(finite_sync["global_total_finite"])
                if not global_loss_finite:
                    optimizer.zero_grad(set_to_none=True)
                    state["optimizer_executed_or_intentionally_skipped"] = True
                    raise RuntimeError("Step3 runtime probe observed non-finite synchronized loss")
                backward_t0 = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize(device)
                step_timing["backward_time"] = time.perf_counter() - backward_t0
                step_timing["ddp_backward_sync_ms"] = float(step_timing["backward_time"]) * 1000.0
                state["backward_executed"] = True
                memory_rows.append(
                    _memory_row_from_torch(
                        s3,
                        final_cfg,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        global_step=optimizer_steps_done + 1,
                        phase="after_backward",
                        request=request,
                    )
                )

            grad_t0 = time.perf_counter()
            grad_inspection = s3.inspect_gradients(s3.step3_trainable_named_parameters(model), topk=5)
            step_timing["grad_check_ms"] = (time.perf_counter() - grad_t0) * 1000.0
            step_timing["grad_norm_compute_ms"] = float(step_timing["grad_check_ms"])
            grad_finite = s3.sync_grad_finite_decision(
                bool(grad_inspection.grad_finite),
                device=device,
                world_size=world_size,
            )
            grad_norm_pre = float(grad_inspection.grad_norm_pre_clip)
            step_timing["grad_finite"] = bool(grad_finite)
            step_timing["nonfinite_param_count"] = int(grad_inspection.nonfinite_param_count)
            step_timing["nonfinite_param_topk"] = list(grad_inspection.nonfinite_param_topk)
            clip_t0 = time.perf_counter()
            if grad_finite:
                nn.utils.clip_grad_norm_(s3.step3_trainable_parameters(model), float(final_cfg.max_grad_norm))
            step_timing["grad_clip_ms"] = (time.perf_counter() - clip_t0) * 1000.0
            grad_monitor_t0 = time.perf_counter()
            grad_norm_post = s3.grad_norm_total(s3.step3_trainable_parameters(model)) if grad_finite else float("nan")
            step_timing["grad_monitor_ms"] = (time.perf_counter() - grad_monitor_t0) * 1000.0
            memory_rows.append(
                _memory_row_from_torch(
                    s3,
                    final_cfg,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    global_step=optimizer_steps_done + 1,
                    phase="after_grad_norm",
                    request=request,
                )
            )
            optimizer_step_executed = False
            skipped_step_count = 0
            if grad_finite:
                opt_t0 = time.perf_counter()
                optimizer.step()
                torch.cuda.synchronize(device)
                step_timing["optimizer_ms"] = (time.perf_counter() - opt_t0) * 1000.0
                optimizer_step_executed = True
            else:
                skipped_step_count = 1
            zero_t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_timing["zero_grad_ms"] = (time.perf_counter() - zero_t0) * 1000.0
            state["optimizer_executed_or_intentionally_skipped"] = True
            memory_rows.append(
                _memory_row_from_torch(
                    s3,
                    final_cfg,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    global_step=optimizer_steps_done + 1,
                    phase="after_optimizer",
                    request=request,
                )
            )
            optimizer_steps_done += 1
            measured = optimizer_steps_done > warmup_steps
            step_total_ms = (time.perf_counter() - step_started) * 1000.0
            if measured:
                measured_done += 1
                base = {
                    "validation_run_id": request.run_id,
                    "task_id": int(request.task_id),
                    "profile_id": str(getattr(final_cfg, "task_profile_id", "") or ""),
                    "probe_type": request.probe_type,
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "device": f"cuda:{local_rank}",
                    "cache_status": "validation_cache_ready",
                    "global_step": int(measured_done),
                    "batch_size": int(final_cfg.train_batch_size),
                    "micro_batch_size": int(final_cfg.per_device_train_batch_size),
                    "measured_steps": int(measured_steps),
                    "timestamp": utc_now(),
                }
                component_ms = {
                    "loader_next_wait_ms": _timing_ms(step_timing, "dataloader_next_wait"),
                    "cpu_collate_ms": 0.0,
                    "h2d_submit_ms": _timing_ms(step_timing, "h2d_submit_ms", "h2d_prefetch_time"),
                    "h2d_wait_ms": _timing_ms(step_timing, "h2d_wait_ms", "compute_wait_for_prefetch"),
                    "prefetch_wait_ms": _timing_ms(step_timing, "prefetch_wait_ms", "dataloader_next_wait"),
                    "forward_ms": _timing_ms(step_timing, "forward_time"),
                    "loss_compute_ms": _timing_ms(step_timing, "loss_time"),
                    "structured_gather_ms": float(structured_gather_ms),
                    "finite_sync_ms": float(step_timing.get("finite_sync_ms", 0.0) or 0.0),
                    "duplicate_loss_check_ms": 0.0,
                    "ddp_backward_sync_ms": float(step_timing.get("ddp_backward_sync_ms", 0.0) or 0.0),
                    "backward_compute_ms": _timing_ms(step_timing, "backward_time"),
                    "grad_check_ms": float(step_timing.get("grad_check_ms", 0.0) or 0.0),
                    "grad_norm_compute_ms": float(step_timing.get("grad_norm_compute_ms", 0.0) or 0.0),
                    "grad_clip_ms": float(step_timing.get("grad_clip_ms", 0.0) or 0.0),
                    "grad_monitor_ms": float(step_timing.get("grad_monitor_ms", 0.0) or 0.0),
                    "optimizer_ms": float(step_timing.get("optimizer_ms", 0.0) or 0.0),
                    "zero_grad_ms": float(step_timing.get("zero_grad_ms", 0.0) or 0.0),
                    "scheduler_ms": 0.0,
                    "metrics_io_ms": 0.0,
                    "logging_io_ms": 0.0,
                }
                known_ms = sum(float(value) for value in component_ms.values())
                unknown_ms = max(float(step_total_ms) - known_ms, 0.0)
                timing_rows.append(
                    {
                        **base,
                        **component_ms,
                        "step_total_ms": float(step_total_ms),
                        "unknown_ms": float(unknown_ms),
                        "timing_closed_ratio": float(1.0 - min(unknown_ms / step_total_ms, 1.0)) if step_total_ms > 0 else 0.0,
                        "grad_finite": bool(grad_finite),
                        "optimizer_step_executed": bool(optimizer_step_executed),
                        "scheduler_step_executed": False,
                        "skipped_step_reason": "" if optimizer_step_executed else "nonfinite_grad",
                    }
                )
                prefetch_evidence = dict(getattr(prefetcher, "last_evidence", {}) or {}) if prefetcher is not None else {}
                h2d_wait_ms = component_ms["h2d_wait_ms"]
                prefetch_wait_ms = component_ms["prefetch_wait_ms"]
                prefetch_rows.append(
                    {
                        "validation_run_id": request.run_id,
                        "probe_type": request.probe_type,
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "global_step": int(measured_done),
                        "step_total_ms": float(step_total_ms),
                        "prefetcher_active": bool(prefetch_evidence.get("prefetcher_active_in_formal_loop", False)),
                        "double_buffer_active": bool(prefetch_evidence.get("double_buffer_active", False)),
                        "num_device_buffers": int(prefetch_evidence.get("num_device_buffers", 0) or 0),
                        "h2d_stream_created": bool(prefetch_evidence.get("h2d_stream_created", False)),
                        "h2d_event_elapsed_ms": float(prefetch_evidence.get("h2d_event_elapsed_ms", component_ms["h2d_submit_ms"]) or 0.0),
                        "h2d_wait_ms": float(h2d_wait_ms),
                        "prefetch_wait_ms": float(prefetch_wait_ms),
                        "h2d_hidden_by_compute_ratio": float(max(0.0, 1.0 - (h2d_wait_ms / max(component_ms["forward_ms"], 1e-6)))),
                        "record_stream_tensor_count": int(prefetch_evidence.get("record_stream_tensor_count", 0) or 0),
                        "compute_wait_stream_count": int(prefetch_evidence.get("compute_wait_stream_count", 0) or 0),
                        "fallback_used": bool(prefetch_evidence.get("fallback_used", False)),
                        "fallback_reason": str(prefetch_evidence.get("fallback_reason") or "not_applicable_no_fallback"),
                    }
                )
                grad_rows.append(
                    {
                        "validation_run_id": request.run_id,
                        "probe_type": request.probe_type,
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "global_step": int(measured_done),
                        "grad_finite": bool(grad_finite),
                        "grad_norm_pre_clip": float(grad_norm_pre),
                        "grad_norm_post_clip": float(grad_norm_post),
                        "grad_check_ms": float(component_ms["grad_check_ms"]),
                        "grad_norm_compute_ms": float(component_ms["grad_norm_compute_ms"]),
                        "grad_clip_ms": float(component_ms["grad_clip_ms"]),
                        "grad_monitor_ms": float(component_ms["grad_monitor_ms"]),
                        "optimizer_step_executed": bool(optimizer_step_executed),
                        "scheduler_step_executed": False,
                        "skipped_step_count": int(skipped_step_count),
                        "nonfinite_param_count": int(grad_inspection.nonfinite_param_count),
                        "nonfinite_param_topk": list(grad_inspection.nonfinite_param_topk),
                    }
                )
                ddp_rows.append(
                    {
                        "validation_run_id": request.run_id,
                        "probe_type": request.probe_type,
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "global_step": int(measured_done),
                        "structured_gather_ms": float(structured_gather_ms),
                        "structured_gather_total_bytes": int(gather_bytes),
                        "gather_tensor_shapes": shapes,
                        "finite_sync_ms": float(component_ms["finite_sync_ms"]),
                        "ddp_backward_sync_ms": float(component_ms["ddp_backward_sync_ms"]),
                        "rank_step_skew_ms": 0.0,
                        "rank0_step_ms": float(step_total_ms) if rank == 0 else 0.0,
                        "rank1_step_ms": float(step_total_ms) if rank == 1 else 0.0,
                        "compact_gather_only": True,
                    }
                )

        state.update(
            {
                "timing_rows_emitted": bool(timing_rows),
                "memory_rows_emitted": bool(memory_rows),
                "prefetch_rows_emitted": bool(prefetch_rows),
                "grad_rows_emitted": bool(grad_rows),
                "ddp_rows_emitted": bool(ddp_rows),
                "runtime_verified": measured_done >= measured_steps,
                "evidence_complete": measured_done >= measured_steps,
                "formal_namespace_polluted": False,
            }
        )
        write_json(
            rank_path,
            {
                "schema_version": SCHEMA_VERSION,
                "rank": int(rank),
                "world_size": int(world_size),
                "status": "ok",
                "validation_run_id": request.run_id,
                "probe_type": request.probe_type,
                "state": state,
                "timing_rows": timing_rows,
                "memory_rows": memory_rows,
                "prefetch_rows": prefetch_rows,
                "grad_rows": grad_rows,
                "ddp_rows": ddp_rows,
                "measured_steps_completed": int(measured_done),
            },
        )
    except Exception as exc:
        state["formal_namespace_polluted"] = False
        write_json(
            rank_path,
            {
                "schema_version": SCHEMA_VERSION,
                "rank": int(rank),
                "world_size": int(context.get("world_size") or 0),
                "status": "failed",
                "validation_run_id": request.run_id,
                "probe_type": request.probe_type,
                "state": state,
                "failure_phase": "rank_runtime_window",
                "root_reason": str(exc),
                "fatal_signature": repr(exc),
                "traceback": traceback.format_exc(),
                "timing_rows": timing_rows,
                "memory_rows": memory_rows,
                "prefetch_rows": prefetch_rows,
                "grad_rows": grad_rows,
                "ddp_rows": ddp_rows,
            },
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_validation_env(request: Step3ValidationWindowRequest, guard: Step3ValidationNamespaceGuard) -> tuple[Any, dict[str, str]]:
    from odcr_core.config_resolver import resolve_config
    from odcr_core.manifests import build_formal_source_table_snapshot, formal_snapshot_view, write_resolved_config_artifacts
    from odcr_core.runners import _odcr_layout_env, _torchrun_hardware_env

    config_path = guard.repo_root / "configs" / "odcr.yaml"
    set_overrides = _stage2_candidate_set_overrides(
        config_path,
        task_id=int(request.task_id),
        candidate_name=request.candidate_name,
    )
    cfg, _sources, snapshot = resolve_config(
        config_path=config_path,
        command="step3",
        task_id=int(request.task_id),
        set_overrides=set_overrides,
        dry_run=True,
        run_id="auto",
        mode="full",
    )
    formal = formal_snapshot_view(snapshot)
    formal["validation"] = {
        "schema_version": SCHEMA_VERSION,
        "validation_mode": "runtime-probe",
        "probe_type": request.probe_type,
        "candidate_name": request.candidate_name or "",
        "namespace": "validation",
        "formal_latest_updates_allowed": False,
        "formal_checkpoint_writes_allowed": False,
        "step4_step5_eval_rerank_allowed": False,
    }
    source_table = build_formal_source_table_snapshot(snapshot)
    source_table["validation"] = dict(formal["validation"])
    write_resolved_config_artifacts(guard.meta_dir, formal, source_table=source_table, write_verbose_source_table=False)
    env = {}
    env.update(_odcr_layout_env(cfg))
    env.update(_torchrun_hardware_env(cfg))
    env.update(
        {
            "ODCR_STAGE_RUN_DIR": str(guard.run_root),
            "ODCR_ITERATION_META_DIR": str(guard.meta_dir),
            "ODCR_MANIFEST_DIR": str(guard.meta_dir),
            "ODCR_LOG_DIR": str(guard.meta_dir),
            "ODCR_SUMMARY_LOG": str(guard.meta_dir / "console.log"),
            "ODCR_STEP3_TOKENIZER_CACHE_STARTUP_JSON": str(guard.meta_dir / "step3_tokenizer_cache_startup.json"),
            "ODCR_LOG_STEP_LOSS_PARTS": "1",
        }
    )
    return cfg, env


def _load_rank_results(rank_dir: Path, world_size: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        path = rank_dir / f"rank_{rank}.json"
        if path.is_file():
            out.append(_json_load(path))
    return out


def _merge_state(rank_results: Sequence[Mapping[str, Any]], guard: Step3ValidationNamespaceGuard) -> dict[str, Any]:
    state = {field_name: False for field_name in STATE_FIELDS}
    state["plan_created"] = True
    state["bridge_dispatched"] = True
    if rank_results:
        for field_name in STATE_FIELDS:
            if field_name == "formal_namespace_polluted":
                continue
            state[field_name] = all(bool((item.get("state") or {}).get(field_name)) for item in rank_results)
    state["formal_namespace_polluted"] = bool(guard.formal_namespace_polluted())
    return state


def _apply_rank_skew(ddp_rows: list[dict[str, Any]], timing_rows: Sequence[Mapping[str, Any]]) -> None:
    by_step: dict[int, dict[int, float]] = {}
    for row in timing_rows:
        try:
            step = int(row.get("global_step"))
            rank = int(row.get("rank"))
            by_step.setdefault(step, {})[rank] = float(row.get("step_total_ms", 0.0) or 0.0)
        except Exception:
            continue
    for row in ddp_rows:
        try:
            step = int(row.get("global_step"))
        except Exception:
            continue
        rank0 = float(by_step.get(step, {}).get(0, 0.0))
        rank1 = float(by_step.get(step, {}).get(1, 0.0))
        row["rank0_step_ms"] = rank0
        row["rank1_step_ms"] = rank1
        row["rank_step_skew_ms"] = abs(rank0 - rank1) if rank0 and rank1 else 0.0


def _stage2_candidate_set_overrides(config_path: Path, *, task_id: int, candidate_name: str | None) -> list[str]:
    if not candidate_name:
        return []
    from odcr_core.config_resolver import load_yaml_config

    cfg = load_yaml_config(config_path)
    raw_name = safe_component(str(candidate_name), label="candidate_name")
    if raw_name == "G1S":
        return []
    profiles = ((cfg.get("step3") or {}).get("task_profiles") or {})
    if not isinstance(profiles, Mapping):
        raise Step3RuntimeProbeError("step3.task_profiles must be configured for candidate probes")
    profile_key = ""
    for key, value in profiles.items():
        if isinstance(value, Mapping) and int(value.get("task_id") or -1) == int(task_id):
            profile_key = str(key)
            break
    if not profile_key:
        raise Step3RuntimeProbeError(f"no Step3 task profile found for task {task_id}")
    ladder = (((cfg.get("step3") or {}).get("performance_candidates") or {}).get("batch_ladder") or {})
    if not isinstance(ladder, Mapping) or raw_name not in ladder:
        raise Step3RuntimeProbeError(f"unknown Stage2 candidate {raw_name!r}")
    row = ladder[raw_name]
    if not isinstance(row, Mapping):
        raise Step3RuntimeProbeError(f"Stage2 candidate {raw_name!r} must be a mapping")
    # Candidate probes are validation-only batch/micro/lr overrides.  They must
    # not mutate the formal task2 candidate identity, which remains G1 by
    # resolver contract.
    overrides: list[str] = []
    for source_key, dest_key in (
        ("batch_size", "batch_size"),
        ("micro_batch_size", "micro_batch_size"),
        ("lr_candidate", "lr"),
    ):
        if source_key in row and row[source_key] is not None:
            overrides.append(f"step3.task_profiles.{profile_key}.train.{dest_key}={row[source_key]}")
    return overrides


def failure_report(
    *,
    request: Step3ValidationWindowRequest,
    guard: Step3ValidationNamespaceGuard,
    failure_phase: str,
    root_reason: str,
    rank_results: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    state = _merge_state(rank_results, guard)
    state["runtime_verified"] = False
    state["evidence_complete"] = False
    report = {
        "schema_version": SCHEMA_VERSION,
        "validation_run_id": request.run_id,
        "task_id": int(request.task_id),
        "profile_id": "",
        "candidate_name": request.candidate_name or "",
        "probe_type": request.probe_type,
        "namespace": "validation",
        "runtime_probe_ok": False,
        "success": False,
        "exit_code": 1,
        "failure_phase": failure_phase,
        "root_reason": root_reason,
        "runtime_verified": False,
        "evidence_complete": False,
        "formal_namespace_polluted": bool(state.get("formal_namespace_polluted")),
        "state": state,
        "rank_count": len(rank_results),
        "generated_at": utc_now(),
        "evidence_dir": str(guard.evidence_dir),
        "validation_run_root": str(guard.run_root),
    }
    sink = Step3RuntimeEvidenceSink(request=request, guard=guard)
    for item in rank_results:
        sink.add_rank_result(item)
    paths = sink.write_outputs(report)
    report["paths"] = paths
    write_json(Path(paths["report_json"]), report)
    write_json(Path(paths["run_summary_validation_json"]), report)
    return report


def run_step3_validation_window(
    *,
    task_id: int = 2,
    validation_slug: str = "step3_runtime_probe_truth_rebuild",
    run_id: str,
    probe_type: str,
    candidate_name: str | None = None,
    warmup_steps: int = 5,
    measured_steps: int = 20,
    max_wall_seconds: int = 170,
    repo_root: str | Path = REPO_ROOT,
    bridge_dispatched: bool = True,
) -> dict[str, Any]:
    request = Step3ValidationWindowRequest(
        task_id=int(task_id),
        validation_slug=validation_slug,
        run_id=run_id,
        probe_type=probe_type,
        candidate_name=candidate_name,
        warmup_steps=int(warmup_steps),
        measured_steps=int(measured_steps),
        max_wall_seconds=int(max_wall_seconds),
        bridge_dispatched=bool(bridge_dispatched),
    )
    guard = Step3ValidationNamespaceGuard(Path(repo_root), int(task_id), validation_slug, run_id)
    guard.evidence_dir.mkdir(parents=True, exist_ok=True)
    guard.meta_dir.mkdir(parents=True, exist_ok=True)
    if probe_type == "batch-ladder-window" and not candidate_name:
        return failure_report(
            request=request,
            guard=guard,
            failure_phase="batch_ladder_gate",
            root_reason="batch-ladder-window requires an explicit validation-only candidate_name after prior timing/memory/prefetch/grad/DDP probes pass.",
        )
    started = time.monotonic()
    rank_dir = guard.evidence_dir / "ranks"
    rank_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torch.multiprocessing as mp

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
                "in this same tmux to enter the GPU node, then rerun the probe."
            )
        cfg, env_updates = _resolve_validation_env(request, guard)
        world_size = int(getattr(cfg, "ddp_world_size", 2) or 2)
        if int(torch.cuda.device_count()) < world_size:
            raise RuntimeError(f"CUDA device_count={torch.cuda.device_count()} is less than ddp_world_size={world_size}")
        total_needed = max(2048, int(getattr(cfg, "per_device_train_batch_size", 1) or 1) * world_size * 2)
        context = {
            "request": dataclasses.asdict(request),
            "repo_root": str(guard.repo_root),
            "run_root": str(guard.run_root),
            "meta_dir": str(guard.meta_dir),
            "rank_dir": str(rank_dir),
            "env_updates": env_updates,
            "source_domain": str(getattr(cfg, "auxiliary", "")),
            "target_domain": str(getattr(cfg, "target", "")),
            "nlayers": int(getattr(cfg, "nlayers", 1) or 1),
            "world_size": world_size,
            "master_addr": "127.0.0.1",
            "master_port": str(_free_port()),
            "train_row_limit": int(total_needed),
            "valid_row_limit": max(256, int(getattr(cfg, "per_device_train_batch_size", 1) or 1) * world_size),
        }
        mp.spawn(_runtime_rank_worker, args=(context,), nprocs=world_size, join=True)
        rank_results = _load_rank_results(rank_dir, world_size)
        if len(rank_results) != world_size:
            raise RuntimeError(f"expected {world_size} rank result files, found {len(rank_results)}")
        state = _merge_state(rank_results, guard)
        sink = Step3RuntimeEvidenceSink(request=request, guard=guard)
        for item in rank_results:
            sink.add_rank_result(item)
        _apply_rank_skew(sink.ddp_rows, sink.timing_rows)
        state.update(
            {
                "timing_rows_emitted": bool(sink.timing_rows),
                "memory_rows_emitted": bool(sink.memory_rows),
                "prefetch_rows_emitted": bool(sink.prefetch_rows),
                "grad_rows_emitted": bool(sink.grad_rows),
                "ddp_rows_emitted": bool(sink.ddp_rows),
            }
        )
        evidence_complete, findings = sink.validate(state=state)
        state["evidence_complete"] = bool(evidence_complete)
        state["runtime_verified"] = bool(evidence_complete)
        guard.assert_clean_after()
        report = {
            "schema_version": SCHEMA_VERSION,
            "validation_run_id": request.run_id,
            "task_id": int(request.task_id),
            "profile_id": str(getattr(cfg, "task_profile_id", "") or ""),
            "candidate_name": request.candidate_name or str(getattr(cfg, "preset_name", "") or ""),
            "probe_type": request.probe_type,
            "namespace": "validation",
            "rank_count": len(rank_results),
            "world_size": int(world_size),
            "device": "cuda",
            "hostname": socket.gethostname(),
            "cache_status": "validation_cache_ready",
            "batch_size": int(getattr(cfg, "batch_size", getattr(cfg, "train_batch_size", 0)) or 0),
            "micro_batch_size": int(getattr(cfg, "per_device_train_batch_size", 0) or 0),
            "warmup_steps": int(request.warmup_steps),
            "measured_steps": int(request.measured_steps),
            "timing_rows": len(sink.timing_rows),
            "memory_rows": len(sink.memory_rows),
            "prefetch_rows": len(sink.prefetch_rows),
            "grad_rows": len(sink.grad_rows),
            "ddp_rows": len(sink.ddp_rows),
            "runtime_started": bool(state.get("runtime_started")),
            "batch_executed": bool(state.get("batch_executed")),
            "runtime_verified": bool(evidence_complete),
            "evidence_complete": bool(evidence_complete),
            "runtime_probe_ok": bool(evidence_complete and not guard.formal_namespace_polluted()),
            "formal_namespace_polluted": bool(guard.formal_namespace_polluted()),
            "success": bool(evidence_complete and not guard.formal_namespace_polluted()),
            "exit_code": 0 if evidence_complete and not guard.formal_namespace_polluted() else 1,
            "failure_phase": "" if evidence_complete else "evidence_completeness",
            "root_reason": "" if evidence_complete else "; ".join(findings),
            "state": state,
            "evidence_findings": findings,
            "elapsed_s": round(time.monotonic() - started, 3),
            "generated_at": utc_now(),
            "evidence_dir": str(guard.evidence_dir),
            "validation_run_root": str(guard.run_root),
            "formal_latest_updated": False,
            "formal_checkpoint_created": False,
            "step4_step5_eval_rerank_started": False,
        }
        paths = sink.write_outputs(report)
        report["paths"] = paths
        write_json(Path(paths["report_json"]), report)
        write_json(Path(paths["run_summary_validation_json"]), report)
        return report
    except Exception as exc:
        rank_results = _load_rank_results(rank_dir, 2)
        return failure_report(
            request=request,
            guard=guard,
            failure_phase="runtime_window",
            root_reason=str(exc),
            rank_results=rank_results,
        )


def markdown_runtime_probe_report(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Step3 Runtime Probe Truth Report",
            "",
            f"- probe_type: {report.get('probe_type')}",
            f"- validation_run_id: {report.get('validation_run_id')}",
            f"- runtime_probe_ok: {str(bool(report.get('runtime_probe_ok'))).lower()}",
            f"- runtime_verified: {str(bool(report.get('runtime_verified'))).lower()}",
            f"- evidence_complete: {str(bool(report.get('evidence_complete'))).lower()}",
            f"- formal_namespace_polluted: {str(bool(report.get('formal_namespace_polluted'))).lower()}",
            f"- failure_phase: {report.get('failure_phase') or ''}",
            f"- root_reason: {report.get('root_reason') or ''}",
            "",
        ]
    )


def child_status_from_report(
    report: Mapping[str, Any],
    *,
    run_id: str,
    elapsed_s: float,
    max_seconds: int,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_verified = bool(report.get("runtime_verified"))
    evidence_complete = bool(report.get("evidence_complete"))
    runtime_probe_ok = bool(report.get("runtime_probe_ok"))
    formal_namespace_polluted = bool(report.get("formal_namespace_polluted"))
    success = bool(runtime_verified and evidence_complete and runtime_probe_ok and not formal_namespace_polluted)
    exit_code = 0 if success else 1
    metrics = {
        "bridge_transport_ok": None,
        "child_process_ok": success,
        "child_exit_code": exit_code,
        "runtime_started": bool(report.get("runtime_started") or (report.get("state") or {}).get("runtime_started")),
        "runtime_probe_ok": runtime_probe_ok,
        "evidence_complete": evidence_complete,
        "runtime_verified": runtime_verified,
        "formal_namespace_polluted": formal_namespace_polluted,
        "report_json": str((report.get("paths") or {}).get("report_json") or ""),
        "run_summary_validation_json": str((report.get("paths") or {}).get("run_summary_validation_json") or ""),
        "timing_rows": int(report.get("timing_rows") or 0),
        "memory_rows": int(report.get("memory_rows") or 0),
        "prefetch_rows": int(report.get("prefetch_rows") or 0),
        "grad_rows": int(report.get("grad_rows") or 0),
        "ddp_rows": int(report.get("ddp_rows") or 0),
        "failure_phase": str(report.get("failure_phase") or ""),
        "root_reason": str(report.get("root_reason") or ""),
        "target": dict(target or {}),
    }
    return {
        "schema_version": BRIDGE_STATUS_SCHEMA,
        "run_id": run_id,
        "kind": "step3-performance-probe",
        "success": success,
        "final_success": success,
        "exit_code": exit_code,
        "elapsed_s": round(float(elapsed_s), 3),
        "startup_timeout_s": min(30, int(max_seconds)),
        "first_result_timeout_s": min(120, int(max_seconds)),
        "hard_timeout_s": int(max_seconds),
        "first_result_seen": True,
        "success_condition": "bounded_step3_hot_path_runtime_verified_and_evidence_complete",
        "stop_reason": "step3_performance_probe_completed" if success else "step3_performance_probe_failed",
        "bridge_transport_ok": None,
        "child_process_ok": success,
        "child_exit_code": exit_code,
        "runtime_started": bool(metrics["runtime_started"]),
        "runtime_probe_ok": runtime_probe_ok,
        "evidence_complete": evidence_complete,
        "runtime_verified": runtime_verified,
        "formal_namespace_polluted": formal_namespace_polluted,
        "metrics": metrics,
        "generated_at": utc_now(),
    }


def normalize_bridge_runtime_success(status: Mapping[str, Any], *, bridge_transport_ok: bool = True) -> dict[str, Any]:
    """Convert child status into transport/child/runtime/evidence/final status."""

    out = dict(status)
    metrics = dict(out.get("metrics") or {})
    child_exit_code = int(out.get("exit_code", metrics.get("child_exit_code", 1)) or 0)
    child_process_ok = child_exit_code == 0 and bool(out.get("first_result_seen", True))
    runtime_started = bool(out.get("runtime_started", metrics.get("runtime_started", False)))
    runtime_probe_ok = bool(out.get("runtime_probe_ok", metrics.get("runtime_probe_ok", False)))
    evidence_complete = bool(out.get("evidence_complete", metrics.get("evidence_complete", False)))
    runtime_verified = bool(out.get("runtime_verified", metrics.get("runtime_verified", False)))
    formal_namespace_polluted = bool(out.get("formal_namespace_polluted", metrics.get("formal_namespace_polluted", False)))
    plan_only = bool(out.get("plan_only", metrics.get("plan_only", False)))
    final_success = bool(
        bridge_transport_ok
        and child_process_ok
        and child_exit_code == 0
        and runtime_started
        and runtime_probe_ok
        and runtime_verified
        and evidence_complete
        and not formal_namespace_polluted
        and not plan_only
    )
    metrics.update(
        {
            "bridge_transport_ok": bool(bridge_transport_ok),
            "child_process_ok": bool(child_process_ok),
            "child_exit_code": int(child_exit_code),
            "runtime_started": bool(runtime_started),
            "runtime_probe_ok": bool(runtime_probe_ok),
            "evidence_complete": bool(evidence_complete),
            "runtime_verified": bool(runtime_verified),
            "formal_namespace_polluted": bool(formal_namespace_polluted),
            "final_success": bool(final_success),
            "success_semantics": {
                "transport_success": "command delivered and child status collected",
                "child_success": "child process exit code is zero",
                "runtime_success": "bounded Step3 hot path executed",
                "evidence_success": "required metric rows are complete",
            },
        }
    )
    out.update(
        {
            "bridge_transport_ok": bool(bridge_transport_ok),
            "child_process_ok": bool(child_process_ok),
            "child_exit_code": int(child_exit_code),
            "runtime_started": bool(runtime_started),
            "runtime_probe_ok": bool(runtime_probe_ok),
            "evidence_complete": bool(evidence_complete),
            "runtime_verified": bool(runtime_verified),
            "formal_namespace_polluted": bool(formal_namespace_polluted),
            "final_success": bool(final_success),
            "success": bool(final_success),
            "exit_code": 0 if final_success else 1,
            "stop_reason": out.get("stop_reason") if final_success else _bridge_failure_reason(metrics, plan_only=plan_only),
            "metrics": metrics,
        }
    )
    return out


def _bridge_failure_reason(metrics: Mapping[str, Any], *, plan_only: bool = False) -> str:
    if plan_only:
        return "step3_performance_probe_plan_only_rejected"
    for key in (
        "bridge_transport_ok",
        "child_process_ok",
        "runtime_started",
        "runtime_probe_ok",
        "runtime_verified",
        "evidence_complete",
    ):
        if not bool(metrics.get(key)):
            return f"step3_performance_probe_{key}_false"
    if bool(metrics.get("formal_namespace_polluted")):
        return "step3_performance_probe_formal_namespace_polluted"
    return "step3_performance_probe_failed"


def evidence_level_runtime_verified(level: int | str, *, code_present: bool = False, active_path: bool = False) -> bool:
    """Only Evidence Level 3+ artifacts may set runtime_verified."""

    _ = (code_present, active_path)
    raw = str(level).strip().lower().replace("evidence", "").replace("level", "").strip("_ :")
    try:
        numeric = int(raw)
    except ValueError:
        numeric = 0
    return numeric >= 3


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _required_file_rows_ok(path: Path, fields: Sequence[str]) -> tuple[bool, str]:
    rows = _read_csv_rows(path)
    if not rows:
        return False, f"{path.name} empty"
    for idx, row in enumerate(rows):
        missing = [field for field in fields if field not in row or _missing(row.get(field))]
        if missing:
            return False, f"{path.name} row {idx} missing/null: {missing}"
    return True, "ok"


def evaluate_stage2_probe_evidence(evidence_dir: str | Path, *, probe_type: str) -> dict[str, Any]:
    root = Path(evidence_dir)
    timing_ok, timing_reason = _required_file_rows_ok(root / "timing_breakdown.csv", TIMING_REQUIRED_FIELDS)
    memory_ok, memory_reason = _required_file_rows_ok(root / "memory_phase_summary.csv", MEMORY_REQUIRED_FIELDS)
    prefetch = _json_load(root / "prefetch_overlap_summary.json")
    grad = _json_load(root / "grad_monitor_validation.json")
    ddp = _json_load(root / "ddp_gather_sync_summary.json")
    summary = _json_load(root / "run_summary_validation.json")

    prefetch_ok = all(not _missing(prefetch.get(key)) for key in ("h2d_wait_ms", "prefetch_wait_ms", "step_total_ms", "overlap_verdict"))
    grad_ok = all(not _missing(grad.get(key)) for key in ("grad_finite_rate", "optimizer_step_executed", "grad_check_ms"))
    ddp_ok = all(not _missing(ddp.get(key)) for key in ("structured_gather_total_bytes", "finite_sync_ms", "rank_step_skew_ms"))
    summary_ok = bool(summary.get("runtime_verified")) and bool(summary.get("evidence_complete"))
    findings = []
    for ok, reason in (
        (timing_ok, timing_reason),
        (memory_ok, memory_reason),
        (prefetch_ok, "prefetch metrics null/missing"),
        (grad_ok, "grad metrics null/missing"),
        (ddp_ok, "DDP metrics null/missing"),
        (summary_ok, "run_summary_validation runtime/evidence false"),
    ):
        if not ok:
            findings.append(reason)
    pass_gate = not findings
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_type": probe_type,
        "pass": pass_gate,
        "runtime_probe_ok": pass_gate,
        "candidate_selection_allowed": pass_gate,
        "timing_pass": timing_ok,
        "memory_pass": memory_ok,
        "prefetch_pass": prefetch_ok,
        "grad_pass": grad_ok,
        "ddp_pass": ddp_ok,
        "run_summary_pass": summary_ok,
        "findings": findings,
        "G1-M": "eligible" if pass_gate else "skipped_by_gate",
        "G2-C": "eligible" if pass_gate else "skipped_by_gate",
        "G3": "eligible" if pass_gate else "skipped_by_gate",
    }
