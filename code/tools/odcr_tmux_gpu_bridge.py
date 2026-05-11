#!/usr/bin/env python3
"""Runtime-first tmux GPU executor for ODCR validation probes.

This is a developer/Codex tool, not an ODCR user entrypoint.  It never creates
or manages GPU allocations.  It can only target a user-created tmux pane that is
already inside a live Slurm GPU step.  The bridge sends one generated command
file to that pane, records transport/child/runtime status, and keeps validation
outputs outside formal namespaces unless the user explicitly confirms a formal
run in a future request.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from odcr_core.step3_runtime_probe import normalize_bridge_runtime_success
from odcr_core.evidence_level import E3_GPU_TRANSPORT, E4_GPU_SHARD_FORWARD_BOUNDED, evidence_level_rank, parse_evidence_level

AI_ANALYSIS = REPO_ROOT / "AI_analysis"
RAW_LOG_DIR = AI_ANALYSIS / "01_raw_logs"
SUMMARY_DIR = AI_ANALYSIS / "04_phase_summaries"
REPORT_DIR = AI_ANALYSIS / "05_final_reports"
RUNTIME_DIR = AI_ANALYSIS / "runtime"
D4C_PYTHON = Path("/public/home/zhangliml/miniconda3/envs/D4C/bin/python")
SAFE_TMPDIR = Path("/public/home/zhangliml/tmp/codex")
BRIDGE_STATUS_SCHEMA = "odcr_tmux_gpu_bridge_status/1.0"
GPU_PANE_SCHEMA = "odcr_gpu_pane/1.0"
# Retired stale state file trust: every mode fresh-discovers and validates the current pane.
TMUX_PANE_FORMAT = "\t".join(
    (
        "#{session_name}",
        "#{window_index}",
        "#{pane_index}",
        "#{pane_id}",
        "#{pane_pid}",
        "#{pane_current_command}",
        "#{pane_current_path}",
        "#{pane_active}",
        "#{pane_dead}",
        "#{pane_in_mode}",
    )
)


class BridgeError(RuntimeError):
    """Expected fail-fast bridge error with a stable stop reason."""

    def __init__(self, message: str, *, stop_reason: str = "target_invalid", details: Any = None) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.details = details


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner:
    """Small wrapper so tests can mock every external command."""

    def run(self, args: Sequence[str], *, timeout: float | None = None) -> CommandResult:
        proc = subprocess.run(
            list(args),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return CommandResult(tuple(str(part) for part in args), proc.returncode, proc.stdout, proc.stderr)


@dataclass(frozen=True)
class OperationSpec:
    name: str
    auto_timeout_s: int
    max_timeout_s: int
    startup_timeout_s: int
    first_result_timeout_s: int
    success_condition: str
    stop_reason_success: str
    send_allowed: bool


SEND_METHODS = (
    "pane-id-literal-enter",
    "target-name-literal-enter",
    "pane-id-literal-cm",
    "buffer-paste-enter",
)
DEFAULT_SEND_METHOD = "pane-id-literal-enter"
UNLOCK_OPERATIONS = ("copy-mode-cancel", "escape", "q")


OPERATION_SPECS: dict[str, OperationSpec] = {
    "discover": OperationSpec(
        "discover",
        20,
        60,
        5,
        20,
        "exactly_one_valid_gpu_tmux_target_or_fail_fast",
        "target_validated",
        False,
    ),
    "validate-only": OperationSpec(
        "validate-only",
        60,
        120,
        10,
        60,
        "target_unique_slurm_gpu_repo_nvidia_smi_and_torch_cuda_2plus_valid",
        "target_validated",
        True,
    ),
    "unlock-pane": OperationSpec(
        "unlock-pane",
        20,
        60,
        5,
        20,
        "pane_mode_cancelled_and_target_revalidated",
        "pane_unlocked",
        False,
    ),
    "cuda-probe": OperationSpec(
        "cuda-probe",
        45,
        120,
        10,
        45,
        "torch_cuda_available_device_count_and_gpu_name",
        "first_cuda_probe_completed",
        True,
    ),
    "marker-probe": OperationSpec(
        "marker-probe",
        20,
        20,
        5,
        15,
        "begin_end_marker_and_send_ok_written_to_ai_analysis",
        "marker_probe_completed",
        True,
    ),
    "preprocess-dryrun": OperationSpec(
        "preprocess-dryrun",
        90,
        180,
        15,
        90,
        "preprocess_b_or_c_dry_run_exit_zero",
        "preprocess_dryrun_completed",
        True,
    ),
    "bge-smoke": OperationSpec(
        "bge-smoke",
        240,
        360,
        20,
        240,
        "local_bge_single_batch_encode_completed",
        "first_bge_encode_completed",
        True,
    ),
    "micro-benchmark": OperationSpec(
        "micro-benchmark",
        300,
        420,
        20,
        300,
        "one_warmup_and_one_measured_batch_completed",
        "first_micro_benchmark_completed",
        True,
    ),
    "real-data-probe": OperationSpec(
        "real-data-probe",
        180,
        180,
        20,
        180,
        "ai_analysis_real_data_short_window_probe_completed",
        "real_data_probe_completed",
        True,
    ),
    "step3-startup-validation": OperationSpec(
        "step3-startup-validation",
        180,
        180,
        60,
        180,
        "step3_startup_validation_2rank_cache_then_nccl_completed",
        "step3_startup_validation_completed",
        True,
    ),
    "step3-performance-probe": OperationSpec(
        "step3-performance-probe",
        180,
        180,
        30,
        120,
        "bounded_step3_hot_path_runtime_verified_and_evidence_complete",
        "step3_performance_probe_completed",
        True,
    ),
    "repo-command": OperationSpec(
        "repo-command",
        180,
        900,
        20,
        180,
        "repo_local_command_exit_zero_and_namespace_clean",
        "repo_command_completed",
        True,
    ),
    "repo-script": OperationSpec(
        "repo-script",
        180,
        900,
        20,
        900,
        "repo_local_script_exit_zero_and_namespace_clean",
        "repo_script_completed",
        True,
    ),
    "repo-module": OperationSpec(
        "repo-module",
        180,
        900,
        20,
        180,
        "repo_local_module_exit_zero_and_namespace_clean",
        "repo_module_completed",
        True,
    ),
    "command-file": OperationSpec(
        "command-file",
        180,
        900,
        20,
        180,
        "generated_command_file_exit_zero_and_namespace_clean",
        "command_file_completed",
        True,
    ),
    "long-run": OperationSpec(
        "long-run",
        0,
        604800,
        20,
        60,
        "detached_managed_launcher_started_and_status_files_written",
        "managed_launcher_started",
        True,
    ),
    "collect": OperationSpec(
        "collect",
        20,
        60,
        5,
        20,
        "log_status_and_end_marker_collected",
        "collected",
        False,
    ),
}
GLOBAL_MAX_TIMEOUT_S = 900


@dataclass(frozen=True)
class ResolvedTimeouts:
    startup_timeout_s: int
    first_result_timeout_s: int
    hard_timeout_s: int | None
    detached: bool = False
    emergency_timeout_s: int | None = None


@dataclass(frozen=True)
class PaneCandidate:
    socket: str
    session: str
    target: str
    pane_id: str
    pane_pid: int
    pane_command: str
    cwd: str
    active: bool
    dead: bool
    in_mode: bool
    srun_pid: int | None
    srun_command: str
    job_id: str
    step_id: str
    node: str
    gpu: str
    step_state: str
    job_state: str

    def to_state(self) -> dict[str, Any]:
        now = _now_iso()
        return {
            "schema_version": GPU_PANE_SCHEMA,
            "socket": self.socket,
            "target": self.target,
            "pane_id": self.pane_id,
            "pane_pid": self.pane_pid,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "node": self.node,
            "repo": str(REPO_ROOT),
            "created_at": now,
            "validated_at": now,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "socket": self.socket,
            "session": self.session,
            "target": self.target,
            "pane_id": self.pane_id,
            "pane_pid": self.pane_pid,
            "pane_command": self.pane_command,
            "cwd": self.cwd,
            "active": self.active,
            "dead": self.dead,
            "in_mode": self.in_mode,
            "srun_pid": self.srun_pid,
            "srun_command": self.srun_command,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "node": self.node,
            "gpu": self.gpu,
            "step_state": self.step_state,
            "job_state": self.job_state,
        }


@dataclass(frozen=True)
class InvalidPane:
    socket: str
    target: str
    pane_id: str
    reason: str
    pane_command: str = ""
    cwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: tuple[PaneCandidate, ...]
    invalid: tuple[InvalidPane, ...]
    sockets_considered: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "invalid_candidates": [item.to_dict() for item in self.invalid],
            "sockets_considered": list(self.sockets_considered),
        }


@dataclass(frozen=True)
class GeneratedPaths:
    run_id: str
    script: Path
    log: Path
    status: Path
    summary: Path
    report: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "script": str(self.script),
            "log": str(self.log),
            "status": str(self.status),
            "summary": str(self.summary),
            "report": str(self.report),
        }


@dataclass
class BridgeOptions:
    mode: str
    stage: str | None = None
    probe_stage: str = "b"
    benchmark_kind: str | None = None
    embed_batch_size: int = 512
    read_chunk_rows: int = 100_000
    group_shard_size: int = 4_096
    workers: int = 2
    bf16_enabled: bool = True
    tf32_enabled: bool = True
    grouped_text_cache_enabled: bool = True
    task_id: int = 2
    smoke_candidate: str | None = None
    candidate_name: str | None = None
    worker_profile: str | None = None
    max_batches: int = 1
    max_steps: int = 1
    warmup_optimizer_steps: int = 10
    measured_optimizer_steps: int = 50
    max_wall_seconds: int = 900
    max_epochs: int = 2
    max_optimizer_steps: int = 2000
    validate_every_steps: int = 500
    validate_every_epoch: bool = True
    timeout: str = "auto"
    dry_run: bool = False
    no_send: bool = False
    strict: bool = False
    socket: str | None = None
    target: str | None = None
    run_id: str | None = None
    send_method: str = DEFAULT_SEND_METHOD
    validation_slug: str = "step3_tmux_gpu_bridge_startup_validation_closeout"
    performance_probe_type: str = "timing-profile-window"
    command_argv: tuple[str, ...] = ()
    script_path: str | None = None
    module_name: str | None = None
    command_file: str | None = None
    output_dir: str | None = None
    user_confirmed_formal: bool = False


STEP4_BRIDGE_MAX_BOUNDED_LIMIT = 32768
STEP4_PREFLIGHT_REQUIRED_ARTIFACTS = (
    "preflight_summary.json",
    "rcr_distribution.json",
    "required_fields_check.json",
    "manifest_preview.json",
    "index_contract_preview.json",
    "lineage_preview.json",
    "cpu_gpu_utilization_snapshot.json",
)


def step4_validation_allowed_output_roots() -> tuple[Path, ...]:
    return (
        AI_ANALYSIS / "06_probe_evidence",
        AI_ANALYSIS / "07_runtime_evidence",
        REPO_ROOT / "runs" / "step3_validation",
        REPO_ROOT / "runs" / "step4_preflight",
        REPO_ROOT / "runs" / "step4_validation",
    )


def command_file_allowed_roots() -> tuple[Path, ...]:
    return step4_validation_allowed_output_roots()
FORMAL_NAMESPACE_FIXED_WATCH_PATHS = (
    "runs/step4/task2",
    "runs/step4/task2/latest.json",
    "runs/step5/task2/latest.json",
    "runs/eval/task2/latest.json",
    "runs/rerank/task2/latest.json",
)
FORMAL_NAMESPACE_WATCH_GLOBS = (
    "runs/step3/task*/latest.json",
    "runs/step3/task*/*/model/best.pth",
    "runs/step3/task*/*/model/best_observed.pth",
    "runs/step3/task*/*/model/latest.pth",
    "runs/step3/task*/*/state/checkpoint_lineage.json",
    "runs/step4/task2/*",
    "runs/step4/task2/*/meta/run_summary.json",
    "runs/step4/task2/*/" + "odcr_routing_train" + ".csv",
    "runs/step4/task2/*/meta/stage_status.json",
    "runs/step4/task2/*/meta/manifest.json",
    "runs/step4/task2/*/meta/index_contract.json",
    "runs/step5/task2/*",
    "runs/eval/task2/*",
    "runs/rerank/task2/*",
)


@dataclass(frozen=True)
class BridgeCommandClassification:
    allowed: bool
    reason: str
    stop_reason: str = "forbidden_mode"
    command_kind: str = "unknown"
    stage: str | None = None
    operation: str | None = None
    task: int | None = None
    validation_namespace: str | None = None
    bounded_limit_name: str | None = None
    bounded_limit_value: int | None = None
    output_roots: tuple[str, ...] = ()
    evidence_roots: tuple[str, ...] = ()
    runtime_evidence_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class BridgeCommandPolicy:
    """Closed command semantics for repo-local GPU bridge execution."""

    allocation_commands = {"srun", "sbatch", "scancel", "odcr-enter-gpu"}
    background_commands = {"nohup", "disown"}
    destructive_commands = {"rm", "mv", "cp"}
    odcr_denied_stages = {"step5", "eval", "rerank"}
    step4_bounded_flags = ("--max-samples", "--max-batches", "--max-rows")
    step4_mode_flags = ("--preflight", "--prepare-cache")
    step4_formal_flags = {"--write-latest", "--formal", "--full", "--run-full"}

    @classmethod
    def classify_repo_command(
        cls,
        argv: Sequence[str],
        *,
        output_dir: Path | None = None,
        user_confirmed_formal: bool = False,
    ) -> BridgeCommandClassification:
        parts = tuple(str(item) for item in argv if str(item) != "")
        if not parts:
            return cls._deny("missing command", stop_reason="target_invalid")
        command_name = Path(parts[0]).name
        if command_name in cls.background_commands or any(Path(part).name in cls.background_commands for part in parts):
            return cls._deny("destructive/allocation/background command rejected: background command")
        if any(Path(part).name in cls.allocation_commands for part in parts):
            return cls._deny("destructive/allocation/background command rejected: allocation command")
        if cls._contains_background_token(parts):
            return cls._deny("destructive/allocation/background command rejected: background token")
        if command_name == "rm":
            return cls._deny("destructive/allocation/background command rejected: rm")
        if command_name in {"mv", "cp"} and cls._targets_formal_namespace(parts[1:]):
            return cls._deny("formal namespace target rejected: mv/cp to formal namespace", stop_reason="formal_namespace_blocked")

        entry = cls._odcr_entry(parts)
        if entry is not None:
            stage_index = entry + 1
            if len(parts) <= stage_index:
                return cls._allow("odcr help command", command_kind="odcr")
            stage = parts[stage_index]
            stage_args = parts[stage_index + 1 :]
            if stage in cls.odcr_denied_stages:
                return cls._deny(f"{stage} command rejected", command_kind="odcr", stage=stage)
            if stage == "step4":
                return cls._classify_step4(stage_args, output_dir=output_dir, user_confirmed_formal=user_confirmed_formal)
            if stage == "step3" and "--dry-run" not in stage_args and not user_confirmed_formal:
                return cls._deny("formal Step3 command rejected", command_kind="odcr", stage=stage)
            return cls._allow("non-formal ODCR command allowed", command_kind="odcr", stage=stage)

        if command_name in {"bash", "sh"}:
            if len(parts) < 2 or parts[1] in {"-c", "--command"}:
                return cls._deny("arbitrary shell rejected")
            script_path = cls._resolve_repo_path(parts[1])
            if script_path is None or not script_path.is_file():
                return cls._deny("arbitrary shell rejected")
            return cls._allow("repo-local shell script allowed", command_kind="repo_script")

        if command_name.startswith("python"):
            return cls._allow("repo-local python command allowed", command_kind="repo_python")

        candidate = cls._resolve_repo_path(parts[0])
        if candidate is not None and candidate.is_file():
            return cls._allow("repo-local executable allowed", command_kind="repo_executable")
        return cls._deny("repo-command must start with ./odcr, python, or a repo-local executable/script")

    @classmethod
    def is_allowed_step4_validation_command(cls, argv: Sequence[str]) -> bool:
        classification = cls.classify_repo_command(argv)
        return classification.allowed and classification.stage == "step4"

    @classmethod
    def _classify_step4(
        cls,
        stage_args: Sequence[str],
        *,
        output_dir: Path | None,
        user_confirmed_formal: bool,
    ) -> BridgeCommandClassification:
        del user_confirmed_formal
        args = tuple(stage_args)
        operation_flags = [flag for flag in cls.step4_mode_flags if flag in args]
        task = cls._int_flag(args, "--task")
        if not operation_flags:
            return cls._deny("formal Step4 command rejected: missing --preflight/--prepare-cache", command_kind="odcr", stage="step4", task=task)
        if len(operation_flags) > 1:
            return cls._deny("Step4 command rejected: choose only one of --preflight/--prepare-cache", command_kind="odcr", stage="step4", task=task)
        operation = operation_flags[0].lstrip("-").replace("-", "_")
        for flag in cls.step4_formal_flags:
            if flag in args:
                return cls._deny("formal Step4 command rejected", command_kind="odcr", stage="step4", operation=operation, task=task)
        if cls._flag_value(args, "--mode") == "full":
            return cls._deny("formal Step4 command rejected: --mode full", command_kind="odcr", stage="step4", operation=operation, task=task)
        namespace = cls._flag_value(args, "--validation-namespace")
        if namespace is None:
            return cls._deny("missing --validation-namespace", command_kind="odcr", stage="step4", operation=operation, task=task)
        namespace_problem = validate_step4_validation_namespace(namespace)
        if namespace_problem:
            return cls._deny(namespace_problem, command_kind="odcr", stage="step4", operation=operation, task=task, validation_namespace=namespace)
        limit_name, limit_value, limit_error = cls._bounded_limit(args)
        if limit_error:
            return cls._deny(limit_error, command_kind="odcr", stage="step4", operation=operation, task=task, validation_namespace=namespace)
        assert limit_name is not None and limit_value is not None
        output_values = tuple(
            value
            for flag in ("--output", "--output-dir", "--run-dir")
            for value in [cls._flag_value(args, flag)]
            if value
        )
        for value in output_values:
            if cls._targets_formal_namespace((value,)):
                return cls._deny(
                    "formal namespace target rejected",
                    stop_reason="formal_namespace_blocked",
                    command_kind="odcr",
                    stage="step4",
                    operation=operation,
                    task=task,
                    validation_namespace=namespace,
                )
            output_path = cls._resolve_output_path(value)
            if not _path_under_any(output_path, step4_validation_allowed_output_roots()):
                return cls._deny(
                    "output root rejected: Step4 validation output must stay under AI_analysis, runs/step4_preflight, or runs/step4_validation",
                    stop_reason="formal_namespace_blocked",
                    command_kind="odcr",
                    stage="step4",
                    operation=operation,
                    task=task,
                    validation_namespace=namespace,
                )
        if output_dir is not None and not _path_under_any(output_dir, step4_validation_allowed_output_roots()):
            return cls._deny(
                "output root rejected: bridge runtime output_dir is outside validation roots",
                stop_reason="formal_namespace_blocked",
                command_kind="odcr",
                stage="step4",
                operation=operation,
                task=task,
                validation_namespace=namespace,
            )
        task_id = int(task or 2)
        evidence_roots = (
            str((REPO_ROOT / "runs" / "step4_preflight" / f"task{task_id}" / namespace).resolve()),
            str((REPO_ROOT / "runs" / "step4_validation" / f"task{task_id}" / namespace).resolve()),
        )
        return cls._allow(
            "bounded Step4 validation command allowed",
            command_kind="odcr",
            stage="step4",
            operation=operation,
            task=task_id,
            validation_namespace=namespace,
            bounded_limit_name=limit_name,
            bounded_limit_value=limit_value,
            output_roots=tuple(str(Path(value).as_posix()) for value in output_values),
            evidence_roots=evidence_roots,
            runtime_evidence_required=operation == "preflight",
        )

    @classmethod
    def _odcr_entry(cls, parts: Sequence[str]) -> int | None:
        if parts[0] == "./odcr":
            return 0
        if Path(parts[0]).name.startswith("python"):
            if len(parts) >= 2 and parts[1] == "code/odcr.py":
                return 1
            if len(parts) >= 3 and parts[1] == "-m" and parts[2] == "code.odcr":
                return 2
        return None

    @classmethod
    def _bounded_limit(cls, args: Sequence[str]) -> tuple[str | None, int | None, str | None]:
        for flag in cls.step4_bounded_flags:
            raw = cls._flag_value(args, flag)
            if raw is None:
                continue
            try:
                value = int(raw)
            except ValueError:
                return flag, None, f"invalid bounded limit for {flag}"
            if value <= 0:
                return flag, value, "bounded limit must be positive"
            if value > STEP4_BRIDGE_MAX_BOUNDED_LIMIT:
                return flag, value, f"bounded limit exceeds bridge cap {STEP4_BRIDGE_MAX_BOUNDED_LIMIT}"
            return flag, value, None
        return None, None, "missing bounded limit"

    @staticmethod
    def _flag_value(args: Sequence[str], flag: str) -> str | None:
        for idx, item in enumerate(args):
            if item == flag:
                if idx + 1 >= len(args):
                    return ""
                return str(args[idx + 1])
            prefix = flag + "="
            if item.startswith(prefix):
                return item[len(prefix) :]
        return None

    @classmethod
    def _int_flag(cls, args: Sequence[str], flag: str) -> int | None:
        value = cls._flag_value(args, flag)
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @staticmethod
    def _contains_background_token(parts: Sequence[str]) -> bool:
        return any(part in {"&", "&&"} or part.endswith("&") for part in parts)

    @staticmethod
    def _targets_formal_namespace(values: Sequence[str]) -> bool:
        joined = " ".join(str(value).replace("\\", "/").lower() for value in values)
        formal_terms = (
            "runs/step4/task",
            "runs/step5",
            "runs/eval",
            "runs/rerank",
            "latest.json",
            "odcr_routing_train.csv",
            "/data/",
            "/merged/",
        )
        return any(term in joined for term in formal_terms)

    @staticmethod
    def _resolve_repo_path(raw: str) -> Path | None:
        try:
            candidate = Path(str(raw)).expanduser()
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            else:
                candidate = candidate.resolve()
            return candidate if _is_relative_to(candidate, REPO_ROOT) else None
        except OSError:
            return None

    @staticmethod
    def _resolve_output_path(raw: str) -> Path:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        else:
            path = path.resolve()
        return path

    @staticmethod
    def _allow(reason: str, **kwargs: Any) -> BridgeCommandClassification:
        return BridgeCommandClassification(True, reason, stop_reason="allowed", **kwargs)

    @staticmethod
    def _deny(reason: str, *, stop_reason: str = "forbidden_mode", **kwargs: Any) -> BridgeCommandClassification:
        return BridgeCommandClassification(False, reason, stop_reason=stop_reason, **kwargs)


def validate_step4_validation_namespace(namespace: str) -> str | None:
    raw = str(namespace or "").strip()
    lowered = raw.lower()
    if not raw:
        return "missing --validation-namespace"
    if raw in {".", ".."} or ".." in raw or "/" in raw or "\\" in raw:
        return "bad validation namespace rejected"
    if "latest" in lowered or "formal" in lowered or "runs/step4/task" in lowered:
        return "bad validation namespace rejected"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", raw):
        return "bad validation namespace rejected"
    return None


def _path_under_any(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _now_run_id(kind: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"bridge_{stamp}_{_safe_run_component(kind)}"


def _safe_run_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "run"


def validate_run_id(run_id: str) -> str:
    raw = str(run_id or "").strip()
    if not raw:
        raise BridgeError("run_id must be non-empty", stop_reason="target_invalid")
    if len(raw) > 96:
        raise BridgeError("run_id is too long", stop_reason="target_invalid")
    if raw in {".", ".."} or ".." in raw or "/" in raw or "\\" in raw:
        raise BridgeError("run_id must not contain path traversal", stop_reason="target_invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw):
        raise BridgeError("run_id contains unsafe characters", stop_reason="target_invalid")
    return raw


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _default_runtime_output_dir(run_id: str) -> Path:
    return AI_ANALYSIS / "06_probe_evidence" / "tmux_gpu_runtime_executor" / validate_run_id(run_id)


def resolve_runtime_output_dir(raw: str | None, run_id: str, *, user_confirmed_formal: bool = False) -> Path:
    base = Path(raw).expanduser() if raw else _default_runtime_output_dir(run_id)
    if not base.is_absolute():
        base = (REPO_ROOT / base).resolve()
    else:
        base = base.resolve()
    allowed = tuple(root.resolve() for root in step4_validation_allowed_output_roots())
    if not user_confirmed_formal and not any(base == root or root in base.parents for root in allowed):
        raise BridgeError(
            "validation GPU output_dir must stay under AI_analysis/06_probe_evidence, AI_analysis/07_runtime_evidence, runs/step3_validation, runs/step4_preflight, or runs/step4_validation",
            stop_reason="formal_namespace_blocked",
            details={"output_dir": str(base), "allowed_roots": [str(root) for root in allowed]},
        )
    return base


def resolve_repo_local_path(raw: str, *, label: str, must_exist: bool = True) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise BridgeError(f"{label} must be non-empty", stop_reason="target_invalid")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not _is_relative_to(path, REPO_ROOT):
        raise BridgeError(f"{label} must be inside the ODCR repository", stop_reason="forbidden_mode", details=str(path))
    if must_exist and not path.is_file():
        raise BridgeError(f"{label} does not exist: {path}", stop_reason="target_invalid")
    return path


def validate_module_name(module_name: str) -> str:
    raw = str(module_name or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", raw):
        raise BridgeError(f"unsafe Python module name: {module_name!r}", stop_reason="forbidden_mode")
    if raw.split(".", 1)[0] not in {"odcr_core", "executors", "tools", "configs"}:
        raise BridgeError(
            "repo-module must target a repo-local code package",
            stop_reason="forbidden_mode",
            details={"module": raw},
        )
    return raw


def resolve_timeouts(mode: str, requested: str | int = "auto") -> ResolvedTimeouts:
    spec = OPERATION_SPECS[mode]
    if mode == "long-run":
        if requested == "auto":
            emergency: int | None = None
        else:
            try:
                emergency = int(requested)
            except (TypeError, ValueError) as exc:
                raise BridgeError(f"invalid timeout {requested!r}", stop_reason="target_invalid") from exc
            if emergency <= 0:
                raise BridgeError("long-run emergency timeout must be positive when set", stop_reason="target_invalid")
            if emergency > spec.max_timeout_s:
                raise BridgeError(
                    f"timeout {emergency}s exceeds {mode} emergency safety limit {spec.max_timeout_s}s",
                    stop_reason="hard_timeout",
                    details={"mode": mode, "timeout": emergency, "max_timeout_s": spec.max_timeout_s},
                )
        return ResolvedTimeouts(
            startup_timeout_s=spec.startup_timeout_s,
            first_result_timeout_s=spec.first_result_timeout_s,
            hard_timeout_s=None,
            detached=True,
            emergency_timeout_s=emergency,
        )
    if requested == "auto":
        hard = spec.auto_timeout_s
    else:
        try:
            hard = int(requested)
        except (TypeError, ValueError) as exc:
            raise BridgeError(f"invalid timeout {requested!r}", stop_reason="target_invalid") from exc
    if hard <= 0:
        raise BridgeError("timeout must be positive", stop_reason="target_invalid")
    if hard > GLOBAL_MAX_TIMEOUT_S or hard > spec.max_timeout_s:
        raise BridgeError(
            f"timeout {hard}s exceeds {mode} safety limit {spec.max_timeout_s}s",
            stop_reason="hard_timeout",
            details={"mode": mode, "timeout": hard, "max_timeout_s": spec.max_timeout_s},
        )
    return ResolvedTimeouts(
        startup_timeout_s=min(spec.startup_timeout_s, hard),
        first_result_timeout_s=min(spec.first_result_timeout_s, hard),
        hard_timeout_s=hard,
    )


def make_generated_paths(run_id: str) -> GeneratedPaths:
    run_id = validate_run_id(run_id)
    return GeneratedPaths(
        run_id=run_id,
        script=RAW_LOG_DIR / f"tmux_bridge_{run_id}.sh",
        log=RAW_LOG_DIR / f"tmux_bridge_{run_id}.log",
        status=RAW_LOG_DIR / f"tmux_bridge_{run_id}.status.json",
        summary=SUMMARY_DIR / f"tmux_bridge_{run_id}_summary.md",
        report=REPORT_DIR / f"tmux_bridge_{run_id}_report.md",
    )


def _ensure_analysis_dirs() -> None:
    for path in (RAW_LOG_DIR, SUMMARY_DIR, REPORT_DIR, RUNTIME_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _mode_status_result_path(paths: GeneratedPaths, mode: str) -> Path:
    if mode in {"step3-startup-validation", "step3-performance-probe"}:
        return paths.status.with_name(paths.status.name + ".child.json")
    return paths.status


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def formal_namespace_watch_paths(repo_root: Path = REPO_ROOT) -> list[Path]:
    root = repo_root.resolve()
    paths: set[Path] = {root / rel for rel in FORMAL_NAMESPACE_FIXED_WATCH_PATHS}
    for pattern in FORMAL_NAMESPACE_WATCH_GLOBS:
        paths.update(root.glob(pattern))
    return sorted(paths, key=lambda p: p.as_posix())


def snapshot_formal_namespace(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve()
    snapshot: dict[str, dict[str, Any]] = {}
    for path in formal_namespace_watch_paths(root):
        try:
            rel = path.resolve().relative_to(root).as_posix()
        except ValueError:
            rel = str(path.resolve())
        if path.exists():
            stat_result = path.stat()
            snapshot[rel] = {
                "exists": True,
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "size": int(stat_result.st_size),
                "mtime_ns": int(stat_result.st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        else:
            snapshot[rel] = {"exists": False}
    return snapshot


def formal_namespace_polluted(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> bool:
    return dict(before) != dict(after)


def _json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def step4_preflight_evidence_dirs(
    classification: BridgeCommandClassification | Mapping[str, Any] | None,
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    dirs: list[Path] = []
    if output_dir is not None:
        dirs.append(output_dir.resolve())
    payload = classification.to_dict() if isinstance(classification, BridgeCommandClassification) else dict(classification or {})
    if payload.get("stage") == "step4" and payload.get("operation") == "preflight":
        namespace = str(payload.get("validation_namespace") or "")
        task = int(payload.get("task") or 2)
        if namespace:
            dirs.extend(
                [
                    (REPO_ROOT / "runs" / "step4_preflight" / f"task{task}" / namespace).resolve(),
                    (REPO_ROOT / "runs" / "step4_validation" / f"task{task}" / namespace).resolve(),
                ]
            )
    seen: set[str] = set()
    unique: list[Path] = []
    for path in dirs:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _step4_evidence_level_from(*payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        if payload.get("evidence_level"):
            return parse_evidence_level(payload)
    return ""


def _step4_runtime_evidence_ok(
    *,
    summary: Mapping[str, Any],
    gpu_snapshot: Mapping[str, Any],
    evidence_complete: bool,
    formal_flags_ok: bool,
) -> tuple[bool, dict[str, Any]]:
    try:
        evidence_level = _step4_evidence_level_from(summary, gpu_snapshot)
        evidence_level_ok = evidence_level_rank(evidence_level) >= evidence_level_rank(E4_GPU_SHARD_FORWARD_BOUNDED)
    except Exception:
        evidence_level = ""
        evidence_level_ok = False
    checks = {
        "evidence_level": evidence_level,
        "evidence_level_ok": evidence_level_ok,
        "gpu_runtime_evidence": summary.get("gpu_runtime_evidence") is True or gpu_snapshot.get("gpu_runtime_evidence") is True,
        "actual_gpu_forward_executed": summary.get("actual_gpu_forward_executed") is True
        or gpu_snapshot.get("actual_gpu_forward_executed") is True,
        "actual_model_loaded_on_gpu": summary.get("actual_model_loaded_on_gpu") is True
        or gpu_snapshot.get("actual_model_loaded_on_gpu") is True,
        "force_gpu_forward": summary.get("force_gpu_forward") is True or gpu_snapshot.get("force_gpu_forward") is True,
        "cuda_available": gpu_snapshot.get("cuda_available") is True,
        "evidence_complete": bool(evidence_complete),
        "formal_pollution": not bool(formal_flags_ok),
        "formal_pollution_false": bool(formal_flags_ok),
    }
    runtime_ok = all(
        bool(checks[key])
        for key in (
            "evidence_level_ok",
            "gpu_runtime_evidence",
            "actual_gpu_forward_executed",
            "actual_model_loaded_on_gpu",
            "force_gpu_forward",
            "cuda_available",
            "evidence_complete",
            "formal_pollution_false",
        )
    )
    return runtime_ok, checks


def parse_step4_preflight_evidence(
    *,
    output_dir: Path | None = None,
    classification: BridgeCommandClassification | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_dirs = step4_preflight_evidence_dirs(classification, output_dir=output_dir)
    for directory in candidate_dirs:
        artifacts = {name: directory / name for name in STEP4_PREFLIGHT_REQUIRED_ARTIFACTS}
        if not any(path.is_file() for path in artifacts.values()):
            continue
        missing = [name for name, path in artifacts.items() if not path.is_file()]
        payloads = {name: _json_or_empty(path) for name, path in artifacts.items() if path.is_file()}
        summary = payloads.get("preflight_summary.json", {})
        distribution = payloads.get("rcr_distribution.json", {})
        required = payloads.get("required_fields_check.json", {})
        lineage = payloads.get("lineage_preview.json", {})
        gpu_snapshot = payloads.get("cpu_gpu_utilization_snapshot.json", {})
        expected_namespace = ""
        expected_max_samples = None
        if classification is not None:
            class_payload = classification.to_dict() if isinstance(classification, BridgeCommandClassification) else dict(classification)
            expected_namespace = str(class_payload.get("validation_namespace") or "")
            expected_max_samples = class_payload.get("bounded_limit_value")
        namespace_ok = not expected_namespace or str(summary.get("validation_namespace") or "") == expected_namespace
        try:
            sample_count = int(summary.get("sample_count") or distribution.get("sample_count") or 0)
        except (TypeError, ValueError):
            sample_count = 0
        try:
            max_samples = int(summary.get("max_samples") or expected_max_samples or 0)
        except (TypeError, ValueError):
            max_samples = 0
        bounded_ok = sample_count > 0 and (max_samples <= 0 or sample_count <= max_samples)
        rcr_counts_present = all(
            key in distribution
            for key in (
                "route_scorer_count",
                "route_explainer_count",
                "train_keep_count",
                "confidence_bucket_distribution",
                "sample_weight_hint",
            )
        )
        formal_flags_ok = summary.get("formal_latest_write") is False and summary.get("formal_export_write") is False
        upstream_ok = bool(summary.get("upstream_step3_run_id") or lineage.get("upstream_step3_run_id") or lineage.get("lineage_hash"))
        required_ok = bool(required.get("passed")) and not required.get("missing")
        evidence_complete = bool(
            not missing
            and namespace_ok
            and bounded_ok
            and rcr_counts_present
            and formal_flags_ok
            and upstream_ok
            and required_ok
        )
        runtime_evidence_ok, runtime_checks = _step4_runtime_evidence_ok(
            summary=summary,
            gpu_snapshot=gpu_snapshot,
            evidence_complete=evidence_complete,
            formal_flags_ok=formal_flags_ok,
        )
        gpu_transport_ok = bool(gpu_snapshot.get("cuda_available"))
        gpu_runtime_observed = bool(runtime_checks["gpu_runtime_evidence"])
        return {
            "schema_version": "odcr_step4_bridge_preflight_evidence/1",
            "evidence_level": runtime_checks["evidence_level"] or (E3_GPU_TRANSPORT if gpu_transport_ok else ""),
            "candidate_dirs": [str(path) for path in candidate_dirs],
            "evidence_dir": str(directory),
            "artifact_paths": {name: str(path) for name, path in artifacts.items()},
            "missing_artifacts": missing,
            "evidence_complete": evidence_complete,
            "runtime_evidence_ok": runtime_evidence_ok,
            "gpu_transport_ok": gpu_transport_ok,
            "gpu_runtime_observed": gpu_runtime_observed,
            "not_step4_runtime_evidence": not runtime_evidence_ok,
            "validation_namespace": summary.get("validation_namespace"),
            "sample_count": sample_count,
            "max_samples": max_samples,
            "route_scorer_count": distribution.get("route_scorer_count"),
            "route_explainer_count": distribution.get("route_explainer_count"),
            "train_keep_count": distribution.get("train_keep_count"),
            "confidence_bucket_distribution": distribution.get("confidence_bucket_distribution"),
            "sample_weight_hint_stats": distribution.get("sample_weight_hint"),
            "formal_latest_write": summary.get("formal_latest_write"),
            "formal_export_write": summary.get("formal_export_write"),
            "upstream_step3_run_id": summary.get("upstream_step3_run_id") or lineage.get("upstream_step3_run_id"),
            "checks": {
                "namespace_ok": namespace_ok,
                "bounded_ok": bounded_ok,
                "rcr_counts_present": rcr_counts_present,
                "formal_flags_ok": formal_flags_ok,
                "upstream_ok": upstream_ok,
                "required_fields_ok": required_ok,
                "child_process_ok_required_by_bridge_status": True,
                **runtime_checks,
            },
        }
    if output_dir is not None:
        report = _json_or_empty(output_dir / "report.json")
        if report.get("cuda_available") is True:
            return {
                "schema_version": "odcr_step4_bridge_preflight_evidence/1",
                "evidence_level": E3_GPU_TRANSPORT,
                "candidate_dirs": [str(path) for path in candidate_dirs],
                "evidence_dir": str(output_dir),
                "artifact_paths": {"report": str(output_dir / "report.json")},
                "missing_artifacts": list(STEP4_PREFLIGHT_REQUIRED_ARTIFACTS),
                "evidence_complete": False,
                "runtime_evidence_ok": False,
                "gpu_transport_ok": True,
                "gpu_runtime_observed": False,
                "not_step4_runtime_evidence": True,
                "checks": {
                    "cuda_available": True,
                    "cuda_probe_alone_is_not_step4_runtime_evidence": True,
                    "child_process_ok_required_by_bridge_status": True,
                },
            }
    return {
        "schema_version": "odcr_step4_bridge_preflight_evidence/1",
        "evidence_level": "",
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "evidence_dir": "",
        "artifact_paths": {},
        "missing_artifacts": list(STEP4_PREFLIGHT_REQUIRED_ARTIFACTS),
        "evidence_complete": False,
        "runtime_evidence_ok": False,
        "gpu_transport_ok": False,
        "gpu_runtime_observed": False,
        "not_step4_runtime_evidence": True,
        "checks": {},
    }


def _default_socket_exists(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except OSError:
        return False


def _env_socket_dirs() -> list[Path]:
    raw = os.environ.get("ODCR_GPU_TMUX_SOCKET_DIRS", "")
    dirs: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            dirs.append(Path(part))
    return dirs


def candidate_socket_paths(uid: int | None = None) -> list[Path]:
    user_id = os.getuid() if uid is None else uid
    bases = [
        Path(f"/tmp/tmux-{user_id}") / "odcr_gpu",
        Path(f"/public/home/zhangliml/tmp/codex/tmux-{user_id}") / "odcr_gpu",
        Path(f"/run/user/{user_id}/codex-tmp/tmux-{user_id}") / "odcr_gpu",
        Path(f"/run/user/{user_id}/tmux-{user_id}") / "odcr_gpu",
    ]
    for extra in _env_socket_dirs():
        bases.append(extra if extra.name == "odcr_gpu" else extra / "odcr_gpu")
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in bases:
        key = str(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    return ordered


def _split_tmux_row(row: str) -> tuple[str, str, str, str, int, str, str, bool, bool, bool] | None:
    parts = row.rstrip("\n").split("\t")
    if len(parts) != 10:
        return None
    session, window_index, pane_index, pane_id, pane_pid, command, cwd, active, dead, in_mode = parts
    try:
        pid = int(pane_pid)
    except ValueError:
        return None
    return (
        session,
        window_index,
        pane_index,
        pane_id,
        pid,
        command,
        cwd,
        active == "1",
        dead == "1",
        in_mode == "1",
    )


def _parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]+)=([^\s]+)", text):
        values[key] = value
    return values


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    args: str


def _parse_ps(text: str) -> dict[int, ProcInfo]:
    procs: dict[int, ProcInfo] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        procs[pid] = ProcInfo(pid, ppid, parts[2])
    return procs


def _iter_descendants(root_pid: int, procs: Mapping[int, ProcInfo]) -> Iterable[ProcInfo]:
    children: dict[int, list[ProcInfo]] = {}
    for proc in procs.values():
        children.setdefault(proc.ppid, []).append(proc)
    stack = list(children.get(root_pid, ()))
    while stack:
        proc = stack.pop(0)
        yield proc
        stack.extend(children.get(proc.pid, ()))


def _is_srun_command(command: str) -> bool:
    return bool(re.search(r"(^|\s|/)srun(\s|$)", command))


def _parse_job_id(command: str) -> str | None:
    match = re.search(r"--jobid(?:=|\s+)(\d+)", command)
    return match.group(1) if match else None


def _requested_gpu_count_from_srun(command: str) -> int | None:
    """Best-effort GPU count from direct interactive srun commands."""
    patterns = (
        r"--gres(?:=|\s+)gpu(?::[A-Za-z0-9_.-]+)?:(\d+)",
        r"--gpus(?:=|\s+)(\d+)",
        r"--gres(?:=|\s+)[^\s,]*gpu[^\s,]*:(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, command)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _gpu_count_from_tres(text: str) -> int | None:
    matches = re.findall(r"(?:gres/)?gpu(?:[:=][A-Za-z0-9_.-]+)?[:=](\d+)", text, flags=re.IGNORECASE)
    counts: list[int] = []
    for value in matches:
        try:
            counts.append(int(value))
        except ValueError:
            continue
    return max(counts) if counts else None


def _gpu_text(step_values: Mapping[str, str], job_values: Mapping[str, str]) -> str:
    parts = [
        step_values.get("TRES", ""),
        step_values.get("TresPerNode", ""),
        job_values.get("TRES", ""),
        job_values.get("TresPerNode", ""),
    ]
    return ",".join(part for part in parts if part)


def _has_gpu_tres(text: str) -> bool:
    lowered = text.lower()
    return "gres/gpu" in lowered or "trespernode=gpu" in lowered or "gpu:" in lowered


def _is_gpu_node_or_tres(node: str, gpu: str) -> bool:
    return node.lower().startswith("gpu") or _has_gpu_tres(gpu)


def _target_matches(candidate: PaneCandidate | InvalidPane, requested: str) -> bool:
    return requested in {candidate.target, candidate.pane_id}


class TmuxGpuBridge:
    def __init__(
        self,
        *,
        runner: SubprocessRunner | None = None,
        socket_exists: Callable[[Path], bool] = _default_socket_exists,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runner = runner or SubprocessRunner()
        self.socket_exists = socket_exists
        self.sleep = sleep
        self.clock = clock

    def discover_socket(self, socket: Path, *, allow_in_mode: bool = False) -> DiscoveryResult:
        sockets = (str(socket),)
        if not self.socket_exists(socket):
            return DiscoveryResult((), (InvalidPane(str(socket), "", "", "socket_missing_or_not_socket"),), sockets)
        panes = self._tmux_list_panes(socket)
        ps_table = self._ps_table()
        valid: list[PaneCandidate] = []
        invalid: list[InvalidPane] = []
        for row in panes:
            candidate, problem = self._validate_row(socket, row, ps_table, allow_in_mode=allow_in_mode)
            if candidate is not None:
                valid.append(candidate)
            else:
                invalid.append(problem)
        return DiscoveryResult(tuple(valid), tuple(invalid), sockets)

    def discover_all(self, *, allow_in_mode: bool = False) -> DiscoveryResult:
        sockets = tuple(str(path) for path in candidate_socket_paths())
        all_valid: list[PaneCandidate] = []
        all_invalid: list[InvalidPane] = []
        for socket_path in candidate_socket_paths():
            result = self.discover_socket(socket_path, allow_in_mode=allow_in_mode)
            all_valid.extend(result.candidates)
            all_invalid.extend(result.invalid)
        return DiscoveryResult(tuple(all_valid), tuple(all_invalid), sockets)

    def resolve_target(self, options: BridgeOptions) -> tuple[PaneCandidate, DiscoveryResult, str]:
        cli_socket = options.socket or os.environ.get("ODCR_GPU_TMUX_SOCKET")
        cli_target = options.target or os.environ.get("ODCR_GPU_TMUX_TARGET")
        if cli_target and not cli_socket:
            raise BridgeError(
                "explicit tmux target requires --socket or ODCR_GPU_TMUX_SOCKET",
                stop_reason="target_invalid",
            )
        if cli_socket:
            result = self.discover_socket(Path(cli_socket))
            candidates = list(result.candidates)
            if cli_target:
                candidates = [candidate for candidate in candidates if _target_matches(candidate, cli_target)]
            return self._pick_unique(candidates, result, source="explicit")

        result = self.discover_all()
        return self._pick_unique(list(result.candidates), result, source="discovery")

    def resolve_unlock_target(self, options: BridgeOptions) -> tuple[PaneCandidate, DiscoveryResult, str]:
        cli_socket = options.socket or os.environ.get("ODCR_GPU_TMUX_SOCKET")
        cli_target = options.target or os.environ.get("ODCR_GPU_TMUX_TARGET")
        if cli_target and not cli_socket:
            raise BridgeError(
                "explicit tmux target requires --socket or ODCR_GPU_TMUX_SOCKET",
                stop_reason="target_invalid",
            )
        if cli_socket:
            result = self.discover_socket(Path(cli_socket), allow_in_mode=True)
            candidates = list(result.candidates)
            if cli_target:
                candidates = [candidate for candidate in candidates if _target_matches(candidate, cli_target)]
            target, discovery, source = self._pick_unique(candidates, result, source="explicit")
            return self._require_pane_mode(target, discovery, source)

        result = self.discover_all(allow_in_mode=True)
        target, discovery, source = self._pick_unique(list(result.candidates), result, source="discovery")
        return self._require_pane_mode(target, discovery, source)

    def run_mode(self, options: BridgeOptions) -> dict[str, Any]:
        if options.mode == "collect":
            if not options.run_id:
                raise BridgeError("collect requires --run-id", stop_reason="target_invalid")
            return collect_run(options.run_id)

        timeouts = resolve_timeouts(options.mode, options.timeout)
        kind = self._kind(options)
        run_id = options.run_id or _now_run_id(kind)
        paths = make_generated_paths(run_id)
        _ensure_analysis_dirs()

        if options.mode in {"repo-command", "repo-script", "repo-module", "command-file", "long-run"} and (
            options.dry_run or options.no_send
        ):
            try:
                return self._run_runtime_policy_only(options, run_id, paths, timeouts)
            except BridgeError as exc:
                return self._record_failure(paths, options.mode, timeouts, exc)

        if options.mode == "unlock-pane":
            try:
                return self._run_unlock_pane(options, run_id, paths, timeouts)
            except BridgeError as exc:
                return self._record_failure(paths, options.mode, timeouts, exc)

        try:
            target, discovery, target_source = self.resolve_target(options)
        except BridgeError as exc:
            return self._record_failure(paths, options.mode, timeouts, exc)
        if options.mode == "discover":
            status = self._status(
                run_id=run_id,
                kind=kind,
                success=True,
                exit_code=0,
                elapsed_s=0.0,
                timeouts=timeouts,
                first_result_seen=True,
                stop_reason="target_validated",
                metrics={"target_source": target_source, "candidate_count": len(discovery.candidates)},
            )
            self._write_local_outputs(paths, status, target, discovery, mode=options.mode)
            return {**status, "target": target.to_dict(), "discovery": discovery.to_dict(), "paths": paths.to_dict()}

        spec = OPERATION_SPECS[options.mode]
        if not spec.send_allowed:
            raise BridgeError(f"mode {options.mode} cannot send commands", stop_reason="forbidden_mode")
        script = build_script(options, run_id, paths, timeouts, target)
        write_script(paths.script, script, allow_formal=bool(options.user_confirmed_formal and options.mode == "long-run"))
        if options.dry_run or options.no_send:
            dry_run_success = options.mode != "step3-performance-probe"
            dry_run_exit = 0 if dry_run_success else 1
            runtime_command = ()
            runtime_output_dir = ""
            if options.mode in {"repo-command", "repo-script", "repo-module", "command-file", "long-run"}:
                runtime_output_dir = str(
                    resolve_runtime_output_dir(
                        options.output_dir,
                        run_id,
                        user_confirmed_formal=options.user_confirmed_formal,
                    )
                )
                runtime_command = runtime_command_argv(options, run_id)
            status = self._status(
                run_id=run_id,
                kind=kind,
                success=dry_run_success,
                exit_code=dry_run_exit,
                elapsed_s=0.0,
                timeouts=timeouts,
                first_result_seen=False,
                stop_reason=(
                    "step3_performance_probe_not_dispatched"
                    if options.mode == "step3-performance-probe"
                    else ("dry_run" if options.dry_run else "no_send")
                ),
                metrics={
                    "target_source": target_source,
                    "send_command": send_line(paths, timeouts),
                    "send_method": options.send_method,
                    "bridge_transport_ok": False,
                    "gpu_transport_ok": False,
                    "pane_validated": True,
                    "command_dispatched": False,
                    "child_process_ok": False,
                    "child_exit_code": None,
                    "runtime_started": False,
                    "runtime_probe_ok": False,
                    "runtime_evidence_ok": None,
                    "runtime_verified": False,
                    "evidence_complete": False,
                    "formal_pollution": False,
                    "formal_namespace_polluted": False,
                    "final_success": False,
                    "runtime_command_argv": list(runtime_command),
                    "runtime_output_dir": runtime_output_dir,
                },
            )
            if options.mode == "step3-performance-probe":
                status = normalize_bridge_runtime_success(status, bridge_transport_ok=False)
            elif options.mode in {"repo-command", "repo-script", "repo-module", "command-file", "long-run"}:
                status = normalize_repo_runtime_success(status, gpu_transport_ok=False)
            self._write_local_outputs(paths, status, target, discovery, mode=options.mode)
            return {
                **status,
                "target": target.to_dict(),
                "paths": paths.to_dict(),
                "sent": False,
                "send_command": send_line(paths, timeouts),
                "send_method": options.send_method,
            }

        command = send_line(paths, timeouts)
        self._send_keys(target, command, options.send_method)
        capture_after_send = self._capture_diagnostics(target, paths.run_id, command, "after_send")
        status = self._poll_for_result(paths, options.mode, timeouts)
        if options.mode == "step3-performance-probe":
            status = normalize_bridge_runtime_success(status, bridge_transport_ok=True)
        elif options.mode in {"repo-command", "repo-script", "repo-module", "command-file", "long-run"}:
            status = normalize_repo_runtime_success(status, gpu_transport_ok=True)
        capture_after_poll = self._capture_diagnostics(target, paths.run_id, command, "after_poll")
        status = self._augment_status(
            paths,
            status,
            {
                "target_source": target_source,
                "send_command": command,
                "send_method": options.send_method,
                "capture_after_send": capture_after_send,
                "capture_after_poll": capture_after_poll,
            },
        )
        final_status = {
            **status,
            "target": target.to_dict(),
            "paths": paths.to_dict(),
            "sent": True,
            "send_method": options.send_method,
        }
        _write_json(paths.status, final_status)
        self._write_summary_report(paths, final_status, target, DiscoveryResult((target,), (), (target.socket,)), options.mode)
        return final_status

    def _policy_only_target(self) -> PaneCandidate:
        return PaneCandidate(
            socket="policy-only",
            session="policy-only",
            target="policy-only",
            pane_id="%policy",
            pane_pid=0,
            pane_command="policy-only",
            cwd=str(REPO_ROOT),
            active=False,
            dead=False,
            in_mode=False,
            srun_pid=None,
            srun_command="",
            job_id="UNSENT",
            step_id="UNSENT",
            node="UNSENT",
            gpu="UNSENT",
            step_state="UNSENT",
            job_state="UNSENT",
        )

    def _run_runtime_policy_only(
        self,
        options: BridgeOptions,
        run_id: str,
        paths: GeneratedPaths,
        timeouts: ResolvedTimeouts,
    ) -> dict[str, Any]:
        output_dir = resolve_runtime_output_dir(
            options.output_dir,
            run_id,
            user_confirmed_formal=options.user_confirmed_formal,
        )
        command = runtime_command_argv(options, run_id)
        classification = BridgeCommandPolicy.classify_repo_command(
            command,
            output_dir=output_dir,
            user_confirmed_formal=options.user_confirmed_formal,
        )
        if not classification.allowed:
            raise BridgeError(classification.reason, stop_reason=classification.stop_reason, details=classification.to_dict())
        script = build_script(options, run_id, paths, timeouts, self._policy_only_target())
        write_script(paths.script, script, allow_formal=bool(options.user_confirmed_formal and options.mode == "long-run"))
        status = self._status(
            run_id=run_id,
            kind=options.mode,
            success=True,
            exit_code=0,
            elapsed_s=0.0,
            timeouts=timeouts,
            first_result_seen=False,
            stop_reason="dry_run" if options.dry_run else "no_send",
            metrics={
                "policy_validation_only": True,
                "target_source": "policy-only",
                "send_command": send_line(paths, timeouts),
                "send_method": options.send_method,
                "bridge_transport_ok": None,
                "gpu_transport_ok": None,
                "pane_validated": False,
                "command_dispatched": False,
                "child_process_ok": None,
                "child_exit_code": None,
                "runtime_started": False,
                "runtime_probe_ok": False,
                "runtime_evidence_ok": None,
                "runtime_verified": False,
                "evidence_complete": False,
                "formal_pollution": False,
                "formal_namespace_polluted": False,
                "formal_pollution_check_passed": True,
                "final_success": False,
                "command_allowed_by_policy": True,
                "command_policy": classification.to_dict(),
                "runtime_command_argv": list(command),
                "runtime_output_dir": str(output_dir),
            },
        )
        target = self._policy_only_target()
        discovery = DiscoveryResult((target,), (), ("policy-only",))
        self._write_local_outputs(paths, status, target, discovery, mode=options.mode)
        return {
            **status,
            "target": target.to_dict(),
            "paths": paths.to_dict(),
            "sent": False,
            "send_command": send_line(paths, timeouts),
            "send_method": options.send_method,
        }

    def _pick_unique(
        self,
        candidates: Sequence[PaneCandidate],
        discovery: DiscoveryResult,
        *,
        source: str,
    ) -> tuple[PaneCandidate, DiscoveryResult, str]:
        if len(candidates) == 1:
            return candidates[0], discovery, source
        if len(candidates) == 0:
            raise BridgeError(
                "no validated GPU tmux pane was found",
                stop_reason="target_not_unique",
                details=discovery.to_dict(),
            )
        raise BridgeError(
            "more than one validated GPU tmux pane was found",
            stop_reason="target_not_unique",
            details=discovery.to_dict(),
        )

    def _require_pane_mode(
        self,
        target: PaneCandidate,
        discovery: DiscoveryResult,
        source: str,
    ) -> tuple[PaneCandidate, DiscoveryResult, str]:
        if target.in_mode:
            return target, discovery, source
        raise BridgeError(
            "unlock-pane requires the validated target pane to be in tmux pane mode",
            stop_reason="target_invalid",
            details={"target": target.to_dict(), "discovery": discovery.to_dict(), "target_source": source},
        )

    def _tmux_list_panes(self, socket: Path) -> list[tuple[str, str, str, str, int, str, str, bool, bool, bool]]:
        result = self.runner.run(
            ["tmux", "-S", str(socket), "list-panes", "-a", "-F", TMUX_PANE_FORMAT],
            timeout=5,
        )
        if result.returncode != 0:
            return []
        rows: list[tuple[str, str, str, str, int, str, str, bool, bool, bool]] = []
        for line in result.stdout.splitlines():
            row = _split_tmux_row(line)
            if row is not None:
                rows.append(row)
        return rows

    def _ps_table(self) -> dict[int, ProcInfo]:
        result = self.runner.run(["ps", "-e", "-o", "pid=", "-o", "ppid=", "-o", "args="], timeout=5)
        if result.returncode != 0:
            return {}
        return _parse_ps(result.stdout)

    def _validate_row(
        self,
        socket: Path,
        row: tuple[str, str, str, str, int, str, str, bool, bool, bool],
        ps_table: Mapping[int, ProcInfo],
        *,
        allow_in_mode: bool = False,
    ) -> tuple[PaneCandidate | None, InvalidPane]:
        session, window_index, pane_index, pane_id, pane_pid, command, cwd, active, dead, in_mode = row
        target = f"{session}:{window_index}.{pane_index}"
        base_invalid = InvalidPane(str(socket), target, pane_id, "invalid", command, cwd)
        if session != "odcr":
            return None, dataclasses.replace(base_invalid, reason="session_not_odcr")
        if cwd != str(REPO_ROOT):
            return None, dataclasses.replace(base_invalid, reason="cwd_not_repo")
        if dead:
            return None, dataclasses.replace(base_invalid, reason="pane_dead")
        if in_mode and not allow_in_mode:
            return None, dataclasses.replace(base_invalid, reason="pane_in_mode")

        srun_pid, srun_command = self._find_srun(pane_pid, command, ps_table)
        if not srun_command:
            return None, dataclasses.replace(base_invalid, reason="no_child_srun")
        job_id = _parse_job_id(srun_command)
        if not job_id:
            inferred = self._infer_direct_srun_job(srun_command, cwd)
            if not inferred:
                return None, dataclasses.replace(base_invalid, reason="srun_jobid_missing")
            job_id, job_values = inferred
            node = job_values.get("NodeList") or job_values.get("BatchHost") or ""
            gpu = _gpu_text({}, job_values)
            job_state = job_values.get("JobState", "")
            if job_state != "RUNNING":
                return None, dataclasses.replace(base_invalid, reason="job_not_running")
            if not _is_gpu_node_or_tres(node, gpu):
                return None, dataclasses.replace(base_invalid, reason="not_gpu_node_or_tres")
            if not _has_gpu_tres(gpu):
                return None, dataclasses.replace(base_invalid, reason="gpu_tres_missing")
            return (
                PaneCandidate(
                    socket=str(socket),
                    session=session,
                    target=target,
                    pane_id=pane_id,
                    pane_pid=pane_pid,
                    pane_command=command,
                    cwd=cwd,
                    active=active,
                    dead=dead,
                    in_mode=in_mode,
                    srun_pid=srun_pid,
                    srun_command=srun_command,
                    job_id=job_id,
                    step_id=f"{job_id}.interactive",
                    node=node,
                    gpu=gpu,
                    step_state=job_state,
                    job_state=job_state,
                ),
                base_invalid,
            )

        step_id = f"{job_id}.0"
        step_result = self.runner.run(["scontrol", "show", "step", step_id], timeout=8)
        job_result = self.runner.run(["scontrol", "show", "job", job_id], timeout=8)
        if job_result.returncode != 0:
            return None, dataclasses.replace(base_invalid, reason="scontrol_job_failed")

        job_values = _parse_key_values(job_result.stdout)
        if step_result.returncode != 0:
            node = job_values.get("NodeList") or job_values.get("BatchHost") or ""
            gpu = _gpu_text({}, job_values)
            job_state = job_values.get("JobState", "")
            if job_state != "RUNNING":
                return None, dataclasses.replace(base_invalid, reason="job_not_running")
            if not _is_gpu_node_or_tres(node, gpu):
                return None, dataclasses.replace(base_invalid, reason="not_gpu_node_or_tres")
            if not _has_gpu_tres(gpu):
                return None, dataclasses.replace(base_invalid, reason="gpu_tres_missing")
            return (
                PaneCandidate(
                    socket=str(socket),
                    session=session,
                    target=target,
                    pane_id=pane_id,
                    pane_pid=pane_pid,
                    pane_command=command,
                    cwd=cwd,
                    active=active,
                    dead=dead,
                    in_mode=in_mode,
                    srun_pid=srun_pid,
                    srun_command=srun_command,
                    job_id=job_id,
                    step_id=f"{job_id}.interactive",
                    node=node,
                    gpu=gpu,
                    step_state=job_state,
                    job_state=job_state,
                ),
                base_invalid,
            )

        step_values = _parse_key_values(step_result.stdout)
        step_state = step_values.get("State", "")
        job_state = job_values.get("JobState", "")
        node = step_values.get("NodeList") or job_values.get("NodeList") or ""
        gpu = _gpu_text(step_values, job_values)
        if step_state != "RUNNING":
            return None, dataclasses.replace(base_invalid, reason="step_not_running")
        if not _is_gpu_node_or_tres(node, gpu):
            return None, dataclasses.replace(base_invalid, reason="not_gpu_node_or_tres")
        if not _has_gpu_tres(gpu):
            return None, dataclasses.replace(base_invalid, reason="gpu_tres_missing")

        return (
            PaneCandidate(
                socket=str(socket),
                session=session,
                target=target,
                pane_id=pane_id,
                pane_pid=pane_pid,
                pane_command=command,
                cwd=cwd,
                active=active,
                dead=dead,
                in_mode=in_mode,
                srun_pid=srun_pid,
                srun_command=srun_command,
                job_id=job_id,
                step_id=step_id,
                node=node,
                gpu=gpu,
                step_state=step_state,
                job_state=job_state,
            ),
            base_invalid,
        )

    def _run_unlock_pane(
        self,
        options: BridgeOptions,
        run_id: str,
        paths: GeneratedPaths,
        timeouts: ResolvedTimeouts,
    ) -> dict[str, Any]:
        target, discovery, target_source = self.resolve_unlock_target(options)
        script = build_unlock_pane_evidence_script(run_id, paths, timeouts, target)
        write_script(paths.script, script)

        attempts: list[dict[str, Any]] = []
        final_target = target
        success = False
        for index, operation in enumerate(UNLOCK_OPERATIONS, start=1):
            result = self._unlock_pane_once(target, operation)
            relaxed_target, relaxed_discovery = self._find_same_target(target, allow_in_mode=True)
            strict_target, _strict_discovery = self._find_same_target(target, allow_in_mode=False)
            final_target = relaxed_target or strict_target or final_target
            attempt = {
                "attempt": index,
                "operation": operation,
                "args": list(result.args),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "post_relaxed_candidate_found": relaxed_target is not None,
                "post_strict_validate_success": strict_target is not None,
                "post_pane_in_mode": relaxed_target.in_mode if relaxed_target is not None else None,
                "post_discovery": relaxed_discovery.to_dict(),
            }
            attempts.append(attempt)
            if result.returncode != 0:
                continue
            if strict_target is not None:
                final_target = strict_target
                success = True
                break

        status = self._status(
            run_id=run_id,
            kind="unlock-pane",
            success=success,
            exit_code=0 if success else 1,
            elapsed_s=0.0,
            timeouts=timeouts,
            first_result_seen=True,
            stop_reason=OPERATION_SPECS["unlock-pane"].stop_reason_success if success else "pane_in_mode_unlock_failed",
            metrics={
                "target_source": target_source,
                "initial_pane_in_mode": target.in_mode,
                "final_pane_in_mode": final_target.in_mode,
                "attempt_count": len(attempts),
                "attempts": attempts,
            },
        )
        self._write_unlock_outputs(paths, status, target, final_target, discovery, attempts)
        return {
            **status,
            "target": final_target.to_dict(),
            "paths": paths.to_dict(),
            "sent": False,
            "unlock_operations": [attempt["operation"] for attempt in attempts],
        }

    def _unlock_pane_once(self, target: PaneCandidate, operation: str) -> CommandResult:
        if operation == "copy-mode-cancel":
            args = ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "-X", "cancel"]
        elif operation == "escape":
            args = ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "Escape"]
        elif operation == "q":
            args = ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "q"]
        else:
            raise BridgeError(f"unsupported unlock operation: {operation}", stop_reason="forbidden_mode")
        return self.runner.run(args, timeout=5)

    def _find_same_target(
        self,
        target: PaneCandidate,
        *,
        allow_in_mode: bool,
    ) -> tuple[PaneCandidate | None, DiscoveryResult]:
        discovery = self.discover_socket(Path(target.socket), allow_in_mode=allow_in_mode)
        for candidate in discovery.candidates:
            if (
                candidate.pane_id == target.pane_id
                and candidate.target == target.target
                and candidate.job_id == target.job_id
            ):
                return candidate, discovery
        return None, discovery

    def _find_srun(
        self,
        pane_pid: int,
        pane_command: str,
        ps_table: Mapping[int, ProcInfo],
    ) -> tuple[int | None, str]:
        if _is_srun_command(pane_command):
            proc = ps_table.get(pane_pid)
            if proc and _is_srun_command(proc.args):
                return proc.pid, proc.args
        for proc in _iter_descendants(pane_pid, ps_table):
            if _is_srun_command(proc.args):
                return proc.pid, proc.args
        return None, ""

    def _infer_direct_srun_job(self, srun_command: str, cwd: str) -> tuple[str, dict[str, str]] | None:
        """Infer a live Slurm job for direct `srun --pty` panes without --jobid.

        Some ODCR GPU panes are entered with a direct interactive srun rather
        than `srun --jobid=<id>`. The pane is still user-created and valid, but
        the bridge must fresh-validate the current Slurm job from scheduler
        state instead of trusting stale pane files.
        """
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        squeue_cmd = ["squeue", "-h", "-t", "R", "-o", "%i"]
        if user:
            squeue_cmd[1:1] = ["-u", user]
        squeue = self.runner.run(squeue_cmd, timeout=8)
        if squeue.returncode != 0:
            return None
        requested_gpus = _requested_gpu_count_from_srun(srun_command)
        matches: list[tuple[str, dict[str, str]]] = []
        for raw_job_id in squeue.stdout.splitlines():
            job_id = raw_job_id.strip().split()[0] if raw_job_id.strip() else ""
            if not job_id or not job_id.isdigit():
                continue
            job_result = self.runner.run(["scontrol", "show", "job", job_id], timeout=8)
            if job_result.returncode != 0:
                continue
            job_values = _parse_key_values(job_result.stdout)
            if job_values.get("JobState") != "RUNNING":
                continue
            if job_values.get("WorkDir") != cwd:
                continue
            gpu = _gpu_text({}, job_values)
            if not _has_gpu_tres(gpu):
                continue
            available_gpus = _gpu_count_from_tres(gpu)
            if requested_gpus is not None and available_gpus is not None and available_gpus < requested_gpus:
                continue
            matches.append((job_id, job_values))
        if len(matches) == 1:
            return matches[0]
        return None

    def _send_keys(self, target: PaneCandidate, command: str, send_method: str = DEFAULT_SEND_METHOD) -> None:
        if send_method not in SEND_METHODS:
            raise BridgeError(f"unsupported send method: {send_method}", stop_reason="target_invalid")
        if "\n" in command or "\r" in command:
            raise BridgeError("generated send command must be a single line", stop_reason="forbidden_mode")

        if send_method == "target-name-literal-enter":
            commands = (
                ("literal", ["tmux", "-S", target.socket, "send-keys", "-t", target.target, "-l", command]),
                ("enter", ["tmux", "-S", target.socket, "send-keys", "-t", target.target, "Enter"]),
            )
        elif send_method == "pane-id-literal-cm":
            commands = (
                ("literal", ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "-l", command]),
                ("enter", ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "C-m"]),
            )
        elif send_method == "buffer-paste-enter":
            commands = (
                ("set_buffer", ["tmux", "-S", target.socket, "set-buffer", command]),
                ("paste_buffer", ["tmux", "-S", target.socket, "paste-buffer", "-t", target.pane_id]),
                ("enter", ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "Enter"]),
            )
        else:
            commands = (
                ("literal", ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "-l", command]),
                ("enter", ["tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "Enter"]),
            )

        results = [(label, self.runner.run(args, timeout=5)) for label, args in commands]
        failed = [(label, result) for label, result in results if result.returncode != 0]
        if failed:
            raise BridgeError(
                "tmux send-keys failed",
                stop_reason="target_invalid",
                details={
                    "send_method": send_method,
                    "failed_steps": [
                        {
                            "step": label,
                            "args": list(result.args),
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "returncode": result.returncode,
                        }
                        for label, result in failed
                    ],
                },
            )

    def _capture_diagnostics(self, target: PaneCandidate, run_id: str, command: str, phase: str) -> dict[str, Any]:
        result = self.runner.run(
            ["tmux", "-S", target.socket, "capture-pane", "-t", target.pane_id, "-p", "-S", "-120"],
            timeout=5,
        )
        text = result.stdout if result.returncode == 0 else ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        recent_lines = lines[-12:]
        prompt_seen = any(line.endswith("$") and "bash" in line for line in recent_lines)
        return {
            "phase": phase,
            "capture_target": target.pane_id,
            "returncode": result.returncode,
            "stderr": result.stderr,
            "run_id_seen": run_id in text,
            "command_seen": command in text or Path(command.split()[-1]).name in text,
            "begin_marker_seen": f"ODCR_BRIDGE_BEGIN_{run_id}" in text,
            "end_marker_seen": f"ODCR_BRIDGE_END_{run_id}" in text,
            "bash_prompt_seen": prompt_seen,
            "last_nonempty_line": recent_lines[-1] if recent_lines else "",
        }

    def _augment_status(self, paths: GeneratedPaths, status: Mapping[str, Any], extra_metrics: Mapping[str, Any]) -> dict[str, Any]:
        augmented = dict(status)
        metrics = dict(augmented.get("metrics") or {})
        metrics.update(extra_metrics)
        augmented["metrics"] = metrics
        _write_json(paths.status, augmented)
        return augmented

    def _record_failure(
        self,
        paths: GeneratedPaths,
        mode: str,
        timeouts: ResolvedTimeouts,
        exc: BridgeError,
    ) -> dict[str, Any]:
        status = self._status(
            run_id=paths.run_id,
            kind=mode,
            success=False,
            exit_code=1,
            elapsed_s=0.0,
            timeouts=timeouts,
            first_result_seen=False,
            stop_reason=exc.stop_reason,
            metrics={"error": str(exc), "details": exc.details},
        )
        write_script(paths.script, build_failure_evidence_script(paths.run_id, paths, mode, str(exc)))
        self._write_failure_outputs(paths, status, mode)
        return {**status, "paths": paths.to_dict(), "sent": False}

    def _poll_for_result(self, paths: GeneratedPaths, mode: str, timeouts: ResolvedTimeouts) -> dict[str, Any]:
        start = self.clock()
        begin_seen = False
        first_result_timeout = False
        first_result_timeout_elapsed_s: float | None = None
        result_status_path = _mode_status_result_path(paths, mode)
        while True:
            elapsed = self.clock() - start
            log_text = paths.log.read_text(encoding="utf-8") if paths.log.is_file() else ""
            begin_seen = begin_seen or f"ODCR_BRIDGE_BEGIN_{paths.run_id}" in log_text
            if result_status_path.is_file():
                try:
                    status = _read_json(result_status_path)
                except json.JSONDecodeError:
                    status = None
                if isinstance(status, dict):
                    if first_result_timeout:
                        status = self._annotate_first_result_timeout_recovery(
                            status,
                            elapsed_s=elapsed,
                            timeout_elapsed_s=first_result_timeout_elapsed_s,
                        )
                    return status
            if not begin_seen and elapsed > timeouts.startup_timeout_s:
                return self._write_timeout_status(paths, mode, timeouts, "startup_timeout", elapsed)
            if begin_seen and elapsed > timeouts.first_result_timeout_s:
                if mode in {"repo-command", "repo-script", "repo-module", "command-file"}:
                    if not first_result_timeout:
                        first_result_timeout = True
                        first_result_timeout_elapsed_s = elapsed
                    if timeouts.hard_timeout_s is None:
                        return self._write_timeout_status(
                            paths,
                            mode,
                            timeouts,
                            "first_result_timeout",
                            elapsed,
                            metrics={
                                "first_result_timeout": True,
                                "first_result_timeout_recovered": False,
                                "child_process_started": True,
                                "child_process_still_running": True,
                                "final_artifact_completed": False,
                                "evidence_complete": False,
                                "formal_pollution": False,
                            },
                        )
                else:
                    return self._write_timeout_status(paths, mode, timeouts, "first_result_timeout", elapsed)
            if timeouts.hard_timeout_s is not None and elapsed > timeouts.hard_timeout_s:
                metrics = {}
                if first_result_timeout:
                    metrics = {
                        "first_result_timeout": True,
                        "first_result_timeout_recovered": False,
                        "first_result_timeout_elapsed_s": round(float(first_result_timeout_elapsed_s or elapsed), 3),
                        "child_process_started": bool(begin_seen),
                        "child_process_still_running": bool(begin_seen and f"ODCR_BRIDGE_END_{paths.run_id}" not in log_text),
                        "final_artifact_completed": False,
                        "evidence_complete": False,
                        "formal_pollution": False,
                    }
                return self._write_timeout_status(paths, mode, timeouts, "hard_timeout", elapsed, metrics=metrics)
            self.sleep(0.5)

    def _write_timeout_status(
        self,
        paths: GeneratedPaths,
        mode: str,
        timeouts: ResolvedTimeouts,
        stop_reason: str,
        elapsed: float,
        *,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not paths.log.is_file():
            _atomic_write_text(
                paths.log,
                "\n".join(
                    (
                        f"run_id={paths.run_id}",
                        f"mode={mode}",
                        f"stop_reason={stop_reason}",
                        "bridge_script_begin_marker_seen=false",
                        "bridge_script_end_marker_seen=false",
                        "",
                    )
                ),
            )
        status = self._status(
            run_id=paths.run_id,
            kind=mode,
            success=False,
            exit_code=124,
            elapsed_s=round(elapsed, 3),
            timeouts=timeouts,
            first_result_seen=False,
            stop_reason=stop_reason,
            metrics=dict(metrics or {}),
        )
        _write_json(paths.status, status)
        return status

    def _annotate_first_result_timeout_recovery(
        self,
        status: Mapping[str, Any],
        *,
        elapsed_s: float,
        timeout_elapsed_s: float | None,
    ) -> dict[str, Any]:
        out = dict(status)
        metrics = dict(out.get("metrics") or {})
        child_returncode_raw = metrics.get("child_returncode", metrics.get("child_exit_code", out.get("exit_code")))
        try:
            child_returncode = int(child_returncode_raw)
        except (TypeError, ValueError):
            child_returncode = None
        evidence_complete = bool(metrics.get("evidence_complete", False))
        final_artifact_completed = bool(metrics.get("final_artifact_completed", evidence_complete))
        formal_pollution = bool(metrics.get("formal_pollution", metrics.get("formal_namespace_polluted", False)))
        recovered = bool(
            child_returncode == 0
            and final_artifact_completed
            and evidence_complete
            and not formal_pollution
        )
        metrics.update(
            {
                "first_result_timeout": True,
                "first_result_timeout_recovered": recovered,
                "first_result_timeout_elapsed_s": round(float(timeout_elapsed_s or elapsed_s), 3),
                "child_process_started": True,
                "child_process_still_running": False,
                "child_returncode": child_returncode,
                "final_artifact_completed": final_artifact_completed,
                "evidence_complete": evidence_complete,
                "formal_pollution": formal_pollution,
            }
        )
        out["metrics"] = metrics
        return out

    def _status(
        self,
        *,
        run_id: str,
        kind: str,
        success: bool,
        exit_code: int,
        elapsed_s: float,
        timeouts: ResolvedTimeouts,
        first_result_seen: bool,
        stop_reason: str,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        spec = OPERATION_SPECS[kind if kind in OPERATION_SPECS else kind.split("_")[0]]
        return {
            "schema_version": BRIDGE_STATUS_SCHEMA,
            "run_id": run_id,
            "kind": kind,
            "success": bool(success),
            "exit_code": int(exit_code),
            "elapsed_s": float(elapsed_s),
            "startup_timeout_s": timeouts.startup_timeout_s,
            "first_result_timeout_s": timeouts.first_result_timeout_s,
            "hard_timeout_s": timeouts.hard_timeout_s,
            "first_result_seen": bool(first_result_seen),
            "success_condition": spec.success_condition,
            "stop_reason": stop_reason,
            "metrics": dict(metrics),
        }

    def _write_local_outputs(
        self,
        paths: GeneratedPaths,
        status: Mapping[str, Any],
        target: PaneCandidate,
        discovery: DiscoveryResult,
        *,
        mode: str,
    ) -> None:
        log_lines = [
            f"ODCR_BRIDGE_BEGIN_{paths.run_id}",
            f"mode={mode}",
            f"target={target.socket}:{target.target}",
            f"job={target.job_id}",
            f"step={target.step_id}",
            f"node={target.node}",
            f"gpu={target.gpu}",
            f"stop_reason={status.get('stop_reason')}",
            f"ODCR_BRIDGE_END_{paths.run_id}",
            "",
        ]
        _atomic_write_text(paths.log, "\n".join(log_lines))
        _write_json(paths.status, status)
        self._write_summary_report(paths, status, target, discovery, mode)

    def _write_summary_report(
        self,
        paths: GeneratedPaths,
        status: Mapping[str, Any],
        target: PaneCandidate,
        discovery: DiscoveryResult,
        mode: str,
    ) -> None:
        summary = textwrap.dedent(
            f"""\
            # Tmux Bridge {paths.run_id} Summary

            - mode: {mode}
            - success: {status.get("success")}
            - stop_reason: {status.get("stop_reason")}
            - target: {target.socket}:{target.target}
            - pane_id: {target.pane_id}
            - job_id: {target.job_id}
            - step_id: {target.step_id}
            - node: {target.node}
            - gpu: {target.gpu}
            - first_result_seen: {status.get("first_result_seen")}
            - hard_timeout_s: {status.get("hard_timeout_s")}
            """
        )
        report = textwrap.dedent(
            f"""\
            # Tmux Bridge {paths.run_id} Report

            ## Result

            Status: {status.get("success")}
            Stop reason: {status.get("stop_reason")}
            Mode: {mode}
            Success condition: {status.get("success_condition")}

            ## Target

            ```json
            {json.dumps(target.to_dict(), indent=2, sort_keys=True)}
            ```

            ## Status

            ```json
            {json.dumps(dict(status), indent=2, sort_keys=True)}
            ```

            ## Discovery

            Valid candidates: {len(discovery.candidates)}
            Invalid candidates: {len(discovery.invalid)}

            ## Boundaries

            - No formal preprocess_a/b/c was started.
            - No formal Step3/Step4/Step5 was entered.
            - No eval/rerank was run.
            - No GPU allocation command was executed by Codex.
            - No tmux session was created, killed, switched, or attached.
            """
        )
        _atomic_write_text(paths.summary, summary)
        _atomic_write_text(paths.report, report)

    def _write_unlock_outputs(
        self,
        paths: GeneratedPaths,
        status: Mapping[str, Any],
        initial_target: PaneCandidate,
        final_target: PaneCandidate,
        discovery: DiscoveryResult,
        attempts: Sequence[Mapping[str, Any]],
    ) -> None:
        log_lines = [
            f"ODCR_BRIDGE_BEGIN_{paths.run_id}",
            "mode=unlock-pane",
            f"target={initial_target.socket}:{initial_target.target}",
            f"pane_id={initial_target.pane_id}",
            f"initial_pane_in_mode={initial_target.in_mode}",
            f"final_pane_in_mode={final_target.in_mode}",
            f"stop_reason={status.get('stop_reason')}",
        ]
        for attempt in attempts:
            log_lines.append(
                "unlock_attempt="
                + json.dumps(
                    {
                        "attempt": attempt.get("attempt"),
                        "operation": attempt.get("operation"),
                        "returncode": attempt.get("returncode"),
                        "post_strict_validate_success": attempt.get("post_strict_validate_success"),
                        "post_pane_in_mode": attempt.get("post_pane_in_mode"),
                    },
                    sort_keys=True,
                )
            )
        log_lines.extend((f"ODCR_BRIDGE_END_{paths.run_id}", ""))
        _atomic_write_text(paths.log, "\n".join(log_lines))
        _write_json(paths.status, status)
        self._write_summary_report(paths, status, final_target, discovery, "unlock-pane")

    def _write_failure_outputs(self, paths: GeneratedPaths, status: Mapping[str, Any], mode: str) -> None:
        log_lines = [
            f"ODCR_BRIDGE_BEGIN_{paths.run_id}",
            f"mode={mode}",
            "sent=false",
            f"stop_reason={status.get('stop_reason')}",
            f"error={status.get('metrics', {}).get('error', '')}",
            f"ODCR_BRIDGE_END_{paths.run_id}",
            "",
        ]
        _atomic_write_text(paths.log, "\n".join(log_lines))
        _write_json(paths.status, status)
        summary = textwrap.dedent(
            f"""\
            # Tmux Bridge {paths.run_id} Summary

            - mode: {mode}
            - success: {status.get("success")}
            - stop_reason: {status.get("stop_reason")}
            - sent: false
            """
        )
        report = textwrap.dedent(
            f"""\
            # Tmux Bridge {paths.run_id} Report

            ## Result

            Status: {status.get("success")}
            Stop reason: {status.get("stop_reason")}
            Mode: {mode}

            ## Status

            ```json
            {json.dumps(dict(status), indent=2, sort_keys=True)}
            ```

            ## Boundaries

            - No target-pane shell command was sent.
            - No formal preprocess_a/b/c was started.
            - No formal Step3/Step4/Step5 was entered.
            - No eval/rerank was run.
            - No GPU allocation command was executed by Codex.
            - No tmux session was created, killed, switched, or attached.
            """
        )
        _atomic_write_text(paths.summary, summary)
        _atomic_write_text(paths.report, report)

    def _kind(self, options: BridgeOptions) -> str:
        if options.mode == "preprocess-dryrun":
            return options.mode
        if options.mode == "micro-benchmark":
            return options.mode
        return options.mode


def build_unlock_pane_evidence_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    target: PaneCandidate,
) -> str:
    del paths, timeouts
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        # Evidence stub for controlled unlock-pane run {run_id}.
        # The bridge does not run this file in the target pane.
        # Fixed operations are performed by odcr_tmux_gpu_bridge.py only:
        # tmux -S {target.socket} send-keys -t {target.pane_id} -X cancel
        # tmux -S {target.socket} send-keys -t {target.pane_id} Escape
        # tmux -S {target.socket} send-keys -t {target.pane_id} q
        exit 0
        """
    )


def build_failure_evidence_script(run_id: str, paths: GeneratedPaths, mode: str, error: str) -> str:
    del paths, error
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        # Evidence stub for bridge failure {run_id}.
        # Mode: {mode}
        # No target-pane shell command was sent.
        exit 1
        """
    )


def send_line(paths: GeneratedPaths, timeouts: ResolvedTimeouts) -> str:
    try:
        script_path = paths.script.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        script_path = str(paths.script)
    if timeouts.detached:
        return f"bash {script_path}"
    return f"timeout {timeouts.hard_timeout_s}s bash {script_path}"


def write_script(path: Path, script: str, *, allow_formal: bool = False) -> None:
    validate_script_safety(script, allow_formal=allow_formal)
    _atomic_write_text(path, script)
    path.chmod(0o700)


def validate_script_safety(script: str, *, allow_formal: bool = False) -> None:
    lowered = script.lower()
    forbidden_regexes = [
        (r"\bnohup\b", "nohup"),
        (r"&", "background_or_fd_ampersand"),
        (r"\bdisown\b", "disown"),
        (r"\bsbatch\b", "sbatch"),
        (r"\bsrun\b", "srun"),
        (r"\bscancel\b", "scancel"),
        (r"\bodcr-enter-gpu\b", "odcr-enter-gpu"),
        (r"\brm\s+-rf\b", "rm -rf"),
        (r"\bkill\b", "kill"),
        (r"\bpkill\b", "pkill"),
        (r"\bmv\s+data\b", "mv data"),
        (r"\brm\s+data\b", "rm data"),
        (r"\brm\s+merged\b", "rm merged"),
    ]
    for pattern, label in forbidden_regexes:
        if re.search(pattern, lowered):
            raise BridgeError(f"generated bridge script contains forbidden token: {label}", stop_reason="forbidden_mode")
    if re.search(r"\./odcr\s+preprocess\s+a\b", lowered):
        raise BridgeError("preprocess a is forbidden", stop_reason="forbidden_mode")
    for stage in ("b", "c"):
        for match in re.finditer(rf"\./odcr\s+preprocess\s+{stage}\b[^\n]*", lowered):
            if "--dry-run" not in match.group(0):
                raise BridgeError(f"preprocess {stage} without --dry-run is forbidden", stop_reason="forbidden_mode")
    always_forbidden = (
        (r"code/tools/odcr_step3_real_data_probe\.py", "retired Step3 real-data probe"),
        (r"\bstep3-ddp-smoke\b", "retired step3-ddp-smoke"),
        (r"\bstep3-short-pilot\b", "retired step3-short-pilot"),
        (r"\btrainmodel_ddp\b", "formal step3 train loop"),
        (r"(?:\./odcr|python\s+code/odcr\.py)\s+step5\b", "step5"),
        (r"(?:\./odcr|python\s+code/odcr\.py)\s+eval\b", "eval"),
        (r"(?:\./odcr|python\s+code/odcr\.py)\s+rerank\b", "rerank"),
    )
    for pattern, label in always_forbidden:
        if re.search(pattern, lowered):
            raise BridgeError(f"generated bridge script contains forbidden token: {label}", stop_reason="forbidden_mode")
    if not allow_formal:
        formal_step3_patterns = (
            (r"(?:\./odcr|python\s+code/odcr\.py)\s+step3\b(?![^\n]*--dry-run)", "formal step3"),
            (r"code/executors/step3_entry\.py", "formal step3 entry"),
            (r"\btorchrun\b[^\n]*\bstep3\b", "torchrun step3"),
            (r"\bbest\.pth\b", "formal checkpoint target"),
            (r"\bcheckpoint_lineage\.json\b", "formal checkpoint lineage"),
        )
        for pattern, label in formal_step3_patterns:
            if re.search(pattern, lowered):
                raise BridgeError(f"generated bridge script contains forbidden token: {label}", stop_reason="forbidden_mode")
        if re.search(r"\b(?:data|merged)/", lowered):
            raise BridgeError("formal data/merged writes are forbidden in bridge scripts", stop_reason="forbidden_mode")
    for line in _iter_script_command_lines(script):
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            raise BridgeError(f"generated bridge script contains unparsable command: {exc}", stop_reason="forbidden_mode") from exc
        classification = BridgeCommandPolicy.classify_repo_command(
            parts,
            output_dir=(AI_ANALYSIS / "06_probe_evidence").resolve(),
            user_confirmed_formal=allow_formal,
        )
        if not classification.allowed:
            raise BridgeError(classification.reason, stop_reason=classification.stop_reason, details=classification.to_dict())


def _iter_script_command_lines(script: str) -> Iterable[str]:
    for raw in script.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("./odcr") or line.startswith("python code/odcr.py") or line.startswith("python -m code.odcr"):
            yield line


def _validation_without_formal_patterns() -> tuple[tuple[str, str], ...]:
    return (
        (r"runs/step3/task\d+/latest\.json", "formal Step3 latest"),
        (r"runs/step3/task\d+/[^ \n;]+/model/(?:best|best_observed|latest)\.pth", "formal Step3 checkpoint"),
        (r"runs/step3/task\d+/[^ \n;]+/state/checkpoint_lineage\.json", "formal Step3 checkpoint lineage"),
        (r"(?:\./odcr|python\s+code/odcr\.py)\s+step3\b(?![^\n]*--dry-run)", "formal Step3 launch"),
        (r"(?:\./odcr|python\s+code/odcr\.py)\s+step5\b", "Step5 launch from validation"),
        (r"(?:\./odcr|python\s+code/odcr\.py)\s+eval\b", "eval launch from validation"),
        (r"(?:\./odcr|python\s+code/odcr\.py)\s+rerank\b", "rerank launch from validation"),
        (r"\bbest\.pth\b", "formal best.pth target"),
        (r"\blatest\.json\b", "formal latest pointer"),
    )


def validate_runtime_command_safety(
    command_text: str,
    *,
    output_dir: Path,
    user_confirmed_formal: bool = False,
) -> None:
    validate_script_safety(command_text, allow_formal=user_confirmed_formal)
    if user_confirmed_formal:
        return
    lowered = command_text.lower()
    for line in _iter_script_command_lines(command_text):
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            raise BridgeError(f"validation GPU command is unparsable: {exc}", stop_reason="forbidden_mode") from exc
        classification = BridgeCommandPolicy.classify_repo_command(parts, output_dir=output_dir)
        if not classification.allowed:
            raise BridgeError(classification.reason, stop_reason=classification.stop_reason, details=classification.to_dict())
    for pattern, label in _validation_without_formal_patterns():
        if re.search(pattern, lowered):
            raise BridgeError(
                f"validation GPU command would touch a formal boundary without user_confirmed_formal=true: {label}",
                stop_reason="formal_namespace_blocked",
                details={"label": label, "output_dir": str(output_dir)},
            )


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _validate_command_argv(argv: Sequence[str], *, user_confirmed_formal: bool, output_dir: Path) -> tuple[str, ...]:
    parts = tuple(str(item) for item in argv if str(item) != "")
    if not parts:
        raise BridgeError("repo-command requires a command after --", stop_reason="target_invalid")
    classification = BridgeCommandPolicy.classify_repo_command(
        parts,
        output_dir=output_dir,
        user_confirmed_formal=user_confirmed_formal,
    )
    if not classification.allowed:
        raise BridgeError(classification.reason, stop_reason=classification.stop_reason, details=classification.to_dict())
    validate_runtime_command_safety(shlex.join(parts), output_dir=output_dir, user_confirmed_formal=user_confirmed_formal)
    return parts


def runtime_command_argv(options: BridgeOptions, run_id: str) -> tuple[str, ...]:
    output_dir = resolve_runtime_output_dir(options.output_dir, run_id, user_confirmed_formal=options.user_confirmed_formal)
    mode = options.mode
    if mode in {"repo-command", "long-run"}:
        return _validate_command_argv(
            options.command_argv,
            user_confirmed_formal=options.user_confirmed_formal,
            output_dir=output_dir,
        )
    if mode == "repo-script":
        script = resolve_repo_local_path(str(options.script_path or ""), label="repo script")
        args = tuple(options.command_argv)
        if script.suffix == ".py":
            command = (str(D4C_PYTHON), _repo_relative(script), *args)
        else:
            command = ("bash", _repo_relative(script), *args)
        return _validate_command_argv(command, user_confirmed_formal=options.user_confirmed_formal, output_dir=output_dir)
    if mode == "repo-module":
        module = validate_module_name(str(options.module_name or ""))
        command = (str(D4C_PYTHON), "-m", module, *tuple(options.command_argv))
        return _validate_command_argv(command, user_confirmed_formal=options.user_confirmed_formal, output_dir=output_dir)
    if mode == "command-file":
        command_file = resolve_repo_local_path(str(options.command_file or ""), label="command file")
        allowed_roots = tuple(root.resolve() for root in command_file_allowed_roots())
        if not any(command_file == root or root in command_file.parents for root in allowed_roots):
            raise BridgeError(
                "generated command-file must live under AI_analysis/06_probe_evidence, AI_analysis/07_runtime_evidence, runs/step3_validation, runs/step4_preflight, or runs/step4_validation",
                stop_reason="forbidden_mode",
                details=str(command_file),
            )
        validate_runtime_command_safety(
            command_file.read_text(encoding="utf-8", errors="ignore"),
            output_dir=output_dir,
            user_confirmed_formal=options.user_confirmed_formal,
        )
        command = ("bash", _repo_relative(command_file), *tuple(options.command_argv))
        return _validate_command_argv(command, user_confirmed_formal=options.user_confirmed_formal, output_dir=output_dir)
    raise BridgeError(f"{mode} is not a generic runtime command mode", stop_reason="forbidden_mode")


def _script_header(paths: GeneratedPaths) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cd {REPO_ROOT}
        export TMPDIR={SAFE_TMPDIR}
        export TMP="$TMPDIR"
        export TEMP="$TMPDIR"
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        export HF_EVALUATE_OFFLINE=1
        LOG={paths.log}
        STATUS={paths.status}
        PYTHON={D4C_PYTHON}
        """
    )


def _marker_script_header(paths: GeneratedPaths) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cd {REPO_ROOT}
        export TMPDIR={SAFE_TMPDIR}
        export TMP="$TMPDIR"
        export TEMP="$TMPDIR"
        LOG={paths.log}
        STATUS={paths.status}
        mkdir -p "$(dirname "$LOG")" "$(dirname "$STATUS")"
        : > "$LOG"
        """
    )


def _python_common_prelude(paths: GeneratedPaths, run_id: str, kind: str, timeouts: ResolvedTimeouts, success_condition: str) -> str:
    return f"""
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

RUN_ID = {run_id!r}
KIND = {kind!r}
LOG_PATH = Path({str(paths.log)!r})
STATUS_PATH = Path({str(paths.status)!r})
SUCCESS_CONDITION = {success_condition!r}
STARTUP_TIMEOUT_S = {timeouts.startup_timeout_s}
FIRST_RESULT_TIMEOUT_S = {timeouts.first_result_timeout_s}
HARD_TIMEOUT_S = {timeouts.hard_timeout_s}
SCHEMA = {BRIDGE_STATUS_SCHEMA!r}
START = time.monotonic()
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG = LOG_PATH.open("a", encoding="utf-8")

def emit(*parts):
    line = " ".join(str(part) for part in parts)
    print(line, flush=True)
    LOG.write(line + "\\n")
    LOG.flush()

def finish(success, exit_code, stop_reason, metrics=None, first_result_seen=False):
    elapsed = round(time.monotonic() - START, 3)
    status = {{
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "kind": KIND,
        "success": bool(success),
        "exit_code": int(exit_code),
        "elapsed_s": elapsed,
        "startup_timeout_s": STARTUP_TIMEOUT_S,
        "first_result_timeout_s": FIRST_RESULT_TIMEOUT_S,
        "hard_timeout_s": HARD_TIMEOUT_S,
        "first_result_seen": bool(first_result_seen),
        "success_condition": SUCCESS_CONDITION,
        "stop_reason": stop_reason,
        "metrics": metrics or {{}},
    }}
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return int(exit_code)

def run_cmd(args):
    emit("$", " ".join(args))
    proc = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            emit(line)
    if proc.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(args))
    return proc.stdout
"""


def build_validate_only_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    target: PaneCandidate,
) -> str:
    spec = OPERATION_SPECS["validate-only"]
    body = _python_common_prelude(paths, run_id, "validate-only", timeouts, spec.success_condition)
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    emit("validate_only_target", {target.socket!r}, {target.target!r})
    emit("validate_only_job", {target.job_id!r}, {target.step_id!r}, {target.node!r})
    host_out = run_cmd(["hostname"]).strip()
    emit("$", "pwd")
    repo_path = os.getcwd()
    emit(repo_path)
    smi = run_cmd(["nvidia-smi", "-L"])
    import torch
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "UNSET")
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    device_names = [torch.cuda.get_device_name(idx) for idx in range(device_count)]
    emit("hostname", host_out)
    emit("SLURM_JOB_ID=" + slurm_job_id)
    emit("CUDA_VISIBLE_DEVICES=" + cuda_visible_devices)
    emit("torch_cuda_available", cuda_available)
    emit("torch_device_count", device_count)
    for idx, name in enumerate(device_names):
        emit("torch_device", idx, name)
    metrics = {{
        "target_socket": {target.socket!r},
        "target": {target.target!r},
        "pane_id": {target.pane_id!r},
        "job_id": {target.job_id!r},
        "step_id": {target.step_id!r},
        "node": {target.node!r},
        "gpu_tres": {target.gpu!r},
        "hostname": host_out,
        "repo_path": repo_path,
        "cuda_visible_devices": cuda_visible_devices,
        "slurm_job_id": slurm_job_id,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": device_names,
        "nvidia_smi_lines": [line for line in smi.splitlines() if line.strip()],
    }}
    if repo_path != {str(REPO_ROOT)!r}:
        raise RuntimeError(f"target pane cwd is not repo root: {{repo_path}}")
    if not cuda_available or device_count < 2:
        raise RuntimeError("Current tmux CUDA validation requires torch.cuda device_count >= 2.")
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(True, 0, {spec.stop_reason_success!r}, metrics, True)
except Exception as exc:
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, "script_exit_nonzero", {{"error": repr(exc)}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def build_marker_probe_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    _target: PaneCandidate,
) -> str:
    spec = OPERATION_SPECS["marker-probe"]
    status = {
        "schema_version": BRIDGE_STATUS_SCHEMA,
        "run_id": run_id,
        "kind": "marker-probe",
        "success": True,
        "exit_code": 0,
        "elapsed_s": 0.0,
        "startup_timeout_s": timeouts.startup_timeout_s,
        "first_result_timeout_s": timeouts.first_result_timeout_s,
        "hard_timeout_s": timeouts.hard_timeout_s,
        "first_result_seen": True,
        "success_condition": spec.success_condition,
        "stop_reason": spec.stop_reason_success,
        "metrics": {"send_marker": f"ODCR_BRIDGE_SEND_OK_{run_id}"},
    }
    status_json = json.dumps(status, indent=2, sort_keys=True)
    body = f"""emit() {{
  line="$*"
  printf '%s\\n' "$line" | tee -a "$LOG"
}}
emit "ODCR_BRIDGE_BEGIN_{run_id}"
emit '$ hostname'
host_out="$(hostname)"
emit "$host_out"
emit '$ pwd'
pwd_out="$(pwd)"
emit "$pwd_out"
emit "ODCR_BRIDGE_SEND_OK_{run_id}"
cat > "$STATUS" <<'JSON'
{status_json}
JSON
emit "ODCR_BRIDGE_END_{run_id}"
"""
    return _marker_script_header(paths) + body


def build_cuda_probe_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    _target: PaneCandidate,
) -> str:
    spec = OPERATION_SPECS["cuda-probe"]
    body = _python_common_prelude(paths, run_id, "cuda-probe", timeouts, spec.success_condition)
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    host_out = run_cmd(["hostname"]).strip()
    emit("$", "pwd")
    emit(os.getcwd())
    emit("TMUX=" + os.environ.get("TMUX", "UNSET"))
    emit("SLURM_JOB_ID=" + os.environ.get("SLURM_JOB_ID", "UNSET"))
    emit("TMPDIR=" + os.environ.get("TMPDIR", "UNSET"))
    emit("XDG_RUNTIME_DIR=" + os.environ.get("XDG_RUNTIME_DIR", "UNSET"))
    run_cmd(["which", "python"])
    smi = run_cmd(["nvidia-smi", "-L"])
    import torch
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    emit("torch_cuda_available", cuda_available)
    emit("torch_device_count", device_count)
    for idx, name in enumerate(device_names):
        emit("torch_device", idx, name)
    metrics = {{
        "hostname": host_out,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": device_names,
        "nvidia_smi_lines": [line for line in smi.splitlines() if line.strip()],
    }}
    if not cuda_available or device_count < 1 or not device_names:
        emit("ODCR_BRIDGE_END_" + RUN_ID)
        exit_code = finish(False, 2, "cuda_unavailable", metrics, False)
    else:
        emit("ODCR_BRIDGE_END_" + RUN_ID)
        exit_code = finish(True, 0, {spec.stop_reason_success!r}, metrics, True)
except Exception as exc:
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, "script_exit_nonzero", {{"error": repr(exc)}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def _model_path_reader_body() -> str:
    return """
def read_sentence_model_path():
    import yaml
    cfg_path = Path("configs/odcr.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    env = cfg.get("env") or {}
    value = env.get("sentence_embed_model")
    if not value:
        raise RuntimeError("configs/odcr.yaml env.sentence_embed_model is required")
    return str(value)

def read_embed_dim():
    import yaml
    cfg_path = Path("configs/odcr.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    env = cfg.get("env") or {}
    value = env.get("embed_dim")
    if value is None:
        raise RuntimeError("configs/odcr.yaml env.embed_dim is required")
    return int(value)
"""


def build_bge_smoke_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    _target: PaneCandidate,
) -> str:
    spec = OPERATION_SPECS["bge-smoke"]
    body = _python_common_prelude(paths, run_id, "bge-smoke", timeouts, spec.success_condition)
    body += _model_path_reader_body()
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    phase = "dependency_import"
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    phase = "cuda_check"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the target tmux pane")
    device = torch.device("cuda")
    model_path = read_sentence_model_path()
    phase = "local_model"
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"local BGE model path does not exist: {{model_path}}")
    texts = ["ODCR short GPU smoke test.", "BGE local single batch."]
    t0 = time.monotonic()
    phase = "tokenizer_load"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    phase = "model_load"
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.to("cuda").eval()
    expected_hidden = read_embed_dim()
    actual_hidden = int(getattr(model.config, "hidden_size", 0))
    if actual_hidden != expected_hidden:
        raise ValueError(f"BGE hidden_size={{actual_hidden}} does not match expected smoke dim {{expected_hidden}}")
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    load_seconds = round(time.monotonic() - t0, 3)
    t1 = time.monotonic()
    phase = "tokenize"
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {{key: value.to(device) for key, value in encoded.items()}}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    autocast_enabled = bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False
    phase = "forward"
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=autocast_enabled):
            outputs = model(**encoded)
            # CLS pooling matches the formal preprocess_b embedding path.
            pooled = outputs.last_hidden_state[:, 0, :].to(torch.float32)
            vectors_t = F.normalize(pooled, p=2, dim=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    encode_seconds = round(time.monotonic() - t1, 3)
    vectors = vectors_t.detach().cpu().numpy()
    shape = list(vectors.shape)
    dtype = str(vectors.dtype)
    norm = float(np.linalg.norm(vectors[0]))
    memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    emit("BGE load_seconds", load_seconds)
    emit("BGE encode_seconds", encode_seconds)
    emit("BGE shape", shape)
    emit("BGE dtype", dtype)
    emit("BGE pooling", "cls")
    emit("BGE norm0", round(norm, 6))
    emit("GPU memory", memory)
    metrics = {{"load_seconds": load_seconds, "encode_seconds": encode_seconds, "shape": shape, "dtype": dtype, "pooling": "cls", "norm0": norm, "gpu_max_memory_allocated": memory, "autocast_bf16": autocast_enabled, "model_path": model_path}}
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(True, 0, {spec.stop_reason_success!r}, metrics, True)
except Exception as exc:
    message = repr(exc)
    lowered = message.lower()
    if "out of memory" in lowered:
        stop_reason = "oom"
    elif phase == "dependency_import":
        stop_reason = "dependency_import_error"
    elif phase == "local_model":
        stop_reason = "local_model_missing"
    elif phase == "tokenizer_load":
        stop_reason = "tokenizer_load_failed"
    elif phase == "model_load":
        stop_reason = "model_load_failed"
    elif phase == "cuda_check":
        stop_reason = "cuda_unavailable"
    elif phase in {{"tokenize", "forward"}}:
        stop_reason = "forward_failed"
    else:
        stop_reason = "script_exit_nonzero"
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, stop_reason, {{"error": repr(exc), "failure_class": stop_reason, "phase": phase}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def build_preprocess_dryrun_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    _target: PaneCandidate,
    stage: str,
) -> str:
    if stage not in {"b", "c"}:
        raise BridgeError("preprocess dry-run stage must be b or c", stop_reason="forbidden_mode")
    spec = OPERATION_SPECS["preprocess-dryrun"]
    command = ["./odcr", "preprocess", stage, "--dry-run"]
    body = _python_common_prelude(paths, run_id, "preprocess-dryrun", timeouts, spec.success_condition)
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    output = run_cmd({command!r})
    expected_terms = ["gpu", "model", "batch"]
    metrics = {{"stage": {stage!r}, "output_lines": len(output.splitlines()), "contains_expected_terms": all(term in output.lower() for term in expected_terms)}}
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(True, 0, {spec.stop_reason_success!r}, metrics, True)
except Exception as exc:
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, "script_exit_nonzero", {{"error": repr(exc), "stage": {stage!r}}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def build_micro_benchmark_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    _target: PaneCandidate,
    benchmark_kind: str,
    *,
    embed_batch_size: int = 512,
    read_chunk_rows: int = 100_000,
    group_shard_size: int = 4_096,
    workers: int = 2,
    bf16_enabled: bool = True,
    tf32_enabled: bool = True,
    grouped_text_cache_enabled: bool = True,
) -> str:
    if benchmark_kind != "bge-single-batch":
        raise BridgeError("only bge-single-batch micro benchmark is allowed", stop_reason="forbidden_mode")
    embed_batch_size = max(1, int(embed_batch_size))
    read_chunk_rows = max(1, int(read_chunk_rows))
    group_shard_size = max(1, int(group_shard_size))
    workers = max(1, int(workers))
    spec = OPERATION_SPECS["micro-benchmark"]
    benchmark_config = {
        "benchmark_kind": benchmark_kind,
        "embed_batch_size": embed_batch_size,
        "read_chunk_rows": read_chunk_rows,
        "group_shard_size": group_shard_size,
        "workers": workers,
        "bf16_enabled": bool(bf16_enabled),
        "tf32_enabled": bool(tf32_enabled),
        "grouped_text_cache_enabled": bool(grouped_text_cache_enabled),
        "warmup_batches": 1,
        "measured_batches": 1,
        "loader": "transformers.AutoTokenizer/AutoModel",
        "local_files_only": True,
    }
    body = _python_common_prelude(paths, run_id, "micro-benchmark", timeouts, spec.success_condition)
    body += _model_path_reader_body()
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    benchmark_config = {benchmark_config!r}
    phase = "dependency_import"
    import threading
    import torch
    from transformers import AutoModel, AutoTokenizer
    phase = "cuda_check"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the target tmux pane")
    device_count = int(torch.cuda.device_count())
    device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    if device_count < 1:
        raise RuntimeError("CUDA device_count is zero")
    torch.backends.cuda.matmul.allow_tf32 = bool(benchmark_config["tf32_enabled"])
    torch.backends.cudnn.allow_tf32 = bool(benchmark_config["tf32_enabled"])
    workers_used = min(int(benchmark_config["workers"]), device_count)
    device_indices = list(range(workers_used))
    model_path = read_sentence_model_path()
    expected_hidden = read_embed_dim()
    phase = "local_model"
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"local BGE model path does not exist: {{model_path}}")
    sample_text = (
        "ODCR preprocess B profile text for local BGE embedding benchmark. "
        "It mixes content evidence, review style, rating context, and domain semantics. "
    ) * 4
    batch = [sample_text + f" sample {{idx}}" for idx in range(int(benchmark_config["embed_batch_size"]))]
    autocast_enabled = bool(benchmark_config["bf16_enabled"] and torch.cuda.is_bf16_supported())
    workers_state = []
    load_started = time.monotonic()
    for worker_id, device_idx in enumerate(device_indices):
        phase = f"tokenizer_load_worker_{{worker_id}}"
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        phase = f"model_load_worker_{{worker_id}}"
        device = torch.device(f"cuda:{{device_idx}}")
        with torch.cuda.device(device):
            model = AutoModel.from_pretrained(model_path, local_files_only=True)
            model.to(device).eval()
            actual_hidden = int(getattr(model.config, "hidden_size", 0))
            if actual_hidden != expected_hidden:
                raise ValueError(f"BGE hidden_size={{actual_hidden}} does not match expected dim {{expected_hidden}}")
        workers_state.append({{"worker_id": worker_id, "device_idx": device_idx, "device": device, "tokenizer": tokenizer, "model": model}})
    if torch.cuda.is_available():
        for item in workers_state:
            torch.cuda.synchronize(item["device"])
    load_seconds = time.monotonic() - load_started

    def encode_once(item, *, measured):
        device = item["device"]
        tokenizer = item["tokenizer"]
        model = item["model"]
        tokenize_started = time.monotonic()
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        tokenize_s = time.monotonic() - tokenize_started
        encoded = {{key: value.to(device) for key, value in encoded.items()}}
        torch.cuda.synchronize(device)
        forward_started = time.monotonic()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = model(**encoded)
                pooled = outputs.last_hidden_state[:, 0, :].detach().to(torch.float32)
        torch.cuda.synchronize(device)
        forward_s = time.monotonic() - forward_started
        shape = [int(pooled.shape[0]), int(pooled.shape[1])]
        if measured:
            return {{"worker_id": item["worker_id"], "device_idx": item["device_idx"], "tokenize_s": tokenize_s, "forward_s": forward_s, "shape": shape}}
        return {{"worker_id": item["worker_id"], "device_idx": item["device_idx"], "tokenize_s": tokenize_s, "forward_s": forward_s, "shape": shape}}

    def run_parallel(*, measured):
        results = []
        errors = []
        lock = threading.Lock()
        def worker(item):
            try:
                result = encode_once(item, measured=measured)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append({{"worker_id": item["worker_id"], "device_idx": item["device_idx"], "error": repr(exc)}})
        threads = [threading.Thread(target=worker, args=(item,)) for item in workers_state]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("worker failed: " + repr(errors))
        return sorted(results, key=lambda item: int(item["worker_id"]))

    emit("micro_benchmark_warmup_batches", 1)
    phase = "forward"
    warmup_results = run_parallel(measured=False)
    for item in workers_state:
        torch.cuda.synchronize(item["device"])
        torch.cuda.reset_peak_memory_stats(item["device"])
    emit("micro_benchmark_measured_batches", 1)
    t0 = time.monotonic()
    measured_results = run_parallel(measured=True)
    for item in workers_state:
        torch.cuda.synchronize(item["device"])
    latency = time.monotonic() - t0
    total_samples = int(len(batch) * workers_used)
    samples_per_sec = total_samples / latency if latency > 0 else 0.0
    memory_by_device = []
    for item in workers_state:
        with torch.cuda.device(item["device"]):
            memory_by_device.append({{"device_idx": item["device_idx"], "max_memory_allocated": int(torch.cuda.max_memory_allocated(item["device"]))}})
    memory = max((int(item["max_memory_allocated"]) for item in memory_by_device), default=0)
    tokenize_s = max((float(item["tokenize_s"]) for item in measured_results), default=0.0)
    forward_s = max((float(item["forward_s"]) for item in measured_results), default=0.0)
    emit("latency_seconds", round(latency, 6))
    emit("samples_per_sec", round(samples_per_sec, 3))
    emit("batch_size", int(benchmark_config["embed_batch_size"]))
    emit("workers_used", workers_used)
    emit("total_samples", total_samples)
    emit("tokenize_seconds", round(tokenize_s, 6))
    emit("forward_seconds", round(forward_s, 6))
    emit("gpu_max_memory_allocated", memory)
    metrics = {{
        "kind": {benchmark_kind!r},
        "parameters": benchmark_config,
        "warmup_batches": 1,
        "measured_batches": 1,
        "load_seconds": load_seconds,
        "latency_seconds": latency,
        "samples_per_sec": samples_per_sec,
        "batch_size": int(benchmark_config["embed_batch_size"]),
        "total_samples": total_samples,
        "workers_requested": int(benchmark_config["workers"]),
        "workers_used": workers_used,
        "device_count": device_count,
        "device_names": device_names,
        "tokenize_s": tokenize_s,
        "forward_s": forward_s,
        "worker_results": measured_results,
        "warmup_results": warmup_results,
        "gpu_max_memory_allocated": memory,
        "max_vram_gb": memory / (1024 ** 3),
        "gpu_memory_by_device": memory_by_device,
        "autocast_bf16": autocast_enabled,
        "tf32_enabled": bool(benchmark_config["tf32_enabled"]),
        "model_path": model_path,
    }}
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(True, 0, {spec.stop_reason_success!r}, metrics, True)
except Exception as exc:
    lowered = repr(exc).lower()
    if "out of memory" in lowered:
        stop_reason = "oom"
    elif phase == "dependency_import":
        stop_reason = "dependency_import_error"
    elif phase == "cuda_check":
        stop_reason = "cuda_unavailable"
    elif "tokenizer_load" in phase:
        stop_reason = "tokenizer_load_failed"
    elif "model_load" in phase:
        stop_reason = "model_load_failed"
    elif phase == "forward":
        stop_reason = "forward_failed"
    else:
        stop_reason = "script_exit_nonzero"
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, stop_reason, {{"error": repr(exc), "failure_class": stop_reason, "phase": phase, "parameters": benchmark_config}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def build_real_data_probe_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    _target: PaneCandidate,
    probe_stage: str = "b",
) -> str:
    if probe_stage not in {"b", "c"}:
        raise BridgeError("real-data-probe requires --probe-stage b or c", stop_reason="forbidden_mode")
    spec = OPERATION_SPECS["real-data-probe"]
    body = _python_common_prelude(paths, run_id, "real-data-probe", timeouts, spec.success_condition)
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    phase = "real_data_probe_script"
    optimized_probe = Path("AI_analysis/runtime/optimize_preprocess_bc_cpu_gpu_pipeline_probe.py")
    legacy_probe = Path("AI_analysis/runtime/preprocess_{probe_stage}_real_probe/preprocess_{probe_stage}_real_data_probe.py")
    if optimized_probe.is_file():
        probe_script = optimized_probe
        cmd = [
            sys.executable,
            str(probe_script),
            "--stage",
            {probe_stage!r},
            "--bridge-run-id",
            RUN_ID,
        ]
        metrics_path = Path("AI_analysis/runtime/optimize_preprocess_bc_cpu_gpu_pipeline/gpu_probe_results.json")
    else:
        probe_script = legacy_probe
        if not probe_script.is_file():
            raise FileNotFoundError(f"real data probe script missing: {{probe_script}}")
        cmd = [
            sys.executable,
            str(probe_script),
            "gpu-probe",
            "--bridge-run-id",
            RUN_ID,
            "--max-runtime-s",
            str(HARD_TIMEOUT_S - 5),
        ]
        metrics_path = Path("AI_analysis/runtime/preprocess_{probe_stage}_real_probe/gpu_results.json")
    emit("$", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            emit(line)
    metrics = {{"probe_stage": {probe_stage!r}, "probe_script": str(probe_script), "probe_returncode": proc.returncode}}
    if metrics_path.is_file():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.update(payload.get("bridge_metrics") or payload)
        except Exception as metrics_exc:
            metrics["metrics_read_error"] = repr(metrics_exc)
    if proc.returncode != 0:
        raise RuntimeError(f"real data probe failed with code {{proc.returncode}}")
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(True, 0, {spec.stop_reason_success!r}, metrics, True)
except Exception as exc:
    lowered = repr(exc).lower()
    if "out of memory" in lowered:
        stop_reason = "oom"
    elif phase == "real_data_probe_script":
        stop_reason = "real_data_probe_failed"
    else:
        stop_reason = "script_exit_nonzero"
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, stop_reason, {{"error": repr(exc), "failure_class": stop_reason, "phase": phase}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def build_repo_runtime_executor_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    target: PaneCandidate,
    options: BridgeOptions,
) -> str:
    del target
    spec = OPERATION_SPECS[options.mode]
    output_dir = resolve_runtime_output_dir(
        options.output_dir,
        run_id,
        user_confirmed_formal=options.user_confirmed_formal,
    )
    command = runtime_command_argv(options, run_id)
    command_text = shlex.join(command)
    validate_runtime_command_safety(
        command_text,
        output_dir=output_dir,
        user_confirmed_formal=options.user_confirmed_formal,
    )
    classification = BridgeCommandPolicy.classify_repo_command(
        command,
        output_dir=output_dir,
        user_confirmed_formal=options.user_confirmed_formal,
    )
    if not classification.allowed:
        raise BridgeError(classification.reason, stop_reason=classification.stop_reason, details=classification.to_dict())
    body = _python_common_prelude(paths, run_id, options.mode, timeouts, spec.success_condition)
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    command = {list(command)!r}
    command_policy = {classification.to_dict()!r}
    output_dir = Path({str(output_dir)!r})
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ODCR_RUNTIME_OUTPUT_DIR"] = str(output_dir)
    env["ODCR_GPU_RUNTIME_RUN_ID"] = RUN_ID
    env["ODCR_GPU_RUNTIME_MODE"] = KIND
    env["PYTHONPATH"] = str(Path("code").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    emit("cwd", str(Path.cwd()))
    emit("output_dir", str(output_dir))
    emit("command_json", json.dumps(command, sort_keys=True))

    def digest(path):
        import hashlib
        if not path.is_file():
            return None
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    formal_fixed = [
        "runs/step4/task2",
        "runs/step4/task2/" + "latest" + ".json",
        "runs/step5/task2/" + "latest" + ".json",
        "runs/eval/task2/" + "latest" + ".json",
        "runs/rerank/task2/" + "latest" + ".json",
    ]
    formal_globs = [
        "runs/step3/task*/" + "latest" + ".json",
        "runs/step3/task*/*/model/" + "best" + ".pth",
        "runs/step3/task*/*/model/" + "best_observed" + ".pth",
        "runs/step3/task*/*/model/" + "latest" + ".pth",
        "runs/step3/task*/*/state/" + "checkpoint_lineage" + ".json",
        "runs/step4/task2/*",
        "runs/step4/task2/*/meta/run_summary.json",
        "runs/step4/task2/*/" + "odcr_routing_train" + ".csv",
        "runs/step4/task2/*/meta/stage_status.json",
        "runs/step4/task2/*/meta/manifest.json",
        "runs/step4/task2/*/meta/index_contract.json",
        "runs/step5/task2/*",
        "runs/eval/task2/*",
        "runs/rerank/task2/*",
    ]

    def watched_paths():
        root = Path({str(REPO_ROOT)!r})
        paths = set(root / rel for rel in formal_fixed)
        for pattern in formal_globs:
            paths.update(root.glob(pattern))
        return sorted(paths, key=lambda p: p.as_posix())

    def snapshot_formal():
        root = Path({str(REPO_ROOT)!r})
        payload = {{}}
        for path in watched_paths():
            try:
                rel = path.resolve().relative_to(root).as_posix()
            except Exception:
                rel = str(path.resolve())
            if path.exists():
                st = path.stat()
                payload[rel] = {{
                    "exists": True,
                    "is_file": path.is_file(),
                    "is_dir": path.is_dir(),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                    "sha256": digest(path),
                }}
            else:
                payload[rel] = {{"exists": False}}
        return payload

    def read_json_or_empty(path):
        if not path.is_file():
            return {{}}
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {{}}
        return obj if isinstance(obj, dict) else {{}}

    def parse_step4_preflight():
        rank_by_level = {{
            "E0_static_config": 0,
            "E1_schema_preview": 1,
            "E2_cpu_real_data_no_model": 2,
            "E3_gpu_transport": 3,
            "E4_gpu_shard_forward_bounded": 4,
            "E5_formal_full_run": 5,
        }}
        def parse_level(*payloads):
            for payload in payloads:
                raw = str(payload.get("evidence_level") or "")
                if raw in rank_by_level:
                    return raw
                prefix = raw.split("_", 1)[0]
                for level in rank_by_level:
                    if level.startswith(prefix + "_"):
                        return level
            return ""
        def runtime_evidence(summary, gpu_snapshot, evidence_complete, formal_flags_ok):
            level = parse_level(summary, gpu_snapshot)
            checks = {{
                "evidence_level": level,
                "evidence_level_ok": rank_by_level.get(level, -1) >= rank_by_level["E4_gpu_shard_forward_bounded"],
                "gpu_runtime_evidence": summary.get("gpu_runtime_evidence") is True or gpu_snapshot.get("gpu_runtime_evidence") is True,
                "actual_gpu_forward_executed": summary.get("actual_gpu_forward_executed") is True or gpu_snapshot.get("actual_gpu_forward_executed") is True,
                "actual_model_loaded_on_gpu": summary.get("actual_model_loaded_on_gpu") is True or gpu_snapshot.get("actual_model_loaded_on_gpu") is True,
                "force_gpu_forward": summary.get("force_gpu_forward") is True or gpu_snapshot.get("force_gpu_forward") is True,
                "cuda_available": gpu_snapshot.get("cuda_available") is True,
                "evidence_complete": bool(evidence_complete),
                "formal_pollution_false": bool(formal_flags_ok),
                "child_process_ok_required_by_bridge_status": True,
            }}
            ok = all(bool(checks[key]) for key in ("evidence_level_ok", "gpu_runtime_evidence", "actual_gpu_forward_executed", "actual_model_loaded_on_gpu", "force_gpu_forward", "cuda_available", "evidence_complete", "formal_pollution_false"))
            return ok, checks
        required_names = {list(STEP4_PREFLIGHT_REQUIRED_ARTIFACTS)!r}
        dirs = [output_dir]
        if command_policy.get("stage") == "step4" and command_policy.get("operation") == "preflight":
            ns = str(command_policy.get("validation_namespace") or "")
            task = int(command_policy.get("task") or 2)
            if ns:
                dirs.extend([
                    Path({str(REPO_ROOT)!r}) / "runs" / "step4_preflight" / f"task{{task}}" / ns,
                    Path({str(REPO_ROOT)!r}) / "runs" / "step4_validation" / f"task{{task}}" / ns,
                ])
        seen_dirs = []
        unique_dirs = []
        for item in dirs:
            key = str(item.resolve())
            if key not in seen_dirs:
                seen_dirs.append(key)
                unique_dirs.append(item.resolve())
        for directory in unique_dirs:
            artifacts = {{name: directory / name for name in required_names}}
            if not any(path.is_file() for path in artifacts.values()):
                continue
            missing = [name for name, path in artifacts.items() if not path.is_file()]
            payloads = {{name: read_json_or_empty(path) for name, path in artifacts.items() if path.is_file()}}
            summary = payloads.get("preflight_summary.json", {{}})
            distribution = payloads.get("rcr_distribution.json", {{}})
            required = payloads.get("required_fields_check.json", {{}})
            lineage = payloads.get("lineage_preview.json", {{}})
            gpu_snapshot = payloads.get("cpu_gpu_utilization_snapshot.json", {{}})
            expected_ns = str(command_policy.get("validation_namespace") or "")
            namespace_ok = (not expected_ns) or str(summary.get("validation_namespace") or "") == expected_ns
            try:
                sample_count = int(summary.get("sample_count") or distribution.get("sample_count") or 0)
            except Exception:
                sample_count = 0
            try:
                max_samples = int(summary.get("max_samples") or command_policy.get("bounded_limit_value") or 0)
            except Exception:
                max_samples = 0
            bounded_ok = sample_count > 0 and (max_samples <= 0 or sample_count <= max_samples)
            rcr_counts_present = all(key in distribution for key in ("route_scorer_count", "route_explainer_count", "train_keep_count", "confidence_bucket_distribution", "sample_weight_hint"))
            formal_flags_ok = summary.get("formal_latest_write") is False and summary.get("formal_export_write") is False
            upstream_ok = bool(summary.get("upstream_step3_run_id") or lineage.get("upstream_step3_run_id") or lineage.get("lineage_hash"))
            required_ok = bool(required.get("passed")) and not required.get("missing")
            evidence_complete = bool(not missing and namespace_ok and bounded_ok and rcr_counts_present and formal_flags_ok and upstream_ok and required_ok)
            runtime_evidence_ok, runtime_checks = runtime_evidence(summary, gpu_snapshot, evidence_complete, formal_flags_ok)
            gpu_transport_ok = bool(gpu_snapshot.get("cuda_available"))
            gpu_runtime_observed = bool(runtime_checks["gpu_runtime_evidence"])
            return {{
                "schema_version": "odcr_step4_bridge_preflight_evidence/1",
                "evidence_level": runtime_checks["evidence_level"] or ("E3_gpu_transport" if gpu_transport_ok else ""),
                "candidate_dirs": [str(path) for path in unique_dirs],
                "evidence_dir": str(directory),
                "artifact_paths": {{name: str(path) for name, path in artifacts.items()}},
                "missing_artifacts": missing,
                "evidence_complete": evidence_complete,
                "runtime_evidence_ok": runtime_evidence_ok,
                "gpu_transport_ok": gpu_transport_ok,
                "gpu_runtime_observed": gpu_runtime_observed,
                "not_step4_runtime_evidence": not runtime_evidence_ok,
                "validation_namespace": summary.get("validation_namespace"),
                "sample_count": sample_count,
                "max_samples": max_samples,
                "route_scorer_count": distribution.get("route_scorer_count"),
                "route_explainer_count": distribution.get("route_explainer_count"),
                "train_keep_count": distribution.get("train_keep_count"),
                "confidence_bucket_distribution": distribution.get("confidence_bucket_distribution"),
                "sample_weight_hint_stats": distribution.get("sample_weight_hint"),
                "formal_latest_write": summary.get("formal_latest_write"),
                "formal_export_write": summary.get("formal_export_write"),
                "upstream_step3_run_id": summary.get("upstream_step3_run_id") or lineage.get("upstream_step3_run_id"),
                "checks": {{
                    "namespace_ok": namespace_ok,
                    "bounded_ok": bounded_ok,
                    "rcr_counts_present": rcr_counts_present,
                    "formal_flags_ok": formal_flags_ok,
                    "upstream_ok": upstream_ok,
                    "required_fields_ok": required_ok,
                    **runtime_checks,
                }},
            }}
        probe_report = read_json_or_empty(output_dir / "report.json")
        if probe_report.get("cuda_available") is True:
            return {{
                "schema_version": "odcr_step4_bridge_preflight_evidence/1",
                "evidence_level": "E3_gpu_transport",
                "candidate_dirs": [str(path) for path in unique_dirs],
                "evidence_dir": str(output_dir),
                "artifact_paths": {{"report": str(output_dir / "report.json")}},
                "missing_artifacts": required_names,
                "evidence_complete": False,
                "runtime_evidence_ok": False,
                "gpu_transport_ok": True,
                "gpu_runtime_observed": False,
                "not_step4_runtime_evidence": True,
                "checks": {{
                    "cuda_available": True,
                    "cuda_probe_alone_is_not_step4_runtime_evidence": True,
                    "child_process_ok_required_by_bridge_status": True,
                }},
            }}
        return {{
            "schema_version": "odcr_step4_bridge_preflight_evidence/1",
            "evidence_level": "",
            "candidate_dirs": [str(path) for path in unique_dirs],
            "evidence_dir": "",
            "artifact_paths": {{}},
            "missing_artifacts": required_names,
            "evidence_complete": False,
            "runtime_evidence_ok": False,
            "gpu_transport_ok": False,
            "gpu_runtime_observed": False,
            "not_step4_runtime_evidence": True,
            "checks": {{}},
        }}

    before = snapshot_formal()
    started_child = time.monotonic()
    proc = subprocess.run(command, cwd={str(REPO_ROOT)!r}, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    child_elapsed = round(time.monotonic() - started_child, 3)
    if proc.stdout:
        for line in proc.stdout.splitlines():
            emit(line)
    after = snapshot_formal()
    pollution = before != after
    report_path = output_dir / "report.json"
    runtime_evidence_ok = None
    evidence_complete = False
    step4_preflight_evidence = {{}}
    report_payload = {{}}
    if command_policy.get("stage") == "step4" and command_policy.get("operation") == "preflight":
        step4_preflight_evidence = parse_step4_preflight()
        runtime_evidence_ok = bool(proc.returncode == 0 and step4_preflight_evidence.get("runtime_evidence_ok"))
        evidence_complete = bool(step4_preflight_evidence.get("evidence_complete"))
        final_artifact_completed = bool(not step4_preflight_evidence.get("missing_artifacts"))
    elif report_path.is_file():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            if "runtime_verified" in report_payload or "evidence_complete" in report_payload:
                runtime_evidence_ok = bool(report_payload.get("runtime_verified")) and bool(report_payload.get("evidence_complete"))
                evidence_complete = bool(report_payload.get("evidence_complete"))
            final_artifact_completed = True
        except Exception as report_exc:
            report_payload = {{"read_error": repr(report_exc)}}
            runtime_evidence_ok = False
            final_artifact_completed = False
    else:
        final_artifact_completed = False
    env_snapshot = {{
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "SLURM_JOB_ID": env.get("SLURM_JOB_ID", "UNSET"),
        "SLURM_STEP_ID": env.get("SLURM_STEP_ID", "UNSET"),
        "HOSTNAME": env.get("HOSTNAME", "UNSET"),
        "PYTHONPATH": env.get("PYTHONPATH", ""),
    }}
    child_ok = proc.returncode == 0
    success = bool(child_ok and not pollution and (runtime_evidence_ok is not False))
    metrics = {{
        "gpu_transport_ok": None,
        "pane_validated": True,
        "command_dispatched": True,
        "command_argv": command,
        "command_text": {command_text!r},
        "cwd": {str(REPO_ROOT)!r},
        "env": env_snapshot,
        "output_dir": str(output_dir),
        "child_process_started": True,
        "child_process_still_running": False,
        "child_process_ok": child_ok,
        "child_exit_code": int(proc.returncode),
        "child_returncode": int(proc.returncode),
        "child_elapsed_s": child_elapsed,
        "first_result_timeout": False,
        "first_result_timeout_recovered": False,
        "final_artifact_completed": final_artifact_completed,
        "runtime_evidence_ok": runtime_evidence_ok,
        "runtime_evidence_split_present": True,
        "evidence_complete": evidence_complete,
        "command_allowed_by_policy": True,
        "command_policy": command_policy,
        "step4_preflight_evidence": step4_preflight_evidence,
        "runtime_report_json": str(report_path) if report_path.is_file() else "",
        "runtime_report_summary": report_payload if report_payload else {{}},
        "formal_pollution": pollution,
        "formal_namespace_polluted": pollution,
        "formal_pollution_check_passed": not pollution,
        "formal_watch_before": before,
        "formal_watch_after": after,
        "stdout_line_count": len(proc.stdout.splitlines()) if proc.stdout else 0,
    }}
    emit("child_exit_code", proc.returncode)
    emit("formal_pollution", str(pollution).lower())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    stop_reason = {spec.stop_reason_success!r} if success else ("formal_namespace_polluted" if pollution else "child_exit_nonzero")
    exit_code = finish(success, 0 if success else int(proc.returncode or 1), stop_reason, metrics, True)
except Exception as exc:
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, "runtime_executor_failed", {{"error": repr(exc), "gpu_transport_ok": None, "pane_validated": True, "command_dispatched": False}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def build_long_run_managed_launcher_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    target: PaneCandidate,
    options: BridgeOptions,
) -> str:
    del target
    spec = OPERATION_SPECS["long-run"]
    output_dir = resolve_runtime_output_dir(
        options.output_dir,
        run_id,
        user_confirmed_formal=options.user_confirmed_formal,
    )
    command = runtime_command_argv(options, run_id)
    command_text = shlex.join(command)
    validate_runtime_command_safety(
        command_text,
        output_dir=output_dir,
        user_confirmed_formal=options.user_confirmed_formal,
    )
    body = _python_common_prelude(paths, run_id, "long-run", timeouts, spec.success_condition)
    body += f"""
exit_code = 0
try:
    emit("ODCR_BRIDGE_BEGIN_" + RUN_ID)
    command = {list(command)!r}
    command_text = {command_text!r}
    output_dir = Path({str(output_dir)!r})
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    command_sh = output_dir / "command.sh"
    status_path = output_dir / "status.json"
    heartbeat_path = output_dir / "heartbeat.json"
    pid_path = output_dir / "pid"
    manager_pid_path = output_dir / "manager.pid"
    monitor_path = output_dir / "managed_launcher.py"
    command_sh.write_text(
        "#!/usr/bin/env bash\\n"
        "set -euo pipefail\\n"
        f"cd {{Path.cwd()}}\\n"
        "export PYTHONPATH=\\\"" + str(Path("code").resolve()) + ":${{PYTHONPATH:-}}\\\"\\n"
        + command_text
        + "\\n",
        encoding="utf-8",
    )
    command_sh.chmod(0o700)
    monitor_source = r'''
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

run_id = {run_id!r}
repo_root = Path({str(REPO_ROOT)!r})
command_sh = Path({str(output_dir / "command.sh")!r})
status_path = Path({str(output_dir / "status.json")!r})
heartbeat_path = Path({str(output_dir / "heartbeat.json")!r})
pid_path = Path({str(output_dir / "pid")!r})
stdout_path = Path({str(output_dir / "stdout.log")!r})
stderr_path = Path({str(output_dir / "stderr.log")!r})
emergency_timeout_s = {timeouts.emergency_timeout_s!r}
schema = "odcr_detached_managed_launcher/1"

def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    tmp.replace(path)

def write_status(state, **extra):
    payload = {{
        "schema_version": schema,
        "run_id": run_id,
        "state": state,
        "command_sh": str(command_sh),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "pid_path": str(pid_path),
        "heartbeat_path": str(heartbeat_path),
        "emergency_timeout_s": emergency_timeout_s,
        "hostname": socket.gethostname(),
        "updated_at": now(),
    }}
    payload.update(extra)
    write_json(status_path, payload)
    write_json(heartbeat_path, {{"schema_version": schema, "run_id": run_id, "state": state, "heartbeat_at": now()}})

started_at = now()
write_status("starting", started_at=started_at, manager_pid=os.getpid())
stdout_path.parent.mkdir(parents=True, exist_ok=True)
with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
    proc = subprocess.Popen(
        ["bash", str(command_sh)],
        cwd=str(repo_root),
        stdin=subprocess.DEVNULL,
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
    )
    pid_path.write_text(str(proc.pid) + "\\n", encoding="utf-8")
    write_status("running", started_at=started_at, manager_pid=os.getpid(), child_pid=proc.pid)
    timed_out = False
    try:
        returncode = proc.wait(timeout=emergency_timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
    finished_at = now()
    final_state = "timeout" if timed_out else ("completed" if returncode == 0 else "failed")
    write_status(
        final_state,
        started_at=started_at,
        finished_at=finished_at,
        manager_pid=os.getpid(),
        child_pid=proc.pid,
        returncode=returncode,
        timed_out=timed_out,
    )
    sys.exit(124 if timed_out else int(returncode or 0))
'''
    monitor_path.write_text(monitor_source, encoding="utf-8")
    monitor_path.chmod(0o700)
    env = dict(os.environ)
    env["ODCR_RUNTIME_OUTPUT_DIR"] = str(output_dir)
    env["ODCR_GPU_RUNTIME_RUN_ID"] = RUN_ID
    env["ODCR_GPU_RUNTIME_MODE"] = KIND
    env["PYTHONPATH"] = str(Path("code").resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    manager = subprocess.Popen(
        [sys.executable, str(monitor_path)],
        cwd={str(REPO_ROOT)!r},
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    manager_pid_path.write_text(str(manager.pid) + "\\n", encoding="utf-8")
    launch_status = {{
        "schema_version": SCHEMA,
        "run_id": RUN_ID,
        "kind": KIND,
        "success": True,
        "exit_code": 0,
        "elapsed_s": round(time.monotonic() - START, 3),
        "startup_timeout_s": STARTUP_TIMEOUT_S,
        "first_result_timeout_s": FIRST_RESULT_TIMEOUT_S,
        "hard_timeout_s": HARD_TIMEOUT_S,
        "first_result_seen": True,
        "success_condition": SUCCESS_CONDITION,
        "stop_reason": {spec.stop_reason_success!r},
        "metrics": {{
            "detached": True,
            "managed_launcher": True,
            "manager_pid": manager.pid,
            "command_sh": str(command_sh),
            "managed_status_path": str(status_path),
            "managed_heartbeat_path": str(heartbeat_path),
            "managed_pid_path": str(pid_path),
            "managed_stdout_log": str(stdout_path),
            "managed_stderr_log": str(stderr_path),
            "output_dir": str(output_dir),
            "emergency_timeout_s": {timeouts.emergency_timeout_s!r},
            "command_text": command_text,
        }},
    }}
    STATUS_PATH.write_text(json.dumps(launch_status, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    emit("managed_launcher_started", str(output_dir))
    emit("manager_pid", manager.pid)
    emit("managed_status_path", str(status_path))
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = 0
except Exception as exc:
    emit("bridge_error", repr(exc))
    emit(traceback.format_exc())
    emit("ODCR_BRIDGE_END_" + RUN_ID)
    exit_code = finish(False, 1, "managed_launcher_failed", {{"error": repr(exc), "detached": True}}, False)
finally:
    LOG.close()
sys.exit(exit_code)
"""
    return _script_header(paths) + f'"$PYTHON" - <<\'PY\'\n{body}\nPY\n'


def build_step3_startup_validation_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    target: PaneCandidate,
    *,
    task_id: int = 2,
    validation_slug: str = "step3_tmux_gpu_bridge_startup_validation_closeout",
) -> str:
    task_id = int(task_id)
    if task_id != 2:
        raise BridgeError("step3-startup-validation currently allows only task 2", stop_reason="forbidden_mode")
    slug = _safe_run_component(validation_slug)
    if slug != validation_slug:
        raise BridgeError("step3-startup-validation validation slug is unsafe", stop_reason="forbidden_mode")
    max_seconds = max(5, min(int(timeouts.hard_timeout_s) - 5, 175))
    return (
        _script_header(paths)
        + "\n"
        + '"$PYTHON" code/tools/odcr_step3_startup_validation.py '
        + f"--task {task_id} --mode startup-only --namespace validation "
        + f"--slug {slug} --run-id {run_id} --max-seconds {max_seconds} "
        + f"--bridge-status-path {_mode_status_result_path(paths, 'step3-startup-validation')} "
        + f"--bridge-log-path {paths.log} "
        + f"--target-socket {target.socket} --target-pane {target.pane_id} "
        + f"--target-job-id {target.job_id} --target-node {target.node}\n"
    )


STEP3_PERFORMANCE_PROBE_TYPES = (
    "timing-profile-window",
    "prefetch-ab",
    "grad-monitor-window",
    "memory-phase-window",
    "ddp-gather-sync-window",
    "quality-checkpoint-window",
    "batch-ladder-window",
)


def build_step3_performance_probe_script(
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    target: PaneCandidate,
    *,
    task_id: int = 2,
    validation_slug: str = "step3_runtime_probe_truth_rebuild",
    probe_type: str = "timing-profile-window",
    candidate_name: str | None = None,
) -> str:
    task_id = int(task_id)
    if task_id != 2:
        raise BridgeError("step3-performance-probe currently allows only task 2", stop_reason="forbidden_mode")
    slug = _safe_run_component(validation_slug)
    if slug != validation_slug:
        raise BridgeError("step3-performance-probe validation slug is unsafe", stop_reason="forbidden_mode")
    probe = str(probe_type or "").strip()
    if probe not in STEP3_PERFORMANCE_PROBE_TYPES:
        raise BridgeError(f"unsupported Step3 performance probe type: {probe!r}", stop_reason="forbidden_mode")
    max_seconds = max(20, min(int(timeouts.hard_timeout_s) - 5, 175))
    candidate_arg = f"--candidate-name {shlex.quote(candidate_name)} " if candidate_name else ""
    return (
        _script_header(paths)
        + "\n"
        + '"$PYTHON" code/tools/odcr_step3_performance_probe.py '
        + f"--probe-type {probe} --namespace validation "
        + candidate_arg
        + f"--task {task_id} --slug {slug} --run-id {run_id} "
        + f"--warmup-steps 5 --measured-steps 20 --max-seconds {max_seconds} "
        + f"--bridge-status-path {_mode_status_result_path(paths, 'step3-performance-probe')} "
        + f"--bridge-log-path {paths.log} "
        + f"--target-socket {target.socket} --target-pane {target.pane_id} "
        + f"--target-job-id {target.job_id} --target-node {target.node}\n"
    )


def build_script(
    options: BridgeOptions,
    run_id: str,
    paths: GeneratedPaths,
    timeouts: ResolvedTimeouts,
    target: PaneCandidate,
) -> str:
    if options.mode == "validate-only":
        return build_validate_only_script(run_id, paths, timeouts, target)
    if options.mode == "marker-probe":
        return build_marker_probe_script(run_id, paths, timeouts, target)
    if options.mode == "cuda-probe":
        return build_cuda_probe_script(run_id, paths, timeouts, target)
    if options.mode == "bge-smoke":
        return build_bge_smoke_script(run_id, paths, timeouts, target)
    if options.mode == "preprocess-dryrun":
        if not options.stage:
            raise BridgeError("preprocess-dryrun requires --stage b or --stage c", stop_reason="forbidden_mode")
        return build_preprocess_dryrun_script(run_id, paths, timeouts, target, options.stage)
    if options.mode == "micro-benchmark":
        return build_micro_benchmark_script(
            run_id,
            paths,
            timeouts,
            target,
            options.benchmark_kind or "bge-single-batch",
            embed_batch_size=options.embed_batch_size,
            read_chunk_rows=options.read_chunk_rows,
            group_shard_size=options.group_shard_size,
            workers=options.workers,
            bf16_enabled=options.bf16_enabled,
            tf32_enabled=options.tf32_enabled,
            grouped_text_cache_enabled=options.grouped_text_cache_enabled,
        )
    if options.mode == "real-data-probe":
        return build_real_data_probe_script(run_id, paths, timeouts, target, options.probe_stage)
    if options.mode == "step3-startup-validation":
        return build_step3_startup_validation_script(
            run_id,
            paths,
            timeouts,
            target,
            task_id=options.task_id,
            validation_slug=options.validation_slug,
        )
    if options.mode == "step3-performance-probe":
        return build_step3_performance_probe_script(
            run_id,
            paths,
            timeouts,
            target,
            task_id=options.task_id,
            validation_slug=options.validation_slug,
            probe_type=options.performance_probe_type,
            candidate_name=options.candidate_name,
        )
    if options.mode in {"repo-command", "repo-script", "repo-module", "command-file"}:
        return build_repo_runtime_executor_script(run_id, paths, timeouts, target, options)
    if options.mode == "long-run":
        return build_long_run_managed_launcher_script(run_id, paths, timeouts, target, options)
    raise BridgeError(f"unsupported script mode: {options.mode}", stop_reason="forbidden_mode")


def normalize_repo_runtime_success(status: Mapping[str, Any], *, gpu_transport_ok: bool = True) -> dict[str, Any]:
    out = dict(status)
    metrics = dict(out.get("metrics") or {})
    child_exit_code = int(metrics.get("child_exit_code", out.get("exit_code", 1)) or 0)
    child_returncode = int(metrics.get("child_returncode", child_exit_code) or 0)
    child_process_ok = bool(metrics.get("child_process_ok", child_exit_code == 0))
    formal_pollution = bool(metrics.get("formal_pollution", metrics.get("formal_namespace_polluted", False)))
    runtime_evidence_ok = metrics.get("runtime_evidence_ok")
    runtime_required_failed = runtime_evidence_ok is False
    evidence_complete = bool(metrics.get("evidence_complete", False))
    final_artifact_completed = bool(metrics.get("final_artifact_completed", evidence_complete))
    first_result_timeout = bool(metrics.get("first_result_timeout", out.get("stop_reason") == "first_result_timeout"))
    first_result_timeout_recovered = bool(metrics.get("first_result_timeout_recovered", False))
    final_success = bool(gpu_transport_ok and child_process_ok and child_exit_code == 0 and not formal_pollution and not runtime_required_failed)
    timeout_stop_reasons = {"startup_timeout", "first_result_timeout", "hard_timeout"}
    failure_stop_reason = (
        out.get("stop_reason")
        if out.get("stop_reason") in timeout_stop_reasons
        else (
            "formal_namespace_polluted"
            if formal_pollution
            else ("runtime_evidence_failed" if runtime_required_failed else "child_exit_nonzero")
        )
    )
    metrics.update(
        {
            "gpu_transport_ok": bool(gpu_transport_ok),
            "bridge_transport_ok": bool(gpu_transport_ok),
            "pane_validated": bool(metrics.get("pane_validated", True)),
            "command_dispatched": bool(metrics.get("command_dispatched", True)),
            "child_process_ok": bool(child_process_ok),
            "child_exit_code": int(child_exit_code),
            "child_returncode": int(child_returncode),
            "child_process_started": bool(metrics.get("child_process_started", metrics.get("command_dispatched", True))),
            "child_process_still_running": bool(metrics.get("child_process_still_running", False)),
            "first_result_timeout": first_result_timeout,
            "first_result_timeout_recovered": first_result_timeout_recovered,
            "final_artifact_completed": final_artifact_completed,
            "runtime_evidence_ok": runtime_evidence_ok,
            "runtime_evidence_split_present": bool(metrics.get("runtime_evidence_split_present", True)),
            "evidence_complete": evidence_complete,
            "formal_pollution": bool(formal_pollution),
            "formal_namespace_polluted": bool(formal_pollution),
            "formal_pollution_check_passed": not bool(formal_pollution),
            "command_allowed_by_policy": bool(metrics.get("command_allowed_by_policy", True)),
            "final_success": bool(final_success),
        }
    )
    out.update(
        {
            "gpu_transport_ok": bool(gpu_transport_ok),
            "bridge_transport_ok": bool(gpu_transport_ok),
            "pane_validated": bool(metrics["pane_validated"]),
            "command_dispatched": bool(metrics["command_dispatched"]),
            "child_process_ok": bool(child_process_ok),
            "child_exit_code": int(child_exit_code),
            "child_returncode": int(child_returncode),
            "first_result_timeout": first_result_timeout,
            "first_result_timeout_recovered": first_result_timeout_recovered,
            "final_artifact_completed": final_artifact_completed,
            "runtime_evidence_ok": runtime_evidence_ok,
            "runtime_evidence_split_present": bool(metrics["runtime_evidence_split_present"]),
            "evidence_complete": evidence_complete,
            "formal_pollution": bool(formal_pollution),
            "formal_namespace_polluted": bool(formal_pollution),
            "formal_pollution_check_passed": not bool(formal_pollution),
            "command_allowed_by_policy": bool(metrics["command_allowed_by_policy"]),
            "final_success": bool(final_success),
            "success": bool(final_success),
            "exit_code": 0 if final_success else (child_exit_code or 1),
            "stop_reason": out.get("stop_reason") if final_success else failure_stop_reason,
            "metrics": metrics,
        }
    )
    return out


def collect_run(run_id: str) -> dict[str, Any]:
    paths = make_generated_paths(run_id)
    log_exists = paths.log.is_file()
    status_exists = paths.status.is_file()
    summary_exists = paths.summary.is_file()
    report_exists = paths.report.is_file()
    log_text = paths.log.read_text(encoding="utf-8") if log_exists else ""
    end_marker = f"ODCR_BRIDGE_END_{run_id}" in log_text
    status_data: dict[str, Any] = {}
    if status_exists:
        try:
            status_data = _read_json(paths.status)
        except json.JSONDecodeError:
            status_data = {}
    complete = bool(status_data and end_marker)
    incomplete_stop_reason = "TIMEOUT_INCOMPLETE"
    source_success = bool(status_data.get("success", False))
    managed_status: dict[str, Any] = {}
    managed_status_path = ""
    if str(status_data.get("kind") or "") == "long-run":
        metrics = status_data.get("metrics") if isinstance(status_data.get("metrics"), dict) else {}
        managed_status_path = str(metrics.get("managed_status_path") or "")
        if managed_status_path:
            managed_path = Path(managed_status_path)
            if managed_path.is_file():
                try:
                    managed_status = _read_json(managed_path)
                except json.JSONDecodeError:
                    managed_status = {}
        state = str(managed_status.get("state") or "")
        if state:
            source_success = source_success and state == "completed" and int(managed_status.get("returncode") or 0) == 0
        else:
            source_success = False
            incomplete_stop_reason = "MANAGED_LONG_RUN_INCOMPLETE"
    result = {
        "schema_version": BRIDGE_STATUS_SCHEMA,
        "run_id": run_id,
        "kind": "collect",
        "success": bool(complete and source_success),
        "exit_code": 0 if complete and source_success else 1,
        "elapsed_s": 0.0,
        "startup_timeout_s": OPERATION_SPECS["collect"].startup_timeout_s,
        "first_result_timeout_s": OPERATION_SPECS["collect"].first_result_timeout_s,
        "hard_timeout_s": OPERATION_SPECS["collect"].auto_timeout_s,
        "first_result_seen": bool(complete),
        "success_condition": OPERATION_SPECS["collect"].success_condition,
        "stop_reason": "collected" if complete and source_success else incomplete_stop_reason,
        "metrics": {
            "log_exists": log_exists,
            "status_exists": status_exists,
            "summary_exists": summary_exists,
            "report_exists": report_exists,
            "end_marker_seen": end_marker,
            "source_status": status_data,
            "managed_status_path": managed_status_path,
            "managed_status": managed_status,
            "bridge_transport_ok": bool(complete),
            "child_process_ok": bool((status_data.get("child_process_ok") if status_data else False)),
            "runtime_probe_ok": bool((status_data.get("runtime_probe_ok") if status_data else False)),
            "runtime_verified": bool((status_data.get("runtime_verified") if status_data else False)),
            "evidence_complete": bool((status_data.get("evidence_complete") if status_data else False)),
            "formal_namespace_polluted": bool((status_data.get("formal_namespace_polluted") if status_data else False)),
            "rank_evidence": "runtime_status_collected",
        },
        "paths": paths.to_dict(),
    }
    return result


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Plan target/script only; do not send keys.")
    parser.add_argument("--timeout", default="auto", help="auto or explicit seconds; capped by mode safety limits.")
    parser.add_argument("--socket", help="Explicit tmux socket path.")
    parser.add_argument("--target", help="Explicit tmux target such as odcr:0.0 or %%0.")
    parser.add_argument("--run-id", help="Optional run id; defaults to bridge_YYYYMMDD_HHMMSS_kind.")
    parser.add_argument("--no-send", action="store_true", help="Generate script/status/report without send-keys.")
    parser.add_argument("--strict", action="store_true", help="Fail fast if target discovery is not unique.")
    parser.add_argument(
        "--send-method",
        choices=SEND_METHODS,
        default=DEFAULT_SEND_METHOD,
        help="Controlled tmux transport method for generated bridge command files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Validation output directory; defaults under AI_analysis/06_probe_evidence and may use Step4 validation roots.",
    )
    parser.add_argument(
        "--user-confirmed-formal",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled ODCR tmux GPU bridge.")
    sub = parser.add_subparsers(dest="mode", required=True)

    for mode in (
        "discover",
        "validate-only",
        "unlock-pane",
        "marker-probe",
        "cuda-probe",
        "bge-smoke",
    ):
        _add_common_args(sub.add_parser(mode))

    real_data = sub.add_parser("real-data-probe")
    _add_common_args(real_data)
    real_data.add_argument("--probe-stage", choices=("b", "c"), default="b")

    step3_startup = sub.add_parser("step3-startup-validation")
    _add_common_args(step3_startup)
    step3_startup.add_argument("--task-id", type=int, default=2)
    step3_startup.add_argument(
        "--validation-slug",
        default="step3_tmux_gpu_bridge_startup_validation_closeout",
    )

    step3_perf = sub.add_parser("step3-performance-probe")
    _add_common_args(step3_perf)
    step3_perf.add_argument("--task-id", type=int, default=2)
    step3_perf.add_argument(
        "--validation-slug",
        default="step3_runtime_probe_truth_rebuild",
    )
    step3_perf.add_argument(
        "--probe-type",
        choices=STEP3_PERFORMANCE_PROBE_TYPES,
        default="timing-profile-window",
    )
    step3_perf.add_argument("--candidate-name", help="Validation-only batch ladder candidate name.")

    preprocess = sub.add_parser("preprocess-dryrun")
    _add_common_args(preprocess)
    preprocess.add_argument("--stage", choices=("b", "c"), required=True)

    micro = sub.add_parser("micro-benchmark")
    _add_common_args(micro)
    micro.add_argument("--kind", choices=("bge-single-batch",), default="bge-single-batch")
    micro.add_argument("--embed-batch-size", type=int, default=512)
    micro.add_argument("--read-chunk-rows", type=int, default=100_000)
    micro.add_argument("--group-shard-size", type=int, default=4_096)
    micro.add_argument("--workers", type=int, default=2)
    micro.add_argument("--bf16", action="store_true", default=True, dest="bf16_enabled")
    micro.add_argument("--no-bf16", action="store_false", dest="bf16_enabled")
    micro.add_argument("--tf32", action="store_true", default=True, dest="tf32_enabled")
    micro.add_argument("--no-tf32", action="store_false", dest="tf32_enabled")
    micro.add_argument("--grouped-text-cache", action="store_true", default=True, dest="grouped_text_cache_enabled")
    micro.add_argument("--no-grouped-text-cache", action="store_false", dest="grouped_text_cache_enabled")

    repo_command = sub.add_parser("repo-command")
    _add_common_args(repo_command)
    repo_command.add_argument("command", nargs=argparse.REMAINDER, help="Repo-local command after --.")

    repo_script = sub.add_parser("repo-script")
    _add_common_args(repo_script)
    repo_script.add_argument("--script", required=True, help="Repo-local script path.")
    repo_script.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script after --.")

    repo_module = sub.add_parser("repo-module")
    _add_common_args(repo_module)
    repo_module.add_argument("--module", required=True, help="Repo-local Python module, for example tools.odcr_step3_performance_probe.")
    repo_module.add_argument("module_args", nargs=argparse.REMAINDER, help="Arguments passed to the module after --.")

    command_file = sub.add_parser("command-file")
    _add_common_args(command_file)
    command_file.add_argument(
        "--file",
        required=True,
        help="Generated command file under AI_analysis/06_probe_evidence, AI_analysis/07_runtime_evidence, runs/step3_validation, runs/step4_preflight, or runs/step4_validation.",
    )
    command_file.add_argument("file_args", nargs=argparse.REMAINDER, help="Arguments passed to the command file after --.")

    long_run = sub.add_parser("long-run")
    _add_common_args(long_run)
    long_run.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Repo-local long-running command after --. Default timeout is detached/no hard cap.",
    )

    collect = sub.add_parser("collect")
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--timeout", default="auto", help=argparse.SUPPRESS)
    collect.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    collect.add_argument("--no-send", action="store_true", help=argparse.SUPPRESS)
    collect.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    collect.add_argument("--socket", help=argparse.SUPPRESS)
    collect.add_argument("--target", help=argparse.SUPPRESS)
    collect.add_argument("--send-method", choices=SEND_METHODS, default=DEFAULT_SEND_METHOD, help=argparse.SUPPRESS)
    collect.add_argument("--output-dir", help=argparse.SUPPRESS)
    collect.add_argument("--user-confirmed-formal", action="store_true", help=argparse.SUPPRESS)
    return parser


def options_from_args(args: argparse.Namespace) -> BridgeOptions:
    command_argv = tuple(
        item for item in (
            getattr(args, "command", None)
            or getattr(args, "script_args", None)
            or getattr(args, "module_args", None)
            or getattr(args, "file_args", None)
            or ()
        )
    )
    if command_argv and command_argv[0] == "--":
        command_argv = command_argv[1:]
    return BridgeOptions(
        mode=args.mode,
        stage=getattr(args, "stage", None),
        probe_stage=getattr(args, "probe_stage", "b"),
        benchmark_kind=getattr(args, "kind", None),
        embed_batch_size=int(getattr(args, "embed_batch_size", 512)),
        read_chunk_rows=int(getattr(args, "read_chunk_rows", 100_000)),
        group_shard_size=int(getattr(args, "group_shard_size", 4_096)),
        workers=int(getattr(args, "workers", 2)),
        bf16_enabled=bool(getattr(args, "bf16_enabled", True)),
        tf32_enabled=bool(getattr(args, "tf32_enabled", True)),
        grouped_text_cache_enabled=bool(getattr(args, "grouped_text_cache_enabled", True)),
        task_id=int(getattr(args, "task_id", 2)),
        smoke_candidate=getattr(args, "smoke_candidate", None),
        candidate_name=getattr(args, "candidate_name", None),
        worker_profile=getattr(args, "worker_profile", None),
        max_batches=int(getattr(args, "max_batches", 1)),
        max_steps=int(getattr(args, "max_steps", 1)),
        warmup_optimizer_steps=int(getattr(args, "warmup_optimizer_steps", 10)),
        measured_optimizer_steps=int(getattr(args, "measured_optimizer_steps", 50)),
        max_wall_seconds=int(getattr(args, "max_wall_seconds", 180)),
        max_epochs=int(getattr(args, "max_epochs", 2)),
        max_optimizer_steps=int(getattr(args, "max_optimizer_steps", 2000)),
        validate_every_steps=int(getattr(args, "validate_every_steps", 500)),
        validate_every_epoch=bool(getattr(args, "validate_every_epoch", True)),
        timeout=getattr(args, "timeout", "auto"),
        dry_run=bool(getattr(args, "dry_run", False)),
        no_send=bool(getattr(args, "no_send", False)),
        strict=bool(getattr(args, "strict", False)),
        socket=getattr(args, "socket", None),
        target=getattr(args, "target", None),
        run_id=getattr(args, "run_id", None),
        send_method=getattr(args, "send_method", DEFAULT_SEND_METHOD),
        validation_slug=getattr(args, "validation_slug", "step3_tmux_gpu_bridge_startup_validation_closeout"),
        performance_probe_type=getattr(args, "probe_type", "timing-profile-window"),
        command_argv=command_argv,
        script_path=getattr(args, "script", None),
        module_name=getattr(args, "module", None),
        command_file=getattr(args, "file", None),
        output_dir=getattr(args, "output_dir", None),
        user_confirmed_formal=bool(getattr(args, "user_confirmed_formal", False)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    bridge = TmuxGpuBridge()
    try:
        result = bridge.run_mode(options_from_args(args))
    except BridgeError as exc:
        payload = {
            "schema_version": BRIDGE_STATUS_SCHEMA,
            "success": False,
            "exit_code": 1,
            "stop_reason": exc.stop_reason,
            "error": str(exc),
            "details": exc.details,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
