"""Closed registry for ODCR runtime probes and bridge repo commands."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RUNTIME_STAGES = ("preprocess", "step3", "step4", "step5", "eval", "rerank", "pipeline")
STEP4_BRIDGE_MAX_BOUNDED_LIMIT = 32768


@dataclass(frozen=True)
class RuntimeCommandSpec:
    command_id: str
    stage: str
    read_only: bool
    can_write: bool
    allow_gpu: bool
    allow_formal_run: bool
    timeout_seconds: int
    output_policy: str
    ai_analysis_subdir: str
    argv_template: tuple[str, ...] = ()
    help_text: str = ""


@dataclass(frozen=True)
class StageDispatchAdmission:
    allowed: bool
    reason: str
    stop_reason: str = "forbidden_mode"
    command_id: str | None = None
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
    read_only: bool = False
    can_write: bool = False
    allow_gpu: bool = False
    allow_formal_run: bool = False
    timeout_seconds: int | None = None
    output_policy: str | None = None
    ai_analysis_subdir: str | None = None


REPO_COMMAND_REGISTRY: Mapping[str, RuntimeCommandSpec] = {
    "runtime_bridge_validate_only": RuntimeCommandSpec(
        command_id="runtime_bridge_validate_only",
        stage="pipeline",
        read_only=True,
        can_write=True,
        allow_gpu=True,
        allow_formal_run=False,
        timeout_seconds=60,
        output_policy="AI_analysis raw/summary/report only",
        ai_analysis_subdir="01_raw_logs",
        help_text="Validate current tmux GPU pane without launching training.",
    ),
    "runtime_bridge_marker_probe": RuntimeCommandSpec(
        command_id="runtime_bridge_marker_probe",
        stage="pipeline",
        read_only=True,
        can_write=True,
        allow_gpu=True,
        allow_formal_run=False,
        timeout_seconds=20,
        output_policy="AI_analysis marker transport evidence only",
        ai_analysis_subdir="01_raw_logs",
        help_text="Send a short bridge marker script.",
    ),
    "runtime_bridge_cuda_probe": RuntimeCommandSpec(
        command_id="runtime_bridge_cuda_probe",
        stage="pipeline",
        read_only=True,
        can_write=True,
        allow_gpu=True,
        allow_formal_run=False,
        timeout_seconds=45,
        output_policy="AI_analysis CUDA transport evidence only",
        ai_analysis_subdir="01_raw_logs",
        help_text="Probe torch CUDA visibility in the current tmux pane.",
    ),
    "step3_bounded_probe": RuntimeCommandSpec(
        command_id="step3_bounded_probe",
        stage="step3",
        read_only=False,
        can_write=True,
        allow_gpu=True,
        allow_formal_run=False,
        timeout_seconds=180,
        output_policy="AI_analysis and runs/step3_validation only",
        ai_analysis_subdir="06_probe_evidence",
        help_text="Bounded Step3 validation probe, never formal training.",
    ),
    "step4_bounded_preflight": RuntimeCommandSpec(
        command_id="step4_bounded_preflight",
        stage="step4",
        read_only=False,
        can_write=True,
        allow_gpu=True,
        allow_formal_run=False,
        timeout_seconds=180,
        output_policy="AI_analysis, runs/step4_preflight, or runs/step4_validation only",
        ai_analysis_subdir="06_probe_evidence",
        help_text="Bounded Step4 RCR preflight validation.",
    ),
    "step4_prepare_cache_validation": RuntimeCommandSpec(
        command_id="step4_prepare_cache_validation",
        stage="step4",
        read_only=False,
        can_write=True,
        allow_gpu=True,
        allow_formal_run=False,
        timeout_seconds=180,
        output_policy="AI_analysis, runs/step4_preflight, or runs/step4_validation only",
        ai_analysis_subdir="06_probe_evidence",
        help_text="Bounded Step4 cache validation.",
    ),
}


def list_runtime_commands() -> tuple[RuntimeCommandSpec, ...]:
    return tuple(REPO_COMMAND_REGISTRY.values())


def get_runtime_command(command_id: str) -> RuntimeCommandSpec:
    try:
        return REPO_COMMAND_REGISTRY[command_id]
    except KeyError as exc:
        raise KeyError(f"unknown runtime command_id: {command_id}") from exc


def _deny(reason: str, *, stop_reason: str = "forbidden_mode", **kwargs: Any) -> StageDispatchAdmission:
    return StageDispatchAdmission(False, reason, stop_reason=stop_reason, **kwargs)


def _allow(spec: RuntimeCommandSpec, reason: str, **kwargs: Any) -> StageDispatchAdmission:
    return StageDispatchAdmission(
        True,
        reason,
        stop_reason="allowed",
        command_id=spec.command_id,
        stage=spec.stage,
        read_only=spec.read_only,
        can_write=spec.can_write,
        allow_gpu=spec.allow_gpu,
        allow_formal_run=spec.allow_formal_run,
        timeout_seconds=spec.timeout_seconds,
        output_policy=spec.output_policy,
        ai_analysis_subdir=spec.ai_analysis_subdir,
        **kwargs,
    )


def stage_dispatch_admission_to_dict(admission: StageDispatchAdmission) -> dict[str, Any]:
    return dataclasses.asdict(admission)


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


def _int_flag(args: Sequence[str], flag: str) -> int | None:
    value = _flag_value(args, flag)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _bounded_limit(args: Sequence[str]) -> tuple[str | None, int | None, str | None]:
    for flag in ("--max-samples", "--max-batches", "--max-rows"):
        raw = _flag_value(args, flag)
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


def _odcr_entry(parts: Sequence[str]) -> int | None:
    if not parts:
        return None
    if parts[0] == "./odcr":
        return 0
    if Path(parts[0]).name.startswith("python"):
        if len(parts) >= 2 and parts[1] == "code/odcr.py":
            return 1
        if len(parts) >= 3 and parts[1] == "-m" and parts[2] == "code.odcr":
            return 2
    return None


def _contains_background_token(parts: Sequence[str]) -> bool:
    return any(part in {"&", "&&"} or part.endswith("&") for part in parts)


def _targets_formal_namespace(values: Sequence[str]) -> bool:
    joined = " ".join(str(value).replace("\\", "/").lower() for value in values)
    formal_terms = (
        "runs/step3/task",
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


def _classify_step4(stage_args: Sequence[str], *, output_dir: Path | None = None) -> StageDispatchAdmission:
    args = tuple(stage_args)
    operation_flags = [flag for flag in ("--preflight", "--prepare-cache") if flag in args]
    task = _int_flag(args, "--task")
    if not operation_flags:
        return _deny("formal Step4 command rejected: missing --preflight/--prepare-cache", command_kind="odcr", stage="step4", task=task)
    if len(operation_flags) > 1:
        return _deny("Step4 command rejected: choose only one of --preflight/--prepare-cache", command_kind="odcr", stage="step4", task=task)
    operation = operation_flags[0].lstrip("-").replace("-", "_")
    for flag in ("--write-latest", "--formal", "--full", "--run-full"):
        if flag in args:
            return _deny("formal Step4 command rejected", command_kind="odcr", stage="step4", operation=operation, task=task)
    if _flag_value(args, "--mode") == "full":
        return _deny("formal Step4 command rejected: --mode full", command_kind="odcr", stage="step4", operation=operation, task=task)
    namespace = _flag_value(args, "--validation-namespace")
    if namespace is None:
        return _deny("missing --validation-namespace", command_kind="odcr", stage="step4", operation=operation, task=task)
    namespace_problem = validate_step4_validation_namespace(namespace)
    if namespace_problem:
        return _deny(namespace_problem, command_kind="odcr", stage="step4", operation=operation, task=task, validation_namespace=namespace)
    limit_name, limit_value, limit_error = _bounded_limit(args)
    if limit_error:
        return _deny(limit_error, command_kind="odcr", stage="step4", operation=operation, task=task, validation_namespace=namespace)
    output_values = tuple(
        value
        for flag in ("--output", "--output-dir", "--run-dir")
        for value in [_flag_value(args, flag)]
        if value
    )
    if _targets_formal_namespace(output_values):
        return _deny(
            "formal namespace target rejected",
            stop_reason="formal_namespace_blocked",
            command_kind="odcr",
            stage="step4",
            operation=operation,
            task=task,
            validation_namespace=namespace,
        )
    if output_dir is not None and _targets_formal_namespace((str(output_dir),)):
        return _deny(
            "formal namespace target rejected",
            stop_reason="formal_namespace_blocked",
            command_kind="odcr",
            stage="step4",
            operation=operation,
            task=task,
            validation_namespace=namespace,
        )
    spec = get_runtime_command("step4_bounded_preflight" if operation == "preflight" else "step4_prepare_cache_validation")
    task_id = int(task or 2)
    evidence_roots = (
        f"runs/step4_preflight/task{task_id}/{namespace}",
        f"runs/step4_validation/task{task_id}/{namespace}",
    )
    return _allow(
        spec,
        "registered bounded Step4 validation command allowed",
        command_kind="odcr",
        operation=operation,
        task=task_id,
        validation_namespace=namespace,
        bounded_limit_name=limit_name,
        bounded_limit_value=limit_value,
        output_roots=tuple(str(value) for value in output_values),
        evidence_roots=evidence_roots,
        runtime_evidence_required=operation == "preflight",
    )


def classify_repo_command(
    argv: Sequence[str],
    *,
    output_dir: Path | None = None,
    user_confirmed_formal: bool = False,
) -> StageDispatchAdmission:
    del user_confirmed_formal
    parts = tuple(str(item) for item in argv if str(item) != "")
    if not parts:
        return _deny("missing command", stop_reason="target_invalid")
    command_name = Path(parts[0]).name
    if command_name in {"nohup", "disown"} or any(Path(part).name in {"nohup", "disown"} for part in parts):
        return _deny("registered repo-command rejected: background command")
    if any(Path(part).name in {"srun", "sbatch", "scancel", "odcr-enter-gpu"} for part in parts):
        return _deny("registered repo-command rejected: allocation command")
    if _contains_background_token(parts):
        return _deny("registered repo-command rejected: background token")
    if command_name in {"rm", "mv", "cp", "kill", "pkill"}:
        return _deny("registered repo-command rejected: destructive command")

    entry = _odcr_entry(parts)
    if entry is not None:
        stage_index = entry + 1
        if len(parts) <= stage_index:
            return _deny("ODCR help is not a registered bridge repo-command")
        stage = parts[stage_index]
        stage_args = parts[stage_index + 1 :]
        if stage == "step4":
            return _classify_step4(stage_args, output_dir=output_dir)
        if stage in {"preprocess", "step3", "step5", "eval", "rerank", "pipeline"}:
            return _deny(f"{stage} command is not registered for bridge repo-command dispatch", command_kind="odcr", stage=stage)
        return _deny(f"unknown ODCR stage for bridge repo-command dispatch: {stage}", command_kind="odcr")

    if (
        Path(parts[0]).name.startswith("python")
        and len(parts) >= 2
        and parts[1] == "code/tools/odcr_step3_performance_probe.py"
        and "--help" in parts
    ):
        spec = get_runtime_command("step3_bounded_probe")
        return _allow(spec, "registered Step3 performance probe help command allowed", command_kind="repo_python", operation="help")

    return _deny("repo-command must match a registered ODCR runtime command_id")


def runtime_probe_bridge_args(
    *,
    stage: str,
    task: int,
    profile: str | None,
    bounded: bool,
    probe_kind: str | None = None,
    dry_run: bool = False,
    no_send: bool = False,
    run_id: str | None = None,
) -> tuple[str, ...]:
    if stage not in RUNTIME_STAGES:
        raise ValueError(f"unknown runtime stage: {stage}")
    if stage != "step3":
        raise ValueError(f"runtime probe for {stage} is registered as help/fail-fast placeholder only")
    if not bounded:
        raise ValueError("step3 runtime probe requires --bounded")
    probe = str(probe_kind or "epoch-boundary-memory").strip()
    allowed_probe_kinds = {
        "timing-profile-window",
        "prefetch-ab",
        "grad-monitor-window",
        "memory-phase-window",
        "epoch-boundary-memory",
        "epoch2-numerical-stability",
        "ddp-gather-sync-window",
        "quality-checkpoint-window",
        "batch-ladder-window",
        "sidecar-gradient-firewall",
    }
    if probe not in allowed_probe_kinds:
        raise ValueError(f"unknown Step3 runtime probe kind: {probe!r}")
    args = [
        "step3-performance-probe",
        "--task-id",
        str(int(task)),
        "--probe-type",
        probe,
    ]
    if probe == "epoch2-numerical-stability":
        args.extend(["--warmup-steps", "500", "--measured-steps", "500", "--max-seconds", "895"])
    if run_id:
        args.extend(["--run-id", run_id])
    if profile:
        args.extend(["--candidate-name", str(profile)])
    if dry_run:
        args.append("--dry-run")
    if no_send:
        args.append("--no-send")
    return tuple(args)
