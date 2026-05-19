"""Allowlisted tmux GPU bridge for ODCR runtime validation."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from odcr_core.aux.evidence.ai_analysis_writer import get_writer
from odcr_core.aux.runtime.bounded_probe import write_probe_report
from odcr_core.aux.runtime.command_registry import LEGACY_FORBIDDEN_COMMANDS, RuntimeCommandError, require_command
from odcr_core.aux.runtime.gpu_handshake import collect_handshake, write_handshake
from odcr_core.aux.runtime.gpu_pane_handoff import (
    HandoffError,
    OLD_GPU_PANE_STATE_ROLE,
    STALE_HANDOFF_STOP_REASON,
    TARGET_SOURCE_HANDOFF,
    current_handoff_path,
    legacy_gpu_pane_state_path,
    load_current_handoff,
    selection_from_handoff_payload,
)
from odcr_core.aux.runtime.pane_discovery import (
    CommandResult,
    DiscoveryResult,
    PaneCandidate,
    SubprocessRunner,
    candidate_socket_paths,
    classify_command,
    discover_panes,
    select_unique_pane,
)
from odcr_core.aux.runtime.runtime_report import write_runtime_report
from odcr_core.aux.runtime.stage_dispatch import probe_command_name
from odcr_core.evidence_level import (
    E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
    E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
AI_ANALYSIS = REPO_ROOT / "AI_analysis"
RAW_LOG_DIR = AI_ANALYSIS / "01_raw_logs"
REPORT_DIR = AI_ANALYSIS / "05_final_reports"
RUNTIME_STATUS_NAME = "aux_runtime_gpu_handshake.status.json"
RUNTIME_LOG_NAME = "aux_runtime_gpu_handshake.log"
RUNTIME_REPORT_NAME = "aux_runtime_gpu_validation_report.md"
PROBE_STATUS_NAME = "aux_runtime_step5_e4_probe.status.json"
PROBE_E5_STATUS_NAME = "aux_runtime_step5_e5_lifecycle_probe.status.json"
PANE_MODE_RECOVERY_STATUS_NAME = "aux_runtime_pane_mode_recovery.json"
GLOBAL_INVENTORY_NAME = "gpu_tmux_global_pane_inventory.json"
CURRENT_HANDOFF_PATH = current_handoff_path(REPO_ROOT)
STATE_HINT_PATH = legacy_gpu_pane_state_path(REPO_ROOT)
PANE_MODE_EXIT_KEYS = ("Escape", "q")
ACTIVE_COMPUTE_PANE_COMMANDS = ("torchrun", "python", "python3", "accelerate", "deepspeed")
RUNNABLE_PANE_COMMAND_CLASSES = ("shell", "srun")
TARGET_SOURCE_CLI = "cli_explicit"
TARGET_SOURCE_GLOBAL = "global_discovery"
TARGET_SOURCE_STATE = "state_hint"
TARGET_SOURCE_LIVE_DISCOVERY = "live_discovery_cuda_probe"
STEP5_FRESH_HANDOFF_REQUIRED_STAGES = {"step5", "step5A", "step5B"}
NO_LIVE_CUDA_AFTER_STALE = "no_live_cuda_pane_after_stale_handoff"
AMBIGUOUS_LIVE_CUDA_PANES = "ambiguous_live_cuda_panes"


class BridgeError(RuntimeError):
    def __init__(self, message: str, *, stop_reason: str = "runtime_bridge_error", details: Any = None) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.details = details


@dataclass(frozen=True)
class BridgeOptions:
    mode: str
    stage: str | None = None
    task_id: int = 2
    socket: str | None = None
    target: str | None = None
    run_id: str | None = None
    dry_run: bool = False
    no_send: bool = False
    timeout: int | None = None
    from_step4: str | None = None
    global_discovery: bool = False
    all_sockets: bool = False
    all_panes: bool = False
    json_output: bool = False


@dataclass(frozen=True)
class GeneratedPaths:
    status: Path
    log: Path
    report: Path

    def to_dict(self) -> dict[str, str]:
        return {"status": str(self.status), "log": str(self.log), "report": str(self.report)}


def _safe_suffix(value: str) -> str:
    out = []
    for char in str(value):
        out.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(out).strip("_")[:120] or "default"


def generated_paths(suffix: str | None = None) -> GeneratedPaths:
    extra = f".{_safe_suffix(suffix)}" if suffix else ""
    return GeneratedPaths(
        status=RAW_LOG_DIR / f"aux_runtime_gpu_handshake{extra}.status.json",
        log=RAW_LOG_DIR / f"aux_runtime_gpu_handshake{extra}.log",
        report=REPORT_DIR / f"aux_runtime_gpu_validation_report{extra}.md",
    )


def _json_default(value: Any) -> str:
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_probe_evidence_level(value: str | None) -> str:
    raw = str(value or "").strip()
    if raw in {"E5", E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE}:
        return E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE
    return E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION


def _current_handoff_selection() -> dict[str, Any]:
    try:
        payload = load_current_handoff(REPO_ROOT)
    except HandoffError as exc:
        return {
            "exists": True,
            "valid": False,
            "path": str(CURRENT_HANDOFF_PATH),
            "error": str(exc),
            "details": exc.details,
        }
    if payload is None:
        return {
            "exists": False,
            "valid": False,
            "path": str(CURRENT_HANDOFF_PATH),
        }
    try:
        selection = selection_from_handoff_payload(payload)
    except HandoffError as exc:
        return {
            "exists": True,
            "valid": False,
            "path": str(CURRENT_HANDOFF_PATH),
            "error": str(exc),
            "details": exc.details,
            "payload": payload,
        }
    return {
        "exists": True,
        "valid": True,
        "path": str(CURRENT_HANDOFF_PATH),
        "socket": selection["socket"],
        "target": selection["target"],
        "payload": payload,
    }


def _cli_conflict_with_handoff(options: BridgeOptions, handoff: Mapping[str, Any]) -> dict[str, Any] | None:
    if not (options.socket or options.target):
        return None
    if not bool(handoff.get("exists")):
        return None
    payload = handoff.get("payload") if isinstance(handoff.get("payload"), Mapping) else {}
    handoff_socket = str(handoff.get("socket") or "").strip()
    handoff_target = str(handoff.get("target") or "").strip()
    admin_tmux = payload.get("admin_tmux") if isinstance(payload.get("admin_tmux"), Mapping) else {}
    tmux_metadata = payload.get("tmux_metadata") if isinstance(payload.get("tmux_metadata"), Mapping) else {}
    acceptable_targets = {
        handoff_target,
        str(admin_tmux.get("target") or "").strip(),
        str(admin_tmux.get("pane_id") or "").strip(),
        str(tmux_metadata.get("target") or "").strip(),
    }
    conflicts: dict[str, Any] = {}
    if options.socket and handoff_socket and str(options.socket) != handoff_socket:
        conflicts["socket"] = {"cli": str(options.socket), "handoff": handoff_socket}
    if options.target and handoff_target and str(options.target) not in acceptable_targets:
        conflicts["target"] = {"cli": str(options.target), "handoff": sorted(v for v in acceptable_targets if v)}
    if not conflicts:
        return None
    return {
        "schema_version": "odcr_gpu_pane_handoff_cli_conflict/1",
        "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
        "cli_overrides_handoff": True,
        "target_source": TARGET_SOURCE_CLI,
        "conflicts": conflicts,
    }


def _write_failure_report(
    kind: str,
    message: str,
    *,
    stage: str | None = None,
    task: int | None = None,
    stop_reason: str = "runtime_bridge_error",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "odcr_runtime_gpu_handshake/1",
        "kind": kind,
        "stage": stage,
        "task": task,
        "success": False,
        "stop_reason": stop_reason,
        "error": message,
    }
    if details:
        payload["details"] = dict(details)
        if "stale_state_used" in details:
            payload["stale_state_used"] = bool(details.get("stale_state_used"))
        if "stale_handoff_detected" in details:
            payload["stale_handoff_detected"] = bool(details.get("stale_handoff_detected"))
    writer = get_writer(REPO_ROOT)
    writer.raw_log(
        RUNTIME_LOG_NAME,
        message,
        source="tmux_gpu_bridge",
        stage=stage,
        task=task,
        validation_result=payload,
        errors=[message],
    )
    writer.final_report(
        RUNTIME_REPORT_NAME,
        "# ODCR Runtime GPU Validation\n\n"
        f"- kind: {kind}\n"
        f"- stage: {stage or 'n/a'}\n"
        f"- task: {task or 'n/a'}\n"
        "- result: FAIL\n"
        f"- error: {message}\n",
        source="tmux_gpu_bridge",
        stage=stage,
        task=task,
        validation_result=payload,
        errors=[message],
    )
    writer.runtime_diagnostic(
        RUNTIME_STATUS_NAME,
        payload,
        source="tmux_gpu_bridge",
        stage=stage,
        task=task,
        validation_result=payload,
        errors=[message],
    )
    return payload


def _forbidden_legacy_message(mode: str) -> str:
    return (
        f"{mode} is retired and fail-fast. Use ./odcr runtime bridge "
        "discover|validate-only|marker-probe|cuda-probe or ./odcr runtime probe --stage ... --bounded."
    )


def reject_legacy_mode(mode: str) -> None:
    if mode in LEGACY_FORBIDDEN_COMMANDS:
        raise BridgeError(_forbidden_legacy_message(mode), stop_reason="forbidden_legacy_mode")


def _command_for_child(kind: str, paths: GeneratedPaths, *, stage: str | None, task: int, require_cuda: bool) -> tuple[str, ...]:
    argv = (
        sys.executable,
        str(REPO_ROOT / "code" / "odcr.py"),
        "runtime",
        "bridge",
        "_handshake-child",
        "--kind",
        kind,
        "--status-path",
        str(paths.status),
        "--log-path",
        str(paths.log),
        "--report-path",
        str(paths.report),
        "--repo-root",
        str(REPO_ROOT),
    )
    if stage:
        argv += ("--stage", stage)
    if task:
        argv += ("--task", str(task))
    if require_cuda:
        argv += ("--require-cuda",)
    return argv


def _command_for_probe_child(
    *,
    stage: str,
    task: int,
    status_path: Path,
    config_path: str,
    sets: Sequence[str],
    candidate_id: str | None,
    timeout: int | None,
    from_step4: str | None,
    evidence_level: str,
) -> tuple[str, ...]:
    argv = (
        sys.executable,
        str(REPO_ROOT / "code" / "odcr.py"),
        "runtime",
        "probe",
        "--stage",
        str(stage),
        "--task",
        str(int(task)),
        "--bounded",
        "--probe-child",
        "--status-path",
        str(status_path),
        "--config",
        str(config_path),
    )
    if candidate_id is not None and str(candidate_id).strip():
        argv += ("--candidate-id", str(candidate_id).strip())
    for item in sets:
        argv += ("--set", str(item))
    if timeout is not None:
        argv += ("--timeout", str(int(timeout)))
    if from_step4 is not None and str(from_step4).strip():
        argv += ("--from-step4-run", str(from_step4).strip())
    argv += ("--evidence-level", _normalize_probe_evidence_level(evidence_level))
    return argv


def _pane_candidate_from_invalid(entry: Mapping[str, Any]) -> PaneCandidate:
    return PaneCandidate(
        socket=str(entry.get("socket") or ""),
        session=str(entry.get("session") or ""),
        target=str(entry.get("target") or ""),
        window_index=str(entry.get("window_index") or ""),
        window_name=str(entry.get("window_name") or ""),
        pane_index=str(entry.get("pane_index") or ""),
        pane_id=str(entry.get("pane_id") or entry.get("target") or ""),
        pane_pid=int(entry.get("pane_pid") or 0),
        pane_command=str(entry.get("pane_command") or ""),
        cwd=str(entry.get("cwd") or ""),
        active=bool(entry.get("active")),
        dead=bool(entry.get("dead")),
        in_mode=bool(entry.get("in_mode")),
        pane_title=str(entry.get("pane_title") or ""),
        pane_tty=str(entry.get("pane_tty") or ""),
        pane_start_command=str(entry.get("pane_start_command") or ""),
        cwd_match_repo=bool(entry.get("cwd_match_repo")),
        command_class=str(entry.get("command_class") or classify_command(str(entry.get("pane_command") or ""))),
        last_visible_line_hash=entry.get("last_visible_line_hash") if entry.get("last_visible_line_hash") else None,
    )


def _pane_candidate_from_payload(entry: Mapping[str, Any]) -> PaneCandidate:
    return _pane_candidate_from_invalid(entry)


def _matches_requested_target(entry: Mapping[str, Any], requested: str | None) -> bool:
    if not requested:
        return True
    raw = str(requested)
    return raw == str(entry.get("target") or "") or raw == str(entry.get("pane_id") or "")


def _find_candidate(discovery: DiscoveryResult, candidate: PaneCandidate) -> PaneCandidate | None:
    for item in discovery.candidates:
        if item.socket == candidate.socket and (item.pane_id == candidate.pane_id or item.target == candidate.target):
            return item
    return None


def _find_in_mode(discovery: DiscoveryResult, candidate: PaneCandidate) -> Mapping[str, Any] | None:
    for item in discovery.invalid:
        if item.get("reason") != "pane_in_mode":
            continue
        if str(item.get("socket") or "") != candidate.socket:
            continue
        if str(item.get("pane_id") or "") == candidate.pane_id or str(item.get("target") or "") == candidate.target:
            return item
    return None


def _write_json_artifact(name: str, payload: Mapping[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _candidate_key(candidate: PaneCandidate) -> str:
    return f"{candidate.socket}|{candidate.target}|{candidate.pane_id}"


def _non_send_filter_reasons(candidate: PaneCandidate) -> list[str]:
    reasons: list[str] = []
    if candidate.dead:
        reasons.append("dead")
    if candidate.in_mode:
        reasons.append("pane_in_mode")
    if not str(candidate.cwd or "").strip():
        reasons.append("no_cwd")
    command_class = str(candidate.command_class or classify_command(candidate.pane_command))
    if command_class == "active_compute_app":
        reasons.append("active_compute_app_busy")
    elif command_class not in RUNNABLE_PANE_COMMAND_CLASSES:
        reasons.append("command_not_shell_or_srun")
    return reasons


def _is_admin_hostname(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or text.startswith("admin")


def _current_tmux_selection() -> dict[str, str]:
    raw = str(os.environ.get("TMUX") or "").strip()
    pane = str(os.environ.get("TMUX_PANE") or "").strip()
    socket = raw.split(",", 1)[0].strip() if raw else ""
    return {"socket": socket, "pane": pane}


def _handoff_selection_candidates(handoff: Mapping[str, Any] | None) -> dict[str, set[str]]:
    payload = handoff.get("payload") if isinstance(handoff, Mapping) and isinstance(handoff.get("payload"), Mapping) else {}
    admin_tmux = payload.get("admin_tmux") if isinstance(payload.get("admin_tmux"), Mapping) else {}
    selection = payload.get("selection_key") if isinstance(payload.get("selection_key"), Mapping) else {}
    socket = str((handoff or {}).get("socket") or admin_tmux.get("socket") or selection.get("socket") or "").strip()
    targets = {
        str((handoff or {}).get("target") or "").strip(),
        str(admin_tmux.get("pane") or "").strip(),
        str(admin_tmux.get("pane_id") or "").strip(),
        str(admin_tmux.get("target") or "").strip(),
        str(selection.get("pane") or "").strip(),
    }
    return {"sockets": {socket} if socket else set(), "targets": {item for item in targets if item}}


def _candidate_matches_selection(candidate: PaneCandidate, selection: Mapping[str, set[str]]) -> bool:
    sockets = selection.get("sockets") or set()
    targets = selection.get("targets") or set()
    if sockets and candidate.socket not in sockets:
        return False
    if not targets:
        return bool(sockets)
    return candidate.pane_id in targets or candidate.target in targets


def _candidate_matches_current_tmux(candidate: PaneCandidate) -> bool:
    current = _current_tmux_selection()
    if not current["socket"] or not current["pane"]:
        return False
    return candidate.socket == current["socket"] and candidate.pane_id == current["pane"]


def _slurm_job_id(status: Mapping[str, Any]) -> int | None:
    raw = str(status.get("SLURM_JOB_ID") or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_compute_app_rows(snapshot: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(snapshot, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for raw in str(snapshot.get("stdout") or "").splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid: int | None = int(parts[0])
        except ValueError:
            pid = None
        try:
            used_memory_mib: float | None = float(parts[2])
        except ValueError:
            used_memory_mib = None
        rows.append(
            {
                "pid": pid,
                "process_name": parts[1],
                "used_memory_mib": used_memory_mib,
                "raw": raw,
            }
        )
    return rows


class TmuxGpuBridge:
    def __init__(
        self,
        *,
        runner: SubprocessRunner | None = None,
        socket_exists: Any | None = None,
        clock: Any | None = None,
        sleep: Any | None = None,
    ) -> None:
        self.runner = runner or SubprocessRunner()
        self.socket_exists = socket_exists
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep

    def discover(self, options: BridgeOptions) -> dict[str, Any]:
        require_command("bridge.discover")
        socket_paths = (Path(options.socket),) if options.socket else None
        global_mode = bool(options.global_discovery or options.all_sockets or options.all_panes)
        discovery = discover_panes(
            runner=self.runner,
            socket_paths=socket_paths,
            socket_exists=self.socket_exists,
            all_sockets=bool(options.all_sockets or options.global_discovery),
            include_filtered=bool(options.all_panes or options.global_discovery),
            capture_hash=global_mode,
            repo_root=REPO_ROOT,
        )
        payload = discovery.to_dict()
        payload["global_tmux_discovery_enabled"] = global_mode
        payload["all_sockets"] = bool(options.all_sockets or options.global_discovery)
        payload["all_panes"] = bool(options.all_panes or options.global_discovery)
        payload["stale_state_used"] = False
        payload["target_source"] = self._target_source_hint(options)
        payload["state_hint_path"] = str(STATE_HINT_PATH)
        payload["old_gpu_pane_state_role"] = OLD_GPU_PANE_STATE_ROLE
        payload["current_gpu_pane_json_path"] = str(CURRENT_HANDOFF_PATH)
        payload["success"] = bool(discovery.panes) if global_mode else len(discovery.candidates) == 1
        if global_mode:
            _write_json_artifact(GLOBAL_INVENTORY_NAME, payload)
        get_writer(REPO_ROOT).runtime_diagnostic(
            "aux_runtime_discovery.json",
            payload,
            source="tmux_gpu_bridge",
            stage="runtime",
            validation_result=payload["success"],
        )
        return payload

    def _target_source_hint(self, options: BridgeOptions) -> str:
        if options.socket or options.target:
            return TARGET_SOURCE_CLI
        handoff = _current_handoff_selection()
        if bool(handoff.get("valid")) or bool(handoff.get("exists")):
            return TARGET_SOURCE_HANDOFF
        return TARGET_SOURCE_GLOBAL if (options.global_discovery or options.all_sockets or options.all_panes) else "missing_current_gpu_pane_handoff"

    def _priority_target_specs(self, options: BridgeOptions) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []

        def add(
            source: str,
            socket: str | None,
            target: str | None,
            *,
            state_hint: Mapping[str, Any] | None = None,
            handoff: Mapping[str, Any] | None = None,
            error: str | None = None,
            active: bool = True,
        ) -> None:
            if not socket and not target and source != TARGET_SOURCE_STATE and not error:
                return
            specs.append(
                {
                    "source": source,
                    "socket": socket,
                    "target": target,
                    "state_hint": dict(state_hint) if isinstance(state_hint, Mapping) else None,
                    "handoff": dict(handoff) if isinstance(handoff, Mapping) else None,
                    "error": error,
                    "active": bool(active),
                }
            )

        handoff = _current_handoff_selection()
        cli_explicit = bool(options.socket or options.target)
        if bool(handoff.get("valid")):
            add(TARGET_SOURCE_HANDOFF, str(handoff.get("socket") or ""), str(handoff.get("target") or ""), handoff=handoff)
        elif bool(handoff.get("exists")):
            add(TARGET_SOURCE_HANDOFF, None, None, handoff=handoff, error=str(handoff.get("error") or "invalid_current_gpu_pane_handoff"))
        if cli_explicit:
            add(
                TARGET_SOURCE_CLI,
                options.socket,
                options.target,
                handoff=handoff,
                error=None,
            )
        add(TARGET_SOURCE_STATE, None, None, state_hint=None, active=False)
        return specs

    def _select_target_for_spec(self, spec: Mapping[str, Any]) -> tuple[PaneCandidate, DiscoveryResult, str]:
        target, discovery = select_unique_pane(
            socket=spec.get("socket") if spec.get("socket") else None,
            target=spec.get("target") if spec.get("target") else None,
            runner=self.runner,
            socket_exists=self.socket_exists,
        )
        return target, discovery, str(spec.get("source") or TARGET_SOURCE_HANDOFF)

    def resolve_target(self, options: BridgeOptions) -> tuple[PaneCandidate, DiscoveryResult, str]:
        attempts: list[dict[str, Any]] = []
        specs = self._priority_target_specs(options)
        cli_available = any(str(item.get("source") or "") == TARGET_SOURCE_CLI for item in specs)
        for spec in specs:
            source = str(spec.get("source") or "")
            if source == TARGET_SOURCE_HANDOFF and spec.get("error"):
                attempts.append(
                    {
                        "target_source": source,
                        "socket": spec.get("socket"),
                        "target": spec.get("target"),
                        "warning_reason": STALE_HANDOFF_STOP_REASON,
                        "error": spec.get("error"),
                        "stale_handoff_stops_discovery": False,
                    }
                )
                if cli_available:
                    continue
                raise BridgeError(
                    "AI_analysis/runtime/current_gpu_pane.json is present but invalid or stale. "
                    "Please run odcr-enter-gpu <JOBID> inside the GPU tmux pane to refresh it.",
                    stop_reason=STALE_HANDOFF_STOP_REASON,
                    details={
                        "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                        "target_priority_attempts": attempts,
                        "handoff": spec.get("handoff"),
                        "stale_state_used": False,
                        "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
                    },
                )
            if source == TARGET_SOURCE_STATE:
                attempts.append(
                    {
                        "target_source": source,
                        "socket": spec.get("socket"),
                        "target": spec.get("target"),
                        "role": OLD_GPU_PANE_STATE_ROLE,
                        "active_target_selection": False,
                    }
                )
                continue
            try:
                return self._select_target_for_spec(spec)
            except RuntimeError as exc:
                attempts.append(
                    {
                        "target_source": source,
                        "socket": spec.get("socket"),
                        "target": spec.get("target"),
                        "error": str(exc),
                    }
                )
                if source == TARGET_SOURCE_HANDOFF:
                    raise BridgeError(
                        "AI_analysis/runtime/current_gpu_pane.json did not resolve to exactly one runnable tmux pane. "
                        "Please rerun odcr-enter-gpu <JOBID> inside the GPU tmux pane to refresh it.",
                        stop_reason=STALE_HANDOFF_STOP_REASON,
                        details={
                            "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                            "target_priority_attempts": attempts,
                            "handoff": spec.get("handoff"),
                            "stale_state_used": False,
                            "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
                        },
                    ) from exc
                if source == TARGET_SOURCE_CLI:
                    raise BridgeError(
                        "Explicit CLI tmux target did not resolve to exactly one runnable tmux pane. "
                        "Check the socket/target pair from the current GPU pane handoff.",
                        stop_reason="cli_target_discovery_failed",
                        details={
                            "target_priority_attempts": attempts,
                            "stale_state_used": False,
                            "state_hint_path": str(STATE_HINT_PATH),
                            "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
                            "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                        },
                    ) from exc
        raise BridgeError(
            "No active GPU pane handoff is available. Use current_gpu_pane.json handoff v2 or pass an explicit CLI socket/target.",
            stop_reason="missing_current_gpu_pane_handoff",
            details={
                "target_priority_attempts": attempts,
                "stale_state_used": False,
                "state_hint_path": str(STATE_HINT_PATH),
                "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
                "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
            },
        )

    def _compute_app_snapshot(self) -> dict[str, Any]:
        args = (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
        try:
            result = self.runner.run(args, timeout=10)
        except Exception as exc:
            return {
                "argv": list(args),
                "returncode": None,
                "stdout": "",
                "stderr": repr(exc),
                "rows": [],
                "active_compute_apps": [],
                "active": False,
                "query_failed": True,
                "query_failure_is_not_cuda_blocker": True,
            }
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {
            "argv": list(args),
            "returncode": int(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "rows": rows,
            "active_compute_apps": rows if result.returncode == 0 else [],
            "active": bool(rows) if result.returncode == 0 else False,
            "query_failed": result.returncode != 0,
            "query_failure_is_not_cuda_blocker": result.returncode != 0,
        }

    def _compute_app_guard_from_status(
        self,
        *,
        status: Mapping[str, Any],
        candidate: PaneCandidate,
        target_source: str,
    ) -> dict[str, Any]:
        snapshot = status.get("nvidia-smi-compute-apps") if isinstance(status.get("nvidia-smi-compute-apps"), Mapping) else {}
        rows = _parse_compute_app_rows(snapshot)
        safe = snapshot.get("returncode") == 0 and not rows
        guard = {
            "schema_version": "odcr_gpu_bridge_compute_app_guard/1",
            "generated_at": _utc_now(),
            "target_source": target_source,
            "socket": candidate.socket,
            "pane": candidate.pane_id,
            "target": candidate.target,
            "hostname": status.get("hostname"),
            "cuda_visible_devices": status.get("CUDA_VISIBLE_DEVICES"),
            "device_count": status.get("torch.cuda.device_count"),
            "query": dict(snapshot),
            "rows": rows,
            "pass": bool(safe),
            "blocked": not bool(safe),
            "status": "pass" if safe else "blocked_unknown_compute_apps",
            "unknown_compute_apps": rows if rows else [],
            "kill_attempted": False,
            "scancel_attempted": False,
        }
        _write_json_artifact("gpu_bridge_live_discovery_compute_app_guard.json", guard)
        return guard

    def _live_cuda_status_ok(self, status: Mapping[str, Any], compute_guard: Mapping[str, Any]) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if status.get("success") is not True:
            reasons.append("cuda_probe_failed")
        if _is_admin_hostname(status.get("hostname")):
            reasons.append("admin_or_missing_hostname")
        if not str(status.get("CUDA_VISIBLE_DEVICES") or "").strip():
            reasons.append("missing_cuda_visible_devices")
        if int(status.get("torch.cuda.device_count") or 0) < 2:
            reasons.append("device_count_lt_2")
        if status.get("torch.cuda.is_available") is not True:
            reasons.append("torch_cuda_unavailable")
        nvidia = status.get("nvidia-smi") if isinstance(status.get("nvidia-smi"), Mapping) else {}
        if nvidia.get("returncode") != 0:
            reasons.append("nvidia_smi_failed")
        if compute_guard.get("pass") is not True:
            reasons.append("compute_app_guard_blocked")
        return not reasons, reasons

    def _eligible_live_candidate(self, candidate: PaneCandidate) -> tuple[bool, list[str]]:
        reasons = _non_send_filter_reasons(candidate)
        if candidate.dead:
            reasons.append("pane_dead")
        if not bool(candidate.cwd_match_repo):
            reasons.append("cwd_not_repo")
        command_class = str(candidate.command_class or classify_command(candidate.pane_command))
        if command_class not in RUNNABLE_PANE_COMMAND_CLASSES:
            reasons.append("pane_not_allowlisted_for_child_probe")
        return not reasons, sorted(set(reasons))

    def _write_candidate_ranking_report(
        self,
        *,
        stale_handoff: Mapping[str, Any] | None,
        discovery: DiscoveryResult,
        records: Sequence[Mapping[str, Any]],
        selected: Mapping[str, Any] | None,
        stop_reason: str | None,
    ) -> None:
        payload = {
            "schema_version": "odcr_gpu_bridge_candidate_ranking/1",
            "generated_at": _utc_now(),
            "stale_handoff_detected": bool(stale_handoff),
            "stale_handoff_warning": stale_handoff,
            "target_source": TARGET_SOURCE_LIVE_DISCOVERY if selected else None,
            "selection_policy": [
                "live_cuda_probe_success",
                "hostname_non_admin",
                "torch_cuda_available",
                "device_count_gte_2",
                "cuda_visible_devices_nonempty",
                "slurm_job_id_nonempty",
                "compute_app_guard_pass",
                "cwd_match_repo",
                "allowlisted_child_probe",
                "safe_pane_mode",
                "handoff_match",
                "current_tmux_match",
                "latest_slurm_job_id",
            ],
            "discovery": discovery.to_dict(),
            "candidate_records": list(records),
            "selected": dict(selected) if isinstance(selected, Mapping) else None,
            "stop_reason": stop_reason,
            "admin_pane_selected": bool(
                selected
                and _is_admin_hostname(((selected.get("child_status") or {}) if isinstance(selected, Mapping) else {}).get("hostname"))
            ),
            "stale_state_used": False,
            "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
        }
        _write_json_artifact("gpu_bridge_candidate_ranking_report.json", payload)

    def _write_stale_policy_report(
        self,
        *,
        stale_handoff: Mapping[str, Any] | None,
        live_attempted: bool,
        live_success: bool,
        stop_reason: str | None,
        selected_source: str | None,
    ) -> None:
        payload = {
            "schema_version": "odcr_gpu_bridge_stale_handoff_policy/1",
            "generated_at": _utc_now(),
            "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
            "current_gpu_pane_is_priority_hint": True,
            "stale_handoff_detected": bool(stale_handoff),
            "stale_handoff_warning": stale_handoff,
            "stale_handoff_stops_discovery": False,
            "stale_handoff_used_as_target": False,
            "admin_fallback_allowed": False,
            "live_discovery_attempted_after_stale_handoff": bool(live_attempted),
            "live_discovery_cuda_probe_success": bool(live_success),
            "selected_cuda_target_source": selected_source,
            "stale_state_used": False,
            "stop_reason": stop_reason,
        }
        _write_json_artifact("gpu_bridge_stale_handoff_policy_report.json", payload)

    def _choose_live_candidate(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        handoff: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        successful = [record for record in records if record.get("live_cuda_eligible") is True]
        if not successful:
            raise BridgeError(
                "No unique safe CUDA tmux pane was found after the stale current_gpu_pane.json warning.",
                stop_reason=NO_LIVE_CUDA_AFTER_STALE,
                details={"stale_state_used": False, "records": list(records)},
            )
        handoff_selection = _handoff_selection_candidates(handoff)
        handoff_matches = [
            record
            for record in successful
            if _candidate_matches_selection(_pane_candidate_from_payload(record["pane"]), handoff_selection)
        ]
        if len(handoff_matches) == 1:
            return handoff_matches[0]
        if len(handoff_matches) > 1:
            successful = handoff_matches
        else:
            current_matches = [
                record for record in successful if _candidate_matches_current_tmux(_pane_candidate_from_payload(record["pane"]))
            ]
            if len(current_matches) == 1:
                return current_matches[0]
            if len(current_matches) > 1:
                successful = current_matches
        jobs = [(record, _slurm_job_id(record.get("child_status") or {})) for record in successful]
        numeric_jobs = [(record, job) for record, job in jobs if job is not None]
        if numeric_jobs:
            newest = max(job for _record, job in numeric_jobs)
            newest_matches = [record for record, job in numeric_jobs if job == newest]
            if len(newest_matches) == 1:
                return newest_matches[0]
            successful = newest_matches
        if len(successful) == 1:
            return successful[0]
        raise BridgeError(
            "Multiple live CUDA tmux panes passed probe and ranking could not pick one without ambiguity.",
            stop_reason=AMBIGUOUS_LIVE_CUDA_PANES,
            details={"stale_state_used": False, "records": list(records)},
        )

    def _resolve_live_discovery_cuda_target(
        self,
        options: BridgeOptions,
        *,
        stale_handoff: Mapping[str, Any] | None,
        timeout_s: int,
    ) -> tuple[PaneCandidate, DiscoveryResult, str, dict[str, Any]]:
        discovery = discover_panes(
            runner=self.runner,
            socket_paths=None,
            socket_exists=self.socket_exists,
            all_sockets=True,
            include_filtered=True,
            capture_hash=True,
            repo_root=REPO_ROOT,
        )
        records: list[dict[str, Any]] = []
        latest_discovery = discovery
        for pane in discovery.panes:
            candidate = pane
            recovery: dict[str, Any] | None = None
            if candidate.in_mode:
                recovered, refreshed, recovery = self._recover_pane_mode(candidate, options, latest_discovery)
                latest_discovery = refreshed
                if recovered is None:
                    records.append(
                        {
                            "pane": candidate.to_dict(),
                            "target_source": TARGET_SOURCE_LIVE_DISCOVERY,
                            "live_cuda_eligible": False,
                            "skip_reasons": [str((recovery or {}).get("result") or "pane_mode_recovery_failed")],
                            "pane_mode_recovery": recovery,
                        }
                    )
                    continue
                candidate = recovered
            eligible, reasons = self._eligible_live_candidate(candidate)
            record: dict[str, Any] = {
                "pane": candidate.to_dict(),
                "target_source": TARGET_SOURCE_LIVE_DISCOVERY,
                "pane_mode_recovery": recovery,
                "static_eligible_for_child_probe": bool(eligible),
                "skip_reasons": list(reasons),
                "live_cuda_eligible": False,
            }
            if not eligible:
                records.append(record)
                continue
            bridge_result = self._run_handshake_on_target(
                candidate,
                mode="cuda-probe",
                spec_name="bridge.cuda_probe",
                require_cuda=True,
                stage=options.stage,
                task=options.task_id,
                timeout_s=timeout_s,
                dry_run=False,
                no_send=False,
                target_source=TARGET_SOURCE_LIVE_DISCOVERY,
            )
            status = bridge_result.get("child_status") if isinstance(bridge_result.get("child_status"), Mapping) else {}
            compute_guard = self._compute_app_guard_from_status(
                status=status,
                candidate=candidate,
                target_source=TARGET_SOURCE_LIVE_DISCOVERY,
            )
            ok, live_reasons = self._live_cuda_status_ok(status, compute_guard)
            record.update(
                {
                    "bridge_result": bridge_result,
                    "child_status": dict(status),
                    "compute_app_guard": compute_guard,
                    "live_cuda_probe_success": bool(bridge_result.get("success")),
                    "hostname_non_admin": not _is_admin_hostname(status.get("hostname")),
                    "device_count": status.get("torch.cuda.device_count"),
                    "cuda_visible_devices": status.get("CUDA_VISIBLE_DEVICES"),
                    "slurm_job_id": status.get("SLURM_JOB_ID"),
                    "live_cuda_eligible": bool(ok),
                    "skip_reasons": list(dict.fromkeys([*record["skip_reasons"], *live_reasons])),
                    "handoff_match": _candidate_matches_selection(candidate, _handoff_selection_candidates(stale_handoff)),
                    "current_tmux_match": _candidate_matches_current_tmux(candidate),
                }
            )
            records.append(record)
        selected_record: Mapping[str, Any] | None = None
        stop_reason: str | None = None
        try:
            selected_record = self._choose_live_candidate(records=records, handoff=stale_handoff)
        except BridgeError as exc:
            stop_reason = exc.stop_reason
            self._write_candidate_ranking_report(
                stale_handoff=stale_handoff,
                discovery=latest_discovery,
                records=records,
                selected=None,
                stop_reason=stop_reason,
            )
            self._write_stale_policy_report(
                stale_handoff=stale_handoff,
                live_attempted=True,
                live_success=False,
                stop_reason=stop_reason,
                selected_source=None,
            )
            raise
        selected_candidate = _pane_candidate_from_payload(selected_record["pane"])
        self._write_candidate_ranking_report(
            stale_handoff=stale_handoff,
            discovery=latest_discovery,
            records=records,
            selected=selected_record,
            stop_reason=None,
        )
        self._write_stale_policy_report(
            stale_handoff=stale_handoff,
            live_attempted=True,
            live_success=True,
            stop_reason=None,
            selected_source=TARGET_SOURCE_LIVE_DISCOVERY,
        )
        return selected_candidate, latest_discovery, TARGET_SOURCE_LIVE_DISCOVERY, {
            "schema_version": "odcr_gpu_bridge_live_discovery_selection/1",
            "stale_handoff_detected": bool(stale_handoff),
            "stale_handoff_warning": dict(stale_handoff or {}),
            "stale_handoff_blocked_execution": False,
            "live_discovery_cuda_probe_success": True,
            "srun_pane_rejected_static_only": False,
            "live_cuda_probe_overrides_static_pane_command": (
                bool(selected_record.get("live_cuda_probe_success"))
                and str((selected_record.get("pane") or {}).get("command_class") or "") != "shell"
            ),
            "selected": dict(selected_record),
            "candidate_records": records,
            "stale_state_used": False,
        }

    def _pane_mode_recovery_guard(self, candidate: PaneCandidate) -> dict[str, Any]:
        command = Path(str(candidate.pane_command or "")).name.lower()
        command_active = any(token == command or token in command for token in ACTIVE_COMPUTE_PANE_COMMANDS)
        compute_apps = self._compute_app_snapshot()
        safe = not command_active and not bool(compute_apps.get("active"))
        reason = "safe_mode_exit_key_allowed"
        if command_active:
            reason = "pane_command_looks_like_active_compute_app"
        elif bool(compute_apps.get("active")):
            reason = "nvidia_smi_compute_apps_nonempty"
        return {
            "safe": bool(safe),
            "reason": reason,
            "pane_command": candidate.pane_command,
            "active_compute_pane_command": bool(command_active),
            "compute_apps": compute_apps,
        }

    def _send_mode_exit_key(self, candidate: PaneCandidate, key: str) -> dict[str, Any]:
        if key not in PANE_MODE_EXIT_KEYS:
            raise BridgeError(f"unregistered pane-mode recovery key: {key}", stop_reason="forbidden_recovery_key")
        result = self.runner.run(("tmux", "-S", candidate.socket, "send-keys", "-t", candidate.pane_id, key), timeout=10)
        return {
            "key": key,
            "argv": list(result.args),
            "returncode": int(result.returncode),
            "stderr": result.stderr.strip(),
            "success": result.returncode == 0,
        }

    def _in_mode_entries(self, discovery: DiscoveryResult, options: BridgeOptions) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        for item in discovery.invalid:
            if item.get("reason") != "pane_in_mode":
                continue
            if options.socket and str(item.get("socket") or "") != str(options.socket):
                continue
            if not _matches_requested_target(item, options.target):
                continue
            entries.append(item)
        return entries

    def _write_pane_recovery_evidence(self, evidence: Mapping[str, Any], *, stage: str | None, task: int | None) -> None:
        writer = get_writer(REPO_ROOT)
        writer.runtime_diagnostic(
            PANE_MODE_RECOVERY_STATUS_NAME,
            dict(evidence),
            source="tmux_gpu_bridge",
            stage=stage,
            task=task,
            validation_result={"success": bool(evidence.get("success")), "result": evidence.get("result")},
            errors=[] if bool(evidence.get("success")) else [str(evidence.get("result") or "pane_mode_recovery_failed")],
        )

    def _recover_pane_mode(
        self,
        candidate: PaneCandidate,
        options: BridgeOptions,
        before: DiscoveryResult,
    ) -> tuple[PaneCandidate | None, DiscoveryResult, dict[str, Any]]:
        evidence: dict[str, Any] = {
            "schema_version": "odcr_runtime_pane_mode_recovery/1",
            "generated_at": _utc_now(),
            "target": candidate.target,
            "pane_id": candidate.pane_id,
            "socket": candidate.socket,
            "pane_command": candidate.pane_command,
            "before_in_mode": True,
            "after_in_mode": True,
            "recovery_keys_sent": [],
            "attempts": [],
            "retry_count": 0,
            "stale_state_used": False,
            "tmux_session_control_used": False,
            "result": "not_attempted",
            "success": False,
        }
        guard = self._pane_mode_recovery_guard(candidate)
        evidence["compute_app_guard"] = guard
        if not bool(guard.get("safe")):
            evidence["result"] = "blocked_by_compute_app_guard"
            self._write_pane_recovery_evidence(evidence, stage=options.stage, task=options.task_id)
            return None, before, evidence
        last_discovery = before
        for key in PANE_MODE_EXIT_KEYS:
            sent = self._send_mode_exit_key(candidate, key)
            evidence["recovery_keys_sent"].append(key)
            evidence["retry_count"] = len(evidence["recovery_keys_sent"])
            refreshed = discover_panes(
                runner=self.runner,
                socket_paths=(Path(candidate.socket),),
                socket_exists=self.socket_exists,
                repo_root=REPO_ROOT,
            )
            runnable = _find_candidate(refreshed, candidate)
            still_in_mode = _find_in_mode(refreshed, candidate) is not None
            evidence["attempts"].append(
                {
                    "key": key,
                    "send": sent,
                    "after_in_mode": bool(still_in_mode),
                    "candidate_count": len(refreshed.candidates),
                }
            )
            last_discovery = refreshed
            if not bool(sent.get("success")):
                evidence["result"] = "tmux_mode_exit_key_send_failed"
                break
            if runnable is not None:
                evidence["after_in_mode"] = False
                evidence["result"] = "pane_mode_recovered"
                evidence["success"] = True
                self._write_pane_recovery_evidence(evidence, stage=options.stage, task=options.task_id)
                return runnable, refreshed, evidence
            if not still_in_mode:
                evidence["after_in_mode"] = False
                evidence["result"] = "pane_left_mode_but_not_runnable"
                break
            self.sleep(0.2)
        else:
            evidence["result"] = "pane_remains_in_mode_after_safe_recovery_attempts"
        evidence["after_in_mode"] = _find_in_mode(last_discovery, candidate) is not None
        if evidence["result"] == "not_attempted":
            evidence["result"] = "pane_remains_in_mode_after_safe_recovery_attempts"
        self._write_pane_recovery_evidence(evidence, stage=options.stage, task=options.task_id)
        return None, last_discovery, evidence

    def resolve_target_with_recovery(
        self,
        options: BridgeOptions,
    ) -> tuple[PaneCandidate, DiscoveryResult, str, dict[str, Any] | None]:
        try:
            target, discovery, source = self.resolve_target(options)
            return target, discovery, source, None
        except BridgeError as exc:
            if exc.stop_reason == STALE_HANDOFF_STOP_REASON:
                raise
            socket_paths = (Path(options.socket),) if options.socket else None
            discovery = discover_panes(runner=self.runner, socket_paths=socket_paths, socket_exists=self.socket_exists)
            entries = self._in_mode_entries(discovery, options)
            if len(discovery.candidates) == 0 and len(entries) == 1:
                candidate = _pane_candidate_from_invalid(entries[0])
                recovered, refreshed, evidence = self._recover_pane_mode(candidate, options, discovery)
                if recovered is not None:
                    return recovered, refreshed, "pane_mode_recovery", evidence
                raise BridgeError(
                    str(evidence.get("result") or "pane_mode_recovery_failed"),
                    stop_reason=str(evidence.get("result") or "pane_mode_recovery_failed"),
                    details=evidence,
                ) from exc
            raise exc

    def _send_registered_command(self, target: PaneCandidate, command: tuple[str, ...]) -> None:
        line = shlex.join(command)
        if any(token in line for token in ("bash -c", "python -c", "srun", "sbatch", "scancel", "odcr-enter-gpu", "nohup", " &")):
            raise BridgeError("registered tmux command contains forbidden token", stop_reason="forbidden_command")
        result = self.runner.run(("tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "-l", line), timeout=10)
        if result.returncode != 0:
            raise BridgeError(f"tmux send command failed: {result.stderr}", stop_reason="tmux_send_failed")
        enter = self.runner.run(("tmux", "-S", target.socket, "send-keys", "-t", target.pane_id, "Enter"), timeout=10)
        if enter.returncode != 0:
            raise BridgeError(f"tmux send enter failed: {enter.stderr}", stop_reason="tmux_send_failed")

    def _step5_handoff_admission(self, *, stage: str, socket: str | None, target: str | None) -> dict[str, Any]:
        if str(stage) not in STEP5_FRESH_HANDOFF_REQUIRED_STAGES:
            return {"required": False, "ok": True}
        handoff = _current_handoff_selection()
        if bool(handoff.get("exists")) and not bool(handoff.get("valid")):
            return {
                "required": True,
                "ok": True,
                "target_source": TARGET_SOURCE_LIVE_DISCOVERY,
                "previous_target_source": TARGET_SOURCE_HANDOFF,
                "warning_reason": STALE_HANDOFF_STOP_REASON,
                "stale_handoff_detected": True,
                "stale_handoff_stops_discovery": False,
                "live_discovery_required": True,
                "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                "handoff": handoff,
                "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
            }
        if bool(handoff.get("valid")):
            return {
                "required": True,
                "ok": True,
                "target_source": TARGET_SOURCE_HANDOFF,
                "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                "socket": handoff.get("socket"),
                "target": handoff.get("target"),
                "cli_explicit_ignored_by_fresh_handoff": bool(socket or target),
                "handoff_cli_conflict": _cli_conflict_with_handoff(
                    BridgeOptions(mode="cuda-probe", stage=stage, socket=socket, target=target),
                    handoff,
                )
                if (socket or target)
                else None,
                "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
            }
        if socket or target:
            return {
                "required": True,
                "ok": True,
                "target_source": TARGET_SOURCE_CLI,
                "cli_explicit": True,
                "handoff_present": False,
                "handoff_cli_conflict": None,
            }
        if not bool(handoff.get("exists")):
            return {
                "required": True,
                "ok": False,
                "target_source": None,
                "stop_reason": "missing_current_gpu_pane_handoff",
                "error": (
                    "AI_analysis/runtime/current_gpu_pane.json is missing. "
                    "Please run odcr-enter-gpu <JOBID> inside the GPU tmux pane; "
                    "that command automatically refreshes current_gpu_pane.json."
                ),
                "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
            }
        return {"required": True, "ok": False, "stop_reason": "unreachable_handoff_admission_state"}

    def _wait_status(self, paths: GeneratedPaths, timeout_s: int) -> dict[str, Any]:
        started = self.clock()
        while self.clock() - started <= timeout_s:
            if paths.status.is_file():
                try:
                    payload = json.loads(paths.status.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    pass
            self.sleep(1)
        raise BridgeError(
            f"runtime bridge status timeout after {int(timeout_s)}s waiting for {paths.status}",
            stop_reason="timeout",
            details={"timeout_s": int(timeout_s), "status_path": str(paths.status)},
        )

    def _run_handshake_on_target(
        self,
        target: PaneCandidate,
        *,
        mode: str,
        spec_name: str,
        require_cuda: bool,
        stage: str | None,
        task: int,
        timeout_s: int,
        dry_run: bool = False,
        no_send: bool = False,
        target_source: str = TARGET_SOURCE_GLOBAL,
    ) -> dict[str, Any]:
        paths = generated_paths(f"{mode}_{_candidate_key(target)}")
        payload: dict[str, Any] = {
            "schema_version": "odcr_runtime_bridge_dispatch/1",
            "mode": mode,
            "command": spec_name,
            "command_id": f"{spec_name}:{int(time.time())}:{_safe_suffix(_candidate_key(target))}",
            "target": target.to_dict(),
            "target_source": target_source,
            "fresh_discover": True,
            "stale_state_used": False,
            "sent": False,
            "dry_run": bool(dry_run),
            "paths": paths.to_dict(),
            "success": False,
        }
        if dry_run or no_send:
            payload["success"] = not require_cuda
            payload["no_send"] = bool(no_send)
            return payload
        for path in (paths.status, paths.log, paths.report):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        command = _command_for_child(mode, paths, stage=stage, task=task, require_cuda=require_cuda)
        try:
            self._send_registered_command(target, command)
            payload["sent"] = True
            status = self._wait_status(paths, timeout_s)
        except BridgeError as exc:
            payload.update(
                {
                    "success": False,
                    "stop_reason": exc.stop_reason,
                    "error": str(exc),
                    "timeout_s": timeout_s,
                    "bridge_repair_required": True,
                }
            )
            return payload
        payload["child_status"] = status
        payload["success"] = bool(status.get("success"))
        payload["selected_cuda_pane"] = _candidate_key(target) if payload["success"] and require_cuda else None
        payload["selected_cuda_socket"] = target.socket if payload["success"] and require_cuda else None
        payload["selected_cuda_pane_id"] = target.pane_id if payload["success"] and require_cuda else None
        if target_source == TARGET_SOURCE_HANDOFF and require_cuda and not bool(payload["success"]):
            payload["stop_reason"] = STALE_HANDOFF_STOP_REASON
            payload["stale_handoff"] = True
            payload["bridge_repair_required"] = True
            payload["error"] = (
                "Fresh current_gpu_pane.json target failed CUDA validation. "
                "Please rerun odcr-enter-gpu <JOBID> inside the GPU tmux pane to refresh it."
            )
        return payload

    def run_bridge_mode(self, options: BridgeOptions) -> dict[str, Any]:
        reject_legacy_mode(options.mode)
        mode_to_spec = {
            "discover": "bridge.discover",
            "validate-only": "bridge.validate_only",
            "marker-probe": "bridge.marker_probe",
            "cuda-probe": "bridge.cuda_probe",
        }
        if options.mode not in mode_to_spec:
            raise BridgeError(f"unregistered bridge mode: {options.mode}", stop_reason="unregistered_command")
        spec = require_command(mode_to_spec[options.mode])
        if options.mode == "discover":
            return self.discover(options)
        if bool(options.global_discovery or options.all_sockets or options.all_panes):
            raise BridgeError(
                "Global/default tmux target selection is retired for bridge execution; use current_gpu_pane.json handoff v2 or CLI explicit socket/target.",
                stop_reason="global_target_selection_retired",
            )
        paths = generated_paths()
        live_selection: dict[str, Any] | None = None
        try:
            target, discovery, source, recovery = self.resolve_target_with_recovery(options)
        except BridgeError as exc:
            if (
                exc.stop_reason == STALE_HANDOFF_STOP_REASON
                and not (options.socket or options.target)
                and not options.dry_run
                and not options.no_send
            ):
                try:
                    target, discovery, source, live_selection = self._resolve_live_discovery_cuda_target(
                        options,
                        stale_handoff=(exc.details or {}).get("handoff") if isinstance(exc.details, Mapping) else None,
                        timeout_s=int(options.timeout or max(spec.timeout_s, 120)),
                    )
                    recovery = None
                except BridgeError as live_exc:
                    return _write_failure_report(
                        options.mode,
                        str(live_exc),
                        stage=options.stage,
                        task=options.task_id,
                        stop_reason=live_exc.stop_reason,
                        details=live_exc.details,
                    )
            else:
                return _write_failure_report(
                    options.mode,
                    str(exc),
                    stage=options.stage,
                    task=options.task_id,
                    stop_reason=exc.stop_reason,
                    details=exc.details,
                )
        payload: dict[str, Any] = {
            "schema_version": "odcr_runtime_bridge_dispatch/1",
            "mode": options.mode,
            "command": spec.name,
            "command_id": f"{spec.name}:{int(time.time())}",
            "target": target.to_dict(),
            "target_source": source,
            "discovery": discovery.to_dict(),
            "fresh_discover": True,
            "stale_state_used": False,
            "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
            "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
            "stale_handoff_detected": bool(live_selection and live_selection.get("stale_handoff_detected")),
            "stale_handoff_blocked_execution": False if live_selection else None,
            "live_discovery_cuda_probe_success": bool(live_selection and live_selection.get("live_discovery_cuda_probe_success")),
            "sent": False,
            "dry_run": options.dry_run,
            "paths": paths.to_dict(),
        }
        conflict = _cli_conflict_with_handoff(options, _current_handoff_selection())
        if conflict is not None:
            payload["handoff_cli_conflict"] = conflict
        if recovery is not None:
            payload["pane_mode_recovery"] = recovery
        if live_selection is not None:
            payload["live_discovery_selection"] = live_selection
            selected_record = live_selection.get("selected") if isinstance(live_selection.get("selected"), Mapping) else {}
            selected_pane = selected_record.get("pane") if isinstance(selected_record.get("pane"), Mapping) else {}
            payload["srun_pane_rejected_static_only"] = False
            payload["live_cuda_probe_overrides_static_pane_command"] = (
                bool(selected_record.get("live_cuda_probe_success"))
                and str(selected_pane.get("command_class") or "") != "shell"
            )
        if options.dry_run or options.no_send:
            payload["success"] = not spec.requires_gpu
            return payload
        for path in (paths.status, paths.log, paths.report):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        command = _command_for_child(
            options.mode,
            paths,
            stage=options.stage,
            task=options.task_id,
            require_cuda=spec.requires_gpu,
        )
        self._send_registered_command(target, command)
        payload["sent"] = True
        timeout_s = int(options.timeout or spec.timeout_s)
        payload["timeout_s"] = timeout_s
        try:
            status = self._wait_status(paths, timeout_s)
        except BridgeError as exc:
            payload.update(
                {
                    "success": False,
                    "stop_reason": exc.stop_reason,
                    "error": str(exc),
                    "timeout_s": timeout_s,
                    "bridge_repair_required": True,
                }
            )
            return payload
        payload["child_status"] = status
        payload["success"] = bool(status.get("success"))
        payload["selected_cuda_pane"] = _candidate_key(target) if payload["success"] and spec.requires_gpu else None
        payload["selected_cuda_socket"] = target.socket if payload["success"] and spec.requires_gpu else None
        payload["selected_cuda_pane_id"] = target.pane_id if payload["success"] and spec.requires_gpu else None
        payload["selected_cuda_hostname"] = status.get("hostname") if payload["success"] and spec.requires_gpu else None
        payload["selected_cuda_visible_devices"] = (
            status.get("CUDA_VISIBLE_DEVICES") if payload["success"] and spec.requires_gpu else None
        )
        payload["selected_device_count"] = (
            status.get("torch.cuda.device_count") if payload["success"] and spec.requires_gpu else None
        )
        if payload["success"] and spec.requires_gpu:
            compute_guard = self._compute_app_guard_from_status(status=status, candidate=target, target_source=source)
            payload["compute_app_guard"] = compute_guard
            payload["compute_app_guard_status"] = compute_guard.get("status")
            payload["selected_cuda_candidate"] = {
                "pane": target.to_dict(),
                "child_status": status,
                "compute_app_guard": compute_guard,
                "target_source": source,
            }
            if compute_guard.get("pass") is not True:
                payload["success"] = False
                payload["stop_reason"] = "compute_app_guard_blocked"
                payload["error"] = "Selected CUDA pane has unknown active compute-app processes; refusing to stack launch."
        if source == TARGET_SOURCE_HANDOFF and spec.requires_gpu and not bool(payload["success"]) and not payload.get("stop_reason"):
            payload["stop_reason"] = STALE_HANDOFF_STOP_REASON
            payload["stale_handoff"] = True
            payload["bridge_repair_required"] = True
            payload["error"] = (
                "Fresh current_gpu_pane.json target failed CUDA validation. "
                "Please rerun odcr-enter-gpu <JOBID> inside the GPU tmux pane to refresh it."
            )
        return payload

    def _patch_step5_probe_gpu_target_contract(
        self,
        *,
        probe_result: dict[str, Any],
        bridge_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected = bridge_result.get("selected_cuda_candidate") if isinstance(bridge_result.get("selected_cuda_candidate"), Mapping) else {}
        status = selected.get("child_status") if isinstance(selected.get("child_status"), Mapping) else {}
        compute_guard = selected.get("compute_app_guard") if isinstance(selected.get("compute_app_guard"), Mapping) else {}
        live_selection = (
            bridge_result.get("live_discovery_selection")
            if isinstance(bridge_result.get("live_discovery_selection"), Mapping)
            else {}
        )
        patch = {
            "gpu_target_source": bridge_result.get("target_source"),
            "stale_handoff_detected": bool(bridge_result.get("stale_handoff_detected")),
            "stale_handoff_blocked_execution": bool(bridge_result.get("stale_handoff_blocked_execution") is True),
            "live_discovery_cuda_probe_success": bool(bridge_result.get("live_discovery_cuda_probe_success")),
            "selected_cuda_socket": bridge_result.get("selected_cuda_socket"),
            "selected_cuda_pane": bridge_result.get("selected_cuda_pane_id") or bridge_result.get("selected_cuda_pane"),
            "selected_cuda_hostname": bridge_result.get("selected_cuda_hostname") or status.get("hostname"),
            "selected_cuda_visible_devices": bridge_result.get("selected_cuda_visible_devices") or status.get("CUDA_VISIBLE_DEVICES"),
            "selected_device_count": bridge_result.get("selected_device_count") or status.get("torch.cuda.device_count"),
            "compute_app_guard_status": bridge_result.get("compute_app_guard_status") or compute_guard.get("status"),
            "validation_e4_evidence_id": probe_result.get("validation_e4_evidence_id")
            or ((probe_result.get("all_trainable_grad") or {}).get("evidence_context") or {}).get("evidence_id"),
            "validation_scorer_only": probe_result.get("step5A_validation_scorer_only"),
            "validation_oom": probe_result.get("validation_oom"),
            "stale_state_used": False,
        }
        if live_selection:
            patch["live_discovery_candidate_count"] = len(list(live_selection.get("candidate_records") or []))
        probe_result.update(patch)
        source_path = Path(str(probe_result.get("source_table_path") or ""))
        if source_path.is_file():
            try:
                source = json.loads(source_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                source = {}
            if isinstance(source, dict):
                records = list(source.get("records") or [])
                record_map = {str(item.get("key")): dict(item) for item in records if isinstance(item, Mapping)}
                for key, value in patch.items():
                    record_map[key] = {"key": key, "value": value, "source": "GPU bridge live target admission"}
                source["records"] = list(record_map.values())
                source_path.write_text(
                    json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
                    encoding="utf-8",
                )
        output_dir = Path(str(probe_result.get("output_dir") or ""))
        result_path = output_dir / "result.json"
        if result_path.is_file():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
            if isinstance(existing, dict):
                existing.update(patch)
                result_path.write_text(
                    json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
                    encoding="utf-8",
                )
        return probe_result

    def run_probe(
        self,
        stage: str,
        task: int,
        *,
        socket: str | None = None,
        target: str | None = None,
        config_path: str = "configs/odcr.yaml",
        sets: Sequence[str] = (),
        candidate_id: str | None = None,
        timeout: int | None = None,
        from_step4: str | None = None,
        evidence_level: str | None = None,
        scan: bool = False,
        global_discovery: bool = False,
    ) -> dict[str, Any]:
        normalized_evidence_level = _normalize_probe_evidence_level(evidence_level)
        if normalized_evidence_level == E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE and str(stage) != "step5A":
            bridge_result = {
                "schema_version": "odcr_runtime_bridge_dispatch/1",
                "mode": "cuda-probe",
                "command": "runtime.probe",
                "success": False,
                "target_source": None,
                "stale_state_used": False,
                "stop_reason": "e5_only_supported_for_step5A",
                "error": "E5_step5A_post_train_eval_lifecycle is only valid for --stage step5A.",
                "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                "formal_train_command_emitted": False,
                "synthetic_batch_used_for_formal_gate": False,
                "evidence_level": normalized_evidence_level,
            }
            payload = write_probe_report(
                stage,
                task,
                handshake=None,
                probe_result=None,
                repo_root=REPO_ROOT,
                target_source=None,
                stale_state_used=False,
                handoff_admission={"ok": False, "stop_reason": bridge_result["stop_reason"]},
            )
            payload["bridge"] = bridge_result
            payload["evidence_level"] = normalized_evidence_level
            return payload
        command_name = probe_command_name(stage, bounded=True)
        spec = require_command(command_name)
        handoff_admission = self._step5_handoff_admission(stage=str(stage), socket=socket, target=target)
        if not bool(handoff_admission.get("ok")):
            bridge_result: dict[str, Any] = {
                "schema_version": "odcr_runtime_bridge_dispatch/1",
                "mode": "cuda-probe" if spec.requires_gpu else "validate-only",
                "command": spec.name,
                "success": False,
                "target_source": handoff_admission.get("target_source"),
                "stale_state_used": False,
                "stop_reason": handoff_admission.get("stop_reason"),
                "error": handoff_admission.get("error"),
                "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
                "formal_train_command_emitted": False,
                "synthetic_batch_used_for_formal_gate": False,
            }
            payload = write_probe_report(
                stage,
                task,
                handshake=None,
                probe_result=None,
                repo_root=REPO_ROOT,
                target_source=bridge_result.get("target_source"),
                stale_state_used=False,
                handoff_admission=handoff_admission,
            )
            payload["bridge"] = bridge_result
            payload["target_source"] = bridge_result.get("target_source")
            payload["stale_state_used"] = False
            payload["current_gpu_pane_json_path"] = str(CURRENT_HANDOFF_PATH)
            payload["old_gpu_pane_state_role"] = OLD_GPU_PANE_STATE_ROLE
            payload["formal_train_command_emitted"] = False
            payload["synthetic_batch_used_for_formal_gate"] = False
            return payload
        if bool(global_discovery or scan):
            bridge_result = {
                "schema_version": "odcr_runtime_bridge_dispatch/1",
                "mode": "cuda-probe" if spec.requires_gpu else "validate-only",
                "command": spec.name,
                "success": False,
                "target_source": TARGET_SOURCE_GLOBAL,
                "stale_state_used": False,
                "stop_reason": "global_target_selection_retired",
                "error": (
                    "Global/default tmux target selection is retired for bounded probes; "
                    "use current_gpu_pane.json handoff v2 or CLI explicit socket/target."
                ),
                "current_gpu_pane_json_path": str(CURRENT_HANDOFF_PATH),
                "old_gpu_pane_state_role": OLD_GPU_PANE_STATE_ROLE,
                "formal_train_command_emitted": False,
                "synthetic_batch_used_for_formal_gate": False,
            }
            payload = write_probe_report(
                stage,
                task,
                handshake=None,
                probe_result=None,
                repo_root=REPO_ROOT,
                target_source=bridge_result.get("target_source"),
                stale_state_used=False,
                handoff_admission=handoff_admission,
            )
            payload["bridge"] = bridge_result
            payload["target_source"] = bridge_result.get("target_source")
            return payload
        bridge_result = self.run_bridge_mode(
            BridgeOptions(
                mode="cuda-probe" if spec.requires_gpu else "validate-only",
                stage=stage,
                task_id=task,
                socket=socket,
                target=target,
                timeout=timeout,
                global_discovery=False,
                all_sockets=False,
                all_panes=False,
            )
        )
        handshake = bridge_result.get("child_status") if isinstance(bridge_result, dict) else None
        probe_result: dict[str, Any] | None = None
        if str(stage) in {"step5A", "step5B"} and bool(bridge_result.get("success")):
            try:
                selected = bridge_result.get("selected_cuda_candidate")
                selected_pane = selected.get("pane") if isinstance(selected, Mapping) else None
                if isinstance(selected_pane, Mapping):
                    target_pane = _pane_candidate_from_payload(selected_pane)
                else:
                    target_pane, _discovery, _source, recovery = self.resolve_target_with_recovery(
                        BridgeOptions(mode="cuda-probe", stage=stage, task_id=task, socket=socket, target=target)
                    )
                    if recovery is not None:
                        bridge_result["probe_pane_mode_recovery"] = recovery
                status_path = RAW_LOG_DIR / (
                    PROBE_E5_STATUS_NAME
                    if normalized_evidence_level == E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE
                    else PROBE_STATUS_NAME
                )
                try:
                    status_path.unlink()
                except FileNotFoundError:
                    pass
                cid = (
                    str(candidate_id)
                    if candidate_id
                    else None
                    if normalized_evidence_level == E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE
                    else "A0_C0_R0"
                )
                command = _command_for_probe_child(
                    stage=str(stage),
                    task=int(task),
                    status_path=status_path,
                    config_path=str(config_path),
                    sets=sets,
                    candidate_id=cid,
                    timeout=timeout,
                    from_step4=from_step4,
                    evidence_level=normalized_evidence_level,
                )
                require_command("bridge.probe_child")
                self._send_registered_command(target_pane, command)
                probe_timeout_s = int(timeout or spec.timeout_s)
                probe_result = self._wait_status(
                    GeneratedPaths(status=status_path, log=RAW_LOG_DIR / RUNTIME_LOG_NAME, report=REPORT_DIR / RUNTIME_REPORT_NAME),
                    probe_timeout_s + 60,
                )
                if isinstance(probe_result, dict):
                    probe_result = self._patch_step5_probe_gpu_target_contract(
                        probe_result=probe_result,
                        bridge_result=bridge_result,
                    )
            except BridgeError as exc:
                probe_result = {
                    "schema_version": "odcr_step5_e4_bounded_probe/1",
                    "stage": stage,
                    "task_id": int(task),
                    "candidate_id": str(candidate_id or "A0_C0_R0"),
                    "success": False,
                    "evidence_level": "E3_gpu_transport",
                    "requested_evidence_level": normalized_evidence_level,
                    "error": str(exc),
                }
        payload = write_probe_report(
            stage,
            task,
            handshake=handshake if isinstance(handshake, dict) else None,
            probe_result=probe_result,
            repo_root=REPO_ROOT,
            target_source=str(bridge_result.get("target_source") or ""),
            stale_state_used=bool(bridge_result.get("stale_state_used")),
            handoff_admission=handoff_admission,
        )
        payload["bridge"] = bridge_result
        payload["target_source"] = bridge_result.get("target_source")
        payload["selected_cuda_pane"] = bridge_result.get("selected_cuda_pane")
        payload["selected_cuda_socket"] = bridge_result.get("selected_cuda_socket")
        payload["selected_cuda_pane_id"] = bridge_result.get("selected_cuda_pane_id")
        payload["stale_state_used"] = bool(bridge_result.get("stale_state_used"))
        if probe_result is not None:
            if normalized_evidence_level == E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE:
                payload["step5_e5_probe"] = probe_result
            else:
                payload["step5_e4_probe"] = probe_result
        payload["requested_evidence_level"] = normalized_evidence_level
        payload["report_path"] = str(
            write_runtime_report(
                RUNTIME_REPORT_NAME,
                payload,
                repo_root=REPO_ROOT,
                stage=stage,
                task=task,
                source="bounded_probe",
            )
        )
        return payload


def run_handshake_child(args: argparse.Namespace) -> int:
    return write_handshake(
        kind=args.kind,
        require_cuda=bool(args.require_cuda),
        status_path=args.status_path,
        log_path=args.log_path,
        report_path=args.report_path,
        repo_root=args.repo_root,
        stage=args.stage,
        task=str(args.task) if args.task is not None else None,
    )


def run_probe_child(args: argparse.Namespace) -> int:
    from odcr_core.config_resolver import load_yaml_config, apply_cli_sets
    from odcr_core.step5_runtime_probe import baseline_candidate_from_config, launch_probe

    status_path = Path(str(args.status_path)).expanduser().resolve()
    sets = list(getattr(args, "sets", []) or [])
    candidate_id = str(getattr(args, "candidate_id", None) or "").strip()
    try:
        if not sets or not candidate_id:
            base = load_yaml_config(str(getattr(args, "config", None) or "configs/odcr.yaml"))
            cfg_with_cli, _sources = apply_cli_sets(base, sets)
            e4_cfg = ((cfg_with_cli.get("step5") or {}).get("e4_bounded") or {})
            if not isinstance(e4_cfg, dict):
                raise BridgeError("step5.e4_bounded must be configured for Step5 E4 probe child")
            hw_profile = str(((cfg_with_cli.get("hardware") or {}).get("active") or "default")).strip() or "default"
            min_per_gpu = None
            if _normalize_probe_evidence_level(getattr(args, "evidence_level", None)) == E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE:
                step5_train = ((cfg_with_cli.get("step5") or {}).get("train") or {})
                step5_eval = ((cfg_with_cli.get("step5") or {}).get("eval") or {})
                if isinstance(step5_train, Mapping):
                    min_per_gpu = int(step5_train.get("per_gpu_batch_size") or 0) or None
                if isinstance(step5_eval, Mapping):
                    min_per_gpu = max(
                        int(min_per_gpu or 0),
                        int(step5_eval.get("valid_per_gpu_batch_size") or 0),
                    ) or None
            baseline = baseline_candidate_from_config(
                e4_cfg,
                hardware_profile=hw_profile,
                candidate_id=candidate_id,
                min_per_gpu_batch_size=min_per_gpu,
            )
            if not candidate_id:
                candidate_id = str(baseline["candidate_id"])
            if not sets:
                sets = list(baseline["overrides"])
        result = launch_probe(
            stage=str(args.stage),
            task=int(args.task),
            candidate_id=candidate_id,
            config_path=str(getattr(args, "config", None) or "configs/odcr.yaml"),
            set_overrides=sets,
            from_step4=getattr(args, "from_step4", None),
            evidence_level=getattr(args, "evidence_level", None),
            timeout_s=getattr(args, "timeout", None),
        )
    except Exception as exc:
        result = {
            "schema_version": "odcr_step5_e4_bounded_probe/1",
            "stage": str(args.stage),
            "task_id": int(args.task),
            "candidate_id": candidate_id or "unknown",
            "success": False,
            "evidence_level": "E3_gpu_transport",
            "requested_evidence_level": _normalize_probe_evidence_level(getattr(args, "evidence_level", None)),
            "error": str(exc),
        }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return 0 if bool(result.get("success")) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odcr runtime", description="ODCR allowlisted runtime/tmux/GPU validation.")
    sub = parser.add_subparsers(dest="runtime_command", required=True)
    bridge = sub.add_parser("bridge", help="discover and validate the current tmux GPU pane")
    bridge_sub = bridge.add_subparsers(dest="bridge_command", required=True)
    for name in ("discover", "validate-only", "marker-probe", "cuda-probe"):
        item = bridge_sub.add_parser(name)
        item.add_argument("--socket", default=None)
        item.add_argument("--target", default=None)
        item.add_argument("--global", dest="global_discovery", action="store_true")
        item.add_argument("--all-sockets", action="store_true")
        item.add_argument("--all-panes", action="store_true")
        item.add_argument("--json", dest="json_output", action="store_true")
        item.add_argument("--dry-run", action="store_true")
        item.add_argument("--no-send", action="store_true")
        item.add_argument("--timeout", type=int, default=None)
    child = bridge_sub.add_parser("_handshake-child", help=argparse.SUPPRESS)
    child.add_argument("--kind", required=True)
    child.add_argument("--status-path", required=True)
    child.add_argument("--log-path", required=True)
    child.add_argument("--report-path", required=True)
    child.add_argument("--repo-root", required=True)
    child.add_argument("--stage", default=None)
    child.add_argument("--task", default=None)
    child.add_argument("--require-cuda", action="store_true")

    probe = sub.add_parser("probe", help="run a bounded allowlisted runtime probe")
    probe.add_argument("--stage", choices=("step3", "step4", "step5", "step5A", "step5B"), required=True)
    probe.add_argument("--task", type=int, required=True)
    probe.add_argument("--bounded", action="store_true", required=True)
    probe.add_argument("--socket", default=None)
    probe.add_argument("--target", default=None)
    probe.add_argument("--config", default="configs/odcr.yaml")
    probe.add_argument("--set", dest="sets", action="append", default=[])
    probe.add_argument("--candidate-id", default=None)
    probe.add_argument("--timeout", type=int, default=None)
    probe.add_argument("--from-step4-run", dest="from_step4", default=None)
    probe.add_argument(
        "--evidence-level",
        choices=("E4", "E5", E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION, E5_STEP5A_POST_TRAIN_EVAL_LIFECYCLE),
        default="E4",
    )
    probe.add_argument("--scan", action="store_true")
    probe.add_argument("--global", dest="global_discovery", action="store_true")
    probe.add_argument("--probe-child", action="store_true", help=argparse.SUPPRESS)
    probe.add_argument("--status-path", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.runtime_command == "bridge" and args.bridge_command == "_handshake-child":
            return run_handshake_child(args)
        if args.runtime_command == "probe" and bool(getattr(args, "probe_child", False)):
            return run_probe_child(args)
        tool = TmuxGpuBridge()
        if args.runtime_command == "bridge":
            result = tool.run_bridge_mode(
                BridgeOptions(
                    mode=args.bridge_command,
                    socket=getattr(args, "socket", None),
                    target=getattr(args, "target", None),
                    dry_run=bool(getattr(args, "dry_run", False)),
                    no_send=bool(getattr(args, "no_send", False)),
                    timeout=getattr(args, "timeout", None),
                    global_discovery=bool(getattr(args, "global_discovery", False)),
                    all_sockets=bool(getattr(args, "all_sockets", False)),
                    all_panes=bool(getattr(args, "all_panes", False)),
                    json_output=bool(getattr(args, "json_output", False)),
                )
            )
        elif args.runtime_command == "probe":
            result = tool.run_probe(
                args.stage,
                int(args.task),
                socket=args.socket,
                target=args.target,
                config_path=str(getattr(args, "config", None) or "configs/odcr.yaml"),
                sets=list(getattr(args, "sets", []) or []),
                candidate_id=getattr(args, "candidate_id", None),
                timeout=getattr(args, "timeout", None),
                from_step4=getattr(args, "from_step4", None),
                evidence_level=getattr(args, "evidence_level", None),
                scan=bool(getattr(args, "scan", False)),
                global_discovery=bool(getattr(args, "global_discovery", False)),
            )
        else:
            raise BridgeError(f"unregistered runtime command: {args.runtime_command}", stop_reason="unregistered_command")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
        return 0 if bool(result.get("success")) else 1
    except (BridgeError, RuntimeCommandError) as exc:
        payload = _write_failure_report("runtime", str(exc))
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
