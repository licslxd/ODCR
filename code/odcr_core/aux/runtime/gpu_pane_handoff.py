"""Fresh GPU pane handoff contract for ODCR tmux runtime selection."""

from __future__ import annotations

import getpass
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from odcr_core.file_atomic import atomic_write_json


SCHEMA_VERSION = "odcr_current_gpu_pane_handoff/2"
COMPATIBLE_SCHEMA_VERSIONS = {SCHEMA_VERSION}
ADMIN_PART_SCHEMA_VERSION = "odcr_current_gpu_pane_admin_part/1"
FAILED_SCHEMA_VERSION = "odcr_current_gpu_pane_failed/1"
CURRENT_HANDOFF_REL = Path("AI_analysis/runtime/current_gpu_pane.json")
ADMIN_PART_REL = Path("AI_analysis/runtime/current_gpu_pane.admin_part.json")
FAILED_HANDOFF_REL = Path("AI_analysis/runtime/current_gpu_pane.failed.json")
LEGACY_GPU_PANE_STATE_REL = Path("AI_analysis/runtime/gpu_pane.json")
TARGET_SOURCE_HANDOFF = "current_gpu_pane_handoff"
OLD_GPU_PANE_STATE_ROLE = "historical_hint_only"
STALE_HANDOFF_MESSAGE = "GPU pane handoff not written; Codex bridge must not use stale GPU state."
STALE_HANDOFF_STOP_REASON = "stale_current_gpu_pane_handoff"
DEFAULT_FRESH_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str


class HandoffError(RuntimeError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def current_handoff_path(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve() / CURRENT_HANDOFF_REL


def current_handoff_tmp_path(repo_root: str | Path) -> Path:
    path = current_handoff_path(repo_root)
    return path.with_suffix(path.suffix + ".tmp")


def admin_part_path(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve() / ADMIN_PART_REL


def admin_part_tmp_path(repo_root: str | Path) -> Path:
    path = admin_part_path(repo_root)
    return path.with_suffix(path.suffix + ".tmp")


def failed_handoff_path(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve() / FAILED_HANDOFF_REL


def legacy_gpu_pane_state_path(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve() / LEGACY_GPU_PANE_STATE_REL


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HandoffError(f"{label} is missing: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise HandoffError(f"{label} is unreadable: {path}", details={"error": str(exc)}) from exc
    if not isinstance(payload, dict):
        raise HandoffError(f"{label} must be a JSON object: {path}")
    return payload


def _parse_generated_at(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise HandoffError("current GPU pane handoff missing generated_at_utc.")
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HandoffError("current GPU pane handoff generated_at_utc is invalid.", details={"generated_at_utc": text}) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_freshness(payload: Mapping[str, Any], *, max_age_seconds: int = DEFAULT_FRESH_SECONDS) -> None:
    generated = _parse_generated_at(payload.get("generated_at_utc"))
    now = datetime.now(timezone.utc)
    age = (now - generated).total_seconds()
    if age < -300:
        raise HandoffError(
            "current GPU pane handoff generated_at_utc is in the future.",
            details={"generated_at_utc": generated.isoformat(), "age_seconds": age},
        )
    if age > int(max_age_seconds):
        raise HandoffError(
            "current GPU pane handoff is stale.",
            details={"generated_at_utc": generated.isoformat(), "age_seconds": age, "max_age_seconds": int(max_age_seconds)},
        )


def parse_tmux_env(raw_tmux: str | None, pane: str | None) -> dict[str, str]:
    raw = str(raw_tmux or "").strip()
    pane_value = str(pane or "").strip()
    if not raw:
        raise HandoffError("TMUX is empty; current pane is not inside tmux.")
    if not pane_value:
        raise HandoffError("TMUX_PANE is empty; current pane cannot be selected.")
    parts = raw.split(",", 2)
    socket = parts[0].strip() if parts else ""
    if not socket:
        raise HandoffError("TMUX socket could not be parsed from TMUX.")
    return {
        "raw_TMUX": raw,
        "socket": socket,
        "pane": pane_value,
        "server_pid_diagnostic_only": parts[1].strip() if len(parts) > 1 else "",
        "session_id_diagnostic_only": parts[2].strip() if len(parts) > 2 else "",
        "selection_uses_pid": False,
    }


def _run_subprocess(args: Sequence[str], *, timeout: int = 15) -> CommandResult:
    try:
        proc = subprocess.run(list(args), text=True, capture_output=True, check=False, timeout=timeout)
        return CommandResult(tuple(str(part) for part in args), proc.returncode, proc.stdout.strip(), proc.stderr.strip())
    except Exception as exc:
        return CommandResult(tuple(str(part) for part in args), None, "", repr(exc))


def _result_payload(result: CommandResult) -> dict[str, Any]:
    return {
        "argv": list(result.args),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _torch_probe() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_device_names": [],
        "torch_error": None,
    }
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        payload["torch_cuda_available"] = available
        payload["torch_cuda_device_count"] = count
        if available and count > 0:
            payload["torch_cuda_device_names"] = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    return payload


def _parse_tmux_metadata(stdout: str, *, socket: str, pane: str) -> dict[str, Any]:
    parts = str(stdout or "").split("\t")
    if len(parts) != 6:
        raise HandoffError("tmux metadata probe did not return the expected fields.")
    session, target, pane_id, current_path, current_command, in_mode = parts
    return {
        "session": session,
        "target": target,
        "pane_id": pane_id or pane,
        "pane_current_path": current_path,
        "pane_current_command": current_command,
        "pane_in_mode": str(in_mode).strip() == "1",
        "socket": socket,
    }


def _tmux_metadata(
    *,
    socket: str,
    pane: str,
    runner: Callable[[Sequence[str]], CommandResult],
) -> dict[str, Any]:
    fmt = "\t".join(
        (
            "#{session_name}",
            "#{session_name}:#{window_index}.#{pane_index}",
            "#{pane_id}",
            "#{pane_current_path}",
            "#{pane_current_command}",
            "#{pane_in_mode}",
        )
    )
    result = runner(("tmux", "-S", socket, "display-message", "-p", "-t", pane, fmt))
    if result.returncode != 0:
        raise HandoffError(
            "tmux metadata probe failed for current pane.",
            details={"tmux_metadata_probe": _result_payload(result)},
        )
    return _parse_tmux_metadata(result.stdout, socket=socket, pane=pane)


def _admin_tmux_from_parts(tmux: Mapping[str, Any], metadata: Mapping[str, Any], *, captured_on_host: str) -> dict[str, Any]:
    pane = str(tmux.get("pane") or "").strip()
    return {
        "raw_TMUX": str(tmux.get("raw_TMUX") or ""),
        "socket": str(tmux.get("socket") or ""),
        "pane": pane,
        "server_pid_diagnostic_only": str(tmux.get("server_pid_diagnostic_only") or ""),
        "session_id_diagnostic_only": str(tmux.get("session_id_diagnostic_only") or ""),
        "selection_uses_pid": False,
        "session": str(metadata.get("session") or ""),
        "target": str(metadata.get("target") or ""),
        "pane_id": str(metadata.get("pane_id") or pane),
        "pane_current_path": str(metadata.get("pane_current_path") or ""),
        "pane_current_command": str(metadata.get("pane_current_command") or ""),
        "pane_in_mode": bool(metadata.get("pane_in_mode")),
        "captured_on_host": str(captured_on_host or platform.node()),
    }


def _validate_gpu_environment(payload: Mapping[str, Any]) -> None:
    hostname = str(payload.get("hostname") or "").strip()
    if not hostname or hostname.lower().startswith("admin"):
        raise HandoffError("current host is not a GPU node.", details={"hostname": hostname})
    if not str(payload.get("slurm_job_id") or "").strip():
        raise HandoffError("SLURM_JOB_ID is empty; current pane is not inside the GPU allocation.")
    if not str(payload.get("cuda_visible_devices") or "").strip():
        raise HandoffError("CUDA_VISIBLE_DEVICES is empty; current pane does not expose CUDA devices.")
    cuda = payload.get("cuda") if isinstance(payload.get("cuda"), Mapping) else {}
    nvidia_rc = cuda.get("nvidia_smi_L_returncode")
    if nvidia_rc is None or int(nvidia_rc) != 0:
        raise HandoffError("nvidia-smi -L failed; current pane is not validated as GPU-capable.", details={"cuda": dict(cuda)})
    if not bool(cuda.get("torch_cuda_available")):
        raise HandoffError("torch.cuda.is_available() is false in the current pane.", details={"cuda": dict(cuda)})
    if int(cuda.get("torch_cuda_device_count") or 0) < 2:
        raise HandoffError("torch.cuda.device_count must be at least 2 for ODCR GPU pane handoff.", details={"cuda": dict(cuda)})


def _gpu_environment_from_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hostname": runtime.get("hostname"),
        "slurm_job_id": runtime.get("slurm_job_id"),
        "cuda_visible_devices": runtime.get("cuda_visible_devices"),
        "cuda": {
            "nvidia_smi_L_returncode": runtime.get("nvidia_smi_L_returncode"),
            "nvidia_smi_L_stdout": runtime.get("nvidia_smi_L_stdout"),
            "nvidia_smi_L_stderr": runtime.get("nvidia_smi_L_stderr"),
            "compute_apps_returncode": runtime.get("compute_apps_returncode"),
            "compute_apps_stdout": runtime.get("compute_apps_stdout"),
            "compute_apps_stderr": runtime.get("compute_apps_stderr"),
            "torch_cuda_available": runtime.get("torch_cuda_available"),
            "torch_cuda_device_count": runtime.get("torch_cuda_device_count"),
            "torch_cuda_device_names": runtime.get("torch_cuda_device_names"),
            "torch_error": runtime.get("torch_error"),
        },
    }


def _admin_tmux_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    admin_tmux = payload.get("admin_tmux") if isinstance(payload.get("admin_tmux"), Mapping) else {}
    return dict(admin_tmux)


def _gpu_runtime_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get("gpu_runtime") if isinstance(payload.get("gpu_runtime"), Mapping) else {}
    return dict(runtime)


def validate_current_gpu_pane_payload(payload: Mapping[str, Any], *, max_age_seconds: int = DEFAULT_FRESH_SECONDS) -> None:
    if payload.get("schema_version") not in COMPATIBLE_SCHEMA_VERSIONS:
        raise HandoffError("current GPU pane handoff schema_version is invalid.")
    _validate_freshness(payload, max_age_seconds=max_age_seconds)
    if payload.get("valid_for_bridge_selection") is not True:
        raise HandoffError("current GPU pane handoff is not valid_for_bridge_selection.")
    admin_tmux = _admin_tmux_from_payload(payload)
    if not admin_tmux:
        raise HandoffError("current GPU pane handoff missing admin_tmux object.")
    if bool(admin_tmux.get("selection_uses_pid")):
        raise HandoffError("current GPU pane handoff must not use TMUX server PID for selection.")
    socket = str(admin_tmux.get("socket") or "").strip()
    pane = str(admin_tmux.get("pane") or admin_tmux.get("pane_id") or "").strip()
    if not socket or not pane:
        raise HandoffError("current GPU pane handoff missing socket or pane.")
    selection = payload.get("selection_key") if isinstance(payload.get("selection_key"), Mapping) else {}
    if str(selection.get("socket") or "").strip() != socket or str(selection.get("pane") or "").strip() != pane:
        raise HandoffError("current GPU pane handoff selection_key does not match tmux socket/pane.")
    gpu_runtime = _gpu_runtime_from_payload(payload)
    _validate_gpu_environment(_gpu_environment_from_runtime(gpu_runtime))


def selection_from_handoff_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_current_gpu_pane_payload(payload)
    admin_tmux = _admin_tmux_from_payload(payload)
    return {
        "source": TARGET_SOURCE_HANDOFF,
        "socket": str(admin_tmux.get("socket") or "").strip(),
        "target": str(admin_tmux.get("pane") or admin_tmux.get("pane_id") or "").strip(),
        "payload": dict(payload),
    }


def load_current_handoff(repo_root: str | Path) -> dict[str, Any] | None:
    path = current_handoff_path(repo_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        raise HandoffError(f"current GPU pane handoff is unreadable: {path}", details={"error": str(exc)}) from exc
    if not isinstance(payload, dict):
        raise HandoffError(f"current GPU pane handoff must be a JSON object: {path}")
    return payload


def remove_current_handoff(repo_root: str | Path) -> None:
    for path in (current_handoff_path(repo_root), current_handoff_tmp_path(repo_root)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def prepare_handoff_start(repo_root: str | Path, *, remove_admin_part: bool = False) -> None:
    paths = [
        current_handoff_path(repo_root),
        current_handoff_tmp_path(repo_root),
        failed_handoff_path(repo_root),
        admin_part_tmp_path(repo_root),
    ]
    if remove_admin_part:
        paths.append(admin_part_path(repo_root))
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def write_failed_handoff(
    *,
    repo_root: str | Path,
    mode: str,
    source: str,
    error: str,
    details: Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    remove_current_handoff(repo)
    payload: dict[str, Any] = {
        "schema_version": FAILED_SCHEMA_VERSION,
        "source": str(source),
        "mode": str(mode),
        "generated_at_utc": generated_at_utc or _utc_now(),
        "repo_root": str(repo),
        "valid_for_bridge_selection": False,
        "active_handoff_deleted": True,
        "error": str(error),
        "details": dict(details or {}),
        "message": STALE_HANDOFF_MESSAGE,
    }
    atomic_write_json(failed_handoff_path(repo), payload)
    return payload


def validate_admin_part_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != ADMIN_PART_SCHEMA_VERSION:
        raise HandoffError("admin pre-srun handoff schema_version is invalid.")
    admin_tmux = payload.get("admin_tmux") if isinstance(payload.get("admin_tmux"), Mapping) else {}
    if not admin_tmux:
        raise HandoffError("admin pre-srun handoff missing admin_tmux.")
    if bool(admin_tmux.get("selection_uses_pid")):
        raise HandoffError("admin pre-srun handoff must not use TMUX server PID for selection.")
    if not str(admin_tmux.get("socket") or "").strip() or not str(admin_tmux.get("pane") or "").strip():
        raise HandoffError("admin pre-srun handoff missing socket or pane.")


def build_admin_pre_srun_payload(
    *,
    repo_root: str | Path,
    source: str,
    job_id: str,
    selected_node: str | None = None,
    env: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str]], CommandResult] | None = None,
    hostname: str | None = None,
    cwd: str | None = None,
    user: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    env_map = dict(os.environ if env is None else env)
    runner_fn = runner or (lambda args: _run_subprocess(args))
    tmux = parse_tmux_env(env_map.get("TMUX"), env_map.get("TMUX_PANE"))
    metadata = _tmux_metadata(socket=tmux["socket"], pane=tmux["pane"], runner=runner_fn)
    captured_on_host = str(hostname or platform.node())
    admin_tmux = _admin_tmux_from_parts(tmux, metadata, captured_on_host=captured_on_host)
    payload: dict[str, Any] = {
        "schema_version": ADMIN_PART_SCHEMA_VERSION,
        "source": str(source),
        "generated_at_utc": generated_at_utc or _utc_now(),
        "repo_root": str(repo),
        "cwd": str(cwd or os.getcwd()),
        "user": str(user or getpass.getuser()),
        "selected_job_id": str(job_id or ""),
        "selected_node": str(selected_node or ""),
        "admin_tmux": admin_tmux,
    }
    validate_admin_part_payload(payload)
    return payload


def write_admin_pre_srun(
    *,
    repo_root: str | Path,
    source: str,
    job_id: str,
    selected_node: str | None = None,
    env: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str]], CommandResult] | None = None,
    hostname: str | None = None,
    cwd: str | None = None,
    user: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    try:
        prepare_handoff_start(repo, remove_admin_part=True)
        payload = build_admin_pre_srun_payload(
            repo_root=repo,
            source=source,
            job_id=job_id,
            selected_node=selected_node,
            env=env,
            runner=runner,
            hostname=hostname,
            cwd=cwd,
            user=user,
            generated_at_utc=generated_at_utc,
        )
        atomic_write_json(admin_part_path(repo), payload)
        return payload
    except Exception:
        remove_current_handoff(repo)
        raise


def _collect_gpu_runtime(
    *,
    env: Mapping[str, str],
    runner: Callable[[Sequence[str]], CommandResult],
    torch_probe: Callable[[], Mapping[str, Any]],
    hostname: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    nvidia_l = runner(("nvidia-smi", "-L"))
    compute_apps = runner(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    torch_payload = dict(torch_probe())
    return {
        "hostname": str(hostname or platform.node()),
        "slurm_job_id": str(env.get("SLURM_JOB_ID") or ""),
        "cuda_visible_devices": str(env.get("CUDA_VISIBLE_DEVICES") or ""),
        "cwd": str(cwd or os.getcwd()),
        "nvidia_smi_L_returncode": nvidia_l.returncode,
        "nvidia_smi_L_stdout": nvidia_l.stdout,
        "nvidia_smi_L_stderr": nvidia_l.stderr,
        "compute_apps_returncode": compute_apps.returncode,
        "compute_apps_stdout": compute_apps.stdout,
        "compute_apps_stderr": compute_apps.stderr,
        "torch_cuda_available": bool(torch_payload.get("torch_cuda_available")),
        "torch_cuda_device_count": int(torch_payload.get("torch_cuda_device_count") or 0),
        "torch_cuda_device_names": list(torch_payload.get("torch_cuda_device_names") or []),
        "torch_error": torch_payload.get("torch_error"),
    }


def build_gpu_post_srun_payload(
    *,
    repo_root: str | Path,
    source: str,
    env: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str]], CommandResult] | None = None,
    torch_probe: Callable[[], Mapping[str, Any]] | None = None,
    hostname: str | None = None,
    cwd: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    env_map = dict(os.environ if env is None else env)
    runner_fn = runner or (lambda args: _run_subprocess(args))
    torch_fn = torch_probe or _torch_probe
    admin_part = _read_json_object(admin_part_path(repo), label="admin pre-srun handoff")
    validate_admin_part_payload(admin_part)
    admin_tmux = dict(admin_part.get("admin_tmux") or {})
    gpu_runtime = _collect_gpu_runtime(
        env=env_map,
        runner=runner_fn,
        torch_probe=torch_fn,
        hostname=hostname,
        cwd=cwd,
    )
    selected_job_id = str(admin_part.get("selected_job_id") or "").strip()
    runtime_job_id = str(gpu_runtime.get("slurm_job_id") or "").strip()
    if selected_job_id and runtime_job_id and selected_job_id != runtime_job_id:
        raise HandoffError(
            "GPU runtime SLURM_JOB_ID does not match admin pre-srun selected job id.",
            details={"selected_job_id": selected_job_id, "slurm_job_id": runtime_job_id},
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "generated_at_utc": generated_at_utc or _utc_now(),
        "repo_root": str(repo),
        "valid_for_bridge_selection": True,
        "admin_tmux": admin_tmux,
        "gpu_runtime": gpu_runtime,
        "selection_key": {
            "socket": str(admin_tmux.get("socket") or ""),
            "pane": str(admin_tmux.get("pane") or admin_tmux.get("pane_id") or ""),
            "hostname": str(gpu_runtime.get("hostname") or ""),
            "slurm_job_id": str(gpu_runtime.get("slurm_job_id") or ""),
            "cuda_visible_devices": str(gpu_runtime.get("cuda_visible_devices") or ""),
        },
    }
    validate_current_gpu_pane_payload(payload)
    return payload


def write_gpu_post_srun(
    *,
    repo_root: str | Path,
    source: str,
    env: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str]], CommandResult] | None = None,
    torch_probe: Callable[[], Mapping[str, Any]] | None = None,
    hostname: str | None = None,
    cwd: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    try:
        remove_current_handoff(repo)
        payload = build_gpu_post_srun_payload(
            repo_root=repo,
            source=source,
            env=env,
            runner=runner,
            torch_probe=torch_probe,
            hostname=hostname,
            cwd=cwd,
            generated_at_utc=generated_at_utc,
        )
        atomic_write_json(current_handoff_path(repo), payload)
        return payload
    except Exception as exc:
        details = getattr(exc, "details", None)
        write_failed_handoff(
            repo_root=repo,
            mode="gpu-post-srun",
            source=source,
            error=str(exc),
            details=details if isinstance(details, Mapping) else {},
        )
        raise


def write_current_gpu_pane(
    *,
    repo_root: str | Path,
    source: str,
    env: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str]], CommandResult] | None = None,
    torch_probe: Callable[[], Mapping[str, Any]] | None = None,
    hostname: str | None = None,
    cwd: str | None = None,
    user: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    del user
    return write_gpu_post_srun(
        repo_root=repo_root,
        source=source,
        env=env,
        runner=runner,
        torch_probe=torch_probe,
        hostname=hostname,
        cwd=cwd,
        generated_at_utc=generated_at_utc,
    )
