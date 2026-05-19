"""Current-pane GPU handshake child implementation."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from odcr_core.aux.evidence.ai_analysis_writer import BUCKETS, get_writer


REQUIRED_EVIDENCE_FIELDS = (
    "hostname",
    "TMUX",
    "SLURM_JOB_ID",
    "CUDA_VISIBLE_DEVICES",
    "nvidia-smi",
    "torch.cuda.is_available",
    "torch.cuda.device_count",
    "torch.cuda.current_device",
    "torch.cuda.get_device_name",
    "torch.cuda.device_names",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_text(args: tuple[str, ...], *, timeout: int = 15) -> dict[str, Any]:
    try:
        proc = subprocess.run(list(args), text=True, capture_output=True, check=False, timeout=timeout)
        return {"argv": list(args), "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:
        return {"argv": list(args), "returncode": None, "stdout": "", "stderr": repr(exc)}


def collect_handshake(*, kind: str, require_cuda: bool, stage: str | None = None, task: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "odcr_runtime_gpu_handshake/1",
        "generated_at": _utc_now(),
        "kind": kind,
        "stage": stage,
        "task": task,
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "python": sys.executable,
        "TMUX": os.environ.get("TMUX", ""),
        "TMUX_PANE": os.environ.get("TMUX_PANE", ""),
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia-smi": _run_text(("nvidia-smi", "-L")),
        "nvidia-smi-compute-apps": _run_text(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        ),
    }
    torch_payload: dict[str, Any] = {
        "import_ok": False,
        "torch.cuda.is_available": False,
        "torch.cuda.device_count": 0,
        "torch.cuda.current_device": None,
        "torch.cuda.get_device_name": None,
        "torch.cuda.device_names": [],
        "error": None,
    }
    try:
        import torch

        torch_payload["import_ok"] = True
        available = bool(torch.cuda.is_available())
        torch_payload["torch.cuda.is_available"] = available
        count = int(torch.cuda.device_count())
        torch_payload["torch.cuda.device_count"] = count
        if available and count > 0:
            current = int(torch.cuda.current_device())
            torch_payload["torch.cuda.current_device"] = current
            torch_payload["torch.cuda.get_device_name"] = str(torch.cuda.get_device_name(current))
            torch_payload["torch.cuda.device_names"] = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    except Exception as exc:
        torch_payload["error"] = repr(exc)
    payload.update(torch_payload)
    payload["success"] = (not require_cuda) or (
        bool(payload["torch.cuda.is_available"]) and int(payload["torch.cuda.device_count"] or 0) > 0
    )
    if require_cuda and not payload["success"]:
        payload["error"] = (
            "Validated tmux pane did not expose CUDA during the bridge handshake. "
            "This is recorded as a bridge/pane/tooling failure, not as a current-shell CUDA blocker."
        )
    return payload


def write_handshake(
    *,
    kind: str,
    require_cuda: bool,
    status_path: str | Path,
    log_path: str | Path,
    report_path: str | Path,
    repo_root: str | Path | None = None,
    stage: str | None = None,
    task: str | None = None,
) -> int:
    payload = collect_handshake(kind=kind, require_cuda=require_cuda, stage=stage, task=task)
    writer = get_writer(repo_root)
    errors = [str(payload["error"])] if payload.get("error") else []
    validation_result = {
        "kind": kind,
        "stage": stage,
        "task": task,
        "success": bool(payload["success"]),
        "requires_cuda": bool(require_cuda),
    }

    def writer_name(path: str | Path, bucket: str) -> str:
        raw = Path(path).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (writer.repo_root / raw).resolve()
        bucket_dir = (writer.root / BUCKETS[bucket]).resolve()
        try:
            return resolved.relative_to(bucket_dir).as_posix()
        except ValueError:
            return raw.name

    lines = [
        f"schema_version={payload['schema_version']}",
        f"kind={kind}",
        f"hostname={payload['hostname']}",
        f"TMUX={payload['TMUX']}",
        f"SLURM_JOB_ID={payload['SLURM_JOB_ID']}",
        f"CUDA_VISIBLE_DEVICES={payload['CUDA_VISIBLE_DEVICES']}",
        f"nvidia-smi={json.dumps(payload['nvidia-smi'], sort_keys=True)}",
        f"nvidia-smi-compute-apps={json.dumps(payload['nvidia-smi-compute-apps'], sort_keys=True)}",
        f"torch.cuda.is_available={payload['torch.cuda.is_available']}",
        f"torch.cuda.device_count={payload['torch.cuda.device_count']}",
        f"torch.cuda.current_device={payload['torch.cuda.current_device']}",
        f"torch.cuda.get_device_name={payload['torch.cuda.get_device_name']}",
        f"torch.cuda.device_names={json.dumps(payload['torch.cuda.device_names'], sort_keys=True)}",
        f"success={payload['success']}",
    ]
    if payload.get("error"):
        lines.append(f"error={payload['error']}")
    writer.runtime_diagnostic(
        writer_name(status_path, "raw_log"),
        payload,
        source="gpu_handshake",
        stage=stage,
        task=task,
        validation_result=validation_result,
        errors=errors,
    )
    writer.raw_log(
        writer_name(log_path, "raw_log"),
        "\n".join(lines),
        source="gpu_handshake",
        stage=stage,
        task=task,
        validation_result=validation_result,
        errors=errors,
    )
    writer.final_report(
        writer_name(report_path, "final_report"),
        "# ODCR Runtime GPU Validation\n\n"
        f"- kind: {kind}\n"
        f"- stage: {stage or 'n/a'}\n"
        f"- task: {task or 'n/a'}\n"
        f"- hostname: {payload['hostname']}\n"
        f"- TMUX: {payload['TMUX'] or 'missing'}\n"
        f"- SLURM_JOB_ID: {payload['SLURM_JOB_ID'] or 'missing'}\n"
        f"- CUDA_VISIBLE_DEVICES: {payload['CUDA_VISIBLE_DEVICES'] or 'missing'}\n"
        f"- torch.cuda.is_available: {payload['torch.cuda.is_available']}\n"
        f"- torch.cuda.device_count: {payload['torch.cuda.device_count']}\n"
        f"- current device: {payload['torch.cuda.current_device']}\n"
        f"- device name: {payload['torch.cuda.get_device_name']}\n"
        f"- device names: {payload['torch.cuda.device_names']}\n"
        f"- result: {'PASS' if payload['success'] else 'FAIL'}\n",
        source="gpu_handshake",
        stage=stage,
        task=task,
        validation_result=validation_result,
        errors=errors,
    )
    return 0 if payload["success"] else 1
