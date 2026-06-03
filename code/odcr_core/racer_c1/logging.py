"""Run-meta logging helpers for RACER-C1."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import RacerC1Paths


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_default(value: Any) -> str:
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


@dataclass
class RacerC1RunLogger:
    paths: RacerC1Paths
    log_interval_steps: int = 50

    def initialize(self) -> None:
        for directory in (
            self.paths.meta_dir,
            self.paths.evidence_dir,
            self.paths.train_dir,
            self.paths.predictions_dir,
            self.paths.metrics_dir,
            self.paths.diagnostics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for rel in ("console.log", "full.log", "errors.log", "epoch_metrics.jsonl", "resource_utilization.jsonl", "throughput.jsonl"):
            path = self.paths.meta_dir / rel
            if not path.exists():
                path.write_text("", encoding="utf-8")

    def console(self, message: str) -> None:
        self._write_line(self.paths.meta_dir / "console.log", message)
        self._write_line(self.paths.meta_dir / "full.log", message)

    def full(self, message: str) -> None:
        self._write_line(self.paths.meta_dir / "full.log", message)

    def error(self, message: str) -> None:
        self._write_line(self.paths.meta_dir / "errors.log", message)
        self._write_line(self.paths.meta_dir / "full.log", message)

    def epoch(self, payload: Mapping[str, Any]) -> None:
        append_jsonl(self.paths.meta_dir / "epoch_metrics.jsonl", {"timestamp": utc_now(), **dict(payload)})

    def resource(self, payload: Mapping[str, Any]) -> None:
        append_jsonl(self.paths.meta_dir / "resource_utilization.jsonl", {"timestamp": utc_now(), **dict(payload)})

    def throughput(self, payload: Mapping[str, Any]) -> None:
        append_jsonl(self.paths.meta_dir / "throughput.jsonl", {"timestamp": utc_now(), **dict(payload)})

    def _write_line(self, path: Path, message: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{utc_now()}] {message.rstrip()}\n")


class ThroughputTimer:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.last = self.start

    def lap(self) -> float:
        now = time.perf_counter()
        delta = now - self.last
        self.last = now
        return delta

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


def cpu_snapshot() -> dict[str, Any]:
    load1, load5, load15 = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
    return {
        "schema_version": "odcr_racer_c1_cpu_snapshot/1",
        "loadavg_1m": load1,
        "loadavg_5m": load5,
        "loadavg_15m": load15,
        "cpu_count": os.cpu_count(),
    }


def cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"schema_version": "odcr_racer_c1_cuda_snapshot/1", "available": False, "reason": f"torch_import_failed:{exc}"}
    available = bool(torch.cuda.is_available())
    out: dict[str, Any] = {
        "schema_version": "odcr_racer_c1_cuda_snapshot/1",
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "devices": [],
    }
    if available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            out["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "allocated_gb": round(torch.cuda.memory_allocated(idx) / (1024**3), 3),
                    "reserved_gb": round(torch.cuda.memory_reserved(idx) / (1024**3), 3),
                }
            )
    return out
