"""Formal latest-pointer guard helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class LatestPointerError(RuntimeError):
    pass


def validate_latest_payload(payload: Mapping[str, Any]) -> None:
    status = str(payload.get("status") or payload.get("run_status") or "").lower()
    if status in {"not_ready", "failed", "running"}:
        raise LatestPointerError(f"latest pointer cannot target {status!r} run")
    target = payload.get("run_summary") or payload.get("run_summary_path")
    if target is None and not (payload.get("latest_run_id") or payload.get("run_id")):
        raise LatestPointerError("latest pointer missing run identity")


def read_latest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LatestPointerError("latest pointer must be a JSON object")
    validate_latest_payload(data)
    return data
