"""Strict latest.json -> meta/run_summary.json resolver."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_latest_summary_path(stage_task_dir: str | Path) -> Path:
    latest = Path(stage_task_dir) / "latest.json"
    if not latest.is_file():
        raise FileNotFoundError(f"missing latest.json: {latest}")
    payload = json.loads(latest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("latest_summary_path"):
        raise ValueError(f"latest.json must point to latest_summary_path: {latest}")
    summary = Path(str(payload["latest_summary_path"]))
    if not summary.is_absolute():
        summary = latest.parent.parent.parent / summary
    summary = summary.resolve()
    if summary.name != "run_summary.json" or summary.parent.name != "meta":
        raise ValueError(f"latest_summary_path must target meta/run_summary.json: {summary}")
    if not summary.is_file():
        raise FileNotFoundError(f"run_summary.json not found: {summary}")
    return summary

