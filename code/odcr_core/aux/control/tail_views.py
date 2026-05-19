"""Tail-view policy helpers for new-layout logs."""

from __future__ import annotations

from pathlib import Path


def allowed_tail_log_names() -> tuple[str, ...]:
    return ("console.log", "full.log", "errors.log")


def assert_new_layout_log(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.name not in allowed_tail_log_names() or "meta" not in resolved.parts:
        raise ValueError("odcr tail is new-layout only: latest.json -> run_summary.json -> meta logs")
    return resolved
