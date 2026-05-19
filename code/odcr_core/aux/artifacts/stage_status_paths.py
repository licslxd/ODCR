"""Stage-status path helpers."""

from __future__ import annotations

from pathlib import Path


def stage_status_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "meta" / "stage_status.json"
