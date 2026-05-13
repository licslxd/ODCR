"""Canonical ODCR run-tree layout helpers."""

from __future__ import annotations

from pathlib import Path


def stage_task_dir(repo_root: str | Path, stage: str, task: int) -> Path:
    return Path(repo_root).resolve() / "runs" / str(stage) / f"task{int(task)}"


def run_meta_dir(repo_root: str | Path, stage: str, task: int, run_id: str) -> Path:
    return stage_task_dir(repo_root, stage, task) / str(run_id) / "meta"

