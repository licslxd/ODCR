"""Cache namespace policy helpers."""

from __future__ import annotations

from pathlib import Path


def cache_namespace(repo_root: str | Path, stage: str, task: int, name: str) -> Path:
    return Path(repo_root).resolve() / "cache" / str(stage) / f"task{int(task)}" / str(name)

