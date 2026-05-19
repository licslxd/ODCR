"""Helpers for formal and test run-like directory layouts."""

from __future__ import annotations

from pathlib import Path

from .path_policy import ArtifactPathPolicy


def formal_run_dir(stage: str, task: int | str, run_id: str | int, *, repo_root: str | Path | None = None) -> Path:
    return ArtifactPathPolicy(Path(repo_root).resolve() if repo_root else ArtifactPathPolicy().repo_root).formal_run_dir(stage, task, run_id)


def test_run_like_dir(stage: str, task: int | str, case_id: str | int, *, repo_root: str | Path | None = None) -> Path:
    return ArtifactPathPolicy(Path(repo_root).resolve() if repo_root else ArtifactPathPolicy().repo_root).test_run_like_dir(stage, task, case_id)
