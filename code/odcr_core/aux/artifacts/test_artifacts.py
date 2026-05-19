"""Test artifact root helpers."""

from __future__ import annotations

from pathlib import Path

from .path_policy import ArtifactPathPolicy


def test_artifact_root(*, repo_root: str | Path | None = None) -> Path:
    return ArtifactPathPolicy(Path(repo_root).resolve() if repo_root else ArtifactPathPolicy().repo_root).test_artifact_root()


def assert_under_test_artifacts(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    return ArtifactPathPolicy(Path(repo_root).resolve() if repo_root else ArtifactPathPolicy().repo_root).assert_under_test_artifacts(path)
