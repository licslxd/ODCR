"""Cache boundary helpers."""

from __future__ import annotations

from pathlib import Path

from .path_policy import ArtifactPathPolicy


def cache_dir(stage: str, fingerprint: str, *, repo_root: str | Path | None = None) -> Path:
    return ArtifactPathPolicy(Path(repo_root).resolve() if repo_root else ArtifactPathPolicy().repo_root).cache_dir(stage, fingerprint)


def assert_not_data_or_merged(path: str | Path, *, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).resolve() if repo_root else ArtifactPathPolicy().repo_root.resolve()
    resolved = Path(path).resolve()
    for name in ("data", "merged"):
        try:
            resolved.relative_to((root / name).resolve())
            raise ValueError(f"cache/test path must not write {name}/: {resolved}")
        except ValueError as exc:
            if str(exc).startswith("cache/test path"):
                raise
    return resolved
