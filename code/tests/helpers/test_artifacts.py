from __future__ import annotations

import os
import shutil
from pathlib import Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_test_artifact_root(repo_root: str | Path | None = None) -> Path:
    """Return the repo-local or env-overridden root for test-owned artifacts."""

    explicit_root = repo_root is not None
    root = Path(repo_root).expanduser().resolve() if explicit_root else _default_repo_root()
    override = "" if explicit_root else os.environ.get("ODCR_TEST_ARTIFACT_ROOT", "").strip()
    artifact_root = Path(override).expanduser().resolve() if override else (root / "test_artifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    for child in ("runs_like", "tmp", "logs", "reports", "cache_like"):
        (artifact_root / child).mkdir(parents=True, exist_ok=True)
    return artifact_root


def assert_not_formal_runs_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    """Reject paths inside the real repo ``runs/`` tree."""

    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _default_repo_root()
    candidate = Path(path).expanduser().resolve()
    formal_runs = (root / "runs").resolve()
    try:
        candidate.relative_to(formal_runs)
    except ValueError:
        return candidate
    raise AssertionError(f"test artifact path must not target formal runs/: {candidate}")


def assert_path_under_test_artifacts(path: str | Path, repo_root: str | Path | None = None) -> Path:
    artifact_root = get_test_artifact_root(repo_root)
    candidate = assert_not_formal_runs_path(path, repo_root).resolve()
    try:
        candidate.relative_to(artifact_root)
    except ValueError as exc:
        raise AssertionError(f"expected test artifact under {artifact_root}, got {candidate}") from exc
    return candidate


def make_test_run_root(
    stage: str,
    task: int | str,
    run_id: str,
    repo_root: str | Path | None = None,
) -> Path:
    """Create a run-like directory under ``test_artifacts/runs_like``."""

    stage_name = str(stage).strip().lower().replace("/", "_")
    if not stage_name:
        raise ValueError("stage must be non-empty")
    task_name = f"task{int(task)}" if str(task).isdigit() else str(task).strip().replace("/", "_")
    if not task_name:
        raise ValueError("task must be non-empty")
    rid = str(run_id).strip().replace("/", "_")
    if not rid:
        raise ValueError("run_id must be non-empty")
    run_root = get_test_artifact_root(repo_root) / "runs_like" / stage_name / task_name / rid
    assert_not_formal_runs_path(run_root, repo_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "meta").mkdir(exist_ok=True)
    return run_root


def cleanup_test_artifacts(repo_root: str | Path | None = None, *, force: bool = False) -> None:
    """Remove repo-local test artifacts; env-overridden roots require ``force``."""

    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _default_repo_root()
    artifact_root = get_test_artifact_root(root)
    default_root = (root / "test_artifacts").resolve()
    if artifact_root != default_root and not force:
        raise ValueError("refusing to clean ODCR_TEST_ARTIFACT_ROOT without force=True")
    if artifact_root.exists():
        shutil.rmtree(artifact_root)


def explain_artifact_policy() -> str:
    return (
        "ODCR tests may write only tempfile/tmp_path or repo test_artifacts/. "
        "Run-like test files belong under test_artifacts/runs_like; tests must not "
        "write latest.json, stage_status.json, run_summary.json, logs, cache payloads, "
        "or data tables into formal runs/, data/, merged/, or cache/."
    )
