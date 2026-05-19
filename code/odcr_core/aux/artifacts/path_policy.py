"""Central path policy for aux/runtime/test artifacts."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PathPolicyError(ValueError):
    """Raised when a path would violate ODCR artifact boundaries."""


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[4]


def repo_relative_path(repo_root: str | Path, value: str | Path | None) -> str | None:
    if value is None:
        return None
    root = Path(repo_root).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_json_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_part(value: str | int, *, field: str) -> str:
    text = str(value).strip()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise PathPolicyError(f"unsafe {field}: {value!r}")
    return text


@dataclass(frozen=True)
class ArtifactPathPolicy:
    repo_root: Path = repo_root_from_here()

    def _root(self) -> Path:
        return Path(self.repo_root).resolve()

    def formal_run_dir(self, stage: str, task: int | str, run_id: str | int) -> Path:
        stage_part = _safe_part(stage, field="stage")
        task_part = _safe_part(task, field="task")
        run_part = _safe_part(run_id, field="run_id")
        return self._root() / "runs" / stage_part / f"task{task_part}" / run_part

    def test_run_like_dir(self, stage: str, task: int | str, case_id: str | int) -> Path:
        stage_part = _safe_part(stage, field="stage")
        task_part = _safe_part(task, field="task")
        case_part = _safe_part(case_id, field="case_id")
        return self.test_artifact_root() / "runs_like" / stage_part / f"task{task_part}" / case_part

    def test_tmp_dir(self, case_id: str | int) -> Path:
        return self.test_artifact_root() / "tmp" / _safe_part(case_id, field="case_id")

    def ai_analysis_dir(self, bucket: str) -> Path:
        bucket_part = _safe_part(bucket, field="bucket")
        return self._root() / "AI_analysis" / bucket_part

    def validation_dir(self, name: str) -> Path:
        return self.ai_analysis_dir("05_final_reports") / _safe_part(name, field="validation")

    def raw_log_path(self, name: str) -> Path:
        return self.ai_analysis_dir("01_raw_logs") / _safe_part(name, field="raw_log")

    def cache_dir(self, stage: str, fingerprint: str) -> Path:
        return self._root() / "cache" / _safe_part(stage, field="stage") / _safe_part(fingerprint, field="fingerprint")

    def test_artifact_root(self) -> Path:
        override = os.environ.get("ODCR_TEST_ARTIFACT_ROOT", "").strip()
        if self._root() != repo_root_from_here().resolve():
            return self._root() / "test_artifacts"
        if not override:
            return self._root() / "test_artifacts"
        path = Path(override).expanduser().resolve()
        try:
            path.relative_to(self._root().resolve())
        except ValueError:
            raise PathPolicyError("ODCR_TEST_ARTIFACT_ROOT must stay inside the repo") from None
        return path

    def assert_under_test_artifacts(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        root = self.test_artifact_root().resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise PathPolicyError(f"test artifact path must stay under {root}: {resolved}") from None
        return resolved

    def assert_not_formal_for_validation(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        formal_root = (self._root() / "runs").resolve()
        try:
            resolved.relative_to(formal_root)
        except ValueError:
            return resolved
        raise PathPolicyError(f"validation/probe output must not write formal runs: {resolved}")
