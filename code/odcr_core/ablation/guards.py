"""Hard guards that keep ablations out of formal latest pointers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class AblationGuardError(RuntimeError):
    """Raised when an ablation would pollute the formal namespace."""


def is_ablation_run_id(run_id: str | None) -> bool:
    return str(run_id or "").strip().startswith("ablation_")


def ablation_manifest_path(repo_root: str | Path, *, task: int, run_id: str) -> Path:
    return (
        Path(repo_root).expanduser().resolve()
        / "runs"
        / "step5"
        / f"task{int(task)}"
        / str(run_id)
        / "meta"
        / "ablation_manifest.json"
    )


def _read_manifest_if_present(repo_root: str | Path, *, task: int, run_id: str) -> dict[str, Any] | None:
    path = ablation_manifest_path(repo_root, task=task, run_id=run_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AblationGuardError(f"ablation manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AblationGuardError(f"ablation manifest must be an object: {path}")
    return payload


def _manifest_marks_ablation(payload: Mapping[str, Any] | None) -> bool:
    if not payload:
        return False
    return bool(payload.get("is_ablation") is True or payload.get("forbidden_to_promote_latest") is True)


def assert_not_ablation_promotion_target(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
) -> None:
    stage_name = str(stage or "").strip().lower()
    if stage_name not in {"step5", "train_step5"}:
        return
    manifest = _read_manifest_if_present(repo_root, task=task, run_id=str(run_id))
    if is_ablation_run_id(str(run_id)) or _manifest_marks_ablation(manifest):
        raise AblationGuardError(
            "ablation Step5 runs are forbidden to promote to official latest.json; "
            f"stage=step5 task={int(task)} run_id={run_id}"
        )


def assert_not_ablation_latest_pointer(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
) -> None:
    stage_name = str(stage or "").strip().lower()
    if stage_name not in {"step5", "train_step5"}:
        return
    manifest = _read_manifest_if_present(repo_root, task=task, run_id=str(run_id))
    if is_ablation_run_id(str(run_id)) or _manifest_marks_ablation(manifest):
        raise AblationGuardError(
            "formal Step5 latest.json must not point to an ablation run; "
            f"task={int(task)} run_id={run_id}"
        )


def paper_table_gate(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    required_true = (
        "valid_complete",
        "test_complete",
        "paper_greedy_25",
        "task_local_rating_source",
        "paper_table_allowed",
    )
    missing = [field for field in required_true if snapshot.get(field) is not True]
    if snapshot.get("requires_manual_review") is not False:
        missing.append("requires_manual_review_false")
    return {
        "schema_version": "odcr_ablation_paper_table_gate/1",
        "eligible": not missing,
        "missing_requirements": missing,
        "policy": (
            "valid/test complete + paper_greedy_25 + task-local rating source + "
            "paper_table_allowed + manual review cleared"
        ),
    }
