"""Promotion helper for task-level active upstream pointers.

Promotion only changes the task-level pointer.  Historical run
``stage_status.json`` files stay immutable; whether a run is active is decided
by ``latest.json`` and the strict read-time verifier.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from odcr_core import path_layout, run_naming
from odcr_core.file_atomic import atomic_write_json
from odcr_core.upstream_resolver import resolve_latest, resolve_run, validate_upstream_eligibility


class StagePromotionError(RuntimeError):
    """Raised when a run cannot be promoted to active latest."""


def _repo_relative(repo_root: str | Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    root = Path(repo_root).expanduser().resolve()
    p = Path(raw).expanduser()
    p = (root / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def promote_upstream(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    stage_name = str(stage or "").strip().lower()
    rid = run_naming.parse_run_id(str(run_id))
    target = resolve_run(repo_root=root, stage=stage_name, task=int(task), run_id=rid, repair=True)
    consumer = {
        "step3": "step4",
        "step4": "step5",
        "step5": "eval",
    }.get(stage_name, "")
    target = validate_upstream_eligibility(
        target,
        repo_root=root,
        mode="formal",
        require_latest=False,
        consumer_stage=consumer,
    )
    current_run_id: str | None = None
    try:
        current = resolve_latest(repo_root=root, stage=stage_name, task=int(task), repair=True)
        current_run_id = current.run_id
    except Exception:
        current_run_id = None
    status = dict(target.stage_status)
    summary_rel = status.get("run_summary")
    if not summary_rel:
        raise StagePromotionError(f"cannot promote {stage_name} run {rid}: stage_status.run_summary is missing")
    run_dir = path_layout.get_stage_run_root(root, int(task), "v1", stage_name, rid).resolve()
    task_root = path_layout.get_stage_task_root(root, stage_name, int(task))
    latest_path = task_root / "latest.json"
    event_path = task_root / "promotion_event.json"
    history_path = task_root / "promotion_history.jsonl"
    promoted_at = _utc_now()
    payload = {
        "schema_version": "odcr_latest_pointer/active_stage_status/1",
        "active_run_id": rid,
        "latest_run_id": rid,
        "latest_run_dir": _repo_relative(root, run_dir),
        "latest_summary_path": summary_rel,
        "latest_stage_status_path": _repo_relative(root, run_dir / "meta" / "stage_status.json"),
        "promoted_at": promoted_at,
        "promoted_from": current_run_id,
        "promoted_to": rid,
        "promotion_event_path": _repo_relative(root, event_path),
        "supersedes": [current_run_id] if current_run_id and current_run_id != rid else [],
        "updated_by": "odcr_core.stage_promotion.promote_upstream.pointer_only",
    }
    event_payload = {
        "schema_version": "odcr_stage_promotion_event/1",
        "promoted_at": promoted_at,
        "stage": stage_name,
        "task": int(task),
        "promoted_from": current_run_id,
        "promoted_to": rid,
        "target_stage_status_path": _repo_relative(root, run_dir / "meta" / "stage_status.json"),
        "target_run_summary_path": summary_rel,
        "strict_validation": target.validation,
        "historical_stage_status_immutable": True,
        "failure_history_preserved": bool(status.get("failure_history_preserved")),
    }
    result = {
        "schema_version": "odcr_stage_promotion/1",
        "dry_run": bool(dry_run),
        "stage": stage_name,
        "task": int(task),
        "promote_run_id": rid,
        "previous_active_run_id": current_run_id,
        "would_write": [
            _repo_relative(root, latest_path),
            _repo_relative(root, event_path),
            _repo_relative(root, history_path),
        ],
        "latest_payload": payload,
        "promotion_event": event_payload,
        "target_stage_status": status,
        "superseded_previous": None,
        "historical_stage_status_immutable": True,
    }
    if dry_run:
        return result
    atomic_write_json(event_path, event_payload)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    atomic_write_json(latest_path, payload)
    return result


__all__ = ["StagePromotionError", "promote_upstream"]
