"""Lightweight anti-forgery fixtures for the stage truth guardrail/tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from odcr_core.stage_promotion import promote_upstream
from odcr_core.stage_status import build_and_write_stage_status
from odcr_core.training_checkpoint import (
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    checkpoint_file_sha256,
    write_checkpoint_lineage,
)
from odcr_core.upstream_resolver import UpstreamResolutionError, resolve_upstream


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_step3_fixture(
    repo: Path,
    *,
    task: int,
    run_id: str,
    active: bool = False,
    eligible: bool = True,
    source: str | None = None,
    target: str | None = None,
    latest_status: str | None = None,
    quality_downstream_ready: bool = False,
) -> Path:
    source = source or ("AM_Movies" if task == 2 else "AM_CDs")
    target = target or ("AM_CDs" if task == 2 else "AM_Movies")
    run = repo / "runs" / "step3" / f"task{int(task)}" / str(run_id)
    meta = run / "meta"
    model = run / "model"
    meta.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    checkpoint = model / "best_observed.pth"
    checkpoint.write_bytes(f"checkpoint-{task}-{run_id}".encode("utf-8"))
    write_checkpoint_lineage(
        checkpoint,
        {
            "sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
            "stage": "step3",
            "run_id": str(run_id),
            "task_id": int(task),
            "source_domain": source,
            "target_domain": target,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_file_hash": checkpoint_file_sha256(checkpoint),
            "reason": "test_fixture",
            "replaced_previous": False,
            "selection_scope": "best_observed",
            "checkpoint_epoch": 1,
            "selection_metric": "valid_loss",
            "selection_metric_value": 1.0,
            "selection_direction": "min",
        },
    )
    status = "completed_with_eval_handoff" if eligible else "quality_blocked"
    write_json(
        meta / "resolved_config.json",
        {"task": {"id": int(task), "source": source, "target": target}, "run": {"stage_run_dir": str(run)}},
    )
    write_json(meta / "source_table.json", {"source_table_schema_version": "1.0", "view": "formal", "records": []})
    write_json(
        meta / "run_summary.json",
        {
            "run_summary_schema_version": "1.0",
            "run_id": str(run_id),
            "stage": "step3",
            "task_id": int(task),
            "source_domain": source,
            "target_domain": target,
            "status": status,
            "run_dir": f"runs/step3/task{int(task)}/{run_id}",
            "meta_dir": f"runs/step3/task{int(task)}/{run_id}/meta",
            "train_status": "completed" if eligible else "failed",
            "paper_eval_status": "completed" if eligible else "not_started",
            "downstream_ready": bool(eligible),
            "selected_checkpoint": f"runs/step3/task{int(task)}/{run_id}/model/best_observed.pth",
            "selected_checkpoint_hash": checkpoint_file_sha256(checkpoint),
            "selected_downstream_checkpoint": f"runs/step3/task{int(task)}/{run_id}/model/best_observed.pth",
            "selected_downstream_checkpoint_hash": checkpoint_file_sha256(checkpoint),
            "failure_history": [{"status": "failed", "source": "fixture"}] if eligible else [],
        },
    )
    if eligible:
        write_json(
            meta / "eval_handoff.json",
            {
                "schema_version": "odcr_step3_eval_handoff/1",
                "task_id": int(task),
                "run_id": str(run_id),
                "checkpoint_path": f"runs/step3/task{int(task)}/{run_id}/model/best_observed.pth",
                "checkpoint_hash": checkpoint_file_sha256(checkpoint),
                "train_status": "completed",
                "paper_eval_status": "completed",
                "paper_eval_protocol": "paper_target_only_eval",
                "old_failure_history_preserved": True,
            },
        )
    write_json(
        meta / "quality_audit.json",
        {
            "schema_version": "odcr_step3_quality_audit/1",
            "quality_status": "pass" if quality_downstream_ready else "blocked",
            "downstream_ready": bool(quality_downstream_ready),
            "quality_block_reasons": [] if quality_downstream_ready else ["diagnostic_eval_collapse"],
            "selected_downstream_checkpoint": str(checkpoint.resolve()),
            "selected_downstream_checkpoint_hash": checkpoint_file_sha256(checkpoint),
        },
    )
    build_and_write_stage_status(repo_root=repo, stage="step3", task=int(task), run_id=str(run_id))
    if active:
        write_json(
            repo / "runs" / "step3" / f"task{int(task)}" / "latest.json",
            {
                "schema_version": "odcr_latest_pointer/active_stage_status/1",
                "active_run_id": str(run_id),
                "latest_run_id": str(run_id),
                "latest_run_dir": f"runs/step3/task{int(task)}/{run_id}",
                "latest_summary_path": f"runs/step3/task{int(task)}/{run_id}/meta/run_summary.json",
                "latest_stage_status_path": f"runs/step3/task{int(task)}/{run_id}/meta/stage_status.json",
                **({"latest_status": latest_status} if latest_status is not None else {}),
            },
        )
    return run


def mutate_status(repo: Path, *, task: int, run_id: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    path = repo / "runs" / "step3" / f"task{int(task)}" / str(run_id) / "meta" / "stage_status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    write_json(path, payload)
    return payload


def expect_reject(label: str, func: Callable[[], Any]) -> tuple[bool, str]:
    try:
        func()
    except (UpstreamResolutionError, Exception) as exc:
        return True, str(exc)
    return False, f"{label} unexpectedly passed"


def run_antiforgery_selftest() -> dict[str, Any]:
    results: dict[str, Any] = {"schema_version": "odcr_stage_truth_antiforgery_selftest/1"}
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        write_step3_fixture(repo, task=2, run_id="1", active=False, eligible=False)
        write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True, latest_status="failed")
        write_step3_fixture(repo, task=2, run_id="3", active=False, eligible=True)
        minimal = repo / "runs" / "step3" / "task2" / "3" / "meta" / "stage_status.json"
        write_json(
            minimal,
            {
                "schema_version": "odcr_stage_status/1",
                "stage": "step3",
                "task": 2,
                "task_id": 2,
                "run_id": "3",
                "run_dir": "runs/step3/task2/3",
                "final_status": "completed_with_eval_handoff",
                "downstream_ready": True,
                "ready_for": ["step4"],
                "artifacts": {},
            },
        )
        ok, msg = expect_reject(
            "forged minimal status",
            lambda: resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="3", consumer_stage="step4"),
        )
        results["forged_status_rejected"] = ok
        results["forged_status_message"] = msg
        ok, msg = expect_reject(
            "run1 alias gate",
            lambda: resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="1", consumer_stage="step4"),
        )
        results["alias_run1_rejected"] = ok
        results["alias_run1_message"] = msg
        run2 = repo / "runs" / "step3" / "task2" / "2"
        resolved = resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4")
        results["latest_pointer_only_passed"] = resolved.run_id == "2"
        results["latest_warnings"] = (resolved.validation or {}).get("latest_warnings", [])
        missing_eval = repo / "runs" / "step3" / "task2" / "4"
        write_step3_fixture(repo, task=2, run_id="4", active=False, eligible=True)
        (missing_eval / "meta" / "eval_handoff.json").unlink()
        ok, msg = expect_reject(
            "missing artifact",
            lambda: resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="4", consumer_stage="step4"),
        )
        results["missing_artifact_rejected"] = ok
        results["missing_artifact_message"] = msg
        write_step3_fixture(repo, task=2, run_id="5", active=False, eligible=True)
        mutate_status(repo, task=2, run_id="5", mutate=lambda payload: payload.__setitem__("selected_checkpoint_hash", "0" * 64))
        ok, msg = expect_reject(
            "hash mismatch",
            lambda: resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="5", consumer_stage="step4"),
        )
        results["hash_mismatch_rejected"] = ok
        results["hash_mismatch_message"] = msg
        write_step3_fixture(repo, task=2, run_id="6", active=False, eligible=True)
        ckpt6 = repo / "runs" / "step3" / "task2" / "6" / "model" / "best_observed.pth"
        ckpt6.unlink()
        ok, msg = expect_reject(
            "stale exists",
            lambda: resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="6", consumer_stage="step4"),
        )
        results["stale_exists_rejected"] = ok
        results["stale_exists_message"] = msg
        ok, msg = expect_reject(
            "malformed promotion target",
            lambda: promote_upstream(repo_root=repo, stage="step3", task=2, run_id="3", dry_run=True),
        )
        results["promotion_malformed_target_rejected"] = ok
        results["promotion_malformed_target_message"] = msg
        promotion = promote_upstream(repo_root=repo, stage="step3", task=2, run_id="2", dry_run=True)
        results["promotion_valid_dry_run_passed"] = promotion.get("dry_run") is True
        results["quality_audit_cannot_override_stage_status"] = bool(resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4"))
        _ = run2
    required = (
        "forged_status_rejected",
        "missing_artifact_rejected",
        "hash_mismatch_rejected",
        "stale_exists_rejected",
        "promotion_malformed_target_rejected",
        "latest_pointer_only_passed",
        "alias_run1_rejected",
        "quality_audit_cannot_override_stage_status",
    )
    results["passed"] = all(bool(results.get(key)) for key in required)
    return results


__all__ = [
    "mutate_status",
    "run_antiforgery_selftest",
    "write_json",
    "write_step3_fixture",
]
