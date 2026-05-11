"""Canonical per-run stage status for downstream handoff decisions.

The live truth for a completed run is ``meta/stage_status.json``.  Historical
diagnostics such as Step3 ``quality_audit.json`` may inform a negative status,
but downstream stages must consume this module's payload rather than reading
those diagnostics directly.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from odcr_core import path_layout, run_naming
from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import ODCR_ROUTING_TRAIN_CSV
from odcr_core.step4_export_validator import (
    STEP4_EXPORT_MANIFEST,
    validate_step4_export_ready,
)
from odcr_core.training_checkpoint import CheckpointLineageError, checkpoint_file_sha256, read_checkpoint_lineage


STAGE_STATUS_SCHEMA_VERSION = "odcr_stage_status/1"
STAGE_STATUS_VALIDATOR_VERSION = "odcr_stage_status_validator/2"
QUALITY_AUDIT_SUPERSEDED_SCHEMA_VERSION = "odcr_quality_audit_superseded_by/1"

READY_FOR_BY_STAGE: dict[str, tuple[str, ...]] = {
    "step3": ("step4",),
    "step4": ("step5",),
    "step5": ("eval", "rerank"),
}

BAD_FINAL_STATUSES = {
    "failed",
    "running",
    "partial",
    "interrupted",
    "quality_blocked",
    "superseded",
    "missing_run_summary",
    "missing_run_dir",
    "not_ready",
}

OK_RUN_SUMMARY_STATUSES = {"ok", "completed", "success", "completed_with_eval_handoff", "eval_handoff_accepted"}
BAD_RUN_SUMMARY_STATUSES = {"failed", "running", "partial", "interrupted"}


class StageStatusError(RuntimeError):
    """Raised when a stage status payload is missing or malformed."""


def _canonical_stage(stage: str) -> str:
    raw = str(stage or "").strip().lower()
    return {
        "train_step3": "step3",
        "train_step4": "step4",
        "train_step5": "step5",
        "eval-rerank": "rerank",
    }.get(raw, raw)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_relative(repo_root: str | Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    root = Path(repo_root).expanduser().resolve()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _repo_path(repo_root: str | Path, raw: Any) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    root = Path(repo_root).expanduser().resolve()
    p = Path(text).expanduser()
    return (root / p).resolve() if not p.is_absolute() else p.resolve()


def _load_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise StageStatusError(f"required JSON file missing: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageStatusError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageStatusError(f"JSON root must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_status(repo_root: Path, rel_or_abs: str | None) -> dict[str, Any]:
    path = _repo_path(repo_root, rel_or_abs)
    if path is None:
        return {"path": rel_or_abs, "exists": False}
    return {
        "path": _repo_relative(repo_root, path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "sha256": _file_sha256(path) if path.is_file() else None,
    }


def stage_status_path(run_root: str | Path) -> Path:
    return Path(run_root).expanduser().resolve() / "meta" / "stage_status.json"


def read_stage_status(run_root: str | Path, *, required: bool = True) -> dict[str, Any]:
    path = stage_status_path(run_root)
    payload = _load_json(path, required=required)
    if not payload:
        return {}
    if payload.get("schema_version") != STAGE_STATUS_SCHEMA_VERSION:
        raise StageStatusError(
            f"unsupported stage_status schema {payload.get('schema_version')!r}: {path}"
        )
    return payload


def write_stage_status(repo_root: str | Path, payload: Mapping[str, Any]) -> Path:
    root = Path(repo_root).expanduser().resolve()
    run_dir = _repo_path(root, payload.get("run_dir"))
    if run_dir is None:
        stage = _canonical_stage(str(payload.get("stage") or ""))
        task = int(payload.get("task") or payload.get("task_id") or 0)
        run_id = run_naming.parse_run_id(str(payload.get("run_id") or ""))
        run_dir = path_layout.get_stage_run_root(root, task, "v1", stage, run_id).resolve()
    out = stage_status_path(run_dir)
    return atomic_write_json(out, dict(payload))


def _checkpoint_lineage_path(run_root: Path, checkpoint: Path | None) -> Path | None:
    state_lineage = run_root / "state" / "checkpoint_lineage.json"
    if state_lineage.is_file():
        return state_lineage
    if checkpoint is not None:
        sidecar = Path(str(checkpoint) + ".lineage.json")
        if sidecar.is_file():
            return sidecar
    return None


def _checkpoint_valid(stage: str, checkpoint: Path | None) -> tuple[bool, str | None, dict[str, Any]]:
    if checkpoint is None:
        return False, "selected_checkpoint_missing_from_status", {}
    if not checkpoint.is_file():
        return False, f"selected_checkpoint_missing: {checkpoint}", {}
    try:
        lineage = read_checkpoint_lineage(checkpoint, expected_stage=stage)
    except CheckpointLineageError as exc:
        return False, f"checkpoint_lineage_invalid: {exc}", {}
    expected_hash = str(lineage.get("checkpoint_file_hash") or "")
    if expected_hash:
        actual_hash = checkpoint_file_sha256(checkpoint)
        if expected_hash != actual_hash:
            return False, f"checkpoint_hash_mismatch: sidecar={expected_hash!r} actual={actual_hash!r}", lineage
    return True, None, lineage


def _selected_step3_checkpoint(repo_root: Path, run_root: Path, run_summary: Mapping[str, Any], handoff: Mapping[str, Any]) -> Path | None:
    for raw in (
        handoff.get("checkpoint_path"),
        run_summary.get("selected_downstream_checkpoint"),
        run_summary.get("selected_checkpoint"),
        run_root / "model" / "best_observed.pth",
        path_layout.best_model_path(run_root),
    ):
        path = _repo_path(repo_root, raw)
        if path is not None and path.is_file():
            return path
    return None


def _step3_status(
    *,
    repo_root: Path,
    run_root: Path,
    run_summary: Mapping[str, Any],
    quality_audit: Mapping[str, Any],
    eval_handoff: Mapping[str, Any],
) -> dict[str, Any]:
    summary_status = str(run_summary.get("status") or "").strip().lower()
    handoff_ready = (
        str(eval_handoff.get("schema_version") or "") == "odcr_step3_eval_handoff/1"
        and str(eval_handoff.get("train_status") or "") == "completed"
        and str(eval_handoff.get("paper_eval_status") or "") == "completed"
        and str(eval_handoff.get("paper_eval_protocol") or "") == "paper_target_only_eval"
        and bool(eval_handoff.get("old_failure_history_preserved")) is True
    )
    checkpoint = _selected_step3_checkpoint(repo_root, run_root, run_summary, eval_handoff)
    checkpoint_ok, checkpoint_error, checkpoint_lineage = _checkpoint_valid("step3", checkpoint)
    if handoff_ready:
        final_status = "completed_with_eval_handoff"
        downstream_ready = checkpoint_ok
        ready_for = ["step4"] if downstream_ready else []
        status_source = "eval_handoff"
        reasons = [] if checkpoint_ok else [checkpoint_error or "checkpoint_not_valid"]
        supersedes = ["quality_audit.json", "post_train_eval_failed_status"]
    elif quality_audit and (
        str(quality_audit.get("quality_status") or "").strip().lower() == "blocked"
        or quality_audit.get("downstream_ready") is False
    ):
        final_status = "quality_blocked"
        downstream_ready = False
        ready_for = []
        status_source = "stage_status_from_quality_audit_negative"
        reasons = [str(x) for x in (quality_audit.get("quality_block_reasons") or [])]
        if not reasons:
            reasons = ["quality_audit_downstream_ready_false"]
        supersedes = []
    elif summary_status in BAD_RUN_SUMMARY_STATUSES:
        final_status = summary_status
        downstream_ready = False
        ready_for = []
        status_source = "run_summary"
        reasons = [str(run_summary.get("failure_phase") or run_summary.get("latest_error") or summary_status)]
        supersedes = []
    elif summary_status in OK_RUN_SUMMARY_STATUSES and run_summary.get("downstream_ready") is True:
        final_status = summary_status or "completed"
        downstream_ready = checkpoint_ok
        ready_for = ["step4"] if downstream_ready else []
        status_source = "run_summary"
        reasons = [] if checkpoint_ok else [checkpoint_error or "checkpoint_not_valid"]
        supersedes = []
    else:
        final_status = "not_ready" if summary_status else "missing_run_summary"
        downstream_ready = False
        ready_for = []
        status_source = "run_summary" if summary_status else "missing_run_summary"
        reasons = [f"run_summary.status={summary_status or '(missing)'}"]
        supersedes = []
    return {
        "final_status": final_status,
        "downstream_ready": bool(downstream_ready),
        "ready_for": ready_for,
        "status_source": status_source,
        "rejection_reasons": [r for r in reasons if r],
        "selected_checkpoint": _repo_relative(repo_root, checkpoint) if checkpoint else None,
        "selected_checkpoint_hash": checkpoint_file_sha256(checkpoint) if checkpoint and checkpoint.is_file() else None,
        "checkpoint_lineage": _repo_relative(repo_root, _checkpoint_lineage_path(run_root, checkpoint)),
        "checkpoint_lineage_schema": checkpoint_lineage.get("sidecar_schema_version") if checkpoint_lineage else None,
        "supersedes": supersedes,
        "failure_history_preserved": bool(run_summary.get("failure_history")) or bool(eval_handoff.get("old_failure_history_preserved")),
        "do_not_use_quality_audit_as_final_truth": True,
        "quality_audit_status": quality_audit.get("quality_status") if quality_audit else None,
        "quality_audit_downstream_ready": quality_audit.get("downstream_ready") if quality_audit else None,
    }


def _step4_status(*, repo_root: Path, run_root: Path, run_summary: Mapping[str, Any]) -> dict[str, Any]:
    summary_status = str(run_summary.get("status") or "").strip().lower()
    train_csv = run_root / ODCR_ROUTING_TRAIN_CSV
    validation = validate_step4_export_ready(run_root, repo_root=repo_root)
    if summary_status in OK_RUN_SUMMARY_STATUSES and validation.ready:
        final_status = "completed"
        downstream_ready = True
        ready_for = ["step5"]
        reasons: list[str] = []
    elif summary_status in BAD_RUN_SUMMARY_STATUSES:
        final_status = summary_status
        downstream_ready = False
        ready_for = []
        reasons = [str(run_summary.get("latest_error") or summary_status)]
    else:
        final_status = "not_ready" if summary_status else "missing_run_summary"
        downstream_ready = False
        ready_for = []
        reasons = validation.errors or [f"routing_train_csv_missing: {_repo_relative(repo_root, train_csv)}"]
    return {
        "final_status": final_status,
        "downstream_ready": downstream_ready,
        "ready_for": ready_for,
        "status_source": "step4_export_readiness_validator",
        "rejection_reasons": reasons,
        "selected_export": _repo_relative(repo_root, train_csv),
        "export_manifest": _repo_relative(repo_root, run_root / STEP4_EXPORT_MANIFEST),
        "index_contract": _repo_relative(repo_root, run_root / "index_contract.json"),
        "export_readiness": validation.to_payload(repo_root),
        "selected_checkpoint": None,
        "checkpoint_lineage": None,
        "supersedes": [],
        "failure_history_preserved": bool(run_summary.get("failure_history")),
        "do_not_use_quality_audit_as_final_truth": True,
    }


def _step5_status(*, repo_root: Path, run_root: Path, run_summary: Mapping[str, Any]) -> dict[str, Any]:
    summary_status = str(run_summary.get("status") or "").strip().lower()
    checkpoint = path_layout.best_model_path(run_root)
    checkpoint_ok, checkpoint_error, checkpoint_lineage = _checkpoint_valid("step5", checkpoint)
    if summary_status in OK_RUN_SUMMARY_STATUSES and checkpoint_ok:
        final_status = "completed"
        downstream_ready = True
        ready_for = ["eval", "rerank"]
        reasons: list[str] = []
    elif summary_status in BAD_RUN_SUMMARY_STATUSES:
        final_status = summary_status
        downstream_ready = False
        ready_for = []
        reasons = [str(run_summary.get("latest_error") or summary_status)]
    else:
        final_status = "not_ready" if summary_status else "missing_run_summary"
        downstream_ready = False
        ready_for = []
        reasons = [checkpoint_error or "checkpoint_not_valid"]
    return {
        "final_status": final_status,
        "downstream_ready": downstream_ready,
        "ready_for": ready_for,
        "status_source": "run_summary",
        "rejection_reasons": reasons,
        "selected_checkpoint": _repo_relative(repo_root, checkpoint),
        "selected_checkpoint_hash": checkpoint_file_sha256(checkpoint) if checkpoint.is_file() else None,
        "checkpoint_lineage": _repo_relative(repo_root, _checkpoint_lineage_path(run_root, checkpoint)),
        "checkpoint_lineage_schema": checkpoint_lineage.get("sidecar_schema_version") if checkpoint_lineage else None,
        "supersedes": [],
        "failure_history_preserved": bool(run_summary.get("failure_history")),
        "do_not_use_quality_audit_as_final_truth": True,
    }


def build_stage_status(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    stage_name = _canonical_stage(stage)
    rid = run_naming.parse_run_id(str(run_id))
    run_root = path_layout.get_stage_run_root(root, int(task), "v1", stage_name, rid).resolve()
    meta = run_root / "meta"
    summary_path = meta / "run_summary.json"
    run_summary = _load_json(summary_path, required=False)
    quality_audit = _load_json(meta / "quality_audit.json", required=False)
    eval_handoff = _load_json(meta / "eval_handoff.json", required=False)
    if not run_root.is_dir():
        stage_payload = {
            "final_status": "missing_run_dir",
            "downstream_ready": False,
            "ready_for": [],
            "status_source": "missing_run_dir",
            "rejection_reasons": [f"run_dir_missing: {run_root}"],
            "selected_checkpoint": None,
            "checkpoint_lineage": None,
            "supersedes": [],
            "failure_history_preserved": False,
            "do_not_use_quality_audit_as_final_truth": True,
        }
    elif stage_name == "step3":
        stage_payload = _step3_status(
            repo_root=root,
            run_root=run_root,
            run_summary=run_summary,
            quality_audit=quality_audit,
            eval_handoff=eval_handoff,
        )
    elif stage_name == "step4":
        stage_payload = _step4_status(repo_root=root, run_root=run_root, run_summary=run_summary)
    elif stage_name == "step5":
        stage_payload = _step5_status(repo_root=root, run_root=run_root, run_summary=run_summary)
    else:
        summary_status = str(run_summary.get("status") or "").strip().lower()
        stage_payload = {
            "final_status": summary_status or "missing_run_summary",
            "downstream_ready": False,
            "ready_for": [],
            "status_source": "run_summary",
            "rejection_reasons": [],
            "selected_checkpoint": None,
            "checkpoint_lineage": None,
            "supersedes": [],
            "failure_history_preserved": bool(run_summary.get("failure_history")),
            "do_not_use_quality_audit_as_final_truth": True,
        }

    source_table = meta / "source_table.json"
    resolved_config = meta / "resolved_config.json"
    now = _utc_now()
    required_artifacts = ["run_summary", "source_table", "resolved_config"]
    if stage_name == "step3":
        required_artifacts.extend(["eval_handoff", "selected_checkpoint", "checkpoint_lineage"])
    elif stage_name == "step4":
        required_artifacts.extend(["selected_export", "export_manifest", "index_contract"])
    elif stage_name == "step5":
        required_artifacts.extend(["selected_checkpoint", "checkpoint_lineage"])
    payload: dict[str, Any] = {
        "schema_version": STAGE_STATUS_SCHEMA_VERSION,
        "validator_version": STAGE_STATUS_VALIDATOR_VERSION,
        "generated_at": now,
        "updated_at": now,
        "generated_at_utc": now,
        "stage": stage_name,
        "task": int(task),
        "task_id": int(task),
        "run_id": rid,
        "run_dir": _repo_relative(root, run_root),
        "final_status": stage_payload["final_status"],
        "downstream_ready": bool(stage_payload["downstream_ready"]),
        "ready_for": list(stage_payload["ready_for"]),
        "selected_checkpoint": stage_payload.get("selected_checkpoint"),
        "selected_checkpoint_hash": stage_payload.get("selected_checkpoint_hash"),
        "selected_export": stage_payload.get("selected_export"),
        "export_manifest": stage_payload.get("export_manifest"),
        "index_contract": stage_payload.get("index_contract"),
        "export_readiness": stage_payload.get("export_readiness"),
        "eval_handoff": _repo_relative(root, meta / "eval_handoff.json") if (meta / "eval_handoff.json").is_file() else None,
        "run_summary": _repo_relative(root, summary_path) if summary_path.is_file() else None,
        "checkpoint_lineage": stage_payload.get("checkpoint_lineage"),
        "source_table": _repo_relative(root, source_table) if source_table.is_file() else None,
        "resolved_config": _repo_relative(root, resolved_config) if resolved_config.is_file() else None,
        "status_source": stage_payload["status_source"],
        "rejection_reasons": list(stage_payload.get("rejection_reasons") or []),
        "supersedes": list(stage_payload.get("supersedes") or []),
        "failure_history_preserved": bool(stage_payload.get("failure_history_preserved")),
        "do_not_use_quality_audit_as_final_truth": bool(stage_payload.get("do_not_use_quality_audit_as_final_truth", True)),
        "required_artifacts": required_artifacts,
        "quality_audit": _repo_relative(root, meta / "quality_audit.json") if (meta / "quality_audit.json").is_file() else None,
        "quality_audit_status": stage_payload.get("quality_audit_status"),
        "quality_audit_downstream_ready": stage_payload.get("quality_audit_downstream_ready"),
        "artifacts": {
            "run_summary": _artifact_status(root, _repo_relative(root, summary_path)),
            "eval_handoff": _artifact_status(root, _repo_relative(root, meta / "eval_handoff.json")) if stage_name == "step3" else None,
            "selected_checkpoint": _artifact_status(root, stage_payload.get("selected_checkpoint")),
            "selected_export": _artifact_status(root, stage_payload.get("selected_export")),
            "export_manifest": _artifact_status(root, stage_payload.get("export_manifest")),
            "index_contract": _artifact_status(root, stage_payload.get("index_contract")),
            "checkpoint_lineage": _artifact_status(root, stage_payload.get("checkpoint_lineage")),
            "source_table": _artifact_status(root, _repo_relative(root, source_table)),
            "resolved_config": _artifact_status(root, _repo_relative(root, resolved_config)),
        },
        "upstream": {},
    }
    if run_summary.get("from_step3"):
        payload["upstream"]["from_step3"] = str(run_summary.get("from_step3"))
    elif stage_name == "step4" and "_" in rid:
        payload["upstream"]["from_step3"] = rid.split("_", 1)[0]
    if run_summary.get("from_step4"):
        payload["upstream"]["from_step4"] = str(run_summary.get("from_step4"))
    return payload


def build_and_write_stage_status(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
    write_quality_audit_sidecar: bool = True,
) -> dict[str, Any]:
    run_root = path_layout.get_stage_run_root(
        Path(repo_root).expanduser().resolve(),
        int(task),
        "v1",
        _canonical_stage(stage),
        run_naming.parse_run_id(str(run_id)),
    ).resolve()
    existing = read_stage_status(run_root, required=False)
    if str(existing.get("final_status") or "").strip().lower() == "superseded":
        return existing
    payload = build_stage_status(repo_root=repo_root, stage=stage, task=int(task), run_id=str(run_id))
    out = write_stage_status(repo_root, payload)
    payload["stage_status"] = _repo_relative(repo_root, out)
    if (
        write_quality_audit_sidecar
        and payload.get("stage") == "step3"
        and payload.get("quality_audit")
        and "quality_audit.json" in payload.get("supersedes", [])
    ):
        sidecar = write_quality_audit_superseded_sidecar(repo_root=repo_root, stage_status=payload)
        payload["quality_audit_superseded_sidecar"] = _repo_relative(repo_root, sidecar)
        write_stage_status(repo_root, payload)
    return payload


def write_quality_audit_superseded_sidecar(*, repo_root: str | Path, stage_status: Mapping[str, Any]) -> Path:
    root = Path(repo_root).expanduser().resolve()
    run_dir = _repo_path(root, stage_status.get("run_dir"))
    if run_dir is None:
        raise StageStatusError("stage_status.run_dir is required for quality audit sidecar")
    meta = run_dir / "meta"
    audit = meta / "quality_audit.json"
    if not audit.is_file():
        raise StageStatusError(f"quality_audit.json missing: {audit}")
    payload = {
        "schema_version": QUALITY_AUDIT_SUPERSEDED_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "superseded_by": [
            _repo_relative(root, meta / "eval_handoff.json"),
            _repo_relative(root, meta / "stage_status.json"),
        ],
        "do_not_use_for_downstream_gate": True,
        "reason": "post-train eval failed but later full paper eval accepted through eval_handoff; stage_status is the final downstream truth",
        "original_quality_audit_preserved": True,
        "original_quality_audit": _repo_relative(root, audit),
        "original_quality_audit_hash": _file_sha256(audit),
        "run_id": str(stage_status.get("run_id") or run_dir.name),
        "task": int(stage_status.get("task") or stage_status.get("task_id") or 0),
    }
    return atomic_write_json(meta / "quality_audit.json.superseded_by.json", payload)


def mark_superseded(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
    superseded_by_run_id: str,
) -> dict[str, Any]:
    payload = build_stage_status(repo_root=repo_root, stage=stage, task=int(task), run_id=str(run_id))
    payload["previous_final_status"] = payload.get("final_status")
    payload["previous_downstream_ready"] = payload.get("downstream_ready")
    payload["final_status"] = "superseded"
    payload["downstream_ready"] = False
    payload["ready_for"] = []
    reasons = list(payload.get("rejection_reasons") or [])
    reasons.append(f"superseded_by_run={run_naming.parse_run_id(str(superseded_by_run_id))}")
    payload["rejection_reasons"] = reasons
    payload["status_source"] = "stage_promotion"
    payload["superseded_by_run_id"] = run_naming.parse_run_id(str(superseded_by_run_id))
    payload["superseded_at_utc"] = _utc_now()
    write_stage_status(repo_root, payload)
    return payload


def status_is_downstream_ready(payload: Mapping[str, Any], *, consumer_stage: str | None = None) -> bool:
    if payload.get("schema_version") != STAGE_STATUS_SCHEMA_VERSION:
        return False
    if str(payload.get("final_status") or "").strip().lower() in BAD_FINAL_STATUSES:
        return False
    if payload.get("downstream_ready") is not True:
        return False
    if consumer_stage:
        return _canonical_stage(consumer_stage) in {str(x) for x in (payload.get("ready_for") or [])}
    return True


__all__ = [
    "BAD_FINAL_STATUSES",
    "QUALITY_AUDIT_SUPERSEDED_SCHEMA_VERSION",
    "READY_FOR_BY_STAGE",
    "STAGE_STATUS_SCHEMA_VERSION",
    "STAGE_STATUS_VALIDATOR_VERSION",
    "StageStatusError",
    "build_and_write_stage_status",
    "build_stage_status",
    "mark_superseded",
    "read_stage_status",
    "stage_status_path",
    "status_is_downstream_ready",
    "write_quality_audit_superseded_sidecar",
    "write_stage_status",
]
