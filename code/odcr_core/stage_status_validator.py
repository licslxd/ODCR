"""Strict read-time validation for stage_status upstream evidence.

``stage_status.json`` is an evidence index, not a proof.  The validator in this
module treats every ready claim as untrusted until the referenced artifacts are
re-read from disk and their paths, schemas, hashes, lineage, and task identity
are verified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from odcr_core import path_layout, run_naming
from odcr_core.index_contract import INDEX_CONTRACT_SCHEMA_VERSION, load_index_contract
from odcr_core.stage_status import BAD_FINAL_STATUSES, STAGE_STATUS_SCHEMA_VERSION
from odcr_core.step4_dedicated_exports import validate_step4_dedicated_exports
from odcr_core.step4_pool_exports import validate_step4_pool_exports
from odcr_core.step3_eval_handoff import EVAL_HANDOFF_SCHEMA_VERSION, PAPER_TARGET_ONLY_EVAL
from odcr_core.training_checkpoint import (
    CHECKPOINT_EVENT_LEDGER_SCHEMA_VERSION,
    LINEAGE_GATE_SCHEMA_VERSION,
    CheckpointLineageError,
    checkpoint_file_sha256,
    read_checkpoint_lineage,
)


STAGE_STATUS_VALIDATOR_VERSION = "odcr_stage_status_validator/2"

STEP3_READY_FINAL_STATUSES = {"completed_with_eval_handoff", "eval_handoff_accepted"}

STEP3_STEP4_REQUIRED_FIELDS = (
    "schema_version",
    "validator_version",
    "generated_at",
    "updated_at",
    "stage",
    "task",
    "task_id",
    "run_id",
    "run_dir",
    "final_status",
    "downstream_ready",
    "ready_for",
    "selected_checkpoint",
    "selected_checkpoint_hash",
    "eval_handoff",
    "run_summary",
    "checkpoint_lineage",
    "source_table",
    "resolved_config",
    "status_source",
    "failure_history_preserved",
    "do_not_use_quality_audit_as_final_truth",
    "artifacts",
    "required_artifacts",
)

STEP3_STEP4_REQUIRED_ARTIFACTS = (
    "run_summary",
    "eval_handoff",
    "selected_checkpoint",
    "checkpoint_lineage",
    "source_table",
    "resolved_config",
)

STEP4_READY_FINAL_STATUSES = {"completed"}

STEP4_STEP5_REQUIRED_FIELDS = (
    "schema_version",
    "validator_version",
    "generated_at",
    "updated_at",
    "stage",
    "task",
    "task_id",
    "run_id",
    "run_dir",
    "final_status",
    "downstream_ready",
    "ready_for",
    "selected_export",
    "export_manifest",
    "index_contract",
    "export_readiness",
    "run_summary",
    "source_table",
    "resolved_config",
    "status_source",
    "do_not_use_quality_audit_as_final_truth",
    "artifacts",
    "required_artifacts",
)

STEP4_STEP5_REQUIRED_ARTIFACTS = (
    "run_summary",
    "source_table",
    "resolved_config",
    "selected_export",
    "export_manifest",
    "index_contract",
)

FORBIDDEN_ARTIFACT_PREFIXES = {"AI_analysis", "docs", "tmp", "history", "_archive"}


class StageStatusValidationError(RuntimeError):
    """Raised when a stage_status ready claim cannot be verified."""


@dataclass(frozen=True)
class StageStatusValidation:
    stage: str
    task: int
    run_id: str
    consumer_stage: str
    run_dir: Path
    status_path: Path
    selected_checkpoint: Path | None = None
    selected_checkpoint_hash: str | None = None
    eval_handoff: Path | None = None
    run_summary: Path | None = None
    checkpoint_lineage: Path | None = None
    source_table: Path | None = None
    resolved_config: Path | None = None
    latest_path: Path | None = None
    latest_warnings: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, repo_root: str | Path) -> dict[str, Any]:
        root = Path(repo_root).expanduser().resolve()
        return {
            "schema_version": "odcr_stage_status_validation/1",
            "validator_version": STAGE_STATUS_VALIDATOR_VERSION,
            "stage": self.stage,
            "task": int(self.task),
            "run_id": self.run_id,
            "consumer_stage": self.consumer_stage,
            "run_dir": _repo_relative(root, self.run_dir),
            "status_path": _repo_relative(root, self.status_path),
            "selected_checkpoint": _repo_relative(root, self.selected_checkpoint),
            "selected_checkpoint_hash": self.selected_checkpoint_hash,
            "eval_handoff": _repo_relative(root, self.eval_handoff),
            "run_summary": _repo_relative(root, self.run_summary),
            "checkpoint_lineage": _repo_relative(root, self.checkpoint_lineage),
            "source_table": _repo_relative(root, self.source_table),
            "resolved_config": _repo_relative(root, self.resolved_config),
            "latest_path": _repo_relative(root, self.latest_path),
            "latest_warnings": list(self.latest_warnings),
            "diagnostics": dict(self.diagnostics),
        }


def _canonical_stage(stage: str) -> str:
    raw = str(stage or "").strip().lower()
    return {
        "train_step3": "step3",
        "train_step4": "step4",
        "train_step5": "step5",
        "eval-rerank": "rerank",
    }.get(raw, raw)


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


def _repo_path(repo_root: str | Path, raw: Any, *, field: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise StageStatusValidationError(f"{field} is required")
    root = Path(repo_root).expanduser().resolve()
    path = Path(text).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _forbid_artifact_path(repo_root: Path, path: Path, *, field: str) -> None:
    if not _is_relative_to(path, repo_root):
        raise StageStatusValidationError(f"{field} escapes repo root: {path}")
    rel = path.relative_to(repo_root).parts
    if rel and rel[0] in FORBIDDEN_ARTIFACT_PREFIXES:
        raise StageStatusValidationError(f"{field} points into forbidden namespace {rel[0]}: {path}")


def _require_under(path: Path, parent: Path, *, field: str) -> None:
    if not _is_relative_to(path, parent):
        raise StageStatusValidationError(f"{field} must stay under {parent}: {path}")


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise StageStatusValidationError(f"{field} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageStatusValidationError(f"{field} invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageStatusValidationError(f"{field} JSON root must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StageStatusValidationError(message)


def _required_artifact_keys(payload: Mapping[str, Any], expected: tuple[str, ...]) -> set[str]:
    raw = payload.get("required_artifacts")
    if isinstance(raw, Mapping):
        keys = {str(key) for key in raw.keys()}
    elif isinstance(raw, list):
        keys = {str(item) for item in raw}
    else:
        raise StageStatusValidationError("required_artifacts must be a non-empty list or mapping")
    if not keys:
        raise StageStatusValidationError("required_artifacts must not be empty")
    expected_set = set(expected)
    unknown = keys.difference(expected_set)
    if unknown:
        raise StageStatusValidationError("unknown required_artifacts keys: " + ", ".join(sorted(unknown)))
    missing = expected_set.difference(keys)
    if missing:
        raise StageStatusValidationError("required_artifacts missing keys: " + ", ".join(sorted(missing)))
    return keys


def _artifact_path(payload: Mapping[str, Any], key: str) -> str | None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise StageStatusValidationError("artifacts must be a non-empty object")
    item = artifacts.get(key)
    if not isinstance(item, Mapping):
        raise StageStatusValidationError(f"artifacts.{key} must be an object")
    path = item.get("path")
    return str(path) if path not in (None, "") else None


def _path_from_status(
    *,
    repo_root: Path,
    run_dir: Path,
    payload: Mapping[str, Any],
    field: str,
    artifact_key: str,
    must_be_under: Path,
    must_be_file: bool = True,
) -> Path:
    status_value = payload.get(field)
    if status_value in (None, ""):
        raise StageStatusValidationError(f"{field} is required")
    artifact_value = _artifact_path(payload, artifact_key)
    if artifact_value and _repo_relative(repo_root, artifact_value) != _repo_relative(repo_root, status_value):
        raise StageStatusValidationError(f"artifacts.{artifact_key}.path does not match {field}")
    path = _repo_path(repo_root, status_value, field=field)
    _forbid_artifact_path(repo_root, path, field=field)
    _require_under(path, must_be_under, field=field)
    if must_be_file and not path.is_file():
        raise StageStatusValidationError(f"{field} missing: {path}")
    if run_dir and not _is_relative_to(path, run_dir):
        raise StageStatusValidationError(f"{field} must stay under run_dir: {path}")
    return path


def _validate_latest_pointer(
    *,
    repo_root: Path,
    stage: str,
    task: int,
    run_id: str,
    run_summary: Path,
    status_path: Path,
    latest_payload: Mapping[str, Any] | None,
    latest_path: Path | None,
    require_latest: bool,
    final_status: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if latest_payload is None:
        if require_latest:
            raise StageStatusValidationError("latest.json payload is required for formal latest validation")
        return ()
    raw_latest = str(latest_payload.get("active_run_id") or latest_payload.get("latest_run_id") or "").strip()
    if require_latest:
        _require(raw_latest == run_id, f"latest pointer does not target run{run_id}: latest={raw_latest or '(missing)'}")
        latest_summary_raw = latest_payload.get("latest_summary_path")
        _require(bool(latest_summary_raw), "latest_summary_path is required")
        latest_summary = _repo_path(repo_root, latest_summary_raw, field="latest_summary_path")
        _require(latest_summary.resolve() == run_summary.resolve(), "latest_summary_path does not match run_summary")
        latest_status_path_raw = latest_payload.get("latest_stage_status_path")
        if latest_status_path_raw:
            latest_status_path = _repo_path(repo_root, latest_status_path_raw, field="latest_stage_status_path")
            _require(latest_status_path.resolve() == status_path.resolve(), "latest_stage_status_path does not match stage_status")
    legacy_status = str(latest_payload.get("latest_status") or "").strip()
    if legacy_status and legacy_status != final_status:
        warnings.append(f"deprecated latest_status={legacy_status!r} ignored; stage_status.final_status={final_status!r}")
    _ = latest_path
    _ = stage
    _ = task
    return tuple(warnings)


def _matching_event(event: Mapping[str, Any], checkpoint: Path, checkpoint_hash: str) -> bool:
    path_text = str(event.get("checkpoint_file") or event.get("path") or "").strip()
    if not path_text:
        return False
    event_path = Path(path_text).expanduser()
    if not event_path.is_absolute():
        return False
    return event_path.resolve() == checkpoint.resolve() and str(
        event.get("checkpoint_file_hash") or event.get("hash") or ""
    ) == checkpoint_hash


def _validate_checkpoint_lineage(
    *,
    repo_root: Path,
    stage: str,
    task: int,
    run_id: str,
    checkpoint: Path,
    checkpoint_hash: str,
    lineage_path: Path,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        sidecar = read_checkpoint_lineage(checkpoint, expected_stage=stage)
    except CheckpointLineageError as exc:
        raise StageStatusValidationError(f"checkpoint lineage sidecar invalid: {exc}") from exc
    _require(str(sidecar.get("checkpoint_file_hash") or "") == checkpoint_hash, "checkpoint sidecar hash mismatch")
    _require(int(sidecar.get("task_id") or -1) == int(task), "checkpoint sidecar task_id mismatch")
    _require(str(sidecar.get("run_id") or "") == str(run_id), "checkpoint sidecar run_id mismatch")
    task_payload = resolved_config.get("task") if isinstance(resolved_config.get("task"), Mapping) else {}
    for key, sidecar_key in (("source", "source_domain"), ("target", "target_domain")):
        expected = str(task_payload.get(key) or "").strip()
        if expected:
            actual = str(sidecar.get(sidecar_key) or "").strip()
            _require(actual == expected, f"checkpoint sidecar {sidecar_key} mismatch: {actual!r} != {expected!r}")
    ledger = _load_json(lineage_path, field="checkpoint_lineage")
    schema = str(ledger.get("schema_version") or "")
    if schema == CHECKPOINT_EVENT_LEDGER_SCHEMA_VERSION:
        _require(str(ledger.get("stage") or "") == stage, "checkpoint_lineage stage mismatch")
        _require(int(ledger.get("task_id") or -1) == int(task), "checkpoint_lineage task_id mismatch")
        _require(str(ledger.get("run_id") or "") == str(run_id), "checkpoint_lineage run_id mismatch")
        events = ledger.get("saved_checkpoint_events")
        _require(isinstance(events, list), "checkpoint_lineage saved_checkpoint_events must be a list")
        _require(
            any(isinstance(item, Mapping) and _matching_event(item, checkpoint, checkpoint_hash) for item in events),
            "checkpoint_lineage has no event matching selected_checkpoint path/hash",
        )
    elif schema == LINEAGE_GATE_SCHEMA_VERSION:
        _require(str(ledger.get("checkpoint_file_hash") or "") == checkpoint_hash, "checkpoint_lineage hash mismatch")
    else:
        raise StageStatusValidationError(f"unsupported checkpoint_lineage schema {schema!r}: {lineage_path}")
    return {"sidecar_schema": sidecar.get("schema_version"), "ledger_schema": schema}


def _validate_step3_step4_ready(
    *,
    repo_root: Path,
    task: int,
    run_id: str,
    run_dir: Path,
    status_path: Path,
    payload: Mapping[str, Any],
    latest_payload: Mapping[str, Any] | None,
    latest_path: Path | None,
    require_latest: bool,
) -> StageStatusValidation:
    missing = [field for field in STEP3_STEP4_REQUIRED_FIELDS if payload.get(field) in (None, "", [], {})]
    if missing:
        raise StageStatusValidationError("stage_status missing required ready fields: " + ", ".join(missing))
    _required_artifact_keys(payload, STEP3_STEP4_REQUIRED_ARTIFACTS)
    _require(str(payload.get("validator_version")) == STAGE_STATUS_VALIDATOR_VERSION, "validator_version mismatch")
    final_status = str(payload.get("final_status") or "").strip()
    _require(final_status in STEP3_READY_FINAL_STATUSES, f"final_status {final_status!r} is not an accepted Step3 handoff")
    _require(payload.get("downstream_ready") is True, "downstream_ready must be true for accepted Step3 handoff")
    ready_for = {str(item) for item in payload.get("ready_for") or []}
    _require("step4" in ready_for, "ready_for must include step4")
    _require(str(payload.get("status_source") or "") == "eval_handoff", "status_source must be eval_handoff")
    _require(payload.get("failure_history_preserved") is True, "failure_history_preserved must be true")
    _require(
        payload.get("do_not_use_quality_audit_as_final_truth") is True,
        "do_not_use_quality_audit_as_final_truth must be true",
    )
    meta = run_dir / "meta"
    checkpoint = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="selected_checkpoint",
        artifact_key="selected_checkpoint",
        must_be_under=run_dir / "model",
    )
    checkpoint_hash = _file_sha256(checkpoint)
    _require(
        checkpoint_hash == str(payload.get("selected_checkpoint_hash") or ""),
        "selected_checkpoint_hash mismatch",
    )
    for key in ("selected_checkpoint", "run_summary", "eval_handoff", "checkpoint_lineage", "source_table", "resolved_config"):
        artifact_path = _artifact_path(payload, "selected_checkpoint" if key == "selected_checkpoint" else key)
        if artifact_path:
            item_path = _repo_path(repo_root, artifact_path, field=f"artifacts.{key}.path")
            _forbid_artifact_path(repo_root, item_path, field=f"artifacts.{key}.path")
            if not item_path.is_file():
                raise StageStatusValidationError(f"artifacts.{key}.path missing on disk: {item_path}")
    run_summary_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="run_summary",
        artifact_key="run_summary",
        must_be_under=meta,
    )
    eval_handoff_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="eval_handoff",
        artifact_key="eval_handoff",
        must_be_under=meta,
    )
    checkpoint_lineage_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="checkpoint_lineage",
        artifact_key="checkpoint_lineage",
        must_be_under=run_dir,
    )
    source_table_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="source_table",
        artifact_key="source_table",
        must_be_under=meta,
    )
    resolved_config_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="resolved_config",
        artifact_key="resolved_config",
        must_be_under=meta,
    )
    run_summary = _load_json(run_summary_path, field="run_summary")
    _require(str(run_summary.get("stage") or "") == "step3", "run_summary stage mismatch")
    _require(int(run_summary.get("task_id") or -1) == int(task), "run_summary task_id mismatch")
    _require(str(run_summary.get("run_id") or "") == str(run_id), "run_summary run_id mismatch")
    _require(str(run_summary.get("status") or "") in STEP3_READY_FINAL_STATUSES, "run_summary status mismatch")
    _require(run_summary.get("downstream_ready") is True, "run_summary downstream_ready must be true")
    if run_summary.get("selected_checkpoint_hash"):
        _require(
            str(run_summary.get("selected_checkpoint_hash")) == checkpoint_hash,
            "run_summary selected_checkpoint_hash mismatch",
        )
    eval_handoff = _load_json(eval_handoff_path, field="eval_handoff")
    _require(str(eval_handoff.get("schema_version") or "") == EVAL_HANDOFF_SCHEMA_VERSION, "eval_handoff schema mismatch")
    _require(int(eval_handoff.get("task_id") or -1) == int(task), "eval_handoff task_id mismatch")
    _require(str(eval_handoff.get("run_id") or "") == str(run_id), "eval_handoff run_id mismatch")
    _require(str(eval_handoff.get("train_status") or "") == "completed", "eval_handoff train_status must be completed")
    _require(str(eval_handoff.get("paper_eval_status") or "") == "completed", "eval_handoff paper_eval_status must be completed")
    _require(str(eval_handoff.get("paper_eval_protocol") or "") == PAPER_TARGET_ONLY_EVAL, "eval_handoff protocol mismatch")
    _require(eval_handoff.get("old_failure_history_preserved") is True, "eval_handoff must preserve old failure history")
    _require(str(eval_handoff.get("checkpoint_hash") or "") == checkpoint_hash, "eval_handoff checkpoint_hash mismatch")
    handoff_checkpoint = _repo_path(repo_root, eval_handoff.get("checkpoint_path"), field="eval_handoff.checkpoint_path")
    _require(handoff_checkpoint.resolve() == checkpoint.resolve(), "eval_handoff checkpoint_path mismatch")
    source_table = _load_json(source_table_path, field="source_table")
    _require(isinstance(source_table.get("records"), list), "source_table.records must be a list")
    resolved_config = _load_json(resolved_config_path, field="resolved_config")
    task_payload = resolved_config.get("task") if isinstance(resolved_config.get("task"), Mapping) else {}
    if task_payload:
        _require(int(task_payload.get("id") or -1) == int(task), "resolved_config task.id mismatch")
        for key, summary_key in (("source", "source_domain"), ("target", "target_domain")):
            expected = str(task_payload.get(key) or "").strip()
            actual = str(run_summary.get(summary_key) or "").strip()
            if expected and actual:
                _require(actual == expected, f"run_summary {summary_key} mismatch: {actual!r} != {expected!r}")
    lineage_diag = _validate_checkpoint_lineage(
        repo_root=repo_root,
        stage="step3",
        task=int(task),
        run_id=str(run_id),
        checkpoint=checkpoint,
        checkpoint_hash=checkpoint_hash,
        lineage_path=checkpoint_lineage_path,
        resolved_config=resolved_config,
    )
    latest_warnings = _validate_latest_pointer(
        repo_root=repo_root,
        stage="step3",
        task=int(task),
        run_id=str(run_id),
        run_summary=run_summary_path,
        status_path=status_path,
        latest_payload=latest_payload,
        latest_path=latest_path,
        require_latest=require_latest,
        final_status=final_status,
    )
    return StageStatusValidation(
        stage="step3",
        task=int(task),
        run_id=str(run_id),
        consumer_stage="step4",
        run_dir=run_dir,
        status_path=status_path,
        selected_checkpoint=checkpoint,
        selected_checkpoint_hash=checkpoint_hash,
        eval_handoff=eval_handoff_path,
        run_summary=run_summary_path,
        checkpoint_lineage=checkpoint_lineage_path,
        source_table=source_table_path,
        resolved_config=resolved_config_path,
        latest_path=latest_path,
        latest_warnings=latest_warnings,
        diagnostics={
            "lineage": lineage_diag,
            "quality_audit_diagnostic": {
                "path": payload.get("quality_audit"),
                "status": payload.get("quality_audit_status"),
                "downstream_ready": payload.get("quality_audit_downstream_ready"),
                "ignored_for_final_truth": True,
            },
        },
    )


def _validate_step4_step5_ready(
    *,
    repo_root: Path,
    task: int,
    run_id: str,
    run_dir: Path,
    status_path: Path,
    payload: Mapping[str, Any],
    latest_payload: Mapping[str, Any] | None,
    latest_path: Path | None,
    require_latest: bool,
) -> StageStatusValidation:
    missing = [field for field in STEP4_STEP5_REQUIRED_FIELDS if payload.get(field) in (None, "", [], {})]
    if missing:
        raise StageStatusValidationError("stage_status missing required ready fields: " + ", ".join(missing))
    _required_artifact_keys(payload, STEP4_STEP5_REQUIRED_ARTIFACTS)
    _require(str(payload.get("validator_version")) == STAGE_STATUS_VALIDATOR_VERSION, "validator_version mismatch")
    final_status = str(payload.get("final_status") or "").strip().lower()
    _require(final_status in STEP4_READY_FINAL_STATUSES, f"final_status {final_status!r} is not an accepted Step4 handoff")
    _require(payload.get("downstream_ready") is True, "downstream_ready must be true for accepted Step4 handoff")
    ready_for = {str(item) for item in payload.get("ready_for") or []}
    _require("step5" in ready_for, "ready_for must include step5")
    _require(str(payload.get("status_source") or "") == "step4_export_readiness_validator", "status_source must be step4_export_readiness_validator")
    _require(
        payload.get("do_not_use_quality_audit_as_final_truth") is True,
        "do_not_use_quality_audit_as_final_truth must be true",
    )
    meta = run_dir / "meta"
    run_summary_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="run_summary",
        artifact_key="run_summary",
        must_be_under=meta,
    )
    source_table_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="source_table",
        artifact_key="source_table",
        must_be_under=meta,
    )
    resolved_config_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="resolved_config",
        artifact_key="resolved_config",
        must_be_under=meta,
    )
    selected_export_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="selected_export",
        artifact_key="selected_export",
        must_be_under=run_dir,
    )
    export_manifest_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="export_manifest",
        artifact_key="export_manifest",
        must_be_under=run_dir,
    )
    index_contract_path = _path_from_status(
        repo_root=repo_root,
        run_dir=run_dir,
        payload=payload,
        field="index_contract",
        artifact_key="index_contract",
        must_be_under=run_dir,
    )
    run_summary = _load_json(run_summary_path, field="run_summary")
    _require(str(run_summary.get("stage") or "") == "step4", "run_summary stage mismatch")
    _require(int(run_summary.get("task_id") or -1) == int(task), "run_summary task_id mismatch")
    _require(str(run_summary.get("run_id") or "") == str(run_id), "run_summary run_id mismatch")
    _require(str(run_summary.get("status") or "") in {"ok", "completed", "success"}, "run_summary status mismatch")
    source_table = _load_json(source_table_path, field="source_table")
    _require(isinstance(source_table, Mapping), "source_table must be an object")
    resolved_config = _load_json(resolved_config_path, field="resolved_config")
    task_payload = resolved_config.get("task") if isinstance(resolved_config.get("task"), Mapping) else {}
    if task_payload:
        _require(int(task_payload.get("id") or -1) == int(task), "resolved_config task.id mismatch")
    readiness = payload.get("export_readiness")
    if not isinstance(readiness, Mapping) or readiness.get("ready") is not True:
        raise StageStatusValidationError("export_readiness.ready must be true")
    if readiness.get("errors"):
        raise StageStatusValidationError("export_readiness.errors must be empty")
    diagnostics = readiness.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise StageStatusValidationError("export_readiness.diagnostics must be an object")
    expected_hashes = {
        "selected_export": diagnostics.get("export_sha256"),
        "export_manifest": diagnostics.get("manifest_sha256"),
        "index_contract": diagnostics.get("index_contract_sha256"),
    }
    if not all(str(value or "").strip() for value in expected_hashes.values()):
        raise StageStatusValidationError("export_readiness.diagnostics missing artifact hashes")
    actual_paths = {
        "selected_export": selected_export_path,
        "export_manifest": export_manifest_path,
        "index_contract": index_contract_path,
    }
    for key, expected_hash in expected_hashes.items():
        artifact = payload.get("artifacts", {}).get(key) if isinstance(payload.get("artifacts"), Mapping) else {}
        if isinstance(artifact, Mapping) and artifact.get("sha256"):
            _require(str(artifact.get("sha256")) == str(expected_hash), f"artifacts.{key}.sha256 mismatch")
    _require(_file_sha256(export_manifest_path) == str(expected_hashes["export_manifest"]), "export_manifest hash mismatch during Step4 handoff validation")
    _require(_file_sha256(index_contract_path) == str(expected_hashes["index_contract"]), "index_contract hash mismatch during Step4 handoff validation")
    manifest = _load_json(export_manifest_path, field="export_manifest")
    row_counts = manifest.get("row_counts") if isinstance(manifest.get("row_counts"), Mapping) else {}
    if row_counts:
        _require(int(row_counts.get("total_rows") or -1) == int(readiness.get("row_count") or -2), "export_readiness row_count mismatch")
    contract = load_index_contract(str(index_contract_path))
    _require(str(contract.get("schema_version") or "") == INDEX_CONTRACT_SCHEMA_VERSION, "index_contract schema mismatch")
    train_fp = (contract.get("fingerprints") or {}).get("train_csv") if isinstance(contract.get("fingerprints"), Mapping) else {}
    _require(isinstance(train_fp, Mapping), "index_contract train_csv fingerprint missing")
    _require(train_fp.get("exists") is True and train_fp.get("is_file") is True, "index_contract train_csv fingerprint is stale")
    fp_path = _repo_path(repo_root, train_fp.get("path"), field="index_contract.fingerprints.train_csv.path")
    _require(fp_path.resolve() == selected_export_path.resolve(), "index_contract train_csv path mismatch")
    _require(str(train_fp.get("sha256") or "") == str(expected_hashes["selected_export"]), "index_contract train_csv sha256 mismatch")
    stat = selected_export_path.stat()
    if train_fp.get("size") is not None:
        _require(int(train_fp.get("size")) == int(stat.st_size), "index_contract train_csv size mismatch")
    if train_fp.get("mtime_ns") is not None:
        _require(
            int(train_fp.get("mtime_ns")) == int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            "index_contract train_csv mtime_ns mismatch",
        )
    latest_warnings = _validate_latest_pointer(
        repo_root=repo_root,
        stage="step4",
        task=int(task),
        run_id=str(run_id),
        run_summary=run_summary_path,
        status_path=status_path,
        latest_payload=latest_payload,
        latest_path=latest_path,
        require_latest=require_latest,
        final_status=final_status,
    )
    dedicated_validation = None
    pool_validation = None
    train_input_role = str(payload.get("step5_train_input_role") or "")
    if train_input_role == "pool_manifest_sampling_contract" or payload.get("step5_pool_exports_ready") is not None:
        pool_validation = validate_step4_pool_exports(run_dir, repo_root=repo_root)
        if not pool_validation.ready:
            raise StageStatusValidationError(
                "Step5 pool exports failed validation: "
                + "; ".join(pool_validation.errors or ["unknown"])
            )
        _require(payload.get("step5_pool_exports_ready") is True, "step5_pool_exports_ready must be true")
        _require(
            train_input_role == "pool_manifest_sampling_contract",
            "step5_train_input_role must be pool_manifest_sampling_contract",
        )
        _require(
            payload.get("full_audit_default_train_forbidden") is True,
            "full_audit_default_train_forbidden must be true",
        )
    elif payload.get("step5_dedicated_exports_ready") is not None or payload.get("step5_train_input_role"):
        dedicated_validation = validate_step4_dedicated_exports(
            run_dir,
            repo_root=repo_root,
            expected_source_full_export_sha256=str(expected_hashes["selected_export"]),
        )
        if not dedicated_validation.ready:
            raise StageStatusValidationError(
                "dedicated Step5 exports failed validation: "
                + "; ".join(dedicated_validation.errors or ["unknown"])
            )
        _require(payload.get("step5_dedicated_exports_ready") is True, "step5_dedicated_exports_ready must be true")
        _require(str(payload.get("full_audit_table_role") or "") == "audit_only", "full_audit_table_role must be audit_only")
        _require(
            str(payload.get("step5_train_input_role") or "") == "dedicated_split_exports",
            "step5_train_input_role must be dedicated_split_exports",
        )
    return StageStatusValidation(
        stage="step4",
        task=int(task),
        run_id=str(run_id),
        consumer_stage="step5",
        run_dir=run_dir,
        status_path=status_path,
        run_summary=run_summary_path,
        source_table=source_table_path,
        resolved_config=resolved_config_path,
        latest_path=latest_path,
        latest_warnings=latest_warnings,
        diagnostics={
            "step4_export_readiness": dict(readiness),
            "step4_dedicated_exports_readiness": dedicated_validation.to_payload(repo_root) if dedicated_validation else None,
            "step4_pool_exports_readiness": pool_validation.to_payload(repo_root) if pool_validation else None,
            "read_time_validation": "manifest/index_contract rehashed; selected_export stat and recorded sha verified through refreshed index_contract",
            "selected_export": _repo_relative(repo_root, selected_export_path),
            "export_manifest": _repo_relative(repo_root, export_manifest_path),
            "index_contract": _repo_relative(repo_root, index_contract_path),
        },
    )


def validate_stage_status_evidence(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
    consumer_stage: str,
    status_payload: Mapping[str, Any],
    run_dir: str | Path | None = None,
    latest_payload: Mapping[str, Any] | None = None,
    latest_path: str | Path | None = None,
    require_latest: bool = False,
) -> StageStatusValidation:
    root = Path(repo_root).expanduser().resolve()
    stage_name = _canonical_stage(stage)
    consumer = _canonical_stage(consumer_stage)
    rid = run_naming.parse_stage_run_id(stage_name, str(run_id))
    expected_run_dir = path_layout.get_stage_run_root(root, int(task), "v1", stage_name, rid).resolve()
    actual_run_dir = Path(run_dir).expanduser().resolve() if run_dir is not None else expected_run_dir
    _require(actual_run_dir == expected_run_dir, f"run_dir path mismatch for {stage_name} task{task} run{rid}")
    status_path = actual_run_dir / "meta" / "stage_status.json"
    payload = dict(status_payload)
    _require(payload.get("schema_version") == STAGE_STATUS_SCHEMA_VERSION, "stage_status schema_version mismatch")
    _require(_canonical_stage(str(payload.get("stage") or "")) == stage_name, "stage_status stage mismatch")
    _require(int(payload.get("task") or payload.get("task_id") or -1) == int(task), "stage_status task mismatch")
    _require(str(payload.get("run_id") or "") == rid, "stage_status run_id mismatch")
    status_run_dir = _repo_path(root, payload.get("run_dir"), field="stage_status.run_dir")
    _require(status_run_dir.resolve() == expected_run_dir, "stage_status run_dir mismatch")
    final_status = str(payload.get("final_status") or "").strip().lower()
    ready_claim = (
        payload.get("downstream_ready") is True
        or consumer in {str(item) for item in (payload.get("ready_for") or [])}
        or final_status in STEP3_READY_FINAL_STATUSES
    )
    if final_status in BAD_FINAL_STATUSES and not ready_claim:
        return StageStatusValidation(
            stage=stage_name,
            task=int(task),
            run_id=rid,
            consumer_stage=consumer,
            run_dir=expected_run_dir,
            status_path=status_path,
            latest_path=Path(latest_path).expanduser().resolve() if latest_path else None,
        )
    if stage_name == "step3" and consumer == "step4":
        return _validate_step3_step4_ready(
            repo_root=root,
            task=int(task),
            run_id=rid,
            run_dir=expected_run_dir,
            status_path=status_path,
            payload=payload,
            latest_payload=latest_payload,
            latest_path=Path(latest_path).expanduser().resolve() if latest_path else None,
            require_latest=require_latest,
        )
    if stage_name == "step4" and consumer == "step5":
        return _validate_step4_step5_ready(
            repo_root=root,
            task=int(task),
            run_id=rid,
            run_dir=expected_run_dir,
            status_path=status_path,
            payload=payload,
            latest_payload=latest_payload,
            latest_path=Path(latest_path).expanduser().resolve() if latest_path else None,
            require_latest=require_latest,
        )
    return StageStatusValidation(
        stage=stage_name,
        task=int(task),
        run_id=rid,
        consumer_stage=consumer,
        run_dir=expected_run_dir,
        status_path=status_path,
        latest_path=Path(latest_path).expanduser().resolve() if latest_path else None,
    )


__all__ = [
    "STAGE_STATUS_VALIDATOR_VERSION",
    "StageStatusValidation",
    "StageStatusValidationError",
    "validate_stage_status_evidence",
]
