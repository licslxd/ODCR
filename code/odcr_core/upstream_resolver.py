"""Unified upstream resolver and eligibility gate.

All downstream stages must resolve their producer run through this module.  The
resolver reads the task-level ``latest.json`` pointer, repairs missing
``meta/stage_status.json`` from canonical run artifacts when possible, and
applies the same formal eligibility gate for dry-run and runtime launches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from odcr_core import path_layout, run_naming
from odcr_core.stage_status import (
    BAD_FINAL_STATUSES,
    STAGE_STATUS_SCHEMA_VERSION,
    StageStatusError,
    build_and_write_stage_status,
    read_stage_status,
    status_is_downstream_ready,
)
from odcr_core.stage_status_validator import (
    StageStatusValidationError,
    validate_stage_status_evidence,
)


FORMAL_MODE = "formal"
PROBE_MODES = {"probe", "comparison"}


class UpstreamResolutionError(RuntimeError):
    """Raised when an upstream run cannot be used by a downstream stage."""


def _canonical_stage(stage: str) -> str:
    raw = str(stage or "").strip().lower()
    return {
        "train_step3": "step3",
        "train_step4": "step4",
        "train_step5": "step5",
        "eval-rerank": "rerank",
    }.get(raw, raw)


def _consumer_for(stage: str) -> str:
    stage_name = _canonical_stage(stage)
    return {
        "step3": "step4",
        "step4": "step5",
        "step5": "eval",
    }.get(stage_name, "")


def _stage_label(stage: str) -> str:
    return {
        "step3": "Step3",
        "step4": "Step4",
        "step5": "Step5",
        "eval": "Eval",
        "rerank": "Rerank",
    }.get(_canonical_stage(stage), stage)


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
    path = Path(text).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise UpstreamResolutionError(f"missing {context}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UpstreamResolutionError(f"{context} is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise UpstreamResolutionError(f"{context} must be a JSON object: {path}")
    return payload


@dataclass(frozen=True)
class UpstreamResolution:
    producer_stage: str
    consumer_stage: str
    task: int
    run_id: str
    run_dir: Path
    stage_status: dict[str, Any]
    latest_path: Path
    latest_run_id: str | None
    is_latest: bool
    mode: str
    requested_run: str | None = None
    validation: dict[str, Any] | None = None

    @property
    def eligible(self) -> bool:
        return status_is_downstream_ready(self.stage_status, consumer_stage=self.consumer_stage)

    def to_payload(self, repo_root: str | Path) -> dict[str, Any]:
        return {
            "schema_version": "odcr_upstream_resolution/1",
            "producer_stage": self.producer_stage,
            "consumer_stage": self.consumer_stage,
            "task": int(self.task),
            "run_id": self.run_id,
            "run_dir": _repo_relative(repo_root, self.run_dir),
            "latest_path": _repo_relative(repo_root, self.latest_path),
            "latest_run_id": self.latest_run_id,
            "is_latest": bool(self.is_latest),
            "mode": self.mode,
            "requested_run": self.requested_run,
            "eligible": self.eligible,
            "stage_status_path": _repo_relative(repo_root, self.run_dir / "meta" / "stage_status.json"),
            "stage_status": dict(self.stage_status),
            "stage_status_validation": dict(self.validation or {}),
        }


def resolve_latest(*, repo_root: str | Path, stage: str, task: int, repair: bool = True) -> UpstreamResolution:
    root = Path(repo_root).expanduser().resolve()
    stage_name = _canonical_stage(stage)
    parent = path_layout.get_stage_task_root(root, stage_name, int(task))
    latest_path = parent / "latest.json"
    latest = _load_json(latest_path, context=f"{stage_name} latest.json")
    latest_run_id = str(latest.get("active_run_id") or latest.get("latest_run_id") or "").strip()
    latest_summary_path = str(latest.get("latest_summary_path") or "").strip()
    latest_stage_status_path = str(latest.get("latest_stage_status_path") or "").strip()
    if not latest_run_id or not latest_summary_path:
        raise UpstreamResolutionError(
            f"latest.json pointer is incomplete for {stage_name} task {int(task)}: {latest_path}; "
            "expected latest_run_id and latest_summary_path"
        )
    run_id = run_naming.parse_stage_run_id(stage_name, latest_run_id)
    summary = _repo_path(root, latest_summary_path)
    expected_summary = parent / run_id / "meta" / "run_summary.json"
    if summary is None or summary.resolve() != expected_summary.resolve():
        raise UpstreamResolutionError(
            f"latest.json pointer is damaged for {stage_name} task {int(task)}: "
            f"latest_summary_path must target {expected_summary}, got {summary}"
        )
    if not expected_summary.is_file():
        raise UpstreamResolutionError(
            f"latest.json pointer is damaged for {stage_name} task {int(task)}: "
            f"missing run_summary.json at {expected_summary}"
        )
    if latest_stage_status_path:
        status_pointer = _repo_path(root, latest_stage_status_path)
        expected_status = parent / run_id / "meta" / "stage_status.json"
        if status_pointer is None or status_pointer.resolve() != expected_status.resolve():
            raise UpstreamResolutionError(
                f"latest.json pointer is damaged for {stage_name} task {int(task)}: "
                f"latest_stage_status_path must target {expected_status}, got {status_pointer}"
            )
    return resolve_run(
        repo_root=root,
        stage=stage_name,
        task=int(task),
        run_id=run_id,
        repair=repair,
        latest_payload=latest,
        latest_path=latest_path,
        requested_run="latest",
    )


def resolve_run(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    run_id: str,
    repair: bool = True,
    latest_payload: Mapping[str, Any] | None = None,
    latest_path: Path | None = None,
    requested_run: str | None = None,
) -> UpstreamResolution:
    root = Path(repo_root).expanduser().resolve()
    stage_name = _canonical_stage(stage)
    rid = run_naming.parse_stage_run_id(stage_name, str(run_id))
    run_dir = path_layout.get_stage_run_root(root, int(task), "v1", stage_name, rid).resolve()
    status_path = run_dir / "meta" / "stage_status.json"
    if repair and not status_path.is_file():
        status = build_and_write_stage_status(repo_root=root, stage=stage_name, task=int(task), run_id=rid)
    else:
        try:
            status = read_stage_status(run_dir, required=True)
        except StageStatusError as exc:
            raise UpstreamResolutionError(str(exc)) from exc
    if status.get("schema_version") != STAGE_STATUS_SCHEMA_VERSION:
        raise UpstreamResolutionError(f"unsupported stage_status schema for {stage_name} run {rid}")
    parent = path_layout.get_stage_task_root(root, stage_name, int(task))
    lp = latest_path or (parent / "latest.json")
    latest_run_id = None
    if latest_payload is None and lp.is_file():
        try:
            latest_payload = _load_json(lp, context=f"{stage_name} latest.json")
        except UpstreamResolutionError:
            latest_payload = {}
    if isinstance(latest_payload, Mapping):
        raw_latest = str(latest_payload.get("active_run_id") or latest_payload.get("latest_run_id") or "").strip()
        latest_run_id = run_naming.parse_stage_run_id(stage_name, raw_latest) if raw_latest else None
    return UpstreamResolution(
        producer_stage=stage_name,
        consumer_stage=_consumer_for(stage_name),
        task=int(task),
        run_id=rid,
        run_dir=run_dir,
        stage_status=status,
        latest_path=lp,
        latest_run_id=latest_run_id,
        is_latest=(latest_run_id == rid),
        mode=FORMAL_MODE,
        requested_run=requested_run,
        validation=None,
    )


def _artifact_missing_reasons(repo_root: Path, status: Mapping[str, Any], *, consumer_stage: str) -> list[str]:
    reasons: list[str] = []
    artifacts = status.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return ["stage_status.artifacts missing"]
    for key in ("run_summary", "checkpoint_lineage"):
        item = artifacts.get(key)
        if isinstance(item, Mapping) and item.get("path") and item.get("exists") is not True:
            reasons.append(f"{key}_missing")
    if consumer_stage == "step4":
        for key in ("eval_handoff", "selected_checkpoint"):
            item = artifacts.get(key)
            if isinstance(item, Mapping) and item.get("exists") is not True:
                reasons.append(f"{key}_missing")
    if consumer_stage == "step5":
        for key in ("selected_export", "export_manifest", "index_contract"):
            item = artifacts.get(key)
            if not isinstance(item, Mapping) or not item.get("path"):
                reasons.append(f"{key}_missing")
            elif item.get("exists") is not True:
                reasons.append(f"{key}_missing")
        if status.get("step5_train_input_role") == "pool_manifest_sampling_contract":
            for key in (
                "step5_pool_manifest",
                "step5_sampling_contract",
                "step5_pool_distribution_report",
                "step5_pool_exports_status",
            ):
                item = artifacts.get(key)
                if not isinstance(item, Mapping) or not item.get("path"):
                    reasons.append(f"{key}_missing")
                elif item.get("exists") is not True:
                    reasons.append(f"{key}_missing")
        elif status.get("step5_train_input_role") == "dedicated_split_exports":
            for key in (
                "rating_stability_control_scorer_train_export",
                "step5_explanation_explainer_train_export",
                "step5_train_manifest",
                "route_intersection_report",
                "step5_dedicated_exports_status",
            ):
                item = artifacts.get(key)
                if not isinstance(item, Mapping) or not item.get("path"):
                    reasons.append(f"{key}_missing")
                elif item.get("exists") is not True:
                    reasons.append(f"{key}_missing")
    return reasons


def validate_upstream_eligibility(
    resolution: UpstreamResolution,
    *,
    repo_root: str | Path,
    mode: str = FORMAL_MODE,
    require_latest: bool = True,
    consumer_stage: str | None = None,
) -> UpstreamResolution:
    root = Path(repo_root).expanduser().resolve()
    resolved_mode = str(mode or FORMAL_MODE).strip().lower()
    if resolved_mode not in {FORMAL_MODE, *PROBE_MODES}:
        raise UpstreamResolutionError(f"unsupported upstream mode {mode!r}; expected formal/probe/comparison")
    if resolved_mode in PROBE_MODES:
        raise UpstreamResolutionError(
            "comparison/probe namespace is not implemented yet; refusing to avoid formal pollution."
        )
    status = resolution.stage_status
    consumer = _canonical_stage(consumer_stage or resolution.consumer_stage)
    label = _stage_label(consumer)
    producer_label = _stage_label(resolution.producer_stage)
    current_active = f"run{resolution.latest_run_id}" if resolution.latest_run_id else "(none)"
    final_status = str(status.get("final_status") or "").strip().lower()
    reasons = [str(x) for x in (status.get("rejection_reasons") or []) if str(x)]
    if final_status in BAD_FINAL_STATUSES:
        reason = " / ".join([final_status, *reasons, f"downstream_ready={status.get('downstream_ready')}"])
        raise UpstreamResolutionError(
            f"run{resolution.run_id} is not eligible for {label} {resolved_mode} upstream; "
            f"reason = {reason}; current active run = {current_active}"
        )
    if status.get("downstream_ready") is not True:
        reason = " / ".join([*reasons, f"downstream_ready={status.get('downstream_ready')}"])
        raise UpstreamResolutionError(
            f"run{resolution.run_id} is not eligible for {label} {resolved_mode} upstream; "
            f"reason = {reason}; current active run = {current_active}"
        )
    ready_for = {str(x) for x in (status.get("ready_for") or [])}
    if consumer and consumer not in ready_for:
        raise UpstreamResolutionError(
            f"run{resolution.run_id} is not eligible for {label} {resolved_mode} upstream; "
            f"reason = ready_for_missing_{consumer}; current active run = {current_active}"
        )
    missing = _artifact_missing_reasons(root, status, consumer_stage=consumer)
    if missing:
        raise UpstreamResolutionError(
            f"run{resolution.run_id} is not eligible for {label} {resolved_mode} upstream; "
            f"reason = {' / '.join(missing)}; current active run = {current_active}"
        )
    latest_payload: Mapping[str, Any] | None = None
    if resolution.latest_path.is_file():
        latest_payload = _load_json(resolution.latest_path, context=f"{resolution.producer_stage} latest.json")
    try:
        validation = validate_stage_status_evidence(
            repo_root=root,
            stage=resolution.producer_stage,
            task=int(resolution.task),
            run_id=resolution.run_id,
            consumer_stage=consumer,
            status_payload=status,
            run_dir=resolution.run_dir,
            latest_payload=latest_payload,
            latest_path=resolution.latest_path,
            require_latest=bool(resolved_mode == FORMAL_MODE and require_latest and resolution.is_latest),
        )
    except StageStatusValidationError as exc:
        raise UpstreamResolutionError(
            f"run{resolution.run_id} is not eligible for {label} {resolved_mode} upstream; "
            f"reason = stage_status_strict_validation_failed: {exc}; current active run = {current_active}"
        ) from exc
    if resolved_mode == FORMAL_MODE and require_latest and not resolution.is_latest:
        raise UpstreamResolutionError(
            f"run{resolution.run_id} is not eligible for {label} formal upstream; "
            f"reason = non_latest_eligible_run_requires_promote; current active run = {current_active}; "
            f"promote with ./odcr promote-upstream --stage {resolution.producer_stage} "
            f"--task {int(resolution.task)} --run-id {resolution.run_id}"
        )
    _ = producer_label
    return UpstreamResolution(
        producer_stage=resolution.producer_stage,
        consumer_stage=resolution.consumer_stage,
        task=resolution.task,
        run_id=resolution.run_id,
        run_dir=resolution.run_dir,
        stage_status=resolution.stage_status,
        latest_path=resolution.latest_path,
        latest_run_id=resolution.latest_run_id,
        is_latest=resolution.is_latest,
        mode=resolution.mode,
        requested_run=resolution.requested_run,
        validation=validation.to_payload(root),
    )


def resolve_upstream(
    *,
    repo_root: str | Path,
    stage: str,
    task: int,
    from_run: str | None = None,
    mode: str = FORMAL_MODE,
    consumer_stage: str | None = None,
    repair: bool = True,
) -> UpstreamResolution:
    root = Path(repo_root).expanduser().resolve()
    requested = str(from_run or "").strip()
    if not requested or requested == "latest":
        resolution = resolve_latest(repo_root=root, stage=stage, task=int(task), repair=repair)
    else:
        latest_payload: dict[str, Any] | None = None
        latest_path = path_layout.get_stage_task_root(root, _canonical_stage(stage), int(task)) / "latest.json"
        if latest_path.is_file():
            latest_payload = _load_json(latest_path, context=f"{_canonical_stage(stage)} latest.json")
        resolution = resolve_run(
            repo_root=root,
            stage=stage,
            task=int(task),
            run_id=run_naming.parse_stage_run_id(_canonical_stage(stage), requested),
            repair=repair,
            latest_payload=latest_payload,
            latest_path=latest_path,
            requested_run=requested,
        )
    return validate_upstream_eligibility(
        resolution,
        repo_root=root,
        mode=mode,
        require_latest=(str(mode or FORMAL_MODE).strip().lower() == FORMAL_MODE),
        consumer_stage=consumer_stage or _consumer_for(stage),
    )


__all__ = [
    "FORMAL_MODE",
    "PROBE_MODES",
    "UpstreamResolution",
    "UpstreamResolutionError",
    "resolve_latest",
    "resolve_run",
    "resolve_upstream",
    "validate_upstream_eligibility",
]
