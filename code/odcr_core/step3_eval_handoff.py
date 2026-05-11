"""Step3 eval-only handoff acceptance and downstream gate helpers."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from odcr_core.file_atomic import atomic_write_json
from odcr_core import path_layout
from odcr_core.training_checkpoint import checkpoint_file_sha256


EVAL_HANDOFF_SCHEMA_VERSION = "odcr_step3_eval_handoff/1"
EVAL_HANDOFF_GATE_VERSION = "odcr_step3_eval_handoff_gate/1"
PAPER_TARGET_ONLY_EVAL = "paper_target_only_eval"
DEFAULT_VALID_LABEL = "paper_valid_b6144_full_detached"
DEFAULT_TEST_LABEL = "paper_test_b6144_full_detached"
REQUIRED_METRIC_KEYS = (
    "MAE",
    "RMSE",
    "ROUGE-1",
    "ROUGE-L",
    "BLEU-1",
    "BLEU-2",
    "BLEU-3",
    "BLEU-4",
    "DIST-1",
    "DIST-2",
    "METEOR",
)
ACCEPTED_HANDOFF_STATUSES = {"completed_with_eval_handoff", "eval_handoff_accepted"}


class Step3EvalHandoffError(RuntimeError):
    """Raised when eval-only evidence cannot satisfy the downstream handoff gate."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _repo_relative(repo_root: Path, path: str | Path) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(p)


def _repo_path(repo_root: Path, raw: str | Path | None, *, context: str, required: bool = True) -> Path:
    if raw in (None, ""):
        if required:
            raise Step3EvalHandoffError(f"{context} is required")
        return Path()
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    if required and not path.exists():
        raise Step3EvalHandoffError(f"{context} missing: {path}")
    return path


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise Step3EvalHandoffError(f"{context} missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step3EvalHandoffError(f"{context} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise Step3EvalHandoffError(f"{context} JSON root must be an object: {path}")
    return data


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Step3EvalHandoffError(message)


def _eval_artifact_root(run_root: Path, split: str, label: str) -> Path:
    return run_root / "meta" / "eval_only" / label / f"eval_{PAPER_TARGET_ONLY_EVAL}_{split}"


def default_eval_paths(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root).expanduser().resolve()
    valid_root = _eval_artifact_root(root, "valid", DEFAULT_VALID_LABEL)
    test_root = _eval_artifact_root(root, "test", DEFAULT_TEST_LABEL)
    return {
        "valid_eval_path": valid_root / "eval_summary.json",
        "valid_protocol_path": valid_root / "eval_protocol.json",
        "valid_integrity_path": valid_root / "sample_integrity_report.json",
        "valid_log_path": valid_root.parent / "full.log",
        "valid_errors_path": valid_root.parent / "errors.log",
        "test_eval_path": test_root / "eval_summary.json",
        "test_protocol_path": test_root / "eval_protocol.json",
        "test_integrity_path": test_root / "sample_integrity_report.json",
        "test_log_path": test_root.parent / "full.log",
        "test_errors_path": test_root.parent / "errors.log",
    }


def _extract_metrics(eval_summary: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    metrics = eval_summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise Step3EvalHandoffError(f"{split} eval_summary.metrics is required")
    recommendation = metrics.get("recommendation")
    explanation = metrics.get("explanation")
    rouge = explanation.get("rouge") if isinstance(explanation, Mapping) else None
    bleu = explanation.get("bleu") if isinstance(explanation, Mapping) else None
    dist = explanation.get("dist") if isinstance(explanation, Mapping) else None
    if not isinstance(recommendation, Mapping) or not isinstance(rouge, Mapping):
        raise Step3EvalHandoffError(f"{split} eval metrics are incomplete")
    if not isinstance(bleu, Mapping) or not isinstance(dist, Mapping) or not isinstance(explanation, Mapping):
        raise Step3EvalHandoffError(f"{split} eval text metrics are incomplete")
    out = {
        "MAE": recommendation.get("mae"),
        "RMSE": recommendation.get("rmse"),
        "ROUGE-1": rouge.get("1"),
        "ROUGE-L": rouge.get("l"),
        "BLEU-1": bleu.get("1"),
        "BLEU-2": bleu.get("2"),
        "BLEU-3": bleu.get("3"),
        "BLEU-4": bleu.get("4"),
        "DIST-1": dist.get("1"),
        "DIST-2": dist.get("2"),
        "METEOR": explanation.get("meteor"),
    }
    missing = [key for key in REQUIRED_METRIC_KEYS if out.get(key) in (None, "")]
    if missing:
        raise Step3EvalHandoffError(f"{split} eval metrics missing: {', '.join(missing)}")
    return out


def _validate_eval_bundle(
    *,
    split: str,
    eval_path: Path,
    protocol_path: Path,
    integrity_path: Path,
    log_paths: Iterable[Path],
) -> dict[str, Any]:
    summary = _load_json(eval_path, context=f"{split} eval_summary.json")
    protocol = _load_json(protocol_path, context=f"{split} eval_protocol.json")
    integrity = _load_json(integrity_path, context=f"{split} sample_integrity_report.json")
    _require(str(summary.get("eval_status")) == "completed", f"{split} eval_status must be completed")
    _require(str(summary.get("eval_protocol")) == PAPER_TARGET_ONLY_EVAL, f"{split} eval_protocol mismatch")
    _require(str(protocol.get("protocol")) == PAPER_TARGET_ONLY_EVAL, f"{split} protocol mismatch")
    _require(bool(summary.get("target_only")) is True, f"{split} eval_summary.target_only must be true")
    _require(bool(protocol.get("target_only")) is True, f"{split} protocol.target_only must be true")
    _require(bool(summary.get("bertscore_enabled")) is False, f"{split} eval_summary BERTScore must be disabled")
    _require(bool(protocol.get("bertscore_enabled")) is False, f"{split} protocol BERTScore must be disabled")
    _require(int(summary.get("max_ref_len")) == 25, f"{split} max_ref_len must be 25")
    _require(int(summary.get("max_decode_len")) == 25, f"{split} max_decode_len must be 25")
    _require(int(protocol.get("max_ref_len")) == 25, f"{split} protocol max_ref_len must be 25")
    _require(int(protocol.get("max_decode_len")) == 25, f"{split} protocol max_decode_len must be 25")
    _require(str(integrity.get("status")) == "PASS", f"{split} sample integrity must be PASS")
    _require(bool(integrity.get("count_match")) is True, f"{split} sample integrity count_match must be true")
    metrics = _extract_metrics(summary, split=split)
    log_hits: list[str] = []
    for log_path in log_paths:
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"WorkNCCL|Timeout\(ms\)|NCCL[^\n]{0,120}timeout", text, flags=re.IGNORECASE):
            log_hits.append(str(log_path))
    _require(not log_hits, f"{split} eval-only logs contain NCCL timeout markers: {log_hits}")
    return {
        "summary": summary,
        "protocol": protocol,
        "integrity": integrity,
        "metrics": metrics,
        "sample_count": int(summary.get("sample_count") or integrity.get("sample_count") or 0),
        "eval_batch": int(summary.get("eval_batch_global") or summary.get("selected_eval_batch") or 0),
    }


def _registry_entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Step3EvalHandoffError(f"eval_registry.jsonl missing: {path}")
    entries: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Step3EvalHandoffError(f"eval_registry.jsonl line {line_no} is invalid JSON") from exc
        if isinstance(item, dict):
            entries.append(item)
    return entries


def _registry_has(entries: Iterable[Mapping[str, Any]], *, split: str, eval_path: Path, metrics: Mapping[str, Any]) -> bool:
    for item in entries:
        if str(item.get("pipeline")) != "Step3_two_phase_paper_target_only_eval":
            continue
        if f"split={split}" not in str(item.get("task_description", "")):
            continue
        log_file = str(item.get("log_file") or "")
        if split not in log_file:
            continue
        if Path(log_file).name != "full.log":
            continue
        try:
            mae = float(item.get("mae"))
            rmse = float(item.get("rmse"))
        except (TypeError, ValueError):
            continue
        if abs(mae - float(metrics["MAE"])) < 1e-9 and abs(rmse - float(metrics["RMSE"])) < 1e-9:
            return True
    return False


def _lineage_hash(path: Path) -> str:
    lineage_path = Path(str(path) + ".lineage.json")
    lineage = _load_json(lineage_path, context=f"{path.name}.lineage.json")
    expected_hash = str(lineage.get("checkpoint_file_hash") or "")
    actual_hash = checkpoint_file_sha256(path)
    _require(expected_hash == actual_hash, f"{path.name} lineage hash mismatch: {expected_hash!r} != {actual_hash!r}")
    return actual_hash


def validate_step3_eval_handoff_evidence(
    *,
    repo_root: str | Path,
    task_id: int,
    run_id: str,
    require_test: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run = path_layout.get_stage_run_root(root, int(task_id), "v1", "step3", str(run_id)).resolve()
    meta = run / "meta"
    model = run / "model"
    summary_path = meta / "run_summary.json"
    run_summary = _load_json(summary_path, context="run_summary.json")
    checkpoint = model / "best_observed.pth"
    _require(checkpoint.is_file(), f"best_observed checkpoint missing: {checkpoint}")
    checkpoint_hash = _lineage_hash(checkpoint)
    sibling_hashes = {
        name: _lineage_hash(model / name)
        for name in ("best_observed.pth", "best.pth", "latest.pth")
        if (model / name).is_file()
    }
    best_alias_hashes = {
        name: value
        for name, value in sibling_hashes.items()
        if name in {"best_observed.pth", "best.pth"}
    }
    _require(
        all(value == checkpoint_hash for value in best_alias_hashes.values()),
        f"best checkpoint alias hash mismatch: {best_alias_hashes}",
    )
    latest_hash = sibling_hashes.get("latest.pth")
    _require(
        latest_hash != checkpoint_hash,
        "latest.pth unexpectedly matches best_observed; handoff must explicitly preserve latest as non-selected unless it is reselected by paper-aware policy",
    )
    paths = default_eval_paths(run)
    valid = _validate_eval_bundle(
        split="valid",
        eval_path=paths["valid_eval_path"],
        protocol_path=paths["valid_protocol_path"],
        integrity_path=paths["valid_integrity_path"],
        log_paths=(paths["valid_log_path"], paths["valid_errors_path"]),
    )
    test: dict[str, Any] | None = None
    if require_test:
        test = _validate_eval_bundle(
            split="test",
            eval_path=paths["test_eval_path"],
            protocol_path=paths["test_protocol_path"],
            integrity_path=paths["test_integrity_path"],
            log_paths=(paths["test_log_path"], paths["test_errors_path"]),
        )
    registry_path = root / "runs" / f"task{int(task_id)}" / "meta" / "eval_registry.jsonl"
    registry = _registry_entries(registry_path)
    _require(
        _registry_has(registry, split="valid", eval_path=paths["valid_eval_path"], metrics=valid["metrics"]),
        "eval_registry.jsonl is missing the matching full valid paper_target_only_eval entry",
    )
    if require_test and test is not None:
        _require(
            _registry_has(registry, split="test", eval_path=paths["test_eval_path"], metrics=test["metrics"]),
            "eval_registry.jsonl is missing the matching full test paper_target_only_eval entry",
        )
    return {
        "repo_root": root,
        "run_root": run,
        "meta_dir": meta,
        "run_summary_path": summary_path,
        "run_summary": run_summary,
        "checkpoint": checkpoint,
        "checkpoint_hash": checkpoint_hash,
        "sibling_checkpoint_hashes": sibling_hashes,
        "latest_checkpoint_hash": latest_hash,
        "paths": paths,
        "valid": valid,
        "test": test,
        "registry_path": registry_path,
    }


def _failure_type(summary: Mapping[str, Any]) -> str:
    text = " ".join(
        str(summary.get(key) or "")
        for key in ("fatal_signature", "latest_error", "failure_phase")
    )
    root = summary.get("failure_root_signature")
    if isinstance(root, Mapping):
        text += " " + " ".join(str(root.get(key) or "") for key in ("fatal_signature", "failure_phase"))
    if "NCCL" in text or "WorkNCCL" in text or "Timeout(ms)" in text:
        return "nccl_timeout"
    return "unknown"


def _existing_failure_history(summary: Mapping[str, Any], *, accepted_at: str) -> list[dict[str, Any]]:
    existing = summary.get("failure_history")
    history = [dict(item) for item in existing if isinstance(item, Mapping)] if isinstance(existing, list) else []
    if history:
        return history
    if str(summary.get("status") or "").lower() not in {"failed", "partial", "interrupted"}:
        return history
    failure_root = summary.get("failure_root_signature")
    raw_failure_phase = (
        str(failure_root.get("failure_phase") or "")
        if isinstance(failure_root, Mapping)
        else str(summary.get("failure_phase") or "")
    )
    history.append(
        {
            "preserved_at": accepted_at,
            "status": summary.get("status"),
            "failure_phase": "post_train_eval",
            "raw_failure_phase": raw_failure_phase or None,
            "failure_type": _failure_type(summary),
            "fatal_signature": summary.get("fatal_signature"),
            "latest_error": summary.get("latest_error"),
            "failure_root_signature": failure_root if isinstance(failure_root, Mapping) else None,
            "training_loop_started": summary.get("training_loop_started"),
            "checkpoint_created": summary.get("checkpoint_created"),
            "source": "pre_eval_handoff_run_summary",
        }
    )
    return history


def build_eval_handoff_payload(
    evidence: Mapping[str, Any],
    *,
    task_id: int,
    run_id: str,
    accepted_at: str,
    accepted_by_tool: str,
) -> dict[str, Any]:
    root = Path(evidence["repo_root"])
    run_summary = evidence["run_summary"]
    valid = evidence["valid"]
    test = evidence.get("test")
    paths = evidence["paths"]
    checkpoint = Path(evidence["checkpoint"])
    payload = {
        "schema_version": EVAL_HANDOFF_SCHEMA_VERSION,
        "accepted_at": accepted_at,
        "accepted_by_tool": accepted_by_tool,
        "task_id": int(task_id),
        "run_id": str(run_id),
        "checkpoint_path": _repo_relative(root, checkpoint),
        "checkpoint_hash": evidence["checkpoint_hash"],
        "checkpoint_scope": "best_observed",
        "train_status": "completed",
        "original_run_summary_status": str(run_summary.get("status") or ""),
        "original_failure_phase": "post_train_eval",
        "original_failure_type": _failure_type(run_summary),
        "eval_status": "completed",
        "paper_eval_status": "completed",
        "paper_eval_protocol": PAPER_TARGET_ONLY_EVAL,
        "valid_eval_path": _repo_relative(root, paths["valid_eval_path"]),
        "test_eval_path": _repo_relative(root, paths["test_eval_path"]) if test is not None else None,
        "valid_metrics": valid["metrics"],
        "test_metrics": test["metrics"] if isinstance(test, Mapping) else None,
        "sample_integrity_valid": valid["integrity"],
        "sample_integrity_test": test["integrity"] if isinstance(test, Mapping) else None,
        "no_bertscore": True,
        "target_only": True,
        "max_ref_len": 25,
        "max_decode_len": 25,
        "paper_comparable_single_run": True,
        "caveats": [
            "single_run_not_5_seed_mean",
            "step3_eval_not_full_step4_step5_pipeline",
        ],
        "downstream_gate_recommendation": "ready_for_step4_preparation",
        "old_failure_history_preserved": True,
        "split_protocol_version": {
            "valid": valid["protocol"].get("schema_version"),
            "test": test["protocol"].get("schema_version") if isinstance(test, Mapping) else None,
        },
        "sample_counts": {
            "valid": valid["sample_count"],
            "test": test["sample_count"] if isinstance(test, Mapping) else None,
        },
        "checkpoint_sibling_hashes": evidence.get("sibling_checkpoint_hashes") or {},
        "latest_checkpoint_hash": evidence.get("latest_checkpoint_hash"),
        "latest_checkpoint_selected_downstream": False,
        "eval_registry_path": _repo_relative(root, evidence["registry_path"]),
        "single_run_caveat_recorded": True,
    }
    return payload


def _updated_step3_eval_status(evidence: Mapping[str, Any], handoff_rel: str, *, accepted_at: str) -> dict[str, Any]:
    meta = Path(evidence["meta_dir"])
    status_path = meta / "step3_eval_status.json"
    status = _load_json(status_path, context="step3_eval_status.json") if status_path.is_file() else {}
    status.update(
        {
            "schema_version": status.get("schema_version") or "odcr_step3_eval_status/combined_eval_only/1",
            "updated_at": accepted_at,
            "train_status": "completed",
            "eval_status": "completed",
            "eval_only_status": "completed",
            "paper_eval_status": "completed",
            "quality_status": "paper_evaluated",
            "downstream_ready": True,
            "downstream_ready_reason": "paper_target_only_eval_accepted",
            "eval_handoff_status": "accepted",
            "accepted_eval_handoff_path": handoff_rel,
        }
    )
    return status


def _updated_run_summary(evidence: Mapping[str, Any], handoff_rel: str, handoff: Mapping[str, Any], *, accepted_at: str) -> dict[str, Any]:
    root = Path(evidence["repo_root"])
    run_summary = dict(evidence["run_summary"])
    history = _existing_failure_history(run_summary, accepted_at=accepted_at)
    checkpoint = Path(evidence["checkpoint"])
    checkpoint_rel = _repo_relative(root, checkpoint)
    run_summary.update(
        {
            "status": "completed_with_eval_handoff",
            "train_status": "completed",
            "post_train_eval_status": "failed",
            "eval_only_status": "completed",
            "paper_eval_status": "completed",
            "quality_status": "paper_evaluated",
            "downstream_ready": True,
            "downstream_ready_reason": "paper_target_only_eval_accepted",
            "failure_history": history,
            "accepted_eval_handoff_path": handoff_rel,
            "selected_checkpoint": checkpoint_rel,
            "selected_checkpoint_hash": handoff["checkpoint_hash"],
            "selected_checkpoint_scope": "best_observed",
            "selected_downstream_checkpoint": checkpoint_rel,
            "selected_downstream_checkpoint_hash": handoff["checkpoint_hash"],
            "selected_downstream_checkpoint_scope": "best_observed",
            "paper_eval_valid_metrics": handoff["valid_metrics"],
            "paper_eval_test_metrics": handoff["test_metrics"],
            "paper_comparable_single_run": True,
            "full_pipeline_final": False,
            "eval_handoff_status": "accepted",
            "quality_gate_version": EVAL_HANDOFF_GATE_VERSION,
            "old_failure_history_preserved": True,
            "updated_at": accepted_at,
        }
    )
    return run_summary


def _updated_latest(evidence: Mapping[str, Any], *, accepted_at: str) -> dict[str, Any]:
    root = Path(evidence["repo_root"])
    run = Path(evidence["run_root"])
    latest_path = run.parent / "latest.json"
    latest = _load_json(latest_path, context="latest.json") if latest_path.is_file() else {}
    history = latest.get("original_latest_status_history")
    if not isinstance(history, list):
        history = []
    old_status = str(latest.get("latest_status") or "").strip()
    if old_status and old_status != "completed_with_eval_handoff":
        history.append(
            {
                "latest_status": old_status,
                "updated_at": latest.get("updated_at"),
                "preserved_at": accepted_at,
                "reason": "eval_handoff_accept_preserved_original_latest_status",
            }
        )
    latest.update(
        {
            "schema_version": "odcr_latest_pointer/active_stage_status/1",
            "active_run_id": str(evidence["run_summary"].get("run_id") or run.name),
            "latest_run_id": str(evidence["run_summary"].get("run_id") or run.name),
            "latest_run_dir": _repo_relative(root, run),
            "latest_summary_path": _repo_relative(root, Path(evidence["run_summary_path"])),
            "latest_stage_status_path": _repo_relative(root, run / "meta" / "stage_status.json"),
            "eval_handoff_status": "accepted",
            "original_latest_status_history": history,
            "updated_at": accepted_at,
            "status_claim_source": "stage_status_strict_verifier",
        }
    )
    latest.pop("latest_status", None)
    return latest


def accept_step3_eval_handoff(
    *,
    repo_root: str | Path,
    task_id: int,
    run_id: str,
    dry_run: bool = False,
    require_test: bool = True,
    accepted_by_tool: str = "./odcr step3 --accept-eval-only",
) -> dict[str, Any]:
    evidence = validate_step3_eval_handoff_evidence(
        repo_root=repo_root,
        task_id=int(task_id),
        run_id=str(run_id),
        require_test=require_test,
    )
    accepted_at = _now_iso()
    root = Path(evidence["repo_root"])
    meta = Path(evidence["meta_dir"])
    handoff = build_eval_handoff_payload(
        evidence,
        task_id=int(task_id),
        run_id=str(run_id),
        accepted_at=accepted_at,
        accepted_by_tool=accepted_by_tool,
    )
    handoff_path = meta / "eval_handoff.json"
    handoff_rel = _repo_relative(root, handoff_path)
    run_summary = _updated_run_summary(evidence, handoff_rel, handoff, accepted_at=accepted_at)
    eval_status = _updated_step3_eval_status(evidence, handoff_rel, accepted_at=accepted_at)
    latest = _updated_latest(evidence, accepted_at=accepted_at)
    result = {
        "schema_version": EVAL_HANDOFF_GATE_VERSION,
        "dry_run": bool(dry_run),
        "accepted": True,
        "task_id": int(task_id),
        "run_id": str(run_id),
        "handoff_path": handoff_rel,
        "run_summary_path": _repo_relative(root, evidence["run_summary_path"]),
        "latest_path": _repo_relative(root, Path(evidence["run_root"]).parent / "latest.json"),
        "checkpoint_hash": handoff["checkpoint_hash"],
        "valid_metrics": handoff["valid_metrics"],
        "test_metrics": handoff["test_metrics"],
        "downstream_ready": True,
        "downstream_ready_reason": "paper_target_only_eval_accepted",
        "would_write": [
            handoff_rel,
            _repo_relative(root, meta / "step3_eval_status.json"),
            _repo_relative(root, evidence["run_summary_path"]),
            _repo_relative(root, Path(evidence["run_root"]).parent / "latest.json"),
        ],
    }
    if dry_run:
        result["preview"] = {
            "eval_handoff": handoff,
            "run_summary_status": run_summary.get("status"),
            "latest_status": latest.get("latest_status"),
        }
        return result
    atomic_write_json(handoff_path, handoff)
    atomic_write_json(meta / "step3_eval_status.json", eval_status)
    atomic_write_json(Path(evidence["run_summary_path"]), run_summary)
    atomic_write_json(Path(evidence["run_root"]).parent / "latest.json", latest)
    from odcr_core.stage_status import build_and_write_stage_status

    build_and_write_stage_status(
        repo_root=root,
        stage="step3",
        task=int(task_id),
        run_id=str(run_id),
    )
    return result


def load_eval_handoff(run_root: str | Path, *, required: bool = True) -> dict[str, Any]:
    path = Path(run_root).expanduser().resolve() / "meta" / "eval_handoff.json"
    if not path.is_file():
        if required:
            raise Step3EvalHandoffError(f"Step3 eval handoff sidecar missing: {path}")
        return {}
    return _load_json(path, context="eval_handoff.json")


def validate_accepted_eval_handoff(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    handoff = load_eval_handoff(root)
    _require(str(handoff.get("schema_version")) == EVAL_HANDOFF_SCHEMA_VERSION, "eval_handoff schema mismatch")
    _require(str(handoff.get("train_status")) == "completed", "eval_handoff train_status must be completed")
    _require(str(handoff.get("paper_eval_status")) == "completed", "eval_handoff paper_eval_status must be completed")
    _require(str(handoff.get("paper_eval_protocol")) == PAPER_TARGET_ONLY_EVAL, "eval_handoff protocol mismatch")
    _require(bool(handoff.get("old_failure_history_preserved")) is True, "eval_handoff must preserve old failure history")
    _require(bool(handoff.get("target_only")) is True, "eval_handoff target_only must be true")
    _require(bool(handoff.get("no_bertscore")) is True, "eval_handoff no_bertscore must be true")
    _require(int(handoff.get("max_ref_len")) == 25, "eval_handoff max_ref_len must be 25")
    _require(int(handoff.get("max_decode_len")) == 25, "eval_handoff max_decode_len must be 25")
    for split in ("valid", "test"):
        metrics = handoff.get(f"{split}_metrics")
        _require(isinstance(metrics, Mapping), f"eval_handoff {split}_metrics missing")
        missing = [key for key in REQUIRED_METRIC_KEYS if metrics.get(key) in (None, "")]
        _require(not missing, f"eval_handoff {split}_metrics missing: {', '.join(missing)}")
        integrity = handoff.get(f"sample_integrity_{split}")
        _require(isinstance(integrity, Mapping), f"eval_handoff sample_integrity_{split} missing")
        _require(str(integrity.get("status")) == "PASS", f"eval_handoff sample_integrity_{split} must be PASS")
    checkpoint = _repo_path(root.parents[3] if root.name else Path.cwd(), handoff.get("checkpoint_path"), context="checkpoint_path")
    expected_hash = str(handoff.get("checkpoint_hash") or "")
    actual_hash = checkpoint_file_sha256(checkpoint)
    _require(expected_hash == actual_hash, f"eval_handoff checkpoint hash mismatch: {expected_hash!r} != {actual_hash!r}")
    return handoff


def quality_audit_from_eval_handoff(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    handoff = validate_accepted_eval_handoff(root)
    checkpoint = _repo_path(root.parents[3], handoff.get("checkpoint_path"), context="checkpoint_path")
    return {
        "schema_version": "odcr_step3_quality_audit_from_eval_handoff/1",
        "quality_gate_version": EVAL_HANDOFF_GATE_VERSION,
        "quality_status": "paper_evaluated",
        "downstream_ready": True,
        "quality_block_reasons": [],
        "quality_warnings": list(handoff.get("caveats") or []),
        "selected_downstream_checkpoint": str(checkpoint),
        "selected_downstream_checkpoint_hash": handoff.get("checkpoint_hash"),
        "selected_downstream_checkpoint_scope": "best_observed",
        "paper_eval_status": "completed",
        "paper_eval_protocol": PAPER_TARGET_ONLY_EVAL,
        "paper_comparable_single_run": True,
        "full_pipeline_final": False,
        "accepted_eval_handoff_path": str(root / "meta" / "eval_handoff.json"),
    }
