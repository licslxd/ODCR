"""Step5A rating-only eval handoff aggregation."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from odcr_eval_metrics import CODE1_COMPATIBLE_RATING_PROTOCOL_ID, code1_compatible_rating_protocol
from odcr_core import path_layout
from odcr_core.file_atomic import atomic_write_json
from odcr_core.manifests import write_run_summary_json
from odcr_core.stage_status import build_and_write_stage_status
from odcr_core.training_checkpoint import checkpoint_file_sha256


STEP5A_RATING_HANDOFF_SCHEMA_VERSION = "odcr_step5A_rating_eval_handoff/2"
STEP5A_RATING_METRICS_SCHEMA_VERSION = "odcr_step5A_rating_metrics/1"
STEP5A_RATING_COMPAT_SCHEMA_VERSION = "odcr_step5A_rating_metric_compatibility/1"
STEP5A_RATING_BATCH_DIAGNOSTIC_SCHEMA_VERSION = "odcr_step5A_rating_batch_diagnostic/2"


class Step5RatingHandoffError(RuntimeError):
    """Raised when a Step5A rating-only eval cannot be promoted to handoff."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_relative(root: Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Step5RatingHandoffError(f"missing required JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Step5RatingHandoffError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Step5RatingHandoffError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_csv_data_rows(path: Path) -> int:
    if not path.is_file():
        raise Step5RatingHandoffError(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return max(0, sum(1 for _line in fh) - 1)


def _count_text_lines(path: Path) -> int:
    if not path.is_file():
        raise Step5RatingHandoffError(f"missing text artifact: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _line in fh)


def _split_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Step5RatingHandoffError(f"missing split CSV: {path}")
    row_count = 0
    rating_min: float | None = None
    rating_max: float | None = None
    null_counts = {"rating": 0, "user_idx": 0, "item_idx": 0}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        for required in ("rating", "user_idx", "item_idx"):
            if required not in fields:
                raise Step5RatingHandoffError(f"split CSV missing {required!r}: {path}")
        for row in reader:
            row_count += 1
            for key in null_counts:
                if str(row.get(key, "")).strip() == "":
                    null_counts[key] += 1
            raw_rating = str(row.get("rating", "")).strip()
            if raw_rating:
                val = float(raw_rating)
                rating_min = val if rating_min is None else min(rating_min, val)
                rating_max = val if rating_max is None else max(rating_max, val)
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "source_row_count": int(row_count),
        "rating_min": rating_min,
        "rating_max": rating_max,
        "null_counts": null_counts,
    }


def _metric_value(payload: Mapping[str, Any], key: str) -> float:
    containers: Iterable[Mapping[str, Any]] = (
        payload.get("rating_metrics") if isinstance(payload.get("rating_metrics"), Mapping) else {},
        payload.get("recommendation") if isinstance(payload.get("recommendation"), Mapping) else {},
        (payload.get("paper_metrics") or {}).get("recommendation")
        if isinstance(payload.get("paper_metrics"), Mapping)
        and isinstance((payload.get("paper_metrics") or {}).get("recommendation"), Mapping)
        else {},
        payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {},
    )
    for container in containers:
        if key in container and container.get(key) is not None:
            return float(container.get(key))
    raise Step5RatingHandoffError(f"rating metric {key!r} missing")


def _split_metric_block(metrics: Mapping[str, Any]) -> dict[str, Any]:
    rating_metrics = metrics.get("rating_metrics") if isinstance(metrics.get("rating_metrics"), Mapping) else {}
    sample_count = int(metrics.get("sample_count") or rating_metrics.get("sample_count") or 0)
    return {
        "sample_count": sample_count,
        "mae": _metric_value(metrics, "mae"),
        "rmse": _metric_value(metrics, "rmse"),
        "metrics_path": str(metrics.get("metrics_path") or ""),
    }


def _path_summary(root: Path, raw: str | Path | None, *, expected_rows: int | None = None) -> dict[str, Any]:
    if raw is None or str(raw).strip() == "":
        return {"path": None, "exists": False, "sha256": None, "row_count": None}
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise Step5RatingHandoffError(f"required artifact missing: {path}")
    suffix = path.suffix.lower()
    row_count: int | None = None
    if suffix == ".csv":
        row_count = _count_csv_data_rows(path)
    elif suffix in {".jsonl", ".ndjson"}:
        row_count = _count_text_lines(path)
    if expected_rows is not None and row_count is not None and int(row_count) != int(expected_rows):
        raise Step5RatingHandoffError(
            f"artifact row_count mismatch for {path}: {row_count} != {expected_rows}"
        )
    return {
        "path": _repo_relative(root, path),
        "exists": True,
        "sha256": _sha256(path),
        "row_count": row_count,
        "bytes": path.stat().st_size,
    }


def _prediction_artifacts(root: Path, metrics: Mapping[str, Any], *, expected_rows: int) -> dict[str, Any]:
    artifacts = metrics.get("prediction_artifacts") if isinstance(metrics.get("prediction_artifacts"), Mapping) else {}
    csv_path = artifacts.get("csv")
    jsonl_path = artifacts.get("jsonl")
    if not csv_path and not jsonl_path:
        raise Step5RatingHandoffError(
            "rating eval handoff requires prediction artifacts or digest; missing prediction_artifacts"
        )
    out = {
        "csv": _path_summary(root, csv_path, expected_rows=expected_rows) if csv_path else None,
        "jsonl": _path_summary(root, jsonl_path, expected_rows=expected_rows) if jsonl_path else None,
    }
    if out["csv"] is None and out["jsonl"] is None:
        raise Step5RatingHandoffError("no readable prediction artifacts found")
    return out


def _digest_artifact(root: Path, eval_dir: Path, split: str, *, expected_rows: int) -> dict[str, Any]:
    digest_path = eval_dir / f"rating_{split}_digest.json"
    payload = _load_json(digest_path)
    if payload.get("metric_protocol") != CODE1_COMPATIBLE_RATING_PROTOCOL_ID:
        raise Step5RatingHandoffError(f"{split} digest metric protocol mismatch")
    digest_count = int(payload.get("sample_count") or 0)
    if digest_count != int(expected_rows):
        raise Step5RatingHandoffError(f"{split} digest sample_count mismatch: {digest_count} != {expected_rows}")
    return {
        "path": _repo_relative(root, digest_path),
        "sha256": _sha256(digest_path),
        "sample_count": digest_count,
        "recommendation": dict(payload.get("recommendation") or {}),
        "no_generate": bool(payload.get("no_generate")),
    }


def finalize_step5A_rating_eval_handoff(
    *,
    repo_root: str | Path,
    task: int,
    source_run_id: str,
    checkpoint: str | Path,
    valid_metrics_path: str | Path,
    test_metrics_path: str | Path,
    valid_file: str | Path,
    test_file: str | Path,
    expected_valid_count: int | None = None,
    expected_test_count: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run_root = path_layout.get_stage_run_root(root, int(task), "v1", "step5", source_run_id).resolve()
    meta = run_root / "meta"
    eval_dir = run_root / "eval"
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise Step5RatingHandoffError(f"checkpoint missing: {checkpoint_path}")
    valid_metrics = _load_json(Path(valid_metrics_path).expanduser().resolve())
    test_metrics = _load_json(Path(test_metrics_path).expanduser().resolve())
    for split, payload in (("valid", valid_metrics), ("test", test_metrics)):
        if payload.get("metric_protocol") != CODE1_COMPATIBLE_RATING_PROTOCOL_ID:
            raise Step5RatingHandoffError(f"{split} metric protocol mismatch: {payload.get('metric_protocol')!r}")
        if payload.get("rating_only") is not True or payload.get("no_generate") is not True:
            raise Step5RatingHandoffError(f"{split} is not rating-only no-generate eval")
        if payload.get("generation_touched") is not False or payload.get("step5B_touched") is not False:
            raise Step5RatingHandoffError(f"{split} touched forbidden generation/Step5B surface")
    valid_split = _split_summary(Path(valid_file).expanduser().resolve())
    test_split = _split_summary(Path(test_file).expanduser().resolve())
    valid_block = _split_metric_block(valid_metrics)
    test_block = _split_metric_block(test_metrics)
    if expected_valid_count is not None and int(valid_block["sample_count"]) != int(expected_valid_count):
        raise Step5RatingHandoffError(
            f"valid eval sample_count mismatch: {valid_block['sample_count']} != {expected_valid_count}"
        )
    if expected_test_count is not None and int(test_block["sample_count"]) != int(expected_test_count):
        raise Step5RatingHandoffError(
            f"test eval sample_count mismatch: {test_block['sample_count']} != {expected_test_count}"
        )
    valid_predictions = _prediction_artifacts(root, valid_metrics, expected_rows=int(valid_block["sample_count"]))
    test_predictions = _prediction_artifacts(root, test_metrics, expected_rows=int(test_block["sample_count"]))
    valid_digest = _digest_artifact(root, eval_dir, "valid", expected_rows=int(valid_block["sample_count"]))
    test_digest = _digest_artifact(root, eval_dir, "test", expected_rows=int(test_block["sample_count"]))
    valid_block.update({"prediction_artifacts": valid_predictions, "digest_artifact": valid_digest})
    test_block.update({"prediction_artifacts": test_predictions, "digest_artifact": test_digest})

    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = checkpoint_file_sha256(checkpoint_path)
    protocol = code1_compatible_rating_protocol()
    aggregate_metrics = {
        "schema_version": STEP5A_RATING_METRICS_SCHEMA_VERSION,
        "head": "step5A",
        "task": int(task),
        "source_run_id": str(source_run_id),
        "checkpoint": _repo_relative(root, checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
        "metric_protocol_detail": protocol,
        "valid": valid_block,
        "test": test_block,
        "paper_comparable_single_run": True,
        "paper_comparable_mean_std": False,
        "paper_protocol_single_run": True,
        "paper_protocol_mean_std": False,
        "target_only": True,
        "batch_invariance_required": False,
        "batch_invariance_gate_removed": True,
        "batch_invariance_not_used_for_paper_metric": True,
        "generation_touched": False,
        "step5B_touched": False,
        "rerank_touched": False,
        "created_at": _utc_now(),
    }
    metrics_out_path = eval_dir / "rating_metrics.json"
    batch_report = {
        "schema_version": STEP5A_RATING_BATCH_DIAGNOSTIC_SCHEMA_VERSION,
        "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
        "valid": valid_metrics.get("batch_invariance"),
        "test": test_metrics.get("batch_invariance"),
        "passed": bool(
            (valid_metrics.get("batch_invariance") or {}).get("passed")
            and (test_metrics.get("batch_invariance") or {}).get("passed")
        ),
        "batch_invariance_required": False,
        "batch_invariance_gate_removed": True,
        "batch_invariance_not_used_for_paper_metric": True,
        "gate": "optional_historical_diagnostic",
    }
    batch_out_path = eval_dir / "rating_batch_invariance_report.json"
    compat_report = {
        "schema_version": STEP5A_RATING_COMPAT_SCHEMA_VERSION,
        "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
        **protocol,
        "task": int(task),
        "source_domain": "AM_Movies",
        "target_domain": "AM_CDs",
        "valid_file": _repo_relative(root, valid_file),
        "test_file": _repo_relative(root, test_file),
        "valid_split": valid_split,
        "test_split": test_split,
        "valid_role": "sanity/checkpoint-selection compatibility evidence",
        "test_role": "single-run paper-comparable rating result",
        "valid_metric_path": _repo_relative(root, valid_metrics_path),
        "test_metric_path": _repo_relative(root, test_metrics_path),
        "valid_prediction_artifacts": valid_block["prediction_artifacts"],
        "test_prediction_artifacts": test_block["prediction_artifacts"],
        "batch_invariance_required": False,
        "batch_invariance_gate_removed": True,
        "batch_invariance_not_used_for_paper_metric": True,
        "batch_invariance_report": _repo_relative(root, batch_out_path),
        "step5A_train_only_checkpoint_consumed": True,
        "combined_step5_ready": False,
        "step5B_touched": False,
        "generation_touched": False,
        "rerank_touched": False,
    }
    compat_out_path = eval_dir / "rating_metric_compatibility_report.json"
    handoff = {
        "schema_version": STEP5A_RATING_HANDOFF_SCHEMA_VERSION,
        "status": "ok",
        "task": int(task),
        "head": "step5A",
        "source_run_id": str(source_run_id),
        "checkpoint": _repo_relative(root, checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
        "code1_formula_matched": True,
        "paper_comparable_single_run": True,
        "paper_comparable_mean_std": False,
        "paper_protocol_single_run": True,
        "paper_protocol_mean_std": False,
        "valid": valid_block,
        "test": test_block,
        "prediction_artifact_paths": {
            "valid": valid_block["prediction_artifacts"],
            "test": test_block["prediction_artifacts"],
        },
        "metric_artifact_paths": {
            "aggregate_metrics": _repo_relative(root, metrics_out_path),
            "compatibility_report": _repo_relative(root, compat_out_path),
            "batch_invariance_report": _repo_relative(root, batch_out_path),
            "valid_metrics": _repo_relative(root, valid_metrics_path),
            "test_metrics": _repo_relative(root, test_metrics_path),
        },
        "batch_invariance_required": False,
        "batch_invariance_gate_removed": True,
        "batch_invariance_not_used_for_paper_metric": True,
        "batch_invariance_report": _repo_relative(root, batch_out_path),
        "split_file_paths": {
            "valid": _repo_relative(root, valid_file),
            "test": _repo_relative(root, test_file),
        },
        "split_file_hashes": {
            "valid": valid_split["sha256"],
            "test": test_split["sha256"],
        },
        "split_source_row_counts": {
            "valid": valid_split["source_row_count"],
            "test": test_split["source_row_count"],
        },
        "target_only": True,
        "step5B_touched": False,
        "combined_step5_ready": False,
        "rerank_touched": False,
        "generation_touched": False,
        "gpu_evidence": {
            "valid_eval_mode": valid_metrics.get("eval_mode"),
            "test_eval_mode": test_metrics.get("eval_mode"),
            "valid_runtime_env": (valid_metrics.get("eval_performance") or {}).get("runtime_env"),
            "test_runtime_env": (test_metrics.get("eval_performance") or {}).get("runtime_env"),
        },
        "created_at": _utc_now(),
    }
    handoff_out_path = meta / "eval_handoff.json"

    if dry_run:
        return {
            "status": "dry_run",
            "would_write": {
                "eval_handoff": _repo_relative(root, handoff_out_path),
                "rating_metrics": _repo_relative(root, metrics_out_path),
                "compatibility_report": _repo_relative(root, compat_out_path),
                "batch_invariance_report": _repo_relative(root, batch_out_path),
            },
            "batch_invariance_required": False,
            "batch_invariance_gate_removed": True,
            "valid": valid_block,
            "test": test_block,
            "checkpoint_hash": checkpoint_hash,
        }

    metrics_out = atomic_write_json(metrics_out_path, aggregate_metrics)
    batch_out = atomic_write_json(batch_out_path, batch_report)
    compat_out = atomic_write_json(compat_out_path, compat_report)
    handoff_out = atomic_write_json(handoff_out_path, handoff)

    summary_path = meta / "run_summary.json"
    summary = _load_json(summary_path)
    previous_status = summary.get("status")
    summary.update(
        {
            "status": "completed_with_eval_handoff",
            "validation_status": "ok",
            "eval_status": "completed",
            "needs_eval_handoff": False,
            "train_status": "completed",
            "downstream_ready": False,
            "ready_for": [],
            "step5A_rating_ready": True,
            "head_ready": True,
            "combined_step5_ready": False,
            "step5B_ready": False,
            "step5B_touched": False,
            "rerank_touched": False,
            "generation_touched": False,
            "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
            "paper_comparable_single_run": True,
            "paper_comparable_mean_std": False,
            "paper_protocol_single_run": True,
            "paper_protocol_mean_std": False,
            "batch_invariance_required": False,
            "batch_invariance_gate_removed": True,
            "batch_invariance_not_used_for_paper_metric": True,
            "rating_metrics_path": _repo_relative(root, metrics_out),
            "rating_metric_compatibility_report": _repo_relative(root, compat_out),
            "rating_batch_invariance_report": _repo_relative(root, batch_out),
            "eval_handoff": _repo_relative(root, handoff_out),
            "valid_mae": valid_block["mae"],
            "valid_rmse": valid_block["rmse"],
            "test_mae": test_block["mae"],
            "test_rmse": test_block["rmse"],
            "valid_sample_count": valid_block["sample_count"],
            "test_sample_count": test_block["sample_count"],
            "previous_status_before_eval_handoff": previous_status,
            "updated_at": _utc_now(),
        }
    )
    write_run_summary_json(summary, repo_root=root, update_latest=False)
    stage_status = build_and_write_stage_status(repo_root=root, stage="step5", task=int(task), run_id=str(source_run_id))
    return {
        "status": "ok",
        "eval_handoff": _repo_relative(root, handoff_out),
        "rating_metrics": _repo_relative(root, metrics_out),
        "compatibility_report": _repo_relative(root, compat_out),
        "batch_invariance_report": _repo_relative(root, batch_out),
        "batch_invariance_required": False,
        "batch_invariance_gate_removed": True,
        "stage_status": stage_status,
        "valid": valid_block,
        "test": test_block,
        "checkpoint_hash": checkpoint_hash,
    }
