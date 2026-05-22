from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from odcr_core.config_schema import OneControlConfigError

RATING_SOURCE_SCHEMA_VERSION = "odcr_rating_source/1"
RATING_SOURCE_TYPE = "step3_accepted_scorer"
STEP3_EVAL_HANDOFF_SCHEMA_VERSION = "odcr_step3_eval_handoff/1"


class RatingSourceError(OneControlConfigError):
    """Raised when the configured Step3 rating source is not acceptable."""


def _repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _repo_path(raw: str | Path, *, repo_root: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RatingSourceError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RatingSourceError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RatingSourceError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RatingSourceError(f"rating source checkpoint missing: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metric(payload: Mapping[str, Any], split: str, name: str) -> float:
    metrics = payload.get(f"{split}_metrics")
    if not isinstance(metrics, Mapping):
        raise RatingSourceError(f"rating source eval_handoff missing {split}_metrics")
    if name in metrics:
        return float(metrics[name])
    upper = name.upper()
    if upper in metrics:
        return float(metrics[upper])
    raise RatingSourceError(f"rating source eval_handoff missing {split} {name}")


def resolve_rating_source_config(raw: Mapping[str, Any] | None, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    if not isinstance(raw, Mapping):
        raise RatingSourceError("configs/odcr.yaml:rating_source must be a mapping")
    schema = str(raw.get("schema_version") or RATING_SOURCE_SCHEMA_VERSION).strip()
    if schema != RATING_SOURCE_SCHEMA_VERSION:
        raise RatingSourceError(f"rating_source.schema_version must be {RATING_SOURCE_SCHEMA_VERSION}")
    source_type = str(raw.get("type") or "").strip()
    if source_type != RATING_SOURCE_TYPE:
        raise RatingSourceError(f"rating_source.type must be {RATING_SOURCE_TYPE}")
    task = int(raw.get("task"))
    run = int(raw.get("run"))
    checkpoint = _repo_path(str(raw.get("checkpoint") or ""), repo_root=root)
    eval_handoff = _repo_path(str(raw.get("eval_handoff") or ""), repo_root=root)
    stage_status = raw.get("stage_status")
    status_path = _repo_path(str(stage_status), repo_root=root) if stage_status else eval_handoff.parent / "stage_status.json"
    out = {
        "schema_version": schema,
        "type": source_type,
        "task": task,
        "run": run,
        "checkpoint": _display_path(checkpoint, repo_root=root),
        "checkpoint_hash": str(raw.get("checkpoint_hash") or "").strip(),
        "eval_handoff": _display_path(eval_handoff, repo_root=root),
        "stage_status": _display_path(status_path, repo_root=root),
        "protocol": str(raw.get("protocol") or "").strip(),
        "valid_mae": float(raw.get("valid_mae")),
        "valid_rmse": float(raw.get("valid_rmse")),
        "test_mae": float(raw.get("test_mae")),
        "test_rmse": float(raw.get("test_rmse")),
    }
    if not out["checkpoint_hash"]:
        raise RatingSourceError("rating_source.checkpoint_hash is required")
    if out["protocol"] != "paper_target_only_eval":
        raise RatingSourceError("rating_source.protocol must be paper_target_only_eval")
    return out


def validate_rating_source(payload: Mapping[str, Any], *, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _repo_root(repo_root)
    resolved = resolve_rating_source_config(payload, repo_root=root)
    checkpoint = _repo_path(resolved["checkpoint"], repo_root=root)
    eval_handoff_path = _repo_path(resolved["eval_handoff"], repo_root=root)
    status_path = _repo_path(resolved["stage_status"], repo_root=root)
    actual_hash = _sha256_file(checkpoint)
    if actual_hash != resolved["checkpoint_hash"]:
        raise RatingSourceError(
            f"rating source checkpoint hash mismatch: expected {resolved['checkpoint_hash']} got {actual_hash}"
        )
    handoff = _load_json(eval_handoff_path, label="rating source eval_handoff")
    status = _load_json(status_path, label="rating source stage_status")
    if str(handoff.get("schema_version") or "") != STEP3_EVAL_HANDOFF_SCHEMA_VERSION:
        raise RatingSourceError("rating source eval_handoff schema is not odcr_step3_eval_handoff/1")
    if int(handoff.get("task_id")) != resolved["task"] or int(handoff.get("run_id")) != resolved["run"]:
        raise RatingSourceError("rating source eval_handoff task/run mismatch")
    if str(handoff.get("checkpoint_path") or "") != resolved["checkpoint"]:
        raise RatingSourceError("rating source eval_handoff checkpoint_path mismatch")
    if str(handoff.get("checkpoint_hash") or "") != resolved["checkpoint_hash"]:
        raise RatingSourceError("rating source eval_handoff checkpoint_hash mismatch")
    if str(handoff.get("paper_eval_protocol") or "") != resolved["protocol"]:
        raise RatingSourceError("rating source eval_handoff protocol mismatch")
    if str(handoff.get("paper_eval_status") or "").lower() not in {"completed", "accepted"}:
        raise RatingSourceError("rating source eval_handoff is not completed/accepted")
    if bool(handoff.get("target_only")) is not True:
        raise RatingSourceError("rating source eval_handoff must be target_only")
    metrics = {
        "valid": {"mae": _metric(handoff, "valid", "MAE"), "rmse": _metric(handoff, "valid", "RMSE")},
        "test": {"mae": _metric(handoff, "test", "MAE"), "rmse": _metric(handoff, "test", "RMSE")},
    }
    expected = {
        "valid": {"mae": resolved["valid_mae"], "rmse": resolved["valid_rmse"]},
        "test": {"mae": resolved["test_mae"], "rmse": resolved["test_rmse"]},
    }
    for split in ("valid", "test"):
        for key in ("mae", "rmse"):
            if abs(metrics[split][key] - expected[split][key]) > 1e-8:
                raise RatingSourceError(f"rating source {split} {key} mismatch")
    if str(status.get("final_status") or "") != "completed_with_eval_handoff":
        raise RatingSourceError("rating source stage_status final_status must be completed_with_eval_handoff")
    if bool(status.get("downstream_ready")) is not True:
        raise RatingSourceError("rating source stage_status downstream_ready must be true")
    if str(status.get("selected_checkpoint") or "") != resolved["checkpoint"]:
        raise RatingSourceError("rating source stage_status selected_checkpoint mismatch")
    if str(status.get("selected_checkpoint_hash") or "") != resolved["checkpoint_hash"]:
        raise RatingSourceError("rating source stage_status selected_checkpoint_hash mismatch")
    if str(status.get("eval_handoff") or "") != resolved["eval_handoff"]:
        raise RatingSourceError("rating source stage_status eval_handoff mismatch")
    return {
        **resolved,
        "status": "ok",
        "valid": True,
        "metrics": metrics,
        "artifacts": {
            "checkpoint": resolved["checkpoint"],
            "eval_handoff": resolved["eval_handoff"],
            "stage_status": resolved["stage_status"],
        },
    }


def rating_metrics_from_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "odcr_step3_rating_metrics_reference/1",
        "source_type": str(payload.get("type") or RATING_SOURCE_TYPE),
        "protocol": str(payload.get("protocol") or "paper_target_only_eval"),
        "checkpoint": str(payload.get("checkpoint") or ""),
        "checkpoint_hash": str(payload.get("checkpoint_hash") or ""),
        "eval_handoff": str(payload.get("eval_handoff") or ""),
        "valid": {
            "MAE": float((payload.get("metrics") or {}).get("valid", {}).get("mae", payload.get("valid_mae"))),
            "RMSE": float((payload.get("metrics") or {}).get("valid", {}).get("rmse", payload.get("valid_rmse"))),
        },
        "test": {
            "MAE": float((payload.get("metrics") or {}).get("test", {}).get("mae", payload.get("test_mae"))),
            "RMSE": float((payload.get("metrics") or {}).get("test", {}).get("rmse", payload.get("test_rmse"))),
        },
    }
