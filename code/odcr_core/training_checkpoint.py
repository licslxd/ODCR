"""Canonical checkpoint and lineage-gate helpers.

``model/best.pth`` remains a plain state-dict file.  Compatibility metadata is
stored beside it under ``state/checkpoint_lineage.json`` and is treated as a
hard gate by downstream stages.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from odcr_core.file_atomic import atomic_write_json

LINEAGE_GATE_SCHEMA_VERSION = "odcr_lineage_gate/4A"
CHECKPOINT_LINEAGE_FILENAME = "checkpoint_lineage.json"
STEP5_TRAIN_SCHEMA_VERSION = "odcr_step5_train_schema/4A"
STEP5_EVAL_OUTPUT_SCHEMA_VERSION = "odcr_step5_eval_output/4A"
STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION = "odcr_step5_checkpoint_compat/4A"
MODEL_ARTIFACT_FINGERPRINT_VERSION = "odcr_model_artifact_fingerprint/1"
FILE_FINGERPRINT_VERSION = "odcr_file_fingerprint/1"
_SAMPLE_BYTES = 1024 * 1024


class CheckpointLineageError(RuntimeError):
    """Raised when a checkpoint/export lineage hard gate fails."""


def state_dict_for_canonical_best_pth(
    *,
    ema_enabled: bool,
    ema_model: Any,
    ddp_module: Any,
    underlying_model_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    if ema_enabled:
        if ema_model is None:
            raise RuntimeError(
                "ema_enabled=True 但 ema_model 未初始化；禁止将原始训练权重写入 model/best.pth。"
            )
        return ema_model.module.state_dict()
    return underlying_model_fn(ddp_module).state_dict()


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(data: Any, *, length: int = 32) -> str:
    return hashlib.sha256(stable_json_dumps(data).encode("utf-8")).hexdigest()[: int(length)]


def parse_json_object(raw: str | Mapping[str, Any] | None, *, context: str, required: bool = False) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        if required:
            raise CheckpointLineageError(f"{context} is required for ODCR lineage gate.")
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CheckpointLineageError(f"{context} must be valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise CheckpointLineageError(f"{context} JSON root must be an object.")
    return obj


def current_effective_payload(*, required: bool = False) -> dict[str, Any]:
    return parse_json_object(
        os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON"),
        context="ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON",
        required=required,
    )


def current_one_control_resolved_config_hash(*, extra: Mapping[str, Any] | None = None) -> str:
    payload = current_effective_payload(required=True)
    roots = {
        key: os.environ.get(key, "")
        for key in (
            "ODCR_RESOLVED_DATA_DIR",
            "ODCR_RESOLVED_MERGED_DIR",
            "ODCR_RESOLVED_MODELS_DIR",
            "ODCR_RESOLVED_STEP5_TEXT_MODEL",
            "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "ODCR_RESOLVED_EMBED_DIM",
        )
    }
    return stable_hash(
        {
            "schema_version": LINEAGE_GATE_SCHEMA_VERSION,
            "effective_training_payload": payload,
            "training_semantic_fingerprint": os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT", ""),
            "generation_semantic_fingerprint": os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT", ""),
            "runtime_diagnostics_fingerprint": os.environ.get("ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT", ""),
            "resolved_roots": roots,
            "extra": dict(extra or {}),
        }
    )


def file_fingerprint(path: str | Path, *, sample_only: bool = False) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"schema_version": FILE_FINGERPRINT_VERSION, "path": str(p), "exists": False}
    st = p.stat()
    out: dict[str, Any] = {
        "schema_version": FILE_FINGERPRINT_VERSION,
        "path": str(p),
        "exists": True,
        "is_file": p.is_file(),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }
    if not p.is_file():
        return out
    h = hashlib.sha256()
    with p.open("rb") as handle:
        if sample_only:
            head = handle.read(_SAMPLE_BYTES)
            h.update(head)
            if st.st_size > _SAMPLE_BYTES:
                handle.seek(max(int(st.st_size) - _SAMPLE_BYTES, 0))
                h.update(handle.read(_SAMPLE_BYTES))
            out["sample_sha256"] = h.hexdigest()
        else:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
            out["sha256"] = h.hexdigest()
    return out


def model_artifact_fingerprint(path: str | Path, *, selected_files: Iterable[str] | None = None) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    names = tuple(
        selected_files
        or (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
            "merges.txt",
            "pytorch_model.bin",
            "model.safetensors",
        )
    )
    if p.is_file():
        return {
            "schema_version": MODEL_ARTIFACT_FINGERPRINT_VERSION,
            "path": str(p),
            "kind": "file",
            "file": file_fingerprint(p, sample_only=True),
        }
    files = []
    if p.is_dir():
        for name in names:
            child = p / name
            if child.exists():
                files.append(file_fingerprint(child, sample_only=True))
    return {
        "schema_version": MODEL_ARTIFACT_FINGERPRINT_VERSION,
        "path": str(p),
        "kind": "directory" if p.is_dir() else "missing",
        "selected_files": files,
        "fingerprint_hash": stable_hash(files),
    }


def checkpoint_lineage_path_for_weight(checkpoint_path: str | Path) -> Path:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    run_root = ckpt.parent.parent
    return (run_root / "state" / CHECKPOINT_LINEAGE_FILENAME).resolve()


def _hashable_lineage_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("lineage_hash", None)
    out.pop("created_at_utc", None)
    return out


def attach_lineage_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("schema_version", LINEAGE_GATE_SCHEMA_VERSION)
    out["lineage_hash"] = stable_hash(_hashable_lineage_payload(out))
    return out


def write_checkpoint_lineage(checkpoint_path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = attach_lineage_hash(
        {
            "schema_version": LINEAGE_GATE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **dict(payload),
        }
    )
    path = checkpoint_lineage_path_for_weight(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, out)
    return path


def read_checkpoint_lineage(checkpoint_path: str | Path, *, expected_stage: str | None = None) -> dict[str, Any]:
    path = checkpoint_lineage_path_for_weight(checkpoint_path)
    if not path.is_file():
        raise CheckpointLineageError(
            f"Checkpoint lineage sidecar is missing: {path}. "
            "Refusing to reuse an old checkpoint; rerun the producing stage under Phase 4A lineage gates."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointLineageError(f"Checkpoint lineage sidecar is unreadable: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckpointLineageError(f"Checkpoint lineage sidecar root must be an object: {path}")
    if data.get("schema_version") != LINEAGE_GATE_SCHEMA_VERSION:
        raise CheckpointLineageError(
            f"Unsupported checkpoint lineage schema {data.get('schema_version')!r}; "
            f"expected {LINEAGE_GATE_SCHEMA_VERSION!r}. path={path}"
        )
    actual_hash = stable_hash(_hashable_lineage_payload(data))
    if data.get("lineage_hash") != actual_hash:
        raise CheckpointLineageError(
            f"Checkpoint lineage hash mismatch: stored={data.get('lineage_hash')} computed={actual_hash} path={path}"
        )
    if expected_stage and str(data.get("stage")) != str(expected_stage):
        raise CheckpointLineageError(
            f"Checkpoint lineage stage mismatch: stored={data.get('stage')!r} expected={expected_stage!r} path={path}"
        )
    return data


def require_equal(actual: Any, expected: Any, *, label: str, context: str) -> None:
    if actual != expected:
        raise CheckpointLineageError(
            f"{context} lineage mismatch for {label}: checkpoint/export={actual!r} current={expected!r}"
        )


def require_keys(payload: Mapping[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise CheckpointLineageError(f"{context} missing lineage fields: {', '.join(missing)}")


def payload_section_hash(section: str, *, payload: Mapping[str, Any] | None = None) -> str:
    obj = dict(payload or current_effective_payload(required=True))
    return stable_hash(obj.get(section) or {})
