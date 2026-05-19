"""Canonical checkpoint and lineage-gate helpers.

``model/best.pth`` remains a plain state-dict file when retained as a
compatibility alias.  Compatibility metadata is stored beside each checkpoint
as ``*.lineage.json`` and summarized in ``state/checkpoint_lineage.json`` as an
event ledger treated as a hard gate by downstream stages.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from odcr_core.file_atomic import atomic_write_json

LINEAGE_GATE_SCHEMA_VERSION = "odcr_lineage_gate/4A"
CHECKPOINT_LINEAGE_FILENAME = "checkpoint_lineage.json"
CHECKPOINT_FILE_LINEAGE_SUFFIX = ".lineage.json"
CHECKPOINT_EVENT_LEDGER_SCHEMA_VERSION = "odcr_checkpoint_event_ledger/1"
STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION = "odcr_step3_checkpoint_compat/2"
STEP3_CHECKPOINT_MIN_ACCEPTED_SCHEMA_VERSION = STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION
STEP5_TRAIN_SCHEMA_VERSION = "odcr_step5_train_schema/4A"
STEP5_EVAL_OUTPUT_SCHEMA_VERSION = "odcr_step5_eval_output/4A"
STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION = "odcr_step5_checkpoint_compat/4A"
MODEL_ARTIFACT_FINGERPRINT_VERSION = "odcr_model_artifact_fingerprint/1"
FILE_FINGERPRINT_VERSION = "odcr_file_fingerprint/1"
_SAMPLE_BYTES = 1024 * 1024
STEP3_CHECKPOINT_REQUIRED_FIELDS = (
    "sidecar_schema_version",
    "stage",
    "run_id",
    "task_id",
    "source_domain",
    "target_domain",
    "task_profile_id",
    "profile_isolation_hash",
    "checkpoint_path",
    "checkpoint_file_hash",
    "checkpoint_epoch",
    "selection_metric",
    "selection_metric_value",
    "selection_direction",
    "selection_scope",
    "reason",
    "replaced_previous",
    "global_best_epoch",
    "global_best_metric",
    "after_min_epochs_best_epoch",
    "after_min_epochs_best_metric",
    "epoch_summary_hash",
    "metrics_jsonl_hash",
    "resolved_config_hash",
    "training_runtime_config_hash",
    "quality_status_at_save",
    "quality_status",
    "downstream_ready",
    "grad_inf_count_until_epoch",
    "model_file_hash",
    "code_commit",
    "created_at",
    "git_code_fingerprint",
    "one_control_resolved_config_path",
    "resolved_config",
    "one_control_resolved_config_hash",
    "resolved_config_compatibility_hash",
    "source_table_path",
    "source_table",
    "source_table_hash",
    "source_table_compatibility_hash",
    "source_table_payload_summary",
    "full_run_config_hash",
    "artifact_lineage_hash",
    "semantic_model_compat_hash",
    "data_contract_hash",
    "tokenizer_cache_compat_hash",
    "train_runtime_config_hash",
    "training_runtime_config_path",
    "training_runtime_config_hash",
    "training_runtime_config",
    "optimizer_config_hash",
    "performance_profile_hash",
    "loss_config_hash",
    "env",
    "embed_dim",
    "model_architecture_config_hash",
    "step3_structured_losses_config_hash",
    "step3_loss_semantics_config_hash",
    "step3_loss_semantics_config",
    "step3_optimizer_config",
    "step3_optimizer_config_hash",
    "step3_tokenizer_config",
    "step3_tokenizer_config_hash",
    "step3_evidence_config",
    "step3_evidence_config_hash",
    "step3_scheduler_config",
    "step3_scheduler_config_hash",
    "step3_valid_batch_config",
    "step3_valid_batch_config_hash",
    "step3_scenario_profile",
    "step3_scenario_profile_hash",
    "step3_task_profile",
    "step3_task_profile_hash",
    "ddp_config",
    "precision_config",
    "batch_semantics",
    "preprocess_latest_run_ids",
    "preprocess_run_summary_fingerprints",
    "preprocess_stage_status_fingerprints",
    "preprocess_stage_manifest_fingerprints",
    "preprocess_source_table_fingerprints",
    "preprocess_metrics_fingerprints",
    "preprocess_verify_report_fingerprints",
    "profile_artifact_fingerprints",
    "domain_artifact_fingerprints",
    "source_csv_fingerprints",
    "merged_csv_fingerprints",
    "sentence_embed_model_identity",
    "step3_tokenizer_cache_manifest",
    "schema_contract_versions",
    "compatibility_metadata",
    "metrics_summary",
    "checkpoint_compatibility_hash",
)
STEP3_SOURCE_TABLE_COMPAT_KEYS = (
    "task",
    "step3_structured_losses",
    "step3_loss_semantics",
    "step3_tokenizer",
    "step3_evidence",
    "step3_scenario_profile",
    "step3_task_profile",
    "profile_isolation_hash",
    "step3_cross_rank_structured_gather",
    "embed_dim",
)
STEP3_CHECKPOINT_RECORD_ONLY_FIELDS = (
    # Mirrors compatibility_metadata.record_only_fields written by Step3 sidecars.
    "batch_size",
    "micro_batch_size",
    "step3_optimizer_config_hash",
    "optimizer_config_hash",
    "train_runtime_config_hash",
    "training_runtime_config_hash",
    "performance_profile_hash",
    "step3_scheduler_config_hash",
    "step3_valid_batch_config_hash",
    "ddp_config_hash",
    "precision_config_hash",
    "batch_semantics_hash",
)
STEP3_MODEL_ARCHITECTURE_FIELDS = (
    "nuser",
    "nitem",
    "ntoken",
    "emsize",
    "nlayers",
    "nhead",
    "nhid",
    "dropout",
)
STEP3_MODEL_ARCHITECTURE_INT_FIELDS = frozenset(
    key for key in STEP3_MODEL_ARCHITECTURE_FIELDS if key != "dropout"
)
STEP3_CHECKPOINT_NULLABLE_REQUIRED_FIELDS = frozenset(
    {
        "after_min_epochs_best_epoch",
        "after_min_epochs_best_metric",
    }
)


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


def build_step3_model_architecture_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical Step3 checkpoint-load architecture payload.

    This scope is intentionally limited to fields that define the Step3 model
    module shapes/structure used by downstream checkpoint consumers. Step4 RCR,
    decode, runtime, evidence-level, hardware, batch, and guardrail/doc fields
    must stay outside this hash.
    """

    missing = [key for key in STEP3_MODEL_ARCHITECTURE_FIELDS if key not in payload]
    if missing:
        raise CheckpointLineageError(
            "Step3 model architecture payload missing fields: " + ", ".join(missing)
        )
    out: dict[str, Any] = {}
    for key in STEP3_MODEL_ARCHITECTURE_FIELDS:
        value = payload[key]
        if key in STEP3_MODEL_ARCHITECTURE_INT_FIELDS:
            out[key] = int(value)
        else:
            out[key] = float(value)
    return out


def compute_model_architecture_config_hash(payload: Mapping[str, Any]) -> str:
    return stable_hash(build_step3_model_architecture_config(payload))


def extract_checkpoint_model_architecture_payload(lineage: Mapping[str, Any]) -> dict[str, Any]:
    raw = lineage.get("model_architecture_config")
    if isinstance(raw, Mapping):
        arch = build_step3_model_architecture_config(raw)
    else:
        arch = build_step3_model_architecture_config(lineage)
    stored_hash = str(lineage.get("model_architecture_config_hash") or "")
    computed_hash = compute_model_architecture_config_hash(arch)
    if stored_hash and stored_hash != computed_hash:
        raise CheckpointLineageError(
            "Step3 checkpoint model_architecture_config_hash is internally inconsistent: "
            f"stored={stored_hash!r} computed={computed_hash!r}"
        )
    return arch


def _state_dict_tensor_shape(state: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = state.get(key)
    if value is None:
        value = state.get(f"module.{key}")
    if value is None or not hasattr(value, "shape"):
        raise CheckpointLineageError(f"Step3 checkpoint state_dict missing tensor: {key}")
    return tuple(int(dim) for dim in value.shape)


def extract_checkpoint_state_dict_architecture_payload(
    checkpoint_path: str | Path,
    *,
    fallback_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer the checkpoint-load architecture from actual state_dict shapes.

    Historical Step3 lineage sidecars can record configuration-derived values
    that are not identical to the tensors saved in the checkpoint. Downstream
    consumers need the tensor-derived payload for the hard checkpoint load
    compatibility gate; sidecar values remain useful for non-shape fields that
    cannot be inferred from tensors, such as ``nhead`` and ``dropout``.
    """

    fallback = build_step3_model_architecture_config(fallback_payload or {}) if fallback_payload else {}
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    try:
        import torch

        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exact torch errors vary by version.
        raise CheckpointLineageError(f"Step3 checkpoint state_dict could not be read for shape gate: {checkpoint}") from exc
    if not isinstance(state, Mapping):
        raise CheckpointLineageError(f"Step3 checkpoint state_dict root must be a mapping: {checkpoint}")

    user_shape = _state_dict_tensor_shape(state, "user_embeddings.weight")
    item_shape = _state_dict_tensor_shape(state, "item_embeddings.weight")
    word_shape = _state_dict_tensor_shape(state, "word_embeddings.weight")
    hidden_weight_shape = _state_dict_tensor_shape(state, "hidden2token.weight")
    hidden_bias_shape = _state_dict_tensor_shape(state, "hidden2token.bias")
    if len(user_shape) != 2 or len(item_shape) != 2 or len(word_shape) != 2:
        raise CheckpointLineageError("Step3 checkpoint embedding tensors must be rank-2 for architecture shape gate.")
    if len(hidden_weight_shape) != 2 or len(hidden_bias_shape) != 1:
        raise CheckpointLineageError("Step3 checkpoint hidden2token tensors have invalid ranks for architecture shape gate.")

    emsize = int(user_shape[1])
    consistency = {
        "item_embeddings.weight": int(item_shape[1]),
        "word_embeddings.weight": int(word_shape[1]),
        "hidden2token.weight": int(hidden_weight_shape[1]),
    }
    mismatched = {key: value for key, value in consistency.items() if value != emsize}
    if mismatched:
        raise CheckpointLineageError(
            "Step3 checkpoint embedding/decoder hidden sizes are inconsistent: "
            + ", ".join(f"{key}={value!r} expected={emsize!r}" for key, value in sorted(mismatched.items()))
        )
    ntoken = int(word_shape[0])
    if int(hidden_weight_shape[0]) != ntoken or int(hidden_bias_shape[0]) != ntoken:
        raise CheckpointLineageError(
            "Step3 checkpoint token projection shape is inconsistent: "
            f"word_embeddings={ntoken!r} hidden2token.weight={hidden_weight_shape[0]!r} "
            f"hidden2token.bias={hidden_bias_shape[0]!r}"
        )

    layer_ids: set[int] = set()
    for raw_key in state:
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        match = re.match(r"transformer_encoder\.layers\.(\d+)\.", key)
        if match:
            layer_ids.add(int(match.group(1)))
    nlayers = int(max(layer_ids) + 1) if layer_ids else int(fallback.get("nlayers", 0) or 0)
    if nlayers <= 0:
        raise CheckpointLineageError("Step3 checkpoint transformer layer count could not be inferred.")
    linear1_shape = _state_dict_tensor_shape(state, "transformer_encoder.layers.0.linear1.weight")
    linear2_shape = _state_dict_tensor_shape(state, "transformer_encoder.layers.0.linear2.weight")
    if len(linear1_shape) != 2 or len(linear2_shape) != 2:
        raise CheckpointLineageError("Step3 checkpoint transformer feed-forward tensors have invalid ranks.")
    nhid = int(linear1_shape[0])
    if int(linear1_shape[1]) != emsize or int(linear2_shape[0]) != emsize or int(linear2_shape[1]) != nhid:
        raise CheckpointLineageError(
            "Step3 checkpoint transformer feed-forward shapes are inconsistent: "
            f"linear1={linear1_shape!r} linear2={linear2_shape!r} emsize={emsize!r}"
        )

    return build_step3_model_architecture_config(
        {
            "nuser": int(user_shape[0]),
            "nitem": int(item_shape[0]),
            "ntoken": ntoken,
            "emsize": emsize,
            "nlayers": nlayers,
            "nhead": int(fallback.get("nhead", 0) or 0),
            "nhid": nhid,
            "dropout": float(fallback.get("dropout", 0.0)),
        }
    )


def diff_model_architecture_payloads(
    checkpoint_payload: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_arch = build_step3_model_architecture_config(checkpoint_payload)
    expected_arch = build_step3_model_architecture_config(expected_payload)
    mismatches: dict[str, dict[str, Any]] = {}
    for key in STEP3_MODEL_ARCHITECTURE_FIELDS:
        checkpoint_value = checkpoint_arch.get(key)
        expected_value = expected_arch.get(key)
        if checkpoint_value != expected_value:
            mismatches[key] = {
                "checkpoint": checkpoint_value,
                "expected": expected_value,
            }
    return {
        "checkpoint_payload": checkpoint_arch,
        "expected_payload": expected_arch,
        "checkpoint_hash": compute_model_architecture_config_hash(checkpoint_arch),
        "expected_hash": compute_model_architecture_config_hash(expected_arch),
        "mismatch_keys": sorted(mismatches),
        "mismatches": mismatches,
    }


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


def checkpoint_file_sha256(path: str | Path) -> str:
    fp = file_fingerprint(path)
    sha = str(fp.get("sha256") or "")
    if not fp.get("exists") or not fp.get("is_file") or not sha:
        raise CheckpointLineageError(f"Checkpoint file hash requires an existing file: {Path(path).expanduser()}")
    return sha


def current_source_table_lineage(*, required_file: bool = False) -> dict[str, Any]:
    field_sources = parse_json_object(
        os.environ.get("ODCR_CONFIG_FIELD_SOURCES_JSON"),
        context="ODCR_CONFIG_FIELD_SOURCES_JSON",
        required=True,
    )
    out: dict[str, Any] = {
        "field_sources_hash": stable_hash(field_sources),
        "field_sources": field_sources,
    }
    manifest_dir = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if manifest_dir:
        source_table_path = Path(manifest_dir).expanduser().resolve() / "source_table.json"
        if not source_table_path.is_file():
            raise CheckpointLineageError(
                f"Current source_table lineage requires meta/source_table.json; missing {source_table_path}"
            )
        out["source_table_path"] = str(source_table_path)
        out["source_table_fingerprint"] = file_fingerprint(source_table_path, sample_only=True)  # internal-only lineage fingerprint
        out["source_table_file_hash"] = stable_hash(out["source_table_fingerprint"])
        try:
            payload = json.loads(source_table_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointLineageError(f"Current source_table is unreadable: {source_table_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CheckpointLineageError(f"Current source_table root must be an object: {source_table_path}")
        out["source_table_payload_summary"] = {
            "source_table_schema_version": payload.get("source_table_schema_version"),
            "field_source_count": len(payload.get("field_sources") or {}),
            "record_count": len(payload.get("records") or []),
            "field_source_keys": sorted(str(k) for k in (payload.get("field_sources") or {}).keys()),
        }
    elif required_file:
        raise CheckpointLineageError("ODCR_MANIFEST_DIR is required for source_table lineage.")
    out["hash"] = stable_hash(out)
    return out


def current_resolved_config_lineage(
    *,
    stage: str,
    task_id: int,
    artifact: str,
    mode: str | None = None,
    required_file: bool = False,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "stage": str(stage),
        "task_id": int(task_id),
        "artifact": str(artifact),
    }
    if mode is not None:
        extra["mode"] = str(mode)
    out: dict[str, Any] = {"hash": current_one_control_resolved_config_hash(extra=extra)}
    manifest_dir = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if manifest_dir:
        resolved_path = Path(manifest_dir).expanduser().resolve() / "resolved_config.json"
        if not resolved_path.is_file():
            raise CheckpointLineageError(
                f"Current resolved config lineage requires meta/resolved_config.json; missing {resolved_path}"
            )
        out["resolved_config_path"] = str(resolved_path)
        out["resolved_config_fingerprint"] = file_fingerprint(resolved_path, sample_only=True)  # internal-only lineage fingerprint
        out["resolved_config_file_hash"] = stable_hash(out["resolved_config_fingerprint"])
    elif required_file:
        raise CheckpointLineageError("ODCR_MANIFEST_DIR is required for resolved_config lineage.")
    return out


def current_training_runtime_config_lineage(*, required_file: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    manifest_dir = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if manifest_dir:
        runtime_path = Path(manifest_dir).expanduser().resolve() / "training_runtime_config.json"
        if not runtime_path.is_file():
            if required_file:
                raise CheckpointLineageError(
                    f"Current training runtime config lineage requires meta/training_runtime_config.json; "
                    f"missing {runtime_path}"
                )
            out["training_runtime_config_path"] = str(runtime_path)
            out["training_runtime_config_file_hash"] = ""
            out["hash"] = stable_hash(out)
            return out
        out["training_runtime_config_path"] = str(runtime_path)
        out["training_runtime_config_fingerprint"] = file_fingerprint(runtime_path, sample_only=True)
        out["training_runtime_config_file_hash"] = stable_hash(out["training_runtime_config_fingerprint"])
        try:
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointLineageError(
                f"Current training runtime config is unreadable: {runtime_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise CheckpointLineageError(f"Current training_runtime_config root must be an object: {runtime_path}")
        out["training_runtime_config_payload_summary"] = {
            "training_runtime_config_schema_version": payload.get("training_runtime_config_schema_version"),
            "field_count": len(payload),
            "has_final_training_config": "training_diagnostics" in payload,
        }
    elif required_file:
        raise CheckpointLineageError("ODCR_MANIFEST_DIR is required for training_runtime_config lineage.")
    out["hash"] = stable_hash(out)
    return out


def step3_resolved_config_compatibility_payload(
    *,
    payload: Mapping[str, Any],
    task_id: int,
    source_domain: str,
    target_domain: str,
    embed_dim: int,
    structured_losses: Mapping[str, Any],
    loss_semantics: Mapping[str, Any] | None = None,
    architecture_hash: str,
) -> dict[str, Any]:
    runtime_roots = payload.get("runtime_roots") if isinstance(payload.get("runtime_roots"), Mapping) else {}
    gather = dict(payload.get("step3_cross_rank_structured_gather") or {})
    task_profile = dict(payload.get("step3_task_profile") or {})
    loss_config = {
        "step3_structured_losses": dict(structured_losses),
        "step3_loss_semantics": dict(loss_semantics or {}),
        "step3_cross_rank_structured_gather": gather,
    }
    return {
        "schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "task_id": int(task_id),
        "source_domain": str(source_domain),
        "target_domain": str(target_domain),
        "task_profile_id": payload.get("task_profile_id"),
        "task_profile_key": payload.get("task_profile_key"),
        "profile_isolation_hash": payload.get("profile_isolation_hash"),
        "scenario": payload.get("scenario"),
        "direction": payload.get("direction"),
        "runtime_roots": {
            "data_dir": runtime_roots.get("data_dir"),
            "merged_dir": runtime_roots.get("merged_dir"),
            "models_dir": runtime_roots.get("models_dir"),
            "sentence_embed_model": runtime_roots.get("sentence_embed_model"),
            "step5_text_model": runtime_roots.get("step5_text_model"),
            "embed_dim": runtime_roots.get("embed_dim"),
        },
        "env_embed_dim": int(embed_dim),
        "step3_structured_losses": dict(structured_losses),
        "step3_loss_semantics": dict(loss_semantics or {}),
        "step3_tokenizer": dict(payload.get("step3_tokenizer") or {}),
        "step3_evidence": dict(payload.get("step3_evidence") or {}),
        "step3_scenario_profile": dict(payload.get("step3_scenario_profile") or {}),
        "step3_task_profile": task_profile,
        "step3_task_profile_hash": stable_hash(task_profile),
        "step3_cross_rank_structured_gather": gather,
        "step3_cross_rank_structured_gather_hash": stable_hash(gather),
        "loss_config_hash": str(payload.get("loss_config_hash") or stable_hash(loss_config)),
        "model_architecture_config_hash": str(architecture_hash),
        "representation_contract": {
            "Step3ForwardOutput": "odcr_step3_forward_output/structured_shared_specific_v1",
            "Step3LossBundle": "odcr_step3_loss_bundle/structured_shared_specific_v1",
        },
        "effective_payload_schema_version": payload.get("schema_version"),
    }


def step3_source_table_compatibility_payload(source_table_lineage: Mapping[str, Any]) -> dict[str, Any]:
    field_sources = source_table_lineage.get("field_sources")
    if not isinstance(field_sources, Mapping):
        field_sources = {}
    return {
        "schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "field_sources": {
            key: field_sources.get(key)
            for key in STEP3_SOURCE_TABLE_COMPAT_KEYS
            if key in field_sources
        },
    }


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


def _checkpoint_run_root(checkpoint_path: str | Path) -> Path:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    for parent in [ckpt.parent, *ckpt.parents]:
        if parent.name == "model":
            return parent.parent.resolve()
    return ckpt.parent.parent.resolve()


def checkpoint_lineage_path_for_weight(checkpoint_path: str | Path) -> Path:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    return ckpt.with_name(ckpt.name + CHECKPOINT_FILE_LINEAGE_SUFFIX).resolve()


def checkpoint_event_ledger_path_for_weight(checkpoint_path: str | Path) -> Path:
    return (_checkpoint_run_root(checkpoint_path) / "state" / CHECKPOINT_LINEAGE_FILENAME).resolve()


def _legacy_checkpoint_lineage_path_for_weight(checkpoint_path: str | Path) -> Path:
    return (_checkpoint_run_root(checkpoint_path) / "state" / CHECKPOINT_LINEAGE_FILENAME).resolve()


def _lineage_event_from_payload(payload: Mapping[str, Any], checkpoint_path: str | Path) -> dict[str, Any]:
    is_step3 = str(payload.get("stage") or "") == "step3"
    if is_step3:
        missing_event_semantics = [key for key in ("reason", "replaced_previous") if key not in payload]
        if missing_event_semantics:
            raise CheckpointLineageError(
                "Step3 checkpoint event semantics must be explicit: "
                + ", ".join(missing_event_semantics)
            )
    created_at = str(payload.get("created_at") or payload.get("created_at_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    checkpoint_file = str(payload.get("checkpoint_path") or Path(checkpoint_path).expanduser().resolve())
    event_core = {
        "checkpoint_file": checkpoint_file,
        "checkpoint_file_hash": payload.get("checkpoint_file_hash"),
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "selection_scope": payload.get("selection_scope"),
        "selection_metric": payload.get("selection_metric"),
        "selection_metric_value": payload.get("selection_metric_value"),
        "selection_direction": payload.get("selection_direction"),
        "reason": payload.get("reason"),
        "replaced_previous": bool(payload.get("replaced_previous", False)),
    }
    event_id = stable_hash(
        {
            "stage": payload.get("stage"),
            "run_id": payload.get("run_id"),
            "task_id": payload.get("task_id"),
            **event_core,
            "lineage_hash": payload.get("lineage_hash"),
        }
    )
    return {
        "event_schema_version": CHECKPOINT_EVENT_LEDGER_SCHEMA_VERSION,
        "event_id": event_id,
        "created_at": created_at,
        "created_at_utc": created_at,
        "stage": payload.get("stage"),
        **event_core,
        "global_best_epoch": payload.get("global_best_epoch"),
        "global_best_metric": payload.get("global_best_metric"),
        "after_min_epochs_best_epoch": payload.get("after_min_epochs_best_epoch"),
        "after_min_epochs_best_metric": payload.get("after_min_epochs_best_metric"),
        "resolved_config_hash": payload.get("resolved_config_hash"),
        "training_runtime_config_hash": payload.get("training_runtime_config_hash"),
        "epoch_summary_hash": payload.get("epoch_summary_hash"),
        "metrics_jsonl_hash": payload.get("metrics_jsonl_hash"),
        "quality_status": payload.get("quality_status") or payload.get("quality_status_at_save"),
        "downstream_ready": bool(payload.get("downstream_ready", False)),
        "epoch": payload.get("checkpoint_epoch"),
        "metric": payload.get("selection_metric_value"),
        "metric_name": payload.get("selection_metric"),
        "metric_source": payload.get("metric_source", "meta/epoch_summary.csv.valid_loss"),
        "path": checkpoint_file,
        "hash": payload.get("checkpoint_file_hash"),
    }


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
    if str(payload.get("stage") or "") == "step3":
        missing = [key for key in ("sidecar_schema_version", "checkpoint_file_hash", "reason") if not payload.get(key)]
        missing.extend(key for key in ("replaced_previous",) if key not in payload)
        if missing:
            raise CheckpointLineageError(
                "Step3 checkpoint writer must include complete lineage sidecar fields: "
                + ", ".join(missing)
            )
        actual_hash = checkpoint_file_sha256(checkpoint_path)
        if str(payload.get("checkpoint_file_hash")) != actual_hash:
            raise CheckpointLineageError(
                "Step3 checkpoint writer refused sidecar with stale checkpoint_file_hash: "
                f"payload={payload.get('checkpoint_file_hash')!r} actual={actual_hash!r}"
            )
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
    ledger_path = checkpoint_event_ledger_path_for_weight(checkpoint_path)
    events: list[dict[str, Any]] = []
    if ledger_path.is_file():
        try:
            existing = json.loads(ledger_path.read_text(encoding="utf-8"))
            if isinstance(existing, Mapping):
                raw_events = existing.get("saved_checkpoint_events") or []
                if isinstance(raw_events, list):
                    events = [dict(item) for item in raw_events if isinstance(item, Mapping)]
                elif existing.get("checkpoint_path"):
                    events = [_lineage_event_from_payload(existing, existing.get("checkpoint_path"))]
        except (OSError, json.JSONDecodeError):
            events = []
    event = _lineage_event_from_payload(out, checkpoint_path)
    events.append(event)
    ledger = {
        "schema_version": CHECKPOINT_EVENT_LEDGER_SCHEMA_VERSION,
        "stage": out.get("stage"),
        "run_id": out.get("run_id"),
        "task_id": out.get("task_id"),
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "global_best_overwrite_policy": "never_silently_overwrite_global_best",
        "saved_checkpoint_events": events,
        "latest_event": event,
    }
    ledger["checkpoint_event_ledger_hash"] = stable_hash(ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ledger_path, ledger)
    return path


def read_checkpoint_lineage(
    checkpoint_path: str | Path,
    *,
    expected_stage: str | None = None,
    allow_derived_lineage_hash: bool = False,
) -> dict[str, Any]:
    path = checkpoint_lineage_path_for_weight(checkpoint_path)
    if not path.is_file():
        legacy_path = _legacy_checkpoint_lineage_path_for_weight(checkpoint_path)
        if legacy_path.is_file():
            path = legacy_path
        else:
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
    stored_hash = data.get("lineage_hash")
    if stored_hash != actual_hash:
        if allow_derived_lineage_hash and not stored_hash:
            data["lineage_hash"] = actual_hash
            data["lineage_hash_source"] = "derived_from_checkpoint_lineage_payload_missing_stored_lineage_hash"
            data["lineage_hash_derivation"] = {
                "hash_algorithm": "stable_hash",
                "excluded_fields": ["lineage_hash", "created_at_utc"],
                "source_path": str(path),
            }
        else:
            raise CheckpointLineageError(
                f"Checkpoint lineage hash mismatch: stored={data.get('lineage_hash')} computed={actual_hash} path={path}"
            )
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


def _require_non_empty(payload: Mapping[str, Any], key: str, *, context: str) -> None:
    value = payload.get(key)
    if value is None or value == "" or value == {} or value == []:
        raise CheckpointLineageError(f"{context} missing required non-empty lineage field: {key}")


def _require_path(payload: Mapping[str, Any], path: tuple[str, ...], *, context: str) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            raise CheckpointLineageError(f"{context} missing lineage field: {'.'.join(path)}")
        cur = cur[key]
    if cur is None or cur == "" or cur == {} or cur == []:
        raise CheckpointLineageError(f"{context} empty lineage field: {'.'.join(path)}")
    return cur


def validate_step3_checkpoint_lineage(
    checkpoint_path: str | Path,
    *,
    expected: Mapping[str, Any],
    allow_derived_lineage_hash: bool = False,
) -> dict[str, Any]:
    """Validate a Step3 checkpoint sidecar before downstream consumers load weights."""

    lineage = read_checkpoint_lineage(
        checkpoint_path,
        expected_stage="step3",
        allow_derived_lineage_hash=allow_derived_lineage_hash,
    )
    context = "Step3 checkpoint compatibility"
    require_keys(lineage, STEP3_CHECKPOINT_REQUIRED_FIELDS, context=context)
    for key in STEP3_CHECKPOINT_REQUIRED_FIELDS:
        if key in STEP3_CHECKPOINT_NULLABLE_REQUIRED_FIELDS and key in lineage:
            continue
        _require_non_empty(lineage, key, context=context)
    if lineage.get("sidecar_schema_version") != STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION:
        raise CheckpointLineageError(
            f"Unsupported Step3 checkpoint sidecar schema {lineage.get('sidecar_schema_version')!r}; "
            f"minimum accepted {STEP3_CHECKPOINT_MIN_ACCEPTED_SCHEMA_VERSION!r}."
        )
    ckpt = Path(checkpoint_path).expanduser().resolve()
    stored_path = Path(str(lineage.get("checkpoint_path"))).expanduser().resolve()
    if stored_path != ckpt:
        raise CheckpointLineageError(
            f"Step3 checkpoint compatibility path mismatch: sidecar={stored_path} current={ckpt}"
        )
    actual_hash = checkpoint_file_sha256(ckpt)
    if lineage.get("checkpoint_file_hash") != actual_hash:
        raise CheckpointLineageError(
            "Step3 checkpoint compatibility hash mismatch: "
            f"sidecar={lineage.get('checkpoint_file_hash')!r} current={actual_hash!r}"
        )
    file_section = lineage.get("checkpoint_file")
    if not isinstance(file_section, Mapping) or file_section.get("sha256") != actual_hash:
        raise CheckpointLineageError("Step3 checkpoint compatibility checkpoint_file.sha256 is missing or stale.")
    tokenizer_manifest = lineage.get("step3_tokenizer_cache_manifest")
    if not isinstance(tokenizer_manifest, Mapping):
        raise CheckpointLineageError("Step3 checkpoint compatibility missing tokenizer cache manifest lineage.")
    tokenizer_manifest_hash = stable_hash(tokenizer_manifest)
    if lineage.get("step3_tokenizer_cache_manifest_hash") != tokenizer_manifest_hash:
        raise CheckpointLineageError(
            "Step3 checkpoint compatibility tokenizer cache manifest hash mismatch: "
            f"sidecar={lineage.get('step3_tokenizer_cache_manifest_hash')!r} computed={tokenizer_manifest_hash!r}"
        )
    source_table_section = lineage.get("source_table")
    if not isinstance(source_table_section, Mapping):
        raise CheckpointLineageError("Step3 checkpoint compatibility missing frozen source_table lineage.")
    frozen_source_table_compat_hash = stable_hash(step3_source_table_compatibility_payload(source_table_section))
    if lineage.get("source_table_compatibility_hash") != frozen_source_table_compat_hash:
        raise CheckpointLineageError(
            "Step3 checkpoint compatibility frozen source_table hash mismatch: "
            f"sidecar={lineage.get('source_table_compatibility_hash')!r} "
            f"computed={frozen_source_table_compat_hash!r}"
        )
    for key, expected_value in expected.items():
        if lineage.get(key) != expected_value:
            raise CheckpointLineageError(
                f"Step3 checkpoint compatibility mismatch for {key}: "
                f"checkpoint={lineage.get(key)!r} current={expected_value!r}"
            )
    required_nested = (
        ("env", "embed_dim"),
        ("batch_semantics", "formula_proof", "matches"),
        ("schema_contract_versions", "preprocess_contract_version"),
        ("schema_contract_versions", "step3_tokenizer_cache_schema_version"),
        ("schema_contract_versions", "step3_upstream_gate_schema_version"),
        ("compatibility_metadata", "minimum_accepted_schema_version"),
        ("compatibility_metadata", "downstream_compare_fields"),
    )
    for path in required_nested:
        value = _require_path(lineage, path, context=context)
        if path == ("batch_semantics", "formula_proof", "matches") and value is not True:
            raise CheckpointLineageError("Step3 checkpoint compatibility batch formula proof is false.")
    return lineage


def payload_section_hash(section: str, *, payload: Mapping[str, Any] | None = None) -> str:
    obj = dict(payload or current_effective_payload(required=True))
    return stable_hash(obj.get(section) or {})
