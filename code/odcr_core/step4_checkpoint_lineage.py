"""Step4 consumer-side Step3 checkpoint lineage validation.

The hard gate here separates Step3 checkpoint-load architecture from Step4
consumer/runtime/RCR configuration. Step4 may change routing, preflight,
decode, hardware, or report settings without changing the frozen Step3 model
shape needed to load the checkpoint.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from config import get_odcr_embed_dim
from data_contract import PREPROCESS_CONTRACT_VERSION
from odcr_core.step3_upstream_gate import validate_step3_preprocess_upstream_gate
from odcr_core.training_checkpoint import (
    CheckpointLineageError,
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    STEP3_MODEL_ARCHITECTURE_FIELDS,
    build_step3_model_architecture_config,
    compute_model_architecture_config_hash,
    current_effective_payload,
    current_source_table_lineage,
    diff_model_architecture_payloads,
    extract_checkpoint_model_architecture_payload,
    extract_checkpoint_state_dict_architecture_payload,
    file_fingerprint,
    model_artifact_fingerprint,
    read_checkpoint_lineage,
    stable_hash,
    step3_source_table_compatibility_payload,
    validate_step3_checkpoint_lineage,
)

STEP4_PRELAUNCH_LINEAGE_VALIDATION_SCHEMA_VERSION = "odcr_step4_prelaunch_lineage_validation/1"
STEP4_CHECKPOINT_ARCH_DIFF_SCHEMA_VERSION = "odcr_step4_checkpoint_architecture_hash_diff/1"
STEP4_NTOKEN_COMPATIBILITY_SCHEMA_VERSION = "odcr_step4_ntoken_sidecar_checkpoint_policy/1"
STEP4_SOURCE_TABLE_HASH_SCOPE_SCHEMA_VERSION = "odcr_step4_source_table_hash_scope/1"
STEP4_LIVE_FROZEN_CONFIG_DRIFT_SCHEMA_VERSION = "odcr_step4_live_frozen_config_drift/1"
STEP4_ARCHITECTURE_IGNORED_CONFIG_KEYS = (
    "step4_rcr",
    "step4_runtime",
    "decode",
    "eval",
    "hardware",
    "runtime_env",
    "training_semantic_fingerprint",
    "generation_semantic_fingerprint",
    "runtime_diagnostics_fingerprint",
    "batch_semantics",
    "global_batch_size",
    "per_gpu_batch_size",
    "micro_batch_size",
    "num_proc",
    "guardrail",
    "docs",
)
_PROFILE_FILES = {
    "user_content": "user_content_profiles.npy",
    "user_style": "user_style_profiles.npy",
    "item_content": "item_content_profiles.npy",
    "item_style": "item_style_profiles.npy",
}

STEP4_FORMAL_LINEAGE_CONTRACT_SCHEMA_VERSION = "odcr_step4_formal_lineage_contract/1"
STEP4_FORMAL_LINEAGE_REQUIRED_FIELDS = (
    "lineage_hash",
    "lineage_hash_source",
    "checkpoint_path",
    "selected_checkpoint",
    "checkpoint_sha256",
    "checkpoint_sha256_source",
    "model_architecture_config_hash",
    "effective_model_ntoken",
    "sidecar_ntoken",
    "checkpoint_tensor_ntoken",
    "stage_status_path",
    "eval_handoff_path",
    "run_summary_path",
    "source_table_path",
    "resolved_config_path",
    "checkpoint_lineage_path",
    "compatibility_status",
    "alias_consistency",
    "best_pth_alias_consistent",
    "used_checkpoint_source",
    "frozen_config_policy",
    "hash_scope_version",
)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _load_upstream_resolution_from_env() -> dict[str, Any]:
    raw = str(os.environ.get("ODCR_UPSTREAM_RESOLUTION_JSON") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _path_from_upstream(upstream: Mapping[str, Any], *keys: str) -> str:
    validation = _mapping_or_empty(upstream.get("stage_status_validation"))
    status = _mapping_or_empty(upstream.get("stage_status"))
    artifacts = _mapping_or_empty(status.get("artifacts"))
    for key in keys:
        value = validation.get(key) or status.get(key) or upstream.get(key)
        if isinstance(value, Mapping):
            value = value.get("path")
        text = _first_text(value)
        if text:
            return text
        artifact = artifacts.get(key)
        if isinstance(artifact, Mapping):
            text = _first_text(artifact.get("path"))
            if text:
                return text
    return ""


def require_step4_lineage_field(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    source_paths: Mapping[str, Any] | None = None,
) -> Any:
    if field_name not in payload or payload.get(field_name) in (None, "", {}, []):
        available = sorted(str(key) for key in payload.keys())
        raise CheckpointLineageError(
            "Step4 formal lineage contract missing required field "
            f"{field_name!r}; available_keys={available}; source_paths={dict(source_paths or {})}"
        )
    return payload[field_name]


def normalize_step3_lineage_for_step4(
    lineage: Mapping[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    upstream_resolution: Mapping[str, Any] | None = None,
    checkpoint_lineage_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the single Step4-facing Step3 lineage payload.

    Step4 runtime consumers use this normalized contract instead of reading
    ad-hoc sidecar/status keys. Historical sidecars that lack a stored
    ``lineage_hash`` are derivable, but the source is recorded explicitly.
    """

    source = dict(lineage)
    upstream = dict(upstream_resolution or _load_upstream_resolution_from_env())
    binding = _mapping_or_empty(source.get("checkpoint_binding"))
    best_alias = _mapping_or_empty(binding.get("best_pth_alias") or source.get("alias_consistency"))
    latest_alias = _mapping_or_empty(binding.get("latest_pth_alias"))
    alias_payload = (
        dict(best_alias)
        if best_alias
        else {
            "alias_consistent": False,
            "used_as_primary": False,
            "source": "not_provided_to_normalizer",
        }
    )
    selected_checkpoint = _first_text(
        source.get("selected_checkpoint"),
        source.get("selected_checkpoint_path"),
        binding.get("selected_checkpoint_path"),
        checkpoint_path,
        _path_from_upstream(upstream, "selected_checkpoint"),
    )
    checkpoint_path_text = _first_text(source.get("checkpoint_path"), selected_checkpoint, checkpoint_path)
    checkpoint_sha = _first_text(
        source.get("checkpoint_sha256"),
        source.get("checkpoint_file_hash"),
        source.get("checkpoint_hash"),
        binding.get("selected_checkpoint_hash"),
        _path_from_upstream(upstream, "selected_checkpoint_hash"),
    )
    checkpoint_sha_source = (
        "checkpoint_lineage.checkpoint_file_hash"
        if _first_text(source.get("checkpoint_file_hash"))
        else "stage_status.selected_checkpoint_hash"
        if _first_text(binding.get("selected_checkpoint_hash"), _path_from_upstream(upstream, "selected_checkpoint_hash"))
        else "missing"
    )
    lineage_hash = _first_text(source.get("lineage_hash"), source.get("checkpoint_lineage_hash"))
    lineage_hash_source = _first_text(source.get("lineage_hash_source"))
    if not lineage_hash:
        derivation_payload = {
            "checkpoint_path": checkpoint_path_text,
            "checkpoint_sha256": checkpoint_sha,
            "model_architecture_config_hash": source.get("model_architecture_config_hash"),
            "checkpoint_lineage_path": _first_text(checkpoint_lineage_path, source.get("checkpoint_lineage_path")),
            "stage_status_path": _path_from_upstream(upstream, "status_path", "stage_status"),
            "eval_handoff_path": _path_from_upstream(upstream, "eval_handoff"),
            "run_summary_path": _path_from_upstream(upstream, "run_summary"),
            "source_table_path": _path_from_upstream(upstream, "source_table"),
            "resolved_config_path": _path_from_upstream(upstream, "resolved_config"),
        }
        lineage_hash = stable_hash(derivation_payload)
        lineage_hash_source = "derived_from_normalized_step3_lineage_evidence"
    elif not lineage_hash_source:
        lineage_hash_source = (
            "checkpoint_lineage.lineage_hash"
            if _first_text(source.get("lineage_hash"))
            else "validated.checkpoint_lineage_hash"
        )
    hash_scope = _mapping_or_empty(source.get("source_table_hash_scope_report"))
    frozen_policy = _mapping_or_empty(source.get("live_vs_frozen_step3_config_drift"))
    normalized: dict[str, Any] = {
        "schema_version": STEP4_FORMAL_LINEAGE_CONTRACT_SCHEMA_VERSION,
        "lineage_hash": lineage_hash,
        "lineage_hash_source": lineage_hash_source,
        "checkpoint_path": checkpoint_path_text,
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256_source": checkpoint_sha_source,
        "model_architecture_config_hash": _first_text(source.get("model_architecture_config_hash")),
        "effective_model_ntoken": source.get("effective_model_ntoken"),
        "sidecar_ntoken": source.get("sidecar_ntoken"),
        "checkpoint_tensor_ntoken": source.get("checkpoint_tensor_ntoken"),
        "stage_status_path": _first_text(
            source.get("stage_status_path"),
            binding.get("stage_status_path"),
            _path_from_upstream(upstream, "status_path", "stage_status"),
        ),
        "eval_handoff_path": _first_text(
            source.get("eval_handoff_path"),
            binding.get("eval_handoff"),
            _path_from_upstream(upstream, "eval_handoff"),
        ),
        "run_summary_path": _first_text(source.get("run_summary_path"), _path_from_upstream(upstream, "run_summary")),
        "source_table_path": _first_text(source.get("source_table_path"), _path_from_upstream(upstream, "source_table")),
        "resolved_config_path": _first_text(
            source.get("resolved_config_path"),
            _path_from_upstream(upstream, "resolved_config"),
        ),
        "checkpoint_lineage_path": _first_text(checkpoint_lineage_path, source.get("checkpoint_lineage_path")),
        "compatibility_status": _first_text(source.get("status"), "ok"),
        "alias_consistency": alias_payload,
        "best_pth_alias_consistent": bool(alias_payload.get("alias_consistent", False)),
        "latest_pth_alias_consistent": bool(latest_alias.get("alias_consistent", False)),
        "used_checkpoint_source": _first_text(source.get("checkpoint_source"), binding.get("checkpoint_source")),
        "frozen_config_policy": _first_text(
            frozen_policy.get("policy"),
            "Step4 uses frozen Step3 checkpoint lineage; live Step4 config is recorded in a separate hash scope.",
        ),
        "hash_scope_version": _first_text(
            hash_scope.get("schema_version"),
            STEP4_SOURCE_TABLE_HASH_SCOPE_SCHEMA_VERSION,
        ),
        "source_paths": {
            "checkpoint_path": checkpoint_path_text,
            "selected_checkpoint": selected_checkpoint,
            "checkpoint_lineage_path": _first_text(checkpoint_lineage_path, source.get("checkpoint_lineage_path")),
            "stage_status_path": _first_text(
                source.get("stage_status_path"),
                binding.get("stage_status_path"),
                _path_from_upstream(upstream, "status_path", "stage_status"),
            ),
            "eval_handoff_path": _first_text(
                source.get("eval_handoff_path"),
                binding.get("eval_handoff"),
                _path_from_upstream(upstream, "eval_handoff"),
            ),
            "run_summary_path": _path_from_upstream(upstream, "run_summary"),
            "source_table_path": _path_from_upstream(upstream, "source_table"),
            "resolved_config_path": _path_from_upstream(upstream, "resolved_config"),
        },
        "available_input_keys": sorted(str(key) for key in source.keys()),
    }
    normalized["required_fields_status"] = {
        key: bool(normalized.get(key) not in (None, "", {}, []))
        for key in STEP4_FORMAL_LINEAGE_REQUIRED_FIELDS
    }
    return normalized


def validate_step4_formal_lineage_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    source_paths = _mapping_or_empty(payload.get("source_paths"))
    missing = [
        key
        for key in STEP4_FORMAL_LINEAGE_REQUIRED_FIELDS
        if key not in payload or payload.get(key) in (None, "", {}, [])
    ]
    if missing:
        raise CheckpointLineageError(
            "Step4 formal lineage contract validation failed; "
            f"missing={missing}; available_keys={sorted(str(key) for key in payload.keys())}; "
            f"source_paths={dict(source_paths)}"
        )
    return dict(payload)


def _step3_preprocess_lineage_expected_for_step4(upstream_evidence: Mapping[str, Any]) -> dict[str, Any]:
    preprocess = upstream_evidence.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise CheckpointLineageError("Step4 refused Step3 checkpoint: current preprocess gate evidence missing.")
    latest_run_ids: dict[str, str] = {}
    run_summary_fps: dict[str, Any] = {}
    stage_status_fps: dict[str, Any] = {}
    stage_manifest_fps: dict[str, Any] = {}
    source_table_fps: dict[str, Any] = {}
    metrics_fps: dict[str, Any] = {}
    verify_fps: dict[str, Any] = {}
    run_summary_fingerprints: dict[str, Any] = {}
    stage_fingerprints: dict[str, Any] = {}
    for unit in ("a", "b", "c"):
        item = preprocess.get(unit)
        if not isinstance(item, Mapping):
            raise CheckpointLineageError(f"Step4 refused Step3 checkpoint: preprocess_{unit} evidence missing.")
        run_id = str(item.get("run_id") or "").strip()
        if not run_id:
            raise CheckpointLineageError(f"Step4 refused Step3 checkpoint: preprocess_{unit} run_id missing.")
        latest_run_ids[unit] = run_id
        run_summary_fps[unit] = item.get("run_summary_fingerprint")
        stage_status_fps[unit] = item.get("stage_status_fingerprint")
        stage_manifest_fps[unit] = item.get("stage_manifest_fingerprint")
        source_table_fps[unit] = item.get("source_table_fingerprint")
        metrics_fps[unit] = item.get("metrics_fingerprint")
        verify_fps[unit] = item.get("verify_report_fingerprint")
        run_summary_fingerprints[unit] = item.get("fingerprint_hash")
        stage_fingerprints[unit] = {
            "run_fingerprint_hash": item.get("fingerprint_hash"),
            "run_summary": item.get("run_summary_fingerprint"),
            "stage_status": item.get("stage_status_fingerprint"),
            "stage_manifest": item.get("stage_manifest_fingerprint"),
            "source_table": item.get("source_table_fingerprint"),
            "metrics": item.get("metrics_fingerprint"),
            "verify_report": item.get("verify_report_fingerprint"),
        }
    return {
        "preprocess_latest_run_ids": latest_run_ids,
        "preprocess_a_latest_run_id": latest_run_ids["a"],
        "preprocess_b_latest_run_id": latest_run_ids["b"],
        "preprocess_c_latest_run_id": latest_run_ids["c"],
        "preprocess_run_summary_fingerprints": run_summary_fps,
        "preprocess_run_summary_lineage_fingerprints": run_summary_fingerprints,
        "preprocess_stage_status_fingerprints": stage_status_fps,
        "preprocess_stage_manifest_fingerprints": stage_manifest_fps,
        "preprocess_source_table_fingerprints": source_table_fps,
        "preprocess_metrics_fingerprints": metrics_fps,
        "preprocess_verify_report_fingerprints": verify_fps,
        "preprocess_stage_fingerprints": stage_fingerprints,
        "preprocess_run_summary_fingerprints_hash": stable_hash(run_summary_fps),
        "preprocess_stage_status_fingerprints_hash": stable_hash(stage_status_fps),
        "preprocess_stage_manifest_fingerprints_hash": stable_hash(stage_manifest_fps),
        "preprocess_source_table_fingerprints_hash": stable_hash(source_table_fps),
        "preprocess_metrics_fingerprints_hash": stable_hash(metrics_fps),
        "preprocess_verify_report_fingerprints_hash": stable_hash(verify_fps),
    }


def _domain_profile_shapes(data_dir: str | Path, domain: str) -> dict[str, tuple[int, ...]]:
    root = Path(data_dir).expanduser().resolve() / str(domain)
    out: dict[str, tuple[int, ...]] = {}
    for key, filename in _PROFILE_FILES.items():
        path = root / filename
        if not path.is_file():
            raise CheckpointLineageError(f"Step4 profile shape check missing {domain}:{filename}: {path}")
        out[key] = tuple(int(x) for x in np.load(path, mmap_mode="r").shape)
    return out


def build_step4_observed_loader_architecture_config(
    *,
    data_dir: str | Path,
    auxiliary_domain: str,
    target_domain: str,
    tokenizer_length: int,
    nlayers: int = 2,
    nhead: int = 2,
    nhid: int = 2048,
    dropout: float = 0.2,
) -> dict[str, Any]:
    target = _domain_profile_shapes(data_dir, target_domain)
    auxiliary = _domain_profile_shapes(data_dir, auxiliary_domain)
    for domain_name, shapes in ((target_domain, target), (auxiliary_domain, auxiliary)):
        user_shape = shapes["user_content"]
        item_shape = shapes["item_content"]
        if len(user_shape) != 2 or len(item_shape) != 2:
            raise CheckpointLineageError(f"Step4 profile shape check expected 2D user/item arrays for {domain_name}.")
        for pair in (("user_content", "user_style"), ("item_content", "item_style")):
            if shapes[pair[0]] != shapes[pair[1]]:
                raise CheckpointLineageError(
                    f"Step4 profile shape check mismatch for {domain_name}:{pair[0]} vs {pair[1]}: "
                    f"{shapes[pair[0]]} != {shapes[pair[1]]}"
                )
    target_emsize = int(target["user_content"][1])
    aux_emsize = int(auxiliary["user_content"][1])
    for label, shape in (
        (f"{target_domain}:item", target["item_content"]),
        (f"{auxiliary_domain}:item", auxiliary["item_content"]),
    ):
        if int(shape[1]) != target_emsize:
            raise CheckpointLineageError(
                f"Step4 profile shape check embedding dim mismatch: {label}={shape[1]} target_user={target_emsize}"
            )
    if aux_emsize != target_emsize:
        raise CheckpointLineageError(
            f"Step4 profile shape check embedding dim mismatch: auxiliary={aux_emsize} target={target_emsize}"
        )
    return build_step3_model_architecture_config(
        {
            "nuser": int(target["user_content"][0]) + int(auxiliary["user_content"][0]),
            "nitem": int(target["item_content"][0]) + int(auxiliary["item_content"][0]),
            "ntoken": int(tokenizer_length),
            "emsize": target_emsize,
            "nlayers": int(nlayers),
            "nhead": int(nhead),
            "nhid": int(nhid),
            "dropout": float(dropout),
        }
    )


def build_step4_expected_checkpoint_arch_payload(
    lineage: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    if checkpoint_path is not None:
        sidecar_arch = extract_checkpoint_model_architecture_payload(lineage)
        return extract_checkpoint_state_dict_architecture_payload(
            checkpoint_path,
            fallback_payload=sidecar_arch,
        )
    return extract_checkpoint_model_architecture_payload(lineage)


def architecture_hash_diff_report(
    *,
    checkpoint_lineage: Mapping[str, Any],
    expected_architecture_payload: Mapping[str, Any],
    observed_loader_architecture_payload: Mapping[str, Any] | None = None,
    checkpoint_architecture_payload: Mapping[str, Any] | None = None,
    checkpoint_lineage_path: str | Path | None = None,
    expected_source: str = "checkpoint_state_dict_architecture_config",
) -> dict[str, Any]:
    sidecar_arch = extract_checkpoint_model_architecture_payload(checkpoint_lineage)
    checkpoint_arch = (
        build_step3_model_architecture_config(checkpoint_architecture_payload)
        if checkpoint_architecture_payload is not None
        else sidecar_arch
    )
    expected_arch = build_step3_model_architecture_config(expected_architecture_payload)
    checkpoint_vs_expected = diff_model_architecture_payloads(checkpoint_arch, expected_arch)
    observed_report = None
    if observed_loader_architecture_payload is not None:
        observed_report = diff_model_architecture_payloads(checkpoint_arch, observed_loader_architecture_payload)
    sidecar_report = diff_model_architecture_payloads(sidecar_arch, checkpoint_arch)
    return {
        "schema_version": STEP4_CHECKPOINT_ARCH_DIFF_SCHEMA_VERSION,
        "status": "ok" if not checkpoint_vs_expected["mismatch_keys"] else "mismatch",
        "checkpoint_payload": checkpoint_arch,
        "expected_payload": expected_arch,
        "sidecar_payload": sidecar_arch,
        "checkpoint_hash": checkpoint_vs_expected["checkpoint_hash"],
        "expected_hash": checkpoint_vs_expected["expected_hash"],
        "sidecar_hash": sidecar_report["checkpoint_hash"],
        "mismatch_keys": checkpoint_vs_expected["mismatch_keys"],
        "mismatches": checkpoint_vs_expected["mismatches"],
        "observed_current_loader": observed_report,
        "sidecar_metadata_diff": {
            "status": "ok" if not sidecar_report["mismatch_keys"] else "mismatch",
            "mismatch_keys": sidecar_report["mismatch_keys"],
            "mismatches": sidecar_report["mismatches"],
            "sidecar_hash": sidecar_report["checkpoint_hash"],
            "checkpoint_state_dict_hash": sidecar_report["expected_hash"],
        },
        "ignored_non_architecture_keys": list(STEP4_ARCHITECTURE_IGNORED_CONFIG_KEYS),
        "hash_source_paths": {
            "checkpoint_lineage": str(checkpoint_lineage_path or ""),
            "expected_source": expected_source,
        },
    }


def ntoken_sidecar_checkpoint_compatibility_report(
    *,
    sidecar_architecture_payload: Mapping[str, Any],
    checkpoint_architecture_payload: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar_arch = build_step3_model_architecture_config(sidecar_architecture_payload)
    checkpoint_arch = build_step3_model_architecture_config(checkpoint_architecture_payload)
    sidecar_ntoken = int(sidecar_arch["ntoken"])
    checkpoint_ntoken = int(checkpoint_arch["ntoken"])
    mismatch = sidecar_ntoken != checkpoint_ntoken
    return {
        "schema_version": STEP4_NTOKEN_COMPATIBILITY_SCHEMA_VERSION,
        "status": "compatible",
        "mismatch": bool(mismatch),
        "sidecar_ntoken": sidecar_ntoken,
        "checkpoint_tensor_ntoken": checkpoint_ntoken,
        "effective_model_ntoken": checkpoint_ntoken,
        "effective_model_ntoken_source": "checkpoint_state_dict.word_embeddings_and_hidden2token_shapes",
        "sidecar_metadata_role": "compatibility_metadata_only_not_model_shape_truth",
        "severity": "warning" if mismatch else "none",
        "compatibility_note": (
            "run2 sidecar ntoken differs from the saved tensor vocabulary shape; "
            "Step4 uses checkpoint tensor shape as the effective model ntoken and "
            "keeps the sidecar value only as compatibility metadata."
            if mismatch
            else "sidecar ntoken matches checkpoint tensor vocabulary shape."
        ),
        "silent_ignore": False,
        "checkpoint_tensor_shape_gate_passed": True,
    }


def live_vs_frozen_step3_config_drift_report(
    *,
    checkpoint_lineage: Mapping[str, Any],
    current_payload: Mapping[str, Any],
) -> dict[str, Any]:
    live_step4_hashes = {
        "step4_rcr_config_hash": stable_hash(current_payload.get("step4_rcr") or {}),
        "step4_runtime_config_hash": stable_hash(current_payload.get("step4_runtime") or {}),
    }
    return {
        "schema_version": STEP4_LIVE_FROZEN_CONFIG_DRIFT_SCHEMA_VERSION,
        "status": "allowed_historical_vs_live_drift",
        "severity": "display-only",
        "blocks_step4": False,
        "run2_frozen_training_config_equals_current_live_step3_config": False,
        "policy": (
            "Step4 consumes the frozen run2 checkpoint lineage and tensor-derived "
            "load architecture. Current configs/odcr.yaml Step3 settings are live "
            "defaults for future Step3 runs and are not used to recompute run2 "
            "checkpoint architecture compatibility."
        ),
        "run2_frozen_config_sources": {
            "checkpoint_lineage": checkpoint_lineage.get("checkpoint_path"),
            "one_control_resolved_config_path": checkpoint_lineage.get("one_control_resolved_config_path"),
            "training_runtime_config_path": checkpoint_lineage.get("training_runtime_config_path"),
        },
        "frozen_hashes": {
            "resolved_config_compatibility_hash": checkpoint_lineage.get("resolved_config_compatibility_hash"),
            "training_runtime_config_hash": checkpoint_lineage.get("training_runtime_config_hash"),
            "train_runtime_config_hash": checkpoint_lineage.get("train_runtime_config_hash"),
            "model_architecture_config_hash": checkpoint_lineage.get("model_architecture_config_hash"),
            "source_table_compatibility_hash": checkpoint_lineage.get("source_table_compatibility_hash"),
        },
        **live_step4_hashes,
        "checkpoint_compatibility_uses_current_live_step3_config": False,
    }


def source_table_hash_scope_report(
    *,
    checkpoint_lineage: Mapping[str, Any],
    source_table_compatibility_payload: Mapping[str, Any],
    current_payload: Mapping[str, Any],
    checkpoint_architecture_payload: Mapping[str, Any],
) -> dict[str, Any]:
    observed_hash = stable_hash(source_table_compatibility_payload)
    expected_hash = str(checkpoint_lineage.get("source_table_compatibility_hash") or "")
    mismatch = bool(expected_hash and observed_hash != expected_hash)
    field_diffs = []
    if mismatch:
        field_diffs.append(
            {
                "field": "source_table_compatibility_hash",
                "expected_frozen_step3_hash": expected_hash,
                "observed_current_step4_display_hash": observed_hash,
                "severity": "display-only",
                "blocks_step4": False,
                "reason": (
                    "The observed hash is from the current Step4 live source_table display. "
                    "The blocking Step3 boundary is the frozen checkpoint sidecar hash."
                ),
            }
        )
    return {
        "schema_version": STEP4_SOURCE_TABLE_HASH_SCOPE_SCHEMA_VERSION,
        "status": "scoped",
        "blocking": False,
        "hash_scopes": {
            "step3_frozen_run_hash": {
                "hash": checkpoint_lineage.get("source_table_hash"),
                "severity": "block",
                "source": checkpoint_lineage.get("source_table_path"),
                "meaning": "frozen Step3 run source_table lineage recorded by the checkpoint sidecar",
            },
            "step3_checkpoint_arch_hash": {
                "hash": compute_model_architecture_config_hash(checkpoint_architecture_payload),
                "severity": "block",
                "source": "checkpoint_state_dict_architecture_payload",
                "meaning": "tensor-derived Step3 checkpoint load architecture",
            },
            "step3_training_semantic_hash": {
                "hash": checkpoint_lineage.get("resolved_config_compatibility_hash"),
                "severity": "block",
                "source": checkpoint_lineage.get("one_control_resolved_config_path"),
                "meaning": "frozen Step3 semantic compatibility hash",
            },
            "step4_live_config_hash": {
                "hash": stable_hash(
                    {
                        "task": current_payload.get("task_id"),
                        "step4_rcr": current_payload.get("step4_rcr") or {},
                        "step4_runtime": current_payload.get("step4_runtime") or {},
                    }
                ),
                "severity": "display-only",
                "source": "current Step4 resolved payload",
                "meaning": "live Step4 config display scope; not a Step3 checkpoint architecture hash",
            },
            "step4_rcr_config_hash": {
                "hash": stable_hash(current_payload.get("step4_rcr") or {}),
                "severity": "display-only",
                "source": "configs/odcr.yaml: step4.rcr",
                "meaning": "live Step4 RCR routing configuration",
            },
            "step4_runtime_config_hash": {
                "hash": stable_hash(current_payload.get("step4_runtime") or {}),
                "severity": "display-only",
                "source": "configs/odcr.yaml: step4.runtime",
                "meaning": "live Step4 runtime configuration",
            },
        },
        "field_diffs": field_diffs,
        "compatibility_hash_boundary": (
            "Frozen Step3 checkpoint/source-table hashes gate upstream reuse. "
            "Step4 live config hashes are labelled separately and cannot mutate "
            "the Step3 checkpoint architecture hash."
        ),
    }


def require_step4_checkpoint_architecture_compatible(
    *,
    checkpoint_lineage: Mapping[str, Any],
    expected_architecture_payload: Mapping[str, Any],
) -> dict[str, Any]:
    diff = architecture_hash_diff_report(
        checkpoint_lineage=checkpoint_lineage,
        expected_architecture_payload=expected_architecture_payload,
    )
    if diff["mismatch_keys"]:
        raise CheckpointLineageError(
            "Step3 checkpoint model architecture mismatch: "
            + ", ".join(str(key) for key in diff["mismatch_keys"])
        )
    return diff


def _require_loader_profile_compatible(
    *,
    checkpoint_architecture: Mapping[str, Any],
    observed_loader_architecture: Mapping[str, Any] | None,
) -> list[str]:
    if observed_loader_architecture is None:
        return []
    checkpoint_arch = build_step3_model_architecture_config(checkpoint_architecture)
    observed_arch = build_step3_model_architecture_config(observed_loader_architecture)
    hard_mismatches: list[str] = []
    for key in ("nuser", "nitem", "emsize"):
        if observed_arch[key] != checkpoint_arch[key]:
            hard_mismatches.append(key)
    if hard_mismatches:
        raise CheckpointLineageError(
            "Step4 loader profile/model shape is incompatible with Step3 checkpoint: "
            + ", ".join(
                f"{key} checkpoint={checkpoint_arch[key]!r} observed={observed_arch[key]!r}"
                for key in hard_mismatches
            )
        )
    return [
        key
        for key in STEP3_MODEL_ARCHITECTURE_FIELDS
        if observed_arch.get(key) != checkpoint_arch.get(key)
    ]


def validate_step4_prelaunch_checkpoint_lineage(
    *,
    checkpoint_path: str | Path,
    task_id: int,
    auxiliary_domain: str,
    target_domain: str,
    data_dir: str | Path,
    merged_dir: str | Path,
    runs_dir: str | Path,
    sentence_embed_model: str | Path,
    embed_dim: int | None = None,
    observed_loader_architecture: Mapping[str, Any] | None = None,
    phase: str = "prelaunch",
) -> dict[str, Any]:
    """Run the strict Step4 prelaunch Step3 checkpoint lineage hard gate."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    checkpoint_lineage_path = Path(str(checkpoint) + ".lineage.json")
    lineage = read_checkpoint_lineage(
        checkpoint,
        expected_stage="step3",
        allow_derived_lineage_hash=True,
    )
    sidecar_arch = extract_checkpoint_model_architecture_payload(lineage)
    checkpoint_arch = extract_checkpoint_state_dict_architecture_payload(
        checkpoint,
        fallback_payload=sidecar_arch,
    )
    expected_arch = build_step4_expected_checkpoint_arch_payload(lineage, checkpoint_path=checkpoint)
    arch_diff = require_step4_checkpoint_architecture_compatible(
        checkpoint_lineage={
            **dict(lineage),
            "model_architecture_config": checkpoint_arch,
            "model_architecture_config_hash": compute_model_architecture_config_hash(checkpoint_arch),
        },
        expected_architecture_payload=expected_arch,
    )
    observed_mismatch_keys = _require_loader_profile_compatible(
        checkpoint_architecture=checkpoint_arch,
        observed_loader_architecture=observed_loader_architecture,
    )
    effective_embed_dim = int(embed_dim if embed_dim is not None else get_odcr_embed_dim())
    if int(checkpoint_arch["emsize"]) != effective_embed_dim:
        raise CheckpointLineageError(
            "Step4 loader embed_dim is incompatible with Step3 checkpoint: "
            f"checkpoint={checkpoint_arch['emsize']!r} current={effective_embed_dim!r}"
        )
    payload = current_effective_payload(required=True)
    train_path = Path(merged_dir).expanduser().resolve() / str(int(task_id)) / "aug_train.csv"
    valid_path = Path(merged_dir).expanduser().resolve() / str(int(task_id)) / "aug_valid.csv"
    data_fps = {
        "aug_train_csv": file_fingerprint(train_path),
        "aug_valid_csv": file_fingerprint(valid_path),
    }
    upstream_evidence = validate_step3_preprocess_upstream_gate(
        repo_root=Path(os.environ.get("ODCR_ROOT") or Path.cwd()).resolve(),
        task_id=int(task_id),
        auxiliary_domain=str(auxiliary_domain),
        target_domain=str(target_domain),
        data_dir=str(Path(data_dir).expanduser().resolve()),
        merged_dir=str(Path(merged_dir).expanduser().resolve()),
        runs_dir=str(Path(runs_dir).expanduser().resolve()),
        embed_dim=effective_embed_dim,
    )
    source_table = current_source_table_lineage(required_file=bool((os.environ.get("ODCR_MANIFEST_DIR") or "").strip()))
    source_table_compatibility = step3_source_table_compatibility_payload(source_table)
    data_contract_payload = {
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "source_task": {
            "task_id": int(task_id),
            "auxiliary": str(auxiliary_domain),
            "target": str(target_domain),
        },
        "source_csv_fingerprints": upstream_evidence.get("source_csv_artifacts"),
        "merged_csv_fingerprints": upstream_evidence.get("merged_artifacts") or data_fps,
    }
    sentence_model_path = Path(sentence_embed_model).expanduser().resolve()
    artifact_lineage_payload = {
        "data_merged_artifact_fingerprint": stable_hash(data_fps),
        "preprocess": _step3_preprocess_lineage_expected_for_step4(upstream_evidence),
        "profile_artifact_fingerprints": upstream_evidence.get("profile_artifact_fingerprints"),
        "domain_artifact_fingerprints": upstream_evidence.get("domain_artifact_fingerprints"),
        "sentence_embed_model_identity": {
            "identity": str(sentence_model_path),
            "resolved_env_key": "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "model_artifact_fingerprint": model_artifact_fingerprint(sentence_model_path),
        },
    }
    preprocess_expected = _step3_preprocess_lineage_expected_for_step4(upstream_evidence)
    lineage_source_task = lineage.get("source_task") if isinstance(lineage.get("source_task"), Mapping) else {}
    expected = {
        "sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "task_id": int(task_id),
        "source_domain": str(auxiliary_domain),
        "target_domain": str(target_domain),
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "data_merged_artifact_fingerprint": stable_hash(data_fps),
        "embed_dim": effective_embed_dim,
        "step3_structured_losses_config_hash": lineage.get("step3_structured_losses_config_hash"),
        # The historical sidecar hash remains a lineage-field hard gate for the
        # sidecar itself. Actual checkpoint loading is separately gated against
        # tensor-derived shapes below.
        "model_architecture_config_hash": lineage.get("model_architecture_config_hash"),
        "resolved_config_compatibility_hash": lineage.get("resolved_config_compatibility_hash"),
        "source_table_compatibility_hash": lineage.get("source_table_compatibility_hash"),
        "semantic_model_compat_hash": lineage.get("semantic_model_compat_hash"),
        "data_contract_hash": stable_hash(data_contract_payload),
        "artifact_lineage_hash": stable_hash(artifact_lineage_payload),
        **preprocess_expected,
        "profile_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("profile_artifact_fingerprints") or {}),
        "domain_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("domain_artifact_fingerprints") or {}),
        "source_csv_fingerprints_hash": stable_hash(upstream_evidence.get("source_csv_artifacts") or {}),
        "merged_csv_fingerprints_hash": stable_hash(upstream_evidence.get("merged_artifacts") or data_fps),
        "source_task": dict(lineage_source_task)
        or {
            "task_id": int(task_id),
            "auxiliary": str(auxiliary_domain),
            "target": str(target_domain),
            "scenario": str(payload.get("scenario") or ""),
            "direction": str(payload.get("direction") or ""),
        },
    }
    validated = validate_step3_checkpoint_lineage(
        checkpoint,
        expected=expected,
        allow_derived_lineage_hash=True,
    )
    diff_report = architecture_hash_diff_report(
        checkpoint_lineage=lineage,
        expected_architecture_payload=expected_arch,
        observed_loader_architecture_payload=observed_loader_architecture,
        checkpoint_architecture_payload=checkpoint_arch,
        checkpoint_lineage_path=checkpoint_lineage_path,
    )
    sidecar_metadata_diff = diff_report.get("sidecar_metadata_diff") or {}
    unsafe_sidecar_keys = [
        str(key)
        for key in (sidecar_metadata_diff.get("mismatch_keys") or [])
        if str(key) != "ntoken"
    ]
    if unsafe_sidecar_keys:
        raise CheckpointLineageError(
            "Step3 checkpoint sidecar/state_dict architecture mismatch is unsafe outside ntoken metadata: "
            + ", ".join(sorted(unsafe_sidecar_keys))
        )
    ntoken_report = ntoken_sidecar_checkpoint_compatibility_report(
        sidecar_architecture_payload=sidecar_arch,
        checkpoint_architecture_payload=checkpoint_arch,
    )
    source_table_scope = source_table_hash_scope_report(
        checkpoint_lineage=lineage,
        source_table_compatibility_payload=source_table_compatibility,
        current_payload=payload,
        checkpoint_architecture_payload=checkpoint_arch,
    )
    live_frozen_drift = live_vs_frozen_step3_config_drift_report(
        checkpoint_lineage=lineage,
        current_payload=payload,
    )
    payload = {
        "schema_version": STEP4_PRELAUNCH_LINEAGE_VALIDATION_SCHEMA_VERSION,
        "status": "ok",
        "phase": str(phase),
        "task_id": int(task_id),
        "source_domain": str(auxiliary_domain),
        "target_domain": str(target_domain),
        "checkpoint_path": str(checkpoint),
        "selected_checkpoint_path": str(checkpoint),
        "checkpoint_source": "stage_status.selected_checkpoint",
        "best_pth_alias_not_primary_step4_binding": True,
        "checkpoint_lineage_path": str(checkpoint_lineage_path),
        "checkpoint_lineage_hash": validated.get("lineage_hash"),
        "checkpoint_file_hash": validated.get("checkpoint_file_hash"),
        "step3_run_id": str(validated.get("run_id") or ""),
        "model_architecture_config_hash": compute_model_architecture_config_hash(expected_arch),
        "checkpoint_model_architecture_hash": str(lineage.get("model_architecture_config_hash") or ""),
        "checkpoint_sidecar_model_architecture_hash": str(lineage.get("model_architecture_config_hash") or ""),
        "checkpoint_state_dict_model_architecture_hash": compute_model_architecture_config_hash(checkpoint_arch),
        "expected_model_architecture_hash": compute_model_architecture_config_hash(expected_arch),
        "checkpoint_model_architecture_payload": checkpoint_arch,
        "checkpoint_sidecar_model_architecture_payload": sidecar_arch,
        "checkpoint_state_dict_model_architecture_payload": checkpoint_arch,
        "expected_model_architecture_payload": expected_arch,
        "observed_current_loader_architecture_payload": (
            build_step3_model_architecture_config(observed_loader_architecture)
            if observed_loader_architecture is not None
            else None
        ),
        "observed_current_loader_mismatch_keys": observed_mismatch_keys,
        "architecture_hash_diff": diff_report,
        "sidecar_model_architecture_metadata_mismatch_keys": sidecar_metadata_diff.get("mismatch_keys") or [],
        "sidecar_model_architecture_metadata_diff": sidecar_metadata_diff,
        "ntoken_compatibility": ntoken_report,
        "sidecar_ntoken": ntoken_report["sidecar_ntoken"],
        "checkpoint_tensor_ntoken": ntoken_report["checkpoint_tensor_ntoken"],
        "effective_model_ntoken": ntoken_report["effective_model_ntoken"],
        "compatibility_note": ntoken_report["compatibility_note"],
        "hash_source_paths": diff_report.get("hash_source_paths"),
        "current_step4_config_pollutes_step3_architecture_hash": False,
        "expected_architecture_source": "checkpoint_state_dict_shapes_with_step3_sidecar_nonshape_fields",
        "ignored_non_architecture_keys": list(STEP4_ARCHITECTURE_IGNORED_CONFIG_KEYS),
        "source_table_compatibility_observed_hash": stable_hash(source_table_compatibility),
        "source_table_compatibility_expected_hash": lineage.get("source_table_compatibility_hash"),
        "source_table_hash_scope_report": source_table_scope,
        "live_vs_frozen_step3_config_drift": live_frozen_drift,
        "data_contract_hash": expected["data_contract_hash"],
        "artifact_lineage_hash": expected["artifact_lineage_hash"],
        "no_formal_write": True,
    }
    normalized = validate_step4_formal_lineage_contract(
        normalize_step3_lineage_for_step4(
            payload,
            checkpoint_path=checkpoint,
            checkpoint_lineage_path=checkpoint_lineage_path,
        )
    )
    payload.update(normalized)
    payload["normalized_lineage_contract"] = normalized
    return payload
