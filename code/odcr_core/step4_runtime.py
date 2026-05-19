"""Step4 pre-DDP cache, validation preflight, and runtime policy helpers."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    build_step4_export_lineage,
    step4_rcr_required_fields_hash,
)
from odcr_core.odcr_cf_routing import ODCFRoutingConfig, attach_odcr_cf_routing
from odcr_core.evidence_level import mark_schema_preview
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST
from odcr_core.training_checkpoint import file_fingerprint, read_checkpoint_lineage, stable_hash
from odcr_core.step4_checkpoint_lineage import (
    STEP4_LIVE_FROZEN_CONFIG_DRIFT_SCHEMA_VERSION,
    STEP4_PRELAUNCH_LINEAGE_VALIDATION_SCHEMA_VERSION,
    STEP4_SOURCE_TABLE_HASH_SCOPE_SCHEMA_VERSION,
    build_step4_observed_loader_architecture_config,
    normalize_step3_lineage_for_step4,
    require_step4_lineage_field,
    validate_step4_prelaunch_checkpoint_lineage,
    validate_step4_formal_lineage_contract,
)


STEP4_PREFLIGHT_SCHEMA_VERSION = "odcr_step4_bounded_preflight/1"
STEP4_CACHE_PREPARE_SCHEMA_VERSION = "odcr_step4_prepare_cache/1"
STEP4_RUNTIME_ENV_KNOBS = (
    "ODCR_STEP4_DECODE_THREADS",
    "ODCR_STEP4_DECODE_CHUNK",
    "ODCR_STEP4_PARTIAL_FORMAT",
    "ODCR_STEP4_PERF_LOG_INTERVAL",
)
STEP4_FORMAL_DRY_RUN_REPLAY_SCHEMA_VERSION = "odcr_step4_formal_runtime_contract_replay/1"


class Step4RuntimeError(RuntimeError):
    """Raised when Step4 runtime policy would be unsafe."""


def step4_formal_dry_run_meta_dir(cfg: Any) -> Path:
    run_id = str(getattr(cfg, "step4_run", None) or getattr(cfg, "run_name", None) or "dry_run")
    return (
        Path(cfg.repo_root).resolve()
        / "AI_analysis"
        / "01_raw_logs"
        / "step4_formal_runtime_contract"
        / "dry_run_meta"
        / f"task{int(cfg.task_id)}"
        / f"run{run_id}"
        / "meta"
    )


def write_step4_dry_run_resolved_artifacts(cfg: Any, snapshot: Mapping[str, Any]) -> Path:
    from odcr_core.manifests import write_resolved_config_artifacts

    meta = step4_formal_dry_run_meta_dir(cfg)
    write_resolved_config_artifacts(meta, dict(snapshot))
    return meta


@contextmanager
def _patched_env(updates: Mapping[str, str]):
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update({str(k): str(v) for k, v in updates.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _cfg_step4_runtime(cfg: Any) -> dict[str, Any]:
    raw = str(getattr(cfg, "step4_runtime_config_json", "") or "{}")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Step4RuntimeError(f"invalid step4_runtime_config_json: {exc}") from exc
    if not isinstance(obj, dict):
        raise Step4RuntimeError("step4_runtime_config_json must decode to object")
    return obj


def step4_runtime_env(cfg: Any, *, mode: str = "formal") -> dict[str, str]:
    runtime = _cfg_step4_runtime(cfg)
    return {
        "ODCR_STEP4_RUNTIME_CONFIG_JSON": json.dumps(runtime, sort_keys=True),
        "ODCR_STEP4_MODE": str(mode),
        "ODCR_STEP4_RUN_ID": str(getattr(cfg, "step4_run", "") or ""),
        "ODCR_STEP4_RCR_CONFIG_JSON": str(getattr(cfg, "step4_rcr_config_json", "") or ""),
        "ODCR_STEP3_TOKENIZER_MAX_LENGTH": str(int(getattr(cfg, "tokenizer_max_length", 0) or 0)),
        "ODCR_STEP3_EVIDENCE_MAX_LENGTH": str(int(getattr(cfg, "evidence_max_length", 0) or 0)),
        "ODCR_UPSTREAM_RESOLUTION_JSON": str(getattr(cfg, "upstream_resolution_json", "") or "{}"),
    }


def reject_step4_formal_env_overrides(*, mode: str = "formal", environ: Mapping[str, str] | None = None) -> None:
    env = environ or os.environ
    if str(mode or "formal") != "formal":
        return
    active = [key for key in STEP4_RUNTIME_ENV_KNOBS if str(env.get(key) or "").strip()]
    if active:
        raise Step4RuntimeError(
            "Step4 formal runtime refuses bare ODCR_STEP4_* perf/env knobs: "
            + ", ".join(active)
            + ". Use configs/odcr.yaml: step4.runtime so resolved_config/source_table record the setting."
        )


def _repo_path(repo_root: str | Path, raw: Any, *, field: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise Step4RuntimeError(f"Step4 upstream resolution missing {field}; refusing best.pth alias fallback.")
    root = Path(repo_root).expanduser().resolve()
    path = Path(text).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _upstream_resolution_payload(cfg: Any) -> dict[str, Any]:
    raw = str(getattr(cfg, "upstream_resolution_json", "") or "").strip()
    if not raw:
        raise Step4RuntimeError("Step4 requires upstream_resolution_json from stage_status resolver.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Step4RuntimeError(f"Step4 upstream_resolution_json is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise Step4RuntimeError("Step4 upstream_resolution_json must decode to an object.")
    return payload


def step3_selected_checkpoint_binding(cfg: Any) -> dict[str, Any]:
    payload = _upstream_resolution_payload(cfg)
    validation = payload.get("stage_status_validation")
    if not isinstance(validation, Mapping):
        validation = {}
    stage_status = payload.get("stage_status")
    if not isinstance(stage_status, Mapping):
        stage_status = {}
    selected_raw = validation.get("selected_checkpoint") or stage_status.get("selected_checkpoint")
    selected = _repo_path(cfg.repo_root, selected_raw, field="stage_status.selected_checkpoint")
    if not selected.is_file():
        raise Step4RuntimeError(f"Step4 selected checkpoint is missing: {selected}")
    expected_hash = str(validation.get("selected_checkpoint_hash") or stage_status.get("selected_checkpoint_hash") or "")
    selected_fp = {"exists": True, "is_file": True}
    selected_hash = expected_hash
    if not selected_hash:
        selected_fp = file_fingerprint(selected)
        selected_hash = str(selected_fp.get("sha256") or "")
    if expected_hash and selected_hash != expected_hash:
        raise Step4RuntimeError(
            "Step4 selected checkpoint hash mismatch: "
            f"stage_status={expected_hash!r} actual={selected_hash!r}"
        )
    run_dir = _repo_path(cfg.repo_root, payload.get("run_dir") or getattr(cfg, "step3_checkpoint_dir", ""), field="upstream run_dir")
    best_alias = (run_dir / "model" / "best.pth").resolve()
    latest_alias = (run_dir / "model" / "latest.pth").resolve()

    def _alias_report(path: Path, name: str) -> dict[str, Any]:
        sidecar = _load_json(Path(str(path) + ".lineage.json"))
        fp = {"exists": path.exists(), "is_file": path.is_file()}
        alias_hash = str(sidecar.get("checkpoint_file_hash") or "")
        if path.is_file() and not alias_hash:
            fp = file_fingerprint(path)
            alias_hash = str(fp.get("sha256") or "")
        return {
            "name": name,
            "path": str(path),
            "exists": bool(fp.get("exists") and fp.get("is_file")),
            "file_hash": alias_hash,
            "selected_checkpoint_hash": selected_hash,
            "alias_consistent": bool(alias_hash and alias_hash == selected_hash),
            "used_as_primary": False,
        }

    best_report = _alias_report(best_alias, "best.pth")
    latest_report = _alias_report(latest_alias, "latest.pth")
    return {
        "schema_version": "odcr_step4_selected_checkpoint_binding/1",
        "checkpoint_source": "stage_status.selected_checkpoint",
        "selected_checkpoint_path": str(selected),
        "selected_checkpoint_hash": selected_hash,
        "stage_status_path": validation.get("status_path") or payload.get("stage_status_path"),
        "eval_handoff": validation.get("eval_handoff") or stage_status.get("eval_handoff"),
        "run_dir": str(run_dir),
        "best_pth_alias": best_report,
        "latest_pth_alias": latest_report,
        "alias_consistent": bool(best_report["alias_consistent"]),
        "best_pth_alias_not_primary_step4_binding": True,
    }


def _layout_env(cfg: Any, *, manifest_dir_override: str | Path | None = None) -> dict[str, str]:
    manifest_dir = Path(manifest_dir_override).resolve() if manifest_dir_override is not None else Path(cfg.manifest_dir).resolve()
    out = {
        "ODCR_ROOT": str(Path(cfg.repo_root).resolve()),
        "ODCR_STAGE_RUN_DIR": str(Path(cfg.checkpoint_dir).resolve()),
        "ODCR_MANIFEST_DIR": str(manifest_dir),
        "ODCR_LOG_DIR": str(manifest_dir),
        "ODCR_STEP4_RUN_ID": str(getattr(cfg, "step4_run", "") or ""),
        "ODCR_RESOLVED_DATA_DIR": str(Path(cfg.data_dir).resolve()),
        "ODCR_RESOLVED_MERGED_DIR": str(Path(cfg.merged_dir).resolve()),
        "ODCR_RESOLVED_RUNS_DIR": str(Path(cfg.runs_dir).resolve()),
        "ODCR_RESOLVED_CACHE_DIR": str(Path(cfg.cache_dir).resolve()),
        "ODCR_RESOLVED_MODELS_DIR": str(Path(cfg.models_dir).resolve()),
        "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(Path(cfg.step5_text_model).resolve()),
        "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(Path(cfg.sentence_embed_model).resolve()),
        "ODCR_RESOLVED_EMBED_DIM": str(int(cfg.embed_dim)),
        "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON": str(getattr(cfg, "effective_training_payload_json", "") or "{}"),
    }
    _fs = str(getattr(cfg, "config_field_sources_json", "") or "").strip()
    if _fs:
        out["ODCR_CONFIG_FIELD_SOURCES_JSON"] = _fs
    _tfp = str(getattr(cfg, "training_semantic_fingerprint", "") or "").strip()
    if _tfp:
        out["ODCR_TRAINING_SEMANTIC_FINGERPRINT"] = _tfp
    _gfp = str(getattr(cfg, "generation_semantic_fingerprint", "") or "").strip()
    if _gfp:
        out["ODCR_GENERATION_SEMANTIC_FINGERPRINT"] = _gfp
    _rd = str(getattr(cfg, "runtime_diagnostics_fingerprint", "") or "").strip()
    if _rd:
        out["ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT"] = _rd
    if getattr(cfg, "step3_checkpoint_dir", None):
        out["ODCR_STEP3_RUN_DIR"] = str(Path(cfg.step3_checkpoint_dir).resolve())
        binding = step3_selected_checkpoint_binding(cfg)
        out["ODCR_STEP3_SELECTED_CHECKPOINT"] = binding["selected_checkpoint_path"]
        out["ODCR_STEP3_SELECTED_CHECKPOINT_HASH"] = binding["selected_checkpoint_hash"]
        out["ODCR_STEP3_SELECTED_CHECKPOINT_SOURCE"] = binding["checkpoint_source"]
    out.update(step4_runtime_env(cfg, mode="preflight"))
    return out


def validate_step4_prelaunch_lineage_for_config(
    cfg: Any,
    *,
    phase: str = "prelaunch",
    manifest_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Validate Step3 checkpoint compatibility before dry-run, preflight, or torchrun."""

    with _patched_env(_layout_env(cfg, manifest_dir_override=manifest_dir_override)):
        from executors.step3_train_core import get_odcr_text_tokenizer

        observed_arch = build_step4_observed_loader_architecture_config(
            data_dir=cfg.data_dir,
            auxiliary_domain=str(cfg.auxiliary),
            target_domain=str(cfg.target),
            tokenizer_length=len(get_odcr_text_tokenizer()),
            nlayers=int(getattr(cfg, "nlayers", 2) or 2),
            nhead=int(getattr(cfg, "nhead", 2) or 2),
            nhid=int(getattr(cfg, "nhid", 2048) or 2048),
            dropout=float(getattr(cfg, "dropout", 0.2) or 0.2),
        )
        checkpoint_binding = step3_selected_checkpoint_binding(cfg)
        checkpoint_path = Path(checkpoint_binding["selected_checkpoint_path"])
        payload = validate_step4_prelaunch_checkpoint_lineage(
            checkpoint_path=checkpoint_path,
            task_id=int(cfg.task_id),
            auxiliary_domain=str(cfg.auxiliary),
            target_domain=str(cfg.target),
            data_dir=cfg.data_dir,
            merged_dir=cfg.merged_dir,
            runs_dir=cfg.runs_dir,
            sentence_embed_model=cfg.sentence_embed_model,
            embed_dim=int(cfg.embed_dim),
            observed_loader_architecture=observed_arch,
            phase=phase,
        ) | {"checkpoint_binding": checkpoint_binding, "alias_consistency": checkpoint_binding["best_pth_alias"]}
        normalized = validate_step4_formal_lineage_contract(
            normalize_step3_lineage_for_step4(
                payload,
                checkpoint_path=checkpoint_path,
                checkpoint_lineage_path=payload.get("checkpoint_lineage_path"),
            )
        )
        payload.update(normalized)
        payload["normalized_lineage_contract"] = normalized
        return payload


def validate_step4_prelaunch_lineage_lightweight_for_config(
    cfg: Any,
    *,
    phase: str = "dry_run",
    manifest_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only Step4 lineage preview for default dry-run.

    This deliberately avoids importing Step3 model code or loading the Step3
    checkpoint tensor payload. Deep tensor-shape validation remains on the
    formal/preflight path; dry-run is a bounded schema/status preview.
    """

    with _patched_env(_layout_env(cfg, manifest_dir_override=manifest_dir_override)):
        binding = step3_selected_checkpoint_binding(cfg)
        checkpoint = Path(binding["selected_checkpoint_path"]).expanduser().resolve()
        checkpoint_lineage_path = Path(str(checkpoint) + ".lineage.json")
        lineage = read_checkpoint_lineage(
            checkpoint,
            expected_stage="step3",
            allow_derived_lineage_hash=True,
        )
        payload = _upstream_resolution_payload(cfg)
        validation = payload.get("stage_status_validation") if isinstance(payload.get("stage_status_validation"), Mapping) else {}
        status = payload.get("stage_status") if isinstance(payload.get("stage_status"), Mapping) else {}
        arch = lineage.get("model_architecture_config") if isinstance(lineage.get("model_architecture_config"), Mapping) else {}
        ntoken = arch.get("ntoken") if isinstance(arch, Mapping) else None
        preview_payload = {
            "schema_version": STEP4_PRELAUNCH_LINEAGE_VALIDATION_SCHEMA_VERSION,
            "status": "ok",
            "phase": str(phase),
            "task_id": int(cfg.task_id),
            "source_domain": str(cfg.auxiliary),
            "target_domain": str(cfg.target),
            "checkpoint_path": str(checkpoint),
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_path": str(checkpoint),
            "checkpoint_source": "stage_status.selected_checkpoint",
            "best_pth_alias_not_primary_step4_binding": True,
            "checkpoint_lineage_path": str(checkpoint_lineage_path),
            "checkpoint_lineage_hash": str(lineage.get("lineage_hash") or ""),
            "lineage_hash": str(lineage.get("lineage_hash") or ""),
            "lineage_hash_source": str(lineage.get("lineage_hash_source") or "checkpoint_lineage.lineage_hash"),
            "checkpoint_file_hash": str(lineage.get("checkpoint_file_hash") or binding.get("selected_checkpoint_hash") or ""),
            "checkpoint_sha256": str(lineage.get("checkpoint_file_hash") or binding.get("selected_checkpoint_hash") or ""),
            "checkpoint_sha256_source": (
                "checkpoint_lineage.checkpoint_file_hash"
                if str(lineage.get("checkpoint_file_hash") or "")
                else "stage_status.selected_checkpoint_hash"
            ),
            "step3_run_id": str(lineage.get("run_id") or status.get("run_id") or validation.get("run_id") or ""),
            "model_architecture_config_hash": str(lineage.get("model_architecture_config_hash") or ""),
            "checkpoint_model_architecture_hash": str(lineage.get("model_architecture_config_hash") or ""),
            "checkpoint_sidecar_model_architecture_hash": str(lineage.get("model_architecture_config_hash") or ""),
            "checkpoint_state_dict_model_architecture_hash": "not_checked_dry_run_lightweight",
            "expected_model_architecture_hash": str(lineage.get("model_architecture_config_hash") or ""),
            "effective_model_ntoken": ntoken if ntoken not in (None, "") else "not_checked_dry_run_lightweight",
            "sidecar_ntoken": ntoken if ntoken not in (None, "") else "not_checked_dry_run_lightweight",
            "checkpoint_tensor_ntoken": "not_checked_dry_run_lightweight",
            "stage_status_path": str(validation.get("status_path") or status.get("stage_status") or ""),
            "eval_handoff_path": str(validation.get("eval_handoff") or status.get("eval_handoff") or ""),
            "run_summary_path": str(validation.get("run_summary") or status.get("run_summary") or ""),
            "source_table_path": str(validation.get("source_table") or status.get("source_table") or ""),
            "resolved_config_path": str(validation.get("resolved_config") or status.get("resolved_config") or ""),
            "alias_consistency": binding["best_pth_alias"],
            "best_pth_alias_consistent": bool((binding.get("best_pth_alias") or {}).get("alias_consistent")),
            "latest_pth_alias_consistent": bool((binding.get("latest_pth_alias") or {}).get("alias_consistent")),
            "used_checkpoint_source": binding["checkpoint_source"],
            "compatibility_status": "ok",
            "checkpoint_binding": binding,
            "source_table_hash_scope_report": {
                "schema_version": STEP4_SOURCE_TABLE_HASH_SCOPE_SCHEMA_VERSION,
                "status": "skipped_dry_run_lightweight",
                "blocking": False,
                "reason": "default Step4 dry-run is bounded to status/header/sidecar reads",
            },
            "live_vs_frozen_step3_config_drift": {
                "schema_version": STEP4_LIVE_FROZEN_CONFIG_DRIFT_SCHEMA_VERSION,
                "status": "skipped_dry_run_lightweight",
                "policy": "Deep Step3 frozen-vs-live drift validation is reserved for formal/preflight validation.",
            },
            "deep_tensor_shape_gate": "skipped_dry_run_lightweight",
            "no_model_import": True,
            "no_checkpoint_tensor_load": True,
            "dry_run_lightweight": True,
            "no_formal_write": True,
        }
        normalized = validate_step4_formal_lineage_contract(
            normalize_step3_lineage_for_step4(
                preview_payload,
                checkpoint_path=checkpoint,
                upstream_resolution=payload,
                checkpoint_lineage_path=checkpoint_lineage_path,
            )
        )
        preview_payload.update(normalized)
        preview_payload["normalized_lineage_contract"] = normalized
        return preview_payload


def _checkpoint_lineage_for_cache(cfg: Any) -> tuple[str, dict[str, Any]]:
    ckpt_dir = Path(str(cfg.step3_checkpoint_dir or "")).expanduser().resolve()
    binding = step3_selected_checkpoint_binding(cfg)
    checkpoint = Path(binding["selected_checkpoint_path"])
    sidecar = Path(str(checkpoint) + ".lineage.json")
    payload = _load_json(sidecar)
    normalized = normalize_step3_lineage_for_step4(
        {
            **payload,
            "checkpoint_binding": binding,
            "checkpoint_path": str(checkpoint),
            "selected_checkpoint_path": binding["selected_checkpoint_path"],
            "checkpoint_file_hash": payload.get("checkpoint_file_hash") or binding.get("selected_checkpoint_hash"),
            "checkpoint_lineage_path": str(sidecar),
            "status": "ok",
        },
        checkpoint_path=checkpoint,
        checkpoint_lineage_path=sidecar,
    )
    lineage_hash = str(require_step4_lineage_field(normalized, "lineage_hash"))
    return lineage_hash, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_source": binding["checkpoint_source"],
        "selected_checkpoint_path": binding["selected_checkpoint_path"],
        "best_pth_alias": binding["best_pth_alias"],
        "checkpoint_hash": str(normalized.get("checkpoint_sha256") or ""),
        "checkpoint_lineage_path": str(sidecar),
        "checkpoint_lineage_hash": lineage_hash,
        "normalized_lineage_contract": normalized,
    }


def prepare_step4_encoded_cache(cfg: Any, *, dry_run: bool = False, build_allowed: bool = True) -> dict[str, Any]:
    """Build or validate the Step4 encoded cache before any DDP/NCCL setup."""
    manifest_override = step4_formal_dry_run_meta_dir(cfg) if dry_run else None
    with _patched_env(_layout_env(cfg, manifest_dir_override=manifest_override)):
        task = int(cfg.task_id)
        aug_csv = Path(cfg.merged_dir).resolve() / str(task) / "aug_train.csv"
        if not aug_csv.is_file():
            raise Step4RuntimeError(f"missing Step4 source CSV: {aug_csv}")
        lineage_hash, lineage = _checkpoint_lineage_for_cache(cfg)
        if dry_run:
            preview_rows = int(max(1, min(64, int(getattr(cfg, "dry_run_sample_rows", 64) or 64))))
            preview_df = pd.read_csv(aug_csv, nrows=preview_rows)
            columns = [str(col) for col in preview_df.columns]
            required_columns = {"domain", "explanation", "item"}
            missing = sorted(required_columns - set(columns))
            target_preview_count = (
                int((preview_df["domain"].astype(str) == "target").sum())
                if "domain" in preview_df.columns
                else 0
            )
            return {
                "schema_version": STEP4_CACHE_PREPARE_SCHEMA_VERSION,
                "stage": "step4",
                "phase": "prepare-cache-preview",
                "task": task,
                "source": cfg.auxiliary,
                "target": cfg.target,
                "upstream_step3_run_id": str(cfg.from_run),
                "cache_dir": None,
                "cache_hit": None,
                "cache_reason": "not_checked_dry_run_lightweight",
                "row_count": None,
                "sample_count": target_preview_count,
                "csv_path": str(aug_csv),
                "csv_header": columns,
                "bounded_sample_rows": int(len(preview_df)),
                "missing_required_columns": missing,
                "fingerprint_hash": None,
                "step3_checkpoint": lineage,
                "no_torch_distributed_collective": True,
                "dry_run": True,
                "dry_run_lightweight": True,
                "full_csv_read": False,
                "cache_manifest_checked": False,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

        from datasets import Dataset, load_from_disk
        from executors import step4_engine
        from paths_config import require_step5_text_model_dir

        train_df = pd.read_csv(aug_csv)
        train_df["item"] = train_df["item"].astype(str)
        train_df = train_df[train_df["explanation"].notna()].reset_index(drop=True)
        target_df = train_df[train_df["domain"] == "target"].copy()
        target_df["domain"] = "auxiliary"
        target_df["sample_id"] = range(len(target_df))
        processor = step4_engine.Processor(
            cfg.auxiliary,
            cfg.target,
            max_length=int(cfg.tokenizer_max_length),
            evidence_length=int(cfg.evidence_max_length),
        )
        fingerprint = step4_engine._step4_encoded_cache_fingerprint(
            task,
            str(aug_csv),
            cfg.auxiliary,
            cfg.target,
            require_step5_text_model_dir(),
            int(processor.max_length),
            lineage_hash,
        )
        cache_dir = Path(step4_engine._step4_encoded_cache_dir(task, fingerprint))
        valid, reason = step4_engine._step4_encoded_cache_manifest_matches(
            str(cache_dir),
            expected_fingerprint=fingerprint,
            expected_rows=len(target_df),
        )
        payload = {
            "schema_version": STEP4_CACHE_PREPARE_SCHEMA_VERSION,
            "stage": "step4",
            "phase": "prepare-cache",
            "task": task,
            "source": cfg.auxiliary,
            "target": cfg.target,
            "upstream_step3_run_id": str(cfg.from_run),
            "cache_dir": str(cache_dir),
            "cache_hit": bool(valid),
            "cache_reason": reason,
            "row_count": int(len(target_df)),
            "sample_count": int(len(target_df)),
            "fingerprint_hash": str(fingerprint.get("fingerprint_hash") or ""),
            "step3_checkpoint": lineage,
            "no_torch_distributed_collective": True,
            "dry_run": bool(dry_run),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if valid:
            return payload
        if not build_allowed:
            raise Step4RuntimeError(f"Step4 encoded cache missing/invalid ({reason}); run --prepare-cache before formal DDP.")

        partial = cache_dir.with_name(cache_dir.name + f".partial.{os.getpid()}")
        failed_marker = Path(step4_engine._step4_encoded_cache_failed_marker_path(str(cache_dir)))
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        failed_marker.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            dataset = Dataset.from_pandas(target_df)
            encoded = dataset.map(lambda sample: processor(sample), num_proc=int(cfg.num_proc), desc="Step4 prepare-cache")
            encoded.save_to_disk(str(partial))
            step4_engine._write_step4_encoded_cache_manifest(
                str(partial),
                fingerprint=fingerprint,
                row_count=len(encoded),
            )
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            os.replace(str(partial), str(cache_dir))
            if failed_marker.exists():
                failed_marker.unlink()
            loaded = load_from_disk(str(cache_dir))
            if len(loaded) != len(target_df):
                raise Step4RuntimeError("prepared cache row count mismatch after atomic publish")
        except Exception as exc:
            with open(failed_marker, "w", encoding="utf-8") as handle:
                handle.write(f"failed_at={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n{exc}\n")
            shutil.rmtree(partial, ignore_errors=True)
            raise
        payload["cache_hit"] = False
        payload["cache_reason"] = "built"
        payload["cache_built"] = True
        return payload


def _safe_validation_namespace(namespace: str | None) -> str:
    raw = str(namespace or "").strip() or f"step4_preflight_{int(time.time())}"
    if raw in {"latest", "formal"} or "/" in raw or "\\" in raw or ".." in raw:
        raise Step4RuntimeError(f"invalid Step4 validation namespace: {raw!r}")
    return raw


def _preflight_dir(cfg: Any, namespace: str) -> Path:
    return Path(cfg.repo_root).resolve() / "runs" / "step4_preflight" / f"task{int(cfg.task_id)}" / namespace


def _load_step4_rcr_json_for_preflight(cfg: Any, candidate_config: str | Path | None) -> tuple[str, dict[str, Any]]:
    if not candidate_config:
        raw = str(getattr(cfg, "step4_rcr_config_json", "") or "")
        return raw, {"candidate_config": None, "candidate_id": None, "source": "resolved_config"}
    path = Path(candidate_config)
    if not path.is_absolute():
        path = Path(cfg.repo_root).resolve() / path
    if not path.is_file():
        raise Step4RuntimeError(f"candidate config not found: {path}")
    superseded_sidecar = Path(str(path) + ".superseded_by_real_gpu_evidence.json")
    if superseded_sidecar.is_file():
        raise Step4RuntimeError(
            "candidate config is superseded CPU-preview evidence and cannot be used for Step4 tuning/preflight: "
            f"{path}"
        )
    try:
        import yaml
    except ImportError as exc:
        raise Step4RuntimeError("PyYAML is required for --candidate-config") from exc
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, Mapping):
        raise Step4RuntimeError(f"candidate config must be a mapping: {path}")
    step4 = raw_payload.get("step4")
    rcr = step4.get("rcr") if isinstance(step4, Mapping) else None
    if not isinstance(rcr, Mapping):
        raise Step4RuntimeError(f"candidate config is missing step4.rcr: {path}")
    candidate_id = raw_payload.get("candidate_id") or raw_payload.get("candidate") or path.stem
    return (
        json.dumps(dict(rcr), ensure_ascii=False, sort_keys=True),
        {
            "candidate_config": str(path),
            "candidate_id": str(candidate_id),
            "source": "candidate_config.step4.rcr",
        },
    )


def _tree_fingerprint(path: Path, *, lightweight: bool = False, max_entries: int = 256) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "file_count": 0}
    if path.is_file():
        if lightweight:
            stat = path.stat()
            return {
                "path": str(path),
                "exists": True,
                "is_file": True,
                "sha256": None,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "file_count": 1,
                "fingerprint_mode": "stat_only",
            }
        fp = file_fingerprint(str(path))
        return {
            "path": str(path),
            "exists": True,
            "is_file": True,
            "sha256": fp.get("sha256"),
            "size": fp.get("size"),
            "mtime_ns": fp.get("mtime_ns"),
            "file_count": 1,
        }
    if lightweight:
        records: list[dict[str, Any]] = []
        truncated = False
        for idx, item in enumerate(sorted(path.iterdir(), key=lambda p: p.name)):
            if idx >= max_entries:
                truncated = True
                break
            try:
                stat = item.stat()
            except OSError:
                continue
            records.append(
                {
                    "name": item.name,
                    "is_file": item.is_file(),
                    "is_dir": item.is_dir(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        return {
            "path": str(path),
            "exists": True,
            "is_file": False,
            "sha256": stable_hash(records),
            "entry_count": len(records),
            "file_count": sum(1 for item in records if item.get("is_file")),
            "truncated": bool(truncated),
            "fingerprint_mode": "stat_listing",
        }
    records: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        try:
            rel = item.relative_to(path).as_posix()
        except ValueError:
            rel = item.name
        fp = file_fingerprint(str(item))
        records.append(
            {
                "rel": rel,
                "sha256": fp.get("sha256"),
                "size": fp.get("size"),
                "mtime_ns": fp.get("mtime_ns"),
            }
        )
    return {
        "path": str(path),
        "exists": True,
        "is_file": False,
        "sha256": stable_hash(records),
        "file_count": len(records),
    }


def _formal_namespace_snapshot(cfg: Any, *, lightweight: bool = False) -> dict[str, Any]:
    root = Path(cfg.repo_root).resolve()
    task = int(cfg.task_id)
    step3_dir = Path(str(cfg.step3_checkpoint_dir or "")).resolve() if getattr(cfg, "step3_checkpoint_dir", None) else root / "runs" / "step3" / f"task{task}" / "2"
    selected_checkpoint = step3_selected_checkpoint_binding(cfg)["selected_checkpoint_path"] if getattr(cfg, "step3_checkpoint_dir", None) else str(step3_dir / "model" / "best_observed.pth")
    return {
        "schema_version": "odcr_step4_formal_namespace_snapshot/1",
        "step4_latest": _tree_fingerprint(root / "runs" / "step4" / f"task{task}" / "latest.json", lightweight=lightweight),
        "step4_formal_task_dir": _tree_fingerprint(root / "runs" / "step4" / f"task{task}", lightweight=lightweight),
        "step5_latest": _tree_fingerprint(root / "runs" / "step5" / f"task{task}" / "latest.json", lightweight=lightweight),
        "eval_latest": _tree_fingerprint(root / "runs" / "eval" / f"task{task}" / "latest.json", lightweight=lightweight),
        "config": _tree_fingerprint(root / "configs" / "odcr.yaml", lightweight=lightweight),
        "step3_stage_status": _tree_fingerprint(step3_dir / "meta" / "stage_status.json", lightweight=lightweight),
        "step3_eval_handoff": _tree_fingerprint(step3_dir / "meta" / "eval_handoff.json", lightweight=lightweight),
        "step3_selected_checkpoint": _tree_fingerprint(Path(selected_checkpoint), lightweight=lightweight),
    }


def _snapshot_polluted(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    keys = (
        "step4_latest",
        "step4_formal_task_dir",
        "step5_latest",
        "eval_latest",
        "config",
        "step3_stage_status",
        "step3_eval_handoff",
        "step3_selected_checkpoint",
    )
    for key in keys:
        if before.get(key) != after.get(key):
            return True
    return False


def _step4_task_root(cfg: Any) -> Path:
    return Path(cfg.repo_root).resolve() / "runs" / "step4" / f"task{int(cfg.task_id)}"


def next_available_step4_run_id(cfg: Any) -> int:
    parent = _step4_task_root(cfg)
    best = 0
    if parent.is_dir():
        for child in parent.iterdir():
            if child.is_dir() and child.name.isdigit():
                best = max(best, int(child.name))
    return best + 1


def _formal_run_plan(cfg: Any) -> dict[str, Any]:
    run_id = str(getattr(cfg, "step4_run", None) or Path(str(cfg.checkpoint_dir)).name)
    run_dir = Path(cfg.checkpoint_dir).resolve()
    meta_dir = Path(cfg.manifest_dir).resolve()
    full_log = meta_dir / "full.log"
    expected_run_dir = _step4_task_root(cfg) / run_id
    if run_dir != expected_run_dir.resolve():
        raise Step4RuntimeError(
            "Step4 run-id propagation mismatch: "
            f"run_id={run_id!r} run_dir={run_dir} expected={expected_run_dir.resolve()}"
        )
    if meta_dir != (run_dir / "meta").resolve():
        raise Step4RuntimeError(
            "Step4 log/meta plan mismatch: "
            f"meta_dir={meta_dir} expected={(run_dir / 'meta').resolve()}"
        )
    exists = run_dir.exists()
    return {
        "planned_run_id": int(run_id) if run_id.isdigit() else run_id,
        "planned_run_id_text": run_id,
        "planned_run_dir": str(run_dir),
        "planned_meta_dir": str(meta_dir),
        "planned_full_log": str(full_log),
        "run_dir_exists": bool(exists),
        "run_id_overwrite_status": "blocked_existing_run" if exists else "ok_new_run",
        "next_available_run_id": next_available_step4_run_id(cfg),
    }


def validate_step4_formal_runtime_contract_replay(
    cfg: Any,
    *,
    snapshot: Mapping[str, Any] | None = None,
    phase: str = "dry_run",
) -> dict[str, Any]:
    """Dry-run the formal Step4 runtime contract without formal writes."""

    if snapshot is not None:
        dry_meta = write_step4_dry_run_resolved_artifacts(cfg, snapshot)
    else:
        dry_meta = step4_formal_dry_run_meta_dir(cfg)
    plan = _formal_run_plan(cfg)
    if plan["run_dir_exists"]:
        raise Step4RuntimeError(
            "Step4 dry-run refuses an existing formal run directory; "
            f"planned_run_id={plan['planned_run_id_text']} planned_run_dir={plan['planned_run_dir']} "
            f"next_available_run_id={plan['next_available_run_id']}"
        )
    lightweight = str(phase) == "dry_run"
    before = _formal_namespace_snapshot(cfg, lightweight=lightweight)
    if lightweight:
        lineage_validation = validate_step4_prelaunch_lineage_lightweight_for_config(
            cfg,
            phase=phase,
            manifest_dir_override=dry_meta,
        )
        normalized = dict(lineage_validation["normalized_lineage_contract"])
    else:
        lineage_validation = validate_step4_prelaunch_lineage_for_config(
            cfg,
            phase=phase,
            manifest_dir_override=dry_meta,
        )
        normalized = validate_step4_formal_lineage_contract(
            normalize_step3_lineage_for_step4(
                lineage_validation,
                checkpoint_path=lineage_validation.get("selected_checkpoint"),
                checkpoint_lineage_path=lineage_validation.get("checkpoint_lineage_path"),
            )
        )
    cache_preview = prepare_step4_encoded_cache(cfg, dry_run=True, build_allowed=False)
    after = _formal_namespace_snapshot(cfg, lightweight=lightweight)
    polluted = _snapshot_polluted(before, after)
    train_obj = snapshot.get("train") if isinstance(snapshot, Mapping) else {}
    train = train_obj if isinstance(train_obj, Mapping) else {}
    payload = {
        "schema_version": STEP4_FORMAL_DRY_RUN_REPLAY_SCHEMA_VERSION,
        "status": "ok" if not polluted else "formal_namespace_polluted",
        "phase": str(phase),
        "task": int(cfg.task_id),
        **plan,
        "planned_run_id": plan["planned_run_id"],
        "planned_run_dir": plan["planned_run_dir"],
        "planned_full_log": plan["planned_full_log"],
        "run_id_overwrite_status": plan["run_id_overwrite_status"],
        "dry_run_meta_dir": str(dry_meta),
        "selected_checkpoint": normalized["selected_checkpoint"],
        "lineage_hash": normalized["lineage_hash"],
        "lineage_hash_source": normalized["lineage_hash_source"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "checkpoint_sha256_source": normalized["checkpoint_sha256_source"],
        "model_architecture_config_hash": normalized["model_architecture_config_hash"],
        "effective_model_ntoken": normalized["effective_model_ntoken"],
        "required_fields_status": normalized["required_fields_status"],
        "checkpoint_lineage_validation": lineage_validation,
        "normalized_lineage_contract": normalized,
        "pre_ddp_cache_preview": cache_preview,
        "g1_p3_formal_config": {
            "step4_rcr_present": bool(snapshot.get("step4_rcr")) if isinstance(snapshot, Mapping) else True,
            "step4_runtime_present": bool(snapshot.get("step4_runtime")) if isinstance(snapshot, Mapping) else True,
            "runtime_partial_format": (
                (snapshot.get("step4_runtime") or {}).get("partial_format")
                if isinstance(snapshot, Mapping) and isinstance(snapshot.get("step4_runtime"), Mapping)
                else None
            ),
        },
        "no_accum": {
            "batch_semantics_version": train.get("batch_semantics_version"),
            "grad_accum_removed": train.get("grad_accum_removed"),
            "global_batch_size": train.get("global_batch_size"),
            "per_gpu_batch_size": train.get("per_gpu_batch_size"),
            "ddp_world_size": train.get("ddp_world_size"),
        },
        "formal_latest_write": False,
        "formal_export_write": False,
        "will_write_formal_on_actual_run": True,
        "dry_run_no_formal_write": True,
        "formal_namespace_polluted": bool(polluted),
    }
    if polluted:
        raise Step4RuntimeError("Step4 formal dry-run replay changed formal namespace")
    report_path = (
        Path(cfg.repo_root).resolve()
        / "AI_analysis"
        / "05_final_reports"
        / "step4_formal_runtime_contract"
        / "formal_dry_run_replay_report.json"
    )
    atomic_write_json(report_path, payload)
    return payload


def _torchrun_cmd() -> list[str]:
    if shutil.which("torchrun"):
        return ["torchrun"]
    return [sys.executable, "-m", "torch.distributed.run"]


def _preflight_hardware_env(cfg: Any) -> dict[str, str]:
    out = {
        "ODCR_HARDWARE_PROFILE_JSON": str(getattr(cfg, "hardware_profile_json", "") or "{}"),
        "ODCR_HARDWARE_PRESET": str(getattr(cfg, "hardware_preset_id", "") or ""),
        "ODCR_RUNTIME_PRECISION_MODE": "fp32",
        "ODCR_RUNTIME_ALLOW_TF32": "1" if bool(getattr(cfg, "allow_tf32", False)) else "0",
        "ODCR_RUNTIME_AMP_AUTOCAST": "1" if bool(getattr(cfg, "amp_autocast", True)) else "0",
        "ODCR_RUNTIME_GRAD_SCALER": "0",
        "ODCR_STEP3_TOKENIZER_MAX_LENGTH": str(int(getattr(cfg, "tokenizer_max_length", 0) or 0)),
        "ODCR_STEP3_EVIDENCE_MAX_LENGTH": str(int(getattr(cfg, "evidence_max_length", 0) or 0)),
        "OMP_NUM_THREADS": str(int(getattr(cfg, "omp_num_threads", 1) or 1)),
        "MKL_NUM_THREADS": str(int(getattr(cfg, "mkl_num_threads", 1) or 1)),
        "TOKENIZERS_PARALLELISM": "true" if bool(getattr(cfg, "tokenizers_parallelism", False)) else "false",
        "ODCR_GLOBAL_EVAL_BATCH_SIZE": str(int(getattr(cfg, "global_eval_batch_size", 0) or 0)),
        "ODCR_EVAL_PER_GPU_BATCH_SIZE": str(int(getattr(cfg, "eval_per_gpu_batch_size", 0) or 0)),
        "ODCR_DECODE_PROFILE_JSON": str(getattr(cfg, "decode_profile_json", "") or "{}"),
        "ODCR_EVAL_PROFILE_NAME": str(getattr(cfg, "eval_profile_id", "") or ""),
    }
    try:
        launcher = json.loads(str(getattr(cfg, "launcher_env_effective_json", "") or "{}"))
    except json.JSONDecodeError:
        launcher = {}
    if isinstance(launcher, Mapping):
        cvd = launcher.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None and str(cvd).strip():
            out["CUDA_VISIBLE_DEVICES"] = str(cvd).strip()
    return out


def _write_gpu_preflight_input(
    cfg: Any,
    *,
    out_dir: Path,
    max_samples: int,
    rcr_config_json: str,
    candidate_meta: Mapping[str, Any],
    profile_utilization: bool,
) -> dict[str, Any]:
    with _patched_env({**_layout_env(cfg), **_preflight_hardware_env(cfg), "ODCR_STEP4_RCR_CONFIG_JSON": rcr_config_json}):
        from datasets import Dataset
        from executors import step4_engine

        print(
            f"[step4 gpu-shard preflight] preparing validation subset max_samples={int(max_samples)} "
            f"namespace={out_dir.name}",
            flush=True,
        )
        task = int(cfg.task_id)
        aug_csv = Path(cfg.merged_dir).resolve() / str(task) / "aug_train.csv"
        if not aug_csv.is_file():
            raise Step4RuntimeError(f"missing Step4 source CSV: {aug_csv}")
        target_parts: list[pd.DataFrame] = []
        seen = 0
        for chunk in pd.read_csv(aug_csv, chunksize=max(1024, int(max_samples) * 4)):
            chunk["item"] = chunk["item"].astype(str)
            chunk = chunk[chunk["explanation"].notna()]
            target = chunk[chunk["domain"] == "target"]
            if not target.empty:
                target_parts.append(target)
                seen += len(target)
            if seen >= max_samples:
                break
        target_df = (
            pd.concat(target_parts, ignore_index=True).head(max_samples).reset_index(drop=True).copy()
            if target_parts
            else pd.DataFrame()
        )
        if target_df.empty:
            raise Step4RuntimeError("bounded gpu-shard preflight found no target rows")
        target_df["domain"] = "auxiliary"
        target_df["sample_id"] = range(len(target_df))
        input_dir = out_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        target_csv = input_dir / "target_subset.csv"
        target_df.to_csv(target_csv, index=False, encoding="utf-8")
        processor = step4_engine.Processor(
            cfg.auxiliary,
            cfg.target,
            max_length=int(cfg.tokenizer_max_length),
            evidence_length=int(cfg.evidence_max_length),
        )
        encoded_dir = input_dir / "encoded_dataset"
        partial = input_dir / f"encoded_dataset.partial.{os.getpid()}"
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        if encoded_dir.exists():
            shutil.rmtree(encoded_dir, ignore_errors=True)
        t0 = time.perf_counter()
        dataset = Dataset.from_pandas(target_df)
        encoded = dataset.map(lambda sample: processor(sample), num_proc=int(cfg.num_proc), desc="Step4 gpu-shard preflight cache")
        encoded.save_to_disk(str(partial))
        os.replace(str(partial), str(encoded_dir))
        cache_wall = time.perf_counter() - t0
        print(
            f"[step4 gpu-shard preflight] cache ready rows={len(target_df)} wall_s={cache_wall:.3f}",
            flush=True,
        )
        try:
            hw_payload = json.loads(str(getattr(cfg, "hardware_profile_json", "") or "{}"))
        except json.JSONDecodeError:
            hw_payload = {}
        hw_budget = hw_payload.get("worker_budget_formula") if isinstance(hw_payload, Mapping) else {}
        if not isinstance(hw_budget, Mapping):
            hw_budget = {}
        cache_status = {
            "schema_version": "odcr_step4_gpu_shard_cache_status/1",
            "cache_phase_pre_ddp": True,
            "no_nccl_in_cache_phase": True,
            "cache_dir": str(encoded_dir),
            "cache_hit": False,
            "cache_reason": "validation_subset_built_pre_ddp",
            "sample_count": int(len(target_df)),
            "num_proc": int(cfg.num_proc),
            "max_parallel_cpu": int(hw_payload.get("max_parallel_cpu", 0) or 0) if isinstance(hw_payload, Mapping) else 0,
            "reserved_cpu": int(hw_budget.get("reserved_cpu", hw_payload.get("reserved_cpu", 2) if isinstance(hw_payload, Mapping) else 2)),
            "dataloader_num_workers_test": int(hw_payload.get("dataloader_num_workers_test", 0) or 0) if isinstance(hw_payload, Mapping) else 0,
            "worker_budget_formula": dict(hw_budget),
            "wall_seconds": cache_wall,
        }
        atomic_write_json(out_dir / "cache_status.json", cache_status)
        binding = step3_selected_checkpoint_binding(cfg)
        payload = {
            "schema_version": "odcr_step4_gpu_shard_preflight_payload/1",
            "task": task,
            "source": str(cfg.auxiliary),
            "target": str(cfg.target),
            "validation_namespace": out_dir.name,
            "out_dir": str(out_dir),
            "target_subset_csv": str(target_csv),
            "encoded_dataset_dir": str(encoded_dir),
            "max_samples": int(max_samples),
            "sample_count": int(len(target_df)),
            "candidate": dict(candidate_meta),
            "profile_utilization": bool(profile_utilization),
            "global_eval_batch_size": int(getattr(cfg, "global_eval_batch_size", 0) or 0),
            "eval_per_gpu_batch_size": int(getattr(cfg, "eval_per_gpu_batch_size", 0) or 0),
            "checkpoint_path": binding["selected_checkpoint_path"],
            "checkpoint_source": binding["checkpoint_source"],
            "checkpoint_binding": binding,
            "step3_checkpoint_dir": str(Path(str(cfg.step3_checkpoint_dir)).resolve()),
            "step3_run_id": str(cfg.from_run),
            "tokenizer_max_length": int(cfg.tokenizer_max_length),
            "evidence_max_length": int(cfg.evidence_max_length),
            "nlayers": int(getattr(cfg, "nlayers", 2) or 2),
            "nhead": int(getattr(cfg, "nhead", 2) or 2),
            "nhid": int(getattr(cfg, "nhid", 2048) or 2048),
            "dropout": float(getattr(cfg, "dropout", 0.2) or 0.2),
            "rcr_config": json.loads(rcr_config_json),
            "cache_status": cache_status,
        }
        payload_path = input_dir / "payload.json"
        atomic_write_json(payload_path, payload)
        return payload


def _run_gpu_shard_preflight(
    cfg: Any,
    *,
    out_dir: Path,
    max_samples: int,
    rcr_config_json: str,
    candidate_meta: Mapping[str, Any],
    profile_utilization: bool,
) -> dict[str, Any]:
    before = _formal_namespace_snapshot(cfg)
    payload = _write_gpu_preflight_input(
        cfg,
        out_dir=out_dir,
        max_samples=max_samples,
        rcr_config_json=rcr_config_json,
        candidate_meta=candidate_meta,
        profile_utilization=profile_utilization,
    )
    env = dict(os.environ)
    env.update(_layout_env(cfg))
    env.update(_preflight_hardware_env(cfg))
    env["ODCR_STAGE_RUN_DIR"] = str(out_dir.resolve())
    env["ODCR_MANIFEST_DIR"] = str(out_dir.resolve())
    env["ODCR_LOG_DIR"] = str(out_dir.resolve())
    env["ODCR_STEP3_RUN_DIR"] = str(Path(str(cfg.step3_checkpoint_dir)).resolve())
    env["ODCR_STEP4_RCR_CONFIG_JSON"] = rcr_config_json
    env["ODCR_STEP4_MODE"] = "preflight"
    env["PYTHONPATH"] = str(Path(cfg.code_dir).resolve()) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    cmd = [
        *_torchrun_cmd(),
        "--standalone",
        f"--nproc_per_node={int(cfg.ddp_world_size)}",
        "-m",
        "odcr_core.step4_gpu_preflight_runner",
        "--payload",
        str(Path(payload["out_dir"]) / "input" / "payload.json"),
    ]
    print(
        f"[step4 gpu-shard preflight] launching torchrun world_size={int(cfg.ddp_world_size)} "
        f"namespace={out_dir.name}",
        flush=True,
    )
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(Path(cfg.code_dir).resolve()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    wall = time.perf_counter() - t0
    atomic_write_json(
        out_dir / "gpu_shard_command.json",
        {
            "schema_version": "odcr_step4_gpu_shard_command/1",
            "cmd": cmd,
            "returncode": proc.returncode,
            "wall_seconds": wall,
            "stdout_tail": proc.stdout[-12000:],
            "stderr_tail": proc.stderr[-12000:],
        },
    )
    after = _formal_namespace_snapshot(cfg)
    pollution = _snapshot_polluted(before, after)
    pollution_report = {
        "schema_version": "odcr_step4_formal_pollution_report/1",
        "formal_pollution": bool(pollution),
        "before": before,
        "after": after,
        "checked_paths": list(before.keys()),
    }
    atomic_write_json(out_dir / "formal_pollution_report.json", pollution_report)
    if proc.returncode != 0:
        raise Step4RuntimeError(
            "Step4 gpu-shard preflight failed "
            f"returncode={proc.returncode}; see {out_dir / 'gpu_shard_command.json'}"
        )
    summary = _load_json(out_dir / "preflight_summary.json")
    if not summary:
        raise Step4RuntimeError(f"gpu-shard preflight did not write summary: {out_dir}")
    summary["formal_pollution"] = bool(pollution)
    summary["formal_latest_write"] = False
    summary["formal_export_write"] = False
    atomic_write_json(out_dir / "preflight_summary.json", summary)
    if pollution:
        raise Step4RuntimeError("bounded gpu-shard preflight changed formal namespace")
    return summary


def _rcr_distribution(df: pd.DataFrame) -> dict[str, Any]:
    sw = pd.to_numeric(df["sample_weight_hint"], errors="coerce")
    bucket = pd.to_numeric(df["confidence_bucket"], errors="coerce").fillna(-1).astype(int)
    return {
        "sample_count": int(len(df)),
        "route_scorer_count": int((pd.to_numeric(df["route_scorer"], errors="coerce").fillna(0).astype(int) == 1).sum()),
        "route_explainer_count": int((pd.to_numeric(df["route_explainer"], errors="coerce").fillna(0).astype(int) == 1).sum()),
        "train_keep_count": int((pd.to_numeric(df["train_keep"], errors="coerce").fillna(0).astype(int) == 1).sum()),
        "confidence_bucket_distribution": {str(k): int(v) for k, v in bucket.value_counts().sort_index().items()},
        "sample_weight_hint": {
            "min": float(sw.min()) if len(sw) else None,
            "mean": float(sw.mean()) if len(sw) else None,
            "max": float(sw.max()) if len(sw) else None,
        },
    }


def run_step4_bounded_preflight(
    cfg: Any,
    *,
    max_samples: int | None = None,
    validation_namespace: str | None = None,
    preflight_mode: str = "preview",
    force_gpu_forward: bool = False,
    profile_utilization: bool = False,
    candidate_config: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a bounded non-formal Step4 readiness preview on real task data.

    When CUDA is not available this performs a CPU schema/path/RCR preview and
    marks ``gpu_runtime_evidence=false``.  It never writes formal Step4 latest
    or formal exports.
    """
    namespace = _safe_validation_namespace(validation_namespace)
    out_dir = _preflight_dir(cfg, namespace)
    if "runs/step4/task" in out_dir.as_posix():
        raise Step4RuntimeError("preflight output would enter formal Step4 namespace")
    runtime = _cfg_step4_runtime(cfg)
    if max_samples is None:
        max_samples = int(runtime.get("preflight_default_max_samples", 128))
    max_samples = max(1, int(max_samples))
    mode = str(preflight_mode or "preview").strip().lower().replace("_", "-")
    if mode not in {"preview", "gpu-shard"}:
        raise Step4RuntimeError(f"invalid Step4 preflight mode: {preflight_mode!r}")
    rcr_config_json, candidate_meta = _load_step4_rcr_json_for_preflight(cfg, candidate_config)
    lineage_validation = validate_step4_prelaunch_lineage_for_config(
        cfg,
        phase="preflight_dry_run" if dry_run else f"preflight_{mode}",
        manifest_dir_override=step4_formal_dry_run_meta_dir(cfg) if dry_run else None,
    )
    if dry_run:
        return {
            "schema_version": STEP4_PREFLIGHT_SCHEMA_VERSION,
            "status": "dry_run",
            "task": int(cfg.task_id),
            "validation_namespace": namespace,
            "output_dir": str(out_dir),
            "max_samples": max_samples,
            "preflight_mode": mode,
            "force_gpu_forward": bool(force_gpu_forward),
            "profile_utilization": bool(profile_utilization),
            "candidate": candidate_meta,
            "checkpoint_lineage_validation": lineage_validation,
            "formal_latest_write": False,
            "formal_export_write": False,
        }
    if mode == "gpu-shard":
        if not force_gpu_forward:
            raise Step4RuntimeError("gpu-shard preflight requires --force-gpu-forward")
        out_dir.mkdir(parents=True, exist_ok=True)
        return _run_gpu_shard_preflight(
            cfg,
            out_dir=out_dir,
            max_samples=max_samples,
            rcr_config_json=rcr_config_json,
            candidate_meta=candidate_meta,
            profile_utilization=profile_utilization,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    formal_latest = Path(cfg.repo_root).resolve() / "runs" / "step4" / f"task{int(cfg.task_id)}" / "latest.json"
    before_latest_hash = file_fingerprint(str(formal_latest)).get("sha256") if formal_latest.exists() else None
    checkpoint_lineage_hash, checkpoint_lineage = _checkpoint_lineage_for_cache(cfg)
    aug_csv = Path(cfg.merged_dir).resolve() / str(int(cfg.task_id)) / "aug_train.csv"
    target_parts: list[pd.DataFrame] = []
    seen = 0
    for chunk in pd.read_csv(aug_csv, chunksize=max(1024, int(max_samples) * 4)):
        chunk["item"] = chunk["item"].astype(str)
        chunk = chunk[chunk["explanation"].notna()]
        target = chunk[chunk["domain"] == "target"]
        if not target.empty:
            target_parts.append(target)
            seen += len(target)
        if seen >= max_samples:
            break
    target_df = (
        pd.concat(target_parts, ignore_index=True).head(max_samples).reset_index(drop=True).copy()
        if target_parts
        else pd.DataFrame()
    )
    if target_df.empty:
        raise Step4RuntimeError("bounded preflight found no target rows")
    cf_df = target_df.copy()
    rating_target = pd.to_numeric(cf_df["rating"], errors="coerce").fillna(0.0)
    cf_df = cf_df.assign(
        domain="auxiliary",
        entropy=0.0,
        rating_target=rating_target,
        rating_counterfactual=rating_target,
        rating_delta=0.0,
        rating_stability_score=1.0,
        shared_latent_similarity=1.0,
        specific_latent_shift=0.5,
    )
    schema_evidence = mark_schema_preview(
        {
            "cpu_preview_fake_score_source": "step4_runtime.run_step4_bounded_preflight",
            "cpu_preview_proxy_fields": {
                "rating_delta": 0.0,
                "rating_stability_score": 1.0,
                "shared_latent_similarity": 1.0,
                "specific_latent_shift": 0.5,
            },
        }
    )
    rcr_config = ODCFRoutingConfig.from_json(rcr_config_json, require=True)
    routed = attach_odcr_cf_routing(target_df, cf_df, cfg=rcr_config)
    dist_payload = mark_schema_preview(_rcr_distribution(routed))
    required_fields = list((rcr_config.export or {}).get("required_fields") or [])
    missing_required = [name for name in required_fields if name not in routed.columns]
    required_check = mark_schema_preview(
        {
        "schema_version": "odcr_step4_required_fields_check/1",
        "passed": not missing_required,
        "missing": missing_required,
        "required_fields": required_fields,
        }
    )
    frozen_lineage = {
        "upstream_step3_run_id": str(cfg.from_run),
        "step3_checkpoint_path": checkpoint_lineage["checkpoint_path"],
        "step3_checkpoint_hash": checkpoint_lineage["checkpoint_hash"],
        "step3_checkpoint_lineage_hash": checkpoint_lineage_hash,
        "step3_stage_status_hash": _upstream_artifact_hash(cfg, "status_path"),
        "step3_eval_handoff_hash": _upstream_artifact_hash(cfg, "eval_handoff"),
    }
    lineage = build_step4_export_lineage(
        task_id=int(cfg.task_id),
        auxiliary_domain=str(cfg.auxiliary),
        target_domain=str(cfg.target),
        step3_checkpoint_lineage_hash=checkpoint_lineage_hash,
        step4_rcr_config=rcr_config.to_dict(),
        step4_run=str(getattr(cfg, "step4_run", "") or "preflight"),
        frozen_step3_lineage=frozen_lineage,
    )
    manifest_preview = mark_schema_preview(
        {
        "schema_version": "odcr_step4_manifest_preview/1",
        "export_manifest_name": STEP4_EXPORT_MANIFEST,
        "row_count": int(len(routed)),
        "rcr_required_fields_hash": step4_rcr_required_fields_hash(),
        "step4_export_lineage": lineage,
        "formal_export": False,
        }
    )
    index_preview = mark_schema_preview(
        {
        "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
        "file": INDEX_CONTRACT_FILENAME,
        "task_id": int(cfg.task_id),
        "step4_run": str(getattr(cfg, "step4_run", "") or "preflight"),
        "step4_export_lineage": lineage,
        }
    )
    import torch

    gpu_payload = mark_schema_preview(
        {
        "schema_version": "odcr_step4_cpu_gpu_utilization_snapshot/1",
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_runtime_evidence": False,
        "runtime_mode": "cpu_schema_rcr_preview",
        "not_step4_runtime_evidence": True,
        }
    )
    sample_hash = stable_hash(
        {
            "columns": list(routed.columns),
            "head": routed.head(min(16, len(routed))).to_dict(orient="records"),
        }
    )
    summary = mark_schema_preview(
        {
        "schema_version": STEP4_PREFLIGHT_SCHEMA_VERSION,
        "status": "ok",
        "task": int(cfg.task_id),
        "validation_namespace": namespace,
        "output_dir": str(out_dir),
        "max_samples": max_samples,
        "sample_count": int(len(routed)),
        "sample_rows_hash": sample_hash,
        "uses_real_task_data": True,
        "uses_real_run2_checkpoint": str(cfg.from_run) == "2",
        "upstream_step3_run_id": str(cfg.from_run),
        "checkpoint": checkpoint_lineage,
        "checkpoint_lineage_validation": lineage_validation,
        "rcr_distribution": dist_payload,
        "confidence_bucket_distribution": dist_payload["confidence_bucket_distribution"],
        "sample_weight_hint_stats": dist_payload["sample_weight_hint"],
        "required_fields_check": required_check,
        "formal_latest_write": False,
        "formal_export_write": False,
        "preflight_mode": mode,
        "candidate": candidate_meta,
        "gpu_runtime_evidence": False,
        "actual_gpu_forward_executed": False,
        "actual_model_loaded_on_gpu": False,
        "force_gpu_forward": False,
        "not_step4_runtime_evidence": True,
        **schema_evidence,
        }
    )
    artifacts = {
        "rcr_distribution.json": dist_payload,
        "required_fields_check.json": required_check,
        "manifest_preview.json": manifest_preview,
        "index_contract_preview.json": index_preview,
        "lineage_preview.json": lineage,
        "cpu_gpu_utilization_snapshot.json": gpu_payload,
        "preflight_summary.json": summary,
    }
    for name, payload in artifacts.items():
        atomic_write_json(out_dir / name, payload)
    after_latest_hash = file_fingerprint(str(formal_latest)).get("sha256") if formal_latest.exists() else None
    if before_latest_hash != after_latest_hash:
        raise Step4RuntimeError("bounded preflight changed formal Step4 latest.json")
    return summary


def _upstream_artifact_hash(cfg: Any, key: str) -> str:
    try:
        payload = json.loads(str(getattr(cfg, "upstream_resolution_json", "") or "{}"))
    except json.JSONDecodeError:
        payload = {}
    path = None
    if isinstance(payload, Mapping):
        validation = payload.get("stage_status_validation")
        if isinstance(validation, Mapping):
            path = validation.get(key)
    if not path:
        return ""
    p = Path(str(path))
    if not p.is_absolute():
        p = Path(cfg.repo_root).resolve() / p
    fp = file_fingerprint(str(p))
    return str(fp.get("sha256") or "")


__all__ = [
    "STEP4_CACHE_PREPARE_SCHEMA_VERSION",
    "STEP4_PREFLIGHT_SCHEMA_VERSION",
    "STEP4_RUNTIME_ENV_KNOBS",
    "Step4RuntimeError",
    "prepare_step4_encoded_cache",
    "reject_step4_formal_env_overrides",
    "run_step4_bounded_preflight",
    "step4_runtime_env",
    "step4_formal_dry_run_meta_dir",
    "validate_step4_prelaunch_lineage_for_config",
    "validate_step4_formal_runtime_contract_replay",
    "write_step4_dry_run_resolved_artifacts",
]
