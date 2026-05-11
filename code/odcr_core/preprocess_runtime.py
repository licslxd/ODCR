from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from odcr_core.preprocess_registry import instantiate_preprocess_preset
from odcr_core.preprocess_metadata import (
    csv_header_metadata as _contract_csv_header_metadata,
    dataset_current_headers,
    parse_dataset_shell_log,
    preprocess_a_reproducibility_evidence,
    refresh_preprocess_a_stage_payload,
    task_current_headers,
    unit_current_headers,
    validate_header_collection,
)
from odcr_core.preprocess_schema import (
    COMBINE_TASK_MAP,
    PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
    PreprocessAConfig,
    PreprocessBConfig,
    PreprocessCConfig,
    PreprocessConfig,
    PreprocessHardwareConfig,
    apply_preprocess_cli_overrides,
    preprocess_b_expected_shape_dtype,
    preprocess_b_output_artifact_contract,
    preprocess_c_expected_shape_dtype,
    preprocess_c_output_artifact_contract,
    render_preprocess_stage_contract,
    resolve_combine_task_ids,
    validate_preprocess_config,
)
from odcr_core.preprocess_status import (
    PreprocessStageManifest,
    PreprocessStageStatus,
    PreprocessUnitStatus,
    PreprocessWorkerResult,
    ResolvedPreprocessStagePaths,
    ensure_preprocess_stage_dirs,
    preprocess_unit_status_path,
    read_preprocess_unit_status,
    resolve_preprocess_stage_paths,
    write_preprocess_completed_stamp,
    write_preprocess_stage_manifest,
    write_preprocess_stage_status,
    write_preprocess_unit_status,
)
from odcr_core.training_checkpoint import file_fingerprint, model_artifact_fingerprint, stable_hash
from odcr_core.manifests import build_run_summary, write_resolved_config_artifacts, write_run_summary_json
from data_contract import (
    CONTENT_PROFILE_TEXT_COLUMNS,
    DOMAIN_CONTENT_TEXT_COLUMNS,
    DOMAIN_STYLE_TEXT_COLUMNS,
    PREPROCESS_CONTRACT_VERSION,
    STYLE_PROFILE_TEXT_COLUMNS,
    assert_no_deprecated_preprocess_detail_columns,
    expected_preprocess_column_order,
    render_preprocess_contract_snapshot,
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _render_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def _repo_relative_path(repo_root: Path, value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_run_git(repo_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ("git", *args),
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _fingerprint_existing(path: Path, *, sample_only: bool = False) -> dict[str, Any]:
    return file_fingerprint(path, sample_only=sample_only)


def _csv_header_metadata(path: Path, *, contract_kind: str) -> dict[str, Any]:
    return _contract_csv_header_metadata(path, contract_kind=contract_kind)


def _preprocess_a_chunk_size() -> int:
    from preprocess_data import CANONICAL_PREPROCESS_CHUNK_SIZE

    return int(CANONICAL_PREPROCESS_CHUNK_SIZE)


_LEGACY_ODCR_ROOT_ENV = {
    "ODCR_DATA_DIR",
    "ODCR_MERGED_DATA_DIR",
    "ODCR_MODELS_DIR",
    "ODCR_STEP5_TEXT_MODEL",
    "ODCR_SENTENCE_EMBED_MODEL",
    "ODCR_EMBED_DIM",
}


def _config_stage_label(config: PreprocessConfig) -> str:
    if config.stage == "preprocess_a":
        return "preprocess_cpu"
    if config.stage == "preprocess_b":
        return "compute_embeddings"
    return "infer_domain_semantics"


def resolve_preprocess_cli_config(args: argparse.Namespace) -> PreprocessConfig:
    preset_config = instantiate_preprocess_preset(str(args.preset))
    if getattr(args, "stage", None) and str(args.stage) != preset_config.stage:
        raise ValueError(
            f"Preset {args.preset!r} resolves to stage {preset_config.stage!r}, "
            f"but CLI requested stage {args.stage!r}."
        )
    return apply_preprocess_cli_overrides(preset_config, args)


class PreprocessRuntime:
    def __init__(self, config: PreprocessConfig) -> None:
        self.config = validate_preprocess_config(config)
        self.repo_root = Path(__file__).resolve().parents[2]
        self._assert_resolved_payload()
        self.data_root = Path(self.config.resolved.data_dir).resolve()
        self.merged_root = Path(self.config.resolved.merged_dir).resolve()
        self.cache_root = Path(self.config.resolved.cache_dir).resolve()
        self.models_root = Path(self.config.resolved.models_dir).resolve()
        self.sentence_embed_model_path = Path(
            self.config.resolved.sentence_embed_model_path
            or self.config.resolved.sentence_embed_model
        ).resolve()
        self.step5_text_model_path = Path(self.config.resolved.step5_text_model).resolve()
        self.embed_dim = int(self.config.resolved.embed_dim)
        self.started_at = _utc_now()
        self.timestamp_tag = _timestamp_tag()
        self.stage_label = _config_stage_label(self.config)
        self.combine_task_ids: tuple[int, ...] = (
            resolve_combine_task_ids(self.config.datasets) if self.config.stage == "preprocess_a" else ()
        )
        self.paths = resolve_preprocess_stage_paths(
            repo_root=str(self.repo_root),
            meta_root=self.config.paths.meta_root,
            shell_log_dir=self.config.paths.shell_log_dir,
            stage=self.config.stage,
            timestamp_tag=self.timestamp_tag,
        )
        self.run_root = Path(self.paths.meta_root).resolve().parent
        self._log_lock = threading.Lock()
        self._worker_results: list[PreprocessWorkerResult] = []
        self.stage_fingerprint = self._build_stage_fingerprint()
        self.stage_fingerprint_hash = stable_hash(self.stage_fingerprint)
        self.stage_metadata = self._build_stage_metadata()

    def _assert_resolved_payload(self) -> None:
        resolved = self.config.resolved
        required = {
            "data_dir": resolved.data_dir,
            "merged_dir": resolved.merged_dir,
            "runs_dir": resolved.runs_dir,
            "cache_dir": resolved.cache_dir,
            "models_dir": resolved.models_dir,
            "step5_text_model": resolved.step5_text_model,
            "sentence_embed_model": resolved.sentence_embed_model,
            "sentence_embed_model_path": resolved.sentence_embed_model_path,
            "embed_dim": resolved.embed_dim,
        }
        missing = [key for key, value in required.items() if not str(value).strip() or str(value).strip() == "0"]
        if missing:
            raise ValueError(
                "PreprocessRuntime requires a resolved One-Control preprocess payload; "
                f"missing={missing}. Launch through ./odcr or python code/odcr.py."
            )
        if int(resolved.embed_dim) <= 0:
            raise ValueError("PreprocessRuntime resolved embed_dim must be a positive integer.")
        if not bool(resolved.local_files_only):
            raise ValueError("preprocess requires resolved env.local_files_only=true for offline model loading.")
        if self.config.stage in ("preprocess_b", "preprocess_c"):
            assert isinstance(self.config, (PreprocessBConfig, PreprocessCConfig))
            if tuple(resolved.gpu_ids) != tuple(self.config.hardware.gpu_ids):
                raise ValueError(
                    f"{self.config.stage} resolved gpu_ids={resolved.gpu_ids} "
                    f"must match hardware.gpu_ids={self.config.hardware.gpu_ids}"
                )
            if bool(resolved.bf16) != bool(self.config.bf16_enabled):
                raise ValueError(f"{self.config.stage} resolved bf16 does not match stage config.")
            if bool(resolved.tf32) != bool(self.config.tf32_enabled):
                raise ValueError(f"{self.config.stage} resolved tf32 does not match stage config.")

    def run(self) -> None:
        if not self.config.runtime.dry_run:
            ensure_preprocess_stage_dirs(self.paths)
            Path(self.paths.completed_stamp_path).unlink(missing_ok=True)

        self._write_config_artifacts()
        self._write_run_summary(status="running", finished_at=None, error_message=None)
        self._write_stage_manifest()
        self._write_stage_status(status="running", finished_at=None, error_message=None)

        try:
            if self.config.stage == "preprocess_a":
                exit_code = self._run_preprocess_a()
            else:
                exit_code = self._run_preprocess_gpu()
        except Exception as exc:
            finished_at = _utc_now()
            if not self.config.runtime.dry_run:
                Path(self.paths.completed_stamp_path).unlink(missing_ok=True)
            self._write_stage_status(
                status="failed",
                finished_at=finished_at,
                error_message=f"{self.config.stage} raised: {exc}",
            )
            self._write_run_summary(
                status="failed",
                finished_at=finished_at,
                error_message=f"{self.config.stage} raised: {exc}",
            )
            raise

        finished_at = _utc_now()
        if exit_code == 0:
            if self.config.stage == "preprocess_a":
                self._refresh_preprocess_a_metadata_after_success()
                self._write_config_artifacts()
                self._write_stage_manifest()
            if not self.config.runtime.dry_run:
                write_preprocess_completed_stamp(
                    self.paths.completed_stamp_path,
                    stage=self.config.stage,
                    preset=self.config.preset_name,
                    started_at=self.started_at,
                    finished_at=finished_at,
                )
            self._write_stage_status(status="ok", finished_at=finished_at, error_message=None)
            self._write_run_summary(status="ok", finished_at=finished_at, error_message=None)
            self._log(f"Completed preprocess stage {self.config.stage}")
            return

        if not self.config.runtime.dry_run:
            Path(self.paths.completed_stamp_path).unlink(missing_ok=True)
        self._write_stage_status(
            status="failed",
            finished_at=finished_at,
            error_message=f"{self.config.stage} failed with exit code {exit_code}",
        )
        self._write_run_summary(
            status="failed",
            finished_at=finished_at,
            error_message=f"{self.config.stage} failed with exit code {exit_code}",
        )
        raise SystemExit(exit_code)

    def _refresh_preprocess_a_metadata_after_success(self) -> None:
        refreshed, issues = refresh_preprocess_a_stage_payload(
            {
                "run_id": self.config.run_id,
                "fingerprint_hash": self.stage_fingerprint_hash,
                "metadata": self.stage_metadata,
            },
            repo_root=self.repo_root,
            meta_root=self.paths.meta_root,
        )
        if issues:
            raise RuntimeError(
                "preprocess_a completed but declared output metadata failed current-header validation: "
                + "; ".join(issues)
            )
        self.stage_metadata = dict(refreshed["metadata"])

    def _log(self, message: str) -> None:
        with self._log_lock:
            print(message, flush=True)
            if self.config.runtime.dry_run:
                return
            seen: set[Path] = set()
            for raw_path in (self.paths.full_log_path, self.paths.console_log_path):
                log_path = Path(raw_path)
                if log_path in seen:
                    continue
                seen.add(log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(message)
                    if not message.endswith("\n"):
                        f.write("\n")

    def _unit_status_path(self, unit_kind: str, unit_name: str) -> str:
        return preprocess_unit_status_path(self.paths, unit_kind=unit_kind, unit_name=unit_name)

    def _dataset_log_path(self, dataset: str, suffix: str) -> str:
        return str(
            (
                Path(self.paths.shell_log_dir)
                / f"{self.config.stage}__{dataset}__{suffix}.log"
            ).resolve()
        )

    def _task_log_path(self, task_id: int) -> str:
        return str(
            (
                Path(self.paths.shell_log_dir)
                / f"{self.config.stage}__task_{task_id}.log"
            ).resolve()
        )

    def _dataset_output_paths(self, dataset: str) -> tuple[str, ...]:
        if self.config.stage == "preprocess_a":
            return (
                str((self.data_root / dataset / "processed.csv").resolve()),
                str((self.data_root / dataset / "train.csv").resolve()),
                str((self.data_root / dataset / "valid.csv").resolve()),
                str((self.data_root / dataset / "test.csv").resolve()),
            )
        if self.config.stage == "preprocess_b":
            return (
                str((self.data_root / dataset / "user_content_profiles.npy").resolve()),
                str((self.data_root / dataset / "user_style_profiles.npy").resolve()),
                str((self.data_root / dataset / "item_content_profiles.npy").resolve()),
                str((self.data_root / dataset / "item_style_profiles.npy").resolve()),
            )
        return (
            str((self.data_root / dataset / "domain_content.npy").resolve()),
            str((self.data_root / dataset / "domain_style.npy").resolve()),
        )

    def _task_output_paths(self, task_id: int) -> tuple[str, ...]:
        return (
            str((self.merged_root / str(task_id) / "aug_train.csv").resolve()),
            str((self.merged_root / str(task_id) / "aug_valid.csv").resolve()),
        )

    def _config_snapshot_hash(self) -> str:
        return stable_hash(self.config.to_dict())

    def _canonical_column_hash(self) -> str:
        return stable_hash(
            {
                "processed": list(expected_preprocess_column_order()),
                "split": list(expected_preprocess_column_order(require_split_indices=True)),
                "merged": list(
                    expected_preprocess_column_order(require_split_indices=True, require_domain=True)
                ),
                "contract": render_preprocess_contract_snapshot(),
            }
        )

    def _schema_code_fingerprint(self) -> dict[str, Any]:
        rels = [
            "code/data_contract.py",
            "code/odcr_core/preprocess_schema.py",
            "code/odcr_core/preprocess_runtime.py",
            "code/odcr_core/preprocess_status.py",
            "configs/odcr.yaml",
            *render_preprocess_stage_contract(self.config.stage).get("producer_scripts", ()),
        ]
        out: dict[str, Any] = {}
        for rel in sorted(set(str(item) for item in rels)):
            path = self.repo_root / rel
            out[rel] = file_fingerprint(path)
        return out

    def _build_stage_fingerprint(self) -> dict[str, Any]:
        return {
            "gate_schema_version": "odcr_preprocess_skip_completed_gate/4A",
            "stage": self.config.stage,
            "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
            "canonical_column_hash": self._canonical_column_hash(),
            "source_datasets": list(self.config.datasets),
            "combine_task_ids": list(self.combine_task_ids),
            "path_roots": {
                "repo_root": str(self.repo_root),
                "data_root": str(self.data_root),
                "merged_root": str(self.merged_root),
                "runs_root": str(Path(self.config.resolved.runs_dir).resolve()),
                "cache_root": str(self.cache_root),
                "models_root": str(self.models_root),
                "meta_root": self.paths.meta_root,
            },
            "input_output_roots": {
                "inputs": {
                    "data_root": str(self.data_root),
                    "merged_root": str(self.merged_root),
                    "model_root": str(self.models_root),
                },
                "outputs": {
                    "data_root": str(self.data_root),
                    "merged_root": str(self.merged_root),
                    "cache_root": str(self.cache_root),
                    "meta_root": self.paths.meta_root,
                },
            },
            "model_paths": {
                "sentence_embed_model": str(self.sentence_embed_model_path),
                "step5_text_model": str(self.step5_text_model_path),
            },
            "model_fingerprints": {
                "sentence_embed_model": model_artifact_fingerprint(self.sentence_embed_model_path),
                "step5_text_model": model_artifact_fingerprint(self.step5_text_model_path),
            },
            "embed_dim": int(self.embed_dim),
            "offline": bool(self.config.resolved.offline),
            "local_files_only": bool(self.config.resolved.local_files_only),
            "config_snapshot_hash": self._config_snapshot_hash(),
            "code_schema_fingerprint": self._schema_code_fingerprint(),
        }

    def _task_source_domains(self, task_id: int) -> tuple[str, str] | None:
        for cur_task_id, source, target in COMBINE_TASK_MAP:
            if int(cur_task_id) == int(task_id):
                return source, target
        return None

    def _unit_input_sources(self, unit_kind: str, unit_name: str) -> dict[str, Any]:
        if unit_kind == "dataset":
            dataset = str(unit_name)
            if self.config.stage == "preprocess_a":
                paths = {"reviews_pickle": self.data_root / dataset / "reviews.pickle"}
            else:
                paths = {"train_csv": self.data_root / dataset / "train.csv"}
            return {key: file_fingerprint(path) for key, path in paths.items()}
        task_id = int(unit_name)
        pair = self._task_source_domains(task_id)
        if pair is None:
            return {}
        source, target = pair
        return {
            "source_domain": source,
            "target_domain": target,
            "source_train_csv": file_fingerprint(self.data_root / source / "train.csv"),
            "source_valid_csv": file_fingerprint(self.data_root / source / "valid.csv"),
            "target_train_csv": file_fingerprint(self.data_root / target / "train.csv"),
            "target_valid_csv": file_fingerprint(self.data_root / target / "valid.csv"),
        }

    def _unit_fingerprint(self, unit_kind: str, unit_name: str, outputs: tuple[str, ...]) -> dict[str, Any]:
        payload = {
            **self.stage_fingerprint,
            "unit": {
                "kind": str(unit_kind),
                "name": str(unit_name),
                "input_sources": self._unit_input_sources(unit_kind, unit_name),
                "output_files": [str(Path(path).resolve()) for path in outputs],
            },
        }
        payload["unit_fingerprint_hash"] = stable_hash(payload)
        return payload

    def _git_metadata(self) -> dict[str, Any]:
        status = _safe_run_git(self.repo_root, "status", "--short") or ""
        lines = [line for line in status.splitlines() if line.strip()]
        return {
            "branch": _safe_run_git(self.repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "",
            "commit": _safe_run_git(self.repo_root, "rev-parse", "HEAD") or "",
            "dirty": bool(lines),
            "dirty_status_count": len(lines),
            "dirty_status_sample": lines[:50],
        }

    def _common_metadata(self) -> dict[str, Any]:
        return {
            "metadata_schema_version": "odcr_preprocess_metadata/1.0",
            "run_id": self.config.run_id,
            "stage": self.config.stage,
            "stage_unit": self.config.stage[-1],
            "datasets": list(self.config.datasets),
            "combine_task_ids": list(self.combine_task_ids),
            "repo_root": str(self.repo_root),
            "git": self._git_metadata(),
            "config_snapshot_hash": self._config_snapshot_hash(),
            "resolved_config_path": str((Path(self.paths.meta_root) / "resolved_config.json").resolve()),
            "source_table_path": str((Path(self.paths.meta_root) / "source_table.json").resolve()),
            "run_summary_path": str((Path(self.paths.meta_root) / "run_summary.json").resolve()),
            "stage_manifest_path": self.paths.stage_manifest_path,
            "stage_status_path": self.paths.stage_status_path,
            "latest_path": str((self.run_root.parent / "latest.json").resolve()),
            "contract_version": PREPROCESS_CONTRACT_VERSION,
            "canonical_column_hash": self._canonical_column_hash(),
            "code_schema_fingerprint": self._schema_code_fingerprint(),
            "data_dir": str(self.data_root),
            "merged_dir": str(self.merged_root),
            "cache_dir": str(self.cache_root),
            "runs_dir": str(Path(self.config.resolved.runs_dir).resolve()),
            "models_dir": str(self.models_root),
            "offline": bool(self.config.resolved.offline),
            "local_files_only": bool(self.config.resolved.local_files_only),
            "resolved_payload": {
                "data_dir": self.config.resolved.data_dir,
                "merged_dir": self.config.resolved.merged_dir,
                "runs_dir": self.config.resolved.runs_dir,
                "cache_dir": self.config.resolved.cache_dir,
                "models_dir": self.config.resolved.models_dir,
                "sentence_embed_model": self.config.resolved.sentence_embed_model,
                "sentence_embed_model_path": self.config.resolved.sentence_embed_model_path,
                "step5_text_model": self.config.resolved.step5_text_model,
                "embed_dim": int(self.config.resolved.embed_dim),
                "gpu_ids": list(self.config.resolved.gpu_ids),
                "bf16": bool(self.config.resolved.bf16),
                "tf32": bool(self.config.resolved.tf32),
            },
            "metadata_responsibilities": {
                "resolved_config.json": "resolved One-Control preprocess payload snapshot; not a second YAML read",
                "source_table.json": "per-key source/value table for active roots, params, cache, lineage, and outputs",
                "run_summary.json": "formal run handoff and latest.json target",
                "stage_manifest.json": "stage-level static contract, paths, fingerprints, and reproducibility metadata",
                "stage_status.json": "live/final status plus unit status aggregation and mismatch policy",
                "latest.json": "parent pointer to meta/run_summary.json only",
            },
        }

    def _expected_csv_contract_metadata(self) -> dict[str, Any]:
        processed = list(expected_preprocess_column_order())
        split = list(expected_preprocess_column_order(require_split_indices=True))
        merged = list(expected_preprocess_column_order(require_split_indices=True, require_domain=True))
        return {
            "processed_columns": processed,
            "split_columns": split,
            "merged_columns": merged,
            "processed_header_hash": stable_hash(processed),
            "split_header_hash": stable_hash(split),
            "merged_header_hash": stable_hash(merged),
            "deprecated_field_reject_policy": (
                "retired detail columns and Step4 posterior route_scorer/route_explainer "
                "fail fast in preprocess processed/split/merged CSVs"
            ),
            "review_non_empty_validation_policy": "canonical text fields with allow_empty=false are validated as non-empty",
        }

    def _preprocess_a_metadata(self) -> dict[str, Any]:
        dataset_outputs = {
            dataset: {
                "raw_input": _fingerprint_existing(self.data_root / dataset / "reviews.pickle", sample_only=True),
                "processed_csv": str((self.data_root / dataset / "processed.csv").resolve()),
                "train_csv": str((self.data_root / dataset / "train.csv").resolve()),
                "valid_csv": str((self.data_root / dataset / "valid.csv").resolve()),
                "test_csv": str((self.data_root / dataset / "test.csv").resolve()),
                "current_headers": dataset_current_headers(self.data_root, dataset),
            }
            for dataset in self.config.datasets
        }
        task_outputs = {
            str(task_id): {
                "source_target": self._task_source_domains(task_id),
                "aug_train_csv": str((self.merged_root / str(task_id) / "aug_train.csv").resolve()),
                "aug_valid_csv": str((self.merged_root / str(task_id) / "aug_valid.csv").resolve()),
                "current_headers": task_current_headers(self.merged_root, task_id),
            }
            for task_id in self.combine_task_ids
        }
        return {
            "workers": int(self.config.runtime.workers or 1),
            "resume": bool(self.config.runtime.resume),
            "skip_completed": bool(self.config.runtime.skip_completed),
            "canonical_asset_chunk_size": _preprocess_a_chunk_size(),
            "canonical_asset_chunk_size_source": "code/preprocess_data.py::CANONICAL_PREPROCESS_CHUNK_SIZE",
            "csv_contract": self._expected_csv_contract_metadata(),
            "dataset_inputs_outputs": dataset_outputs,
            "merged_task_outputs": task_outputs,
        }

    def _preprocess_b_metadata(self) -> dict[str, Any]:
        assert isinstance(self.config, PreprocessBConfig)
        source_csvs = {
            dataset: _fingerprint_existing(self.data_root / dataset / "train.csv", sample_only=True)
            for dataset in self.config.datasets
        }
        output_paths = {
            dataset: {
                "user_content_profiles": str((self.data_root / dataset / "user_content_profiles.npy").resolve()),
                "user_style_profiles": str((self.data_root / dataset / "user_style_profiles.npy").resolve()),
                "item_content_profiles": str((self.data_root / dataset / "item_content_profiles.npy").resolve()),
                "item_style_profiles": str((self.data_root / dataset / "item_style_profiles.npy").resolve()),
            }
            for dataset in self.config.datasets
        }
        return {
            "sentence_embed_model_path": str(self.sentence_embed_model_path),
            "sentence_embed_model_fingerprint": model_artifact_fingerprint(self.sentence_embed_model_path),
            "embed_dim": int(self.embed_dim),
            "gpu_ids": list(self.config.hardware.gpu_ids),
            "workers": int(self.config.runtime.workers or len(self.config.hardware.gpu_ids)),
            "embed_batch_size": int(self.config.embed_batch_size),
            "read_chunk_rows": int(self.config.read_chunk_rows),
            "group_shard_size": int(self.config.group_shard_size),
            "tokenizer_parallelism_enabled": bool(self.config.tokenizer_parallelism_enabled),
            "tokenizer_threads_per_worker": int(self.config.tokenizer_threads_per_worker),
            "tokenizer_total_threads": int(self.config.tokenizer_total_threads),
            "prefetch_batches": int(self.config.prefetch_batches),
            "pin_memory": bool(self.config.pin_memory),
            "non_blocking_h2d": bool(self.config.non_blocking_h2d),
            "async_prefetch_enabled": bool(self.config.async_prefetch_enabled),
            "token_aware_batching_enabled": bool(self.config.token_aware_batching_enabled),
            "max_tokens_per_gpu_batch": self.config.max_tokens_per_gpu_batch,
            "cpu_cores_reserved": int(self.config.cpu_cores_reserved),
            "cpu_cores_available": int(self.config.cpu_cores_available),
            "cpu_cores_configured": int(self.config.tokenizer_total_threads),
            "grouped_cache_enabled": bool(self.config.grouped_text_cache_enabled),
            "grouped_cache_dir": str(self.config.grouped_text_cache_dir),
            "grouped_cache_version": str(self.config.grouped_text_cache_version),
            "bf16": bool(self.config.bf16_enabled),
            "tf32": bool(self.config.tf32_enabled),
            "local_files_only": bool(self.config.resolved.local_files_only),
            "profile_output_paths": output_paths,
            "expected_shape_dtype": preprocess_b_expected_shape_dtype(),
            "profile_output_artifact_contract": preprocess_b_output_artifact_contract(),
            "source_csv_fingerprints": source_csvs,
            "selected_text_columns": {
                "content": list(CONTENT_PROFILE_TEXT_COLUMNS),
                "style": list(STYLE_PROFILE_TEXT_COLUMNS),
            },
            "cache_key_fields": [
                "cache_version",
                "code_semantics_version",
                "format_version",
                "preprocess_contract_version",
                "canonical_column_hash",
                "canonical_text_source_contract.selected_columns",
                "dataset",
                "source_file.size",
                "source_file.mtime_ns",
                "source_file.sha256",
                "spec",
                "sentence_embed_model.artifact_fingerprint",
                "sentence_embed_model.tokenizer_model_identity",
                "sentence_embed_model.odcr_embed_dim",
                "grouped_text_semantics",
                "sharding.group_shard_size",
                "sharding.read_chunk_rows",
            ],
            "cache_stale_policy": (
                "cache miss/rebuild when contract, selected columns, source CSV fingerprint, model fingerprint, "
                "embed_dim, read_chunk_rows, group_shard_size, cache version, or grouped semantics differ"
            ),
        }

    def _preprocess_c_metadata(self) -> dict[str, Any]:
        assert isinstance(self.config, PreprocessCConfig)
        source_csvs = {
            dataset: _fingerprint_existing(self.data_root / dataset / "train.csv", sample_only=True)
            for dataset in self.config.datasets
        }
        output_paths = {
            dataset: {
                "domain_content": str((self.data_root / dataset / "domain_content.npy").resolve()),
                "domain_style": str((self.data_root / dataset / "domain_style.npy").resolve()),
            }
            for dataset in self.config.datasets
        }
        return {
            "sentence_embed_model_path": str(self.sentence_embed_model_path),
            "sentence_embed_model_fingerprint": model_artifact_fingerprint(self.sentence_embed_model_path),
            "embed_dim": int(self.embed_dim),
            "gpu_ids": list(self.config.hardware.gpu_ids),
            "workers": int(self.config.runtime.workers or len(self.config.hardware.gpu_ids)),
            "chunk_batch_size": int(self.config.chunk_batch_size),
            "tokenizer_parallelism_enabled": bool(self.config.tokenizer_parallelism_enabled),
            "tokenizer_threads_per_worker": int(self.config.tokenizer_threads_per_worker),
            "tokenizer_total_threads": int(self.config.tokenizer_total_threads),
            "prefetch_batches": int(self.config.prefetch_batches),
            "pin_memory": bool(self.config.pin_memory),
            "non_blocking_h2d": bool(self.config.non_blocking_h2d),
            "async_prefetch_enabled": bool(self.config.async_prefetch_enabled),
            "scheduling_policy": str(self.config.scheduling_policy),
            "cpu_cores_reserved": int(self.config.cpu_cores_reserved),
            "cpu_cores_available": int(self.config.cpu_cores_available),
            "cpu_cores_configured": int(self.config.tokenizer_total_threads),
            "token_window_max_total_tokens": 512,
            "token_window_payload_budget": "512 - tokenizer.num_special_tokens_to_add(pair=False)",
            "tokenizer_hotpath_enabled": bool(self.config.tokenizer_hotpath_enabled),
            "token_window_cache_enabled": bool(self.config.token_window_cache_enabled),
            "token_window_cache_dir": str(self.config.token_window_cache_dir),
            "token_window_cache_version": str(self.config.token_window_cache_version),
            "token_window_cache_shard_size": int(self.config.token_window_cache_shard_size),
            "bf16": bool(self.config.bf16_enabled),
            "tf32": bool(self.config.tf32_enabled),
            "local_files_only": bool(self.config.resolved.local_files_only),
            "domain_output_paths": output_paths,
            "expected_shape_dtype": preprocess_c_expected_shape_dtype(),
            "domain_shape_contract_version": PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
            "domain_output_artifact_contract": preprocess_c_output_artifact_contract(),
            "source_profile_fingerprints": {
                dataset: {
                    "user_content_profiles": _fingerprint_existing(self.data_root / dataset / "user_content_profiles.npy", sample_only=True),
                    "user_style_profiles": _fingerprint_existing(self.data_root / dataset / "user_style_profiles.npy", sample_only=True),
                    "item_content_profiles": _fingerprint_existing(self.data_root / dataset / "item_content_profiles.npy", sample_only=True),
                    "item_style_profiles": _fingerprint_existing(self.data_root / dataset / "item_style_profiles.npy", sample_only=True),
                }
                for dataset in self.config.datasets
            },
            "source_csv_fingerprints": source_csvs,
            "selected_text_columns": {
                "content": list(DOMAIN_CONTENT_TEXT_COLUMNS),
                "style": list(DOMAIN_STYLE_TEXT_COLUMNS),
            },
            "cache_key_fields": [
                "dataset",
                "domain",
                "source_file.size",
                "source_file.mtime_ns",
                "source_file.sample_sha256",
                "preprocess_contract_version",
                "canonical_column_hash",
                "canonical_text_source_contract.selected_columns",
                "tokenizer.identity",
                "sentence_embed_model.artifact_fingerprint",
                "sentence_embed_model.odcr_embed_dim",
                "max_total_tokens",
                "payload_budget",
                "chunking_contract_version",
                "token_window_cache_version",
                "tokenizer_hotpath_enabled",
                "prepend_space_between_cells",
                "empty_text_placeholder",
                "cache_scope",
                "probe_chunk_limit",
            ],
            "cache_stale_policy": (
                "cache miss/rebuild when contract, selected columns, tokenizer identity, model fingerprint, "
                "embed_dim, token-window parameters, cache version, source data, or hotpath setting differ"
            ),
        }

    def _build_stage_metadata(self) -> dict[str, Any]:
        metadata = self._common_metadata()
        if self.config.stage == "preprocess_a":
            stage_specific = self._preprocess_a_metadata()
        elif self.config.stage == "preprocess_b":
            stage_specific = self._preprocess_b_metadata()
        else:
            stage_specific = self._preprocess_c_metadata()
        metadata["stage_specific"] = stage_specific
        metadata["stage_specific_hash"] = stable_hash(stage_specific)
        metadata["lineage_stale_policy"] = (
            "contract/schema/data/model/config/cache fingerprint mismatch is stale and must rebuild or fail fast; "
            "old v2.x or old route_scorer/route_explainer preprocess artifacts are not silently reused"
        )
        if self.config.stage == "preprocess_a":
            metadata["reproducibility_evidence"] = preprocess_a_reproducibility_evidence(
                repo_root=self.repo_root,
                stage_payload={
                    "run_id": self.config.run_id,
                    "fingerprint_hash": self.stage_fingerprint_hash,
                    "metadata": metadata,
                },
            )
        return metadata

    def _assert_csv_header_contract(self, path: str, *, require_split_indices: bool, require_domain: bool) -> None:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise RuntimeError(f"{path} is empty; cannot reuse as {PREPROCESS_CONTRACT_VERSION}.") from exc
        assert_no_deprecated_preprocess_detail_columns(header, source_label=path)
        expected = list(
            expected_preprocess_column_order(
                require_split_indices=require_split_indices,
                require_domain=require_domain,
            )
        )
        if list(header) != expected:
            raise RuntimeError(
                f"{path} does not match {PREPROCESS_CONTRACT_VERSION}; refusing skip_completed reuse. "
                f"expected={expected}, actual={list(header)}"
            )

    def _assert_unit_outputs_current_contract(self, unit_kind: str, outputs: tuple[str, ...]) -> None:
        if self.config.stage != "preprocess_a":
            return
        if unit_kind == "dataset":
            if len(outputs) >= 1:
                self._assert_csv_header_contract(outputs[0], require_split_indices=False, require_domain=False)
            for path in outputs[1:]:
                self._assert_csv_header_contract(path, require_split_indices=True, require_domain=False)
        else:
            for path in outputs:
                self._assert_csv_header_contract(path, require_split_indices=True, require_domain=True)

    def _task_forced(self, task_id: int) -> bool:
        if not self.config.runtime.force_datasets:
            return False
        forced = set(self.config.runtime.force_datasets)
        for cur_task_id, source, target in COMBINE_TASK_MAP:
            if cur_task_id == task_id:
                return source in forced or target in forced
        return False

    def _unit_already_complete(self, unit_kind: str, unit_name: str, outputs: tuple[str, ...]) -> bool:
        if not self.config.runtime.resume or not self.config.runtime.skip_completed:
            return False
        payload = read_preprocess_unit_status(self._unit_status_path(unit_kind, unit_name))
        if not payload:
            return False
        if payload.get("status") != "ok":
            return False
        current_fp = self._unit_fingerprint(unit_kind, unit_name, outputs)
        current_hash = str(current_fp["unit_fingerprint_hash"])
        stored_hash = str(payload.get("fingerprint_hash") or "")
        stored_fp = payload.get("fingerprint") if isinstance(payload.get("fingerprint"), dict) else None
        stored_contract = str((stored_fp or {}).get("preprocess_contract_version") or "")
        if not stored_hash or not stored_fp:
            raise RuntimeError(
                f"{self.config.stage} skip_completed refused for {unit_kind}={unit_name}: "
                "previous status lacks Phase 4A fingerprint. Rerun with --no-skip-completed or force the unit."
            )
        if stored_contract != PREPROCESS_CONTRACT_VERSION:
            raise RuntimeError(
                f"{self.config.stage} skip_completed refused for {unit_kind}={unit_name}: "
                f"stored preprocess contract {stored_contract!r} != {PREPROCESS_CONTRACT_VERSION!r}. Rerun preprocess."
            )
        if stored_hash != current_hash:
            raise RuntimeError(
                f"{self.config.stage} skip_completed refused for {unit_kind}={unit_name}: "
                f"fingerprint mismatch stored={stored_hash} current={current_hash}. Rerun preprocess."
            )
        if not all(Path(path).exists() for path in outputs):
            return False
        self._assert_unit_outputs_current_contract(unit_kind, outputs)
        return True

    def _preprocess_a_unit_current_headers(
        self,
        unit_kind: str,
        output_files: tuple[str, ...],
        *,
        status: str,
    ) -> dict[str, Any] | None:
        if self.config.stage != "preprocess_a":
            return None
        headers = unit_current_headers(unit_kind, output_files)
        if status == "ok":
            issues = validate_header_collection(headers)
            if issues:
                raise RuntimeError(
                    f"preprocess_a {unit_kind} status cannot be ok with invalid current_headers: "
                    + "; ".join(issues)
                )
        return headers

    def _preprocess_a_unit_metadata(
        self,
        *,
        unit_kind: str,
        unit_name: str,
        shell_log_path: str | None,
    ) -> dict[str, Any] | None:
        if self.config.stage != "preprocess_a":
            return None
        if unit_kind == "dataset":
            parsed = parse_dataset_shell_log(shell_log_path or "", unit_name)
            return {
                "split_policy": parsed.get("split_policy") or {},
                "cleaning": parsed.get("cleaning") or {},
            }
        return {
            "source_target": self._task_source_domains(int(unit_name)),
            "merge_policy": "concatenate auxiliary and target train/valid splits with domain transport labels",
        }

    def _preprocess_gpu_unit_metadata(
        self,
        *,
        unit_kind: str,
        unit_name: str,
        output_files: tuple[str, ...],
    ) -> dict[str, Any] | None:
        if unit_kind != "dataset" or self.config.stage not in {"preprocess_b", "preprocess_c"}:
            return None
        if self.config.stage == "preprocess_b":
            contract = preprocess_b_output_artifact_contract()
            expected = preprocess_b_expected_shape_dtype()
            kind = "profile_matrix"
        else:
            contract = preprocess_c_output_artifact_contract()
            expected = preprocess_c_expected_shape_dtype()
            kind = "domain_vector"
        return {
            "artifact_contract_kind": kind,
            "expected_shape_dtype": expected,
            "output_artifact_contract": {
                Path(path).name: contract.get(Path(path).name)
                for path in output_files
                if Path(path).name in contract
            },
            "unit_name": unit_name,
        }

    def _write_unit_status(
        self,
        *,
        unit_kind: str,
        unit_name: str,
        status: str,
        output_files: tuple[str, ...],
        shell_log_path: str | None,
        started_at: str | None = None,
        finished_at: str | None = None,
        reason: str | None = None,
        error_message: str | None = None,
        worker_id: int | None = None,
        gpu_id: int | None = None,
        command: list[str] | None = None,
        unit_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.config.runtime.dry_run:
            return
        fingerprint = self._unit_fingerprint(unit_kind, unit_name, output_files)
        current_headers = self._preprocess_a_unit_current_headers(
            unit_kind,
            output_files,
            status=status,
        )
        provided_unit_metadata = dict(unit_metadata or {})
        unit_metadata = self._preprocess_a_unit_metadata(
            unit_kind=unit_kind,
            unit_name=unit_name,
            shell_log_path=shell_log_path,
        )
        base_metadata = self._preprocess_gpu_unit_metadata(
            unit_kind=unit_kind,
            unit_name=unit_name,
            output_files=output_files,
        )
        merged_metadata: dict[str, Any] = {}
        if base_metadata:
            merged_metadata.update(base_metadata)
        if unit_metadata:
            merged_metadata.update(unit_metadata)
        if provided_unit_metadata:
            merged_metadata.update(provided_unit_metadata)
        unit_metadata = merged_metadata or None
        payload = PreprocessUnitStatus(
            stage=self.config.stage,
            preset=self.config.preset_name,
            unit_kind=unit_kind,  # type: ignore[arg-type]
            unit_name=unit_name,
            status=status,  # type: ignore[arg-type]
            started_at=started_at,
            finished_at=finished_at,
            shell_log_path=shell_log_path,
            output_files=output_files,
            reason=reason,
            error_message=error_message,
            worker_id=worker_id,
            gpu_id=gpu_id,
            command=tuple(command) if command else None,
            fingerprint=fingerprint,
            fingerprint_hash=str(fingerprint["unit_fingerprint_hash"]),
            current_headers=current_headers,
            metadata=unit_metadata,
        )
        write_preprocess_unit_status(self._unit_status_path(unit_kind, unit_name), payload)

    def _read_all_statuses(self, unit_kind: str) -> dict[str, dict[str, Any]]:
        root = Path(self.paths.datasets_status_dir if unit_kind == "dataset" else self.paths.tasks_status_dir)
        out: dict[str, dict[str, Any]] = {}
        if not root.is_dir():
            return out
        for path in sorted(root.glob("*.status.json")):
            payload = read_preprocess_unit_status(path)
            if payload is not None:
                out[str(payload["unit_name"])] = payload
        return out

    def _write_stage_manifest(self) -> None:
        manifest = PreprocessStageManifest(
            stage=self.config.stage,
            preset=self.config.preset_name,
            description=self.config.description,
            stage_label=self.stage_label,
            started_at=self.started_at,
            datasets=self.config.datasets,
            combine_task_ids=self.combine_task_ids,
            config_snapshot=self.config.to_dict(),
            contract_snapshot=render_preprocess_stage_contract(self.config.stage),
            fingerprint=self.stage_fingerprint,
            fingerprint_hash=self.stage_fingerprint_hash,
            paths={
                "repo_root": str(self.repo_root),
                "data_root": str(self.data_root),
                "merged_root": str(self.merged_root),
                "runs_root": str(Path(self.config.resolved.runs_dir).resolve()),
                "cache_root": str(self.cache_root),
                "models_root": str(self.models_root),
                "sentence_embed_model": str(self.sentence_embed_model_path),
                "embed_dim": str(self.embed_dim),
                "meta_root": self.paths.meta_root,
                "shell_log_dir": self.paths.shell_log_dir,
                "stage_log_path": self.paths.stage_log_path,
                "console_log_path": self.paths.console_log_path,
                "full_log_path": self.paths.full_log_path,
                "errors_log_path": self.paths.errors_log_path,
                "metrics_path": str((Path(self.paths.meta_root) / "metrics.json").resolve())
                if self.config.stage in {"preprocess_b", "preprocess_c"}
                else "",
                "verify_report_path": str((Path(self.paths.meta_root) / "verify_report.json").resolve())
                if self.config.stage in {"preprocess_b", "preprocess_c"}
                else "",
                "stage_manifest_path": self.paths.stage_manifest_path,
                "stage_status_path": self.paths.stage_status_path,
                "completed_stamp_path": self.paths.completed_stamp_path,
                "datasets_status_dir": self.paths.datasets_status_dir,
                "tasks_status_dir": self.paths.tasks_status_dir,
            },
            metadata=self.stage_metadata,
        )
        if not self.config.runtime.dry_run:
            write_preprocess_stage_manifest(self.paths.stage_manifest_path, manifest)

    def _source_table_field_sources(self) -> dict[str, str]:
        unit = self.config.stage[-1]
        sources = dict(self.config.resolved.sources)
        common = {
            "config": "configs/odcr.yaml",
            "preprocess": f"configs/odcr.yaml:preprocess.{unit}",
            "run_id": "run_naming.allocate_child_dir(runs/preprocess/<unit>)",
            "stage": "resolved preprocess command",
            "datasets": f"configs/odcr.yaml:preprocess.{unit}.datasets or preprocess.datasets",
            "repo_root": "code/odcr_core/preprocess_runtime.py",
            "git_branch": "git rev-parse --abbrev-ref HEAD",
            "git_commit": "git rev-parse HEAD",
            "git_dirty": "git status --short",
            "config_snapshot_hash": "PreprocessConfig.to_dict() stable hash",
            "resolved_config_path": "runs/preprocess/<unit>/<run_id>/meta/resolved_config.json",
            "source_table_path": "runs/preprocess/<unit>/<run_id>/meta/source_table.json",
            "contract_version": "code/data_contract.py::PREPROCESS_CONTRACT_VERSION",
            "canonical_column_hash": "code/data_contract.py expected_preprocess_column_order/render_preprocess_contract_snapshot",
            "code_schema_fingerprint": "code/odcr_core/preprocess_runtime.py::_schema_code_fingerprint",
            "lineage_stale_policy": "preprocess runtime/cache fail-fast policy",
            "data_dir": sources.get("data_dir", "project.data_dir"),
            "merged_dir": sources.get("merged_dir", "project.merged_dir"),
            "runs_dir": sources.get("runs_dir", "project.run_root"),
            "cache_dir": sources.get("cache_dir", "project.cache_dir"),
            "models_dir": sources.get("models_dir", "env.models_dir"),
            "sentence_embed_model": sources.get("sentence_embed_model", "env.sentence_embed_model"),
            "sentence_embed_model_path": sources.get("sentence_embed_model_path", "env.sentence_embed_model"),
            "step5_text_model": sources.get("step5_text_model", "env.step5_text_model"),
            "embed_dim": sources.get("embed_dim", "env.embed_dim"),
            "offline": sources.get("offline", "env.offline"),
            "local_files_only": sources.get("local_files_only", "env.local_files_only"),
        }
        if self.config.stage == "preprocess_a":
            common.update(
                {
                    "preprocess.a.workers": "configs/odcr.yaml:preprocess.a.workers",
                    "preprocess.a.resume": "configs/odcr.yaml:preprocess.a.resume",
                    "preprocess.a.skip_completed": "configs/odcr.yaml:preprocess.a.skip_completed",
                    "preprocess.a.canonical_asset_chunk_size": "code/preprocess_data.py::CANONICAL_PREPROCESS_CHUNK_SIZE",
                    "preprocess.a.raw_input_fingerprint": "project.data_dir/<dataset>/reviews.pickle",
                    "preprocess.a.output_paths": "project.data_dir/project.merged_dir resolved roots",
                    "preprocess.a.csv_header_hash": "code/data_contract.py canonical processed/split/merged headers",
                    "preprocess.a.deprecated_field_reject_policy": "code/data_contract.py fail-fast validators",
                    "preprocess.a.review_non_empty_validation_policy": "code/data_contract.py text validation",
                    "preprocess.a.split_policy_summary": "split_data.py cold user/item valid/test filter policy",
                    "preprocess.a.cleaning_summary": "preprocess_data.py empty-review and k-core cleaning logs",
                    "preprocess.a.reproducibility_evidence": "preprocess_a git/code/config/contract fingerprint evidence",
                }
            )
        elif self.config.stage == "preprocess_b":
            common.update(
                {
                    "gpu_ids": sources.get("gpu_ids", "preprocess.b.gpu_ids or hardware.preprocess.gpu_ids"),
                    "bf16": sources.get("bf16", "preprocess.b.bf16_enabled"),
                    "tf32": sources.get("tf32", "preprocess.b.tf32_enabled"),
                    "preprocess.b.workers": "configs/odcr.yaml:preprocess.b.workers or len(gpu_ids)",
                    "preprocess.b.embed_batch_size": "configs/odcr.yaml:preprocess.b.batch_size -> child --embed-batch-size",
                    "preprocess.b.read_chunk_rows": "configs/odcr.yaml:preprocess.b.read_chunk_rows -> child --read-chunk-rows",
                    "preprocess.b.group_shard_size": "configs/odcr.yaml:preprocess.b.group_shard_size -> child --group-shard-size",
                    "preprocess.b.tokenizer_parallelism_enabled": "configs/odcr.yaml:preprocess.b.tokenizer_parallelism_enabled -> child --tokenizer-parallelism/--no-tokenizer-parallelism",
                    "preprocess.b.tokenizer_threads_per_worker": "configs/odcr.yaml:preprocess.b.tokenizer_threads_per_worker -> child --tokenizer-threads-per-worker/RAYON_NUM_THREADS",
                    "preprocess.b.tokenizer_total_threads": "configs/odcr.yaml:preprocess.b.tokenizer_total_threads",
                    "preprocess.b.prefetch_batches": "configs/odcr.yaml:preprocess.b.prefetch_batches -> child --prefetch-batches",
                    "preprocess.b.pin_memory": "configs/odcr.yaml:preprocess.b.pin_memory -> child --pin-memory/--no-pin-memory",
                    "preprocess.b.non_blocking_h2d": "configs/odcr.yaml:preprocess.b.non_blocking_h2d -> child --non-blocking-h2d/--no-non-blocking-h2d",
                    "preprocess.b.async_prefetch_enabled": "configs/odcr.yaml:preprocess.b.async_prefetch_enabled -> child --async-prefetch/--no-async-prefetch",
                    "preprocess.b.token_aware_batching_enabled": "configs/odcr.yaml:preprocess.b.token_aware_batching_enabled -> child --token-aware-batching/--no-token-aware-batching",
                    "preprocess.b.max_tokens_per_gpu_batch": "configs/odcr.yaml:preprocess.b.max_tokens_per_gpu_batch",
                    "preprocess.b.cpu_cores_reserved": "configs/odcr.yaml:preprocess.b.cpu_cores_reserved",
                    "preprocess.b.cpu_cores_available": "configs/odcr.yaml:preprocess.b.cpu_cores_available",
                    "preprocess.b.cpu_cores_configured": "preprocess.b.tokenizer_total_threads",
                    "preprocess.b.grouped_cache_enabled": "configs/odcr.yaml:preprocess.b.grouped_text_cache_enabled",
                    "preprocess.b.grouped_cache_version": "configs/odcr.yaml:preprocess.b.grouped_text_cache_version",
                    "preprocess.b.profile_output_paths": "project.data_dir/<dataset>/*_profiles.npy",
                    "preprocess.b.expected_shape_dtype": "code/compute_embeddings.py output verifier",
                    "preprocess.b.profile_output_artifact_contract": "code/odcr_core/preprocess_schema.py profile matrix contract",
                    "preprocess.b.source_csv_fingerprint": "project.data_dir/<dataset>/train.csv",
                    "preprocess.b.cache_key_fields": "code/compute_embeddings.py::_build_grouped_text_cache_request",
                    "preprocess.b.cache_stale_policy": "cache miss/rebuild on contract/source/model/config mismatch",
                }
            )
        else:
            common.update(
                {
                    "gpu_ids": sources.get("gpu_ids", "preprocess.c.gpu_ids or hardware.preprocess.gpu_ids"),
                    "bf16": sources.get("bf16", "preprocess.c.bf16_enabled"),
                    "tf32": sources.get("tf32", "preprocess.c.tf32_enabled"),
                    "preprocess.c.workers": "configs/odcr.yaml:preprocess.c.workers or len(gpu_ids)",
                    "preprocess.c.chunk_batch_size": "configs/odcr.yaml:preprocess.c.chunk_batch_size -> child --chunk-batch-size",
                    "preprocess.c.tokenizer_parallelism_enabled": "configs/odcr.yaml:preprocess.c.tokenizer_parallelism_enabled -> child --tokenizer-parallelism/--no-tokenizer-parallelism",
                    "preprocess.c.tokenizer_threads_per_worker": "configs/odcr.yaml:preprocess.c.tokenizer_threads_per_worker -> child --tokenizer-threads-per-worker/RAYON_NUM_THREADS",
                    "preprocess.c.tokenizer_total_threads": "configs/odcr.yaml:preprocess.c.tokenizer_total_threads",
                    "preprocess.c.prefetch_batches": "configs/odcr.yaml:preprocess.c.prefetch_batches -> child --prefetch-batches",
                    "preprocess.c.pin_memory": "configs/odcr.yaml:preprocess.c.pin_memory -> child --pin-memory/--no-pin-memory",
                    "preprocess.c.non_blocking_h2d": "configs/odcr.yaml:preprocess.c.non_blocking_h2d -> child --non-blocking-h2d/--no-non-blocking-h2d",
                    "preprocess.c.async_prefetch_enabled": "configs/odcr.yaml:preprocess.c.async_prefetch_enabled -> child --async-prefetch/--no-async-prefetch",
                    "preprocess.c.scheduling_policy": "configs/odcr.yaml:preprocess.c.scheduling_policy",
                    "preprocess.c.cpu_cores_reserved": "configs/odcr.yaml:preprocess.c.cpu_cores_reserved",
                    "preprocess.c.cpu_cores_available": "configs/odcr.yaml:preprocess.c.cpu_cores_available",
                    "preprocess.c.cpu_cores_configured": "preprocess.c.tokenizer_total_threads",
                    "preprocess.c.token_window_max_total_tokens": "code/infer_domain_semantics.py::MAX_TOTAL_TOKENS",
                    "preprocess.c.token_window_payload_budget": "tokenizer.num_special_tokens_to_add(pair=False)",
                    "preprocess.c.tokenizer_hotpath_enabled": "configs/odcr.yaml:preprocess.c.tokenizer_hotpath_enabled",
                    "preprocess.c.token_window_cache_enabled": "configs/odcr.yaml:preprocess.c.token_window_cache_enabled",
                    "preprocess.c.token_window_cache_version": "configs/odcr.yaml:preprocess.c.token_window_cache_version",
                    "preprocess.c.token_window_cache_shard_size": "configs/odcr.yaml:preprocess.c.token_window_cache_shard_size",
                    "preprocess.c.domain_output_paths": "project.data_dir/<dataset>/domain_*.npy",
                    "preprocess.c.expected_shape_dtype": "code/infer_domain_semantics.py output verifier",
                    "preprocess.c.domain_shape_contract_version": "code/odcr_core/preprocess_schema.py domain vector contract",
                    "preprocess.c.domain_output_artifact_contract": "code/odcr_core/preprocess_schema.py domain vector contract",
                    "preprocess.c.source_profile_fingerprints": "project.data_dir/<dataset>/*_profiles.npy",
                    "preprocess.c.source_csv_fingerprint": "project.data_dir/<dataset>/train.csv",
                    "preprocess.c.cache_key_fields": "code/infer_domain_semantics.py::_token_window_cache_fingerprint",
                    "preprocess.c.cache_stale_policy": "cache miss/rebuild on contract/tokenizer/source/model/config mismatch",
                }
            )
        return common

    def _source_table_payload(self) -> dict[str, Any]:
        field_sources = self._source_table_field_sources()
        values = self.stage_metadata
        stage_specific = values.get("stage_specific", {}) if isinstance(values.get("stage_specific"), dict) else {}
        value_by_key = {
            "run_id": self.config.run_id,
            "stage": self.config.stage,
            "datasets": list(self.config.datasets),
            "repo_root": str(self.repo_root),
            "git_branch": values["git"]["branch"],
            "git_commit": values["git"]["commit"],
            "git_dirty": values["git"]["dirty"],
            "config_snapshot_hash": values["config_snapshot_hash"],
            "resolved_config_path": values["resolved_config_path"],
            "source_table_path": values["source_table_path"],
            "contract_version": PREPROCESS_CONTRACT_VERSION,
            "canonical_column_hash": values["canonical_column_hash"],
            "data_dir": str(self.data_root),
            "merged_dir": str(self.merged_root),
            "runs_dir": str(Path(self.config.resolved.runs_dir).resolve()),
            "cache_dir": str(self.cache_root),
            "models_dir": str(self.models_root),
            "sentence_embed_model": self.config.resolved.sentence_embed_model,
            "sentence_embed_model_path": str(self.sentence_embed_model_path),
            "step5_text_model": str(self.step5_text_model_path),
            "embed_dim": int(self.embed_dim),
            "offline": bool(self.config.resolved.offline),
            "local_files_only": bool(self.config.resolved.local_files_only),
            "gpu_ids": list(self.config.resolved.gpu_ids),
            "bf16": bool(self.config.resolved.bf16),
            "tf32": bool(self.config.resolved.tf32),
        }
        for key, value in stage_specific.items():
            value_by_key[f"{self.config.stage}.{key}"] = value
            value_by_key[f"preprocess.{self.config.stage[-1]}.{key}"] = value
        if self.config.stage == "preprocess_a":
            value_by_key["preprocess.a.raw_input_fingerprint"] = stage_specific.get("dataset_inputs_outputs")
            value_by_key["preprocess.a.output_paths"] = {
                "datasets": stage_specific.get("dataset_inputs_outputs"),
                "merged_tasks": stage_specific.get("merged_task_outputs"),
            }
            value_by_key["preprocess.a.csv_header_hash"] = stage_specific.get("csv_contract")
            value_by_key["preprocess.a.deprecated_field_reject_policy"] = (
                stage_specific.get("csv_contract", {}).get("deprecated_field_reject_policy")
                if isinstance(stage_specific.get("csv_contract"), dict)
                else None
            )
            value_by_key["preprocess.a.review_non_empty_validation_policy"] = (
                stage_specific.get("csv_contract", {}).get("review_non_empty_validation_policy")
                if isinstance(stage_specific.get("csv_contract"), dict)
                else None
            )
            value_by_key["preprocess.a.split_policy_summary"] = stage_specific.get("split_policy_summary")
            value_by_key["preprocess.a.cleaning_summary"] = stage_specific.get("cleaning_summary")
            value_by_key["preprocess.a.reproducibility_evidence"] = self.stage_metadata.get(
                "reproducibility_evidence"
            )
        elif self.config.stage == "preprocess_b":
            value_by_key["preprocess.b.source_csv_fingerprint"] = stage_specific.get("source_csv_fingerprints")
            value_by_key["preprocess.b.profile_output_paths"] = stage_specific.get("profile_output_paths")
            value_by_key["preprocess.b.expected_shape_dtype"] = stage_specific.get("expected_shape_dtype")
            value_by_key["preprocess.b.cache_key_fields"] = stage_specific.get("cache_key_fields")
            value_by_key["preprocess.b.cache_stale_policy"] = stage_specific.get("cache_stale_policy")
        else:
            value_by_key["preprocess.c.source_csv_fingerprint"] = stage_specific.get("source_csv_fingerprints")
            value_by_key["preprocess.c.source_profile_fingerprints"] = stage_specific.get("source_profile_fingerprints")
            value_by_key["preprocess.c.domain_output_paths"] = stage_specific.get("domain_output_paths")
            value_by_key["preprocess.c.expected_shape_dtype"] = stage_specific.get("expected_shape_dtype")
            value_by_key["preprocess.c.cache_key_fields"] = stage_specific.get("cache_key_fields")
            value_by_key["preprocess.c.cache_stale_policy"] = stage_specific.get("cache_stale_policy")
        records = [
            {
                "key": key,
                "source": source,
                "value": value_by_key.get(key, value_by_key.get(f"{self.config.stage}.{key}", None)),
            }
            for key, source in sorted(field_sources.items())
        ]
        return {
            "source_table_schema_version": "1.1",
            "generated_at_utc": _utc_now(),
            "source_policy": "One-Control resolved payload and runtime transport only; no child YAML/env fallback",
            "field_sources": field_sources,
            "records": records,
        }

    def _write_config_artifacts(self) -> None:
        if self.config.runtime.dry_run:
            return
        write_resolved_config_artifacts(
            self.paths.meta_root,
            self.config.to_dict(),
            source_table=self._source_table_payload(),
        )

    def _write_run_summary(
        self,
        *,
        status: str,
        finished_at: str | None,
        error_message: str | None,
    ) -> None:
        if self.config.runtime.dry_run:
            return
        unit = self.config.stage[-1]
        metrics_path = None
        verify_report_path = None
        key_artifacts: dict[str, Any] = {
            "stage_manifest": self.paths.stage_manifest_path,
            "stage_status": self.paths.stage_status_path,
            "completed_stamp": self.paths.completed_stamp_path,
        }
        if self.config.stage in {"preprocess_b", "preprocess_c"}:
            metrics_path = Path(self.paths.meta_root) / "metrics.json"
            verify_report_path = Path(self.paths.meta_root) / "verify_report.json"
            key_artifacts["metrics"] = metrics_path
            key_artifacts["verify_report"] = verify_report_path
        summary = build_run_summary(
            repo_root=self.repo_root,
            run_dir=self.run_root,
            meta_dir=self.paths.meta_root,
            run_id=self.config.run_id,
            stage="preprocess",
            unit=unit,
            status=status,
            started_at=self.started_at,
            finished_at=finished_at,
            command=f"./odcr preprocess {unit}",
            console_log_path=self.paths.console_log_path,
            full_log_path=self.paths.full_log_path,
            errors_log_path=self.paths.errors_log_path,
            metrics_path=metrics_path,
            lineage_path=self.paths.stage_status_path,
            manifest_path=self.paths.stage_manifest_path,
            key_artifacts=key_artifacts,
            latest_error=error_message,
            validation_status="ok" if status == "ok" else ("failed" if status == "failed" else "pending"),
        )
        if verify_report_path is not None:
            summary["verify_report_path"] = _repo_relative_path(self.repo_root, verify_report_path)
        if self.config.stage == "preprocess_c":
            summary["domain_shape_contract_version"] = PREPROCESS_C_DOMAIN_CONTRACT_VERSION
        summary["fingerprint_hash"] = self.stage_fingerprint_hash
        summary["preprocess_metadata"] = {
            "metadata_schema_version": self.stage_metadata["metadata_schema_version"],
            "stage_specific_hash": self.stage_metadata["stage_specific_hash"],
            "contract_version": PREPROCESS_CONTRACT_VERSION,
            "canonical_column_hash": self.stage_metadata["canonical_column_hash"],
            "data_dir": str(self.data_root),
            "merged_dir": str(self.merged_root),
            "cache_dir": str(self.cache_root),
            "runs_dir": str(Path(self.config.resolved.runs_dir).resolve()),
            "models_dir": str(self.models_root),
            "offline": bool(self.config.resolved.offline),
            "local_files_only": bool(self.config.resolved.local_files_only),
            "stage_manifest_path": self.paths.stage_manifest_path,
            "stage_status_path": self.paths.stage_status_path,
            "lineage_stale_policy": self.stage_metadata["lineage_stale_policy"],
        }
        if self.config.stage == "preprocess_c":
            summary["preprocess_metadata"]["domain_shape_contract_version"] = PREPROCESS_C_DOMAIN_CONTRACT_VERSION
        write_run_summary_json(summary, repo_root=self.repo_root, update_latest=True)

    def _write_stage_status(
        self,
        *,
        status: str,
        finished_at: str | None,
        error_message: str | None,
    ) -> None:
        if self.config.runtime.dry_run:
            return
        payload = PreprocessStageStatus(
            stage=self.config.stage,
            preset=self.config.preset_name,
            status=status,  # type: ignore[arg-type]
            started_at=self.started_at,
            finished_at=finished_at,
            description=self.config.description,
            stage_label=self.stage_label,
            datasets=self.config.datasets,
            combine_task_ids=self.combine_task_ids,
            dataset_statuses=self._read_all_statuses("dataset"),
            task_statuses=self._read_all_statuses("task"),
            worker_results=tuple(item.to_dict() for item in self._worker_results),
            config_snapshot=self.config.to_dict(),
            contract_snapshot=render_preprocess_stage_contract(self.config.stage),
            fingerprint=self.stage_fingerprint,
            fingerprint_hash=self.stage_fingerprint_hash,
            paths={
                "repo_root": str(self.repo_root),
                "data_root": str(self.data_root),
                "merged_root": str(self.merged_root),
                "runs_root": str(Path(self.config.resolved.runs_dir).resolve()),
                "cache_root": str(self.cache_root),
                "models_root": str(self.models_root),
                "sentence_embed_model": str(self.sentence_embed_model_path),
                "embed_dim": str(self.embed_dim),
                "meta_root": self.paths.meta_root,
                "shell_log_dir": self.paths.shell_log_dir,
                "stage_log_path": self.paths.stage_log_path,
                "console_log_path": self.paths.console_log_path,
                "full_log_path": self.paths.full_log_path,
                "errors_log_path": self.paths.errors_log_path,
                "metrics_path": str((Path(self.paths.meta_root) / "metrics.json").resolve())
                if self.config.stage in {"preprocess_b", "preprocess_c"}
                else "",
                "verify_report_path": str((Path(self.paths.meta_root) / "verify_report.json").resolve())
                if self.config.stage in {"preprocess_b", "preprocess_c"}
                else "",
                "stage_manifest_path": self.paths.stage_manifest_path,
                "stage_status_path": self.paths.stage_status_path,
                "completed_stamp_path": self.paths.completed_stamp_path,
                "datasets_status_dir": self.paths.datasets_status_dir,
                "tasks_status_dir": self.paths.tasks_status_dir,
            },
            metadata=self.stage_metadata,
            error_message=error_message,
        )
        write_preprocess_stage_status(self.paths.stage_status_path, payload)

    def _assert_outputs_refreshed(self, outputs: tuple[str, ...], start_time: float) -> None:
        for output in outputs:
            path = Path(output)
            if not path.exists():
                raise FileNotFoundError(f"Missing expected output: {output}")
            if path.stat().st_mtime < start_time:
                raise RuntimeError(f"Output was not refreshed during this subprocess: {output}")

    def _run_logged_subprocess(
        self,
        *,
        label: str,
        command: list[str],
        unit_log_path: str,
        outputs: tuple[str, ...],
        expect_refresh: bool,
    ) -> int:
        rendered = _render_command(command)
        self._log(f"[{label}] {rendered}")
        if self.config.runtime.dry_run:
            if outputs:
                for output in outputs:
                    self._log(f"[{label}] expected output: {output}")
            return 0

        start_time = time.time() - 1e-6
        Path(unit_log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(unit_log_path).open("a", encoding="utf-8") as log_file:
            log_file.write(f"[command] {rendered}\n")
            log_file.flush()
            proc = subprocess.run(
                command,
                cwd=str(self.repo_root),
                env=self._child_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode == 0 and expect_refresh:
            self._assert_outputs_refreshed(outputs, start_time)
        return int(proc.returncode)

    def _child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in list(env):
            if (key in _LEGACY_ODCR_ROOT_ENV) or (key.startswith("ODCR_") and not key.startswith("ODCR_RESOLVED_")):
                env.pop(key, None)
        resolved = self.config.resolved
        env.update(
            {
                "ODCR_RESOLVED_DATA_DIR": str(self.data_root),
                "ODCR_RESOLVED_MERGED_DIR": str(self.merged_root),
                "ODCR_RESOLVED_RUNS_DIR": str(Path(resolved.runs_dir).resolve()),
                "ODCR_RESOLVED_CACHE_DIR": str(self.cache_root),
                "ODCR_RESOLVED_MODELS_DIR": str(self.models_root),
                "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(self.step5_text_model_path),
                "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(self.sentence_embed_model_path),
                "ODCR_RESOLVED_EMBED_DIM": str(self.embed_dim),
                "ODCR_RESOLVED_OFFLINE": "1" if resolved.offline else "0",
                "ODCR_RESOLVED_LOCAL_FILES_ONLY": "1" if resolved.local_files_only else "0",
            }
        )
        if isinstance(self.config, (PreprocessBConfig, PreprocessCConfig)):
            env.update(
                {
                    "ODCR_RESOLVED_TOKENIZER_PARALLELISM": (
                        "1" if self.config.tokenizer_parallelism_enabled else "0"
                    ),
                    "ODCR_RESOLVED_TOKENIZER_THREADS_PER_WORKER": str(
                        int(self.config.tokenizer_threads_per_worker)
                    ),
                    "ODCR_RESOLVED_TOKENIZER_TOTAL_THREADS": str(int(self.config.tokenizer_total_threads)),
                    "ODCR_RESOLVED_PREFETCH_BATCHES": str(int(self.config.prefetch_batches)),
                    "ODCR_RESOLVED_CPU_CORES_RESERVED": str(int(self.config.cpu_cores_reserved)),
                    "ODCR_RESOLVED_CPU_CORES_AVAILABLE": str(int(self.config.cpu_cores_available)),
                }
            )
        return env

    def _run_preprocess_a(self) -> int:
        self._log(
            f"preprocess_a canonical entry: python code/odcr.py preprocess --stage preprocess_a --preset {self.config.preset_name}"
        )
        forced_task_ids = {task_id for task_id in self.combine_task_ids if self._task_forced(task_id)}
        pending_datasets: list[str] = []
        pending_tasks: list[int] = []

        for dataset in self.config.datasets:
            outputs = self._dataset_output_paths(dataset)
            if self._unit_already_complete("dataset", dataset, outputs) and dataset not in self.config.runtime.force_datasets:
                self._write_unit_status(
                    unit_kind="dataset",
                    unit_name=dataset,
                    status="skipped",
                    output_files=outputs,
                    shell_log_path=self.paths.stage_log_path,
                    started_at=self.started_at,
                    finished_at=self.started_at,
                    reason="resume_skip_ok_outputs_present",
                )
            else:
                pending_datasets.append(dataset)
                self._write_unit_status(
                    unit_kind="dataset",
                    unit_name=dataset,
                    status="pending",
                    output_files=outputs,
                    shell_log_path=self.paths.stage_log_path,
                    reason="pending_run",
                )

        for task_id in self.combine_task_ids:
            outputs = self._task_output_paths(task_id)
            if (
                self._unit_already_complete("task", str(task_id), outputs)
                and task_id not in forced_task_ids
            ):
                self._write_unit_status(
                    unit_kind="task",
                    unit_name=str(task_id),
                    status="skipped",
                    output_files=outputs,
                    shell_log_path=self.paths.stage_log_path,
                    started_at=self.started_at,
                    finished_at=self.started_at,
                    reason="resume_skip_ok_outputs_present",
                )
            else:
                pending_tasks.append(task_id)
                self._write_unit_status(
                    unit_kind="task",
                    unit_name=str(task_id),
                    status="pending",
                    output_files=outputs,
                    shell_log_path=self.paths.stage_log_path,
                    reason="pending_run",
                )

        if not pending_datasets and not pending_tasks:
            self._log("[preprocess_a] no pending datasets or tasks")
            return 0

        for dataset in pending_datasets:
            dataset_outputs = self._dataset_output_paths(dataset)
            dataset_log = self._dataset_log_path(dataset, "cpu")
            started_at = _utc_now()
            self._write_unit_status(
                unit_kind="dataset",
                unit_name=dataset,
                status="running",
                output_files=dataset_outputs,
                shell_log_path=dataset_log,
                started_at=started_at,
                reason="preprocess_data_then_split_data",
            )
            rc = self._run_logged_subprocess(
                label=f"preprocess_data[{dataset}]",
                command=[
                    self.config.runtime.python_bin,
                    "code/preprocess_data.py",
                    "--datasets",
                    dataset,
                    "--data-dir",
                    str(self.data_root),
                ],
                unit_log_path=dataset_log,
                outputs=(dataset_outputs[0],),
                expect_refresh=True,
            )
            if rc != 0:
                self._write_unit_status(
                    unit_kind="dataset",
                    unit_name=dataset,
                    status="failed",
                    output_files=dataset_outputs,
                    shell_log_path=dataset_log,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    reason="preprocess_data_failed",
                    error_message=f"preprocess_data.py exited with code {rc}",
                )
                return rc
            rc = self._run_logged_subprocess(
                label=f"split_data[{dataset}]",
                command=[
                    self.config.runtime.python_bin,
                    "code/split_data.py",
                    "--datasets",
                    dataset,
                    "--data-dir",
                    str(self.data_root),
                ],
                unit_log_path=dataset_log,
                outputs=dataset_outputs[1:],
                expect_refresh=True,
            )
            if rc != 0:
                self._write_unit_status(
                    unit_kind="dataset",
                    unit_name=dataset,
                    status="failed",
                    output_files=dataset_outputs,
                    shell_log_path=dataset_log,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    reason="split_data_failed",
                    error_message=f"split_data.py exited with code {rc}",
                )
                return rc
            self._write_unit_status(
                unit_kind="dataset",
                unit_name=dataset,
                status="ok",
                output_files=dataset_outputs,
                shell_log_path=dataset_log,
                started_at=started_at,
                finished_at=_utc_now(),
                reason="completed",
            )

        for task_id in pending_tasks:
            outputs = self._task_output_paths(task_id)
            task_log = self._task_log_path(task_id)
            started_at = _utc_now()
            self._write_unit_status(
                unit_kind="task",
                unit_name=str(task_id),
                status="running",
                output_files=outputs,
                shell_log_path=task_log,
                started_at=started_at,
                reason="combine_data",
            )
            rc = self._run_logged_subprocess(
                label=f"combine_data[task={task_id}]",
                command=[
                    self.config.runtime.python_bin,
                    "code/combine_data.py",
                    "--task-id",
                    str(task_id),
                    "--data-dir",
                    str(self.data_root),
                    "--merged-dir",
                    str(self.merged_root),
                ],
                unit_log_path=task_log,
                outputs=outputs,
                expect_refresh=True,
            )
            if rc != 0:
                self._write_unit_status(
                    unit_kind="task",
                    unit_name=str(task_id),
                    status="failed",
                    output_files=outputs,
                    shell_log_path=task_log,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    reason="combine_data_failed",
                    error_message=f"combine_data.py exited with code {rc}",
                )
                return rc
            self._write_unit_status(
                unit_kind="task",
                unit_name=str(task_id),
                status="ok",
                output_files=outputs,
                shell_log_path=task_log,
                started_at=started_at,
                finished_at=_utc_now(),
                reason="completed",
            )
        return 0

    def _gpu_worker_ids(self, hardware: PreprocessHardwareConfig) -> list[tuple[int, int]]:
        workers = int(self.config.runtime.workers or 0)
        gpu_ids = list(hardware.gpu_ids)[:workers]
        return [(idx + 1, gpu_id) for idx, gpu_id in enumerate(gpu_ids)]

    def _assert_gpu_admission(self) -> dict[str, Any]:
        assert isinstance(self.config, (PreprocessBConfig, PreprocessCConfig))
        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"{self.config.stage} requires PyTorch CUDA admission before loading BGE-large; "
                "torch import failed. Enter the GPU session with: "
                "tmux -L odcr_gpu new-session -A -s odcr"
            ) from exc

        configured = tuple(int(item) for item in self.config.hardware.gpu_ids)
        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        names: list[str] = []
        if device_count > 0:
            for idx in range(device_count):
                try:
                    names.append(str(torch.cuda.get_device_name(idx)))
                except Exception:
                    names.append("<unavailable>")
        bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        tf32_matmul = bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False))
        tf32_cudnn = bool(getattr(torch.backends.cudnn, "allow_tf32", False))
        report = {
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": device_count,
            "configured_gpu_ids": list(configured),
            "device_names": names,
            "bf16_supported": bf16_supported,
            "tf32_matmul_allow": tf32_matmul,
            "tf32_cudnn_allow": tf32_cudnn,
        }
        self._log(f"[{self.config.stage}] gpu_admission={json.dumps(report, sort_keys=True)}")
        invalid = [gpu_id for gpu_id in configured if gpu_id >= device_count]
        if not report["cuda_available"] or device_count <= 0 or invalid:
            raise RuntimeError(
                f"{self.config.stage} requires visible CUDA GPUs before loading BGE-large; "
                f"admission={json.dumps(report, sort_keys=True)}. "
                "Enter or launch inside the GPU tmux session: tmux -L odcr_gpu new-session -A -s odcr"
            )
        return report

    def _prepare_gpu_pending_datasets(self) -> list[str]:
        pending: list[str] = []
        for dataset in self.config.datasets:
            outputs = self._dataset_output_paths(dataset)
            if (
                self._unit_already_complete("dataset", dataset, outputs)
                and dataset not in self.config.runtime.force_datasets
            ):
                self._write_unit_status(
                    unit_kind="dataset",
                    unit_name=dataset,
                    status="skipped",
                    output_files=outputs,
                    shell_log_path=self.paths.stage_log_path,
                    started_at=self.started_at,
                    finished_at=self.started_at,
                    reason="resume_skip_ok_outputs_present",
                )
                continue
            pending.append(dataset)
            self._write_unit_status(
                unit_kind="dataset",
                unit_name=dataset,
                status="pending",
                output_files=outputs,
                shell_log_path=self.paths.stage_log_path,
                reason="pending_run",
            )
        return pending

    def _gpu_dataset_command(
        self,
        config: PreprocessBConfig | PreprocessCConfig,
        *,
        dataset: str,
        gpu_id: int,
    ) -> list[str]:
        base = [
            config.runtime.python_bin,
            "code/compute_embeddings.py" if config.stage == "preprocess_b" else "code/infer_domain_semantics.py",
            "--datasets",
            dataset,
            "--cuda-device",
            str(gpu_id),
            "--data-dir",
            str(self.data_root),
            "--models-dir",
            str(self.models_root),
            "--sentence-embed-model",
            str(self.sentence_embed_model_path),
            "--embed-dim",
            str(self.embed_dim),
        ]
        if config.stage == "preprocess_b":
            base.extend(["--embed-batch-size", str(config.embed_batch_size)])
            base.extend(["--read-chunk-rows", str(config.read_chunk_rows)])
            base.extend(["--group-shard-size", str(config.group_shard_size)])
            base.append(
                "--tokenizer-parallelism"
                if config.tokenizer_parallelism_enabled
                else "--no-tokenizer-parallelism"
            )
            base.extend(["--tokenizer-threads-per-worker", str(config.tokenizer_threads_per_worker)])
            base.extend(["--tokenizer-total-threads", str(config.tokenizer_total_threads)])
            base.extend(["--prefetch-batches", str(config.prefetch_batches)])
            base.append("--pin-memory" if config.pin_memory else "--no-pin-memory")
            base.append("--non-blocking-h2d" if config.non_blocking_h2d else "--no-non-blocking-h2d")
            base.append("--async-prefetch" if config.async_prefetch_enabled else "--no-async-prefetch")
            base.append(
                "--token-aware-batching"
                if config.token_aware_batching_enabled
                else "--no-token-aware-batching"
            )
            if config.max_tokens_per_gpu_batch is not None:
                base.extend(["--max-tokens-per-gpu-batch", str(config.max_tokens_per_gpu_batch)])
            base.extend(["--cpu-cores-reserved", str(config.cpu_cores_reserved)])
            base.extend(["--cpu-cores-available", str(config.cpu_cores_available)])
            base.append("--grouped-text-cache" if config.grouped_text_cache_enabled else "--no-grouped-text-cache")
            base.extend(["--grouped-text-cache-dir", str(config.grouped_text_cache_dir)])
            base.extend(["--grouped-text-cache-version", str(config.grouped_text_cache_version)])
            base.append("--bf16" if config.bf16_enabled else "--no-bf16")
            base.append("--tf32" if config.tf32_enabled else "--no-tf32")
            if config.runtime.verify_only:
                base.append("--verify-only")
                base.extend(["--verify-sample-size", str(config.verify_sample_size)])
                base.extend(["--verify-seed", str(config.verify_seed)])
                if config.verify_user_indices:
                    base.extend(
                        ["--verify-user-indices", ",".join(str(item) for item in config.verify_user_indices)]
                    )
                if config.verify_item_indices:
                    base.extend(
                        ["--verify-item-indices", ",".join(str(item) for item in config.verify_item_indices)]
                    )
        else:
            base.extend(["--chunk-batch-size", str(config.chunk_batch_size)])
            base.append(
                "--tokenizer-parallelism"
                if config.tokenizer_parallelism_enabled
                else "--no-tokenizer-parallelism"
            )
            base.extend(["--tokenizer-threads-per-worker", str(config.tokenizer_threads_per_worker)])
            base.extend(["--tokenizer-total-threads", str(config.tokenizer_total_threads)])
            base.extend(["--prefetch-batches", str(config.prefetch_batches)])
            base.append("--pin-memory" if config.pin_memory else "--no-pin-memory")
            base.append("--non-blocking-h2d" if config.non_blocking_h2d else "--no-non-blocking-h2d")
            base.append("--async-prefetch" if config.async_prefetch_enabled else "--no-async-prefetch")
            base.extend(["--cpu-cores-reserved", str(config.cpu_cores_reserved)])
            base.extend(["--cpu-cores-available", str(config.cpu_cores_available)])
            base.append("--bf16" if config.bf16_enabled else "--no-bf16")
            base.append("--tf32" if config.tf32_enabled else "--no-tf32")
            base.append("--tokenizer-hotpath" if config.tokenizer_hotpath_enabled else "--no-tokenizer-hotpath")
            base.append("--token-window-cache" if config.token_window_cache_enabled else "--no-token-window-cache")
            base.extend(["--token-window-cache-dir", str(config.token_window_cache_dir)])
            base.extend(["--token-window-cache-version", str(config.token_window_cache_version)])
            base.extend(["--token-window-cache-shard-size", str(config.token_window_cache_shard_size)])
            if config.runtime.verify_only:
                base.append("--verify-only")
        return base

    def _latest_preprocess_c_metrics(self) -> dict[str, Any] | None:
        metrics_root = Path(self.config.resolved.runs_dir).resolve() / "preprocess" / "c"
        candidates = sorted(metrics_root.glob("*/meta/metrics.json"), key=lambda path: path.stat().st_mtime if path.exists() else 0)
        for path in reversed(candidates):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _estimate_preprocess_c_workloads(self, datasets: list[str]) -> dict[str, dict[str, Any]]:
        metrics = self._latest_preprocess_c_metrics()
        token_windows_by_dataset: dict[str, int] = {}
        elapsed_by_dataset: dict[str, float] = {}
        if isinstance(metrics, dict):
            for item in metrics.get("per_domain", []) or []:
                if not isinstance(item, dict):
                    continue
                dataset = str(item.get("dataset") or "")
                if dataset not in datasets:
                    continue
                token_windows_by_dataset[dataset] = token_windows_by_dataset.get(dataset, 0) + int(
                    item.get("token_window_count") or 0
                )
                elapsed_by_dataset[dataset] = elapsed_by_dataset.get(dataset, 0.0) + float(item.get("elapsed_s") or 0.0)
        estimates: dict[str, dict[str, Any]] = {}
        for dataset in datasets:
            train_csv = self.data_root / dataset / "train.csv"
            train_size = int(train_csv.stat().st_size) if train_csv.exists() else 0
            token_windows = int(token_windows_by_dataset.get(dataset, 0))
            historical_elapsed_s = float(elapsed_by_dataset.get(dataset, 0.0))
            if token_windows > 0:
                estimated = float(token_windows)
                basis = "historical_token_window_count"
            elif historical_elapsed_s > 0:
                estimated = historical_elapsed_s
                basis = "historical_elapsed_s"
            else:
                estimated = float(max(train_size, 1))
                basis = "train_csv_size_bytes"
            estimates[dataset] = {
                "dataset": dataset,
                "estimated_workload": estimated,
                "workload_basis": basis,
                "historical_token_windows": token_windows,
                "historical_elapsed_s": historical_elapsed_s,
                "train_csv_size_bytes": train_size,
            }
        return estimates

    def _lpt_worker_assignments(
        self,
        *,
        workers: list[tuple[int, int]],
        pending: list[str],
        estimates: dict[str, dict[str, Any]],
    ) -> tuple[dict[int, list[str]], dict[int, float]]:
        assignments = {worker_id: [] for worker_id, _ in workers}
        totals = {worker_id: 0.0 for worker_id, _ in workers}
        sorted_pending = sorted(
            pending,
            key=lambda dataset: (
                -float(estimates.get(dataset, {}).get("estimated_workload") or 0.0),
                dataset,
            ),
        )
        for dataset in sorted_pending:
            worker_id = min(totals, key=lambda item: (totals[item], item))
            assignments[worker_id].append(dataset)
            totals[worker_id] += float(estimates.get(dataset, {}).get("estimated_workload") or 0.0)
        return assignments, totals

    def _gpu_worker_assignments(
        self,
        *,
        workers: list[tuple[int, int]],
        pending: list[str],
    ) -> tuple[dict[int, list[str]], dict[str, dict[str, Any]], dict[int, float], str]:
        if not isinstance(self.config, PreprocessCConfig) or self.config.scheduling_policy == "dataset_order":
            assignments = {worker_id: [] for worker_id, _ in workers}
            for idx, dataset in enumerate(pending):
                worker_id = workers[idx % len(workers)][0]
                assignments[worker_id].append(dataset)
            estimates = {
                dataset: {
                    "dataset": dataset,
                    "estimated_workload": float(idx + 1),
                    "workload_basis": "dataset_order",
                }
                for idx, dataset in enumerate(pending)
            }
            totals = {
                worker_id: float(sum(estimates[dataset]["estimated_workload"] for dataset in datasets))
                for worker_id, datasets in assignments.items()
            }
            return assignments, estimates, totals, "dataset_order"
        estimates = self._estimate_preprocess_c_workloads(pending)
        assignments, totals = self._lpt_worker_assignments(
            workers=workers,
            pending=pending,
            estimates=estimates,
        )
        return assignments, estimates, totals, "lpt_by_token_windows"

    def _run_preprocess_gpu(self) -> int:
        assert isinstance(self.config, (PreprocessBConfig, PreprocessCConfig))
        if not self.config.runtime.dry_run:
            self._assert_gpu_admission()
        pending = self._prepare_gpu_pending_datasets()
        workers = self._gpu_worker_ids(self.config.hardware)
        self._log(
            f"{self.config.stage} canonical entry: python code/odcr.py preprocess --stage {self.config.stage} --preset {self.config.preset_name}"
        )
        self._log(f"[{self.config.stage}] gpu_ids={self.config.hardware.gpu_ids} workers={len(workers)}")
        self._log(f"[{self.config.stage}] pending_datasets={pending or ['<none>']}")

        if not pending:
            return 0

        assignments, workload_estimates, worker_estimated_totals, scheduling_policy = self._gpu_worker_assignments(
            workers=workers,
            pending=pending,
        )
        nonzero_estimates = [value for value in worker_estimated_totals.values() if value > 0]
        imbalance_ratio = (
            (max(nonzero_estimates) / max(min(nonzero_estimates), 1e-12)) if nonzero_estimates else 1.0
        )
        static_assignments_enabled = isinstance(self.config, PreprocessCConfig) and scheduling_policy == "lpt_by_token_windows"
        scheduling_payload = {
            "policy": scheduling_policy,
            "assignments": assignments,
            "estimated_workload": workload_estimates,
            "worker_estimated_total": worker_estimated_totals,
            "imbalance_ratio": round(float(imbalance_ratio), 6),
            "assignment_mode": "static_lpt" if static_assignments_enabled else "dynamic_queue",
        }
        self._log(f"[{self.config.stage}] scheduling={json.dumps(scheduling_payload, sort_keys=True)}")

        work_queue: queue.Queue[str] | None = None
        if not static_assignments_enabled:
            work_queue = queue.Queue()
            for dataset in pending:
                work_queue.put(dataset)

        failure_event = threading.Event()
        results_lock = threading.Lock()
        worker_actual_totals: dict[int, float] = {worker_id: 0.0 for worker_id, _ in workers}

        def worker_main(worker_id: int, gpu_id: int) -> None:
            handled: list[str] = []
            exit_code = 0
            worker_actual_total_s = 0.0
            static_iter = iter(assignments.get(worker_id, []))
            while not failure_event.is_set():
                if static_assignments_enabled:
                    try:
                        dataset = next(static_iter)
                    except StopIteration:
                        break
                else:
                    assert work_queue is not None
                    try:
                        dataset = work_queue.get_nowait()
                    except queue.Empty:
                        break
                if failure_event.is_set():
                    break
                outputs = self._dataset_output_paths(dataset)
                dataset_log = self._dataset_log_path(dataset, f"worker{worker_id}")
                command = self._gpu_dataset_command(self.config, dataset=dataset, gpu_id=gpu_id)
                label = f"{self.stage_label}[worker={worker_id} gpu={gpu_id} dataset={dataset}]"
                started_at = _utc_now()
                dataset_started_perf = time.perf_counter()
                self._write_unit_status(
                    unit_kind="dataset",
                    unit_name=dataset,
                    status="running",
                    output_files=outputs,
                    shell_log_path=dataset_log,
                    started_at=started_at,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    reason="subprocess_running",
                    command=command,
                    unit_metadata={
                        "scheduling_policy": scheduling_policy,
                        "estimated_workload": workload_estimates.get(dataset),
                        "assigned_worker": worker_id,
                        "worker_estimated_total": worker_estimated_totals.get(worker_id),
                        "worker_actual_total": round(float(worker_actual_total_s), 6),
                        "scheduling_imbalance_ratio": round(float(imbalance_ratio), 6),
                    },
                )
                try:
                    rc = self._run_logged_subprocess(
                        label=label,
                        command=command,
                        unit_log_path=dataset_log,
                        outputs=outputs,
                        expect_refresh=not self.config.runtime.verify_only,
                    )
                except Exception as exc:  # pragma: no cover - surfaced in status/final exit
                    rc = 1
                    self._write_unit_status(
                        unit_kind="dataset",
                        unit_name=dataset,
                        status="failed",
                        output_files=outputs,
                        shell_log_path=dataset_log,
                        started_at=started_at,
                        finished_at=_utc_now(),
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                        reason="subprocess_exception",
                        error_message=str(exc),
                        command=command,
                    )
                    failure_event.set()
                    exit_code = rc
                    handled.append(dataset)
                    if not static_assignments_enabled and work_queue is not None:
                        work_queue.task_done()
                    break
                dataset_actual_s = time.perf_counter() - dataset_started_perf
                worker_actual_total_s += dataset_actual_s
                if rc == 0:
                    self._write_unit_status(
                        unit_kind="dataset",
                        unit_name=dataset,
                        status="ok",
                        output_files=outputs,
                        shell_log_path=dataset_log,
                        started_at=started_at,
                        finished_at=_utc_now(),
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                        reason="completed",
                        command=command,
                        unit_metadata={
                            "scheduling_policy": scheduling_policy,
                            "estimated_workload": workload_estimates.get(dataset),
                            "assigned_worker": worker_id,
                            "worker_estimated_total": worker_estimated_totals.get(worker_id),
                            "actual_dataset_wall_s": round(float(dataset_actual_s), 6),
                            "worker_actual_total": round(float(worker_actual_total_s), 6),
                            "scheduling_imbalance_ratio": round(float(imbalance_ratio), 6),
                        },
                    )
                else:
                    self._write_unit_status(
                        unit_kind="dataset",
                        unit_name=dataset,
                        status="failed",
                        output_files=outputs,
                        shell_log_path=dataset_log,
                        started_at=started_at,
                        finished_at=_utc_now(),
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                        reason="subprocess_failed",
                        error_message=f"subprocess exited with code {rc}",
                        command=command,
                        unit_metadata={
                            "scheduling_policy": scheduling_policy,
                            "estimated_workload": workload_estimates.get(dataset),
                            "assigned_worker": worker_id,
                            "worker_estimated_total": worker_estimated_totals.get(worker_id),
                            "actual_dataset_wall_s": round(float(dataset_actual_s), 6),
                            "worker_actual_total": round(float(worker_actual_total_s), 6),
                            "scheduling_imbalance_ratio": round(float(imbalance_ratio), 6),
                        },
                    )
                    failure_event.set()
                    exit_code = rc
                handled.append(dataset)
                if not static_assignments_enabled and work_queue is not None:
                    work_queue.task_done()
                if rc != 0:
                    break
            with results_lock:
                worker_actual_totals[worker_id] = worker_actual_total_s
                self._worker_results.append(
                    PreprocessWorkerResult(
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                        exit_code=exit_code,
                        handled_units=tuple(handled),
                    )
                )

        threads: list[threading.Thread] = []
        for worker_id, gpu_id in workers:
            thread = threading.Thread(
                target=worker_main,
                name=f"preprocess-worker-{worker_id}",
                args=(worker_id, gpu_id),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        nonzero_actual = [value for value in worker_actual_totals.values() if value > 0]
        actual_imbalance_ratio = (
            (max(nonzero_actual) / max(min(nonzero_actual), 1e-12)) if nonzero_actual else 1.0
        )
        self._log(
            f"[{self.config.stage}] scheduling_result="
            f"{json.dumps({'policy': scheduling_policy, 'assignment_mode': 'static_lpt' if static_assignments_enabled else 'dynamic_queue', 'worker_estimated_total': worker_estimated_totals, 'worker_actual_total': worker_actual_totals, 'estimated_imbalance_ratio': round(float(imbalance_ratio), 6), 'actual_imbalance_ratio': round(float(actual_imbalance_ratio), 6)}, sort_keys=True)}"
        )

        if failure_event.is_set():
            return 1
        return 0


def run_preprocess_cli(args: argparse.Namespace) -> None:
    config = resolve_preprocess_cli_config(args)
    runtime = PreprocessRuntime(config)
    runtime.run()
