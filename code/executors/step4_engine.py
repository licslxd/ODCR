"""
Step4 执行体核心（ENGINE）：反事实推理生成（与 eval 同属 eval 语义侧）。

全局推理 batch 由父进程根据 ``--eval-profile`` 解析的 ``eval_batch_size`` 传入（torchrun ``--batch-size``），
须满足与 eval 相同的 strict 整除 ``ddp_world_size`` 规则；**不使用** training 的 ``train_batch_size``。

由 ``executors.step4_entry`` 在 torchrun 下调用；用户入口请使用 ``python code/odcr.py step4 … --eval-profile …``。
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
from executors.step3_train_core import *
from base_utils import get_underlying_model
from data_contract import PREPROCESS_CONTRACT_VERSION
from config import (
    BASE_TRAINING_DEFAULTS,
    TASK_DEFAULTS,
    get_dataloader_num_workers,
    get_dataloader_prefetch_factor,
    get_num_proc,
    get_odcr_embed_dim,
)
from datasets import Dataset, load_from_disk
from paths_config import (
    append_log_dual,
    get_data_dir,
    get_odcr_root,
    get_merged_data_dir,
    get_stage_run_dir,
    require_step5_text_model_dir,
)
from perf_monitor import PerfMonitor
from train_diagnostics import odcr_cuda_bf16_autocast
from odcr_core.gather_schema import require_gathered_batch
from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    ODCR_ROUTING_TRAIN_CSV,
    build_index_contract,
    build_step4_export_lineage,
    load_profile_tensors_dual_first,
    parse_training_run_lineage,
)
from odcr_core.step4_training_export import (
    assemble_step4_training_table,
    build_step4_train_manifest,
    write_step4_training_artifacts,
)
from odcr_core.odcr_cf_routing import ODCFRoutingConfig, attach_odcr_cf_routing
from odcr_core.file_atomic import atomic_write_json
from odcr_core.training_checkpoint import (
    CheckpointLineageError,
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    current_effective_payload,
    current_resolved_config_lineage,
    current_one_control_resolved_config_hash,
    current_source_table_lineage,
    file_fingerprint,
    model_artifact_fingerprint,
    read_checkpoint_lineage,
    step3_resolved_config_compatibility_payload,
    step3_source_table_compatibility_payload,
    stable_hash,
    validate_step3_checkpoint_lineage,
)
from odcr_core.step3_upstream_gate import validate_step3_preprocess_upstream_gate
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, TensorDataset

_STEP4_REQUIRES_TORCHRUN_MSG = (
    "step4 runner 仅支持 torchrun / python -m torch.distributed.run DDP。\n"
    "用户日常（仓库根）: python code/odcr.py step4 …\n"
    "请勿在非 torchrun 环境下直接启动 step4_entry。\n"
    "高级排障（须 torchrun，在 code/ 目录）见 docs/ODCR_Scripts_and_Runtime_Guide.md 附录。\n"
    "多卡请设置 CUDA_VISIBLE_DEVICES 并使 nproc_per_node 与可见 GPU 数一致。"
)

# 编码逻辑 / Processor 口径变更时递增，避免误读旧 Arrow 缓存
_STEP4_ENCODE_CACHE_VERSION = "v5_lineage_manifest"
_STEP4_ENCODE_CACHE_SCHEMA_VERSION = "odcr_step4_encoded_cache/1"
_STEP4_ENCODE_CACHE_MANIFEST = "cache_manifest.json"
_STEP4_ENCODE_CACHE_COMPLETED_MARKER = "completed.marker"
_STEP4_ENCODE_CACHE_FAILED_MARKER = "failed.marker"
_STEP4_ENCODE_CACHE_PRODUCER_CODE_VERSION = "executors.step4_engine.encoded_cache/2"
_STEP4_ENCODE_CACHE_REQUIRED_FIELDS = (
    "user_idx",
    "item_idx",
    "rating",
    "explanation",
    "domain",
    "sample_id",
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
)


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
    for unit in ("a", "b", "c"):
        item = preprocess.get(unit)
        if not isinstance(item, Mapping):
            raise CheckpointLineageError(f"Step4 refused Step3 checkpoint: preprocess_{unit} evidence missing.")
        latest_run_ids[unit] = str(item.get("run_id") or "")
        run_summary_fps[unit] = item.get("run_summary_fingerprint")
        stage_status_fps[unit] = item.get("stage_status_fingerprint")
        stage_manifest_fps[unit] = item.get("stage_manifest_fingerprint")
        source_table_fps[unit] = item.get("source_table_fingerprint")
        metrics_fps[unit] = item.get("metrics_fingerprint")
        verify_fps[unit] = item.get("verify_report_fingerprint")
    return {
        "preprocess_latest_run_ids": latest_run_ids,
        "preprocess_run_summary_fingerprints_hash": stable_hash(run_summary_fps),
        "preprocess_stage_status_fingerprints_hash": stable_hash(stage_status_fps),
        "preprocess_stage_manifest_fingerprints_hash": stable_hash(stage_manifest_fps),
        "preprocess_source_table_fingerprints_hash": stable_hash(source_table_fps),
        "preprocess_metrics_fingerprints_hash": stable_hash(metrics_fps),
        "preprocess_verify_report_fingerprints_hash": stable_hash(verify_fps),
    }


def _validate_step3_checkpoint_lineage_for_step4(
    *,
    checkpoint_path: str,
    task_idx: int,
    auxiliary: str,
    target: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = current_effective_payload(required=True)
    train_path = os.path.join(get_merged_data_dir(), str(task_idx), "aug_train.csv")
    valid_path = os.path.join(get_merged_data_dir(), str(task_idx), "aug_valid.csv")
    data_fps = {
        "aug_train_csv": file_fingerprint(train_path),
        "aug_valid_csv": file_fingerprint(valid_path),
    }
    current_arch = {
        "nuser": int(config["nuser"]),
        "nitem": int(config["nitem"]),
        "ntoken": int(config["ntoken"]),
        "emsize": int(config["emsize"]),
        "nlayers": int(config["nlayers"]),
        "nhead": int(config["nhead"]),
        "nhid": int(config["nhid"]),
        "dropout": float(config["dropout"]),
    }
    upstream_evidence = validate_step3_preprocess_upstream_gate(
        repo_root=get_odcr_root(),
        task_id=int(task_idx),
        auxiliary_domain=str(auxiliary),
        target_domain=str(target),
        data_dir=get_data_dir(),
        merged_dir=get_merged_data_dir(),
        runs_dir=os.environ.get("ODCR_RESOLVED_RUNS_DIR") or os.path.join(get_odcr_root(), "runs"),
        embed_dim=int(get_odcr_embed_dim()),
    )
    resolved_compatibility = step3_resolved_config_compatibility_payload(
        payload=payload,
        task_id=int(task_idx),
        source_domain=str(auxiliary),
        target_domain=str(target),
        embed_dim=int(get_odcr_embed_dim()),
        structured_losses=payload.get("step3_structured_losses") or {},
        loss_semantics=payload.get("step3_loss_semantics") or {},
        architecture_hash=stable_hash(current_arch),
    )
    source_table = current_source_table_lineage(required_file=bool((os.environ.get("ODCR_MANIFEST_DIR") or "").strip()))
    source_table_compatibility = step3_source_table_compatibility_payload(source_table)
    data_contract_payload = {
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "source_task": {
            "task_id": int(task_idx),
            "auxiliary": str(auxiliary),
            "target": str(target),
        },
        "source_csv_fingerprints": upstream_evidence.get("source_csv_artifacts"),
        "merged_csv_fingerprints": upstream_evidence.get("merged_artifacts") or data_fps,
    }
    artifact_lineage_payload = {
        "data_merged_artifact_fingerprint": stable_hash(data_fps),
        "preprocess": _step3_preprocess_lineage_expected_for_step4(upstream_evidence),
        "profile_artifact_fingerprints": upstream_evidence.get("profile_artifact_fingerprints"),
        "domain_artifact_fingerprints": upstream_evidence.get("domain_artifact_fingerprints"),
        "sentence_embed_model_identity": {
            "identity": os.path.abspath(os.environ.get("ODCR_RESOLVED_SENTENCE_EMBED_MODEL") or get_sentence_embed_model_dir()),
            "resolved_env_key": "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
            "model_artifact_fingerprint": model_artifact_fingerprint(
                os.environ.get("ODCR_RESOLVED_SENTENCE_EMBED_MODEL") or get_sentence_embed_model_dir()
            ),
        },
    }
    semantic_model_payload = {
        "resolved_config_compatibility": resolved_compatibility,
        "source_table_compatibility": source_table_compatibility,
        "embed_dim": int(get_odcr_embed_dim()),
        "model_architecture_config_hash": stable_hash(current_arch),
        "representation_output_contract_hash": stable_hash(
            {
                "Step3ForwardOutput": "odcr_step3_forward_output/structured_shared_specific_v1",
                "Step3LossBundle": "odcr_step3_loss_bundle/structured_shared_specific_v1",
            }
        ),
        "structured_losses_hash": stable_hash(payload.get("step3_structured_losses") or {}),
        "loss_semantics_hash": stable_hash(payload.get("step3_loss_semantics") or {}),
        "profile_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("profile_artifact_fingerprints") or {}),
        "domain_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("domain_artifact_fingerprints") or {}),
    }
    expected = {
        "sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "task_id": int(task_idx),
        "source_domain": str(auxiliary),
        "target_domain": str(target),
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "data_merged_artifact_fingerprint": stable_hash(data_fps),
        "embed_dim": int(get_odcr_embed_dim()),
        "step3_structured_losses_config_hash": stable_hash(payload.get("step3_structured_losses") or {}),
        "model_architecture_config_hash": stable_hash(current_arch),
        "resolved_config_compatibility_hash": stable_hash(resolved_compatibility),
        "source_table_compatibility_hash": stable_hash(source_table_compatibility),
        "semantic_model_compat_hash": stable_hash(semantic_model_payload),
        "data_contract_hash": stable_hash(data_contract_payload),
        "artifact_lineage_hash": stable_hash(artifact_lineage_payload),
        **_step3_preprocess_lineage_expected_for_step4(upstream_evidence),
        "profile_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("profile_artifact_fingerprints") or {}),
        "domain_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("domain_artifact_fingerprints") or {}),
        "source_csv_fingerprints_hash": stable_hash(upstream_evidence.get("source_csv_artifacts") or {}),
        "merged_csv_fingerprints_hash": stable_hash(upstream_evidence.get("merged_artifacts") or data_fps),
        "source_task": {
            "task_id": int(task_idx),
            "auxiliary": str(auxiliary),
            "target": str(target),
            "scenario": str(payload.get("scenario") or ""),
            "direction": str(payload.get("direction") or ""),
        },
    }
    try:
        # Includes the checkpoint_file_hash hard gate before torch.load.
        return validate_step3_checkpoint_lineage(checkpoint_path, expected=expected)
    except CheckpointLineageError as exc:
        raise CheckpointLineageError(f"Step4 refused Step3 checkpoint: {exc}") from exc


def _require_torchrun_env_vars() -> None:
    for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
        if k not in os.environ:
            raise RuntimeError(_STEP4_REQUIRES_TORCHRUN_MSG)


def _setup_distributed():
    _require_torchrun_env_vars()
    if not torch.cuda.is_available():
        raise RuntimeError("Step 4 DDP 推理需要 CUDA（与 Step 3/5 一致，后端 NCCL）。")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _teardown_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _step4_encoded_cache_fingerprint(
    task_idx: int,
    aug_csv_path: str,
    auxiliary: str,
    target: str,
    t5_resolved: str,
    processor_max_length: int,
    step3_checkpoint_lineage_hash: str,
) -> dict[str, Any]:
    """Hard cache identity: source hash + tokenizer artifact + resolved config lineage."""
    source_fp = file_fingerprint(aug_csv_path)
    tokenizer_fp = model_artifact_fingerprint(t5_resolved)
    required_fields_hash = stable_hash(_STEP4_ENCODE_CACHE_REQUIRED_FIELDS)
    resolved_config_hash = current_one_control_resolved_config_hash(
        extra={"stage": "step4", "task_id": int(task_idx), "artifact": "encoded_cache"}
    )
    payload: dict[str, Any] = {
        "schema_version": _STEP4_ENCODE_CACHE_SCHEMA_VERSION,
        "cache_version": _STEP4_ENCODE_CACHE_VERSION,
        "stage": "step4",
        "task_id": int(task_idx),
        "auxiliary": str(auxiliary),
        "target": str(target),
        "source_aug_csv": source_fp,
        "data_source_hash": str(source_fp.get("sha256") or ""),
        "source_data_path": str(source_fp.get("path") or os.path.abspath(aug_csv_path)),
        "source_data_sha256": str(source_fp.get("sha256") or ""),
        "source_data_size": int(source_fp.get("size", -1)),
        "source_data_mtime_ns": int(source_fp.get("mtime_ns", -1)),
        "tokenizer_model": {
            "path": os.path.abspath(t5_resolved),
            "artifact_fingerprint": tokenizer_fp,
        },
        "tokenizer_path_or_id": os.path.abspath(t5_resolved),
        "tokenizer_fingerprint": tokenizer_fp,
        "tokenizer_config_hash": stable_hash(tokenizer_fp),
        "processor": {
            "name": "executors.step4_engine.Processor",
            "max_length": int(processor_max_length),
            "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
            "required_fields": list(_STEP4_ENCODE_CACHE_REQUIRED_FIELDS),
        },
        "max_length": int(processor_max_length),
        "resolved_config_hash": resolved_config_hash,
        "one_control_resolved_config_hash": resolved_config_hash,
        "rcr_step4_config_hash": stable_hash(
            json.loads(os.environ.get("ODCR_STEP4_RCR_CONFIG_JSON") or "{}")
        ),
        "upstream_step3_run_id": str(os.path.basename(os.path.abspath(os.environ.get("ODCR_STEP3_RUN_DIR") or ""))),
        "step3_checkpoint_hash": _step4_file_sha256(
            os.path.join(os.environ.get("ODCR_STEP3_RUN_DIR") or "", "model", "best.pth")
        )
        if os.environ.get("ODCR_STEP3_RUN_DIR")
        and os.path.isfile(os.path.join(os.environ.get("ODCR_STEP3_RUN_DIR") or "", "model", "best.pth"))
        else "",
        "step3_checkpoint_lineage_hash": str(step3_checkpoint_lineage_hash),
        "index_contract_or_required_fields_hash": required_fields_hash,
        "producer_code_version": _STEP4_ENCODE_CACHE_PRODUCER_CODE_VERSION,
        "training_semantic_fingerprint": os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT", ""),
        "generation_semantic_fingerprint": os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT", ""),
    }
    payload["fingerprint_hash"] = stable_hash(payload)
    return payload


def _step4_encoded_cache_dir(task_idx: int, fingerprint: Mapping[str, Any]) -> str:
    digest = stable_hash(dict(fingerprint), length=24)
    return os.path.join(get_odcr_root(), "cache", "step4_encoded", str(task_idx), digest)


def _step4_encoded_cache_manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _STEP4_ENCODE_CACHE_MANIFEST)


def _step4_encoded_cache_completed_marker_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _STEP4_ENCODE_CACHE_COMPLETED_MARKER)


def _step4_encoded_cache_failed_marker_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _STEP4_ENCODE_CACHE_FAILED_MARKER)


def _load_step4_encoded_cache_manifest(cache_dir: str) -> dict[str, Any] | None:
    path = _step4_encoded_cache_manifest_path(cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _step4_encoded_cache_manifest_gate_fields(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_schema_version": _STEP4_ENCODE_CACHE_SCHEMA_VERSION,
        "schema_version": _STEP4_ENCODE_CACHE_SCHEMA_VERSION,
        "cache_version": _STEP4_ENCODE_CACHE_VERSION,
        "stage": "step4",
        "task_id": int(fingerprint.get("task_id", -1)),
        "source_data_path": str(fingerprint.get("source_data_path") or ""),
        "source_data_sha256": str(fingerprint.get("source_data_sha256") or ""),
        "source_data_size": int(fingerprint.get("source_data_size", -1)),
        "source_data_mtime_ns": int(fingerprint.get("source_data_mtime_ns", -1)),
        "tokenizer_path_or_id": str(fingerprint.get("tokenizer_path_or_id") or ""),
        "tokenizer_fingerprint": fingerprint.get("tokenizer_fingerprint"),
        "tokenizer_config_hash": str(fingerprint.get("tokenizer_config_hash") or ""),
        "max_length": int(fingerprint.get("max_length", -1)),
        "resolved_config_hash": str(fingerprint.get("resolved_config_hash") or ""),
        "rcr_step4_config_hash": str(fingerprint.get("rcr_step4_config_hash") or ""),
        "upstream_step3_run_id": str(fingerprint.get("upstream_step3_run_id") or ""),
        "step3_checkpoint_hash": str(fingerprint.get("step3_checkpoint_hash") or ""),
        "step3_checkpoint_lineage_hash": str(fingerprint.get("step3_checkpoint_lineage_hash") or ""),
        "index_contract_or_required_fields_hash": str(fingerprint.get("index_contract_or_required_fields_hash") or ""),
        "producer_code_version": _STEP4_ENCODE_CACHE_PRODUCER_CODE_VERSION,
    }


def _step4_encoded_cache_manifest_matches(
    cache_dir: str,
    *,
    expected_fingerprint: Mapping[str, Any],
    expected_rows: int,
) -> tuple[bool, str]:
    if os.path.exists(_step4_encoded_cache_failed_marker_path(cache_dir)):
        return False, "failed_marker_present"
    if not _dataset_saved_to_disk(cache_dir):
        return False, "missing_dataset"
    manifest = _load_step4_encoded_cache_manifest(cache_dir)
    if manifest is None:
        return False, "missing_manifest"
    if not os.path.exists(_step4_encoded_cache_completed_marker_path(cache_dir)):
        return False, "missing_completed_marker"
    if str(manifest.get("schema_version")) != _STEP4_ENCODE_CACHE_SCHEMA_VERSION:
        return False, "schema_mismatch"
    if str(manifest.get("cache_version")) != _STEP4_ENCODE_CACHE_VERSION:
        return False, "version_mismatch"
    expected_hash = str(expected_fingerprint.get("fingerprint_hash") or "")
    if str(manifest.get("fingerprint_hash") or "") != expected_hash:
        return False, "fingerprint_mismatch"
    if int(manifest.get("row_count", -1)) != int(expected_rows):
        return False, "row_count_mismatch"
    expected_gate = _step4_encoded_cache_manifest_gate_fields(expected_fingerprint)
    for key, expected_value in expected_gate.items():
        if manifest.get(key) != expected_value:
            return False, f"{key}_mismatch"
    return True, "hit"


def _write_step4_encoded_cache_manifest(
    cache_dir: str,
    *,
    fingerprint: Mapping[str, Any],
    row_count: int,
) -> None:
    row_count = int(row_count)
    atomic_write_json(
        _step4_encoded_cache_manifest_path(cache_dir),
        {
            **_step4_encoded_cache_manifest_gate_fields(fingerprint),
            "cache_version": _STEP4_ENCODE_CACHE_VERSION,
            "fingerprint_hash": str(fingerprint.get("fingerprint_hash") or ""),
            "fingerprint": dict(fingerprint),
            "row_count": row_count,
            "sample_count": row_count,
            "cache_compatibility_hash": str(fingerprint.get("fingerprint_hash") or ""),
            "dataset_format": "huggingface_dataset_save_to_disk",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    completed = _step4_encoded_cache_completed_marker_path(cache_dir)
    with open(completed, "w", encoding="utf-8") as handle:
        handle.write("ok\n")


def _dataset_saved_to_disk(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json"))


def _step4_pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def _step4_partial_format_choice() -> str:
    """parquet（需 pyarrow）/ csv / auto（有 pyarrow 则 parquet）。"""
    v = str(_step4_runtime_config().get("partial_format", "auto")).strip().lower()
    if v in ("parquet", "csv", "auto"):
        return v
    return "auto"


def _step4_runtime_config() -> dict[str, Any]:
    raw = (os.environ.get("ODCR_STEP4_RUNTIME_CONFIG_JSON") or "").strip()
    if not raw:
        return {
            "decode_threads": 0,
            "decode_chunk": 4096,
            "partial_format": "auto",
            "perf_log_interval": 10,
            "partial_wait_timeout_seconds": 600,
        }
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ODCR_STEP4_RUNTIME_CONFIG_JSON invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("ODCR_STEP4_RUNTIME_CONFIG_JSON must be an object")
    return obj


def _reject_step4_formal_bare_runtime_env() -> None:
    mode = (os.environ.get("ODCR_STEP4_MODE") or "formal").strip()
    if mode != "formal":
        return
    bad = [
        key
        for key in (
            "ODCR_STEP4_DECODE_THREADS",
            "ODCR_STEP4_DECODE_CHUNK",
            "ODCR_STEP4_PARTIAL_FORMAT",
            "ODCR_STEP4_PERF_LOG_INTERVAL",
        )
        if (os.environ.get(key) or "").strip()
    ]
    if bad:
        raise RuntimeError(
            "Step4 formal runtime refuses bare perf/env overrides: "
            + ", ".join(bad)
            + ". Configure configs/odcr.yaml: step4.runtime."
        )


def _step4_partial_suffix_and_kind(fmt_choice: str) -> tuple[str, str]:
    if fmt_choice == "csv":
        return ".csv", "csv"
    if fmt_choice == "parquet":
        if _step4_pyarrow_available():
            return ".parquet", "parquet"
        return ".csv", "csv"
    if _step4_pyarrow_available():
        return ".parquet", "parquet"
    return ".csv", "csv"


def _step4_write_partial_df(df: pd.DataFrame, path: str, kind: str) -> None:
    if kind == "parquet":
        df.to_parquet(path, index=False, engine="pyarrow")
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def _step4_read_partial_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path, encoding="utf-8")


def _step4_file_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _step4_partial_manifest_path(partial_path: str) -> str:
    return partial_path + ".manifest.json"


def _step4_partial_failed_marker(partial_path: str) -> str:
    return partial_path + ".failed"


def _step4_write_partial_manifest(
    *,
    partial_path: str,
    rank: int,
    world_size: int,
    row_count: int,
    kind: str,
) -> str:
    manifest = {
        "schema_version": "odcr_step4_partial_artifact/1",
        "status": "ok",
        "rank": int(rank),
        "world_size": int(world_size),
        "shard_id": int(rank),
        "path": os.path.abspath(partial_path),
        "row_count": int(row_count),
        "format": str(kind),
        "sha256": _step4_file_sha256(partial_path),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = _step4_partial_manifest_path(partial_path)
    atomic_write_json(out, manifest)
    return out


def _step4_wait_for_partial_manifests(partial_dir: str, *, world_size: int, timeout_s: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + max(1, int(timeout_s))
    while True:
        failed = sorted(Path(partial_dir).glob("*.failed"))
        if failed:
            raise RuntimeError("Step4 partial failed marker present: " + ", ".join(str(p) for p in failed[:5]))
        manifests = sorted(Path(partial_dir).glob("*.manifest.json"))
        if len(manifests) >= int(world_size):
            out = []
            for path in manifests:
                with open(path, "r", encoding="utf-8") as handle:
                    out.append(json.load(handle))
            ranks = {int(item.get("rank", -1)) for item in out}
            if ranks == set(range(int(world_size))):
                return out
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Step4 partial manifest wait timed out after {timeout_s}s in {partial_dir}; "
                "no endless polling is allowed."
            )
        time.sleep(0.1)


def _step4_upstream_artifact_hash(key: str) -> str:
    try:
        payload = json.loads(os.environ.get("ODCR_UPSTREAM_RESOLUTION_JSON") or "{}")
    except json.JSONDecodeError:
        payload = {}
    validation = payload.get("stage_status_validation") if isinstance(payload, Mapping) else None
    raw = validation.get(key) if isinstance(validation, Mapping) else None
    if not raw:
        return ""
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path(get_odcr_root()) / path
    if not path.is_file():
        return ""
    return _step4_file_sha256(str(path))


def _step4_append_primary_log(primary_log_file: str, text: str) -> None:
    """DDP 多进程追加同一主日志：Linux 下用 flock，避免行被拆散。"""
    path = os.path.abspath(os.path.expanduser(primary_log_file))
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
        return
    with open(path, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(text)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Step4PerfLogger:
    """性能与进度行写入主日志 ``full.log``（不再生成 ``step4_perf_*.log``）。"""

    def __init__(self, log_file: str, task_idx: int, rank: int):
        _ = task_idx
        self._path = os.path.abspath(os.path.expanduser(log_file))
        log_dir = os.path.dirname(self._path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.rank = rank

    def line(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{ts}] [rank{self.rank}] {msg}\n"
        print(text, end="", flush=True)
        try:
            _step4_append_primary_log(self._path, text)
        except Exception:
            pass


def _format_seconds_human(seconds: float) -> str:
    """人类可读时长：秒或分钟。"""
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60.0:.2f} min"


def _step4_pct(part: float, whole: float, ndigits: int = 1) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, ndigits)


def _log_step4_final_summary(
    *,
    task_idx: int,
    world_size: int,
    n_rows: int,
    log_file: str,
    step4_end_to_end_wall_s: float,
    preprocess_wall_s: float,
    inference_loop_wall_s: float,
    decode_local_wall_s: float,
    merge_wall_s: float,
    filter_wall_s: float,
    csv_write_wall_s: float,
    barrier_after_inference_wall_s: float,
    collective_gather_paths_wall_s: float,
    trainer_epoch_time_s: float | None,
    inference_only_avg_step_ms: float,
) -> None:
    """rank0：stdout + ``full.log``（经 append_log_dual）；摘要块不重复写入 perf 侧车文件。"""
    e2e = step4_end_to_end_wall_s
    decode_tail_s = decode_local_wall_s
    infer_pct = _step4_pct(inference_loop_wall_s, e2e)
    decode_pct = _step4_pct(decode_tail_s, e2e)
    csv_pct = _step4_pct(csv_write_wall_s, e2e)
    trainer_s_str = f"{float(trainer_epoch_time_s):.4f}" if trainer_epoch_time_s is not None else "n/a"

    mach = (
        "[rank0] step4_final_summary\n"
        f"  step4_end_to_end_wall_s={e2e:.4f}\n"
        f"  total_wall_s__alias_of_step4_e2e={e2e:.4f}\n"
        f"  end_to_end_wall_s__alias_of_step4_e2e={e2e:.4f}\n"
        f"  epoch_time_scope=inference_only\n"
        f"  step4_total_scope=full_task\n"
        f"  preprocess_wall_s={preprocess_wall_s:.4f}\n"
        f"  inference_loop_wall_s={inference_loop_wall_s:.4f}\n"
        f"  decode_tail_wall_s={decode_tail_s:.4f}\n"
        f"  merge_wall_s={merge_wall_s:.4f}\n"
        f"  filter_wall_s={filter_wall_s:.4f}\n"
        f"  csv_write_wall_s={csv_write_wall_s:.4f}\n"
        f"  barrier_after_inference_wall_s={barrier_after_inference_wall_s:.4f}\n"
        f"  collective_gather_paths_wall_s={collective_gather_paths_wall_s:.4f}\n"
        f"  inference_share_pct={infer_pct}\n"
        f"  decode_share_pct={decode_pct}\n"
        f"  csv_share_pct={csv_pct}\n"
        f"  n_rows={n_rows}\n"
        f"  world_size={world_size}\n"
        f"  inference_only_avg_step_ms={inference_only_avg_step_ms:.4f}\n"
        f"  trainer_epoch_time_s={trainer_s_str}\n"
        f"  note=trainer_epoch_time_excludes_decode_merge_csv_tail"
    )
    append_log_dual(log_file, mach + "\n")

    trainer_h = (
        _format_seconds_human(float(trainer_epoch_time_s))
        if trainer_epoch_time_s is not None
        else "n/a"
    )
    human_lines = [
        f"Trainer epoch time shown above: {trainer_h} "
        f"(trainer_epoch_time_s={trainer_s_str} s; scope=inference_only / PerfMonitor epoch wall).",
        f"Actual step4 end-to-end time: {_format_seconds_human(e2e)} "
        f"(step4_end_to_end_wall_s={e2e:.4f} s; scope=full_task incl. preprocess, decode, merge, csv).",
    ]
    append_log_dual(log_file, "\n".join(f"[rank0] {_ln}" for _ln in human_lines) + "\n")

    block = (
        "\n========== Step 4 End-to-End Summary ==========\n"
        f"Task: {task_idx}\n"
        f"Rows: {n_rows}\n"
        f"World size: {world_size}\n"
        f"epoch_time_scope=inference_only | step4_total_scope=full_task\n"
        f"Trainer epoch time shown above: {trainer_h} (trainer_epoch_time_s={trainer_s_str} s)\n"
        f"Actual step4 end-to-end time: {_format_seconds_human(e2e)} (step4_end_to_end_wall_s={e2e:.2f} s)\n"
        f"Preprocess: {preprocess_wall_s:.2f} s ({_format_seconds_human(preprocess_wall_s)})\n"
        f"Inference loop: {inference_loop_wall_s:.2f} s ({_format_seconds_human(inference_loop_wall_s)})\n"
        f"trainer_epoch_time_s (PerfMonitor, inference-only): {trainer_s_str} s ({trainer_h})\n"
        f"Barrier after inference: {barrier_after_inference_wall_s:.2f} s\n"
        f"Decode tail (local tokenizer): {decode_tail_s:.2f} s ({_format_seconds_human(decode_tail_s)})\n"
        f"Merge (rank0 read partials + sort/validate): {merge_wall_s:.2f} s\n"
        f"RCR routing: {filter_wall_s:.2f} s\n"
        f"Collective gather paths: {collective_gather_paths_wall_s:.4f} s\n"
        f"CSV write: {csv_write_wall_s:.2f} s ({_format_seconds_human(csv_write_wall_s)})\n"
        f"step4_end_to_end_wall_s (primary total): {e2e:.2f} s ({_format_seconds_human(e2e)})\n"
        f"Inference / decode / CSV share of step4 e2e: {infer_pct}% / {decode_pct}% / {csv_pct}%\n"
        "Note: trainer_epoch_time_s matches log line 'Epoch 1 | time: Xm' scope (inference loop only).\n"
        "Warning: the 'Epoch 1 time: Xm' line is not the full step4 task duration.\n"
        "==============================================\n"
    )
    print(block, flush=True)
    append_log_dual(log_file, block)


def _decode_pred_token_rows(token_rows, chunk_size: int = 4096, progress_plog: Step4PerfLogger | None = None):
    """各 rank 本地 batch_decode：先一次性打成 (n, max_len) int64 再用 torch.from_numpy 按 chunk 解码。"""
    if not token_rows:
        return []
    cs = int(_step4_runtime_config().get("decode_chunk", chunk_size))
    if cs < 1:
        cs = chunk_size
    n = len(token_rows)
    _pid = get_odcr_text_tokenizer().pad_token_id
    pad_id = int(_pid) if _pid is not None else 0

    t_pack0 = time.perf_counter()
    max_len = 0
    for r in token_rows:
        lr = len(r)
        if lr > max_len:
            max_len = lr
    if max_len == 0:
        return [""] * n
    mat = np.full((n, max_len), pad_id, dtype=np.int64)
    for i, r in enumerate(token_rows):
        if not r:
            continue
        L = len(r)
        mat[i, :L] = r
    pack_wall_s = time.perf_counter() - t_pack0
    if progress_plog is not None:
        progress_plog.line(
            f"decode_token_pack_wall_s={pack_wall_s:.4f} decode_input_rows={n} max_seq_len={max_len}"
        )

    num_chunks = (n + cs - 1) // cs if n else 0
    log_every = 0
    out = [""] * n
    t_prog0 = time.perf_counter()
    for chunk_i, s in enumerate(range(0, n, cs)):
        e = min(s + cs, n)
        block = np.ascontiguousarray(mat[s:e])
        t = torch.from_numpy(block)
        decoded = get_odcr_text_tokenizer().batch_decode(t, skip_special_tokens=True)
        out[s:e] = decoded
        if progress_plog is not None and log_every > 0:
            done = chunk_i + 1
            if done % log_every == 0 or done == num_chunks:
                progress_plog.line(
                    f"decode_chunk_progress chunks={done}/{num_chunks} "
                    f"cum_wall_s={time.perf_counter() - t_prog0:.4f}"
                )
    return out


def _step4_rcr_latent_diagnostics(
    model,
    *,
    user_idx,
    item_idx,
    content_anchor,
    style_anchor,
    content_evidence_ids,
    style_evidence_ids,
    domain_style_anchor_ids,
    local_style_hint_ids,
    polarity_ids,
    evidence_quality_prior,
) -> dict[str, list[float]]:
    """Lightweight posterior diagnostics for Step4 RCR; no training graph is built."""
    target_domain = torch.ones_like(user_idx, dtype=torch.long, device=user_idx.device)
    auxiliary_domain = torch.zeros_like(user_idx, dtype=torch.long, device=user_idx.device)
    lat_target, _, _ = model._compute_latents(
        user_idx,
        item_idx,
        target_domain,
        content_anchor=content_anchor,
        style_anchor=style_anchor,
        content_evidence_ids=content_evidence_ids,
        style_evidence_ids=style_evidence_ids,
        domain_style_anchor_ids=domain_style_anchor_ids,
        local_style_hint_ids=local_style_hint_ids,
        polarity_ids=polarity_ids,
        evidence_quality_prior=evidence_quality_prior,
    )
    lat_cf, _, _ = model._compute_latents(
        user_idx,
        item_idx,
        auxiliary_domain,
        content_anchor=content_anchor,
        style_anchor=style_anchor,
        content_evidence_ids=content_evidence_ids,
        style_evidence_ids=style_evidence_ids,
        domain_style_anchor_ids=domain_style_anchor_ids,
        local_style_hint_ids=local_style_hint_ids,
        polarity_ids=polarity_ids,
        evidence_quality_prior=evidence_quality_prior,
    )
    rating_target = model.recommender(lat_target.shared).float()
    rating_cf = model.recommender(lat_cf.shared).float()
    rating_delta = torch.abs(rating_target - rating_cf)
    rating_stability = (1.0 - torch.clamp(rating_delta / 1.0, 0.0, 1.0)).float()
    shared_sim = ((F.cosine_similarity(lat_target.shared_proj.float(), lat_cf.shared_proj.float(), dim=-1) + 1.0) * 0.5)
    specific_shift = ((1.0 - F.cosine_similarity(lat_target.specific_proj.float(), lat_cf.specific_proj.float(), dim=-1)) * 0.5)
    return {
        "rating_target": rating_target.detach().cpu().numpy().tolist(),
        "rating_counterfactual": rating_cf.detach().cpu().numpy().tolist(),
        "rating_delta": rating_delta.detach().cpu().numpy().tolist(),
        "rating_stability_score": rating_stability.detach().cpu().numpy().tolist(),
        "shared_latent_similarity": shared_sim.clamp(0.0, 1.0).detach().cpu().numpy().tolist(),
        "specific_latent_shift": specific_shift.clamp(0.0, 1.0).detach().cpu().numpy().tolist(),
    }


def _run_one_task(
    task_idx: int,
    batch_size: int,
    nproc: int,
    rank: int,
    world_size: int,
    local_rank: int,
    log_file: str,
):
    _reject_step4_formal_bare_runtime_env()
    step4_e2e_start = time.perf_counter()
    task_config = TASK_DEFAULTS[task_idx]
    auxiliary = task_config["auxiliary"]
    target = task_config["target"]
    _task_ckpt_dir = get_stage_run_dir(task_idx)
    _step3_stage = (os.environ.get("ODCR_STEP3_RUN_DIR") or "").strip()
    _model_root = _step3_stage if _step3_stage else _task_ckpt_dir

    save_file = os.path.join(_model_root, "model", "best.pth")

    device = f"cuda:{local_rank}"
    if batch_size % world_size != 0:
        raise ValueError(
            f"step4 要求 global_eval_batch_size（当前 {batch_size}）能被 world_size={world_size} 整除，"
            "与 eval / step5 valid 相同的 strict 规则；请调整 configs/odcr.yaml 中 "
            "eval.profiles.<profile>.eval_batch_size 或 hardware.profiles.<profile>.ddp_world_size。"
        )
    local_batch = batch_size // world_size
    if rank == 0:
        _epn = (os.environ.get("ODCR_EVAL_PROFILE_NAME") or "").strip()
        print(
            "[step4 eval inference] "
            f"eval_profile_name={_epn!r} "
            f"global_eval_batch_size={batch_size} "
            f"eval_per_gpu_batch_size={local_batch} "
            f"world_size={world_size}",
            flush=True,
        )

    plog = Step4PerfLogger(log_file, task_idx, rank)

    path = os.path.join(get_merged_data_dir(), str(task_idx))
    aug_csv = os.path.join(path, "aug_train.csv")
    train_df = pd.read_csv(aug_csv)
    train_df["item"] = train_df["item"].astype(str)
    # 与 step3 / step5 一致：反事实编码仅使用有 explanation 的行
    train_df = train_df[train_df["explanation"].notna()].reset_index(drop=True)
    config = {
        "task_idx": task_idx,
        "device": device,
        "log_file": log_file,
        "save_file": save_file,
        "learning_rate": task_config["lr"],
        "epochs": 50,
        "batch_size": batch_size,
        "emsize": int(get_odcr_embed_dim()),
        "nlayers": 2,
        "nhid": 2048,
        "ntoken": len(get_odcr_text_tokenizer()),
        "dropout": 0.2,
        "coef": task_config["coef"],
        "nhead": 2,
    }

    dc, ds, uc, us, ic, ist, profile_meta = load_profile_tensors_dual_first(
        data_root=get_data_dir(),
        auxiliary_domain=auxiliary,
        target_domain=target,
        device_idx=device,
    )
    _em_prof = int(uc.shape[-1])
    if _em_prof != int(get_odcr_embed_dim()):
        raise ValueError(
            f"Step4 加载的 profile 隐层维度={_em_prof} 与 ODCR_EMBED_DIM={get_odcr_embed_dim()} 不一致；"
            "请重算 embeddings 或调整 ODCR_EMBED_DIM。"
        )
    config["emsize"] = _em_prof
    if rank == 0:
        print(
            f"[Step4] ODCR dual-channel profiles loaded ({profile_meta.get('profile_mode')}).",
            flush=True,
        )
    tuser_count = int(profile_meta["target_user_count"])
    suser_count = int(profile_meta["aux_user_count"])
    titem_count = int(profile_meta["target_item_count"])
    sitem_count = int(profile_meta["aux_item_count"])
    tuser_uc = uc[:tuser_count]
    suser_uc = uc[tuser_count:tuser_count + suser_count]
    titem_ic = ic[:titem_count]
    sitem_ic = ic[titem_count:titem_count + sitem_count]
    nuser = int(uc.shape[0])
    nitem = int(ic.shape[0])
    mx_u = int(train_df["user_idx"].max())
    mx_i = int(train_df["item_idx"].max())
    if mx_u >= nuser or mx_i >= nitem:
        raise ValueError(
            f"Step4 aug_train 索引越界: max user_idx={mx_u} max item_idx={mx_i}，"
            f"但 profile 拼接空间 nuser={nuser} nitem={nitem}（target+aux）。"
        )
    config["nuser"] = nuser
    config["nitem"] = nitem
    model = Model(
        config.get("nuser"),
        config.get("nitem"),
        config.get("ntoken"),
        config.get("emsize"),
        config.get("nhead"),
        config.get("nhid"),
        config.get("nlayers"),
        config.get("dropout"),
        uc,
        us,
        ic,
        ist,
        dc,
        ds,
    ).to(device)
    step3_lineage = _validate_step3_checkpoint_lineage_for_step4(
        checkpoint_path=str(config.get("save_file")),
        task_idx=int(task_idx),
        auxiliary=str(auxiliary),
        target=str(target),
        config=config,
    )
    _map = device if isinstance(device, str) else f"cuda:{device}"
    model.load_state_dict(torch.load(config.get("save_file"), map_location=_map, weights_only=True))
    _raw_dp = (os.environ.get("ODCR_DECODE_PROFILE_JSON") or "").strip()
    if _raw_dp:
        try:
            _dp = json.loads(_raw_dp)
        except json.JSONDecodeError:
            _dp = {}
        if isinstance(_dp, dict):
            model.decode_strategy = str(_dp.get("decode_strategy", "greedy")).strip().lower()
            model.generate_temperature = float(_dp.get("generate_temperature", 0.8))
            model.generate_top_p = float(_dp.get("generate_top_p", 0.9))
            model.repetition_penalty = float(_dp.get("repetition_penalty", 1.15))
            model.max_explanation_length = int(_dp.get("max_explanation_length", 25))
            model.no_repeat_ngram_size = int(_dp.get("no_repeat_ngram_size") or 0)
            model.min_len = int(_dp.get("min_len") or 0)
            model.soft_max_len = int(_dp.get("soft_max_len") or 0)
            model.hard_max_len = int(_dp.get("hard_max_len") or model.max_explanation_length)
            model.eos_boost_start = int(_dp.get("eos_boost_start", 9999))
            model.eos_boost_value = float(_dp.get("eos_boost_value", 0.0))
            model.tail_temperature = float(_dp.get("tail_temperature", -1.0))
            model.tail_top_p = float(_dp.get("tail_top_p", -1.0))
            model.decode_token_repeat_window = int(_dp.get("decode_token_repeat_window", 4))
            model.decode_token_repeat_max = int(_dp.get("decode_token_repeat_max", 2))
            model.domain_fusion_mode = str(_dp.get("domain_fusion_mode", "gate_cross_attn")).strip().lower()
            model.decoder_eos_id = int(getattr(get_odcr_text_tokenizer(), "eos_token_id", -1) or -1)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    target_df = train_df[train_df["domain"] == "target"].copy()
    target_df["domain"] = "auxiliary"
    target_df["sample_id"] = np.arange(len(target_df), dtype=np.int64)
    target_dataset = Dataset.from_pandas(target_df)
    processor = Processor(
        auxiliary,
        target,
        max_length=int(os.environ.get("ODCR_STEP3_TOKENIZER_MAX_LENGTH") or 0),
        evidence_length=int(os.environ.get("ODCR_STEP3_EVIDENCE_MAX_LENGTH") or 0),
    )
    proc_max_len = int(processor.max_length)

    cache_fingerprint = _step4_encoded_cache_fingerprint(
        task_idx,
        aug_csv,
        auxiliary,
        target,
        require_step5_text_model_dir(),
        proc_max_len,
        str(step3_lineage["lineage_hash"]),
    )
    cache_dir = _step4_encoded_cache_dir(task_idx, cache_fingerprint)
    plog.line(
        "step4_encode_cache_dir="
        f"{cache_dir} fingerprint={cache_fingerprint.get('fingerprint_hash')}"
    )

    perf = None
    if rank == 0:
        perf = PerfMonitor(
            device=local_rank,
            log_file=log_file,
            num_proc=nproc,
            test_num_workers=get_dataloader_num_workers("test"),
        )
        perf.start()

    # ---------- 阶段 preprocess：只验证/加载父进程 pre-DDP cache；DDP 内禁止 cold build ----------
    t_pre0 = time.perf_counter()
    plog.line("pre_ddp_cache_load_start cold_build_allowed=False")
    t_ld = time.perf_counter()
    cache_valid, cache_reason = _step4_encoded_cache_manifest_matches(
        cache_dir,
        expected_fingerprint=cache_fingerprint,
        expected_rows=len(target_df),
    )
    if cache_valid:
        encoded_data = load_from_disk(cache_dir)
    else:
        raise RuntimeError(
            "Step4 encoded cache is not ready before DDP inference. "
            f"reason={cache_reason} dir={cache_dir}. "
            "Run ./odcr step4 --task N --prepare-cache before formal Step4, "
            "or use ./odcr step4 --task N so the parent launcher prepares cache before torchrun."
        )
    if len(encoded_data) != len(target_df):
        raise RuntimeError(
            f"Step4 encoded cache row mismatch on rank {rank}: loaded={len(encoded_data)} expected={len(target_df)}"
        )
    cache_hit = True
    tokenize_wall = time.perf_counter() - t_ld
    barrier_preprocess = 0.0
    plog.line(f"pre_ddp_cache_load_done cache_hit=True load_wall_s={tokenize_wall:.4f} n_rows={len(encoded_data)}")

    encoded_data.set_format("torch")

    n_samples = len(encoded_data)
    full_dataset = TensorDataset(
        encoded_data["sample_id"],
        encoded_data["user_idx"],
        encoded_data["item_idx"],
        encoded_data["rating"],
        encoded_data["explanation_idx"],
        encoded_data["domain_idx"],
        encoded_data["content_anchor_score"],
        encoded_data["style_anchor_score"],
        encoded_data["content_evidence_ids"],
        encoded_data["style_evidence_ids"],
        encoded_data["domain_style_anchor_ids"],
        encoded_data["local_style_hint_ids"],
        encoded_data["polarity_ids"],
        encoded_data["evidence_quality_prior"],
    )

    chunk = (n_samples + world_size - 1) // world_size
    s = rank * chunk
    e = min(s + chunk, n_samples)
    test_dataset = Subset(full_dataset, list(range(s, e)))

    t_dl0 = time.perf_counter()
    test_nw = get_dataloader_num_workers("test")
    pin_mem = torch.cuda.is_available()
    _pf_test = get_dataloader_prefetch_factor(test_nw, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=local_batch,
        shuffle=False,
        num_workers=test_nw,
        pin_memory=pin_mem,
        persistent_workers=test_nw > 0,
        prefetch_factor=_pf_test,
    )
    dataloader_build_wall = time.perf_counter() - t_dl0

    preprocess_wall = time.perf_counter() - t_pre0
    _cache_log = f"cache_hit={cache_hit}" if rank == 0 else "replica_load_from_disk=True"
    plog.line(
        f"preprocess_summary preprocess_wall_s={preprocess_wall:.4f} "
        f"dataloader_build_wall_s={dataloader_build_wall:.4f} "
        f"tokenize_or_load_wall_s_rank={tokenize_wall:.4f} {_cache_log}"
    )

    _model = get_underlying_model(model)
    model.eval()
    local_pred_token_rows = []
    local_entropy_values = []
    local_row_indices = []
    local_rating_target = []
    local_rating_counterfactual = []
    local_rating_delta = []
    local_rating_stability = []
    local_shared_latent_similarity = []
    local_specific_latent_shift = []

    if rank == 0:
        perf.test_num_workers = test_nw
        perf.epoch_start()

    log_interval = max(1, int(_step4_runtime_config().get("perf_log_interval", 10)))
    t_prev_end = time.perf_counter()
    first_batch_wait = None
    step_idx = 0
    t_loop_start = time.perf_counter()

    with torch.no_grad():
        for batch in test_dataloader:
            t_batch_start = time.perf_counter()
            if first_batch_wait is None:
                first_batch_wait = t_batch_start - t_prev_end
                plog.line(f"first_batch_wait_s={first_batch_wait:.4f}")

            data_wait_s = t_batch_start - t_prev_end

            (
                batch_sample_id,
                user_idx,
                item_idx,
                rating,
                tgt_output,
                domain_idx,
                content_anchor_score,
                style_anchor_score,
                content_evidence_ids,
                style_evidence_ids,
                domain_style_anchor_ids,
                local_style_hint_ids,
                polarity_ids,
                evidence_quality_prior,
            ) = batch
            t_gather0 = time.perf_counter()
            gb = require_gathered_batch(
                _model.gather(
                    (
                        user_idx,
                        item_idx,
                        rating,
                        tgt_output,
                        domain_idx,
                        batch_sample_id,
                        content_anchor_score,
                        style_anchor_score,
                        content_evidence_ids,
                        style_evidence_ids,
                        domain_style_anchor_ids,
                        local_style_hint_ids,
                        polarity_ids,
                        evidence_quality_prior,
                    ),
                    device,
                )
            )
            user_idx = gb.user_idx
            item_idx = gb.item_idx
            rating = gb.rating
            tgt_output = gb.tgt_output
            domain_idx = gb.domain_idx
            ca = gb.content_anchor_score
            sa = gb.style_anchor_score
            ce = gb.content_evidence_ids
            se = gb.style_evidence_ids
            dsa = gb.domain_style_anchor_ids
            lsh = gb.local_style_hint_ids
            pol = gb.polarity_ids
            eq = gb.evidence_quality_prior
            gather_h2d_s = time.perf_counter() - t_gather0

            if any(x is None for x in (ca, sa, ce, se, dsa, lsh, pol, eq)):
                raise RuntimeError("Step4 gather 缺少 canonical evidence 张量，无法执行 RCR latent-aware routing。")

            t_gen0 = time.perf_counter()
            with odcr_cuda_bf16_autocast():
                pred_exps, entropy = _model.generate(
                    user_idx,
                    item_idx,
                    domain_idx,
                    content_anchor=ca,
                    style_anchor=sa,
                    content_evidence_ids=ce,
                    style_evidence_ids=se,
                    domain_style_anchor_ids=dsa,
                    local_style_hint_ids=lsh,
                    polarity_ids=pol,
                    evidence_quality_prior=eq,
                )
                rcr_diag = _step4_rcr_latent_diagnostics(
                    _model,
                    user_idx=user_idx,
                    item_idx=item_idx,
                    content_anchor=ca,
                    style_anchor=sa,
                    content_evidence_ids=ce,
                    style_evidence_ids=se,
                    domain_style_anchor_ids=dsa,
                    local_style_hint_ids=lsh,
                    polarity_ids=pol,
                    evidence_quality_prior=eq,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generate_wall_s = time.perf_counter() - t_gen0

            t_ent0 = time.perf_counter()
            ent_cpu = entropy.detach().cpu()
            ent_list = ent_cpu.numpy().tolist()
            entropy_sync_wall_s = time.perf_counter() - t_ent0

            decode_wall_s = 0.0
            local_row_indices.extend(batch_sample_id.detach().cpu().reshape(-1).tolist())
            local_pred_token_rows.extend(pred_exps.detach().cpu().tolist())
            local_entropy_values.extend(ent_list)
            local_rating_target.extend(rcr_diag["rating_target"])
            local_rating_counterfactual.extend(rcr_diag["rating_counterfactual"])
            local_rating_delta.extend(rcr_diag["rating_delta"])
            local_rating_stability.extend(rcr_diag["rating_stability_score"])
            local_shared_latent_similarity.extend(rcr_diag["shared_latent_similarity"])
            local_specific_latent_shift.extend(rcr_diag["specific_latent_shift"])

            t_end = time.perf_counter()
            step_wall_s = t_end - t_batch_start
            bsz = int(user_idx.size(0))
            samples_per_sec = bsz / step_wall_s if step_wall_s > 0 else 0.0
            peak_mb = (
                torch.cuda.max_memory_allocated(local_rank) / (1024**2) if torch.cuda.is_available() else 0.0
            )

            step_idx += 1
            if step_idx % log_interval == 0 or step_idx == 1:
                plog.line(
                    f"step={step_idx} step_wall_s={step_wall_s:.4f} data_wait_s={data_wait_s:.4f} "
                    f"gather_h2d_s={gather_h2d_s:.4f} generate_wall_s={generate_wall_s:.4f} "
                    f"decode_wall_s={decode_wall_s:.4f} entropy_sync_wall_s={entropy_sync_wall_s:.4f} "
                    f"samples_per_sec={samples_per_sec:.2f} max_mem_alloc_MB={peak_mb:.1f}"
                )

            t_prev_end = t_end

    inference_loop_wall_s = time.perf_counter() - t_loop_start
    plog.line(f"inference_loop_wall_s={inference_loop_wall_s:.4f} steps={step_idx}")
    if first_batch_wait is None:
        plog.line("first_batch_wait_s=n/a (empty_dataloader)")

    t_barrier1 = time.perf_counter()
    dist.barrier()
    barrier_after_loop_s = time.perf_counter() - t_barrier1
    plog.line(f"barrier_after_inference wall_s={barrier_after_loop_s:.4f}")

    if len(local_row_indices) != len(local_pred_token_rows) or len(local_entropy_values) != len(
        local_pred_token_rows
    ):
        raise RuntimeError(
            f"rank{rank} 本地 row_idx / token / entropy 条数不一致: "
            f"{len(local_row_indices)} {len(local_pred_token_rows)} {len(local_entropy_values)}"
        )
    if len(local_rating_delta) != len(local_pred_token_rows):
        raise RuntimeError(
            f"rank{rank} 本地 RCR latent diagnostics 条数不一致: "
            f"{len(local_rating_delta)} {len(local_pred_token_rows)}"
        )

    # PerfMonitor 仅统计推理循环，避免尾部 decode/merge 拉长“单步均值”
    trainer_epoch_time_s: float | None = None
    if rank == 0:
        _rec = perf.epoch_end(1, len(test_dataloader), emit_log=True)
        trainer_epoch_time_s = float(_rec["epoch_time"])
        t_pf0 = time.perf_counter()
        perf.finish()
        rank0_perf_finish_wall_s = time.perf_counter() - t_pf0
        plog.line(f"rank0_perf_finish_wall_s={rank0_perf_finish_wall_s:.4f}")
        _epoch_scope_note = (
            "Note: Epoch time above is trainer/PerfMonitor wall time and excludes "
            "step4 decode/merge/csv tail."
        )
        print(_epoch_scope_note, flush=True)
        append_log_dual(log_file, _epoch_scope_note + "\n")

    partial_dir = os.path.join(_task_ckpt_dir, "step4_partials")
    os.makedirs(partial_dir, exist_ok=True)
    partial_base = os.path.join(
        partial_dir, f"step4_partial_task{task_idx}_rank{rank}"
    )
    fmt_choice = _step4_partial_format_choice()
    partial_suffix, partial_kind = _step4_partial_suffix_and_kind(fmt_choice)
    if fmt_choice == "parquet" and partial_kind == "csv":
        plog.line("step4_partial_fallback_csv reason=pyarrow_missing")
    partial_path = partial_base + partial_suffix

    _prev_torch_threads = torch.get_num_threads()
    try:
        _nti = int(_step4_runtime_config().get("decode_threads", 0))
        if _nti > 0:
            torch.set_num_threads(_nti)
            plog.line(f"decode_torch_num_threads={_nti}")
        t_dec_local0 = time.perf_counter()
        local_explanations = _decode_pred_token_rows(local_pred_token_rows, progress_plog=plog)
        decode_local_wall_s = time.perf_counter() - t_dec_local0
    finally:
        torch.set_num_threads(_prev_torch_threads)

    n_loc = len(local_explanations)
    _dchunk = int(_step4_runtime_config().get("decode_chunk", 4096))
    _nchunks = (n_loc + _dchunk - 1) // _dchunk if n_loc else 0
    plog.line(
        f"decode_local_wall_s={decode_local_wall_s:.4f} decode_input_rows={n_loc} "
        f"decode_chunk_size={_dchunk} decode_num_chunks={_nchunks}"
    )

    t_before_partial_phase = time.perf_counter()
    t_prep0 = time.perf_counter()
    part_df = pd.DataFrame(
        {
            "row_idx": local_row_indices,
            "entropy": local_entropy_values,
            "explanation": local_explanations,
            "rating_target": local_rating_target,
            "rating_counterfactual": local_rating_counterfactual,
            "rating_delta": local_rating_delta,
            "rating_stability_score": local_rating_stability,
            "shared_latent_similarity": local_shared_latent_similarity,
            "specific_latent_shift": local_specific_latent_shift,
        }
    )
    partial_prep_wall_s = time.perf_counter() - t_prep0
    plog.line(f"partial_df_prep_wall_s={partial_prep_wall_s:.4f}")

    t_pw0 = time.perf_counter()
    try:
        _step4_write_partial_df(part_df, partial_path, partial_kind)
        partial_manifest_path = _step4_write_partial_manifest(
            partial_path=partial_path,
            rank=rank,
            world_size=world_size,
            row_count=len(part_df),
            kind=partial_kind,
        )
    except Exception as exc:
        with open(_step4_partial_failed_marker(partial_path), "w", encoding="utf-8") as handle:
            handle.write(f"rank={rank}\nfailed_at={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n{exc}\n")
        raise
    partial_write_wall_s = time.perf_counter() - t_pw0
    plog.line(
        f"partial_write_wall_s={partial_write_wall_s:.4f} partial_kind={partial_kind} "
        f"path={partial_path} manifest={partial_manifest_path}"
    )

    t_gather_paths0 = time.perf_counter()
    pre_gather_wait_wall_s = t_gather_paths0 - t_before_partial_phase
    dist.barrier()
    collective_gather_paths_wall_s = time.perf_counter() - t_gather_paths0
    plog.line(f"pre_gather_wait_wall_s={pre_gather_wait_wall_s:.4f}")
    plog.line(f"partial_manifest_barrier_wall_s={collective_gather_paths_wall_s:.4f}")
    plog.line("barrier_before_csv_removed=True")
    plog.line("destroy_process_group_before_cpu_export=True")
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        del model
        del _model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if rank != 0:
        plog.line("non_rank0_gpu_released_before_cpu_tail=True")
        return

    decode_merge_rank0_wall_s = 0.0
    rank0_read_partials_wall_s = 0.0
    rank0_sort_validate_wall_s = 0.0
    rank0_filter_wall_s = 0.0
    csv_write_wall_s = 0.0

    if rank == 0:
        wait_timeout = int(_step4_runtime_config().get("partial_wait_timeout_seconds", 600))
        partial_manifests = _step4_wait_for_partial_manifests(
            partial_dir,
            world_size=world_size,
            timeout_s=wait_timeout,
        )
        t_read0 = time.perf_counter()
        dfs = []
        for item in partial_manifests:
            pth = str(item["path"])
            dfs.append(_step4_read_partial_df(pth))
        rank0_read_partials_wall_s = time.perf_counter() - t_read0
        plog.line(f"rank0_read_partials_wall_s={rank0_read_partials_wall_s:.4f} n_files={len(dfs)}")

        t_sv0 = time.perf_counter()
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        if len(merged) != n_samples:
            raise RuntimeError(
                f"rank0 合并后行数 {len(merged)} 与 target 样本数 {n_samples} 不一致"
            )
        idx_arr = merged["row_idx"].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(idx_arr, np.arange(n_samples, dtype=np.int64)):
            raise RuntimeError("rank0 合并后 row_idx 未覆盖 0..n-1 或存在重复/缺失")
        entropy_values = merged["entropy"].astype(float).tolist()
        prediction_exps = merged["explanation"].astype(str).tolist()
        rank0_sort_validate_wall_s = time.perf_counter() - t_sv0
        plog.line(f"rank0_sort_validate_wall_s={rank0_sort_validate_wall_s:.4f} n={len(merged)}")

        decode_merge_rank0_wall_s = rank0_read_partials_wall_s + rank0_sort_validate_wall_s
        plog.line(f"decode_merge_rank0_wall_s={decode_merge_rank0_wall_s:.4f}")

        t_filt0 = time.perf_counter()
        cf_candidate_df = target_df.copy().reset_index(drop=True)
        cf_candidate_df["explanation"] = prediction_exps
        cf_candidate_df["entropy"] = entropy_values
        for _rcr_col in (
            "rating_target",
            "rating_counterfactual",
            "rating_delta",
            "rating_stability_score",
            "shared_latent_similarity",
            "specific_latent_shift",
        ):
            cf_candidate_df[_rcr_col] = merged[_rcr_col].astype(float).to_numpy(copy=False)
        rcr_config = ODCFRoutingConfig.from_env(require=True)
        routed_cf_df = attach_odcr_cf_routing(target_df, cf_candidate_df, cfg=rcr_config)
        rank0_filter_wall_s = time.perf_counter() - t_filt0
        plog.line(f"rank0_filter_wall_s={rank0_filter_wall_s:.4f} (odcr reliability routing)")

        t_csv0 = time.perf_counter()
        final_df = assemble_step4_training_table(train_df, routed_cf_df, rcr_config=rcr_config)
        os.makedirs(_task_ckpt_dir, exist_ok=True)
        _lineage = parse_training_run_lineage(_task_ckpt_dir)
        _tid = int(_lineage.get("task_id") or task_idx)
        _iter = str(_lineage.get("iteration_id") or "unknown")
        _slug = str(_lineage.get("step4_run") or os.path.basename(os.path.abspath(_task_ckpt_dir)))
        _train_csv_out = os.path.join(_task_ckpt_dir, ODCR_ROUTING_TRAIN_CSV)
        _valid_csv_p = os.path.join(get_data_dir(), target, "valid.csv")
        _test_csv_p = os.path.join(get_data_dir(), target, "test.csv")
        _ic = build_index_contract(
            task_id=_tid,
            iteration_id=_iter,
            step4_run=_slug,
            auxiliary_domain=auxiliary,
            target_domain=target,
            data_root=get_data_dir(),
            train_csv_path=_train_csv_out,
            valid_csv_path=_valid_csv_p,
            test_csv_path=_test_csv_p,
            target_user_count=int(tuser_uc.shape[0]),
            aux_user_count=int(suser_uc.shape[0]),
            target_item_count=int(titem_ic.shape[0]),
            aux_item_count=int(sitem_ic.shape[0]),
        )
        _step4_export_lineage = build_step4_export_lineage(
            task_id=_tid,
            auxiliary_domain=auxiliary,
            target_domain=target,
            step3_checkpoint_lineage_hash=str(step3_lineage["lineage_hash"]),
            step4_rcr_config=rcr_config.to_dict(),
            step4_run=_slug,
            frozen_step3_lineage={
                "upstream_step3_run_id": str(os.path.basename(os.path.abspath(_model_root))),
                "step3_checkpoint_path": os.path.abspath(str(config.get("save_file"))),
                "step3_checkpoint_hash": _step4_file_sha256(str(config.get("save_file"))),
                "step3_checkpoint_lineage_hash": str(step3_lineage["lineage_hash"]),
                "step3_stage_status_hash": _step4_upstream_artifact_hash("status_path") or _step4_upstream_artifact_hash("stage_status"),
                "step3_eval_handoff_hash": _step4_upstream_artifact_hash("eval_handoff"),
            },
        )
        _ic["step4_export_lineage"] = _step4_export_lineage
        _ic_path = os.path.join(_task_ckpt_dir, INDEX_CONTRACT_FILENAME)
        _ic_summary = {
            "nuser_global": _ic["nuser_global"],
            "nitem_global": _ic["nitem_global"],
            "train_index_space": _ic["train_index_space"],
            "valid_index_space": _ic["valid_index_space"],
            "test_index_space": _ic["test_index_space"],
            "step4_export_lineage_hash": _step4_export_lineage["lineage_hash"],
        }
        manifest = build_step4_train_manifest(
            final_df,
            n_cf_candidate_input=int(n_samples),
            n_cf_rcr_scorer_kept=int((routed_cf_df["route_scorer"].astype(int) == 1).sum()),
            rcr_config=rcr_config,
            index_contract_path=_ic_path,
            index_contract_summary=_ic_summary,
            lineage=_step4_export_lineage,
        )
        manifest["partial_artifacts"] = partial_manifests
        csv_out, man_out, ic_out = write_step4_training_artifacts(
            final_df, manifest, _task_ckpt_dir, index_contract=_ic
        )
        csv_write_wall_s = time.perf_counter() - t_csv0
        print(f"Task {task_idx}: 已写入 {csv_out}", flush=True)
        print(f"Task {task_idx}: manifest {man_out}", flush=True)
        print(f"Task {task_idx}: index_contract {ic_out}", flush=True)
        append_log_dual(
            log_file,
            f"[rank0] step4_train_table rows={len(final_df)} manifest={man_out}\n",
        )
        plog.line(f"csv_write_wall_s={csv_write_wall_s:.4f}")

        plog.line("partial_artifacts_retained_for_readiness_validator=True")

    decode_tail_wall_s = decode_local_wall_s + (
        (decode_merge_rank0_wall_s + rank0_filter_wall_s) if rank == 0 else 0.0
    )
    inf_avg_ms = (inference_loop_wall_s / step_idx * 1000.0) if step_idx else 0.0
    step4_end_to_end_wall_s = time.perf_counter() - step4_e2e_start

    if rank == 0:
        _log_step4_final_summary(
            task_idx=task_idx,
            world_size=world_size,
            n_rows=n_samples,
            log_file=log_file,
            step4_end_to_end_wall_s=step4_end_to_end_wall_s,
            preprocess_wall_s=preprocess_wall,
            inference_loop_wall_s=inference_loop_wall_s,
            decode_local_wall_s=decode_local_wall_s,
            merge_wall_s=decode_merge_rank0_wall_s,
            filter_wall_s=rank0_filter_wall_s,
            csv_write_wall_s=csv_write_wall_s,
            barrier_after_inference_wall_s=barrier_after_loop_s,
            collective_gather_paths_wall_s=collective_gather_paths_wall_s,
            trainer_epoch_time_s=trainer_epoch_time_s,
            inference_only_avg_step_ms=inf_avg_ms,
        )

    plog.line(
        f"step4_perf_summary "
        f"inference_loop_wall_s={inference_loop_wall_s:.4f} "
        f"decode_tail_wall_s={decode_tail_wall_s:.4f} "
        f"csv_write_wall_s={csv_write_wall_s:.4f} "
        f"step4_end_to_end_wall_s={step4_end_to_end_wall_s:.4f} "
        f"total_wall_s__alias_of_step4_e2e={step4_end_to_end_wall_s:.4f} "
        f"inference_only_avg_step_ms={inf_avg_ms:.4f} "
        f"note=primary_total_is_step4_end_to_end_wall_s_full_task_preprocess_through_csv"
    )

    plog.line("step4_task_done")
