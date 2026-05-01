from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from data_contract import (
    CONTENT_PROFILE_TEXT_COLUMNS,
    PREPROCESS_CONTRACT_VERSION,
    STYLE_PROFILE_TEXT_COLUMNS,
    assert_no_deprecated_preprocess_detail_columns,
    expected_preprocess_column_order,
    preprocess_csv_dtype_map,
    render_preprocess_contract_snapshot,
)
from odcr_core.file_atomic import atomic_save_numpy
from odcr_core.training_checkpoint import model_artifact_fingerprint

log = logging.getLogger("compute_embeddings")

ALL_DATASETS = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]

CONTENT_TEXT_COLS = CONTENT_PROFILE_TEXT_COLUMNS
STYLE_TEXT_COLS = STYLE_PROFILE_TEXT_COLUMNS
GROUPED_TEXT_CACHE_CODE_VERSION = "preprocess_b_grouped_text_semantics/1.0"
GROUPED_TEXT_CACHE_FORMAT_VERSION = "grouped_text_shards_npz/1"
_SOURCE_FILE_FINGERPRINT_CACHE: dict[tuple[str, int, int], str] = {}
GPU_TMUX_HINT = "tmux -L odcr_gpu new-session -A -s odcr"


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    output_name: str
    group_col: str
    entity_kind: str
    text_kind: str
    column_names: tuple[str, ...]


PROFILE_SPECS = (
    ProfileSpec(
        name="user_content",
        output_name="user_content_profiles.npy",
        group_col="user_idx",
        entity_kind="user",
        text_kind="content",
        column_names=CONTENT_TEXT_COLS,
    ),
    ProfileSpec(
        name="user_style",
        output_name="user_style_profiles.npy",
        group_col="user_idx",
        entity_kind="user",
        text_kind="style",
        column_names=STYLE_TEXT_COLS,
    ),
    ProfileSpec(
        name="item_content",
        output_name="item_content_profiles.npy",
        group_col="item_idx",
        entity_kind="item",
        text_kind="content",
        column_names=CONTENT_TEXT_COLS,
    ),
    ProfileSpec(
        name="item_style",
        output_name="item_style_profiles.npy",
        group_col="item_idx",
        entity_kind="item",
        text_kind="style",
        column_names=STYLE_TEXT_COLS,
    ),
)
PROFILE_SPEC_BY_NAME = {spec.name: spec for spec in PROFILE_SPECS}


@dataclass(frozen=True)
class ProbeConfig:
    probe_only: bool
    max_groups_per_spec: int | None
    max_batches_per_spec: int | None


@dataclass
class PhaseTiming:
    csv_read_s: float = 0.0
    group_text_build_s: float = 0.0
    group_text_cache_fingerprint_s: float = 0.0
    group_text_cache_load_s: float = 0.0
    group_text_cache_write_s: float = 0.0
    tokenize_s: float = 0.0
    gpu_forward_s: float = 0.0
    write_npy_s: float = 0.0
    verify_extra_s: float = 0.0
    total_s: float = 0.0
    rows_read: int = 0
    read_chunks: int = 0
    groups_total: int = 0
    groups_finalized: int = 0
    groups_encoded: int = 0
    group_shards: int = 0
    batches_encoded: int = 0
    verify_sample_count: int = 0
    verify_missing_sample_count: int = 0
    group_text_cache_status: str = "disabled"
    group_text_cache_key: str | None = None
    group_text_cache_shards_loaded: int = 0
    group_text_cache_shards_written: int = 0
    group_text_cache_groups_written: int = 0
    first_tokenize_from_spec_start_s: float | None = None
    first_forward_from_spec_start_s: float | None = None
    first_tokenize_from_dataset_start_s: float | None = None
    first_forward_from_dataset_start_s: float | None = None

    def mark_first_tokenize(self, *, spec_started_at: float, dataset_started_at: float) -> None:
        if self.first_tokenize_from_spec_start_s is not None:
            return
        now = perf_counter()
        self.first_tokenize_from_spec_start_s = now - spec_started_at
        self.first_tokenize_from_dataset_start_s = now - dataset_started_at

    def mark_first_forward(self, *, spec_started_at: float, dataset_started_at: float) -> None:
        if self.first_forward_from_spec_start_s is not None:
            return
        now = perf_counter()
        self.first_forward_from_spec_start_s = now - spec_started_at
        self.first_forward_from_dataset_start_s = now - dataset_started_at

    def to_dict(self) -> dict[str, object]:
        return {
            "csv_read_s": _round_timing(self.csv_read_s),
            "group_text_build_s": _round_timing(self.group_text_build_s),
            "group_text_cache_fingerprint_s": _round_timing(self.group_text_cache_fingerprint_s),
            "group_text_cache_load_s": _round_timing(self.group_text_cache_load_s),
            "group_text_cache_write_s": _round_timing(self.group_text_cache_write_s),
            "tokenize_s": _round_timing(self.tokenize_s),
            "gpu_forward_s": _round_timing(self.gpu_forward_s),
            "write_npy_s": _round_timing(self.write_npy_s),
            "verify_extra_s": _round_timing(self.verify_extra_s),
            "total_s": _round_timing(self.total_s),
            "rows_read": int(self.rows_read),
            "read_chunks": int(self.read_chunks),
            "groups_total": int(self.groups_total),
            "groups_finalized": int(self.groups_finalized),
            "groups_encoded": int(self.groups_encoded),
            "group_shards": int(self.group_shards),
            "batches_encoded": int(self.batches_encoded),
            "verify_sample_count": int(self.verify_sample_count),
            "verify_missing_sample_count": int(self.verify_missing_sample_count),
            "group_text_cache_status": self.group_text_cache_status,
            "group_text_cache_key": self.group_text_cache_key,
            "group_text_cache_shards_loaded": int(self.group_text_cache_shards_loaded),
            "group_text_cache_shards_written": int(self.group_text_cache_shards_written),
            "group_text_cache_groups_written": int(self.group_text_cache_groups_written),
            "first_tokenize_from_spec_start_s": _round_optional_timing(self.first_tokenize_from_spec_start_s),
            "first_forward_from_spec_start_s": _round_optional_timing(self.first_forward_from_spec_start_s),
            "first_tokenize_from_dataset_start_s": _round_optional_timing(
                self.first_tokenize_from_dataset_start_s
            ),
            "first_forward_from_dataset_start_s": _round_optional_timing(
                self.first_forward_from_dataset_start_s
            ),
        }


@dataclass
class SpecGroupAccumulator:
    spec: ProfileSpec
    order_keys: list[int]
    fragments_by_group: dict[int, list[list[str]]]
    target_size: int
    rows_read: int
    read_chunks: int

    @property
    def group_count(self) -> int:
        return len(self.order_keys)

    def active_indices(self) -> np.ndarray:
        return np.asarray(self.order_keys, dtype=np.int64)

    def build_text(self, group_idx: int) -> str:
        fragments = self.fragments_by_group[int(group_idx)]
        return " ".join(piece for column_fragments in fragments for piece in column_fragments if piece)

    def build_texts(self, group_indices: np.ndarray) -> list[str]:
        return [self.build_text(int(group_idx)) for group_idx in group_indices.tolist()]


@dataclass(frozen=True)
class GroupTextShard:
    shard_id: int
    group_indices: np.ndarray
    texts: list[str]


@dataclass(frozen=True)
class GroupedTextCacheConfig:
    enabled: bool
    cache_dir: str
    version: str


@dataclass(frozen=True)
class GroupedTextCacheRequest:
    cache_dir: Path
    dataset: str
    spec: ProfileSpec
    source_path: Path
    cache_key: str
    key_payload: dict[str, object]
    entry_dir: Path
    manifest_path: Path
    stale_candidates: int


@dataclass(frozen=True)
class GroupedTextCacheEntry:
    request: GroupedTextCacheRequest
    manifest: dict[str, object]
    target_size: int
    group_count: int
    rows_read: int
    read_chunks: int
    shards: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class PrecisionConfig:
    bf16_enabled: bool
    tf32_enabled: bool
    autocast_enabled: bool
    autocast_dtype: torch.dtype


def _install_resolved_preprocess_context(args: argparse.Namespace) -> None:
    required = {
        "--data-dir": getattr(args, "data_dir", None),
        "--models-dir": getattr(args, "models_dir", None),
        "--sentence-embed-model": getattr(args, "sentence_embed_model", None),
        "--embed-dim": getattr(args, "embed_dim", None),
    }
    missing = [flag for flag, value in required.items() if not str(value or "").strip()]
    if missing:
        raise RuntimeError(
            "compute_embeddings.py must be launched with the resolved preprocess payload from ./odcr; "
            f"missing {missing}. Refusing to read configs/odcr.yaml as a child-side fallback."
        )
    if int(args.embed_dim) <= 0:
        raise RuntimeError("--embed-dim must be a positive resolved env.embed_dim value.")
    local_files_only = str(os.environ.get("ODCR_RESOLVED_LOCAL_FILES_ONLY") or "").strip().lower()
    if local_files_only not in ("1", "true", "yes", "on"):
        raise RuntimeError("compute_embeddings.py requires ODCR_RESOLVED_LOCAL_FILES_ONLY=1 for formal preprocess_b.")
    os.environ["ODCR_RESOLVED_DATA_DIR"] = os.path.abspath(os.path.expanduser(str(args.data_dir)))
    os.environ["ODCR_RESOLVED_MODELS_DIR"] = os.path.abspath(os.path.expanduser(str(args.models_dir)))
    os.environ["ODCR_RESOLVED_SENTENCE_EMBED_MODEL"] = os.path.abspath(
        os.path.expanduser(str(args.sentence_embed_model))
    )
    os.environ["ODCR_RESOLVED_EMBED_DIM"] = str(int(args.embed_dim))


def _resolved_env_path(name: str) -> str:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        raise RuntimeError(f"{name} is required from the resolved preprocess payload.")
    return os.path.abspath(os.path.expanduser(raw))


def _resolved_data_dir() -> str:
    return _resolved_env_path("ODCR_RESOLVED_DATA_DIR")


def _require_sentence_embed_model_dir() -> str:
    path = _resolved_env_path("ODCR_RESOLVED_SENTENCE_EMBED_MODEL")
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Resolved env.sentence_embed_model does not exist: {path}. "
            "Refusing fallback to configs/odcr.yaml or Hugging Face Hub."
        )
    return path


def _resolved_embed_dim() -> int:
    raw = str(os.environ.get("ODCR_RESOLVED_EMBED_DIM") or "").strip()
    if not raw:
        raise RuntimeError("ODCR_RESOLVED_EMBED_DIM is required from the resolved preprocess payload.")
    value = int(raw)
    if value <= 0:
        raise RuntimeError("ODCR_RESOLVED_EMBED_DIM must be positive.")
    return value


def _select_cuda_device_or_fail(args: argparse.Namespace) -> torch.device:
    if torch.cuda.is_available():
        device_idx = 0 if args.cuda_device is None else int(args.cuda_device)
        device_count = int(torch.cuda.device_count())
        if device_idx < 0 or device_idx >= device_count:
            raise RuntimeError(
                f"preprocess_b requested cuda:{device_idx}, but torch sees {device_count} visible CUDA device(s). "
                f"Use the GPU tmux session: {GPU_TMUX_HINT}"
            )
        return torch.device(f"cuda:{device_idx}")
    if bool(getattr(args, "allow_cpu_debug", False)):
        log.warning("preprocess_b explicit --allow-cpu-debug enabled; using CPU debug mode without formal admission.")
        return torch.device("cpu")
    raise RuntimeError(
        "preprocess_b requires CUDA before loading BGE-large; torch.cuda.is_available() is false. "
        f"Use the GPU tmux session: {GPU_TMUX_HINT}"
    )


def _round_timing(value: float) -> float:
    return round(float(value), 6)


def _round_optional_timing(value: float | None) -> float | None:
    if value is None:
        return None
    return _round_timing(value)


def _safe_join(values) -> str:
    return " ".join(str(v) for v in values if pd.notna(v) and str(v).strip())


def _require_resolved_bool_flag(args: argparse.Namespace, name: str, cli_names: str) -> bool:
    raw = getattr(args, name, None)
    if raw is None:
        raise ValueError(f"{cli_names} is required from the resolved preprocess_b runtime transport.")
    return bool(raw)


def _resolve_datasets(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(ALL_DATASETS)
    datasets = [d.strip() for d in str(raw).split(",") if d.strip()]
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; expected one of {ALL_DATASETS}")
    return datasets


def _resolve_profile_specs(raw: str | None) -> tuple[ProfileSpec, ...]:
    if raw is None or not str(raw).strip():
        return PROFILE_SPECS
    seen: set[str] = set()
    specs: list[ProfileSpec] = []
    for token in str(raw).split(","):
        name = token.strip()
        if not name or name in seen:
            continue
        spec = PROFILE_SPEC_BY_NAME.get(name)
        if spec is None:
            raise ValueError(
                f"Unknown profile spec: {name!r}; expected one of {list(PROFILE_SPEC_BY_NAME.keys())}"
            )
        seen.add(name)
        specs.append(spec)
    if not specs:
        raise ValueError("--specs resolved to an empty selection")
    return tuple(specs)


def _parse_optional_indices(raw: str | None, *, label: str) -> list[int] | None:
    if raw is None or not str(raw).strip():
        return None
    seen: set[int] = set()
    ordered: list[int] = []
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        if not item.isdigit():
            raise ValueError(f"{label} must be a comma-separated list of non-negative integers: {raw!r}")
        idx = int(item)
        if idx not in seen:
            seen.add(idx)
            ordered.append(idx)
    if not ordered:
        raise ValueError(f"{label} resolved to an empty index list")
    return ordered


def _resolve_probe_config(args: argparse.Namespace) -> ProbeConfig:
    probe_only = bool(getattr(args, "probe_only", False))
    max_groups = getattr(args, "probe_max_groups_per_spec", None)
    if max_groups is not None:
        max_groups = int(max_groups)
        if max_groups <= 0:
            raise ValueError("probe_max_groups_per_spec must be positive when provided")
        if not probe_only:
            raise ValueError("--probe-max-groups-per-spec requires --probe-only")

    max_batches = getattr(args, "probe_max_batches_per_spec", None)
    if max_batches is not None:
        max_batches = int(max_batches)
        if max_batches <= 0:
            raise ValueError("probe_max_batches_per_spec must be positive when provided")
        if not probe_only:
            raise ValueError("--probe-max-batches-per-spec requires --probe-only")

    return ProbeConfig(
        probe_only=probe_only,
        max_groups_per_spec=max_groups,
        max_batches_per_spec=max_batches,
    )


def _resolve_precision_config(args: argparse.Namespace, device: torch.device) -> PrecisionConfig:
    bf16_enabled = _require_resolved_bool_flag(args, "bf16_enabled", "--bf16/--no-bf16")
    tf32_enabled = _require_resolved_bool_flag(args, "tf32_enabled", "--tf32/--no-tf32")
    autocast_enabled = bool(device.type == "cuda" and bf16_enabled and torch.cuda.is_bf16_supported())
    if bf16_enabled and device.type != "cuda":
        log.info("bf16 requested but CUDA is unavailable; preprocess_b will run in fp32.")
    elif bf16_enabled and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        log.info("bf16 requested but torch.cuda.is_bf16_supported() is False; preprocess_b will run in fp32.")
    return PrecisionConfig(
        bf16_enabled=bf16_enabled,
        tf32_enabled=tf32_enabled,
        autocast_enabled=autocast_enabled,
        autocast_dtype=torch.bfloat16,
    )


def _configure_tf32(device: torch.device, *, enabled: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    log.info("preprocess_b TF32 matmul/cudnn = %s", "ON" if enabled else "OFF")


def _aggregate_group_text_series(
    df: pd.DataFrame, group_col: str, column_names: tuple[str, ...]
) -> pd.Series:
    if df.empty or group_col not in df.columns:
        return pd.Series(dtype=object)

    gids = df[group_col].to_numpy()
    order_keys: list[int] = []
    seen: set[int] = set()
    for g in gids:
        if g not in seen:
            seen.add(int(g))
            order_keys.append(int(g))

    idx_by_gid: dict[int, list[int]] = {g: [] for g in order_keys}
    for pos, gid in enumerate(gids):
        idx_by_gid[int(gid)].append(pos)

    texts: list[str] = []
    for gid in order_keys:
        row_positions = idx_by_gid[gid]
        vals: list[object] = []
        for column_name in column_names:
            if column_name not in df.columns:
                continue
            col = df[column_name]
            for pos in row_positions:
                vals.append(col.iloc[pos])
        texts.append(_safe_join(vals))
    return pd.Series(texts, index=pd.Index(order_keys, name=group_col))


def _build_content_profile_text(df: pd.DataFrame, group_col: str) -> pd.Series:
    return _aggregate_group_text_series(df, group_col, CONTENT_TEXT_COLS)


def _build_style_profile_text(df: pd.DataFrame, group_col: str) -> pd.Series:
    return _aggregate_group_text_series(df, group_col, STYLE_TEXT_COLS)


def _spec_required_columns(spec: ProfileSpec) -> tuple[str, ...]:
    ordered: list[str] = [spec.group_col]
    for column_name in spec.column_names:
        if column_name not in ordered:
            ordered.append(column_name)
    return tuple(ordered)


def _read_source_header(source_path: str | Path) -> list[str]:
    with open(source_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{source_path} is empty; expected a canonical split CSV header.") from exc


def _normalize_fragment(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_column_hash() -> str:
    payload: dict[str, object] = {
        "processed": list(expected_preprocess_column_order()),
        "split": list(expected_preprocess_column_order(require_split_indices=True)),
        "merged": list(expected_preprocess_column_order(require_split_indices=True, require_domain=True)),
        "contract": render_preprocess_contract_snapshot(),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_model_identity(tokenizer: AutoTokenizer, model: AutoModel) -> dict[str, object]:
    cfg = getattr(model, "config", None)
    return {
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "model_max_length": int(getattr(tokenizer, "model_max_length", 0) or 0),
            "vocab_size": int(len(tokenizer)),
            "special_tokens_overhead": int(tokenizer.num_special_tokens_to_add(pair=False)),
        },
        "model": {
            "class": model.__class__.__name__,
            "name_or_path": str(getattr(cfg, "_name_or_path", "")),
            "model_type": str(getattr(cfg, "model_type", "")),
            "hidden_size": int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "d_model", 0) or 0),
        },
    }


def _resolve_cache_root(cache_dir: str) -> Path:
    root = Path(cache_dir).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def _count_stale_cache_candidates(cache_root: Path, current_key: str) -> int:
    if not cache_root.exists():
        return 0
    count = 0
    for child in cache_root.iterdir():
        if not child.is_dir() or child.name == current_key or child.name.startswith(".tmp-"):
            continue
        if (child / "manifest.json").exists():
            count += 1
    return count


def _build_grouped_text_cache_request(
    *,
    cache_config: GroupedTextCacheConfig,
    dataset: str,
    source_path: str,
    spec: ProfileSpec,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    read_chunk_rows: int,
    group_shard_size: int,
    phase_timing: PhaseTiming,
) -> GroupedTextCacheRequest | None:
    if not cache_config.enabled:
        phase_timing.group_text_cache_status = "disabled"
        return None

    started_at = perf_counter()
    source = Path(source_path).resolve()
    stat = source.stat()
    source_cache_key = (str(source), int(stat.st_size), int(stat.st_mtime_ns))
    source_sha256 = _SOURCE_FILE_FINGERPRINT_CACHE.get(source_cache_key)
    if source_sha256 is None:
        source_sha256 = _sha256_file(source)
        _SOURCE_FILE_FINGERPRINT_CACHE[source_cache_key] = source_sha256
    phase_timing.group_text_cache_fingerprint_s += perf_counter() - started_at

    key_payload: dict[str, object] = {
        "cache_version": str(cache_config.version),
        "code_semantics_version": GROUPED_TEXT_CACHE_CODE_VERSION,
        "format_version": GROUPED_TEXT_CACHE_FORMAT_VERSION,
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "canonical_column_hash": _canonical_column_hash(),
        "canonical_text_source_contract": {
            "content_columns": list(CONTENT_TEXT_COLS),
            "style_columns": list(STYLE_TEXT_COLS),
            "selected_columns": list(spec.column_names),
        },
        "dataset": dataset,
        "source_file": {
            "path_name": source.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": source_sha256,
        },
        "spec": {
            "name": spec.name,
            "output_name": spec.output_name,
            "group_col": spec.group_col,
            "entity_kind": spec.entity_kind,
            "text_kind": spec.text_kind,
            "column_names": list(spec.column_names),
        },
        "sentence_embed_model": {
            "local_dir": str(Path(_require_sentence_embed_model_dir()).resolve()),
            "artifact_fingerprint": model_artifact_fingerprint(_require_sentence_embed_model_dir()),
            "tokenizer_model_identity": _tokenizer_model_identity(tokenizer, model),
            "odcr_embed_dim": int(_resolved_embed_dim()),
        },
        "grouped_text_semantics": {
            "normalizer": "_normalize_fragment: pd.isna->empty, str(value).strip(), skip empty",
            "joiner": "single ASCII space across columns in spec.column_names order",
            "ordering": "first observed group id order from train.csv chunks",
        },
        "sharding": {
            "group_shard_size": int(group_shard_size),
            "read_chunk_rows": int(read_chunk_rows),
        },
    }
    cache_key = hashlib.sha256(_canonical_json_bytes(key_payload)).hexdigest()
    cache_root = _resolve_cache_root(cache_config.cache_dir)
    entry_dir = cache_root / cache_key
    request = GroupedTextCacheRequest(
        cache_dir=cache_root,
        dataset=dataset,
        spec=spec,
        source_path=source,
        cache_key=cache_key,
        key_payload=key_payload,
        entry_dir=entry_dir,
        manifest_path=entry_dir / "manifest.json",
        stale_candidates=_count_stale_cache_candidates(cache_root, cache_key),
    )
    phase_timing.group_text_cache_key = cache_key
    return request


def _entry_from_manifest(
    request: GroupedTextCacheRequest,
    manifest: dict[str, object],
) -> GroupedTextCacheEntry:
    shards_raw = manifest.get("shards", [])
    if not isinstance(shards_raw, list):
        raise ValueError("manifest shards must be a list")
    return GroupedTextCacheEntry(
        request=request,
        manifest=manifest,
        target_size=int(manifest["target_size"]),
        group_count=int(manifest["group_count"]),
        rows_read=int(manifest["rows_read"]),
        read_chunks=int(manifest["read_chunks"]),
        shards=tuple(dict(item) for item in shards_raw),
    )


def _validate_grouped_text_cache_manifest(
    request: GroupedTextCacheRequest,
    manifest: dict[str, object],
) -> str | None:
    if manifest.get("complete") is not True:
        return "manifest_not_complete"
    if manifest.get("cache_key") != request.cache_key:
        return "cache_key_mismatch"
    if manifest.get("key_payload") != request.key_payload:
        return "fingerprint_payload_mismatch"
    if manifest.get("format_version") != GROUPED_TEXT_CACHE_FORMAT_VERSION:
        return "format_version_mismatch"
    for field_name in ("target_size", "group_count", "rows_read", "read_chunks", "shards"):
        if field_name not in manifest:
            return f"missing_{field_name}"
    try:
        entry = _entry_from_manifest(request, manifest)
    except Exception as exc:
        return f"manifest_shape_error:{exc}"
    if entry.group_count < 0 or entry.target_size < 0 or entry.rows_read < 0 or entry.read_chunks < 0:
        return "negative_manifest_count"
    shard_group_total = 0
    for shard in entry.shards:
        rel_path = str(shard.get("path", ""))
        expected_size = shard.get("file_size")
        expected_sha256 = shard.get("sha256")
        group_count = int(shard.get("group_count", -1))
        if not rel_path or expected_size is None or not expected_sha256 or group_count < 0:
            return "invalid_shard_manifest"
        shard_path = request.entry_dir / rel_path
        if not shard_path.exists():
            return f"missing_shard:{rel_path}"
        if int(shard_path.stat().st_size) != int(expected_size):
            return f"shard_size_mismatch:{rel_path}"
        shard_group_total += group_count
    if shard_group_total != entry.group_count:
        return "shard_group_count_mismatch"
    return None


def _load_grouped_text_cache(
    request: GroupedTextCacheRequest,
    *,
    phase_timing: PhaseTiming,
) -> GroupedTextCacheEntry | None:
    load_started_at = perf_counter()
    try:
        if not request.manifest_path.exists():
            phase_timing.group_text_cache_status = "miss"
            stale_note = f" stale_candidates={request.stale_candidates}" if request.stale_candidates else ""
            log.info(
                "[group-text-cache][%s][%s] miss key=%s%s root=%s",
                request.dataset,
                request.spec.name,
                request.cache_key,
                stale_note,
                request.cache_dir,
            )
            return None
        with open(request.manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        reason = _validate_grouped_text_cache_manifest(request, manifest)
        if reason is not None:
            phase_timing.group_text_cache_status = "stale"
            log.info(
                "[group-text-cache][%s][%s] stale key=%s reason=%s action=rebuild root=%s",
                request.dataset,
                request.spec.name,
                request.cache_key,
                reason,
                request.cache_dir,
            )
            return None
        entry = _entry_from_manifest(request, manifest)
        phase_timing.group_text_cache_status = "hit"
        phase_timing.rows_read = entry.rows_read
        phase_timing.read_chunks = entry.read_chunks
        phase_timing.groups_total = entry.group_count
        log.info(
            "[group-text-cache][%s][%s] hit key=%s shards=%d groups=%d target_size=%d root=%s",
            request.dataset,
            request.spec.name,
            request.cache_key,
            len(entry.shards),
            entry.group_count,
            entry.target_size,
            request.cache_dir,
        )
        return entry
    finally:
        phase_timing.group_text_cache_load_s += perf_counter() - load_started_at


def _write_grouped_text_cache(
    request: GroupedTextCacheRequest,
    accumulator: SpecGroupAccumulator,
    *,
    shard_size: int,
    phase_timing: PhaseTiming,
) -> GroupedTextCacheEntry:
    parent = request.entry_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = parent / f".tmp-{request.cache_key}-{os.getpid()}-{perf_counter_ns()}"
    shards_dir = tmp_dir / "shards"
    manifest_shards: list[dict[str, object]] = []
    try:
        shards_dir.mkdir(parents=True, exist_ok=False)
        shard_id = 0
        for start in range(0, accumulator.group_count, shard_size):
            group_ids = accumulator.order_keys[start : start + shard_size]
            build_started_at = perf_counter()
            texts = [accumulator.build_text(int(group_idx)) for group_idx in group_ids]
            phase_timing.group_text_build_s += perf_counter() - build_started_at

            group_indices = np.asarray(group_ids, dtype=np.int64)
            rel_path = f"shards/shard_{shard_id:06d}.npz"
            shard_path = tmp_dir / rel_path
            write_started_at = perf_counter()
            np.savez(
                shard_path,
                group_indices=group_indices,
                texts=np.asarray(texts, dtype=object),
            )
            shard_sha256 = _sha256_file(shard_path)
            shard_size_bytes = int(shard_path.stat().st_size)
            phase_timing.group_text_cache_write_s += perf_counter() - write_started_at
            manifest_shards.append(
                {
                    "shard_id": shard_id,
                    "path": rel_path,
                    "group_count": int(group_indices.size),
                    "first_group_idx": int(group_indices[0]) if group_indices.size else None,
                    "last_group_idx": int(group_indices[-1]) if group_indices.size else None,
                    "file_size": shard_size_bytes,
                    "sha256": shard_sha256,
                }
            )
            shard_id += 1

        manifest: dict[str, object] = {
            "complete": True,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cache_key": request.cache_key,
            "format_version": GROUPED_TEXT_CACHE_FORMAT_VERSION,
            "key_payload": request.key_payload,
            "dataset": request.dataset,
            "spec": request.spec.name,
            "target_size": int(accumulator.target_size),
            "group_count": int(accumulator.group_count),
            "rows_read": int(accumulator.rows_read),
            "read_chunks": int(accumulator.read_chunks),
            "shard_size": int(shard_size),
            "shards": manifest_shards,
        }
        write_started_at = perf_counter()
        with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, sort_keys=True, indent=2)
            fh.write("\n")
        phase_timing.group_text_cache_write_s += perf_counter() - write_started_at

        if request.entry_dir.exists():
            shutil.rmtree(request.entry_dir)
        os.replace(tmp_dir, request.entry_dir)
        phase_timing.group_text_cache_status = "miss_written"
        phase_timing.group_text_cache_shards_written = len(manifest_shards)
        phase_timing.group_text_cache_groups_written = int(accumulator.group_count)
        log.info(
            "[group-text-cache][%s][%s] miss-written key=%s shards=%d groups=%d target_size=%d root=%s",
            request.dataset,
            request.spec.name,
            request.cache_key,
            len(manifest_shards),
            accumulator.group_count,
            accumulator.target_size,
            request.cache_dir,
        )
        return _entry_from_manifest(request, manifest)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _read_spec_group_accumulator(
    *,
    source_path: str,
    spec: ProfileSpec,
    read_chunk_rows: int,
    phase_timing: PhaseTiming,
) -> SpecGroupAccumulator:
    required_columns = _spec_required_columns(spec)
    assert_no_deprecated_preprocess_detail_columns(
        _read_source_header(source_path),
        source_label=f"{source_path} ({spec.name})",
    )
    dtype_map = preprocess_csv_dtype_map(required_columns)
    reader = pd.read_csv(
        source_path,
        usecols=list(required_columns),
        dtype=dtype_map,
        keep_default_na=False,
        na_values=[],
        low_memory=False,
        chunksize=read_chunk_rows,
    )

    order_keys: list[int] = []
    fragments_by_group: dict[int, list[list[str]]] = {}
    target_size = 0
    rows_read = 0
    read_chunks = 0

    while True:
        read_started_at = perf_counter()
        try:
            chunk = next(reader)
        except StopIteration:
            break
        phase_timing.csv_read_s += perf_counter() - read_started_at

        build_started_at = perf_counter()
        read_chunks += 1
        rows_read += int(len(chunk))
        if not chunk.empty:
            gids = chunk[spec.group_col].to_numpy(dtype=np.int64, copy=False)
            if gids.size > 0:
                target_size = max(target_size, int(gids.max()) + 1)

            chunk_order_keys: list[int] = []
            chunk_seen: set[int] = set()
            for gid in gids:
                gid_int = int(gid)
                if gid_int not in chunk_seen:
                    chunk_seen.add(gid_int)
                    chunk_order_keys.append(gid_int)
                    if gid_int not in fragments_by_group:
                        fragments_by_group[gid_int] = [[] for _ in spec.column_names]
                        order_keys.append(gid_int)

            positions_by_gid: dict[int, list[int]] = {gid: [] for gid in chunk_order_keys}
            for pos, gid in enumerate(gids):
                positions_by_gid[int(gid)].append(pos)

            for gid in chunk_order_keys:
                row_positions = positions_by_gid[gid]
                column_fragments = fragments_by_group[gid]
                for col_idx, column_name in enumerate(spec.column_names):
                    values = chunk[column_name].take(row_positions).tolist()
                    target_fragments = column_fragments[col_idx]
                    for value in values:
                        normalized = _normalize_fragment(value)
                        if normalized:
                            target_fragments.append(normalized)
        phase_timing.group_text_build_s += perf_counter() - build_started_at

    phase_timing.rows_read = rows_read
    phase_timing.read_chunks = read_chunks
    phase_timing.groups_total = len(order_keys)
    return SpecGroupAccumulator(
        spec=spec,
        order_keys=order_keys,
        fragments_by_group=fragments_by_group,
        target_size=target_size,
        rows_read=rows_read,
        read_chunks=read_chunks,
    )


def _effective_probe_group_limit(
    group_count: int,
    *,
    batch_size: int,
    probe_config: ProbeConfig,
) -> int | None:
    if not probe_config.probe_only:
        return None
    limit = probe_config.max_groups_per_spec
    if probe_config.max_batches_per_spec is not None:
        batch_cap = int(probe_config.max_batches_per_spec) * int(batch_size)
        limit = batch_cap if limit is None else min(limit, batch_cap)
    if limit is None:
        return None
    return min(int(limit), int(group_count))


def _iter_group_text_shards(
    accumulator: SpecGroupAccumulator,
    *,
    shard_size: int,
    max_groups: int | None,
    phase_timing: PhaseTiming,
) -> Iterator[GroupTextShard]:
    effective_limit = accumulator.group_count if max_groups is None else min(max_groups, accumulator.group_count)
    if effective_limit <= 0:
        return

    shard_ids: list[int] = []
    shard_texts: list[str] = []
    shard_id = 0

    for group_idx in accumulator.order_keys[:effective_limit]:
        build_started_at = perf_counter()
        text = accumulator.build_text(group_idx)
        phase_timing.group_text_build_s += perf_counter() - build_started_at

        shard_ids.append(int(group_idx))
        shard_texts.append(text)
        if len(shard_ids) >= shard_size:
            phase_timing.group_shards += 1
            phase_timing.groups_finalized += len(shard_ids)
            yield GroupTextShard(
                shard_id=shard_id,
                group_indices=np.asarray(shard_ids, dtype=np.int64),
                texts=list(shard_texts),
            )
            shard_id += 1
            shard_ids = []
            shard_texts = []

    if shard_ids:
        phase_timing.group_shards += 1
        phase_timing.groups_finalized += len(shard_ids)
        yield GroupTextShard(
            shard_id=shard_id,
            group_indices=np.asarray(shard_ids, dtype=np.int64),
            texts=list(shard_texts),
        )


def _iter_cached_group_text_shards(
    entry: GroupedTextCacheEntry,
    *,
    max_groups: int | None,
    phase_timing: PhaseTiming,
) -> Iterator[GroupTextShard]:
    effective_limit = entry.group_count if max_groups is None else min(max_groups, entry.group_count)
    if effective_limit <= 0:
        return

    emitted = 0
    for shard_meta in entry.shards:
        if emitted >= effective_limit:
            break
        rel_path = str(shard_meta["path"])
        shard_path = entry.request.entry_dir / rel_path
        load_started_at = perf_counter()
        expected_sha256 = str(shard_meta["sha256"])
        actual_sha256 = _sha256_file(shard_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"grouped-text cache shard checksum mismatch for {entry.request.dataset}/{entry.request.spec.name}: "
                f"{rel_path}"
            )
        with np.load(shard_path, allow_pickle=True) as loaded:
            group_indices = loaded["group_indices"].astype(np.int64, copy=False)
            texts = [str(item) for item in loaded["texts"].tolist()]
        phase_timing.group_text_cache_load_s += perf_counter() - load_started_at

        if len(texts) != int(group_indices.size):
            raise ValueError(
                f"grouped-text cache shard length mismatch for {entry.request.dataset}/{entry.request.spec.name}: "
                f"{rel_path}"
            )
        remaining = effective_limit - emitted
        if remaining < int(group_indices.size):
            group_indices = group_indices[:remaining]
            texts = texts[:remaining]

        phase_timing.group_shards += 1
        phase_timing.groups_finalized += int(group_indices.size)
        phase_timing.group_text_cache_shards_loaded += 1
        emitted += int(group_indices.size)
        yield GroupTextShard(
            shard_id=int(shard_meta["shard_id"]),
            group_indices=group_indices,
            texts=texts,
        )


def _encode_texts_to_numpy(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    batch_size: int,
    precision_config: PrecisionConfig,
    phase_timing: PhaseTiming,
    spec_started_at: float,
    dataset_started_at: float,
    max_batches: int | None = None,
) -> np.ndarray:
    hidden_size = int(model.config.hidden_size)
    if not texts:
        return np.zeros((0, hidden_size), dtype=np.float32)

    outputs_accum: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        if max_batches is not None and phase_timing.batches_encoded >= max_batches:
            break
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        phase_timing.mark_first_tokenize(
            spec_started_at=spec_started_at,
            dataset_started_at=dataset_started_at,
        )
        tokenize_started_at = perf_counter()
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        phase_timing.tokenize_s += perf_counter() - tokenize_started_at

        encoded = {k: v.to(device) for k, v in encoded.items()}
        phase_timing.mark_first_forward(
            spec_started_at=spec_started_at,
            dataset_started_at=dataset_started_at,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_started_at = perf_counter()
        autocast_ctx = (
            torch.autocast(
                device_type="cuda",
                dtype=precision_config.autocast_dtype,
                enabled=precision_config.autocast_enabled,
            )
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                outputs = model(**encoded)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phase_timing.gpu_forward_s += perf_counter() - forward_started_at

        batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().to(torch.float32).numpy()
        outputs_accum.append(batch_embeddings)
        phase_timing.batches_encoded += 1
        phase_timing.groups_encoded += end - start

    if not outputs_accum:
        return np.zeros((0, hidden_size), dtype=np.float32)
    return np.concatenate(outputs_accum, axis=0).astype(np.float32, copy=False)


def _write_profile_matrix(
    *,
    path: str,
    matrix: np.ndarray,
    phase_timing: PhaseTiming,
) -> None:
    write_started_at = perf_counter()
    atomic_save_numpy(path, matrix.astype(np.float32, copy=False))
    phase_timing.write_npy_s += perf_counter() - write_started_at


def _load_sentence_embed_model(device: torch.device) -> tuple[AutoTokenizer, AutoModel]:
    model_dir = _require_sentence_embed_model_dir()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()
    expected_hidden = int(_resolved_embed_dim())
    actual_hidden = int(model.config.hidden_size)
    if actual_hidden != expected_hidden:
        raise ValueError(
            f"Sentence embed hidden_size={actual_hidden} does not match ODCR_EMBED_DIM={expected_hidden}."
        )
    return tokenizer, model


def _output_path(dataset: str, output_name: str) -> str:
    return os.path.join(_resolved_data_dir(), dataset, output_name)


def _validate_output_matrix(
    stored: np.ndarray,
    *,
    target_size: int,
    hidden_size: int,
    output_name: str,
) -> None:
    expected_shape = (target_size, hidden_size)
    if stored.shape != expected_shape:
        raise ValueError(f"{output_name} shape mismatch: got {stored.shape}, expected {expected_shape}")
    if stored.dtype != np.float32:
        raise ValueError(f"{output_name} dtype mismatch: got {stored.dtype}, expected float32")


def _cosine_similarity_rows(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs64 = lhs.astype(np.float64, copy=False)
    rhs64 = rhs.astype(np.float64, copy=False)
    numerator = np.sum(lhs64 * rhs64, axis=1)
    denom = np.linalg.norm(lhs64, axis=1) * np.linalg.norm(rhs64, axis=1)
    denom = np.maximum(denom, 1e-12)
    return numerator / denom


def _resolve_verify_sample_indices(
    *,
    spec: ProfileSpec,
    target_size: int,
    active_indices: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    fixed_user_indices: list[int] | None,
    fixed_item_indices: list[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    fixed = fixed_user_indices if spec.entity_kind == "user" else fixed_item_indices
    present_mask = np.zeros(target_size, dtype=bool)
    present_mask[active_indices] = True
    if fixed is not None:
        selected = np.asarray(fixed, dtype=np.int64)
        if np.any(selected < 0) or np.any(selected >= target_size):
            raise ValueError(
                f"{spec.entity_kind} verify indices out of range for target_size={target_size}: {selected.tolist()}"
            )
        active_selected = selected[present_mask[selected]]
        missing_selected = selected[~present_mask[selected]]
        return active_selected, missing_selected

    if sample_size <= 0 or len(active_indices) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    actual_sample = min(sample_size, len(active_indices))
    sampled = np.sort(rng.choice(active_indices, size=actual_sample, replace=False).astype(np.int64))
    return sampled, np.zeros((0,), dtype=np.int64)


def _process_compute_spec(
    *,
    dataset: str,
    spec: ProfileSpec,
    source_path: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    precision_config: PrecisionConfig,
    read_chunk_rows: int,
    group_shard_size: int,
    grouped_text_cache: GroupedTextCacheConfig,
    probe_config: ProbeConfig,
    dataset_started_at: float,
) -> PhaseTiming:
    spec_started_at = perf_counter()
    phase_timing = PhaseTiming()
    cache_request = _build_grouped_text_cache_request(
        cache_config=grouped_text_cache,
        dataset=dataset,
        source_path=source_path,
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        read_chunk_rows=read_chunk_rows,
        group_shard_size=group_shard_size,
        phase_timing=phase_timing,
    )
    cache_entry = (
        _load_grouped_text_cache(cache_request, phase_timing=phase_timing)
        if cache_request is not None
        else None
    )

    accumulator: SpecGroupAccumulator | None = None
    if cache_entry is None:
        accumulator = _read_spec_group_accumulator(
            source_path=source_path,
            spec=spec,
            read_chunk_rows=read_chunk_rows,
            phase_timing=phase_timing,
        )
        if cache_request is not None:
            cache_entry = _write_grouped_text_cache(
                cache_request,
                accumulator,
                shard_size=group_shard_size,
                phase_timing=phase_timing,
            )

    group_count = cache_entry.group_count if cache_entry is not None else accumulator.group_count
    target_size = cache_entry.target_size if cache_entry is not None else accumulator.target_size
    read_chunks = cache_entry.read_chunks if cache_entry is not None else accumulator.read_chunks
    hidden_size = int(model.config.hidden_size)
    output = (
        None
        if probe_config.probe_only
        else np.zeros((target_size, hidden_size), dtype=np.float32)
    )
    group_limit = _effective_probe_group_limit(
        group_count,
        batch_size=batch_size,
        probe_config=probe_config,
    )

    shard_iter: Iterator[GroupTextShard]
    if cache_entry is not None:
        shard_iter = _iter_cached_group_text_shards(
            cache_entry,
            max_groups=group_limit,
            phase_timing=phase_timing,
        )
    else:
        shard_iter = _iter_group_text_shards(
            accumulator,
            shard_size=group_shard_size,
            max_groups=group_limit,
            phase_timing=phase_timing,
        )

    for shard in shard_iter:
        embeddings = _encode_texts_to_numpy(
            shard.texts,
            model,
            tokenizer,
            device,
            batch_size=batch_size,
            precision_config=precision_config,
            phase_timing=phase_timing,
            spec_started_at=spec_started_at,
            dataset_started_at=dataset_started_at,
            max_batches=probe_config.max_batches_per_spec,
        )
        processed = int(embeddings.shape[0])
        if processed == 0:
            break
        if output is not None:
            output[shard.group_indices[:processed]] = embeddings
        if probe_config.max_batches_per_spec is not None and phase_timing.batches_encoded >= probe_config.max_batches_per_spec:
            break

    if output is not None:
        if phase_timing.groups_encoded != group_count:
            raise RuntimeError(
                f"{spec.output_name} encoded {phase_timing.groups_encoded} groups but expected {group_count}"
            )
        _write_profile_matrix(
            path=_output_path(dataset, spec.output_name),
            matrix=output,
            phase_timing=phase_timing,
        )
        log.info(
            "[%s][%s] wrote %s target_size=%d groups=%d read_chunks=%d",
            dataset,
            spec.name,
            spec.output_name,
            target_size,
            group_count,
            read_chunks,
        )
    else:
        log.info(
            "[probe][%s][%s] skipped write for %s groups_encoded=%d/%d read_chunks=%d batches=%d",
            dataset,
            spec.name,
            spec.output_name,
            phase_timing.groups_encoded,
            group_count,
            read_chunks,
            phase_timing.batches_encoded,
        )

    phase_timing.total_s = perf_counter() - spec_started_at
    return phase_timing


def _verify_spec_output(
    *,
    dataset: str,
    spec: ProfileSpec,
    source_path: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    precision_config: PrecisionConfig,
    read_chunk_rows: int,
    sample_size: int,
    rng: np.random.Generator,
    verify_user_indices: list[int] | None,
    verify_item_indices: list[int] | None,
    dataset_started_at: float,
) -> tuple[PhaseTiming, dict[str, object]]:
    spec_started_at = perf_counter()
    phase_timing = PhaseTiming()
    accumulator = _read_spec_group_accumulator(
        source_path=source_path,
        spec=spec,
        read_chunk_rows=read_chunk_rows,
        phase_timing=phase_timing,
    )
    hidden_size = int(model.config.hidden_size)
    active_indices = accumulator.active_indices()
    verify_started_at = perf_counter()
    output_path = _output_path(dataset, spec.output_name)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Missing verify target: {output_path}")
    stored = np.load(output_path)
    _validate_output_matrix(
        stored,
        target_size=accumulator.target_size,
        hidden_size=hidden_size,
        output_name=spec.output_name,
    )

    present_mask = np.zeros(accumulator.target_size, dtype=bool)
    present_mask[active_indices] = True
    missing_mask = ~present_mask
    zeros_ok = True
    if np.any(missing_mask):
        zeros_ok = bool(np.allclose(stored[missing_mask], 0.0, atol=0.0, rtol=0.0))
        if not zeros_ok:
            raise ValueError(f"{spec.output_name} has non-zero rows on missing {spec.entity_kind} indices")

    active_sample, missing_sample = _resolve_verify_sample_indices(
        spec=spec,
        target_size=accumulator.target_size,
        active_indices=active_indices,
        sample_size=sample_size,
        rng=rng,
        fixed_user_indices=verify_user_indices,
        fixed_item_indices=verify_item_indices,
    )
    phase_timing.verify_sample_count = int(active_sample.size)
    phase_timing.verify_missing_sample_count = int(missing_sample.size)
    phase_timing.verify_extra_s += perf_counter() - verify_started_at

    sample_report: dict[str, object] = {
        "path": output_path,
        "shape": list(stored.shape),
        "dtype": str(stored.dtype),
        "active_count": int(active_indices.size),
        "missing_count": int(np.count_nonzero(missing_mask)),
        "zeros_ok": zeros_ok,
        "sample_indices": active_sample.tolist(),
        "missing_sample_indices": missing_sample.tolist(),
        "allclose": True,
        "cosine_min": None,
        "cosine_mean": None,
        "max_abs_diff": 0.0,
    }

    verify_compare_started_at = perf_counter()
    if missing_sample.size > 0 and not np.allclose(stored[missing_sample], 0.0, atol=0.0, rtol=0.0):
        raise ValueError(f"{spec.output_name} fixed missing indices are not zero-filled: {missing_sample.tolist()}")
    phase_timing.verify_extra_s += perf_counter() - verify_compare_started_at

    if active_sample.size > 0:
        text_build_started_at = perf_counter()
        sample_texts = accumulator.build_texts(active_sample)
        phase_timing.group_text_build_s += perf_counter() - text_build_started_at
        phase_timing.groups_finalized += int(active_sample.size)

        recomputed = _encode_texts_to_numpy(
            sample_texts,
            model,
            tokenizer,
            device,
            batch_size=batch_size,
            precision_config=precision_config,
            phase_timing=phase_timing,
            spec_started_at=spec_started_at,
            dataset_started_at=dataset_started_at,
        )

        compare_started_at = perf_counter()
        stored_rows = stored[active_sample]
        allclose_ok = bool(np.allclose(stored_rows, recomputed, atol=1e-5, rtol=1e-4))
        cosine = _cosine_similarity_rows(stored_rows, recomputed)
        max_abs_diff = float(np.max(np.abs(stored_rows - recomputed)))
        sample_report["allclose"] = allclose_ok
        sample_report["cosine_min"] = float(np.min(cosine))
        sample_report["cosine_mean"] = float(np.mean(cosine))
        sample_report["max_abs_diff"] = max_abs_diff
        phase_timing.verify_extra_s += perf_counter() - compare_started_at
        if not allclose_ok:
            raise ValueError(
                f"{spec.output_name} verify failed on indices {active_sample.tolist()} "
                f"(max_abs_diff={max_abs_diff:.6g})"
            )

    phase_timing.total_s = perf_counter() - spec_started_at
    return phase_timing, sample_report


def _log_phase_summary(
    *,
    dataset: str,
    spec: ProfileSpec,
    phase_timing: PhaseTiming,
    mode: str,
    device: torch.device,
) -> None:
    payload = {
        "dataset": dataset,
        "spec": spec.name,
        "mode": mode,
        "device": str(device),
        "entity_kind": spec.entity_kind,
        "text_kind": spec.text_kind,
        **phase_timing.to_dict(),
    }
    log.info("[phase][%s][%s] %s", dataset, spec.name, json.dumps(payload, sort_keys=True))


def _log_dataset_phase_summary(
    *,
    dataset: str,
    phase_timings: list[PhaseTiming],
    mode: str,
    device: torch.device,
) -> None:
    summary = {
        "dataset": dataset,
        "mode": mode,
        "device": str(device),
        "csv_read_s": _round_timing(sum(item.csv_read_s for item in phase_timings)),
        "group_text_build_s": _round_timing(sum(item.group_text_build_s for item in phase_timings)),
        "group_text_cache_fingerprint_s": _round_timing(
            sum(item.group_text_cache_fingerprint_s for item in phase_timings)
        ),
        "group_text_cache_load_s": _round_timing(sum(item.group_text_cache_load_s for item in phase_timings)),
        "group_text_cache_write_s": _round_timing(sum(item.group_text_cache_write_s for item in phase_timings)),
        "tokenize_s": _round_timing(sum(item.tokenize_s for item in phase_timings)),
        "gpu_forward_s": _round_timing(sum(item.gpu_forward_s for item in phase_timings)),
        "write_npy_s": _round_timing(sum(item.write_npy_s for item in phase_timings)),
        "verify_extra_s": _round_timing(sum(item.verify_extra_s for item in phase_timings)),
        "total_s": _round_timing(sum(item.total_s for item in phase_timings)),
        "rows_read": int(sum(item.rows_read for item in phase_timings)),
        "read_chunks": int(sum(item.read_chunks for item in phase_timings)),
        "groups_total": int(sum(item.groups_total for item in phase_timings)),
        "groups_finalized": int(sum(item.groups_finalized for item in phase_timings)),
        "groups_encoded": int(sum(item.groups_encoded for item in phase_timings)),
        "group_shards": int(sum(item.group_shards for item in phase_timings)),
        "batches_encoded": int(sum(item.batches_encoded for item in phase_timings)),
        "group_text_cache_hits": int(sum(1 for item in phase_timings if item.group_text_cache_status == "hit")),
        "group_text_cache_misses": int(
            sum(1 for item in phase_timings if item.group_text_cache_status in {"miss", "miss_written", "stale"})
        ),
        "group_text_cache_disabled": int(
            sum(1 for item in phase_timings if item.group_text_cache_status == "disabled")
        ),
        "group_text_cache_shards_loaded": int(sum(item.group_text_cache_shards_loaded for item in phase_timings)),
        "group_text_cache_shards_written": int(sum(item.group_text_cache_shards_written for item in phase_timings)),
        "group_text_cache_groups_written": int(sum(item.group_text_cache_groups_written for item in phase_timings)),
        "first_forward_from_dataset_start_s": _round_optional_timing(
            min(
                (
                    item.first_forward_from_dataset_start_s
                    for item in phase_timings
                    if item.first_forward_from_dataset_start_s is not None
                ),
                default=None,
            )
        ),
        "first_tokenize_from_dataset_start_s": _round_optional_timing(
            min(
                (
                    item.first_tokenize_from_dataset_start_s
                    for item in phase_timings
                    if item.first_tokenize_from_dataset_start_s is not None
                ),
                default=None,
            )
        ),
    }
    log.info("[phase-summary][%s] %s", dataset, json.dumps(summary, sort_keys=True))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for raw_arg in sys.argv[1:]:
        if raw_arg == "--gpus" or raw_arg.startswith("--gpus="):
            sys.stderr.write(
                "compute_embeddings.py: error: --gpus has been removed.\n"
                "Use --cuda-device N (single process / single GPU) or CUDA_VISIBLE_DEVICES.\n"
            )
            raise SystemExit(2)

    parser = argparse.ArgumentParser(
        description=(
            "Compute dual-channel user/item embeddings with a spec-by-spec cold path. "
            "Each spec is read, grouped, tokenized, and encoded independently; verify-only "
            "re-encodes sampled rows, and probe-only skips final writes."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        metavar="N",
        help="CUDA device id (default: 0). Formal preprocess_b fails fast when CUDA is unavailable.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Resolved project.data_dir from ./odcr; required for formal preprocess_b.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Resolved env.models_dir from ./odcr; required for formal preprocess_b.",
    )
    parser.add_argument(
        "--sentence-embed-model",
        type=str,
        default=None,
        help="Resolved env.sentence_embed_model path from ./odcr; required for formal preprocess_b.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Resolved env.embed_dim from ./odcr; required for formal preprocess_b.",
    )
    parser.add_argument(
        "--allow-cpu-debug",
        action="store_true",
        default=False,
        help="Explicit test-only CPU debug mode. Formal preprocess_b must not use this flag.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names. Defaults to all datasets.",
    )
    parser.add_argument(
        "--specs",
        type=str,
        default=None,
        help="Optional comma-separated subset of specs. Requires --verify-only or --probe-only.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Embedding batch size from the resolved preprocess_b payload.",
    )
    parser.add_argument(
        "--read-chunk-rows",
        type=int,
        default=None,
        metavar="N",
        help="CSV row chunk size for spec-level cold-path reads.",
    )
    parser.add_argument(
        "--group-shard-size",
        type=int,
        default=None,
        metavar="N",
        help="How many grouped texts to finalize per shard before batched tokenize/forward.",
    )
    parser.add_argument(
        "--grouped-text-cache",
        action="store_true",
        default=None,
        dest="grouped_text_cache_enabled",
        help="Enable grouped-text shard cache for preprocess_b cold-path reuse.",
    )
    parser.add_argument(
        "--no-grouped-text-cache",
        action="store_false",
        dest="grouped_text_cache_enabled",
        help="Disable grouped-text shard cache and build grouped texts directly from train.csv.",
    )
    parser.add_argument(
        "--grouped-text-cache-dir",
        type=str,
        default=None,
        help="Root directory for preprocess_b grouped-text shard cache.",
    )
    parser.add_argument(
        "--grouped-text-cache-version",
        type=str,
        default=None,
        help="Version salt for grouped-text cache invalidation.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=None,
        dest="bf16_enabled",
        help="Enable CUDA bf16 autocast for preprocess_b encoder forward.",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="bf16_enabled",
        help="Disable CUDA bf16 autocast for preprocess_b encoder forward.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        default=None,
        dest="tf32_enabled",
        help="Enable TF32 matmul/cudnn for preprocess_b CUDA steady-state.",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_false",
        dest="tf32_enabled",
        help="Disable TF32 matmul/cudnn for preprocess_b CUDA steady-state.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read existing *_profiles.npy, re-encode sampled rows, and emit a verification report without writing outputs.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run the cold path and phase timing without writing *_profiles.npy outputs.",
    )
    parser.add_argument(
        "--probe-max-groups-per-spec",
        type=int,
        default=None,
        metavar="N",
        help="When probing, only finalize/encode the first N groups for each selected spec.",
    )
    parser.add_argument(
        "--probe-max-batches-per-spec",
        type=int,
        default=None,
        metavar="N",
        help="When probing, only forward the first N batches for each selected spec.",
    )
    parser.add_argument(
        "--verify-sample-size",
        type=int,
        default=8,
        metavar="N",
        help="Sample size per user/item side when --verify-only is used and fixed indices are not supplied.",
    )
    parser.add_argument(
        "--verify-seed",
        type=int,
        default=7,
        metavar="N",
        help="Random seed for verify-only sampling.",
    )
    parser.add_argument(
        "--verify-user-indices",
        type=str,
        default=None,
        help="Optional fixed user_idx list for verify-only, for example '0,7,42'.",
    )
    parser.add_argument(
        "--verify-item-indices",
        type=str,
        default=None,
        help="Optional fixed item_idx list for verify-only, for example '0,9,99'.",
    )
    args = parser.parse_args()
    _install_resolved_preprocess_context(args)

    if args.verify_only and args.probe_only:
        raise ValueError("--verify-only cannot be combined with --probe-only")

    specs = _resolve_profile_specs(args.specs)
    if args.specs is not None and not (args.verify_only or args.probe_only):
        raise ValueError("--specs requires --verify-only or --probe-only to avoid partial writes")

    batch_size = args.embed_batch_size
    if batch_size is None:
        raise ValueError("--embed-batch-size is required from the resolved preprocess_b payload.")
    batch_size = max(1, int(batch_size))
    verify_sample_size = max(0, int(args.verify_sample_size))
    if args.read_chunk_rows is None:
        raise ValueError("--read-chunk-rows is required from the resolved preprocess_b payload.")
    if args.group_shard_size is None:
        raise ValueError("--group-shard-size is required from the resolved preprocess_b payload.")
    if args.grouped_text_cache_enabled is None:
        raise ValueError("--grouped-text-cache/--no-grouped-text-cache is required from the resolved preprocess_b payload.")
    if args.grouped_text_cache_dir is None:
        raise ValueError("--grouped-text-cache-dir is required from the resolved preprocess_b payload.")
    if args.grouped_text_cache_version is None:
        raise ValueError("--grouped-text-cache-version is required from the resolved preprocess_b payload.")
    read_chunk_rows = max(1, int(args.read_chunk_rows))
    group_shard_size = max(1, int(args.group_shard_size))
    grouped_text_cache = GroupedTextCacheConfig(
        enabled=bool(args.grouped_text_cache_enabled),
        cache_dir=str(args.grouped_text_cache_dir),
        version=str(args.grouped_text_cache_version),
    )
    if grouped_text_cache.enabled and not grouped_text_cache.cache_dir.strip():
        raise ValueError("--grouped-text-cache-dir must be non-empty when grouped-text cache is enabled")
    if grouped_text_cache.enabled and not grouped_text_cache.version.strip():
        raise ValueError("--grouped-text-cache-version must be non-empty when grouped-text cache is enabled")

    device = _select_cuda_device_or_fail(args)

    verify_user_indices = _parse_optional_indices(args.verify_user_indices, label="--verify-user-indices")
    verify_item_indices = _parse_optional_indices(args.verify_item_indices, label="--verify-item-indices")
    probe_config = _resolve_probe_config(args)
    precision_config = _resolve_precision_config(args, device)
    _configure_tf32(device, enabled=precision_config.tf32_enabled)

    log.info(
        "compute_embeddings config: device=%s batch_size=%s read_chunk_rows=%s group_shard_size=%s "
        "grouped_text_cache_enabled=%s grouped_text_cache_dir=%s grouped_text_cache_version=%s "
        "bf16_enabled=%s tf32_enabled=%s autocast_enabled=%s "
        "verify_only=%s probe_only=%s probe_max_groups=%s probe_max_batches=%s specs=%s verify_sample_size=%s",
        device,
        batch_size,
        read_chunk_rows,
        group_shard_size,
        grouped_text_cache.enabled,
        grouped_text_cache.cache_dir,
        grouped_text_cache.version,
        precision_config.bf16_enabled,
        precision_config.tf32_enabled,
        precision_config.autocast_enabled,
        args.verify_only,
        probe_config.probe_only,
        probe_config.max_groups_per_spec,
        probe_config.max_batches_per_spec,
        [spec.name for spec in specs],
        verify_sample_size,
    )

    tokenizer, model = _load_sentence_embed_model(device)
    datasets = _resolve_datasets(args.datasets)

    for dataset in datasets:
        source_path = os.path.join(_resolved_data_dir(), dataset, "train.csv")
        dataset_started_at = perf_counter()
        phase_timings: list[PhaseTiming] = []
        log.info("[%s] source=%s", dataset, source_path)

        if args.verify_only:
            report: dict[str, object] = {"dataset": dataset, "artifacts": {}}
            rng = np.random.default_rng(int(args.verify_seed))
            for spec in specs:
                phase_timing, sample_report = _verify_spec_output(
                    dataset=dataset,
                    spec=spec,
                    source_path=source_path,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    precision_config=precision_config,
                    read_chunk_rows=read_chunk_rows,
                    sample_size=verify_sample_size,
                    rng=rng,
                    verify_user_indices=verify_user_indices,
                    verify_item_indices=verify_item_indices,
                    dataset_started_at=dataset_started_at,
                )
                phase_timings.append(phase_timing)
                report["artifacts"][spec.output_name] = sample_report
                _log_phase_summary(
                    dataset=dataset,
                    spec=spec,
                    phase_timing=phase_timing,
                    mode="verify",
                    device=device,
                )
            log.info("[verify][%s] %s", dataset, json.dumps(report, ensure_ascii=False, sort_keys=True))
            _log_dataset_phase_summary(
                dataset=dataset,
                phase_timings=phase_timings,
                mode="verify",
                device=device,
            )
            continue

        for spec in specs:
            phase_timing = _process_compute_spec(
                dataset=dataset,
                spec=spec,
                source_path=source_path,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=batch_size,
                precision_config=precision_config,
                read_chunk_rows=read_chunk_rows,
                group_shard_size=group_shard_size,
                grouped_text_cache=grouped_text_cache,
                probe_config=probe_config,
                dataset_started_at=dataset_started_at,
            )
            phase_timings.append(phase_timing)
            _log_phase_summary(
                dataset=dataset,
                spec=spec,
                phase_timing=phase_timing,
                mode="probe" if probe_config.probe_only else "write",
                device=device,
            )
        _log_dataset_phase_summary(
            dataset=dataset,
            phase_timings=phase_timings,
            mode="probe" if probe_config.probe_only else "write",
            device=device,
        )
        log.info("[%s] dataset finished", dataset)


if __name__ == "__main__":
    main()
