from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, __version__ as TRANSFORMERS_VERSION

from data_contract import (
    DOMAIN_CONTENT_TEXT_COLUMNS,
    DOMAIN_STYLE_TEXT_COLUMNS,
    PREPROCESS_CONTRACT_VERSION,
    expected_preprocess_column_order,
    read_preprocess_csv,
    render_preprocess_contract_snapshot,
)
from odcr_core.file_atomic import atomic_save_numpy, atomic_torch_save, atomic_write_json
from odcr_core.training_checkpoint import model_artifact_fingerprint

log = logging.getLogger("infer_domain_semantics")

ALL_DATASETS = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]

CONTENT_COLS = DOMAIN_CONTENT_TEXT_COLUMNS
STYLE_COLS = DOMAIN_STYLE_TEXT_COLUMNS

MAX_TOTAL_TOKENS = 512
DEFAULT_TOKENIZER_BATCH_SIZE = 1024
TOKEN_WINDOW_CACHE_CONTRACT_VERSION = "preprocess_c/token_window_contract/v3"
EMPTY_TEXT_PLACEHOLDER = "empty_text"
FILE_FINGERPRINT_SAMPLE_BYTES = 1024 * 1024
GPU_TMUX_HINT = "tmux -L odcr_gpu new-session -A -s odcr"


@dataclass(frozen=True)
class DomainSpec:
    name: str
    output_name: str
    column_names: tuple[str, ...]


DOMAIN_SPECS = (
    DomainSpec(name="content", output_name="domain_content.npy", column_names=CONTENT_COLS),
    DomainSpec(name="style", output_name="domain_style.npy", column_names=STYLE_COLS),
)


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
            "infer_domain_semantics.py must be launched with the resolved preprocess payload from ./odcr; "
            f"missing {missing}. Refusing to read configs/odcr.yaml as a child-side fallback."
        )
    if int(args.embed_dim) <= 0:
        raise RuntimeError("--embed-dim must be a positive resolved env.embed_dim value.")
    local_files_only = str(os.environ.get("ODCR_RESOLVED_LOCAL_FILES_ONLY") or "").strip().lower()
    if local_files_only not in ("1", "true", "yes", "on"):
        raise RuntimeError("infer_domain_semantics.py requires ODCR_RESOLVED_LOCAL_FILES_ONLY=1 for formal preprocess_c.")
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
                f"preprocess_c requested cuda:{device_idx}, but torch sees {device_count} visible CUDA device(s). "
                f"Use the GPU tmux session: {GPU_TMUX_HINT}"
            )
        return torch.device(f"cuda:{device_idx}")
    if bool(getattr(args, "allow_cpu_debug", False)):
        log.warning("preprocess_c explicit --allow-cpu-debug enabled; using CPU debug mode without formal admission.")
        return torch.device("cpu")
    raise RuntimeError(
        "preprocess_c requires CUDA before loading BGE-large; torch.cuda.is_available() is false. "
        f"Use the GPU tmux session: {GPU_TMUX_HINT}"
    )


@dataclass(frozen=True)
class TokenWindowCacheConfig:
    enabled: bool
    cache_dir: str
    version: str
    shard_size: int


@dataclass(frozen=True)
class TokenWindowShard:
    window_ids: np.ndarray
    window_lengths: np.ndarray
    shard_index: int
    cache_path: str | None = None
    cache_hit: bool = False

    @property
    def window_count(self) -> int:
        return int(self.window_lengths.shape[0])


@dataclass(frozen=True)
class TokenWindowCacheManifest:
    fingerprint: dict[str, object]
    payload_budget: int
    shard_size: int
    shard_count: int
    window_count: int
    cache_path: str


@dataclass(frozen=True)
class TokenizerPathConfig:
    hotpath_enabled: bool


@dataclass(frozen=True)
class ProbeConfig:
    probe_only: bool
    max_chunks_per_domain: int | None


@dataclass(frozen=True)
class TokenizerPipelineConfig:
    tokenizer_parallelism_enabled: bool
    tokenizer_threads_per_worker: int
    tokenizer_total_threads: int
    prefetch_batches: int
    pin_memory: bool
    non_blocking_h2d: bool
    async_prefetch_enabled: bool
    cpu_cores_reserved: int
    cpu_cores_available: int

    @property
    def cpu_cores_configured(self) -> int:
        return int(self.tokenizer_total_threads)


@dataclass
class DomainTiming:
    csv_read_s: float = 0.0
    token_window_cache_manifest_s: float = 0.0
    token_window_cache_load_s: float = 0.0
    token_window_cache_write_s: float = 0.0
    token_window_cache_build_s: float = 0.0
    tokenizer_queue_wait_s: float = 0.0
    gpu_queue_wait_s: float = 0.0
    h2d_s: float = 0.0
    tokenize_s: float = 0.0
    gpu_forward_s: float = 0.0
    encode_wall_s: float = 0.0
    total_s: float = 0.0
    token_windows: int = 0
    cache_status: str = "disabled"
    cache_shards_loaded: int = 0
    cache_shards_written: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "csv_read_s": round(float(self.csv_read_s), 6),
            "token_window_cache_manifest_s": round(float(self.token_window_cache_manifest_s), 6),
            "token_window_cache_load_s": round(float(self.token_window_cache_load_s), 6),
            "token_window_cache_write_s": round(float(self.token_window_cache_write_s), 6),
            "token_window_cache_build_s": round(float(self.token_window_cache_build_s), 6),
            "tokenizer_queue_wait_s": round(float(self.tokenizer_queue_wait_s), 6),
            "gpu_queue_wait_s": round(float(self.gpu_queue_wait_s), 6),
            "h2d_s": round(float(self.h2d_s), 6),
            "tokenize_s": round(float(self.tokenize_s), 6),
            "gpu_forward_s": round(float(self.gpu_forward_s), 6),
            "encode_wall_s": round(float(self.encode_wall_s), 6),
            "overlap_effectiveness": round(float(_overlap_effectiveness(self)), 6),
            "total_s": round(float(self.total_s), 6),
            "token_window_count": int(self.token_windows),
            "token_window_cache_status": self.cache_status,
            "token_window_cache_shards_loaded": int(self.cache_shards_loaded),
            "token_window_cache_shards_written": int(self.cache_shards_written),
        }


def _overlap_effectiveness(timing: DomainTiming) -> float:
    sequential = float(timing.tokenize_s + timing.h2d_s + timing.gpu_forward_s)
    wall = float(timing.encode_wall_s)
    overlap_capacity = min(float(timing.tokenize_s), float(timing.h2d_s + timing.gpu_forward_s))
    if sequential <= 0.0 or wall <= 0.0 or overlap_capacity <= 0.0:
        return 0.0
    hidden = max(0.0, sequential - wall)
    return max(0.0, min(1.0, hidden / overlap_capacity))


def _resolve_datasets(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(ALL_DATASETS)
    datasets = [d.strip() for d in str(raw).split(",") if d.strip()]
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; expected one of {ALL_DATASETS}")
    return datasets


DOMAIN_SPEC_BY_NAME = {spec.name: spec for spec in DOMAIN_SPECS}


def _resolve_domain_specs(raw: str | None) -> tuple[DomainSpec, ...]:
    if raw is None or not str(raw).strip():
        return tuple(DOMAIN_SPECS)
    names = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = [name for name in names if name not in DOMAIN_SPEC_BY_NAME]
    if unknown:
        valid = sorted(DOMAIN_SPEC_BY_NAME)
        raise ValueError(f"Unknown domains: {unknown}; expected one of {valid}")
    resolved: list[DomainSpec] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        resolved.append(DOMAIN_SPEC_BY_NAME[name])
    return tuple(resolved)


def _make_preprocess_data_loader(*, dataset: str, source_path: str) -> Callable[[], pd.DataFrame]:
    loaded_data: pd.DataFrame | None = None

    def _load() -> pd.DataFrame:
        nonlocal loaded_data
        if loaded_data is None:
            log.info("[%s] reading %s", dataset, source_path)
            loaded_data = read_preprocess_csv(source_path, require_split_indices=True)
        return loaded_data

    return _load


def _iter_domain_cells(data: pd.DataFrame, column_names: tuple[str, ...]) -> Iterable[str]:
    for column_name in column_names:
        if column_name not in data.columns:
            continue
        for value in data[column_name].to_numpy():
            if pd.notna(value):
                text = str(value).strip()
                if text:
                    yield text


def _require_resolved_bool_flag(args: argparse.Namespace, name: str, cli_names: str) -> bool:
    raw = getattr(args, name, None)
    if raw is None:
        raise ValueError(f"{cli_names} is required from the resolved preprocess_c runtime transport.")
    return bool(raw)


def _resolve_precision_config(args: argparse.Namespace, device: torch.device) -> PrecisionConfig:
    bf16_enabled = _require_resolved_bool_flag(args, "bf16_enabled", "--bf16/--no-bf16")
    tf32_enabled = _require_resolved_bool_flag(args, "tf32_enabled", "--tf32/--no-tf32")
    autocast_enabled = bool(device.type == "cuda" and bf16_enabled and torch.cuda.is_bf16_supported())
    if bf16_enabled and device.type != "cuda":
        log.info("bf16 requested but CUDA is unavailable; preprocess_c will run in fp32.")
    elif bf16_enabled and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        log.info("bf16 requested but torch.cuda.is_bf16_supported() is False; preprocess_c will run in fp32.")
    return PrecisionConfig(
        bf16_enabled=bf16_enabled,
        tf32_enabled=tf32_enabled,
        autocast_enabled=autocast_enabled,
        autocast_dtype=torch.bfloat16,
    )


def _resolve_token_window_cache_config(args: argparse.Namespace) -> TokenWindowCacheConfig:
    enabled = _require_resolved_bool_flag(
        args,
        "token_window_cache_enabled",
        "--token-window-cache/--no-token-window-cache",
    )
    if getattr(args, "token_window_cache_dir", None) is None:
        raise ValueError("--token-window-cache-dir is required from the resolved preprocess_c payload.")
    if getattr(args, "token_window_cache_version", None) is None:
        raise ValueError("--token-window-cache-version is required from the resolved preprocess_c payload.")
    if getattr(args, "token_window_cache_shard_size", None) is None:
        raise ValueError("--token-window-cache-shard-size is required from the resolved preprocess_c payload.")
    cache_dir = str(args.token_window_cache_dir).strip()
    version = str(args.token_window_cache_version).strip()
    shard_size = int(args.token_window_cache_shard_size)
    if not cache_dir:
        raise ValueError("token_window_cache_dir must be non-empty when token-window cache is configured")
    if not version:
        raise ValueError("token_window_cache_version must be non-empty when token-window cache is configured")
    if shard_size <= 0:
        raise ValueError("token_window_cache_shard_size must be positive")
    return TokenWindowCacheConfig(enabled=enabled, cache_dir=cache_dir, version=version, shard_size=shard_size)


def _resolve_tokenizer_path_config(args: argparse.Namespace) -> TokenizerPathConfig:
    return TokenizerPathConfig(
        hotpath_enabled=_require_resolved_bool_flag(
            args,
            "tokenizer_hotpath_enabled",
            "--tokenizer-hotpath/--no-tokenizer-hotpath",
        )
    )


def _resolve_probe_config(args: argparse.Namespace) -> ProbeConfig:
    probe_only = bool(getattr(args, "probe_only", False))
    max_chunks = getattr(args, "probe_max_chunks_per_domain", None)
    if max_chunks is not None:
        max_chunks = int(max_chunks)
        if max_chunks <= 0:
            raise ValueError("probe_max_chunks_per_domain must be positive when provided")
        if not probe_only:
            raise ValueError("--probe-max-chunks-per-domain requires --probe-only")
    return ProbeConfig(
        probe_only=probe_only,
        max_chunks_per_domain=max_chunks,
    )


def _required_int_arg(args: argparse.Namespace, name: str, cli_name: str) -> int:
    raw = getattr(args, name, None)
    if raw is None:
        raise ValueError(f"{cli_name} is required from the resolved preprocess_c payload.")
    return int(raw)


def _resolve_tokenizer_pipeline_config(args: argparse.Namespace) -> TokenizerPipelineConfig:
    config = TokenizerPipelineConfig(
        tokenizer_parallelism_enabled=_require_resolved_bool_flag(
            args,
            "tokenizer_parallelism_enabled",
            "--tokenizer-parallelism/--no-tokenizer-parallelism",
        ),
        tokenizer_threads_per_worker=_required_int_arg(
            args,
            "tokenizer_threads_per_worker",
            "--tokenizer-threads-per-worker",
        ),
        tokenizer_total_threads=_required_int_arg(args, "tokenizer_total_threads", "--tokenizer-total-threads"),
        prefetch_batches=_required_int_arg(args, "prefetch_batches", "--prefetch-batches"),
        pin_memory=_require_resolved_bool_flag(args, "pin_memory", "--pin-memory/--no-pin-memory"),
        non_blocking_h2d=_require_resolved_bool_flag(
            args,
            "non_blocking_h2d",
            "--non-blocking-h2d/--no-non-blocking-h2d",
        ),
        async_prefetch_enabled=_require_resolved_bool_flag(
            args,
            "async_prefetch_enabled",
            "--async-prefetch/--no-async-prefetch",
        ),
        cpu_cores_reserved=_required_int_arg(args, "cpu_cores_reserved", "--cpu-cores-reserved"),
        cpu_cores_available=_required_int_arg(args, "cpu_cores_available", "--cpu-cores-available"),
    )
    usable_cpu = int(config.cpu_cores_available) - int(config.cpu_cores_reserved)
    if config.tokenizer_threads_per_worker <= 0 or config.tokenizer_total_threads <= 0:
        raise ValueError("tokenizer thread counts must be positive")
    if config.prefetch_batches < 0:
        raise ValueError("prefetch_batches must be non-negative")
    if config.async_prefetch_enabled and config.prefetch_batches <= 0:
        raise ValueError("async_prefetch_enabled requires prefetch_batches > 0")
    if usable_cpu <= 0 or config.tokenizer_total_threads > usable_cpu:
        raise ValueError("tokenizer_total_threads must fit within cpu_cores_available - cpu_cores_reserved")
    return config


def _install_tokenizer_thread_env(config: TokenizerPipelineConfig) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if config.tokenizer_parallelism_enabled else "false"
    os.environ["RAYON_NUM_THREADS"] = str(int(config.tokenizer_threads_per_worker))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["ODCR_RESOLVED_TOKENIZER_PARALLELISM"] = "1" if config.tokenizer_parallelism_enabled else "0"
    os.environ["ODCR_RESOLVED_TOKENIZER_THREADS_PER_WORKER"] = str(int(config.tokenizer_threads_per_worker))
    os.environ["ODCR_RESOLVED_TOKENIZER_TOTAL_THREADS"] = str(int(config.tokenizer_total_threads))
    os.environ["ODCR_RESOLVED_PREFETCH_BATCHES"] = str(int(config.prefetch_batches))


def _tokenizer_pipeline_summary(config: TokenizerPipelineConfig) -> dict[str, object]:
    return {
        "tokenizer_threads": int(config.tokenizer_threads_per_worker),
        "tokenizer_threads_per_worker": int(config.tokenizer_threads_per_worker),
        "tokenizer_total_threads": int(config.tokenizer_total_threads),
        "prefetch_batches": int(config.prefetch_batches),
        "cpu_cores_available": int(config.cpu_cores_available),
        "cpu_cores_reserved": int(config.cpu_cores_reserved),
        "cpu_cores_configured": int(config.cpu_cores_configured),
        "tokenizers_parallelism_enabled": bool(config.tokenizer_parallelism_enabled),
        "pin_memory": bool(config.pin_memory),
        "non_blocking_h2d": bool(config.non_blocking_h2d),
        "async_prefetch_enabled": bool(config.async_prefetch_enabled),
    }


def _configure_tf32(device: torch.device, *, enabled: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    log.info("preprocess_c TF32 matmul/cudnn = %s", "ON" if enabled else "OFF")


def _token_window_budget(tokenizer: AutoTokenizer, *, max_total_tokens: int) -> int:
    budget = int(max_total_tokens) - int(tokenizer.num_special_tokens_to_add(pair=False))
    if budget <= 0:
        raise ValueError(
            f"Tokenizer special token overhead leaves no payload budget under max_total_tokens={max_total_tokens}"
        )
    return budget


def _tokenizer_identity(tokenizer: AutoTokenizer) -> dict[str, object]:
    return {
        "class": tokenizer.__class__.__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0) or 0),
        "vocab_size": int(len(tokenizer)),
        "special_tokens_overhead": int(tokenizer.num_special_tokens_to_add(pair=False)),
        "transformers_version": str(TRANSFORMERS_VERSION),
    }


def _fingerprint_file(path: str | Path) -> dict[str, object]:
    p = Path(path).expanduser().resolve()
    stat = p.stat()
    hasher = hashlib.sha256()
    with p.open("rb") as handle:
        hasher.update(handle.read(FILE_FINGERPRINT_SAMPLE_BYTES))
        if stat.st_size > FILE_FINGERPRINT_SAMPLE_BYTES:
            tail_offset = max(int(stat.st_size) - FILE_FINGERPRINT_SAMPLE_BYTES, 0)
            handle.seek(tail_offset)
            hasher.update(handle.read(FILE_FINGERPRINT_SAMPLE_BYTES))
    return {
        "path": str(p),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": hasher.hexdigest(),
    }


def _canonical_column_hash() -> str:
    payload = {
        "processed": list(expected_preprocess_column_order()),
        "split": list(expected_preprocess_column_order(require_split_indices=True)),
        "merged": list(expected_preprocess_column_order(require_split_indices=True, require_domain=True)),
        "contract": render_preprocess_contract_snapshot(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _token_window_cache_fingerprint(
    *,
    dataset: str,
    spec: DomainSpec,
    source_path: str,
    tokenizer: AutoTokenizer,
    max_total_tokens: int,
    payload_budget: int,
    cache_version: str,
    probe_chunk_limit: int | None,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "domain": spec.name,
        "source_file": _fingerprint_file(source_path),
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "canonical_column_hash": _canonical_column_hash(),
        "canonical_text_source_contract": {
            "content_columns": list(CONTENT_COLS),
            "style_columns": list(STYLE_COLS),
            "selected_columns": list(spec.column_names),
        },
        "tokenizer": _tokenizer_identity(tokenizer),
        "sentence_embed_model": {
            "local_dir": str(Path(_require_sentence_embed_model_dir()).resolve()),
            "artifact_fingerprint": model_artifact_fingerprint(_require_sentence_embed_model_dir()),
            "odcr_embed_dim": int(_resolved_embed_dim()),
        },
        "max_total_tokens": int(max_total_tokens),
        "payload_budget": int(payload_budget),
        "chunking_contract_version": TOKEN_WINDOW_CACHE_CONTRACT_VERSION,
        "token_window_cache_version": str(cache_version),
        "tokenizer_hotpath_enabled": True,
        "prepend_space_between_cells": True,
        "empty_text_placeholder": EMPTY_TEXT_PLACEHOLDER,
        "cache_scope": "probe" if probe_chunk_limit is not None else "full",
        "probe_chunk_limit": None if probe_chunk_limit is None else int(probe_chunk_limit),
    }


def _cache_digest(fingerprint: dict[str, object]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _token_window_cache_path(
    cache_config: TokenWindowCacheConfig,
    *,
    dataset: str,
    spec: DomainSpec,
    fingerprint: dict[str, object],
) -> Path:
    base_dir = Path(cache_config.cache_dir).expanduser().resolve()
    _ = dataset, spec
    return base_dir / _cache_digest(fingerprint)


def _token_window_cache_manifest_path(cache_path: Path) -> Path:
    return cache_path / "manifest.json"


def _token_window_cache_shard_path(cache_path: Path, shard_index: int) -> Path:
    return cache_path / f"shard_{int(shard_index):05d}.pt"


def _load_token_window_cache_manifest(
    cache_path: Path,
    *,
    expected_fingerprint: dict[str, object],
    payload_budget: int,
    timing: DomainTiming | None = None,
) -> TokenWindowCacheManifest | None:
    started_at = time.perf_counter()
    manifest_path = _token_window_cache_manifest_path(cache_path)
    try:
        if not manifest_path.exists():
            if timing is not None:
                timing.cache_status = "miss"
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            log.warning("[cache] failed to load %s: %s", manifest_path, exc)
            if timing is not None:
                timing.cache_status = "stale"
            return None
    finally:
        if timing is not None:
            timing.token_window_cache_manifest_s += time.perf_counter() - started_at
    if not isinstance(payload, dict):
        log.warning("[cache] invalid payload type in %s: %s", manifest_path, type(payload).__name__)
        if timing is not None:
            timing.cache_status = "stale"
        return None
    if payload.get("fingerprint") != expected_fingerprint:
        log.warning("[cache] fingerprint mismatch for %s; cache will be rebuilt.", cache_path)
        if timing is not None:
            timing.cache_status = "stale"
        return None
    if str(payload.get("cache_contract_version")) != TOKEN_WINDOW_CACHE_CONTRACT_VERSION:
        log.warning("[cache] contract mismatch for %s; cache will be rebuilt.", cache_path)
        return None
    shard_size = int(payload.get("shard_size") or 0)
    shard_count = int(payload.get("shard_count") or 0)
    window_count = int(payload.get("window_count") or 0)
    manifest_payload_budget = int(payload.get("payload_budget") or 0)
    if manifest_payload_budget != payload_budget:
        log.warning(
            "[cache] payload_budget mismatch for %s: manifest=%s expected=%s",
            cache_path,
            manifest_payload_budget,
            payload_budget,
        )
        if timing is not None:
            timing.cache_status = "stale"
        return None
    if shard_size <= 0 or shard_count < 0 or window_count < 0:
        log.warning("[cache] invalid manifest values in %s: %s", cache_path, payload)
        if timing is not None:
            timing.cache_status = "stale"
        return None
    for shard_index in range(shard_count):
        shard_path = _token_window_cache_shard_path(cache_path, shard_index)
        if not shard_path.exists():
            log.warning("[cache] missing shard %s for %s", shard_path.name, cache_path)
            if timing is not None:
                timing.cache_status = "stale"
            return None
    if timing is not None:
        timing.cache_status = "hit"
    return TokenWindowCacheManifest(
        fingerprint=expected_fingerprint,
        payload_budget=payload_budget,
        shard_size=shard_size,
        shard_count=shard_count,
        window_count=window_count,
        cache_path=str(cache_path),
    )


def _iter_piece_text_batches_from_cells(
    cell_iter: Iterable[str],
    *,
    batch_size: int = DEFAULT_TOKENIZER_BATCH_SIZE,
    prepend_space_between_cells: bool = True,
) -> Iterable[list[str]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    batch: list[str] = []
    first_cell = True
    for raw_cell in cell_iter:
        cell = str(raw_cell).strip()
        if not cell:
            continue
        if first_cell or not prepend_space_between_cells:
            batch.append(cell)
        else:
            batch.append(f" {cell}")
        first_cell = False
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_piece_batch(tokenizer: AutoTokenizer, batch_text: list[str]) -> list[list[int]]:
    if not batch_text:
        return []
    encoded = tokenizer(
        batch_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        verbose=False,
    )
    return [list(piece_ids) for piece_ids in encoded["input_ids"]]


def _build_token_window_shard(
    token_windows: list[list[int]],
    *,
    payload_budget: int,
    shard_index: int,
) -> TokenWindowShard:
    window_count = len(token_windows)
    window_ids = np.zeros((window_count, payload_budget), dtype=np.int32)
    window_lengths = np.zeros(window_count, dtype=np.int32)
    for row_idx, window in enumerate(token_windows):
        window_lengths[row_idx] = len(window)
        if window:
            window_ids[row_idx, : len(window)] = np.asarray(window, dtype=np.int32)
    return TokenWindowShard(
        window_ids=window_ids,
        window_lengths=window_lengths,
        shard_index=shard_index,
    )


def _iter_token_window_shards_from_cells(
    *,
    cell_iter: Iterable[str],
    tokenizer: AutoTokenizer,
    max_total_tokens: int,
    shard_size: int,
    batch_size: int = DEFAULT_TOKENIZER_BATCH_SIZE,
    max_chunks: int | None = None,
    prepend_space_between_cells: bool = True,
    timing: DomainTiming | None = None,
) -> Iterable[TokenWindowShard]:
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")
    payload_budget = _token_window_budget(tokenizer, max_total_tokens=max_total_tokens)
    buffer: list[int] = []
    shard_windows: list[list[int]] = []
    emitted_windows = 0
    shard_index = 0
    seen_any_cell = False
    stop_requested = False

    def flush_shard(*, force: bool) -> TokenWindowShard | None:
        nonlocal shard_windows, shard_index
        if not shard_windows:
            return None
        if not force and len(shard_windows) < shard_size:
            return None
        shard = _build_token_window_shard(
            shard_windows,
            payload_budget=payload_budget,
            shard_index=shard_index,
        )
        shard_windows = []
        shard_index += 1
        return shard

    def push_window(window: list[int]) -> TokenWindowShard | None:
        nonlocal emitted_windows, stop_requested, shard_windows
        if max_chunks is not None and emitted_windows >= int(max_chunks):
            stop_requested = True
            return None
        shard_windows.append(list(window))
        emitted_windows += 1
        if max_chunks is not None and emitted_windows >= int(max_chunks):
            stop_requested = True
        return flush_shard(force=False)

    for batch_text in _iter_piece_text_batches_from_cells(
        cell_iter,
        batch_size=batch_size,
        prepend_space_between_cells=prepend_space_between_cells,
    ):
        seen_any_cell = True
        tokenize_started_at = time.perf_counter()
        encoded_batch = _encode_piece_batch(tokenizer, batch_text)
        if timing is not None:
            timing.tokenize_s += time.perf_counter() - tokenize_started_at
        build_started_at = time.perf_counter()
        for piece_ids in encoded_batch:
            if not piece_ids:
                continue
            start = 0
            while start < len(piece_ids):
                remaining = payload_budget - len(buffer)
                if remaining <= 0:
                    maybe_shard = push_window(buffer)
                    buffer = []
                    remaining = payload_budget
                    if maybe_shard is not None:
                        yield maybe_shard
                    if stop_requested:
                        break
                take = min(remaining, len(piece_ids) - start)
                buffer.extend(piece_ids[start : start + take])
                start += take
                if len(buffer) == payload_budget:
                    maybe_shard = push_window(buffer)
                    buffer = []
                    if maybe_shard is not None:
                        yield maybe_shard
                    if stop_requested:
                        break
            if stop_requested:
                break
        if timing is not None:
            timing.token_window_cache_build_s += time.perf_counter() - build_started_at
        if stop_requested:
            break

    if not seen_any_cell:
        buffer.extend(_placeholder_payload_ids(tokenizer, payload_budget=payload_budget))

    if buffer and not stop_requested:
        maybe_shard = push_window(buffer)
        if maybe_shard is not None:
            yield maybe_shard

    final_shard = flush_shard(force=True)
    if final_shard is not None:
        yield final_shard


def _write_token_window_cache_shard(
    cache_root: Path,
    *,
    shard: TokenWindowShard,
    payload_budget: int,
) -> None:
    atomic_torch_save(
        _token_window_cache_shard_path(cache_root, shard.shard_index),
        {
            "shard_index": int(shard.shard_index),
            "payload_budget": int(payload_budget),
            "window_ids": shard.window_ids,
            "window_lengths": shard.window_lengths,
        },
    )


def _iter_and_maybe_cache_token_window_shards(
    *,
    shard_iter: Iterable[TokenWindowShard],
    cache_path: Path | None,
    fingerprint: dict[str, object] | None,
    payload_budget: int,
    shard_size: int,
    persist_cache: bool,
    dataset: str,
    spec: DomainSpec,
    timing: DomainTiming | None = None,
) -> Iterable[TokenWindowShard]:
    tmp_cache_dir: Path | None = None
    shard_count = 0
    window_count = 0
    if persist_cache:
        if cache_path is None or fingerprint is None:
            raise ValueError("cache_path and fingerprint are required when persist_cache=True")
        tmp_cache_dir = cache_path.parent / f".{cache_path.name}.tmp.{uuid.uuid4().hex}"
        shutil.rmtree(tmp_cache_dir, ignore_errors=True)
        tmp_cache_dir.mkdir(parents=True, exist_ok=False)
    try:
        for shard in shard_iter:
            if shard_count == 0:
                log.info(
                    "[hotpath][%s][%s] first_shard_ready token_windows=%d persist_cache=%s shard_size=%d",
                    dataset,
                    spec.name,
                    shard.window_count,
                    persist_cache,
                    shard_size,
                )
            if tmp_cache_dir is not None:
                write_started_at = time.perf_counter()
                _write_token_window_cache_shard(
                    tmp_cache_dir,
                    shard=shard,
                    payload_budget=payload_budget,
                )
                if timing is not None:
                    timing.token_window_cache_write_s += time.perf_counter() - write_started_at
                    timing.cache_shards_written += 1
            shard_count += 1
            window_count += shard.window_count
            yield shard
        if tmp_cache_dir is None:
            return
        write_started_at = time.perf_counter()
        atomic_write_json(
            _token_window_cache_manifest_path(tmp_cache_dir),
            {
                "cache_contract_version": TOKEN_WINDOW_CACHE_CONTRACT_VERSION,
                "fingerprint": fingerprint,
                "payload_budget": int(payload_budget),
                "shard_size": int(shard_size),
                "shard_count": int(shard_count),
                "window_count": int(window_count),
            },
        )
        if timing is not None:
            timing.token_window_cache_write_s += time.perf_counter() - write_started_at
            timing.cache_status = "miss_written"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            log.info(
                "[cache][%s][%s] concurrent writer already populated dir=%s; discard temp cache.",
                dataset,
                spec.name,
                cache_path,
            )
            shutil.rmtree(tmp_cache_dir, ignore_errors=True)
            return
        os.replace(str(tmp_cache_dir), str(cache_path))
        log.info(
            "[cache][%s][%s] wrote dir=%s shards=%d token_windows=%d shard_size=%d",
            dataset,
            spec.name,
            cache_path,
            shard_count,
            window_count,
            shard_size,
        )
    except Exception:
        if tmp_cache_dir is not None:
            shutil.rmtree(tmp_cache_dir, ignore_errors=True)
        raise


def _load_cached_token_window_shard(
    cache_path: Path,
    *,
    shard_index: int,
    payload_budget: int,
    timing: DomainTiming | None = None,
) -> TokenWindowShard:
    shard_path = _token_window_cache_shard_path(cache_path, shard_index)
    started_at = time.perf_counter()
    try:
        try:
            payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(shard_path), map_location="cpu")
    except Exception as exc:
        raise ValueError(f"failed to load shard {shard_path}: {exc}") from exc
    if timing is not None:
        timing.token_window_cache_load_s += time.perf_counter() - started_at
        timing.cache_shards_loaded += 1
    if not isinstance(payload, dict):
        raise ValueError(f"invalid shard payload type in {shard_path}: {type(payload).__name__}")
    if int(payload.get("payload_budget") or 0) != payload_budget:
        raise ValueError(
            f"payload_budget mismatch in {shard_path}: manifest expects {payload_budget}, shard has {payload.get('payload_budget')}"
        )
    window_ids = np.asarray(payload.get("window_ids"))
    window_lengths = np.asarray(payload.get("window_lengths"))
    if window_ids.ndim != 2 or window_ids.shape[1] != payload_budget:
        raise ValueError(f"invalid window_ids shape in {shard_path}: {window_ids.shape}")
    if window_lengths.ndim != 1 or window_lengths.shape[0] != window_ids.shape[0]:
        raise ValueError(f"invalid window_lengths shape in {shard_path}: {window_lengths.shape}")
    return TokenWindowShard(
        window_ids=window_ids.astype(np.int32, copy=False),
        window_lengths=window_lengths.astype(np.int32, copy=False),
        shard_index=int(payload.get("shard_index") or shard_index),
        cache_path=str(shard_path),
        cache_hit=True,
    )


def _iter_cached_token_window_shards(
    cache_path: Path,
    *,
    manifest: TokenWindowCacheManifest,
    max_chunks: int | None = None,
    timing: DomainTiming | None = None,
) -> Iterable[TokenWindowShard]:
    remaining = None if max_chunks is None else int(max_chunks)
    for shard_index in range(manifest.shard_count):
        if remaining is not None and remaining <= 0:
            break
        shard = _load_cached_token_window_shard(
            cache_path,
            shard_index=shard_index,
            payload_budget=manifest.payload_budget,
            timing=timing,
        )
        if remaining is not None and shard.window_count > remaining:
            shard = TokenWindowShard(
                window_ids=shard.window_ids[:remaining],
                window_lengths=shard.window_lengths[:remaining],
                shard_index=shard.shard_index,
                cache_path=shard.cache_path,
                cache_hit=shard.cache_hit,
            )
        yield shard
        if remaining is not None:
            remaining -= shard.window_count


def _placeholder_payload_ids(tokenizer: AutoTokenizer, *, payload_budget: int) -> list[int]:
    placeholder = _encode_piece_batch(tokenizer, [EMPTY_TEXT_PLACEHOLDER])[0]
    if not placeholder:
        unk_id = tokenizer.unk_token_id
        if unk_id is None:
            raise ValueError("Tokenizer returned no ids for empty_text and has no unk_token_id fallback")
        placeholder = [int(unk_id)]
    return placeholder[:payload_budget]


def _prefetch_token_window_shards(
    shard_iter: Iterable[TokenWindowShard],
    *,
    timing: DomainTiming,
    prefetch_batches: int,
) -> Iterable[TokenWindowShard]:
    item_queue: queue.Queue[object] = queue.Queue(maxsize=max(1, int(prefetch_batches)))
    sentinel = object()

    def producer() -> None:
        try:
            for shard in shard_iter:
                put_started_at = time.perf_counter()
                item_queue.put(shard)
                timing.tokenizer_queue_wait_s += time.perf_counter() - put_started_at
        except Exception as exc:  # pragma: no cover - surfaced to consumer
            item_queue.put(exc)
        finally:
            item_queue.put(sentinel)

    thread = threading.Thread(target=producer, name="preprocess-c-token-window-prefetch", daemon=True)
    thread.start()
    while True:
        get_started_at = time.perf_counter()
        item = item_queue.get()
        timing.gpu_queue_wait_s += time.perf_counter() - get_started_at
        if item is sentinel:
            break
        if isinstance(item, Exception):
            thread.join()
            raise item
        yield item  # type: ignore[misc]
    thread.join()


def _legacy_tokenize_piece(tokenizer: AutoTokenizer, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    return list(encoded["input_ids"])


def _legacy_iter_token_windows_from_cells(
    cell_iter: Iterable[str],
    tokenizer: AutoTokenizer,
    *,
    max_total_tokens: int,
) -> Iterable[list[int]]:
    payload_budget = _token_window_budget(tokenizer, max_total_tokens=max_total_tokens)
    buffer: list[int] = []
    seen_any = False
    first_cell = True

    for raw_cell in cell_iter:
        cell = str(raw_cell).strip()
        if not cell:
            continue
        piece = cell if first_cell else f" {cell}"
        first_cell = False
        seen_any = True
        piece_ids = _legacy_tokenize_piece(tokenizer, piece)
        if not piece_ids:
            continue
        start = 0
        while start < len(piece_ids):
            remaining = payload_budget - len(buffer)
            if remaining <= 0:
                yield buffer
                buffer = []
                remaining = payload_budget
            take = min(remaining, len(piece_ids) - start)
            buffer.extend(piece_ids[start : start + take])
            start += take
            if len(buffer) == payload_budget:
                yield buffer
                buffer = []

    if not seen_any:
        placeholder = _legacy_tokenize_piece(tokenizer, EMPTY_TEXT_PLACEHOLDER)
        if not placeholder:
            unk_id = tokenizer.unk_token_id
            if unk_id is None:
                raise ValueError("Tokenizer returned no ids for empty_text and has no unk_token_id fallback")
            placeholder = [int(unk_id)]
        buffer.extend(placeholder[:payload_budget])

    if buffer:
        yield buffer


def _streaming_chunk_mean_embedding_from_shards(
    *,
    shard_iter: Iterable[TokenWindowShard],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    hidden_size: int,
    desc: str,
    forward_batch_size: int,
    max_total_tokens: int,
    precision_config: PrecisionConfig,
    total_chunks: int | None = None,
    pipeline_config: TokenizerPipelineConfig | None = None,
    timing: DomainTiming | None = None,
) -> tuple[np.ndarray, int]:
    sum_vec = torch.zeros(hidden_size, dtype=torch.float64, device="cpu")
    chunk_count = 0
    progress = tqdm(total=total_chunks, desc=desc, unit="chunk")
    iter_for_forward = (
        _prefetch_token_window_shards(
            shard_iter,
            timing=timing,
            prefetch_batches=pipeline_config.prefetch_batches,
        )
        if timing is not None and pipeline_config is not None and pipeline_config.async_prefetch_enabled
        else shard_iter
    )
    encode_started_at = time.perf_counter()
    for shard in iter_for_forward:
        for start in range(0, shard.window_count, forward_batch_size):
            end = min(start + forward_batch_size, shard.window_count)
            pooled = _forward_attention_mask_mean_pool(
                shard.window_ids[start:end],
                shard.window_lengths[start:end],
                tokenizer,
                model,
                device,
                max_total_tokens=max_total_tokens,
                precision_config=precision_config,
                pipeline_config=pipeline_config,
                timing=timing,
            )
            batch_count = end - start
            sum_vec += pooled.detach().cpu().to(torch.float64).sum(dim=0)
            chunk_count += batch_count
            if timing is not None:
                timing.token_windows += int(batch_count)
            progress.update(batch_count)
    progress.close()
    if timing is not None:
        timing.encode_wall_s += time.perf_counter() - encode_started_at
    if chunk_count == 0:
        return np.zeros(hidden_size, dtype=np.float32), 0
    mean = (sum_vec / float(chunk_count)).to(torch.float32).numpy()
    return mean.astype(np.float32, copy=False), int(chunk_count)


def _compute_hotpath_domain_embedding(
    *,
    data_loader: Callable[[], pd.DataFrame],
    spec: DomainSpec,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    hidden_size: int,
    chunk_batch_size: int,
    max_total_tokens: int,
    dataset: str,
    source_path: str,
    cache_config: TokenWindowCacheConfig,
    precision_config: PrecisionConfig,
    pipeline_config: TokenizerPipelineConfig,
    timing: DomainTiming,
    max_chunks_per_domain: int | None = None,
) -> tuple[np.ndarray, int]:
    payload_budget = _token_window_budget(tokenizer, max_total_tokens=max_total_tokens)
    fingerprint: dict[str, object] | None = None
    cache_path: Path | None = None

    if cache_config.enabled:
        fingerprint = _token_window_cache_fingerprint(
            dataset=dataset,
            spec=spec,
            source_path=source_path,
            tokenizer=tokenizer,
            max_total_tokens=max_total_tokens,
            payload_budget=payload_budget,
            cache_version=cache_config.version,
            probe_chunk_limit=max_chunks_per_domain,
        )
        cache_path = _token_window_cache_path(
            cache_config,
            dataset=dataset,
            spec=spec,
            fingerprint=fingerprint,
        )
        cached_manifest = _load_token_window_cache_manifest(
            cache_path,
            expected_fingerprint=fingerprint,
            payload_budget=payload_budget,
            timing=timing,
        )
        if cached_manifest is not None:
            total_chunks = _effective_chunk_count(cached_manifest.window_count, max_chunks=max_chunks_per_domain)
            log.info(
                "[cache][%s][%s] hit dir=%s shards=%d token_windows=%d",
                dataset,
                spec.name,
                cache_path,
                cached_manifest.shard_count,
                cached_manifest.window_count,
            )
            try:
                return _streaming_chunk_mean_embedding_from_shards(
                    shard_iter=_iter_cached_token_window_shards(
                        cache_path,
                        manifest=cached_manifest,
                        max_chunks=max_chunks_per_domain,
                        timing=timing,
                    ),
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    hidden_size=hidden_size,
                    desc=f"{dataset} {spec.name} token-windows",
                    forward_batch_size=chunk_batch_size,
                    max_total_tokens=max_total_tokens,
                    precision_config=precision_config,
                    total_chunks=total_chunks,
                    pipeline_config=pipeline_config,
                    timing=timing,
                )
            except Exception as exc:
                log.warning(
                    "[cache][%s][%s] failed while streaming %s: %s; cache will be rebuilt.",
                    dataset,
                    spec.name,
                    cache_path,
                    exc,
                )
                shutil.rmtree(cache_path, ignore_errors=True)
        if cache_path.exists():
            log.info("[cache][%s][%s] miss dir=%s reason=rebuild", dataset, spec.name, cache_path)
            if cache_path.is_dir():
                shutil.rmtree(cache_path, ignore_errors=True)
            else:
                cache_path.unlink(missing_ok=True)
        else:
            log.info("[cache][%s][%s] miss dir=%s reason=not_found", dataset, spec.name, cache_path)

    read_started_at = time.perf_counter()
    data = data_loader()
    timing.csv_read_s += time.perf_counter() - read_started_at
    persist_cache = bool(cache_config.enabled)
    shard_iter = _iter_token_window_shards_from_cells(
        cell_iter=_iter_domain_cells(data, spec.column_names),
        tokenizer=tokenizer,
        max_total_tokens=max_total_tokens,
        shard_size=cache_config.shard_size,
        max_chunks=max_chunks_per_domain,
        timing=timing,
    )
    cached_iter = _iter_and_maybe_cache_token_window_shards(
        shard_iter=shard_iter,
        cache_path=cache_path,
        fingerprint=fingerprint,
        payload_budget=payload_budget,
        shard_size=cache_config.shard_size,
        persist_cache=persist_cache,
        dataset=dataset,
        spec=spec,
        timing=timing,
    )
    return _streaming_chunk_mean_embedding_from_shards(
        shard_iter=cached_iter,
        tokenizer=tokenizer,
        model=model,
        device=device,
        hidden_size=hidden_size,
        desc=f"{dataset} {spec.name} token-windows",
        forward_batch_size=chunk_batch_size,
        max_total_tokens=max_total_tokens,
        precision_config=precision_config,
        total_chunks=None,
        pipeline_config=pipeline_config,
        timing=timing,
    )


def _prepare_chunk_batch(
    token_windows: np.ndarray,
    token_lengths: np.ndarray,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    max_total_tokens: int,
    pipeline_config: TokenizerPipelineConfig | None = None,
    timing: DomainTiming | None = None,
) -> dict[str, torch.Tensor]:
    if token_windows.ndim != 2:
        raise ValueError(f"token_windows must be rank-2, got shape={token_windows.shape}")
    if token_lengths.ndim != 1 or token_lengths.shape[0] != token_windows.shape[0]:
        raise ValueError(
            f"token_lengths shape mismatch: token_windows={token_windows.shape} token_lengths={token_lengths.shape}"
        )

    built_windows: list[list[int]] = []
    max_batch_tokens = 0
    for row_idx in range(token_windows.shape[0]):
        window_len = int(token_lengths[row_idx])
        payload_ids = token_windows[row_idx, :window_len].tolist()
        features = tokenizer.build_inputs_with_special_tokens(payload_ids)
        if len(features) > max_total_tokens:
            raise ValueError(f"Prepared chunk exceeds max_total_tokens={max_total_tokens}: {len(features)}")
        built_windows.append(features)
        max_batch_tokens = max(max_batch_tokens, len(features))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.sep_token_id
    if pad_token_id is None:
        pad_token_id = 0

    input_ids = torch.full((len(built_windows), max_batch_tokens), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(built_windows), max_batch_tokens), dtype=torch.long)
    for row_idx, features in enumerate(built_windows):
        row = torch.tensor(features, dtype=torch.long)
        input_ids[row_idx, : row.numel()] = row
        attention_mask[row_idx, : row.numel()] = 1
    if pipeline_config is not None and pipeline_config.pin_memory and device.type == "cuda":
        input_ids = input_ids.pin_memory()
        attention_mask = attention_mask.pin_memory()
    h2d_started_at = time.perf_counter()
    non_blocking = bool(
        pipeline_config is not None and pipeline_config.non_blocking_h2d and device.type == "cuda"
    )
    out = {
        "input_ids": input_ids.to(device, non_blocking=non_blocking),
        "attention_mask": attention_mask.to(device, non_blocking=non_blocking),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timing is not None:
        timing.h2d_s += time.perf_counter() - h2d_started_at
    return out


def _legacy_prepare_chunk_batch(
    token_windows: list[list[int]],
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    max_total_tokens: int,
    pipeline_config: TokenizerPipelineConfig | None = None,
    timing: DomainTiming | None = None,
) -> dict[str, torch.Tensor]:
    prepared = []
    for window_ids in token_windows:
        features = tokenizer.prepare_for_model(
            window_ids,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        if len(features["input_ids"]) > max_total_tokens:
            raise ValueError(
                f"Prepared chunk exceeds max_total_tokens={max_total_tokens}: {len(features['input_ids'])}"
            )
        prepared.append(features)
    batch = tokenizer.pad(prepared, padding=True, return_tensors="pt")
    if pipeline_config is not None and pipeline_config.pin_memory and device.type == "cuda":
        batch = {key: value.pin_memory() for key, value in batch.items()}
    h2d_started_at = time.perf_counter()
    non_blocking = bool(
        pipeline_config is not None and pipeline_config.non_blocking_h2d and device.type == "cuda"
    )
    out = {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timing is not None:
        timing.h2d_s += time.perf_counter() - h2d_started_at
    return out


def _effective_chunk_count(chunk_count: int, *, max_chunks: int | None) -> int:
    total = int(chunk_count)
    if max_chunks is None:
        return total
    return min(total, int(max_chunks))


def _forward_attention_mask_mean_pool(
    token_windows: np.ndarray,
    token_lengths: np.ndarray,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    *,
    max_total_tokens: int,
    precision_config: PrecisionConfig,
    pipeline_config: TokenizerPipelineConfig | None = None,
    timing: DomainTiming | None = None,
) -> torch.Tensor:
    inputs = _prepare_chunk_batch(
        token_windows,
        token_lengths,
        tokenizer,
        device,
        max_total_tokens=max_total_tokens,
        pipeline_config=pipeline_config,
        timing=timing,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_started_at = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=precision_config.autocast_dtype,
            enabled=precision_config.autocast_enabled,
        ):
            outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timing is not None:
        timing.gpu_forward_s += time.perf_counter() - forward_started_at
    hidden = outputs.last_hidden_state
    attn = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1.0)
    return pooled.to(torch.float32)


def _legacy_forward_attention_mask_mean_pool(
    token_windows: list[list[int]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    *,
    max_total_tokens: int,
    precision_config: PrecisionConfig,
    pipeline_config: TokenizerPipelineConfig | None = None,
    timing: DomainTiming | None = None,
) -> torch.Tensor:
    inputs = _legacy_prepare_chunk_batch(
        token_windows,
        tokenizer,
        device,
        max_total_tokens=max_total_tokens,
        pipeline_config=pipeline_config,
        timing=timing,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_started_at = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=precision_config.autocast_dtype,
            enabled=precision_config.autocast_enabled,
        ):
            outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timing is not None:
        timing.gpu_forward_s += time.perf_counter() - forward_started_at
    hidden = outputs.last_hidden_state
    attn = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1.0)
    return pooled.to(torch.float32)


def _legacy_streaming_chunk_mean_embedding(
    *,
    chunk_iter: Iterable[list[int]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    hidden_size: int,
    desc: str,
    forward_batch_size: int,
    max_total_tokens: int,
    precision_config: PrecisionConfig,
    max_chunks: int | None = None,
    pipeline_config: TokenizerPipelineConfig | None = None,
    timing: DomainTiming | None = None,
) -> tuple[np.ndarray, int]:
    sum_vec = torch.zeros(hidden_size, dtype=torch.float64, device="cpu")
    chunk_count = 0
    batch: list[list[int]] = []
    progress = tqdm(desc=desc, unit="chunk")
    target_chunk_count = max_chunks

    def flush_batch() -> None:
        nonlocal sum_vec, chunk_count, batch
        if not batch:
            return
        pooled = _legacy_forward_attention_mask_mean_pool(
            batch,
            tokenizer,
            model,
            device,
            max_total_tokens=max_total_tokens,
            precision_config=precision_config,
            pipeline_config=pipeline_config,
            timing=timing,
        )
        sum_vec += pooled.detach().cpu().to(torch.float64).sum(dim=0)
        chunk_count += pooled.shape[0]
        if timing is not None:
            timing.token_windows += int(pooled.shape[0])
        progress.update(len(batch))
        batch = []

    encode_started_at = time.perf_counter()
    for token_window in chunk_iter:
        if target_chunk_count is not None and chunk_count >= target_chunk_count:
            break
        batch.append(token_window)
        if len(batch) >= forward_batch_size:
            if target_chunk_count is not None:
                remaining = target_chunk_count - chunk_count
                if remaining < len(batch):
                    batch = batch[:remaining]
            flush_batch()
            if target_chunk_count is not None and chunk_count >= target_chunk_count:
                break
    flush_batch()
    progress.close()
    if timing is not None:
        timing.encode_wall_s += time.perf_counter() - encode_started_at

    if chunk_count == 0:
        return np.zeros(hidden_size, dtype=np.float32), 0
    mean = (sum_vec / float(chunk_count)).to(torch.float32).numpy()
    return mean.astype(np.float32, copy=False), int(chunk_count)


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


def _compute_domain_embedding(
    *,
    data_loader: Callable[[], pd.DataFrame],
    spec: DomainSpec,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    hidden_size: int,
    chunk_batch_size: int,
    max_total_tokens: int,
    dataset: str,
    source_path: str,
    cache_config: TokenWindowCacheConfig,
    precision_config: PrecisionConfig,
    tokenizer_path_config: TokenizerPathConfig,
    pipeline_config: TokenizerPipelineConfig,
    timing: DomainTiming,
    max_chunks_per_domain: int | None = None,
) -> tuple[np.ndarray, int]:
    if not tokenizer_path_config.hotpath_enabled:
        read_started_at = time.perf_counter()
        data = data_loader()
        timing.csv_read_s += time.perf_counter() - read_started_at
        chunk_iter = _legacy_iter_token_windows_from_cells(
            _iter_domain_cells(data, spec.column_names),
            tokenizer,
            max_total_tokens=max_total_tokens,
        )
        return _legacy_streaming_chunk_mean_embedding(
            chunk_iter=chunk_iter,
            tokenizer=tokenizer,
            model=model,
            device=device,
            hidden_size=hidden_size,
            desc=f"{dataset} {spec.name} token-windows",
            forward_batch_size=chunk_batch_size,
            max_total_tokens=max_total_tokens,
            precision_config=precision_config,
            max_chunks=max_chunks_per_domain,
            pipeline_config=pipeline_config,
            timing=timing,
        )

    return _compute_hotpath_domain_embedding(
        data_loader=data_loader,
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        device=device,
        hidden_size=hidden_size,
        chunk_batch_size=chunk_batch_size,
        max_total_tokens=max_total_tokens,
        dataset=dataset,
        source_path=source_path,
        cache_config=cache_config,
        precision_config=precision_config,
        pipeline_config=pipeline_config,
        timing=timing,
        max_chunks_per_domain=max_chunks_per_domain,
    )


def _output_path(dataset: str, output_name: str) -> str:
    return os.path.join(_resolved_data_dir(), dataset, output_name)


def _validate_domain_output(stored: np.ndarray, *, output_name: str, hidden_size: int) -> None:
    if stored.shape != (hidden_size,):
        raise ValueError(f"{output_name} shape mismatch: got {stored.shape}, expected ({hidden_size},)")
    if stored.dtype != np.float32:
        raise ValueError(f"{output_name} dtype mismatch: got {stored.dtype}, expected float32")


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs64 = lhs.astype(np.float64, copy=False)
    rhs64 = rhs.astype(np.float64, copy=False)
    denom = float(np.linalg.norm(lhs64) * np.linalg.norm(rhs64))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(lhs64, rhs64) / denom)


def _verify_outputs(
    *,
    dataset: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    hidden_size: int,
    chunk_batch_size: int,
    max_total_tokens: int,
    source_path: str,
    cache_config: TokenWindowCacheConfig,
    precision_config: PrecisionConfig,
    tokenizer_path_config: TokenizerPathConfig,
    pipeline_config: TokenizerPipelineConfig,
    domain_specs: tuple[DomainSpec, ...],
) -> dict[str, object]:
    data_loader = _make_preprocess_data_loader(dataset=dataset, source_path=source_path)
    report: dict[str, object] = {"dataset": dataset, "artifacts": {}}
    for spec in domain_specs:
        output_path = _output_path(dataset, spec.output_name)
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Missing verify target: {output_path}")
        stored = np.load(output_path)
        _validate_domain_output(stored, output_name=spec.output_name, hidden_size=hidden_size)
        timing = DomainTiming()
        started_at = time.perf_counter()
        recomputed, chunk_count = _compute_domain_embedding(
            data_loader=data_loader,
            spec=spec,
            tokenizer=tokenizer,
            model=model,
            device=device,
            hidden_size=hidden_size,
            chunk_batch_size=chunk_batch_size,
            max_total_tokens=max_total_tokens,
            dataset=dataset,
            source_path=source_path,
            cache_config=cache_config,
            precision_config=precision_config,
            tokenizer_path_config=tokenizer_path_config,
            pipeline_config=pipeline_config,
            timing=timing,
            max_chunks_per_domain=None,
        )
        timing.total_s = time.perf_counter() - started_at
        allclose_ok = bool(np.allclose(stored, recomputed, atol=1e-5, rtol=1e-4))
        max_abs_diff = float(np.max(np.abs(stored - recomputed)))
        cosine = _cosine_similarity(stored, recomputed)
        if not allclose_ok:
            raise ValueError(
                f"{spec.output_name} verify failed (max_abs_diff={max_abs_diff:.6g}, cosine={cosine:.6f})"
            )
        report["artifacts"][spec.output_name] = {
            "path": output_path,
            "shape": list(stored.shape),
            "dtype": str(stored.dtype),
            "chunk_count": int(chunk_count),
            "allclose": allclose_ok,
            "cosine": cosine,
            "max_abs_diff": max_abs_diff,
            "phase_timing": timing.to_dict(),
        }
    return report


def _sum_domain_timings(phase_timings: list[DomainTiming]) -> DomainTiming:
    out = DomainTiming()
    for item in phase_timings:
        out.tokenize_s += item.tokenize_s
        out.h2d_s += item.h2d_s
        out.gpu_forward_s += item.gpu_forward_s
        out.encode_wall_s += item.encode_wall_s
    return out


def _log_domain_phase_summary(
    *,
    dataset: str,
    spec: DomainSpec,
    timing: DomainTiming,
    mode: str,
    device: torch.device,
    pipeline_config: TokenizerPipelineConfig,
) -> None:
    payload = {
        "dataset": dataset,
        "domain": spec.name,
        "mode": mode,
        "device": str(device),
        **_tokenizer_pipeline_summary(pipeline_config),
        **timing.to_dict(),
    }
    log.info("[phase][%s][%s] %s", dataset, spec.name, json.dumps(payload, sort_keys=True))


def _log_dataset_phase_summary(
    *,
    dataset: str,
    phase_timings: list[DomainTiming],
    mode: str,
    device: torch.device,
    pipeline_config: TokenizerPipelineConfig,
) -> None:
    summary_timing = _sum_domain_timings(phase_timings)
    payload = {
        "dataset": dataset,
        "mode": mode,
        "device": str(device),
        **_tokenizer_pipeline_summary(pipeline_config),
        "csv_read_s": round(float(sum(item.csv_read_s for item in phase_timings)), 6),
        "token_window_cache_manifest_s": round(
            float(sum(item.token_window_cache_manifest_s for item in phase_timings)),
            6,
        ),
        "token_window_cache_load_s": round(float(sum(item.token_window_cache_load_s for item in phase_timings)), 6),
        "token_window_cache_write_s": round(float(sum(item.token_window_cache_write_s for item in phase_timings)), 6),
        "token_window_cache_build_s": round(float(sum(item.token_window_cache_build_s for item in phase_timings)), 6),
        "tokenizer_queue_wait_s": round(float(sum(item.tokenizer_queue_wait_s for item in phase_timings)), 6),
        "gpu_queue_wait_s": round(float(sum(item.gpu_queue_wait_s for item in phase_timings)), 6),
        "h2d_s": round(float(summary_timing.h2d_s), 6),
        "tokenize_s": round(float(summary_timing.tokenize_s), 6),
        "gpu_forward_s": round(float(summary_timing.gpu_forward_s), 6),
        "encode_wall_s": round(float(summary_timing.encode_wall_s), 6),
        "overlap_effectiveness": round(float(_overlap_effectiveness(summary_timing)), 6),
        "total_s": round(float(sum(item.total_s for item in phase_timings)), 6),
        "token_windows": int(sum(item.token_windows for item in phase_timings)),
        "token_window_cache_hits": int(sum(1 for item in phase_timings if item.cache_status == "hit")),
        "token_window_cache_misses": int(
            sum(1 for item in phase_timings if item.cache_status in {"miss", "miss_written", "stale"})
        ),
        "token_window_cache_shards_loaded": int(sum(item.cache_shards_loaded for item in phase_timings)),
        "token_window_cache_shards_written": int(sum(item.cache_shards_written for item in phase_timings)),
    }
    log.info("[phase-summary][%s] %s", dataset, json.dumps(payload, sort_keys=True))


def _run_infer(
    *,
    dataset: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    hidden_size: int,
    chunk_batch_size: int,
    max_total_tokens: int,
    source_path: str,
    cache_config: TokenWindowCacheConfig,
    precision_config: PrecisionConfig,
    tokenizer_path_config: TokenizerPathConfig,
    pipeline_config: TokenizerPipelineConfig,
    domain_specs: tuple[DomainSpec, ...],
    probe_config: ProbeConfig,
) -> None:
    data_dir = os.path.join(_resolved_data_dir(), dataset)
    os.makedirs(data_dir, exist_ok=True)
    data_loader = _make_preprocess_data_loader(dataset=dataset, source_path=source_path)
    phase_timings: list[DomainTiming] = []
    for spec in domain_specs:
        timing = DomainTiming()
        started_at = time.perf_counter()
        embedding, chunk_count = _compute_domain_embedding(
            data_loader=data_loader,
            spec=spec,
            tokenizer=tokenizer,
            model=model,
            device=device,
            hidden_size=hidden_size,
            chunk_batch_size=chunk_batch_size,
            max_total_tokens=max_total_tokens,
            dataset=dataset,
            source_path=source_path,
            cache_config=cache_config,
            precision_config=precision_config,
            tokenizer_path_config=tokenizer_path_config,
            pipeline_config=pipeline_config,
            timing=timing,
            max_chunks_per_domain=probe_config.max_chunks_per_domain,
        )
        timing.total_s = time.perf_counter() - started_at
        phase_timings.append(timing)
        _log_domain_phase_summary(
            dataset=dataset,
            spec=spec,
            timing=timing,
            mode="probe" if probe_config.probe_only else "write",
            device=device,
            pipeline_config=pipeline_config,
        )
        if probe_config.probe_only:
            log.info(
                "[probe][%s] skipped write for %s with token_window_count=%d max_total_tokens=%d",
                dataset,
                spec.output_name,
                chunk_count,
                max_total_tokens,
            )
            continue
        atomic_save_numpy(_output_path(dataset, spec.output_name), embedding.astype(np.float32, copy=False))
        log.info(
            "[%s] wrote %s with token_window_count=%d max_total_tokens=%d",
            dataset,
            spec.output_name,
            chunk_count,
            max_total_tokens,
        )
    _log_dataset_phase_summary(
        dataset=dataset,
        phase_timings=phase_timings,
        mode="probe" if probe_config.probe_only else "write",
        device=device,
        pipeline_config=pipeline_config,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Infer domain_content / domain_style with token-aware windowing. "
            "Chunks are built directly from tokenizer payload tokens, then pooled with attention-mask-aware mean pooling."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        metavar="N",
        help="CUDA device id (default: 0). Formal preprocess_c fails fast when CUDA is unavailable.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Resolved project.data_dir from ./odcr; required for formal preprocess_c.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Resolved env.models_dir from ./odcr; required for formal preprocess_c.",
    )
    parser.add_argument(
        "--sentence-embed-model",
        type=str,
        default=None,
        help="Resolved env.sentence_embed_model path from ./odcr; required for formal preprocess_c.",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Resolved env.embed_dim from ./odcr; required for formal preprocess_c.",
    )
    parser.add_argument(
        "--allow-cpu-debug",
        action="store_true",
        default=False,
        help="Explicit test-only CPU debug mode. Formal preprocess_c must not use this flag.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names. Defaults to all datasets.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated domain names to process. Defaults to content,style.",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Number of token windows to forward per batch from the resolved preprocess_c payload.",
    )
    parser.add_argument("--tokenizer-parallelism", action="store_true", default=None, dest="tokenizer_parallelism_enabled")
    parser.add_argument("--no-tokenizer-parallelism", action="store_false", dest="tokenizer_parallelism_enabled")
    parser.add_argument("--tokenizer-threads-per-worker", type=int, default=None, metavar="N")
    parser.add_argument("--tokenizer-total-threads", type=int, default=None, metavar="N")
    parser.add_argument("--prefetch-batches", type=int, default=None, metavar="N")
    parser.add_argument("--pin-memory", action="store_true", default=None, dest="pin_memory")
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--non-blocking-h2d", action="store_true", default=None, dest="non_blocking_h2d")
    parser.add_argument("--no-non-blocking-h2d", action="store_false", dest="non_blocking_h2d")
    parser.add_argument("--async-prefetch", action="store_true", default=None, dest="async_prefetch_enabled")
    parser.add_argument("--no-async-prefetch", action="store_false", dest="async_prefetch_enabled")
    parser.add_argument("--cpu-cores-reserved", type=int, default=None, metavar="N")
    parser.add_argument("--cpu-cores-available", type=int, default=None, metavar="N")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read existing domain_*.npy, recompute with the current token-aware contract, and emit a verification report.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run selected domains as a throughput probe and do not write domain_*.npy outputs.",
    )
    parser.add_argument(
        "--probe-max-chunks-per-domain",
        type=int,
        default=None,
        metavar="N",
        help="When probing, only forward the first N token windows for each selected domain.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=None,
        dest="bf16_enabled",
        help="Enable preprocess_c CUDA bf16 autocast (default: on).",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="bf16_enabled",
        help="Disable preprocess_c CUDA bf16 autocast.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        default=None,
        dest="tf32_enabled",
        help="Enable preprocess_c TF32 matmul/cudnn (default: on).",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_false",
        dest="tf32_enabled",
        help="Disable preprocess_c TF32 matmul/cudnn.",
    )
    parser.add_argument(
        "--token-window-cache",
        action="store_true",
        default=None,
        dest="token_window_cache_enabled",
        help="Enable token-window cache reuse for preprocess_c (default: on).",
    )
    parser.add_argument(
        "--no-token-window-cache",
        action="store_false",
        dest="token_window_cache_enabled",
        help="Disable token-window cache reuse.",
    )
    parser.add_argument(
        "--token-window-cache-dir",
        type=str,
        default=None,
        help="Directory for preprocess_c token-window cache payloads.",
    )
    parser.add_argument(
        "--token-window-cache-version",
        type=str,
        default=None,
        help="Token-window cache version salt. Change this to force a clean cache miss.",
    )
    parser.add_argument(
        "--token-window-cache-shard-size",
        type=int,
        default=None,
        metavar="N",
        help="Number of token windows per cache shard for preprocess_c hot-path streaming.",
    )
    parser.add_argument(
        "--tokenizer-hotpath",
        action="store_true",
        default=None,
        dest="tokenizer_hotpath_enabled",
        help="Enable batched tokenizer/manual-pad hot path for preprocess_c (default: on).",
    )
    parser.add_argument(
        "--no-tokenizer-hotpath",
        action="store_false",
        dest="tokenizer_hotpath_enabled",
        help="Disable the tokenizer hot path and fall back to the legacy tokenizer/pad path.",
    )
    args = parser.parse_args()
    _install_resolved_preprocess_context(args)

    device = _select_cuda_device_or_fail(args)

    chunk_batch_size = args.chunk_batch_size
    if chunk_batch_size is None:
        raise ValueError("--chunk-batch-size is required from the resolved preprocess_c payload.")
    chunk_batch_size = max(1, int(chunk_batch_size))
    pipeline_config = _resolve_tokenizer_pipeline_config(args)
    _install_tokenizer_thread_env(pipeline_config)
    precision_config = _resolve_precision_config(args, device)
    cache_config = _resolve_token_window_cache_config(args)
    tokenizer_path_config = _resolve_tokenizer_path_config(args)
    probe_config = _resolve_probe_config(args)
    domain_specs = _resolve_domain_specs(args.domains)
    _configure_tf32(device, enabled=precision_config.tf32_enabled)
    if not tokenizer_path_config.hotpath_enabled and cache_config.enabled:
        log.info("tokenizer hot path is disabled; token-window cache will be ignored for this run.")
        cache_config = TokenWindowCacheConfig(
            enabled=False,
            cache_dir=cache_config.cache_dir,
            version=cache_config.version,
            shard_size=cache_config.shard_size,
        )
    if args.verify_only and probe_config.probe_only:
        raise ValueError("--verify-only cannot be combined with --probe-only")

    log.info(
        "infer_domain_semantics config: device=%s chunk_batch_size=%s max_total_tokens=%s verify_only=%s probe_only=%s probe_max_chunks=%s domains=%s bf16=%s autocast=%s tf32=%s tokenizer_hotpath=%s token_window_cache=%s cache_dir=%s cache_version=%s cache_shard_size=%s tokenizer_parallelism=%s tokenizer_threads_per_worker=%s tokenizer_total_threads=%s prefetch_batches=%s pin_memory=%s non_blocking_h2d=%s async_prefetch=%s cpu_cores_available=%s cpu_cores_reserved=%s",
        device,
        chunk_batch_size,
        MAX_TOTAL_TOKENS,
        args.verify_only,
        probe_config.probe_only,
        probe_config.max_chunks_per_domain,
        [spec.name for spec in domain_specs],
        precision_config.bf16_enabled,
        precision_config.autocast_enabled,
        precision_config.tf32_enabled,
        tokenizer_path_config.hotpath_enabled,
        cache_config.enabled,
        cache_config.cache_dir,
        cache_config.version,
        cache_config.shard_size,
        pipeline_config.tokenizer_parallelism_enabled,
        pipeline_config.tokenizer_threads_per_worker,
        pipeline_config.tokenizer_total_threads,
        pipeline_config.prefetch_batches,
        pipeline_config.pin_memory,
        pipeline_config.non_blocking_h2d,
        pipeline_config.async_prefetch_enabled,
        pipeline_config.cpu_cores_available,
        pipeline_config.cpu_cores_reserved,
    )

    tokenizer, model = _load_sentence_embed_model(device)
    hidden_size = int(model.config.hidden_size)
    datasets = _resolve_datasets(args.datasets)

    for dataset in datasets:
        file_path = os.path.join(_resolved_data_dir(), dataset, "train.csv")

        if args.verify_only:
            report = _verify_outputs(
                dataset=dataset,
                tokenizer=tokenizer,
                model=model,
                device=device,
                hidden_size=hidden_size,
                chunk_batch_size=chunk_batch_size,
                max_total_tokens=MAX_TOTAL_TOKENS,
                source_path=file_path,
                cache_config=cache_config,
                precision_config=precision_config,
                tokenizer_path_config=tokenizer_path_config,
                pipeline_config=pipeline_config,
                domain_specs=domain_specs,
            )
            log.info("[verify][%s] %s", dataset, json.dumps(report, ensure_ascii=False, sort_keys=True))
            continue

        _run_infer(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            device=device,
            hidden_size=hidden_size,
            chunk_batch_size=chunk_batch_size,
            max_total_tokens=MAX_TOTAL_TOKENS,
            source_path=file_path,
            cache_config=cache_config,
            precision_config=precision_config,
            tokenizer_path_config=tokenizer_path_config,
            pipeline_config=pipeline_config,
            domain_specs=domain_specs,
            probe_config=probe_config,
        )


if __name__ == "__main__":
    main()
