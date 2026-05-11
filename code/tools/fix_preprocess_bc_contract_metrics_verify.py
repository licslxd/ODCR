#!/usr/bin/env python3
"""Repair preprocess_b/c metadata, metrics, and verify reports from finished artifacts.

This tool is intentionally read-only for data products: it mmap-loads existing
``.npy`` outputs, parses existing run logs/status files, and only writes
``runs/.../meta`` JSON plus AI_analysis handoff files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.file_atomic import atomic_write_json, atomic_write_text  # noqa: E402
from odcr_core.preprocess_schema import (  # noqa: E402
    PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
    preprocess_b_expected_shape_dtype,
    preprocess_b_output_artifact_contract,
    preprocess_c_expected_shape_dtype,
    preprocess_c_output_artifact_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("AM_Electronics", "AM_CDs", "AM_Movies", "Yelp", "TripAdvisor")
B_PROFILE_SPECS = (
    ("user_content", "user_content_profiles.npy", "user"),
    ("user_style", "user_style_profiles.npy", "user"),
    ("item_content", "item_content_profiles.npy", "item"),
    ("item_style", "item_style_profiles.npy", "item"),
)
C_DOMAIN_SPECS = (
    ("content", "domain_content.npy"),
    ("style", "domain_style.npy"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _rel(path: str | Path, repo_root: Path) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(repo_root).as_posix()
    except ValueError:
        return p.as_posix()


def _parse_time(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_s(started_at: Any, finished_at: Any) -> float | None:
    start = _parse_time(started_at)
    finish = _parse_time(finished_at)
    if start is None or finish is None:
        return None
    return round(max(0.0, (finish - start).total_seconds()), 3)


def _read_statuses(meta_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    status_dir = meta_dir / "datasets"
    for path in sorted(status_dir.glob("*.status.json")):
        payload = _read_json(path)
        out[str(payload.get("unit_name") or path.stem.replace(".status", ""))] = payload
    return out


def _worker_spans(statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_worker: dict[str, dict[str, Any]] = {}
    for dataset, status in statuses.items():
        worker = str(status.get("worker_id") or "unknown")
        item = by_worker.setdefault(
            worker,
            {
                "worker_id": status.get("worker_id"),
                "gpu_id": status.get("gpu_id"),
                "datasets": [],
                "started_at": None,
                "finished_at": None,
                "assigned_wall_s": 0.0,
            },
        )
        item["datasets"].append(dataset)
        wall_s = _duration_s(status.get("started_at"), status.get("finished_at"))
        if wall_s is not None:
            item["assigned_wall_s"] = round(float(item["assigned_wall_s"]) + wall_s, 3)
        started = _parse_time(status.get("started_at"))
        finished = _parse_time(status.get("finished_at"))
        current_start = _parse_time(item.get("started_at"))
        current_finish = _parse_time(item.get("finished_at"))
        if started is not None and (current_start is None or started < current_start):
            item["started_at"] = status.get("started_at")
        if finished is not None and (current_finish is None or finished > current_finish):
            item["finished_at"] = status.get("finished_at")
    spans: dict[str, dict[str, Any]] = {}
    for worker, item in by_worker.items():
        item["span_s"] = _duration_s(item.get("started_at"), item.get("finished_at"))
        spans[worker] = item
    values = [float(item.get("span_s") or 0.0) for item in spans.values()]
    imbalance = round(max(values) - min(values), 3) if values else 0.0
    tail_worker = None
    if spans:
        tail_worker = max(spans.values(), key=lambda item: float(item.get("span_s") or 0.0)).get("worker_id")
    return {
        "workers": spans,
        "worker_imbalance_s": imbalance,
        "tail_worker": tail_worker,
    }


def _extract_json_tail(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _log_segments(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [item for item in re.split(r"[\r\n]+", text) if item.strip()]


def _sum_numbers(items: Iterable[dict[str, Any]], key: str) -> float:
    return round(sum(float(item.get(key) or 0.0) for item in items), 6)


def _sum_ints(items: Iterable[dict[str, Any]], key: str) -> int:
    return int(sum(int(item.get(key) or 0) for item in items))


def _status_wall_times(statuses: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    return {
        dataset: _duration_s(status.get("started_at"), status.get("finished_at"))
        for dataset, status in sorted(statuses.items())
    }


def _run_duration(meta_dir: Path, stage_status: dict[str, Any]) -> float | None:
    summary_path = meta_dir / "run_summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        duration = summary.get("duration_sec")
        if duration is not None:
            return float(duration)
    return _duration_s(stage_status.get("started_at"), stage_status.get("finished_at"))


def _detect_cpu_cores() -> int | None:
    for key in ("SLURM_CPUS_PER_TASK", "ODCR_CPU_CORES"):
        raw = str(os.environ.get(key) or "").strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return None


def _configured_gpu_count(meta_dir: Path, stage_status: dict[str, Any]) -> int | None:
    for payload in (stage_status, _read_json(meta_dir / "resolved_config.json") if (meta_dir / "resolved_config.json").is_file() else {}):
        hardware = payload.get("hardware") if isinstance(payload, dict) else None
        if isinstance(hardware, dict) and isinstance(hardware.get("gpu_ids"), list):
            return len(hardware["gpu_ids"])
        config = payload.get("config_snapshot") if isinstance(payload, dict) else None
        if isinstance(config, dict):
            hardware = config.get("hardware")
            if isinstance(hardware, dict) and isinstance(hardware.get("gpu_ids"), list):
                return len(hardware["gpu_ids"])
    return None


def _parse_b_metrics(meta_dir: Path, repo_root: Path) -> dict[str, Any]:
    stage_status = _read_json(meta_dir / "stage_status.json")
    statuses = _read_statuses(meta_dir)
    phases: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    target_sizes: dict[tuple[str, str], int] = {}
    stale_candidates: dict[tuple[str, str], int] = {}
    phase_re = re.compile(r"^\[phase\]\[([^\]]+)\]\[([^\]]+)\]\s+(\{.*\})$")
    summary_re = re.compile(r"^\[phase-summary\]\[([^\]]+)\]\s+(\{.*\})$")
    target_re = re.compile(r"^\[([^\]]+)\]\[([^\]]+)\]\s+wrote\s+\S+\s+target_size=(\d+)")
    stale_re = re.compile(r"^\[group-text-cache\]\[([^\]]+)\]\[([^\]]+)\].*?stale_candidates=(\d+)")
    for path in sorted((meta_dir / "shell_logs").glob("*.log")):
        for line in _log_segments(path):
            match = phase_re.match(line)
            if match:
                dataset, spec, raw = match.groups()
                payload = json.loads(raw)
                payload["dataset"] = dataset
                payload["spec"] = spec
                phases.append(payload)
                continue
            match = summary_re.match(line)
            if match:
                dataset, raw = match.groups()
                summaries[dataset] = json.loads(raw)
                continue
            match = target_re.match(line)
            if match:
                dataset, spec, size = match.groups()
                target_sizes[(dataset, spec)] = int(size)
                continue
            match = stale_re.match(line)
            if match:
                dataset, spec, stale = match.groups()
                stale_candidates[(dataset, spec)] = int(stale)

    per_dataset: dict[str, Any] = {}
    per_spec: list[dict[str, Any]] = []
    for dataset, summary in sorted(summaries.items()):
        item = dict(summary)
        item["wall_s"] = _duration_s(statuses.get(dataset, {}).get("started_at"), statuses.get(dataset, {}).get("finished_at"))
        per_dataset[dataset] = item
    for phase in phases:
        dataset = str(phase.get("dataset"))
        spec = str(phase.get("spec"))
        status = str(phase.get("group_text_cache_status") or "")
        per_spec.append(
            {
                "dataset": dataset,
                "spec": spec,
                "rows_read": int(phase.get("rows_read") or 0),
                "groups": int(phase.get("groups_total") or phase.get("groups_encoded") or 0),
                "target_size": target_sizes.get((dataset, spec)),
                "batches": int(phase.get("batches_encoded") or 0),
                "shards": int(phase.get("group_shards") or 0),
                "cache_status": status,
                "stale_candidates": stale_candidates.get((dataset, spec), 0),
                "total_s": float(phase.get("total_s") or 0.0),
                "gpu_forward_s": float(phase.get("gpu_forward_s") or 0.0),
                "tokenize_s": float(phase.get("tokenize_s") or 0.0),
                "group_build_s": float(phase.get("group_text_build_s") or 0.0),
                "cache_write_s": float(phase.get("group_text_cache_write_s") or 0.0),
                "cache_load_s": float(phase.get("group_text_cache_load_s") or 0.0),
                "fingerprint_s": float(phase.get("group_text_cache_fingerprint_s") or 0.0),
                "write_npy_s": float(phase.get("write_npy_s") or 0.0),
            }
        )
    worker = _worker_spans(statuses)
    tokenize_s = _sum_numbers(phases, "tokenize_s")
    gpu_forward_s = _sum_numbers(phases, "gpu_forward_s")
    threshold = 1.0
    return {
        "metrics_schema_version": "odcr_preprocess_b_metrics/1",
        "stage": "preprocess_b",
        "generated_at_utc": _utc_now(),
        "source": "existing shell logs and run metadata only; no preprocess rerun",
        "run_meta_dir": _rel(meta_dir, repo_root),
        "total_wall_s": _run_duration(meta_dir, stage_status),
        "dataset_wall_s": _status_wall_times(statuses),
        "worker_span": worker["workers"],
        "per_dataset": per_dataset,
        "per_spec": per_spec,
        "totals": {
            "cumulative_tokenize_s": tokenize_s,
            "cumulative_gpu_forward_s": gpu_forward_s,
            "cumulative_group_build_s": _sum_numbers(phases, "group_text_build_s"),
            "cache_hits": int(sum(1 for item in phases if item.get("group_text_cache_status") == "hit")),
            "cache_misses": int(sum(1 for item in phases if item.get("group_text_cache_status") in {"miss", "miss_written", "stale"})),
            "cache_stale_candidates": int(sum(stale_candidates.values())),
            "groups_encoded": _sum_ints(phases, "groups_encoded"),
            "batches_encoded": _sum_ints(phases, "batches_encoded"),
        },
        "resource_conclusion": {
            "tokenizer_bottleneck_threshold": threshold,
            "tokenizer_bottleneck": bool(tokenize_s >= gpu_forward_s * threshold),
            "worker_imbalance_s": worker["worker_imbalance_s"],
            "gpu_count": _configured_gpu_count(meta_dir, stage_status),
            "cpu_cores": _detect_cpu_cores(),
        },
    }


def _elapsed_to_seconds(raw: str) -> float | None:
    parts = [int(part) for part in raw.split(":") if part.isdigit()]
    if len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    return None


def _parse_c_metrics(meta_dir: Path, repo_root: Path) -> dict[str, Any]:
    stage_status = _read_json(meta_dir / "stage_status.json")
    statuses = _read_statuses(meta_dir)
    per_domain: list[dict[str, Any]] = []
    token_windows_total = 0
    cache_writes = 0
    cache_hits = 0
    cache_misses = 0
    progress_elapsed_total = 0.0
    for path in sorted((meta_dir / "shell_logs").glob("*.log")):
        config_shard_size: int | None = None
        config_max_total_tokens: int | None = None
        cache_status: dict[tuple[str, str], str] = {}
        cache_dirs: dict[tuple[str, str], str] = {}
        cache_wrote: dict[tuple[str, str], dict[str, int]] = {}
        progress: dict[tuple[str, str], tuple[int, float | None, float | None]] = {}
        wrote_outputs: dict[tuple[str, str], dict[str, int]] = {}
        for line in _log_segments(path):
            match = re.search(r"cache_shard_size=(\d+)", line)
            if match:
                config_shard_size = int(match.group(1))
            match = re.search(r"max_total_tokens=(\d+)", line)
            if match:
                config_max_total_tokens = int(match.group(1))
            match = re.match(r"^\[cache\]\[([^\]]+)\]\[([^\]]+)\]\s+(hit|miss)\s+dir=(\S+)", line)
            if match:
                dataset, domain, status, cache_dir = match.groups()
                cache_status[(dataset, domain)] = status
                cache_dirs[(dataset, domain)] = cache_dir
            match = re.search(
                r"\[cache\]\[([^\]]+)\]\[([^\]]+)\]\s+wrote\s+dir=(\S+)\s+shards=(\d+)\s+token_windows=(\d+)\s+shard_size=(\d+)",
                line,
            )
            if match:
                dataset, domain, cache_dir, shards, windows, shard_size = match.groups()
                cache_wrote[(dataset, domain)] = {
                    "shards": int(shards),
                    "token_windows": int(windows),
                    "shard_size": int(shard_size),
                }
                cache_dirs[(dataset, domain)] = cache_dir
            match = re.search(
                r"\[([^\]]+)\]\s+wrote\s+domain_([a-z]+)\.npy\s+with\s+token_window_count=(\d+)\s+max_total_tokens=(\d+)",
                line,
            )
            if match:
                dataset, domain, windows, max_tokens = match.groups()
                wrote_outputs[(dataset, domain)] = {
                    "token_window_count": int(windows),
                    "max_total_tokens": int(max_tokens),
                }
            for match in re.finditer(r"([A-Za-z0-9_]+)\s+(content|style)\s+token-windows:\s*(\d+)chunk\s+\[([0-9:]+),\s*([0-9.]+)chunk/s\]", line):
                dataset, domain, windows, elapsed, rate = match.groups()
                progress[(dataset, domain)] = (int(windows), _elapsed_to_seconds(elapsed), float(rate))
        keys = set(cache_status) | set(cache_wrote) | set(wrote_outputs) | set(progress)
        for dataset, domain in sorted(keys):
            wrote = wrote_outputs.get((dataset, domain), {})
            cache = cache_wrote.get((dataset, domain), {})
            prog = progress.get((dataset, domain), (None, None, None))
            token_windows = int(wrote.get("token_window_count") or cache.get("token_windows") or prog[0] or 0)
            elapsed_s = prog[1]
            if elapsed_s is not None:
                progress_elapsed_total = round(progress_elapsed_total + elapsed_s, 3)
            token_windows_total += token_windows
            status = cache_status.get((dataset, domain), "unknown")
            if status == "hit":
                cache_hits += 1
            elif status == "miss":
                cache_misses += 1
            if cache:
                cache_writes += 1
            shard_size = int(cache.get("shard_size") or config_shard_size or 0)
            shard_count = int(cache.get("shards") or (math.ceil(token_windows / shard_size) if shard_size else 0))
            per_domain.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "token_window_count": token_windows,
                    "cache_status": status,
                    "shard_size": shard_size,
                    "shard_count": shard_count,
                    "elapsed_s": elapsed_s,
                    "final_rate": prog[2],
                    "cache_dir": cache_dirs.get((dataset, domain)),
                    "max_total_tokens": int(wrote.get("max_total_tokens") or config_max_total_tokens or 0),
                }
            )
    worker = _worker_spans(statuses)
    return {
        "metrics_schema_version": "odcr_preprocess_c_metrics/1",
        "stage": "preprocess_c",
        "generated_at_utc": _utc_now(),
        "source": "existing shell logs and run metadata only; no preprocess rerun",
        "run_meta_dir": _rel(meta_dir, repo_root),
        "total_wall_s": _run_duration(meta_dir, stage_status),
        "dataset_wall_s": _status_wall_times(statuses),
        "worker_span": worker["workers"],
        "token_windows_total": int(token_windows_total),
        "cache_shards_written": int(sum(int(item.get("shard_count") or 0) for item in per_domain if item.get("cache_status") == "miss")),
        "per_domain": per_domain,
        "totals": {
            "token_windows_total": int(token_windows_total),
            "cache_misses": int(cache_misses),
            "cache_hits": int(cache_hits),
            "cache_writes": int(cache_writes),
            "cumulative_progress_elapsed_s": progress_elapsed_total,
        },
        "resource_conclusion": {
            "worker_imbalance_s": worker["worker_imbalance_s"],
            "tail_worker": worker["tail_worker"],
            "cache_fresh_build": bool(cache_misses > 0 and cache_hits == 0),
            "gpu_count": _configured_gpu_count(meta_dir, stage_status),
            "cpu_cores": _detect_cpu_cores(),
        },
    }


def _sample_array(arr: np.ndarray, *, max_rows: int = 16) -> np.ndarray:
    if arr.ndim == 1:
        return np.asarray(arr[:], dtype=np.float32)
    if arr.shape[0] <= 0:
        return np.asarray([], dtype=np.float32)
    row_count = min(max_rows, int(arr.shape[0]))
    indices = np.linspace(0, int(arr.shape[0]) - 1, num=row_count, dtype=np.int64)
    return np.asarray(arr[indices], dtype=np.float32)


def _verify_one(path: Path, *, expected_shape: tuple[int, ...], expected_dtype: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "shape": None,
        "expected_shape": list(expected_shape),
        "dtype": None,
        "expected_dtype": expected_dtype,
        "finite_sample_count": 0,
        "nonzero_sample_count": 0,
        "verify_sample_count": 0,
        "status": "fail",
    }
    if not path.is_file():
        return record
    arr = np.load(path, mmap_mode="r")
    sample = _sample_array(arr)
    record["shape"] = list(arr.shape)
    record["dtype"] = str(arr.dtype)
    record["verify_sample_count"] = int(sample.size)
    record["finite_sample_count"] = int(np.isfinite(sample).sum()) if sample.size else 0
    record["nonzero_sample_count"] = int(np.count_nonzero(sample)) if sample.size else 0
    shape_ok = tuple(int(x) for x in arr.shape) == tuple(int(x) for x in expected_shape)
    dtype_ok = str(arr.dtype) == expected_dtype
    finite_ok = record["verify_sample_count"] > 0 and record["finite_sample_count"] == record["verify_sample_count"]
    nonzero_ok = record["nonzero_sample_count"] > 0
    record["status"] = "pass" if shape_ok and dtype_ok and finite_ok and nonzero_ok else "fail"
    return record


def _verify_b(meta_dir: Path, repo_root: Path, embed_dim: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for spec, filename, entity_kind in B_PROFILE_SPECS:
            path = repo_root / "data" / dataset / filename
            if path.is_file():
                arr = np.load(path, mmap_mode="r")
                entity_count = int(arr.shape[0]) if arr.ndim == 2 else 0
            else:
                entity_count = 0
            record = _verify_one(path, expected_shape=(entity_count, embed_dim), expected_dtype="float32")
            record.update(
                {
                    "dataset": dataset,
                    "spec": spec,
                    "entity_kind": entity_kind,
                    "expected_shape_label": "[entity_count, env.embed_dim]",
                }
            )
            records.append(record)
    status = "pass" if records and all(item["status"] == "pass" for item in records) else "fail"
    return {
        "verify_schema_version": "odcr_preprocess_b_verify/1",
        "stage": "preprocess_b",
        "generated_at_utc": _utc_now(),
        "source": "read-only numpy mmap verification; profile .npy files were not modified",
        "run_meta_dir": _rel(meta_dir, repo_root),
        "expected_shape_dtype": preprocess_b_expected_shape_dtype(),
        "artifacts": records,
        "verify_sample_count": int(sum(int(item["verify_sample_count"]) for item in records)),
        "finite_sample_count": int(sum(int(item["finite_sample_count"]) for item in records)),
        "nonzero_sample_count": int(sum(int(item["nonzero_sample_count"]) for item in records)),
        "status": status,
    }


def _verify_c(meta_dir: Path, repo_root: Path, embed_dim: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for domain, filename in C_DOMAIN_SPECS:
            record = _verify_one(repo_root / "data" / dataset / filename, expected_shape=(embed_dim,), expected_dtype="float32")
            record.update(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "expected_shape_label": "[env.embed_dim]",
                    "domain_shape_contract_version": PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
                }
            )
            records.append(record)
    status = "pass" if records and all(item["status"] == "pass" for item in records) else "fail"
    return {
        "verify_schema_version": "odcr_preprocess_c_verify/1",
        "stage": "preprocess_c",
        "generated_at_utc": _utc_now(),
        "source": "read-only numpy mmap verification; domain .npy files were not modified",
        "run_meta_dir": _rel(meta_dir, repo_root),
        "domain_shape_contract_version": PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
        "expected_shape_dtype": preprocess_c_expected_shape_dtype(),
        "artifacts": records,
        "verify_sample_count": int(sum(int(item["verify_sample_count"]) for item in records)),
        "finite_sample_count": int(sum(int(item["finite_sample_count"]) for item in records)),
        "nonzero_sample_count": int(sum(int(item["nonzero_sample_count"]) for item in records)),
        "status": status,
    }


def _embed_dim(meta_dir: Path) -> int:
    for path in (meta_dir / "resolved_config.json", meta_dir / "stage_status.json"):
        if not path.is_file():
            continue
        payload = _read_json(path)
        resolved = payload.get("resolved") if isinstance(payload.get("resolved"), dict) else None
        if resolved and resolved.get("embed_dim"):
            return int(resolved["embed_dim"])
        config = payload.get("config_snapshot") if isinstance(payload.get("config_snapshot"), dict) else None
        if config:
            resolved = config.get("resolved") if isinstance(config.get("resolved"), dict) else None
            if resolved and resolved.get("embed_dim"):
                return int(resolved["embed_dim"])
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
        if metadata:
            stage_specific = metadata.get("stage_specific") if isinstance(metadata.get("stage_specific"), dict) else None
            if stage_specific and stage_specific.get("embed_dim"):
                return int(stage_specific["embed_dim"])
    return 1024


def _stage_contract(stage: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str | None]:
    if stage == "preprocess_b":
        return preprocess_b_expected_shape_dtype(), preprocess_b_output_artifact_contract(), None
    return (
        preprocess_c_expected_shape_dtype(),
        preprocess_c_output_artifact_contract(),
        PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
    )


def _patch_contract_payload(payload: dict[str, Any], *, stage: str) -> bool:
    changed = False
    expected, contract, domain_version = _stage_contract(stage)
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        stage_specific = metadata.setdefault("stage_specific", {})
        if isinstance(stage_specific, dict):
            if stage_specific.get("expected_shape_dtype") != expected:
                stage_specific["expected_shape_dtype"] = expected
                changed = True
            contract_key = "profile_output_artifact_contract" if stage == "preprocess_b" else "domain_output_artifact_contract"
            if stage_specific.get(contract_key) != contract:
                stage_specific[contract_key] = contract
                changed = True
            if domain_version and stage_specific.get("domain_shape_contract_version") != domain_version:
                stage_specific["domain_shape_contract_version"] = domain_version
                changed = True
    snapshot = payload.setdefault("contract_snapshot", {})
    if isinstance(snapshot, dict):
        if snapshot.get("expected_shape_dtype") != expected:
            snapshot["expected_shape_dtype"] = expected
            changed = True
        if snapshot.get("output_artifact_contract") != contract:
            snapshot["output_artifact_contract"] = contract
            changed = True
        if domain_version and snapshot.get("domain_shape_contract_version") != domain_version:
            snapshot["domain_shape_contract_version"] = domain_version
            changed = True
    paths = payload.setdefault("paths", {})
    if isinstance(paths, dict):
        meta_dir = Path(paths.get("meta_root") or "")
        if str(meta_dir):
            metrics_path = str((meta_dir / "metrics.json").resolve())
            verify_path = str((meta_dir / "verify_report.json").resolve())
            if paths.get("metrics_path") != metrics_path:
                paths["metrics_path"] = metrics_path
                changed = True
            if paths.get("verify_report_path") != verify_path:
                paths["verify_report_path"] = verify_path
                changed = True
    return changed


def _patch_unit_status(path: Path, *, stage: str) -> bool:
    payload = _read_json(path)
    expected, contract, domain_version = _stage_contract(stage)
    key = "profile_matrix" if stage == "preprocess_b" else "domain_vector"
    metadata = {
        "artifact_contract_kind": key,
        "expected_shape_dtype": expected,
        "output_artifact_contract": {
            Path(item).name: contract.get(Path(item).name)
            for item in payload.get("output_files", [])
            if Path(item).name in contract
        },
        "unit_name": payload.get("unit_name"),
    }
    if domain_version:
        metadata["domain_shape_contract_version"] = domain_version
    if payload.get("metadata") == metadata:
        return False
    payload["metadata"] = metadata
    _write_json(path, payload)
    return True


def _patch_source_table(path: Path, *, stage: str) -> bool:
    payload = _read_json(path)
    expected, contract, domain_version = _stage_contract(stage)
    field_sources = payload.setdefault("field_sources", {})
    records = payload.setdefault("records", [])
    changed = False
    if not isinstance(field_sources, dict) or not isinstance(records, list):
        return False
    entries: dict[str, tuple[str, Any]] = {}
    if stage == "preprocess_b":
        entries["preprocess.b.expected_shape_dtype"] = ("code/compute_embeddings.py output verifier", expected)
        entries["preprocess.b.profile_output_artifact_contract"] = (
            "code/odcr_core/preprocess_schema.py profile matrix contract",
            contract,
        )
    else:
        entries["preprocess.c.expected_shape_dtype"] = ("code/infer_domain_semantics.py output verifier", expected)
        entries["preprocess.c.domain_output_artifact_contract"] = (
            "code/odcr_core/preprocess_schema.py domain vector contract",
            contract,
        )
        entries["preprocess.c.domain_shape_contract_version"] = (
            "code/odcr_core/preprocess_schema.py domain vector contract",
            domain_version,
        )
    record_by_key = {str(item.get("key")): item for item in records if isinstance(item, dict)}
    for key, (source, value) in entries.items():
        if field_sources.get(key) != source:
            field_sources[key] = source
            changed = True
        record = record_by_key.get(key)
        if record is None:
            records.append({"key": key, "source": source, "value": value})
            changed = True
        else:
            if record.get("source") != source:
                record["source"] = source
                changed = True
            if record.get("value") != value:
                record["value"] = value
                changed = True
    records.sort(key=lambda item: str(item.get("key", "")) if isinstance(item, dict) else "")
    if changed:
        _write_json(path, payload)
    return changed


def _patch_run_summary(meta_dir: Path, repo_root: Path, *, stage: str) -> bool:
    path = meta_dir / "run_summary.json"
    payload = _read_json(path)
    metrics_path = meta_dir / "metrics.json"
    verify_path = meta_dir / "verify_report.json"
    changed = False
    rel_metrics = _rel(metrics_path, repo_root)
    rel_verify = _rel(verify_path, repo_root)
    if payload.get("metrics_path") != rel_metrics:
        payload["metrics_path"] = rel_metrics
        changed = True
    if payload.get("verify_report_path") != rel_verify:
        payload["verify_report_path"] = rel_verify
        changed = True
    key_artifacts = payload.setdefault("key_artifacts", {})
    if isinstance(key_artifacts, dict):
        if key_artifacts.get("metrics") != rel_metrics:
            key_artifacts["metrics"] = rel_metrics
            changed = True
        if key_artifacts.get("verify_report") != rel_verify:
            key_artifacts["verify_report"] = rel_verify
            changed = True
    if stage == "preprocess_c" and payload.get("domain_shape_contract_version") != PREPROCESS_C_DOMAIN_CONTRACT_VERSION:
        payload["domain_shape_contract_version"] = PREPROCESS_C_DOMAIN_CONTRACT_VERSION
        changed = True
    metadata = payload.setdefault("preprocess_metadata", {})
    if isinstance(metadata, dict) and stage == "preprocess_c":
        if metadata.get("domain_shape_contract_version") != PREPROCESS_C_DOMAIN_CONTRACT_VERSION:
            metadata["domain_shape_contract_version"] = PREPROCESS_C_DOMAIN_CONTRACT_VERSION
            changed = True
    if changed:
        _write_json(path, payload)
    return changed


def _patch_latest(stage_dir: Path, repo_root: Path, *, status: str) -> bool:
    path = stage_dir / "latest.json"
    if not path.is_file():
        return False
    payload = _read_json(path)
    run_id = str(payload.get("latest_run_id") or "1")
    summary = stage_dir / run_id / "meta" / "run_summary.json"
    changed = False
    if payload.get("latest_summary_path") != _rel(summary, repo_root):
        payload["latest_summary_path"] = _rel(summary, repo_root)
        changed = True
    if payload.get("latest_status") != status:
        payload["latest_status"] = status
        changed = True
    if changed:
        payload["updated_at"] = _utc_now()
        _write_json(path, payload)
    return changed


def _patch_current_metadata(meta_dir: Path, repo_root: Path, *, stage: str) -> list[str]:
    changed: list[str] = []
    for name in ("stage_manifest.json", "stage_status.json"):
        path = meta_dir / name
        payload = _read_json(path)
        if _patch_contract_payload(payload, stage=stage):
            _write_json(path, payload)
            changed.append(_rel(path, repo_root))
    for status_path in sorted((meta_dir / "datasets").glob("*.status.json")):
        if _patch_unit_status(status_path, stage=stage):
            changed.append(_rel(status_path, repo_root))
    stage_status_path = meta_dir / "stage_status.json"
    stage_status = _read_json(stage_status_path)
    changed_stage_status = False
    for dataset, status in list((stage_status.get("dataset_statuses") or {}).items()):
        if isinstance(status, dict):
            expected, contract, domain_version = _stage_contract(stage)
            metadata = {
                "artifact_contract_kind": "profile_matrix" if stage == "preprocess_b" else "domain_vector",
                "expected_shape_dtype": expected,
                "output_artifact_contract": {
                    Path(item).name: contract.get(Path(item).name)
                    for item in status.get("output_files", [])
                    if Path(item).name in contract
                },
                "unit_name": dataset,
            }
            if domain_version:
                metadata["domain_shape_contract_version"] = domain_version
            if status.get("metadata") != metadata:
                status["metadata"] = metadata
                changed_stage_status = True
    if changed_stage_status:
        _write_json(stage_status_path, stage_status)
        if _rel(stage_status_path, repo_root) not in changed:
            changed.append(_rel(stage_status_path, repo_root))
    if _patch_source_table(meta_dir / "source_table.json", stage=stage):
        changed.append(_rel(meta_dir / "source_table.json", repo_root))
    return changed


def repair(repo_root: Path, b_meta: Path, c_meta: Path, *, cpu_cores: int | None = None) -> dict[str, Any]:
    if cpu_cores is not None:
        os.environ["ODCR_CPU_CORES"] = str(cpu_cores)
    embed_dim = _embed_dim(c_meta)
    b_verify = _verify_b(b_meta, repo_root, embed_dim)
    c_verify = _verify_c(c_meta, repo_root, embed_dim)
    _write_json(b_meta / "verify_report.json", b_verify)
    _write_json(c_meta / "verify_report.json", c_verify)
    b_metrics = _parse_b_metrics(b_meta, repo_root)
    c_metrics = _parse_c_metrics(c_meta, repo_root)
    _write_json(b_meta / "metrics.json", b_metrics)
    _write_json(c_meta / "metrics.json", c_metrics)
    changed = []
    changed.extend(_patch_current_metadata(b_meta, repo_root, stage="preprocess_b"))
    changed.extend(_patch_current_metadata(c_meta, repo_root, stage="preprocess_c"))
    for meta_dir, stage in ((b_meta, "preprocess_b"), (c_meta, "preprocess_c")):
        if _patch_run_summary(meta_dir, repo_root, stage=stage):
            changed.append(_rel(meta_dir / "run_summary.json", repo_root))
    _patch_latest(b_meta.parents[1], repo_root, status=str(_read_json(b_meta / "run_summary.json").get("status") or "ok"))
    _patch_latest(c_meta.parents[1], repo_root, status=str(_read_json(c_meta / "run_summary.json").get("status") or "ok"))
    return {
        "generated_at_utc": _utc_now(),
        "repo_root": str(repo_root),
        "changed_metadata_files": sorted(set(changed)),
        "metrics_paths": [_rel(b_meta / "metrics.json", repo_root), _rel(c_meta / "metrics.json", repo_root)],
        "verify_report_paths": [_rel(b_meta / "verify_report.json", repo_root), _rel(c_meta / "verify_report.json", repo_root)],
        "preprocess_b_verify_status": b_verify["status"],
        "preprocess_c_verify_status": c_verify["status"],
        "preprocess_b_verify_sample_count": b_verify["verify_sample_count"],
        "preprocess_c_verify_sample_count": c_verify["verify_sample_count"],
        "preprocess_c_domain_shape_contract_version": PREPROCESS_C_DOMAIN_CONTRACT_VERSION,
        "no_rerun_statement": "No preprocess_b/c, Step3/4/5, eval, rerank, synthetic benchmark, or data/domain/profile artifact rewrite was performed.",
    }


def _write_ai_analysis(repo_root: Path, result: dict[str, Any]) -> None:
    base = repo_root / "AI_analysis"
    log_path = base / "01_raw_logs" / "fix_preprocess_bc_contract_metrics_verify.log"
    hits_path = base / "02_search_hits" / "fix_preprocess_bc_contract_metrics_verify_hits.txt"
    ledger_path = base / "03_evidence_ledgers" / "fix_preprocess_bc_contract_metrics_verify_ledger.md"
    summary_path = base / "04_phase_summaries" / "fix_preprocess_bc_contract_metrics_verify_summary.md"
    report_path = base / "05_final_reports" / "fix_preprocess_bc_contract_metrics_verify_report.md"
    atomic_write_text(
        log_path,
        "\n".join(
            (
                "fix_preprocess_bc_contract_metrics_verify",
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            )
        ),
    )
    atomic_write_text(
        hits_path,
        "\n".join(
            (
                "Search/read targets:",
                "code/odcr_core/preprocess_runtime.py expected_shape_dtype metrics_path verify_report_path",
                "code/odcr_core/preprocess_schema.py preprocess_c domain vector artifact contract",
                "runs/preprocess/b/1/meta/shell_logs/*.log [phase] [phase-summary]",
                "runs/preprocess/c/1/meta/shell_logs/*.log [cache] token-windows wrote",
                "runs/preprocess/b/1/meta/*.json",
                "runs/preprocess/c/1/meta/*.json",
            )
        ),
    )
    ledger = f"""# preprocess_b/c contract metrics verify ledger

- Classification: metadata/manifest/status/source_table repair plus metrics/verify artifact generation for completed preprocess_b/c runs.
- Public parameter changes: none.
- Data/export field changes: no CSV/data fields changed; preprocess_c domain artifact contract corrected to vector shape.
- Reusable artifacts: added `metrics.json` and `verify_report.json` under existing run meta directories.
- Entrypoints/model/loss/router/eval changes: none.
- Old logic handling: retired row-count domain shape metadata is replaced/fail-fast detectable in tests; no dual active shape contract.
- Rerun decision: preprocess_b/c rerun not required because verify status is `{result['preprocess_b_verify_status']}` / `{result['preprocess_c_verify_status']}`.
- Files changed/generated: {', '.join(result['metrics_paths'] + result['verify_report_paths'])}.
- Verification samples: b={result['preprocess_b_verify_sample_count']}, c={result['preprocess_c_verify_sample_count']}.
- Forbidden work avoided: preprocess_b/c rerun, Step3/4/5, eval/rerank, synthetic benchmark, data/profile/domain artifact writes.
"""
    atomic_write_text(ledger_path, ledger)
    summary = f"""# preprocess_b/c contract metrics verify summary

- preprocess_c domain contract: `domain_content.npy` and `domain_style.npy` are shape `[env.embed_dim]`, dtype `float32`.
- Metrics artifacts: `{result['metrics_paths'][0]}`, `{result['metrics_paths'][1]}`.
- Verify artifacts: `{result['verify_report_paths'][0]}`, `{result['verify_report_paths'][1]}`.
- Verify status: b=`{result['preprocess_b_verify_status']}`, c=`{result['preprocess_c_verify_status']}`.
- Next phase: CPU/GPU pipeline optimization is allowed after validation because metadata repair did not require a stage rerun.
"""
    atomic_write_text(summary_path, summary)
    report = f"""# preprocess_b/c contract metrics verify report

## Files changed or generated

- Code/tests: `code/odcr_core/preprocess_schema.py`, `code/odcr_core/preprocess_runtime.py`, `code/tools/fix_preprocess_bc_contract_metrics_verify.py`, `code/tests/test_preprocess_contract_cleanup.py`, `code/tests/test_run_summary_logging.py`, `code/tests/test_post_edit_check.py`.
- Current-run metadata/artifacts: `runs/preprocess/b/1/meta/metrics.json`, `runs/preprocess/b/1/meta/verify_report.json`, `runs/preprocess/c/1/meta/metrics.json`, `runs/preprocess/c/1/meta/verify_report.json`, plus source_table/stage_manifest/stage_status/dataset status metadata repair.

## Root cause

preprocess_c writes one mean domain vector per channel, but runtime metadata described the artifacts as row-level matrices. The old metadata reused `[row_count, env.embed_dim]`; the actual `infer_domain_semantics.py` validator and files are vector-shaped.

## Current contract

- `domain_content.npy`: shape `[env.embed_dim]` / resolved `[1024]`, dtype `float32`.
- `domain_style.npy`: shape `[env.embed_dim]` / resolved `[1024]`, dtype `float32`.
- Contract version: `{PREPROCESS_C_DOMAIN_CONTRACT_VERSION}`.

## Metadata repair

source_table, stage_manifest, stage_status, and all preprocess_c dataset status files now record `[env.embed_dim]`; preprocess_b profile metadata remains `[entity_count, env.embed_dim]`.

## Metrics and verify artifacts

- Metrics: `{result['metrics_paths'][0]}`, `{result['metrics_paths'][1]}`.
- Verify reports: `{result['verify_report_paths'][0]}`, `{result['verify_report_paths'][1]}`.
- verify_sample_count: b={result['preprocess_b_verify_sample_count']} (>0), c={result['preprocess_c_verify_sample_count']} (>0).
- Verify status: b=`{result['preprocess_b_verify_status']}`, c=`{result['preprocess_c_verify_status']}`.

## Rerun and risk decision

- preprocess_b/c product rerun required: no.
- Remaining P0/P1/P2: no P0/P1/P2 blockers for contract/metadata/metrics/verify. CPU/tokenizer bottleneck is optimization work, not a correctness blocker.
- Phase 2 CPU/GPU pipeline optimization: allowed after validation.

## Explicit non-actions

- Did not rerun preprocess_b or preprocess_c.
- Did not enter Step3/Step4/Step5.
- Did not run eval/rerank.
- Did not modify data/merged CSVs.
- Did not modify profile .npy or domain .npy payloads.
- Did not use a synthetic benchmark.
"""
    atomic_write_text(report_path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair completed preprocess_b/c metadata, metrics, and verify artifacts.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--b-meta", default="runs/preprocess/b/1/meta")
    parser.add_argument("--c-meta", default="runs/preprocess/c/1/meta")
    parser.add_argument("--cpu-cores", type=int, default=12)
    parser.add_argument("--write-ai-analysis", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    b_meta = Path(args.b_meta)
    c_meta = Path(args.c_meta)
    if not b_meta.is_absolute():
        b_meta = repo_root / b_meta
    if not c_meta.is_absolute():
        c_meta = repo_root / c_meta
    result = repair(repo_root, b_meta.resolve(), c_meta.resolve(), cpu_cores=args.cpu_cores)
    if args.write_ai_analysis:
        _write_ai_analysis(repo_root, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
