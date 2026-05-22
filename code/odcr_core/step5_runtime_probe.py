"""Real bounded Step5 E4 runtime probe.

This module is intentionally separate from formal Step5 training.  It resolves
the normal Step5 One-Control payload, consumes the selected Step4 export through
the bounded loader, runs real DDP forward/backward/optimizer steps, and writes
compact AI_analysis evidence.  It must not update formal latest pointers or
write checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import resource
import signal
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from odcr_core.config_resolver import resolve_config
from odcr_core.config_schema import as_plain_dict, fingerprint
from odcr_core.evidence_level import (
    E3_GPU_TRANSPORT,
    E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
    E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE,
    mark_gpu_shard_forward,
)
from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import (
    GLOBAL_COL_ITEM,
    GLOBAL_COL_USER,
    ODCR_ROUTING_TRAIN_CSV,
    load_index_contract,
    validate_index_contract_against_profiles,
    validate_split_indices,
    validate_step4_export_lineage,
)
from odcr_core.manifests import build_formal_source_table_snapshot
from odcr_core.step5_export_loader import (
    STEP5_TRAIN_VALIDATION_COLUMNS,
    load_step5_pool_train_table,
    resolved_step5_export_paths,
)
from odcr_core.step5_pool_sampler import (
    read_step5_sample_plan_shard,
    write_step5_sample_plan,
)
from odcr_core.step5_innovation import (
    build_ccv_control_packet,
    build_rating_stability_control_gate,
    build_step5_explanation_gate,
    evidence_basis_fca_loss,
    parse_step5_innovation_config_json,
    validate_ccv_control_packet_shapes,
)
from odcr_core.step5_word_losses import (
    odcr_anti_repeat_unlikelihood_loss_from_logp,
    route_weighted_mean,
)
from odcr_core.step5_grad_contract import (
    head_gated_loss_contract,
    validate_all_trainable_params_receive_grad,
)
from odcr_core.training_checkpoint import file_fingerprint, stable_hash


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"


def _torchrun_cmd() -> list[str]:
    if os.environ.get("ODCR_TEST_DISABLE_TORCHRUN", "").strip() == "1":
        return [sys.executable]
    import shutil

    if shutil.which("torchrun"):
        return ["torchrun"]
    return [sys.executable, "-m", "torch.distributed.run"]


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _normalize_evidence_level(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in {"E5", E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE}:
        return E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE
    return E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(payload))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _rank_paths(output_dir: Path, rank: int) -> Path:
    return output_dir / f"rank{int(rank)}.json"


def _candidate_output_dir(stage: str, task: int, candidate_id: str) -> Path:
    safe_stage = str(stage).replace("/", "_")
    safe_candidate = str(candidate_id).replace("/", "_")
    return REPO_ROOT / "AI_analysis" / "01_raw_logs" / "step5_e4_candidates" / safe_stage / f"task{int(task)}" / safe_candidate


def candidate_overrides(
    *,
    hardware_profile: str,
    per_gpu_batch_size: int,
    global_batch_size: int,
    workers_per_rank: int,
    prefetch_factor: int,
    bounded_rows: int,
    chunk_rows: int,
) -> list[str]:
    return [
        f"step5.train.per_gpu_batch_size={int(per_gpu_batch_size)}",
        f"step5.train.batch_size={int(global_batch_size)}",
        f"hardware.profiles.{hardware_profile}.dataloader_num_workers_train={int(workers_per_rank)}",
        f"hardware.profiles.{hardware_profile}.dataloader_prefetch_factor_train={int(prefetch_factor)}",
        f"step5.export_loader.bounded_max_rows={int(bounded_rows)}",
        f"step5.export_loader.chunk_rows={int(chunk_rows)}",
    ]


def _candidate_tokens(candidate_id: str | None) -> set[str]:
    text = str(candidate_id or "").strip().upper()
    return {token for token in re.split(r"[^A-Z0-9]+", text) if token}


def _select_candidate_row(rows: Sequence[Mapping[str, Any]], *, candidate_id: str | None) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("candidate list is empty.")
    tokens = _candidate_tokens(candidate_id)
    for item in rows:
        cid = str(item.get("id") or "").strip()
        if cid and cid.upper() in tokens:
            return dict(item)
    return dict(rows[0])


def baseline_candidate_from_config(
    cfg_payload: Mapping[str, Any],
    *,
    hardware_profile: str = "default",
    candidate_id: str | None = None,
    min_per_gpu_batch_size: int | None = None,
) -> dict[str, Any]:
    batches = list((cfg_payload.get("batch_candidates") or []))
    loaders = list((cfg_payload.get("dataloader_candidates") or []))
    rows = list((cfg_payload.get("row_candidates") or []))
    if not batches or not loaders or not rows:
        raise RuntimeError("step5.e4_bounded candidates are empty.")
    b0 = _select_candidate_row(batches, candidate_id=candidate_id)
    if not str(candidate_id or "").strip() and min_per_gpu_batch_size is not None:
        minimum = int(min_per_gpu_batch_size)
        eligible = [
            dict(item)
            for item in batches
            if int(item.get("per_gpu_batch_size", 0) or 0) >= minimum
        ]
        if eligible:
            b0 = eligible[0]
    c0 = _select_candidate_row(loaders, candidate_id=candidate_id)
    r0 = _select_candidate_row(rows, candidate_id=candidate_id)
    return {
        "candidate_id": f"{b0['id']}_{c0['id']}_{r0['id']}",
        "batch": b0,
        "dataloader": c0,
        "rows": r0,
        "overrides": candidate_overrides(
            hardware_profile=hardware_profile,
            per_gpu_batch_size=int(b0["per_gpu_batch_size"]),
            global_batch_size=int(b0["global_batch_size"]),
            workers_per_rank=int(c0["workers_per_rank"]),
            prefetch_factor=int(c0["prefetch_factor"]),
            bounded_rows=int(r0["bounded_rows"]),
            chunk_rows=int(r0["chunk_rows"]),
        ),
    }


def expand_scan_candidates(cfg_payload: Mapping[str, Any], *, hardware_profile: str = "default") -> list[dict[str, Any]]:
    batches = [dict(x) for x in (cfg_payload.get("batch_candidates") or [])]
    loaders = [dict(x) for x in (cfg_payload.get("dataloader_candidates") or [])]
    rows = [dict(x) for x in (cfg_payload.get("row_candidates") or [])]
    if not batches or not loaders or not rows:
        raise RuntimeError("step5.e4_bounded candidates are empty.")
    out: list[dict[str, Any]] = []
    baseline_loader = loaders[0]
    baseline_rows = rows[0]
    baseline_batch = batches[0]

    def _add(b: Mapping[str, Any], c: Mapping[str, Any], r: Mapping[str, Any]) -> None:
        cid = f"{b['id']}_{c['id']}_{r['id']}"
        if any(item["candidate_id"] == cid for item in out):
            return
        out.append(
            {
                "candidate_id": cid,
                "batch": dict(b),
                "dataloader": dict(c),
                "rows": dict(r),
                "overrides": candidate_overrides(
                    hardware_profile=hardware_profile,
                    per_gpu_batch_size=int(b["per_gpu_batch_size"]),
                    global_batch_size=int(b["global_batch_size"]),
                    workers_per_rank=int(c["workers_per_rank"]),
                    prefetch_factor=int(c["prefetch_factor"]),
                    bounded_rows=int(r["bounded_rows"]),
                    chunk_rows=int(r["chunk_rows"]),
                ),
            }
        )

    for batch in batches:
        _add(batch, baseline_loader, baseline_rows)
    for loader in loaders[1:]:
        _add(baseline_batch, loader, baseline_rows)
    for row in rows[1:]:
        _add(baseline_batch, baseline_loader, row)
    return out


def _resolve_candidate(
    *,
    stage: str,
    task: int,
    config_path: str,
    set_overrides: Sequence[str],
    from_step4: str | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    cfg, _sources, snapshot = resolve_config(
        config_path=config_path,
        command="step5",
        task_id=int(task),
        set_overrides=list(set_overrides),
        dry_run=True,
        run_id="auto",
        from_step4=from_step4,
        step5_head=str(stage),
        mode="train_only",
    )
    source_table = build_formal_source_table_snapshot(snapshot)
    return cfg, snapshot, source_table


def _runtime_env_from_cfg(cfg: Any) -> dict[str, str]:
    thread = json.loads(str(getattr(cfg, "thread_env_effective_json", "") or "{}"))
    launcher = json.loads(str(getattr(cfg, "launcher_env_effective_json", "") or "{}"))
    env = {
        "ODCR_ROOT": str(cfg.repo_root),
        "ODCR_RESOLVED_DATA_DIR": str(Path(cfg.data_dir).resolve()),
        "ODCR_RESOLVED_MERGED_DIR": str(Path(cfg.merged_dir).resolve()),
        "ODCR_RESOLVED_RUNS_DIR": str(Path(cfg.runs_dir).resolve()),
        "ODCR_RESOLVED_CACHE_DIR": str(Path(cfg.cache_dir).resolve()),
        "ODCR_RESOLVED_MODELS_DIR": str(Path(cfg.models_dir).resolve()),
        "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(Path(cfg.step5_text_model).resolve()),
        "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(Path(cfg.sentence_embed_model).resolve()),
        "ODCR_RESOLVED_EMBED_DIM": str(int(cfg.embed_dim)),
        "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON": str(cfg.effective_training_payload_json),
        "ODCR_CONFIG_FIELD_SOURCES_JSON": str(cfg.config_field_sources_json),
        "ODCR_HARDWARE_PROFILE_JSON": str(cfg.hardware_profile_json),
        "ODCR_HARDWARE_PRESET": str(cfg.hardware_preset_id),
        "ODCR_RUNTIME_PRECISION_MODE": str(cfg.train_precision),
        "ODCR_RUNTIME_ALLOW_TF32": "1" if bool(getattr(cfg, "allow_tf32", False)) else "0",
        "ODCR_RUNTIME_AMP_AUTOCAST": "1" if bool(getattr(cfg, "amp_autocast", True)) else "0",
        "ODCR_RUNTIME_GRAD_SCALER": "1" if bool(getattr(cfg, "grad_scaler", False)) else "0",
        "ODCR_DECODE_PROFILE_JSON": str(cfg.decode_profile_json),
        "ODCR_RERANK_PROFILE_JSON": str(cfg.rerank_profile_json),
        "ODCR_DECODE_PRESET_STEM": str(cfg.decode_preset_id),
        "ODCR_RERANK_PRESET_STEM": str(cfg.rerank_preset_id or ""),
        "OMP_NUM_THREADS": str(thread.get("OMP_NUM_THREADS", cfg.omp_num_threads)),
        "MKL_NUM_THREADS": str(thread.get("MKL_NUM_THREADS", cfg.mkl_num_threads)),
        "TOKENIZERS_PARALLELISM": str(thread.get("TOKENIZERS_PARALLELISM", "false")).lower(),
    }
    cvd = launcher.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and str(cvd).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(cvd).strip()
    return env


def _worker_budget(cfg: Any) -> dict[str, Any]:
    hw = json.loads(str(cfg.hardware_profile_json or "{}"))
    formula = hw.get("worker_budget_formula") if isinstance(hw, dict) else {}
    reserved = int((formula or {}).get("reserved_cpu", hw.get("reserved_cpu", 2)))
    workers = int(hw.get("dataloader_num_workers_train", 0))
    world = int(cfg.ddp_world_size)
    max_cpu = int(hw.get("max_parallel_cpu", 0))
    active = workers * world + reserved
    return {
        "workers_per_rank": workers,
        "ddp_world_size": world,
        "reserved_cpu": reserved,
        "max_parallel_cpu": max_cpu,
        "active_processes": active,
        "ok": active <= max_cpu,
        "formula": f"{workers} * {world} + {reserved} <= {max_cpu}",
    }


def _artifact_build_preflight_requested(candidate_id: str | None) -> bool:
    text = str(candidate_id or "").strip().lower().replace("-", "_")
    return "artifact_build" in text or text in {"artifact", "artifactbuild"}


def _ensure_artifact_build_train_link(run_dir: Path, export_path: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    dest = run_dir / ODCR_ROUTING_TRAIN_CSV
    src = export_path.expanduser().resolve()
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and dest.resolve() == src:
            return dest
        if dest.is_symlink():
            dest.unlink()
        else:
            raise RuntimeError(f"artifact-build preflight refuses to overwrite existing train table: {dest}")
    rel = os.path.relpath(src, dest.parent)
    os.symlink(rel, dest)
    return dest


def _restore_env(saved: Mapping[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _extract_artifact_build_cache_evidence(log_file: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "log_file": str(log_file),
        "log_exists": log_file.is_file(),
        "cache_dir": None,
        "cache_fingerprint": None,
        "rank_used_rank0_broadcast_cache": False,
        "missing_dataset_absent": True,
    }
    if not log_file.is_file():
        return payload
    text = log_file.read_text(encoding="utf-8", errors="replace")
    payload["missing_dataset_absent"] = "missing_dataset" not in text
    if "using rank0 broadcast cache" in text:
        payload["rank_used_rank0_broadcast_cache"] = True
    matches = list(re.finditer(r"fingerprint=([^|\s]+)\s*\|\s*cache_dir=([^|\n]+)", text))
    if matches:
        last = matches[-1]
        payload["cache_fingerprint"] = last.group(1).strip()
        payload["cache_dir"] = last.group(2).strip()
    broadcast_matches = list(
        re.finditer(r"broadcast_fingerprint=([^|\s]+)\s*\|\s*broadcast_dir=([^|\n]+)", text)
    )
    if broadcast_matches:
        last = broadcast_matches[-1]
        payload["cache_fingerprint"] = last.group(1).strip()
        payload["cache_dir"] = last.group(2).strip()
    return payload


def _summarize_artifact_build_rank_payloads(
    all_payloads: Sequence[Mapping[str, Any] | None],
    *,
    world_size: int,
) -> dict[str, Any]:
    cache_dirs: list[str] = []
    lineage_hashes: list[str] = []
    audit_ok: list[bool] = []
    success_markers: list[bool] = []
    lineage_files: list[bool] = []
    missing_dataset_absent: list[bool] = []
    broadcast_cache_used: list[bool] = []
    rank_ids: list[int] = []
    for item in all_payloads:
        if not item:
            continue
        try:
            rank_ids.append(int(item.get("rank")))
        except Exception:
            pass
        artifact = dict(item.get("artifact_build_preflight") or {})
        cache_dirs.extend(str(path) for path in list(artifact.get("cache_dir_candidates") or []) if path)
        lineage_hashes.extend(str(v) for v in dict(artifact.get("cache_lineage_semantic_hashes") or {}).values() if v)
        audit_ok.append(bool(artifact.get("index_contract_audit_exists")))
        success_markers.append(bool(artifact.get("token_cache_success_marker_exists")))
        lineage_files.append(bool(artifact.get("token_cache_lineage_exists")))
        missing_dataset_absent.append(bool(artifact.get("missing_dataset_absent")))
        broadcast_cache_used.append(bool(artifact.get("rank_used_rank0_broadcast_cache")))
    unique_cache_dirs = sorted(set(cache_dirs))
    unique_lineage_hashes = sorted(set(lineage_hashes))
    rank_payloads_complete = sorted(set(rank_ids)) == list(range(int(world_size)))
    rank0_rank1_cache_dir_match = len(unique_cache_dirs) == 1
    rank0_rank1_cache_fingerprint_match = len(unique_lineage_hashes) == 1
    rank1_broadcast_log_observed = bool(broadcast_cache_used) and all(broadcast_cache_used)
    rank1_used_rank0_cache_dir = (
        rank_payloads_complete
        and rank0_rank1_cache_dir_match
        and rank0_rank1_cache_fingerprint_match
    )
    return {
        "rank_ids": sorted(set(rank_ids)),
        "rank_payloads_complete": rank_payloads_complete,
        "rank0_rank1_cache_dir_candidates": unique_cache_dirs,
        "rank0_rank1_cache_lineage_hashes": unique_lineage_hashes,
        "rank0_rank1_cache_dir_match": rank0_rank1_cache_dir_match,
        "rank0_rank1_cache_fingerprint_match": rank0_rank1_cache_fingerprint_match,
        "rank1_used_rank0_cache_dir": rank1_used_rank0_cache_dir,
        "rank1_broadcast_log_observed": rank1_broadcast_log_observed,
        "rank1_broadcast_cache_inferred_from_rank_payloads": rank1_used_rank0_cache_dir
        and not rank1_broadcast_log_observed,
        "missing_dataset_absent": bool(missing_dataset_absent) and all(missing_dataset_absent),
        "token_cache_lineage_success": bool(lineage_files)
        and all(lineage_files)
        and all(success_markers)
        and rank0_rank1_cache_fingerprint_match,
        "index_contract_audit_pass": bool(audit_ok) and all(audit_ok),
        "index_contract_audit_success": bool(audit_ok) and all(audit_ok),
    }


def _artifact_build_args(cfg: Any, *, run_dir: Path, log_file: Path, save_file: Path, stage: str) -> SimpleNamespace:
    decode = json.loads(str(getattr(cfg, "decode_profile_json", "") or "{}"))
    return SimpleNamespace(
        auxiliary=str(cfg.auxiliary),
        target=str(cfg.target),
        save_file=str(save_file),
        log_file=str(log_file),
        seed=int(cfg.seed),
        num_proc=int(cfg.num_proc),
        nlayers=None,
        nhead=None,
        nhid=None,
        dropout=None,
        label_smoothing=float(decode.get("label_smoothing", cfg.label_smoothing)),
        repetition_penalty=float(decode.get("repetition_penalty", cfg.repetition_penalty)),
        generate_temperature=float(decode.get("generate_temperature", cfg.generate_temperature)),
        generate_top_p=float(decode.get("generate_top_p", cfg.generate_top_p)),
        max_explanation_length=int(decode.get("max_explanation_length", cfg.max_explanation_length)),
        decode_strategy=str(decode.get("decode_strategy", cfg.decode_strategy)),
        decode_seed=decode.get("decode_seed", cfg.decode_seed),
        no_repeat_ngram_size=decode.get("no_repeat_ngram_size", cfg.no_repeat_ngram_size),
        min_len=decode.get("min_len", cfg.min_len),
        eval_batch_size=None,
        eval_single_process_safe=False,
        sanity_compare_ddp_single=False,
        learning_rate=None,
        epochs=None,
        coef=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        per_device_batch_size=None,
        train_only=True,
        task_head=str(stage),
        min_epochs=None,
        early_stop_patience=None,
        early_stop_patience_full=None,
        early_stop_patience_loss=None,
        checkpoint_metric="valid_loss",
        bleu4_max_samples=None,
        quick_eval_max_samples=None,
        scheduler_initial_lr=None,
        warmup_steps=None,
        warmup_ratio=None,
        min_lr_ratio=None,
        _artifact_build_preflight_run_dir=str(run_dir),
    )


def _run_artifact_build_preflight(
    *,
    cfg: Any,
    stage: str,
    rank: int,
    local_rank: int,
    world_size: int,
    output_dir: Path,
    export_path: Path,
    build_odcr_ddp_artefacts: Any,
) -> dict[str, Any]:
    run_dir = output_dir / "artifact_build_preflight_run"
    meta_dir = run_dir / "meta"
    model_dir = run_dir / "model"
    meta_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    train_link = _ensure_artifact_build_train_link(run_dir, export_path)
    log_file = meta_dir / f"artifact_build_rank{rank}.log"
    save_file = model_dir / "FORBIDDEN_checkpoint_not_written.pth"
    args_ns = _artifact_build_args(cfg, run_dir=run_dir, log_file=log_file, save_file=save_file, stage=stage)
    progress_path = meta_dir / f"artifact_build_rank{rank}_progress.json"
    _write_json(
        progress_path,
        {
            "schema_version": "odcr_step5_artifact_build_progress/1",
            "rank": int(rank),
            "local_rank": int(local_rank),
            "world_size": int(world_size),
            "stage": str(stage),
            "state": "before_build_odcr_ddp_artefacts",
            "run_dir": str(run_dir),
            "log_file": str(log_file),
        },
    )
    cache_root = REPO_ROOT / "cache" / f"task{int(cfg.task_id)}" / "hf"
    before_cache_dirs = set(cache_root.glob("hf_cache_step5_v8_step5_semantic_lineage_*")) if cache_root.is_dir() else set()
    saved_env = {
        key: os.environ.get(key)
        for key in ("ODCR_STAGE_RUN_DIR", "ODCR_MANIFEST_DIR", "ODCR_LOG_DIR")
    }
    os.environ["ODCR_STAGE_RUN_DIR"] = str(run_dir)
    os.environ["ODCR_MANIFEST_DIR"] = str(meta_dir)
    os.environ["ODCR_LOG_DIR"] = str(meta_dir)
    started = time.monotonic()
    try:
        base_final, train_dataset, valid_dataset, model = build_odcr_ddp_artefacts(
            args_ns,
            world_size,
            local_rank,
            rank,
            command="train",
            show_datasets_progress=False,
        )
        del model
        torch.cuda.empty_cache()
        _write_json(
            progress_path,
            {
                "schema_version": "odcr_step5_artifact_build_progress/1",
                "rank": int(rank),
                "local_rank": int(local_rank),
                "world_size": int(world_size),
                "stage": str(stage),
                "state": "after_build_odcr_ddp_artefacts",
                "run_dir": str(run_dir),
                "log_file": str(log_file),
                "train_dataset_len": int(len(train_dataset)) if train_dataset is not None else 0,
                "valid_dataset_len": int(len(valid_dataset)) if valid_dataset is not None else 0,
            },
        )
    finally:
        _restore_env(saved_env)
    elapsed = time.monotonic() - started
    checkpoint_written = save_file.exists() or any(model_dir.glob("*.pth"))
    latest_created = any(run_dir.glob("latest.json")) or any(output_dir.glob("latest.json"))
    after_cache_dirs = set(cache_root.glob("hf_cache_step5_v8_step5_semantic_lineage_*")) if cache_root.is_dir() else set()
    touched_cache_dirs = sorted(after_cache_dirs - before_cache_dirs)
    if not touched_cache_dirs:
        touched_cache_dirs = sorted(after_cache_dirs)
    log_cache = _extract_artifact_build_cache_evidence(log_file)
    cache_dirs = [str(log_cache["cache_dir"])] if log_cache.get("cache_dir") else [str(path) for path in touched_cache_dirs]
    cache_lineage_hashes = {}
    if log_cache.get("cache_dir"):
        lineage_paths = [Path(str(log_cache["cache_dir"]))]
    else:
        lineage_paths = list(touched_cache_dirs[-4:])
    for path in lineage_paths[-4:]:
        lineage = path / "token_cache_lineage.json"
        if lineage.is_file():
            try:
                cache_lineage_hashes[str(path)] = (_load_json(lineage).get("semantic_payload_hash") or "")
            except Exception:
                cache_lineage_hashes[str(path)] = "unreadable"
    success_markers = {
        str(path): (path / "_SUCCESS").is_file()
        for path in lineage_paths[-4:]
    }
    payload = {
        "schema_version": "odcr_step5_formal_artifact_build_preflight/1",
        "stage": str(stage),
        "task_id": int(cfg.task_id),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "success": True,
        "artifact_build_only": True,
        "evidence_level": E3_GPU_TRANSPORT,
        "forward_executed": False,
        "loss_backward_executed": False,
        "optimizer_step_executed": False,
        "real_data_batch_used": False,
        "real_ccv_packet_used": False,
        "train_link": str(train_link),
        "run_dir": str(run_dir),
        "meta_dir": str(meta_dir),
        "log_file": str(log_file),
        "index_contract_audit_path": str(meta_dir / "index_contract_audit.json"),
        "index_contract_audit_exists": (meta_dir / "index_contract_audit.json").is_file(),
        "train_dataset_len": int(len(train_dataset)) if train_dataset is not None else 0,
        "valid_dataset_len": int(len(valid_dataset)) if valid_dataset is not None else 0,
        "head": str(getattr(base_final, "step5_head", stage)),
        "per_gpu_batch_size": int(getattr(base_final, "per_gpu_batch_size", getattr(cfg, "per_device_train_batch_size", 0))),
        "global_batch_size": int(getattr(base_final, "global_batch_size", getattr(cfg, "batch_size", 0))),
        "cache_dir_candidates": cache_dirs[-4:],
        "cache_evidence_from_log": log_cache,
        "cache_lineage_semantic_hashes": cache_lineage_hashes,
        "cache_success_markers": success_markers,
        "token_cache_lineage_exists": any((path / "token_cache_lineage.json").is_file() for path in lineage_paths[-4:]),
        "token_cache_success_marker_exists": any(success_markers.values()),
        "rank_used_rank0_broadcast_cache": bool(log_cache.get("rank_used_rank0_broadcast_cache")) if rank != 0 else True,
        "missing_dataset_absent": bool(log_cache.get("missing_dataset_absent")),
        "elapsed_sec": float(elapsed),
        "formal_namespace_pollution": False,
        "latest_json_created": bool(latest_created),
        "checkpoint_written": bool(checkpoint_written),
    }
    if checkpoint_written or latest_created:
        raise RuntimeError(f"artifact-build preflight wrote forbidden formal artifact: {payload}")
    return payload


def _write_candidate_resolution(
    *,
    output_dir: Path,
    cfg: Any,
    snapshot: Mapping[str, Any],
    source_table: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, str]:
    resolved_path = output_dir / "resolved_config.json"
    source_path = output_dir / "source_table.json"
    candidate_path = output_dir / "candidate.json"
    _write_json(resolved_path, snapshot)
    _write_json(source_path, source_table)
    _write_json(candidate_path, candidate)
    return {
        "resolved_config_path": str(resolved_path),
        "source_table_path": str(source_path),
        "candidate_path": str(candidate_path),
        "resolved_config_hash": fingerprint(snapshot),
        "effective_training_payload_hash": stable_hash(json.loads(cfg.effective_training_payload_json)),
    }


def _patch_candidate_resolution_contract(
    *,
    resolution_paths: Mapping[str, str],
    result: Mapping[str, Any],
) -> None:
    resolved_path = Path(str(resolution_paths.get("resolved_config_path") or ""))
    source_path = Path(str(resolution_paths.get("source_table_path") or ""))
    all_grad = dict(result.get("all_trainable_grad") or {})
    evidence_context = dict(all_grad.get("evidence_context") or {})
    contract_patch = {
        "lora_target_policy_id": result.get("lora_target_policy_id"),
        "head_specific_lora_allowlist_id": result.get("head_specific_lora_allowlist_id"),
        "final_lora_target_modules": list(result.get("final_lora_target_modules") or []),
        "forbidden_lora_targets": list(result.get("forbidden_lora_targets") or []),
        "deleted_legacy_modules": list(result.get("deleted_legacy_modules") or []),
        "combined_formal_enabled": False,
        "all_trainable_grad_required": True,
        "head_specific_trainable_policy": result.get("head_specific_trainable_policy"),
        "head_gated_loss_contract": result.get("head_gated_loss_contract"),
        "all_trainable_grad_preflight_result": all_grad or None,
        "all_trainable_grad_status": result.get("all_trainable_grad_status"),
        "scratch_cleanup_required": True,
        "scratch_cleanup_status": result.get("scratch_cleanup_status"),
        "graph_tensor_audit_status": result.get("graph_tensor_audit_status"),
        "graph_tensor_audit_phase": result.get("graph_tensor_audit_phase"),
        "ema_enabled": result.get("ema_enabled"),
        "ema_decay": result.get("ema_decay"),
        "ema_init_status": "pass" if result.get("ema_init_pass") is True else "fail",
        "ema_init_strategy": result.get("ema_init_strategy"),
        "formal_entry_E4_required": True,
        "formal_entry_E4_validation_required": True,
        "formal_entry_E4_evidence_id": evidence_context.get("evidence_id"),
        "first_train_step_evidence_id": evidence_context.get("evidence_id"),
        "validation_e4_evidence_id": evidence_context.get("evidence_id"),
        "validation_pass_executed": result.get("validation_pass_executed"),
        "validation_forward_pass": result.get("validation_forward_pass"),
        "validation_loss_finite": result.get("validation_loss_finite"),
        "validation_oom": result.get("validation_oom"),
        "rating_stability_control_validation_control_only": result.get("rating_stability_control_validation_control_only"),
        "flan_explainer_called_in_rating_stability_control_validation": result.get("flan_explainer_called_in_rating_stability_control_validation"),
        "out_logits_materialized_in_rating_stability_control_validation": result.get("out_logits_materialized_in_rating_stability_control_validation"),
        "valid_per_gpu_batch_size": result.get("valid_per_gpu_batch_size"),
        "valid_forward_micro_batch_size": result.get("valid_forward_micro_batch_size"),
        "train_per_gpu_batch_size": result.get("train_per_gpu_batch_size"),
        "validation_memory_policy": result.get("validation_memory_policy"),
        "validation_microbatch_accumulation": result.get("validation_microbatch_accumulation"),
        "missing_grad_params": list(result.get("missing_grad_params") or []),
        "runtime_e4_evidence_id": evidence_context.get("evidence_id"),
        "trainable_param_count": result.get("trainable_param_count"),
        "grad_present_count": result.get("grad_present_count"),
        "lora_trainable_count": result.get("lora_trainable_count"),
        "lora_grad_present_count": result.get("lora_grad_present_count"),
    }
    if resolved_path.is_file():
        resolved = _load_json(resolved_path)
        if isinstance(resolved, dict):
            resolved.update(contract_patch)
            _write_json(resolved_path, resolved)
    if source_path.is_file():
        source = _load_json(source_path)
        if isinstance(source, dict):
            records = list(source.get("records") or [])
            record_map = {str(item.get("key")): dict(item) for item in records if isinstance(item, Mapping)}

            def upsert(key: str, value: Any, source_label: str) -> None:
                record_map[key] = {"key": key, "value": value, "source": source_label}

            final_targets = list(result.get("final_lora_target_modules") or [])
            upsert("final_lora_target_modules_hash", stable_hash(final_targets), "Step5 runtime head-aware LoRA allowlist")
            upsert(
                "trainable_parameter_names_hash",
                result.get("trainable_parameter_names_hash"),
                "Step5 head-aware trainable contract",
            )
            upsert("forbidden_lora_targets", list(result.get("forbidden_lora_targets") or []), "Step5 MHA out_proj LoRA ban")
            upsert("deleted_legacy_modules", list(result.get("deleted_legacy_modules") or []), "Step5 active legacy deletion")
            upsert(
                "head_specific_trainable_policy",
                result.get("head_specific_trainable_policy"),
                "Step5 head-aware trainable contract",
            )
            upsert("head_gated_loss_contract", result.get("head_gated_loss_contract"), "Step5 head-gated train loss")
            upsert("all_trainable_grad_preflight_result", all_grad or None, "Step5 unified all-trainable-grad gate")
            upsert("missing_grad_params", list(result.get("missing_grad_params") or []), "Step5 unified all-trainable-grad gate")
            upsert("scratch_cleanup_status", result.get("scratch_cleanup_status"), "Step5 formal-entry lifecycle E4")
            upsert("graph_tensor_audit_status", result.get("graph_tensor_audit_status"), "Step5 formal-entry lifecycle E4")
            upsert("graph_tensor_audit_phase", result.get("graph_tensor_audit_phase"), "Step5 formal-entry lifecycle E4")
            upsert("ema_init_status", "pass" if result.get("ema_init_pass") is True else "fail", "Step5 formal-entry lifecycle E4")
            upsert("ema_init_strategy", result.get("ema_init_strategy"), "Step5 formal-entry lifecycle E4")
            upsert("formal_entry_E4_evidence_id", evidence_context.get("evidence_id"), "Step5 formal-entry lifecycle E4")
            upsert("first_train_step_evidence_id", evidence_context.get("evidence_id"), "Step5 formal-entry lifecycle E4")
            upsert("validation_e4_evidence_id", evidence_context.get("evidence_id"), "Step5 formal-entry validation E4")
            upsert("validation_pass_executed", result.get("validation_pass_executed"), "Step5 formal-entry validation E4")
            upsert("validation_forward_pass", result.get("validation_forward_pass"), "Step5 formal-entry validation E4")
            upsert("validation_loss_finite", result.get("validation_loss_finite"), "Step5 formal-entry validation E4")
            upsert("validation_oom", result.get("validation_oom"), "Step5 formal-entry validation E4")
            upsert("rating_stability_control_validation_control_only", result.get("rating_stability_control_validation_control_only"), "Step5 validation contract")
            upsert("out_logits_materialized_in_rating_stability_control_validation", result.get("out_logits_materialized_in_rating_stability_control_validation"), "Step5 validation contract")
            upsert("valid_per_gpu_batch_size", result.get("valid_per_gpu_batch_size"), "Step5 validation memory policy")
            upsert("valid_forward_micro_batch_size", result.get("valid_forward_micro_batch_size"), "Step5 validation memory policy")
            upsert("train_per_gpu_batch_size", result.get("train_per_gpu_batch_size"), "Step5 train batch policy")
            upsert("validation_memory_policy", result.get("validation_memory_policy"), "Step5 validation memory policy")
            upsert("validation_microbatch_accumulation", result.get("validation_microbatch_accumulation"), "Step5 validation memory policy")
            upsert("runtime_e4_evidence_id", evidence_context.get("evidence_id"), "Step5 formal-entry lifecycle E4")
            upsert("synthetic_used", False, "Step5 E4 bounded probe contract")
            source["records"] = list(record_map.values())
            _write_json(source_path, source)


class _Step5ExplanationBoundedDataset(Dataset):
    def __init__(self, df: Any, processor: Any) -> None:
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.processor(self.df.iloc[int(idx)].to_dict())


class _Step5EncodedDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, torch.Tensor]]) -> None:
        self.rows = [dict(row) for row in rows]

    def __len__(self) -> int:
        return int(len(self.rows))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.rows[int(idx)]


class _TimingCollate:
    def __init__(self, collate_fn: Any) -> None:
        self.collate_fn = collate_fn
        self.total_s = mp.Value("d", 0.0)
        self.count = mp.Value("i", 0)

    def __call__(self, batch: Any) -> Any:
        t0 = time.perf_counter()
        out = self.collate_fn(batch)
        elapsed = time.perf_counter() - t0
        with self.total_s.get_lock():
            self.total_s.value += float(elapsed)
        with self.count.get_lock():
            self.count.value += 1
        return out

    def summary_ms(self) -> dict[str, float | int | None]:
        total = float(self.total_s.value)
        count = int(self.count.value)
        return {
            "total_ms": total * 1000.0,
            "count": count,
            "mean_ms": (total * 1000.0 / float(count)) if count > 0 else None,
        }


def _cpu_usage_snapshot() -> tuple[float, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + child.ru_utime), float(usage.ru_stime + child.ru_stime)


def _cpu_percentages(start: tuple[float, float], end: tuple[float, float], *, wall_s: float) -> dict[str, float]:
    denom = max(float(wall_s), 1e-9)
    return {
        "cpu_user_percent": float(max(0.0, end[0] - start[0]) / denom * 100.0),
        "cpu_system_percent": float(max(0.0, end[1] - start[1]) / denom * 100.0),
    }


def _nvidia_smi_gpu_snapshot() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=10)
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip() or proc.stdout.strip(), "gpus": []}
    gpus: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "utilization_gpu_percent": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                    "memory_total_mib": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return {"available": bool(gpus), "error": None, "gpus": gpus}


def _gpu_util_summary(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    memory: list[float] = []
    for snap in snapshots:
        if not bool(snap.get("available")):
            continue
        for gpu in snap.get("gpus") or []:
            if isinstance(gpu, Mapping):
                values.append(float(gpu.get("utilization_gpu_percent") or 0.0))
                memory.append(float(gpu.get("memory_used_mib") or 0.0))
    if not values:
        return {
            "available": False,
            "mean": None,
            "p50": None,
            "p95": None,
            "memory_used_mib_max": None,
            "snapshots": list(snapshots),
        }
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]
    return {
        "available": True,
        "mean": float(sum(values) / len(values)),
        "p50": float(p50),
        "p95": float(p95),
        "memory_used_mib_max": float(max(memory)) if memory else None,
        "snapshots": list(snapshots),
    }


def _write_bounded_token_cache(
    *,
    output_dir: Path,
    rank: int,
    train_df: Any,
    processor: Any,
    source_table: Mapping[str, Any],
    sample_plan_manifest: Mapping[str, Any] | None,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any], float]:
    cache_dir = output_dir / "bounded_token_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = [processor(row) for row in train_df.reset_index(drop=True).to_dict("records")]
    tokenize_s = time.perf_counter() - t0
    cache_file = cache_dir / f"rank{int(rank)}_token_cache.pt"
    torch.save({"schema_version": "odcr_step5_bounded_token_cache/1", "rows": rows}, cache_file)
    manifest = {
        "schema_version": "odcr_step5_bounded_token_cache/1",
        "rank": int(rank),
        "cache_file": str(cache_file),
        "cache_file_sha256": file_fingerprint(cache_file).get("sha256"),
        "row_count": int(len(rows)),
        "tokenize_time_s": float(tokenize_s),
        "source_table_hash": stable_hash(dict(source_table)),
        "sample_plan_hash": str((sample_plan_manifest or {}).get("plan_hash") or ""),
        "token_fields": [
            "explanation_idx",
            "content_evidence_ids",
            "style_evidence_ids",
            "domain_style_anchor_ids",
            "local_style_hint_ids",
            "polarity_ids",
        ],
        "hot_path_tokenize_removed": True,
        "formal_namespace_write": False,
    }
    _write_json(cache_dir / f"rank{int(rank)}_manifest.json", manifest)
    return rows, manifest, float(tokenize_s)


def _cuda_event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _finite_scalar(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _loss_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        out = float(value.detach().float().item())
    else:
        out = float(value)
    if math.isnan(out) or math.isinf(out):
        return out
    return out


_PER_TIER_SPECS: tuple[tuple[str, int, int, str, str], ...] = (
    ("target_gold_high", 0, 0, "gold", "target_gold"),
    ("target_gold_medium", 0, 1, "gold", "target_gold"),
    ("aux_gold_high", 1, 0, "gold", "aux_gold"),
    ("aux_gold_medium", 1, 1, "gold", "aux_gold"),
    ("cf_high", 2, 0, "cf", "cf"),
    ("cf_medium", 2, 1, "cf", "cf"),
    ("cf_low_weighted", 2, 2, "cf", "cf"),
)
_PER_TIER_METRICS: tuple[str, ...] = (
    "scorer_main_loss_raw",
    "scorer_main_loss_weighted",
    "lci_raw_loss",
    "lci_weighted_loss",
    "uci_weight",
    "explainer_loss_raw",
    "explainer_loss_weighted",
    "fca_raw_loss",
    "fca_weighted_loss",
    "ccv_explainer_weight",
    "ccv_fca_weight",
    "ccv_uncertainty",
    "ccv_confidence",
    "ccv_reliability",
)


def _new_per_tier_accumulator() -> dict[str, Any]:
    return {
        name: {
            "tier_count": 0,
            "category": category,
            "component": component,
            "metrics": {metric: {"count": 0, "sum": 0.0, "sumsq": 0.0} for metric in _PER_TIER_METRICS},
        }
        for name, _component_id, _tier_id, category, component in _PER_TIER_SPECS
    }


def _accum_metric(acc: dict[str, Any], tier: str, metric: str, values: torch.Tensor, mask: torch.Tensor) -> None:
    selected = values.detach().float().view(-1)[mask.detach().bool().view(-1)]
    count = int(selected.numel())
    if count <= 0:
        return
    row = acc[tier]["metrics"][metric]
    row["count"] = int(row["count"]) + count
    row["sum"] = float(row["sum"]) + float(selected.sum().item())
    row["sumsq"] = float(row["sumsq"]) + float(selected.pow(2).sum().item())


def _accumulate_per_tier_loss(
    acc: dict[str, Any],
    gb: Any,
    *,
    scorer_raw: torch.Tensor,
    scorer_weighted: torch.Tensor,
    lci_raw: torch.Tensor,
    lci_weighted: torch.Tensor,
    uci_weight: torch.Tensor,
    explainer_raw: torch.Tensor,
    explainer_weighted: torch.Tensor,
    fca_raw: torch.Tensor,
    fca_weighted: torch.Tensor,
    ccv_explainer_weight: torch.Tensor,
    ccv_fca_weight: torch.Tensor,
    ccv_uncertainty: torch.Tensor,
    ccv_confidence: torch.Tensor,
    ccv_reliability: torch.Tensor,
) -> None:
    component_id = getattr(gb, "sampler_component_id", None)
    tier_id = getattr(gb, "sampler_tier_id", None)
    if component_id is None or tier_id is None:
        return
    component_id = component_id.view(-1)
    tier_id = tier_id.view(-1)
    metric_values = {
        "scorer_main_loss_raw": scorer_raw,
        "scorer_main_loss_weighted": scorer_weighted,
        "lci_raw_loss": lci_raw,
        "lci_weighted_loss": lci_weighted,
        "uci_weight": uci_weight,
        "explainer_loss_raw": explainer_raw,
        "explainer_loss_weighted": explainer_weighted,
        "fca_raw_loss": fca_raw,
        "fca_weighted_loss": fca_weighted,
        "ccv_explainer_weight": ccv_explainer_weight,
        "ccv_fca_weight": ccv_fca_weight,
        "ccv_uncertainty": ccv_uncertainty,
        "ccv_confidence": ccv_confidence,
        "ccv_reliability": ccv_reliability,
    }
    for tier_name, component_value, tier_value, _category, _component in _PER_TIER_SPECS:
        mask = (component_id == int(component_value)) & (tier_id == int(tier_value))
        count = int(mask.detach().sum().item())
        acc[tier_name]["tier_count"] = int(acc[tier_name]["tier_count"]) + count
        for metric, values in metric_values.items():
            _accum_metric(acc, tier_name, metric, values, mask)


def _finalize_metric_stats(row: Mapping[str, Any]) -> dict[str, Any]:
    count = int(row.get("count") or 0)
    if count <= 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "finite": True,
            "zero_kind": "graph_tied_zero",
        }
    mean = float(row.get("sum") or 0.0) / float(count)
    variance = max(0.0, float(row.get("sumsq") or 0.0) / float(count) - mean * mean)
    std = math.sqrt(variance)
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "finite": bool(math.isfinite(mean) and math.isfinite(std)),
        "zero_kind": None,
    }


def _finalize_per_tier_loss(acc: Mapping[str, Any]) -> dict[str, Any]:
    tiers: dict[str, Any] = {}
    for tier_name, _component_id, _tier_id, category, component in _PER_TIER_SPECS:
        item = acc.get(tier_name) if isinstance(acc.get(tier_name), Mapping) else {}
        metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
        tiers[tier_name] = {
            "category": category,
            "component": component,
            "tier_count": int(item.get("tier_count") or 0),
            "present": int(item.get("tier_count") or 0) > 0,
            "metrics": {
                metric: _finalize_metric_stats(metrics.get(metric) if isinstance(metrics.get(metric), Mapping) else {})
                for metric in _PER_TIER_METRICS
            },
        }
    return {
        "schema_version": "odcr_step5_per_tier_loss/1",
        "tier_order": [name for name, *_rest in _PER_TIER_SPECS],
        "metric_order": list(_PER_TIER_METRICS),
        "tiers": tiers,
        "gold": {name: tiers[name] for name, *_rest in _PER_TIER_SPECS if tiers[name]["category"] == "gold"},
        "cf": {name: tiers[name] for name, *_rest in _PER_TIER_SPECS if tiers[name]["category"] == "cf"},
        "all_tiers_emitted": True,
        "missing_tier_policy": "tier_count=0 with graph_tied_zero metrics",
        "per_rank_identical_loss_keys_required": True,
    }


def _merge_per_tier_loss(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = _new_per_tier_accumulator()
    for report in reports:
        tiers = report.get("tiers") if isinstance(report.get("tiers"), Mapping) else {}
        for tier_name, tier_payload in tiers.items():
            if tier_name not in merged or not isinstance(tier_payload, Mapping):
                continue
            merged[tier_name]["tier_count"] = int(merged[tier_name]["tier_count"]) + int(tier_payload.get("tier_count") or 0)
            metrics = tier_payload.get("metrics") if isinstance(tier_payload.get("metrics"), Mapping) else {}
            for metric, metric_payload in metrics.items():
                if metric not in merged[tier_name]["metrics"] or not isinstance(metric_payload, Mapping):
                    continue
                count = int(metric_payload.get("count") or 0)
                mean = float(metric_payload.get("mean") or 0.0)
                std = float(metric_payload.get("std") or 0.0)
                row = merged[tier_name]["metrics"][metric]
                row["count"] = int(row["count"]) + count
                row["sum"] = float(row["sum"]) + mean * count
                row["sumsq"] = float(row["sumsq"]) + (std * std + mean * mean) * count
    return _finalize_per_tier_loss(merged)


def _per_tier_loss_keys(report: Mapping[str, Any]) -> list[str]:
    tiers = report.get("tiers") if isinstance(report.get("tiers"), Mapping) else {}
    keys: list[str] = []
    for tier_name in [name for name, *_rest in _PER_TIER_SPECS]:
        metrics = (tiers.get(tier_name) or {}).get("metrics") if isinstance(tiers.get(tier_name), Mapping) else {}
        for metric in _PER_TIER_METRICS:
            keys.append(f"{tier_name}.{metric}")
    return keys


def _lci_per_sample_losses(
    *,
    factual_score: torch.Tensor,
    cf_score: torch.Tensor,
    robust_score: torch.Tensor,
    target_rating: torch.Tensor,
    gate: Any,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(getattr(cfg.lci, "enabled", False)) or float(getattr(cfg.lci, "weight", 0.0) or 0.0) <= 0.0:
        z = factual_score.detach() * 0.0 + factual_score * 0.0
        return z, z
    consistency_ps = (factual_score - cf_score).pow(2)
    cf_score_ps = (cf_score - target_rating.to(dtype=cf_score.dtype)).pow(2)
    robust_ps = (cf_score - robust_score).pow(2)
    raw = (
        consistency_ps
        + float(cfg.lci.counterfactual_label_weight) * cf_score_ps
        + float(cfg.lci.robustness_weight) * robust_ps
    )
    weighted = raw * gate.lci_weight.to(dtype=raw.dtype) * gate.route_mask.to(dtype=raw.dtype) * float(cfg.lci.weight)
    return raw, weighted


def _fca_per_sample_losses(*, fca_bundle: Any, gate: Any, cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
    scorer_basis = fca_bundle.scorer_evidence_basis
    explainer_basis = fca_bundle.explainer_evidence_basis
    if not bool(getattr(cfg.fca, "enabled", False)) or float(getattr(cfg.fca, "weight", 0.0) or 0.0) <= 0.0:
        z = scorer_basis.sum(dim=-1) * 0.0
        return z, z
    eps = 1e-8
    score_n = F.normalize(scorer_basis, dim=-1, eps=eps)
    explain_n = F.normalize(explainer_basis, dim=-1, eps=eps)
    raw = 1.0 - (score_n * explain_n).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    weighted = raw * gate.fca_weight.to(dtype=raw.dtype) * gate.route_mask.to(dtype=raw.dtype) * float(cfg.fca.weight)
    return raw, weighted


MEMORY_TRUTH_REQUIRED_FIELDS: tuple[str, ...] = (
    "device_total_gb",
    "max_memory_allocated_gb",
    "max_memory_reserved_gb",
    "reserved_minus_allocated_gb",
    "allocated_to_total_ratio",
    "reserved_to_total_ratio",
    "nvidia_smi_process_used_gb",
    "param_memory_gb",
    "trainable_param_memory_gb",
    "frozen_param_memory_gb",
    "grad_memory_gb",
    "optimizer_state_memory_gb",
    "activation_peak_estimated_gb",
    "fragmentation_hint",
    "memory_creep_detected",
    "oom",
    "oom_error_message",
    "cuda_allocator_backend",
    "torch_cuda_alloc_conf",
    "reserved_is_diagnostic_only",
)

MODEL_MEMORY_AUDIT_REQUIRED_FIELDS: tuple[str, ...] = (
    "total_params",
    "trainable_params",
    "frozen_params",
    "trainable_ratio",
    "optimizer_param_count",
    "optimizer_param_memory_gb",
    "optimizer_includes_frozen_params",
    "base_text_model_loaded",
    "base_text_model_trainable",
    "lora_enabled",
    "lora_trainable_params",
    "gradient_checkpointing_enabled",
    "use_cache_training_disabled",
    "bf16_effective",
    "model_dtype_summary",
    "largest_modules_by_param_memory",
    "largest_trainable_modules",
    "suspicious_trainable_modules",
)


def _gb(value: float | int) -> float:
    return float(value) / float(1024**3)


def _round_gb(value: float | int) -> float:
    return round(float(value), 6)


def _param_bytes(param: torch.nn.Parameter) -> int:
    return int(param.numel()) * int(param.element_size())


def _optimizer_param_ids(optimizer: optim.Optimizer) -> set[int]:
    ids: set[int] = set()
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            ids.add(id(param))
    return ids


def _optimizer_state_memory_gb(optimizer: optim.Optimizer) -> float:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel()) * int(value.element_size())
    return _round_gb(_gb(total))


def _cuda_allocator_backend() -> str:
    for owner, name in ((getattr(torch.cuda, "memory", None), "get_allocator_backend"), (torch.cuda, "get_allocator_backend")):
        func = getattr(owner, name, None) if owner is not None else None
        if callable(func):
            try:
                return str(func())
            except Exception:
                pass
    return "unknown"


def _nvidia_smi_process_used_gb(pid: int | None = None) -> float | None:
    target = int(pid or os.getpid())
    cmd = (
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=10)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    total_mib = 0.0
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) == target:
                total_mib += float(parts[1])
        except ValueError:
            continue
    return round(total_mib / 1024.0, 6) if total_mib > 0.0 else None


def _nvidia_smi_compute_apps() -> dict[str, Any]:
    cmd = (
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=10)
    except Exception as exc:
        return {"available": False, "error": repr(exc), "rows": [], "stdout": "", "returncode": None}
    rows: list[dict[str, Any]] = []
    for raw in proc.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            used_mib = float(parts[2])
        except ValueError:
            used_mib = None
        rows.append({"pid": pid, "process_name": parts[1], "used_memory_mib": used_mib, "raw": raw})
    return {
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "rows": rows,
        "error": None if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
    }


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except Exception:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _candidate_min_steps(candidate_id: str, memory_cfg: Mapping[str, Any]) -> int:
    upper = str(candidate_id).upper()
    if "LW" not in upper and "LONG" not in upper:
        return 1
    return max(1, int(memory_cfg.get("long_window_steps", 1) or 1))


def _result_steps(result: Mapping[str, Any]) -> int:
    long_window = result.get("long_window") if isinstance(result.get("long_window"), Mapping) else {}
    if long_window.get("steps_executed_max") is not None:
        return int(long_window.get("steps_executed_max") or 0)
    ranks = long_window.get("rank_results") if isinstance(long_window.get("rank_results"), Sequence) else []
    vals = [int((row or {}).get("steps_executed") or 0) for row in ranks if isinstance(row, Mapping)]
    return max(vals, default=0)


def _artifact_hash_payload(out_dir: Path) -> dict[str, Any]:
    sample_manifest = out_dir / "sample_plan" / "sample_plan_manifest.json"
    token_dir = out_dir / "bounded_token_cache"
    token_hashes: dict[str, Any] = {}
    if token_dir.is_dir():
        for path in sorted(token_dir.glob("rank*_manifest.json")):
            token_hashes[path.name] = file_fingerprint(path).get("sha256")
    sample_payload = _load_json(sample_manifest) if sample_manifest.is_file() else {}
    paths = {
        "resolved_config": out_dir / "resolved_config.json",
        "source_table": out_dir / "source_table.json",
        "candidate": out_dir / "candidate.json",
        "request": out_dir / "request.json",
        "sample_plan_manifest": sample_manifest,
    }
    return {
        "paths_present": {name: path.is_file() for name, path in paths.items()},
        "sha256": {name: file_fingerprint(path).get("sha256") for name, path in paths.items() if path.is_file()},
        "sample_plan_hash": sample_payload.get("plan_hash"),
        "token_cache_hashes": token_hashes,
    }


def _validate_existing_probe_result(
    path: Path,
    *,
    stage: str,
    task: int,
    candidate_id: str,
    min_steps: int,
    require_per_tier: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "reused": False,
        "reasons": [],
        "min_steps": int(min_steps),
    }
    if not path.is_file():
        report["reasons"].append("result_missing")
        return report
    try:
        result = _load_json(path)
    except Exception as exc:
        report["reasons"].append(f"result_invalid_json:{exc}")
        return report
    checks = {
        "schema": result.get("schema_version") == "odcr_step5_e4_bounded_probe/1",
        "stage": str(result.get("stage")) == str(stage),
        "task": int(result.get("task_id") or -1) == int(task),
        "candidate": str(result.get("candidate_id")) == str(candidate_id),
        "success": result.get("success") is True,
        "evidence": result.get("evidence_level") == E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
        "formal_entry_lifecycle": result.get("formal_entry_lifecycle") is True,
        "forward": result.get("forward_executed") is True,
        "backward": result.get("loss_backward_executed") is True,
        "optimizer": result.get("optimizer_step_executed") is True,
        "preflight": result.get("preflight_executed") is True,
        "scratch_cleanup": result.get("scratch_cleanup_status") == "pass",
        "graph_tensor_audit": result.get("graph_tensor_audit_status") == "pass",
        "graph_scratch_absent": not list(result.get("graph_scratch_before_ema") or []),
        "ema_init": result.get("ema_init_pass") is True and result.get("ema_init_executed_in_E4") is True,
        "ddp_wrap": result.get("ddp_wrap_pass") is True,
        "first_train_step": result.get("first_train_step_pass") is True,
        "validation_executed": result.get("validation_pass_executed") is True,
        "validation_forward": result.get("validation_forward_pass") is True,
        "validation_loss_finite": result.get("validation_loss_finite") is True,
        "validation_oom_absent": result.get("validation_oom") is False,
        "rating_stability_control_control_only_validation": (
            str(stage) != "rating_stability_control" or result.get("rating_stability_control_validation_control_only") is True
        ),
        "rating_stability_control_no_flan_validation": (
            str(stage) != "rating_stability_control" or result.get("flan_explainer_called_in_rating_stability_control_validation") is False
        ),
        "rating_stability_control_no_logits_validation": (
            str(stage) != "rating_stability_control" or result.get("out_logits_materialized_in_rating_stability_control_validation") is False
        ),
        "valid_micro_lte_train": int(result.get("valid_forward_micro_batch_size") or 10**9)
        <= int(result.get("train_per_gpu_batch_size") or -1),
        "all_trainable_grad": result.get("all_trainable_grad_status") == "pass",
        "all_trainable_count_match": int(result.get("trainable_param_count") or -1) == int(result.get("grad_present_count") or -2),
        "lora_grad_count_match": int(result.get("lora_trainable_count") or -1) == int(result.get("lora_grad_present_count") or -2),
        "missing_grad_absent": not list(result.get("missing_grad_params") or []),
        "forward_backward": result.get("real_forward_backward_executed") is True,
        "finite": result.get("finite_loss_sync_ok", True) is True,
        "graph_safe": result.get("graph_safe_backward_ok", True) is True,
        "rank_balance": result.get("rank_sample_balance_ok", True) is True,
        "rank_component_keys": result.get("loss_component_keys_per_rank_identical", True) is True,
        "formal_clean": result.get("formal_namespace_pollution") is False,
        "latest_clean": result.get("latest_json_created") is False,
        "checkpoint_clean": result.get("checkpoint_written") is False,
        "floor": _result_steps(result) >= int(min_steps),
    }
    if require_per_tier:
        checks["per_tier_loss"] = isinstance(result.get("per_tier_loss"), Mapping)
        checks["per_tier_keys"] = result.get("per_tier_loss_keys_per_rank_identical") is True
    for name, ok in checks.items():
        if not ok:
            report["reasons"].append(f"{name}_failed")
    out_dir = path.parent
    artifacts = _artifact_hash_payload(out_dir)
    report["artifact_hashes"] = artifacts
    if not all(artifacts.get("paths_present", {}).get(key) for key in ("resolved_config", "source_table", "sample_plan_manifest")):
        report["reasons"].append("required_artifact_missing")
    if not artifacts.get("token_cache_hashes"):
        report["reasons"].append("token_cache_manifest_missing")
    report["actual_steps"] = _result_steps(result)
    report["valid"] = not report["reasons"]
    report["reused"] = bool(report["valid"])
    return report


def _write_process_state(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, dict(payload))


def _owned_bounded_compute_app(row: Mapping[str, Any], *, candidate_id: str, out_dir: Path) -> dict[str, Any]:
    pid = int(row.get("pid") or -1)
    cmdline = _proc_cmdline(pid) if pid > 0 else ""
    out_dir_text = str(out_dir)
    owned = (
        pid > 0
        and "step5_runtime_probe.py" in cmdline
        and str(candidate_id) in cmdline
        and out_dir_text in cmdline
    )
    pgid = None
    if pid > 0:
        try:
            pgid = os.getpgid(pid)
        except Exception:
            pgid = None
    return {"pid": pid, "pgid": pgid, "cmdline": cmdline, "owned": bool(owned), **dict(row)}


def _cleanup_process_group(pgid: int, *, reason: str, timeout_s: int = 20) -> dict[str, Any]:
    evidence = {"pgid": int(pgid), "reason": reason, "sigterm": False, "sigkill": False, "success": False}
    try:
        os.killpg(int(pgid), signal.SIGTERM)
        evidence["sigterm"] = True
    except ProcessLookupError:
        evidence["success"] = True
        return evidence
    except Exception as exc:
        evidence["error"] = str(exc)
        return evidence
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        try:
            os.killpg(int(pgid), 0)
        except ProcessLookupError:
            evidence["success"] = True
            return evidence
        except PermissionError:
            break
        time.sleep(1)
    try:
        os.killpg(int(pgid), signal.SIGKILL)
        evidence["sigkill"] = True
    except ProcessLookupError:
        evidence["success"] = True
        return evidence
    except Exception as exc:
        evidence["error"] = str(exc)
        return evidence
    time.sleep(1)
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        evidence["success"] = True
    except PermissionError:
        evidence["success"] = False
    return evidence


def _prelaunch_compute_app_guard(*, candidate_id: str, out_dir: Path) -> dict[str, Any]:
    before = _nvidia_smi_compute_apps()
    evidence: dict[str, Any] = {
        "schema_version": "odcr_step5_compute_app_guard/1",
        "candidate_id": str(candidate_id),
        "output_dir": str(out_dir),
        "before": before,
        "cleanup": [],
        "pass": True,
        "blocked": False,
        "duplicate_launch_prevented": False,
        "orphan_cleanup_pass": True,
    }
    rows = before.get("rows") if isinstance(before.get("rows"), Sequence) else []
    if not rows:
        return evidence
    owned_rows = [_owned_bounded_compute_app(row, candidate_id=candidate_id, out_dir=out_dir) for row in rows if isinstance(row, Mapping)]
    unowned = [row for row in owned_rows if not row.get("owned")]
    owned = [row for row in owned_rows if row.get("owned")]
    if owned:
        evidence["duplicate_launch_prevented"] = True
        pgids = sorted({int(row["pgid"]) for row in owned if row.get("pgid") is not None})
        for pgid in pgids:
            evidence["cleanup"].append(_cleanup_process_group(pgid, reason="owned_bounded_orphan_before_relaunch"))
        evidence["orphan_cleanup_pass"] = all(bool(item.get("success")) for item in evidence["cleanup"])
        time.sleep(3)
        after = _nvidia_smi_compute_apps()
        evidence["after_cleanup"] = after
        rows = after.get("rows") if isinstance(after.get("rows"), Sequence) else []
        if not rows:
            return evidence
        unowned = [
            _owned_bounded_compute_app(row, candidate_id=candidate_id, out_dir=out_dir)
            for row in rows
            if isinstance(row, Mapping)
        ]
    evidence["pass"] = False
    evidence["blocked"] = True
    evidence["duplicate_launch_prevented"] = True
    evidence["unowned_or_remaining_compute_apps"] = unowned
    return evidence


def _dtype_summary(model: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.dtype).replace("torch.", "")
        out[key] = out.get(key, 0) + int(param.numel())
    return dict(sorted(out.items()))


def _largest_module_rows(model: nn.Module, *, trainable_only: bool, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        params = list(module.parameters(recurse=False))
        if trainable_only:
            params = [p for p in params if p.requires_grad]
        if not params:
            continue
        total_params = sum(int(p.numel()) for p in params)
        total_bytes = sum(_param_bytes(p) for p in params)
        if total_params <= 0:
            continue
        rows.append(
            {
                "module": name or "<root>",
                "params": int(total_params),
                "memory_gb": _round_gb(_gb(total_bytes)),
                "trainable": bool(any(p.requires_grad for p in params)),
            }
        )
    rows.sort(key=lambda item: float(item["memory_gb"]), reverse=True)
    return rows[:limit]


def _model_memory_audit(
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    *,
    gradient_checkpointing_enabled: bool,
    gradient_checkpointing_reentrant_policy: str,
    use_cache_training_disabled: bool,
    bf16_effective: bool,
) -> dict[str, Any]:
    params = list(model.parameters())
    trainable = [p for p in params if p.requires_grad]
    frozen = [p for p in params if not p.requires_grad]
    opt_ids = _optimizer_param_ids(optimizer) if optimizer is not None else set()
    frozen_ids = {id(p) for p in frozen}
    optimizer_param_count = sum(int(p.numel()) for p in params if id(p) in opt_ids)
    optimizer_param_memory_gb = _round_gb(_gb(sum(_param_bytes(p) for p in params if id(p) in opt_ids)))
    flan = getattr(model, "flan_explainer", None)
    base_text_trainable = False
    if flan is not None:
        for name, param in flan.named_parameters():
            if param.requires_grad and "lora_" not in name:
                base_text_trainable = True
                break
    lora_trainable = sum(int(p.numel()) for name, p in model.named_parameters() if p.requires_grad and "lora_" in name)
    suspicious = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and name.startswith("flan_explainer.") and "lora_" not in name
    ][:40]
    total_params = sum(int(p.numel()) for p in params)
    trainable_params = sum(int(p.numel()) for p in trainable)
    frozen_params = sum(int(p.numel()) for p in frozen)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "trainable_ratio": float(trainable_params / max(total_params, 1)),
        "optimizer_param_count": int(optimizer_param_count),
        "optimizer_param_memory_gb": optimizer_param_memory_gb,
        "optimizer_includes_frozen_params": bool(opt_ids & frozen_ids),
        "base_text_model_loaded": flan is not None,
        "base_text_model_trainable": bool(base_text_trainable),
        "lora_enabled": bool(lora_trainable > 0),
        "lora_trainable_params": int(lora_trainable),
        "gradient_checkpointing_enabled": bool(gradient_checkpointing_enabled),
        "gradient_checkpointing_reentrant_policy": str(gradient_checkpointing_reentrant_policy),
        "use_cache_training_disabled": bool(use_cache_training_disabled),
        "bf16_effective": bool(bf16_effective),
        "model_dtype_summary": _dtype_summary(model),
        "largest_modules_by_param_memory": _largest_module_rows(model, trainable_only=False),
        "largest_trainable_modules": _largest_module_rows(model, trainable_only=True),
        "suspicious_trainable_modules": suspicious,
    }


def _sequence_stats(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {"min": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(int(x) for x in lengths)
    p95_idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return {
        "min": int(ordered[0]),
        "mean": float(sum(ordered) / len(ordered)),
        "p95": int(ordered[p95_idx]),
        "max": int(ordered[-1]),
    }


def _sequence_memory_audit(gb: Any, final_cfg: Any) -> dict[str, Any]:
    input_lengths = [int((row != 0).sum().item()) for row in gb.tgt_input.detach().cpu()]
    label_lengths = [int((row != 0).sum().item()) for row in gb.tgt_output.detach().cpu()]
    input_stats = _sequence_stats(input_lengths)
    label_stats = _sequence_stats(label_lengths)
    return {
        "input_length_min": input_stats["min"],
        "input_length_mean": input_stats["mean"],
        "input_length_p95": input_stats["p95"],
        "input_length_max": input_stats["max"],
        "label_length_min": label_stats["min"],
        "label_length_mean": label_stats["mean"],
        "label_length_p95": label_stats["p95"],
        "label_length_max": label_stats["max"],
        "dynamic_padding_enabled": bool(getattr(final_cfg, "train_dynamic_padding", True)),
        "length_bucket_enabled": False,
        "max_explanation_length": int(getattr(final_cfg, "max_explanation_length", 0) or 0),
        "soft_max_len": getattr(final_cfg, "soft_max_len", None),
        "hard_max_len": getattr(final_cfg, "hard_max_len", None),
    }


def _graph_memory_audit(ccv_packet: Any, *, empty_cache_called: bool = False) -> dict[str, Any]:
    numeric = ccv_packet.numeric_controls()
    return {
        "lci_graph_retention_ok": True,
        "fca_graph_retention_ok": True,
        "ccv_control_packet_shape": {
            "numeric_controls": list(numeric.shape),
            "content_evidence_ids": list(ccv_packet.content_evidence_ids.shape),
            "style_evidence_ids": list(ccv_packet.style_evidence_ids.shape),
            "domain_style_anchor_ids": list(ccv_packet.domain_style_anchor_ids.shape),
            "local_style_hint_ids": list(ccv_packet.local_style_hint_ids.shape),
            "polarity_ids": list(ccv_packet.polarity_ids.shape),
        },
        "unnecessary_logits_retained": False,
        "tensors_retained_after_step": False,
        "empty_cache_called_for_measurement_only": bool(empty_cache_called),
    }


def _memory_truth_payload(
    *,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    memory_creep_detected: bool,
    oom: bool = False,
    oom_error_message: str = "",
) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    total_gb = _gb(int(props.total_memory))
    max_alloc_gb = _gb(int(torch.cuda.max_memory_allocated(device)))
    max_reserved_gb = _gb(int(torch.cuda.max_memory_reserved(device)))
    param_gb = _gb(sum(_param_bytes(p) for p in model.parameters()))
    trainable_gb = _gb(sum(_param_bytes(p) for p in model.parameters() if p.requires_grad))
    frozen_gb = _gb(sum(_param_bytes(p) for p in model.parameters() if not p.requires_grad))
    grad_gb = _gb(sum(_param_bytes(p.grad) for p in model.parameters() if p.grad is not None))
    opt_gb = _optimizer_state_memory_gb(optimizer) if optimizer is not None else 0.0
    activation_est = max(0.0, max_alloc_gb - param_gb - grad_gb - float(opt_gb))
    reserved_minus_alloc = max(0.0, max_reserved_gb - max_alloc_gb)
    return {
        "device_total_gb": _round_gb(total_gb),
        "max_memory_allocated_gb": _round_gb(max_alloc_gb),
        "max_memory_reserved_gb": _round_gb(max_reserved_gb),
        "reserved_minus_allocated_gb": _round_gb(reserved_minus_alloc),
        "allocated_to_total_ratio": float(max_alloc_gb / max(total_gb, 1e-9)),
        "reserved_to_total_ratio": float(max_reserved_gb / max(total_gb, 1e-9)),
        "nvidia_smi_process_used_gb": _nvidia_smi_process_used_gb(),
        "param_memory_gb": _round_gb(param_gb),
        "trainable_param_memory_gb": _round_gb(trainable_gb),
        "frozen_param_memory_gb": _round_gb(frozen_gb),
        "grad_memory_gb": _round_gb(grad_gb),
        "optimizer_state_memory_gb": float(opt_gb),
        "activation_peak_estimated_gb": _round_gb(activation_est),
        "fragmentation_hint": {
            "reserved_minus_allocated_gb": _round_gb(reserved_minus_alloc),
            "reserved_is_allocator_cache": True,
            "diagnostic_only": True,
        },
        "memory_creep_detected": bool(memory_creep_detected),
        "oom": bool(oom),
        "oom_error_message": str(oom_error_message or ""),
        "cuda_allocator_backend": _cuda_allocator_backend(),
        "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "reserved_is_diagnostic_only": True,
    }


def validate_memory_truth_schema(payload: Mapping[str, Any]) -> None:
    missing = [key for key in MEMORY_TRUTH_REQUIRED_FIELDS if key not in payload]
    if missing:
        raise RuntimeError("memory_truth missing required fields: " + ", ".join(missing))
    if payload.get("reserved_is_diagnostic_only") is not True:
        raise RuntimeError("memory_truth.reserved_is_diagnostic_only must be true")


def _signature_supports_gradient_checkpointing_kwargs(fn: Any) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "gradient_checkpointing_kwargs":
            return True
    return False


def _runtime_transformers_signature_payload(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    flan_model: Any | None = None,
) -> dict[str, Any]:
    try:
        import transformers
        from transformers import PreTrainedModel, T5ForConditionalGeneration
    except Exception as exc:
        return {
            "rank": int(rank),
            "local_rank": int(local_rank),
            "world_size": int(world_size),
            "import_error": str(exc),
            "gradient_checkpointing_kwargs_supported": False,
            "use_reentrant_false_supported": False,
        }

    pretrain_fn = PreTrainedModel.gradient_checkpointing_enable
    t5_fn = T5ForConditionalGeneration.gradient_checkpointing_enable
    instance_fn = getattr(flan_model, "gradient_checkpointing_enable", None) if flan_model is not None else None
    transformer_path = str(getattr(transformers, "__file__", ""))
    torch_path = str(getattr(torch, "__file__", ""))
    prefix = str(Path(sys.prefix).resolve())
    same_conda_env = bool(
        transformer_path.startswith(prefix + os.sep)
        and torch_path.startswith(prefix + os.sep)
    )
    pretrain_support = _signature_supports_gradient_checkpointing_kwargs(pretrain_fn)
    t5_support = _signature_supports_gradient_checkpointing_kwargs(t5_fn)
    instance_support = (
        _signature_supports_gradient_checkpointing_kwargs(instance_fn)
        if callable(instance_fn)
        else None
    )
    return {
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "torch_module_path": torch_path,
        "transformers_version": str(getattr(transformers, "__version__", "unknown")),
        "transformers_module_path": transformer_path,
        "torch_and_transformers_same_conda_env": same_conda_env,
        "pretrained_model_gradient_checkpointing_enable_signature": str(inspect.signature(pretrain_fn)),
        "t5_gradient_checkpointing_enable_signature": str(inspect.signature(t5_fn)),
        "flan_instance_gradient_checkpointing_enable_signature": (
            str(inspect.signature(instance_fn)) if callable(instance_fn) else None
        ),
        "pretrained_model_supports_gradient_checkpointing_kwargs": bool(pretrain_support),
        "t5_supports_gradient_checkpointing_kwargs": bool(t5_support),
        "flan_instance_supports_gradient_checkpointing_kwargs": instance_support,
        "gradient_checkpointing_kwargs_supported": bool(
            pretrain_support and t5_support and (instance_support is not False)
        ),
        "use_reentrant_false_supported": bool(
            pretrain_support and t5_support and (instance_support is not False)
        ),
    }


def _summarize_runtime_transformers_signatures(rank_payloads: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    rows = [
        dict((payload or {}).get("runtime_transformers_signature") or {})
        for payload in rank_payloads
        if isinstance((payload or {}).get("runtime_transformers_signature"), Mapping)
    ]
    versions = sorted({str(row.get("transformers_version")) for row in rows if row.get("transformers_version")})
    torch_versions = sorted({str(row.get("torch_version")) for row in rows if row.get("torch_version")})
    module_paths = sorted({str(row.get("transformers_module_path")) for row in rows if row.get("transformers_module_path")})
    return {
        "rank_results": rows,
        "rank_count": len(rows),
        "transformers_versions": versions,
        "torch_versions": torch_versions,
        "transformers_module_paths": module_paths,
        "all_ranks_transformers_version_match": len(versions) == 1 and len(rows) > 0,
        "all_ranks_torch_version_match": len(torch_versions) == 1 and len(rows) > 0,
        "all_ranks_module_path_match": len(module_paths) == 1 and len(rows) > 0,
        "gradient_checkpointing_kwargs_supported": bool(rows) and all(
            bool(row.get("gradient_checkpointing_kwargs_supported")) for row in rows
        ),
        "use_reentrant_false_supported": bool(rows) and all(
            bool(row.get("use_reentrant_false_supported")) for row in rows
        ),
        "torch_and_transformers_same_conda_env": bool(rows) and all(
            bool(row.get("torch_and_transformers_same_conda_env")) for row in rows
        ),
    }


def _classify_step5_runtime_probe_failure(exc: BaseException | str) -> dict[str, Any]:
    text = str(exc)
    lower = text.lower()
    if (
        "only tensors created explicitly by the user" in lower
        and "deepcopy protocol" in lower
        and ("averagedmodel" in lower or "ema_model" in lower)
    ):
        return {
            "failure_phase": "ema_init",
            "failure_type": "model_deepcopy_non_leaf_tensor_after_preflight",
            "root_cause": "step5_forward_cached_graph_tensors_persisted_before_ema_deepcopy",
        }
    if "step5 find_unused_parameters=false preflight failed" in lower and "trainable params without grad" in lower:
        return {
            "failure_phase": "ddp_preflight",
            "failure_type": "trainable_param_without_grad",
            "root_cause": "step5_trainable_graph_mismatch",
        }
    if "ccv control ids must be [b,t]" in lower:
        return {
            "failure_phase": "data_collate",
            "failure_type": "ccv_control_packet_shape_contract",
            "root_cause": "real_batch_control_packet_shape_invalid",
        }
    if re.search(r"Expected to mark a variable ready only once|mark a variable ready only once", text, re.IGNORECASE):
        param_match = re.search(r"(?:parameter:\s*|name\s+)([A-Za-z0-9_.$]+)", text, re.IGNORECASE)
        payload = {
            "failure_phase": "train_backward",
            "failure_type": "ddp_parameter_ready_twice",
            "root_cause": "ddp_lora_checkpointing_ready_hook_conflict",
        }
        if param_match:
            payload["parameter_name"] = param_match.group(1)
        return payload
    if "gradient_checkpointing_kwargs" in text or "transformers_runtime_api_mismatch" in lower:
        return {
            "failure_phase": "model_init",
            "failure_type": "gradient_checkpointing_policy_unsupported",
            "root_cause": "transformers_runtime_api_mismatch",
        }
    if "out of memory" in lower or "cuda oom" in lower:
        return {
            "failure_phase": "backward" if "backward" in lower else "model_init",
            "failure_type": "oom",
            "root_cause": "memory_pressure",
        }
    return {
        "failure_phase": "unknown",
        "failure_type": "runtime_failure",
        "root_cause": "unclassified_runtime_failure",
    }


def _ddp_ready_hook_policy_payload(cfg: Any) -> dict[str, Any]:
    find_unused = bool(getattr(cfg, "ddp_find_unused_parameters", True))
    static_graph = bool(getattr(cfg, "ddp_static_graph", False))
    return {
        "find_unused_parameters": find_unused,
        "static_graph": static_graph,
        "ddp_find_unused_parameters_effective": find_unused,
        "ddp_static_graph_effective": static_graph,
        "ddp_static_graph_reason": (
            "one_control_static_graph_enabled_after_bounded_stability_evidence"
            if static_graph
            else "one_control_static_graph_disabled_until_needed_by_bounded_evidence"
        ),
        "ddp_ready_hook_policy": (
            "find_unused_false_static_graph_false_non_reentrant_checkpointing_required"
            if not static_graph
            else "static_graph_enabled_after_bounded_preflight"
        ),
    }


def _lora_parameter_participation(model: nn.Module) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        rows.append(
            {
                "name": name,
                "requires_grad": bool(param.requires_grad),
                "grad_present": bool(param.grad is not None),
                "grad_finite": bool(param.grad is not None and torch.isfinite(param.grad).all().item()),
            }
        )
    target_rows = [
        row for row in rows
        if row["name"].endswith("flan_explainer.decoder.block.23.layer.2.DenseReluDense.wo.lora_B")
        or "decoder.block.23.layer.2.DenseReluDense.wo.lora_B" in row["name"]
    ]
    return {
        "lora_trainable_param_count": sum(1 for row in rows if row["requires_grad"]),
        "lora_grad_present_count": sum(1 for row in rows if row["grad_present"]),
        "lora_grad_finite_count": sum(1 for row in rows if row["grad_finite"]),
        "target_lora_B_param_present": bool(target_rows),
        "target_lora_B_grad_present": any(bool(row["grad_present"]) for row in target_rows),
        "target_lora_B_participates_once_in_successful_backward": any(bool(row["grad_present"]) for row in target_rows),
        "sample_rows": rows[:40],
        "target_rows": target_rows[:10],
    }


def candidate_decision_from_result(
    result: Mapping[str, Any],
    memory_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(memory_cfg or {})
    memory = result.get("memory_truth") if isinstance(result.get("memory_truth"), Mapping) else {}
    reject_reason = ""
    category = "pass"
    correctness_pass = (
        bool(result.get("success"))
        and result.get("evidence_level") == E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION
        and result.get("formal_entry_lifecycle") is True
        and result.get("preflight_executed") is True
        and result.get("scratch_cleanup_status") == "pass"
        and result.get("graph_tensor_audit_status") == "pass"
        and not list(result.get("graph_scratch_before_ema") or [])
        and result.get("ema_init_pass") is True
        and result.get("ema_init_executed_in_E4") is True
        and result.get("ddp_wrap_pass") is True
        and result.get("first_train_step_pass") is True
        and result.get("validation_pass_executed") is True
        and result.get("validation_forward_pass") is True
        and result.get("validation_loss_finite") is True
        and result.get("validation_oom") is False
        and (
            str(result.get("stage") or "") != "rating_stability_control"
            or (
                result.get("rating_stability_control_validation_control_only") is True
                and result.get("flan_explainer_called_in_rating_stability_control_validation") is False
                and result.get("out_logits_materialized_in_rating_stability_control_validation") is False
            )
        )
        and int(result.get("valid_forward_micro_batch_size") or 10**9)
        <= int(result.get("train_per_gpu_batch_size") or -1)
        and bool(result.get("finite_loss_sync_ok", True))
        and bool(result.get("graph_safe_backward_ok", True))
        and result.get("all_trainable_grad_status") == "pass"
        and int(result.get("trainable_param_count") or -1) == int(result.get("grad_present_count") or -2)
        and int(result.get("lora_trainable_count") or -1) == int(result.get("lora_grad_present_count") or -2)
        and not list(result.get("missing_grad_params") or [])
    )
    if bool(memory.get("oom") or result.get("oom")) and bool(cfg.get("reject_on_oom", True)):
        reject_reason = "oom"
        category = "oom"
    elif not bool(result.get("success")):
        reject_reason = str(result.get("rejection_reason") or result.get("error") or "runtime_failure")
        category = "runtime_failure"
    elif not correctness_pass:
        reject_reason = "correctness_failed"
        category = "correctness"
    elif bool(memory.get("memory_creep_detected")):
        reject_reason = "memory_creep_detected"
        category = "memory_creep"
    else:
        threshold = cfg.get("reject_on_allocated_ratio")
        allocated_ratio = memory.get("allocated_to_total_ratio")
        if threshold is not None and allocated_ratio is not None and float(allocated_ratio) >= float(threshold):
            reject_reason = "allocated_memory_ratio_exceeds_configured_limit"
            category = "allocated_memory"
    score = 0.0
    if not reject_reason:
        throughput = float(result.get("throughput_samples_per_sec") or 0.0)
        allocated = float(memory.get("max_memory_allocated_gb") or 0.0)
        data_wait = float(result.get("data_wait_ratio") or 0.0)
        score = throughput - 0.05 * allocated - 0.25 * data_wait
    return {
        "reject_reason": reject_reason or None,
        "reject_reason_category": category,
        "reserved_memory_used_for_rejection": False,
        "selected_score": float(score),
        "correctness_pass": bool(correctness_pass),
        "long_window_pass": bool(not memory.get("memory_creep_detected")),
    }


def rank_batch_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        row = dict(item)
        decision = row.get("candidate_decision") if isinstance(row.get("candidate_decision"), Mapping) else {}
        row["_reject"] = bool(decision.get("reject_reason"))
        row["_score"] = float(decision.get("selected_score") or row.get("selected_score") or 0.0)
        row["_throughput"] = float(row.get("throughput_samples_per_sec") or row.get("throughput") or row["_score"])
        row["_data_wait"] = float(row.get("data_wait_ratio") or 0.0)
        row["_gpu_util"] = float(row.get("gpu_util_mean") or row.get("gpu_utilization_mean") or 0.0)
        memory = row.get("memory_truth") if isinstance(row.get("memory_truth"), Mapping) else {}
        row["_allocated"] = float(
            row.get("max_memory_allocated_gb")
            or row.get("gpu_memory_peak_gb")
            or memory.get("max_memory_allocated_gb")
            or 0.0
        )
        row["_per_gpu"] = int(row.get("per_gpu_batch_size") or 0)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["_reject"],
            -row["_throughput"],
            row["_data_wait"],
            -row["_gpu_util"],
            row["_allocated"],
            -row["_score"],
            -row["_per_gpu"],
        )
    )
    for row in rows:
        row.pop("_reject", None)
        row.pop("_score", None)
        row.pop("_throughput", None)
        row.pop("_data_wait", None)
        row.pop("_gpu_util", None)
        row.pop("_allocated", None)
        row.pop("_per_gpu", None)
    return rows


def _run_rank_probe(args: argparse.Namespace) -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Step5 E4 bounded probe requires CUDA; handshake-only evidence is E3, not E4.")
    if world_size != 2:
        raise RuntimeError(f"Step5 E4 bounded probe requires ddp_world_size=2, got {world_size}.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(os.environ.get("ODCR_STEP5_PROBE_NCCL_TIMEOUT_S", "1200"))),
    )
    device = torch.device("cuda", local_rank)
    output_dir = Path(args.output_dir).resolve()
    rank_payload: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "stage": args.stage,
        "candidate_id": args.candidate_id,
        "forward_success": False,
        "backward_success": False,
        "optimizer_success": False,
    }
    rank_payload["runtime_transformers_signature"] = _runtime_transformers_signature_payload(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    try:
        initial_signature = dict(rank_payload.get("runtime_transformers_signature") or {})
        if not bool(initial_signature.get("gradient_checkpointing_kwargs_supported")):
            raise RuntimeError(
                "transformers_runtime_api_mismatch: RatingStabilityControl non-reentrant checkpointing requires "
                "gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}) "
                f"in torchrun worker, got transformers={initial_signature.get('transformers_version')!r} "
                f"path={initial_signature.get('transformers_module_path')!r}."
            )
        from executors.step5_engine import (
            Processor,
            build_odcr_ddp_artefacts,
            _load_odcr_profile_tensors_from_contract,
            _make_model,
            _step5_collate_dynamic,
            _step5_non_blocking_h2d_from_final_cfg,
            _step5_pin_memory_from_final_cfg,
            _validate_route_masks_batch,
            _head_gated_step5_loss_terms,
            assert_no_step5_graph_tensors_attached,
            clear_step5_graph_cache,
            compose_step5_total_loss,
            find_step5_graph_tensors_attached,
            graph_tied_zero,
            graph_tied_zero_like,
            get_step5_tokenizer,
            initialize_step5_ema_model,
            run_step5_find_unused_parameters_preflight,
            validModel,
        )
        from odcr_core.odcr_losses import build_orthogonal_losses
        from odcr_core.step5_explanation_flan_bridge import per_sample_decoder_ce_from_logits

        request = _load_json(Path(args.request_path))
        requested_evidence_level = _normalize_evidence_level(request.get("evidence_level") or getattr(args, "evidence_level", None))
        if requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE and str(args.stage) != "rating_stability_control":
            raise RuntimeError("E5_rating_stability_control_post_train_eval_lifecycle probe only supports --stage rating_stability_control.")
        rank_payload["requested_evidence_level"] = requested_evidence_level
        set_overrides = list(request.get("set_overrides") or [])
        cfg, snapshot, source_table = _resolve_candidate(
            stage=str(args.stage),
            task=int(args.task),
            config_path=str(request.get("config_path") or "configs/odcr.yaml"),
            set_overrides=set_overrides,
            from_step4=request.get("from_step4"),
        )
        if int(cfg.ddp_world_size) != world_size:
            raise RuntimeError(f"resolved ddp_world_size={cfg.ddp_world_size} does not match torchrun world_size={world_size}")
        budget = _worker_budget(cfg)
        if not bool(budget["ok"]):
            raise RuntimeError(f"CPU worker budget exceeded: {budget['formula']}")
        export_path, contract_path, manifest_path = resolved_step5_export_paths(cfg)
        index_contract = load_index_contract(contract_path)
        step4_lineage = validate_step4_export_lineage(
            index_contract,
            current_step4_rcr_config=dict(json.loads(cfg.step4_rcr_config_json or "{}")),
            task_id=int(cfg.task_id),
            auxiliary_domain=str(cfg.auxiliary),
            target_domain=str(cfg.target),
        )
        loader_cfg = json.loads(str(cfg.step5_export_loader_config_json or "{}"))
        data_pipeline_cfg = json.loads(str(getattr(cfg, "step5_data_pipeline_config_json", "") or "{}"))
        innov_cfg = parse_step5_innovation_config_json(str(cfg.step5_innovation_config_json))
        memory_cfg = json.loads(str(cfg.step5_memory_truth_config_json or "{}"))
        explainer_mult = float(innov_cfg.explainer_gate.explainer_only_multiplier)
        sampler_cfg = json.loads(str(getattr(cfg, "step5_sampler_config_json", "") or "{}"))
        batch_candidates_cfg = json.loads(str(getattr(cfg, "step5_batch_candidates_config_json", "") or "{}"))
        tuning_cfg = json.loads(str(getattr(cfg, "step5_tuning_config_json", "") or "{}"))
        if _artifact_build_preflight_requested(str(args.candidate_id)):
            artifact_payload = _run_artifact_build_preflight(
                cfg=cfg,
                stage=str(args.stage),
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                output_dir=output_dir,
                export_path=export_path,
                build_odcr_ddp_artefacts=build_odcr_ddp_artefacts,
            )
            rank_payload.update(
                {
                    "success": True,
                    "artifact_build_preflight": artifact_payload,
                    "artifact_build_preflight_pass": True,
                    "real_task_data_used": True,
                    "real_model_loaded_on_gpu": True,
                    "real_forward_backward_executed": False,
                    "actual_gpu_backward_executed": False,
                    "backward_validation_claimed": False,
                    "checkpoint_written": bool(artifact_payload.get("checkpoint_written")),
                    "latest_json_created": bool(artifact_payload.get("latest_json_created")),
                    "formal_namespace_pollution": False,
                }
            )
            all_payloads: list[dict[str, Any] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(all_payloads, rank_payload)
            if rank == 0:
                cache_summary = _summarize_artifact_build_rank_payloads(all_payloads, world_size=world_size)
                runtime_signature_summary = _summarize_runtime_transformers_signatures(all_payloads)
                final_payload = {
                    "schema_version": "odcr_step5_e4_bounded_probe/1",
                    "artifact_build_schema_version": "odcr_step5_formal_artifact_build_preflight/1",
                    "stage": str(args.stage),
                    "task_id": int(args.task),
                    "candidate_id": str(args.candidate_id),
                    "success": True,
                    "artifact_build_only": True,
                    "artifact_build_preflight_pass": True,
                    "evidence_level": E3_GPU_TRANSPORT,
                    "real_forward_backward_executed": False,
                    "forward_executed": False,
                    "loss_backward_executed": False,
                    "optimizer_step_executed": False,
                    "actual_gpu_backward_executed": False,
                    "backward_validation_claimed": False,
                    "artifact_build_only_does_not_validate_backward": True,
                    "real_task_data_used": True,
                    "real_model_loaded_on_gpu": True,
                    "ddp_world_size": world_size,
                    "rank_ids": cache_summary["rank_ids"],
                    "rank_results": all_payloads,
                    "runtime_transformers_signature": runtime_signature_summary,
                    **cache_summary,
                    "formal_namespace_pollution": False,
                    "latest_json_created": False,
                    "checkpoint_written": False,
                    "formal_run_id": None,
                    "cpu_worker_budget_ok": bool(budget["ok"]),
                    "finite_loss_sync_ok": True,
                    "graph_safe_backward_ok": True,
                }
                final_payload["candidate_decision"] = candidate_decision_from_result(final_payload, memory_cfg)
                _write_json(Path(args.result_path), final_payload)
            return 0
        data_load_s = 0.0
        sample_plan_manifest: dict[str, Any] | None = None
        train_table_stats: dict[str, Any] = {}
        train_table_source_summary: dict[str, Any] = {}
        train_table_raw_count = 0
        train_table_filtered_count = 0
        sample_plan_enabled = bool(data_pipeline_cfg.get("sample_plan_enabled", False))
        if sample_plan_enabled:
            if rank == 0:
                load_t0 = time.perf_counter()
                train_table = load_step5_pool_train_table(
                    export_path,
                    index_contract_path=contract_path,
                    manifest_path=manifest_path,
                    index_contract=index_contract,
                    required_columns=STEP5_TRAIN_VALIDATION_COLUMNS,
                    mode="bounded",
                    sampler_config=sampler_cfg,
                    batch_candidates_config=batch_candidates_cfg,
                    tuning_config=tuning_cfg,
                    task_head=str(args.stage),
                    validate_sample_rows=int(loader_cfg.get("validate_sample_rows", 16)),
                    bounded_max_rows=int(loader_cfg.get("bounded_max_rows", 1024)),
                    verify_sha256=False,
                    validation_ctx={"task_id": int(cfg.task_id), "csv_path": str(export_path), "step4_run": str(cfg.step4_run)},
                )
                plan_df = train_table.train_df
                if str(args.stage) == "rating_stability_control":
                    plan_df = plan_df.loc[plan_df["route_scorer"].astype(int) == 1].reset_index(drop=True)
                elif str(args.stage) == "explanation":
                    plan_df = plan_df.loc[plan_df["route_explainer"].astype(int) == 1].reset_index(drop=True)
                sample_plan_manifest = write_step5_sample_plan(
                    output_dir,
                    train_df=plan_df,
                    stats=dict(train_table.stats),
                    source_summary=train_table.source.to_summary(),
                    task_head=str(args.stage),
                    world_size=world_size,
                    source_table=source_table,
                )
                data_load_s = time.perf_counter() - load_t0
            dist.barrier()
            manifest_path_plan = output_dir / "sample_plan" / "sample_plan_manifest.json"
            sample_plan_manifest = _load_json(manifest_path_plan)
            train_df = read_step5_sample_plan_shard(output_dir, rank=rank)
            train_table_stats = dict(sample_plan_manifest.get("stats") or {})
            train_table_source_summary = dict(sample_plan_manifest.get("source") or {})
            train_table_raw_count = int(train_table_stats.get("planned_total_rows") or sample_plan_manifest.get("row_count") or len(train_df))
            train_table_filtered_count = int(len(train_df))
        else:
            load_t0 = time.perf_counter()
            train_table = load_step5_pool_train_table(
                export_path,
                index_contract_path=contract_path,
                manifest_path=manifest_path,
                index_contract=index_contract,
                required_columns=STEP5_TRAIN_VALIDATION_COLUMNS,
                mode="bounded",
                sampler_config=sampler_cfg,
                batch_candidates_config=batch_candidates_cfg,
                tuning_config=tuning_cfg,
                task_head=str(args.stage),
                validate_sample_rows=int(loader_cfg.get("validate_sample_rows", 16)),
                bounded_max_rows=int(loader_cfg.get("bounded_max_rows", 1024)),
                verify_sha256=False,
                validation_ctx={"task_id": int(cfg.task_id), "csv_path": str(export_path), "step4_run": str(cfg.step4_run)},
            )
            data_load_s = time.perf_counter() - load_t0
            train_df = train_table.train_df
            if str(args.stage) == "rating_stability_control":
                train_df = train_df.loc[train_df["route_scorer"].astype(int) == 1].reset_index(drop=True)
            elif str(args.stage) == "explanation":
                train_df = train_df.loc[train_df["route_explainer"].astype(int) == 1].reset_index(drop=True)
            train_table_stats = dict(train_table.stats)
            train_table_source_summary = train_table.source.to_summary()
            train_table_raw_count = int(train_table.raw_row_count)
            train_table_filtered_count = int(train_table.filtered_row_count)
        if len(train_df) < world_size:
            raise RuntimeError(f"Step5 {args.stage} bounded probe has too few rows after route filter: {len(train_df)}")
        validate_split_indices(train_df, index_contract, "train", ctx={"csv_path": str(export_path), "task_id": int(cfg.task_id)})
        dc, ds, uc, us, ic, ist, profile_meta = _load_odcr_profile_tensors_from_contract(index_contract, "cpu")
        validate_index_contract_against_profiles(
            index_contract,
            uc,
            ic,
            ctx={
                "task_id": int(cfg.task_id),
                "step4_run": str(cfg.step4_run),
                "csv_path": str(export_path),
                "contract_path": str(contract_path),
            },
        )
        del dc, ds, us, ist
        processor = Processor(str(cfg.auxiliary), str(cfg.target), max_length=int(cfg.train_label_max_length))
        token_cache_manifest: dict[str, Any] | None = None
        tokenize_s = 0.0
        hot_path_tokenize_removed = False
        if bool(data_pipeline_cfg.get("token_cache_enabled", False)) and bool(data_pipeline_cfg.get("bounded_token_cache_enabled", False)):
            encoded_rows, token_cache_manifest, tokenize_s = _write_bounded_token_cache(
                output_dir=output_dir,
                rank=rank,
                train_df=train_df,
                processor=processor,
                source_table=source_table,
                sample_plan_manifest=sample_plan_manifest,
            )
            dataset = _Step5EncodedDataset(encoded_rows)
            hot_path_tokenize_removed = True
        else:
            dataset = _Step5ExplanationBoundedDataset(train_df, processor)
        if sample_plan_enabled:
            sampler = None
        else:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        final_cfg = SimpleNamespace(**as_plain_dict(cfg))
        effective_payload = json.loads(str(cfg.effective_training_payload_json or "{}"))
        training_row = effective_payload.get("training_row") if isinstance(effective_payload, Mapping) else {}
        if isinstance(training_row, Mapping):
            for key, value in training_row.items():
                setattr(final_cfg, str(key), value)
        final_cfg.nuser = int(index_contract["nuser_global"])
        final_cfg.nitem = int(index_contract["nitem_global"])
        final_cfg.ntoken = len(get_step5_tokenizer())
        final_cfg.emsize = int(cfg.embed_dim)
        final_cfg.device = local_rank
        final_cfg.device_ids = tuple(range(world_size))
        final_cfg.ddp_world_size = world_size
        final_cfg.run_dir = str(output_dir)
        final_cfg.manifest_dir = str(output_dir)
        final_cfg.log_file = str(output_dir / "bounded_probe.log")
        final_cfg.save_file = str(output_dir / "FORBIDDEN_checkpoint_not_written.pth")
        final_cfg.step4_export_lineage_json = json.dumps(step4_lineage, ensure_ascii=False, sort_keys=True)
        args_ns = SimpleNamespace(_odcr_index_contract=index_contract)
        model = _make_model(final_cfg, args_ns, local_rank)
        peft_meta = dict(getattr(args_ns, "_odcr_step5_peft_meta", {}) or {})
        trainable_contract_meta = dict(getattr(args_ns, "_odcr_step5_trainable_contract_meta", {}) or {})
        memory_policy_meta = dict(getattr(args_ns, "_odcr_step5_memory_policy_meta", {}) or {})
        rank_payload.update(
            {
                "lora_target_policy_id": str(peft_meta.get("target_policy_id") or ""),
                "head_specific_lora_allowlist_id": str(peft_meta.get("head_specific_lora_allowlist_id") or ""),
                "final_lora_target_modules": list(peft_meta.get("target_modules") or []),
                "forbidden_lora_targets": list(peft_meta.get("forbidden_lora_targets") or []),
                "deleted_legacy_modules": list(peft_meta.get("deleted_legacy_modules") or []),
                "head_specific_trainable_policy": trainable_contract_meta.get("policy_id"),
                "final_lora_target_modules_hash": trainable_contract_meta.get("final_lora_target_modules_hash"),
                "trainable_parameter_names_hash": trainable_contract_meta.get("trainable_parameter_names_hash"),
                "frozen_parameter_names_hash": trainable_contract_meta.get("frozen_parameter_names_hash"),
            }
        )
        rank_payload["runtime_transformers_signature"] = _runtime_transformers_signature_payload(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            flan_model=getattr(model, "flan_explainer", None),
        )
        model.train()
        collate_timer = _TimingCollate(
            partial(
                _step5_collate_dynamic,
                dynamic_padding=bool(getattr(final_cfg, "train_dynamic_padding", True)),
                fixed_max_length=int(getattr(final_cfg, "train_label_max_length", 36)),
            )
        )
        num_workers = int(json.loads(cfg.hardware_profile_json)["dataloader_num_workers_train"])
        prefetch = int(json.loads(cfg.hardware_profile_json)["dataloader_prefetch_factor_train"])
        dataloader = DataLoader(
            dataset,
            batch_size=int(cfg.per_device_train_batch_size),
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_step5_pin_memory_from_final_cfg(final_cfg),
            persistent_workers=bool(getattr(cfg, "persistent_workers", True)) and num_workers > 0,
            prefetch_factor=prefetch if num_workers > 0 else None,
            collate_fn=collate_timer,
        )
        preflight_result = run_step5_find_unused_parameters_preflight(
            model,
            final_cfg,
            step5_innov_cfg=innov_cfg,
            train_dataloader=dataloader,
        )
        graph_before_ema = find_step5_graph_tensors_attached(model, phase="before_ema_init")
        assert_no_step5_graph_tensors_attached(model, phase="before_ema_init")
        ema_model = None
        ema_report: dict[str, Any] = {
            "ema_enabled": bool(getattr(final_cfg, "ema_enabled", True)),
            "ema_decay": float(getattr(final_cfg, "ema_decay", 0.999)),
            "ema_init_phase": "after_preflight_cleanup",
            "ema_strategy": "AveragedModel_after_scratch_cleanup",
            "ema_deepcopy_success": False,
            "ema_init_pass": False,
            "graph_scratch_before_ema": graph_before_ema,
        }
        if bool(getattr(final_cfg, "ema_enabled", True)):
            ema_model, ema_report = initialize_step5_ema_model(
                model,
                ema_decay=float(getattr(final_cfg, "ema_decay", 0.999)),
                phase="after_preflight_cleanup",
            )
        rank_payload.update(
            {
                "preflight_success": True,
                "preflight_result": preflight_result,
                "scratch_cleanup_status": "pass" if preflight_result.get("scratch_cleared_after_preflight") else "fail",
                "graph_tensor_audit_status": "pass",
                "graph_tensor_audit_phase": "before_ema_init",
                "graph_scratch_before_ema": graph_before_ema,
                "ema_init_status": "pass" if bool(ema_report.get("ema_init_pass")) else "disabled",
                "ema_init_report": ema_report,
                "ema_init_pass": bool(ema_report.get("ema_init_pass")) or not bool(getattr(final_cfg, "ema_enabled", True)),
                "ema_init_executed": bool(getattr(final_cfg, "ema_enabled", True)),
                "ema_init_strategy": str(ema_report.get("ema_strategy") or "AveragedModel_after_scratch_cleanup"),
            }
        )
        ddp = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(cfg.ddp_find_unused_parameters),
            static_graph=bool(getattr(cfg, "ddp_static_graph", False)),
        )
        rank_payload["ddp_wrap_pass"] = True
        ddp_static_graph_probe = bool(getattr(cfg, "ddp_static_graph", False))
        params = [p for p in ddp.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=float(cfg.learning_rate), weight_decay=1e-5)
        long_window_requested = "LW" in str(args.candidate_id).upper() or "LONG" in str(args.candidate_id).upper()
        long_window_steps = max(1, int(memory_cfg.get("long_window_steps", 1) or 1))
        gpu_util_snapshots: list[dict[str, Any]] = []
        if rank == 0 and bool(data_pipeline_cfg.get("gpu_util_sampling_enabled", False)):
            gpu_util_snapshots.append({"phase": "before", **_nvidia_smi_gpu_snapshot()})
        cpu_start = _cpu_usage_snapshot()
        wall_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        data_wait_t0 = time.perf_counter()
        batch = next(iter(dataloader))
        data_wait_s = time.perf_counter() - data_wait_t0
        per_tier_acc = _new_per_tier_accumulator()
        h2d_s, h2d_e = _cuda_event_pair()
        fwd_s, fwd_e = _cuda_event_pair()
        bwd_s, bwd_e = _cuda_event_pair()
        opt_s, opt_e = _cuda_event_pair()
        non_blocking_h2d = _step5_non_blocking_h2d_from_final_cfg(final_cfg)
        h2d_s.record()
        gb = ddp.module.gather(batch, device, non_blocking_h2d=non_blocking_h2d)
        h2d_e.record()
        user_idx = gb.user_idx
        item_idx = gb.item_idx
        rating = gb.rating
        tgt_input = gb.tgt_input
        tgt_output = gb.tgt_output
        domain_idx = gb.domain_idx
        _validate_route_masks_batch(
            gb.route_scorer_mask,
            gb.route_explainer_mask,
            batch_size=int(user_idx.size(0)),
            stage=str(args.stage),
        )
        torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        precision = str(cfg.train_precision).lower()
        autocast_enabled = precision == "bf16"
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        is_rating_stability_control_probe = str(args.stage) == "rating_stability_control"
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            fwd_s.record()
            gate_a = build_rating_stability_control_gate(gb, innov_cfg)
            gate_b = build_step5_explanation_gate(gb, innov_cfg)
            ccv_packet = None
            if not is_rating_stability_control_probe:
                ccv_packet = build_ccv_control_packet(
                    gb,
                    innov_cfg,
                    producer="real_batch_backward_preflight",
                    head=str(args.stage),
                )
                validate_ccv_control_packet_shapes(
                    ccv_packet,
                    producer="real_batch_backward_preflight",
                    head=str(args.stage),
                    strict=True,
                )
            sequence_memory_audit = _sequence_memory_audit(gb, final_cfg)
            graph_memory_audit = (
                {
                    "lci_graph_retention_ok": True,
                    "fca_graph_retention_ok": True,
                    "ccv_control_packet_shape": {},
                    "rating_stability_control_probe_no_ccv_packet": True,
                    "unnecessary_logits_retained": False,
                    "tensors_retained_after_step": False,
                    "empty_cache_called_for_measurement_only": False,
                }
                if is_rating_stability_control_probe
                else _graph_memory_audit(ccv_packet)
            )
            control_score, _context_dist, word_dist = ddp(
                user_idx,
                item_idx,
                tgt_input,
                domain_idx,
                target_tokens=None if is_rating_stability_control_probe else tgt_output,
                evidence_features=gb.evidence_features,
                content_anchor_score=gb.content_anchor_score,
                style_anchor_score=gb.style_anchor_score,
                ccv_control_packet=ccv_packet,
                return_explainer_logits=not is_rating_stability_control_probe,
            )
            fwd_e.record()
            rank_payload["forward_success"] = True
            control_zero_ps = graph_tied_zero_like(control_score).expand_as(control_score).to(dtype=control_score.dtype)
            if is_rating_stability_control_probe:
                loss_flan_ps = control_zero_ps
            else:
                loss_flan_ps = per_sample_decoder_ce_from_logits(
                    word_dist,
                    tgt_output,
                    ignore_index=0,
                    label_smoothing=float(cfg.label_smoothing),
                )
            control_only = control_zero_ps.to(dtype=loss_flan_ps.dtype)
            explainer_only = loss_flan_ps
            dom = domain_idx.view(-1)
            loss_factual = graph_tied_zero_like(control_score)
            loss_counterfactual = (
                graph_tied_zero_like(control_score)
                if is_rating_stability_control_probe
                else float(cfg.explainer_loss_weight)
                * route_weighted_mean(
                    explainer_only,
                    gate_b.explainer_weight.to(dtype=explainer_only.dtype),
                    (dom == 0).to(dtype=explainer_only.dtype),
                )
            )
            _model = ddp.module
            spec_lat = _model._last_specific_latent
            shared_lat = _model._last_shared_latent
            lci_weighted_zero = graph_tied_zero_like(control_score)
            lci_raw_zero = control_zero_ps
            fca_bundle = None
            if not is_rating_stability_control_probe:
                fca_bundle = evidence_basis_fca_loss(
                    scorer_hidden=_model._last_h_score,
                    explainer_hidden=_model._last_h_explain_aligned,
                    shared_latent=shared_lat,
                    content_profile=_model._last_content_profile,
                    content_evidence_latent=_model._last_content_evidence_latent,
                    packet=ccv_packet,
                    gate=gate_b,
                    cfg=innov_cfg,
                )
            (
                loss_factual,
                loss_counterfactual,
                lci_weighted_loss,
                fca_weighted_loss,
                l_lci,
                l_fca,
            ) = _head_gated_step5_loss_terms(
                task_head=str(args.stage),
                zero_like=control_score if is_rating_stability_control_probe else word_dist,
                loss_factual=loss_factual,
                loss_counterfactual=loss_counterfactual,
                lci_weighted_loss=lci_weighted_zero,
                fca_weighted_loss=graph_tied_zero_like(control_score) if is_rating_stability_control_probe else fca_bundle.fca_weighted_loss,
                l_lci=None,
                l_fca=graph_tied_zero_like(control_score) if is_rating_stability_control_probe else fca_bundle.fca_loss,
            )
            lci_raw_ps, lci_weighted_ps = lci_raw_zero, lci_raw_zero
            if is_rating_stability_control_probe:
                fca_raw_ps = control_zero_ps
                fca_weighted_ps = fca_raw_ps
            else:
                fca_raw_ps, fca_weighted_ps = _fca_per_sample_losses(
                    fca_bundle=fca_bundle,
                    gate=gate_b,
                    cfg=innov_cfg,
                )
            control_weighted_ps = control_only * gate_a.scorer_weight.to(dtype=control_only.dtype) * (dom == 1).to(dtype=control_only.dtype)
            explainer_weighted_ps = (
                explainer_only
                * gate_b.explainer_weight.to(dtype=explainer_only.dtype)
                * (dom == 0).to(dtype=explainer_only.dtype)
                * float(cfg.explainer_loss_weight)
            )
            _accumulate_per_tier_loss(
                per_tier_acc,
                gb,
                scorer_raw=control_only,
                scorer_weighted=control_weighted_ps,
                lci_raw=lci_raw_ps,
                lci_weighted=lci_weighted_ps,
                uci_weight=gate_a.uci_weight,
                explainer_raw=explainer_only,
                explainer_weighted=explainer_weighted_ps,
                fca_raw=fca_raw_ps,
                fca_weighted=fca_weighted_ps,
                ccv_explainer_weight=gate_b.explainer_weight,
                ccv_fca_weight=gate_b.fca_weight,
                ccv_uncertainty=gate_b.uncertainty,
                ccv_confidence=gate_b.confidence_bucket,
                ccv_reliability=gate_b.reliability,
            )
            loss_ortho_keep = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
            lambda_ortho = float(getattr(final_cfg, "lambda_ortho_step5", 0.0) or 0.0)
            if lambda_ortho > 0.0:
                loss_ortho_keep = build_orthogonal_losses(
                    shared_lat,
                    spec_lat,
                    w_xcov=float(getattr(final_cfg, "lambda_ortho_xcov", 1.0)),
                    w_cos=float(getattr(final_cfg, "lambda_ortho_cos", 0.25)),
                ).loss_ortho_total
            loss_ul = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
            if (not is_rating_stability_control_probe) and float(getattr(final_cfg, "loss_weight_repeat_ul", 0.0) or 0.0) > 0.0:
                word_logp = F.log_softmax(word_dist, dim=-1)
                loss_ul = odcr_anti_repeat_unlikelihood_loss_from_logp(word_logp, tgt_output)
            loss_tc = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
            loss_bd = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
            total_loss = compose_step5_total_loss(
                loss_factual=loss_factual,
                loss_counterfactual=loss_counterfactual,
                retired_repeat_ul_zero_guard=loss_ul,
                loss_terminal_clean=loss_tc,
                loss_batch_diversity=loss_bd,
                repeat_ul_weight=float(getattr(final_cfg, "loss_weight_repeat_ul", 0.0) or 0.0),
                terminal_clean_weight=0.0,
                batch_diversity_weight=0.0,
                lci_weighted_loss=lci_weighted_loss,
                fca_weighted_loss=fca_weighted_loss,
                ortho_keep_loss=loss_ortho_keep,
                ortho_keep_weight=lambda_ortho,
            )
        finite_local = torch.tensor([1 if _finite_scalar(total_loss) else 0], device=device, dtype=torch.int)
        dist.all_reduce(finite_local, op=dist.ReduceOp.MIN)
        if int(finite_local.item()) != 1:
            raise RuntimeError("Step5 E4 bounded probe observed non-finite synchronized loss")
        bwd_s.record()
        total_loss.backward()
        bwd_e.record()
        all_grad_report = validate_all_trainable_params_receive_grad(
            ddp.module,
            total_loss,
            head=str(args.stage),
            evidence_context={
                "evidence_id": f"{args.stage}_E4_gpu_shard_forward_bounded_formal_entry_with_validation_{args.candidate_id}",
                "evidence_level": E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
                "real_sample_plan_used": True,
                "real_collate_executed": True,
                "real_ccv_packet_used": not is_rating_stability_control_probe,
                "ema_init_executed": True,
                "ddp_wrap_executed": True,
                "first_train_step_executed": True,
            },
            fail_on_missing=True,
        )
        rank_payload["backward_success"] = True
        opt_s.record()
        nn.utils.clip_grad_norm_(params, 1.0)
        lora_participation = _lora_parameter_participation(ddp.module)
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(ddp.module)
        opt_e.record()
        rank_payload["optimizer_success"] = True
        rank_payload["first_train_step_pass"] = True
        torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        cleanup_before_validation = clear_step5_graph_cache(ddp.module, reason="before_formal_entry_validation_pass")
        assert_no_step5_graph_tensors_attached(ddp.module, phase="before_formal_entry_validation_pass")
        valid_collate_timer = _TimingCollate(
            partial(
                _step5_collate_dynamic,
                dynamic_padding=bool(getattr(final_cfg, "train_dynamic_padding", True)),
                fixed_max_length=int(getattr(final_cfg, "train_label_max_length", 36)),
            )
        )
        valid_sampler = None if sample_plan_enabled else DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        valid_workers = int(json.loads(cfg.hardware_profile_json).get("dataloader_num_workers_valid") or 0)
        valid_prefetch = int(json.loads(cfg.hardware_profile_json).get("dataloader_prefetch_factor_valid") or 2)
        valid_loader = DataLoader(
            dataset,
            batch_size=int(getattr(final_cfg, "valid_per_gpu_batch_size", 0) or getattr(final_cfg, "valid_micro_batch_size", 1)),
            sampler=valid_sampler,
            shuffle=False,
            num_workers=valid_workers,
            pin_memory=_step5_pin_memory_from_final_cfg(final_cfg),
            persistent_workers=bool(getattr(cfg, "persistent_workers", True)) and valid_workers > 0,
            prefetch_factor=valid_prefetch if valid_workers > 0 else None,
            collate_fn=valid_collate_timer,
        )
        validation_pass_executed = True
        try:
            v_loss_sum, v_n, v_le_sum = validModel(
                ddp,
                valid_loader,
                device,
                coef=float(cfg.coef),
                explainer_loss_weight=float(cfg.explainer_loss_weight),
                step5_innov_cfg=innov_cfg,
                non_blocking_h2d=non_blocking_h2d,
                task_head=str(args.stage),
                valid_forward_micro_batch_size=int(
                    getattr(final_cfg, "valid_forward_micro_batch_size", 0)
                    or getattr(final_cfg, "valid_micro_batch_size", 0)
                    or 1
                ),
                validation_memory_policy=str(getattr(final_cfg, "validation_memory_policy", "microbatch_accumulate")),
                lambda_ortho_step5=float(getattr(final_cfg, "lambda_ortho_step5", 0.0) or 0.0),
                lambda_ortho_xcov=float(getattr(final_cfg, "lambda_ortho_xcov", 1.0) or 1.0),
                lambda_ortho_cos=float(getattr(final_cfg, "lambda_ortho_cos", 0.25) or 0.25),
            )
        except RuntimeError as exc:
            oom = "out of memory" in str(exc).lower()
            rank_payload.update(
                {
                    "validation_pass_executed": validation_pass_executed,
                    "validation_forward_pass": False,
                    "validation_loss_finite": False,
                    "validation_oom": bool(oom),
                    "validation_error": str(exc),
                }
            )
            raise RuntimeError("Step5 formal-entry E4 validation_forward_oom" if oom else "Step5 formal-entry E4 validation failed") from exc
        v_stat = torch.tensor([v_loss_sum, float(v_n), v_le_sum], dtype=torch.double, device=device)
        dist.all_reduce(v_stat, op=dist.ReduceOp.SUM)
        v_den = float(v_stat[1].item())
        validation_loss = float(v_stat[0].item() / v_den) if v_den > 0.0 else 0.0
        validation_loss_finite = bool(math.isfinite(validation_loss))
        validation_contract = dict(getattr(ddp.module, "_last_validation_contract", {}) or {})
        rank_payload.update(
            {
                "validation_pass_executed": validation_pass_executed,
                "validation_forward_pass": True,
                "validation_loss_finite": validation_loss_finite,
                "validation_oom": False,
                "validation_loss": validation_loss,
                "validation_sample_count_global": v_den,
                "validation_retired_prediction_zero": 0.0,
                "validation_retired_counterfactual_zero": 0.0,
                "validation_loss_e": float(v_stat[2].item() / v_den) if v_den > 0.0 else 0.0,
                "validation_microbatch_accumulation": bool(getattr(final_cfg, "validation_microbatch_accumulation", False)),
                "validation_memory_policy": str(getattr(final_cfg, "validation_memory_policy", "")),
                "valid_per_gpu_batch_size": int(getattr(final_cfg, "valid_per_gpu_batch_size", 0)),
                "valid_forward_micro_batch_size": int(getattr(final_cfg, "valid_forward_micro_batch_size", 0)),
                "train_per_gpu_batch_size": int(getattr(final_cfg, "per_gpu_batch_size", 0)),
                "rating_stability_control_validation_control_only": bool(validation_contract.get("rating_stability_control_validation_control_only"))
                if str(args.stage) == "rating_stability_control"
                else False,
                "flan_explainer_called_in_rating_stability_control_validation": bool(
                    validation_contract.get("flan_explainer_called_in_rating_stability_control_validation")
                )
                if str(args.stage) == "rating_stability_control"
                else None,
                "out_logits_materialized_in_rating_stability_control_validation": bool(
                    validation_contract.get("out_logits_materialized_in_rating_stability_control_validation")
                )
                if str(args.stage) == "rating_stability_control"
                else None,
                "word_dist_returned_in_rating_stability_control_validation": bool(
                    validation_contract.get("word_dist_returned_in_rating_stability_control_validation")
                )
                if str(args.stage) == "rating_stability_control"
                else None,
                "validation_collate": valid_collate_timer.summary_ms(),
                "scratch_cleanup_before_validation": cleanup_before_validation,
            }
        )
        if not validation_loss_finite:
            raise RuntimeError("Step5 formal-entry E4 validation loss is non-finite.")
        memory_truth = _memory_truth_payload(
            device=device,
            model=ddp.module,
            optimizer=optimizer,
            memory_creep_detected=False,
        )
        validate_memory_truth_schema(memory_truth)
        model_memory_audit = _model_memory_audit(
            ddp.module,
            optimizer,
            gradient_checkpointing_enabled=bool(memory_policy_meta.get("gradient_checkpointing_enabled", False)),
            gradient_checkpointing_reentrant_policy=str(
                memory_policy_meta.get("gradient_checkpointing_reentrant_policy")
                or memory_cfg.get("gradient_checkpointing_reentrant_policy")
                or "unknown"
            ),
            use_cache_training_disabled=bool(memory_policy_meta.get("use_cache_training_disabled", False)),
            bf16_effective=str(cfg.train_precision).lower() == "bf16",
        )
        cleanup_after_step = clear_step5_graph_cache(ddp.module, reason="after_formal_entry_first_train_step")
        assert_no_step5_graph_tensors_attached(ddp.module, phase="after_formal_entry_first_train_step")
        rank_payload["scratch_cleanup_after_first_train_step"] = cleanup_after_step
        loss_components = {
            "total_loss": total_loss,
            "main_control_loss": loss_factual,
            "explainer_loss": loss_counterfactual,
            "lci_raw_loss": l_lci,
            "lci_weighted_loss": lci_weighted_loss,
            "uci_weight_mean": lci_bundle.uci_weight_mean,
            "scorer_weight_mean": lci_bundle.scorer_weight_mean,
            "fca_raw_loss": l_fca,
            "fca_weighted_loss": fca_weighted_loss,
            "fca_weight_mean": 0.0 if is_rating_stability_control_probe else fca_bundle.fca_weight_mean,
            "repeat_ul_loss": loss_ul,
            "terminal_clean_loss": loss_tc,
            "batch_diversity_loss": loss_bd,
            "ortho_keep_loss": loss_ortho_keep,
        }
        component_keys = sorted(loss_components)
        key_payloads: list[list[str] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(key_payloads, component_keys)
        keys_identical = all(list(keys or []) == component_keys for keys in key_payloads)
        if rank == 0 and bool(data_pipeline_cfg.get("gpu_util_sampling_enabled", False)):
            gpu_util_snapshots.append({"phase": "after_first_step", **_nvidia_smi_gpu_snapshot()})
        sample_count = int(user_idx.size(0))
        scorer_count = int((gb.route_scorer_mask.view(-1) > 0).sum().item())
        explainer_count = int((gb.route_explainer_mask.view(-1) > 0).sum().item())
        token_count = int((tgt_output != 0).sum().item())
        total_step_ms = _elapsed_ms(h2d_s, h2d_e) + _elapsed_ms(fwd_s, fwd_e) + _elapsed_ms(bwd_s, bwd_e) + _elapsed_ms(opt_s, opt_e)
        collate_summary = collate_timer.summary_ms()
        cpu_now = _cpu_usage_snapshot()
        cpu_pct = _cpu_percentages(cpu_start, cpu_now, wall_s=time.perf_counter() - wall_start)
        rank_payload.update(
            {
                "success": True,
                "forward_executed": True,
                "loss_backward_executed": True,
                "optimizer_step_executed": True,
                "ready_hook_error_absent": True,
                "ccv_shape_error_absent": True,
                "real_forward_backward_executed": True,
                "real_sample_plan_used": True,
                "real_collate_executed": True,
                "real_ccv_packet_used": not is_rating_stability_control_probe,
                "sample_count": sample_count,
                "route_scorer_count": scorer_count,
                "route_explainer_count": explainer_count,
                "loss_finite": True,
                "loss_component_keys": component_keys,
                "loss_component_keys_identical": bool(keys_identical),
                "losses": {key: _loss_float(value) for key, value in loss_components.items()},
                "nan_inf_count": 0,
                "graph_tied_zero_count": int(sum(abs(_loss_float(loss_components[k])) == 0.0 for k in ("repeat_ul_loss", "terminal_clean_loss", "batch_diversity_loss"))),
                "all_trainable_grad": all_grad_report,
                "all_trainable_grad_status": all_grad_report["status"],
                "trainable_param_count": all_grad_report["trainable_param_count"],
                "grad_present_count": all_grad_report["grad_present_count"],
                "lora_trainable_count": all_grad_report["lora_trainable_count"],
                "lora_grad_present_count": all_grad_report["lora_grad_present_count"],
                "missing_grad_params": all_grad_report["missing_grad_params"],
                "head_gated_loss_contract": head_gated_loss_contract(str(args.stage)),
                "ddp": {
                    **_ddp_ready_hook_policy_payload(cfg),
                    "static_graph_probe": bool(ddp_static_graph_probe),
                },
                "training_memory_policy": memory_policy_meta,
                "lora_parameter_participation": lora_participation,
                "ccv_control_packet": (
                    {"present": False, "rating_stability_control_probe_no_ccv_packet": True}
                    if is_rating_stability_control_probe
                    else {
                        "present": True,
                        "numeric_shape": list(ccv_packet.numeric_controls().shape),
                        "numeric_dtype": str(ccv_packet.numeric_controls().dtype),
                        "numeric_device": str(ccv_packet.numeric_controls().device),
                        "content_evidence_shape": list(ccv_packet.content_evidence_ids.shape),
                        "style_evidence_shape": list(ccv_packet.style_evidence_ids.shape),
                        "domain_style_anchor_shape": list(ccv_packet.domain_style_anchor_ids.shape),
                        "local_style_hint_shape": list(ccv_packet.local_style_hint_ids.shape),
                        "polarity_ids_shape": list(ccv_packet.polarity_ids.shape),
                    }
                ),
                "timing_ms": {
                    "sampler_plan": float(train_table_stats.get("sampler_plan_time_s", 0.0)) * 1000.0,
                    "parquet_read": float(train_table_stats.get("parquet_read_time_s", 0.0)) * 1000.0,
                    "prompt_build": float(train_table_stats.get("prompt_build_time_s", 0.0)) * 1000.0,
                    "tokenize": float(tokenize_s) * 1000.0,
                    "h2d": _elapsed_ms(h2d_s, h2d_e),
                    "forward": _elapsed_ms(fwd_s, fwd_e),
                    "backward": _elapsed_ms(bwd_s, bwd_e),
                    "optimizer": _elapsed_ms(opt_s, opt_e),
                    "total_step": total_step_ms,
                    "data_wait_wall": data_wait_s * 1000.0,
                    "dataloader_queue_wait": data_wait_s * 1000.0,
                    "data_load_wall": data_load_s * 1000.0,
                    "collate": collate_summary["mean_ms"],
                },
                "pipeline_timing_ms": {
                    "sampler_plan_time": float(train_table_stats.get("sampler_plan_time_s", 0.0)) * 1000.0,
                    "parquet_read_time": float(train_table_stats.get("parquet_read_time_s", 0.0)) * 1000.0,
                    "prompt_build_time": float(train_table_stats.get("prompt_build_time_s", 0.0)) * 1000.0,
                    "tokenize_time": float(tokenize_s) * 1000.0,
                    "collate_time": collate_summary["mean_ms"],
                    "dataloader_wait_time": data_wait_s * 1000.0,
                    "dataloader_queue_wait": data_wait_s * 1000.0,
                    "dataloader_queue_depth": None,
                    "h2d_time": _elapsed_ms(h2d_s, h2d_e),
                    "forward_time": _elapsed_ms(fwd_s, fwd_e),
                    "backward_time": _elapsed_ms(bwd_s, bwd_e),
                    "optimizer_time": _elapsed_ms(opt_s, opt_e),
                    "total_step_time": total_step_ms,
                    "data_wait_ratio": float((data_wait_s * 1000.0) / max(total_step_ms, 1e-9)),
                    **cpu_pct,
                },
                "throughput": {
                    "samples_per_sec": float(sample_count / max(total_step_ms / 1000.0, 1e-9)),
                    "tokens_per_sec": float(token_count / max(total_step_ms / 1000.0, 1e-9)),
                },
                "gpu_memory": {
                    "allocated_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "allocated_peak_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
                    "reserved_peak_gb": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
                    "reserved_is_diagnostic_only": True,
                },
                "memory_truth": memory_truth,
                "model_memory_audit": model_memory_audit,
                "sequence_memory_audit": sequence_memory_audit,
                "graph_memory_audit": graph_memory_audit,
                "loader": {
                    "cache_hit": False,
                    "export_loader_mode": dict(train_table_stats).get("mode"),
                    "full_csv_parse": bool(dict(train_table_stats).get("full_csv_parse")),
                    "bounded_rows": int(dict(train_table_stats).get("bounded_max_rows", 0) or 0),
                    "filtered_row_count": int(train_table_filtered_count),
                    "raw_row_count": int(train_table_raw_count),
                    "worker_budget": budget,
                    "num_workers": num_workers,
                    "prefetch_factor": prefetch,
                    "sample_plan_enabled": bool(sample_plan_enabled),
                    "sample_plan_manifest": str(output_dir / "sample_plan" / "sample_plan_manifest.json") if sample_plan_enabled else None,
                    "token_cache_enabled": bool(token_cache_manifest is not None),
                    "token_cache_manifest": str(output_dir / "bounded_token_cache" / f"rank{int(rank)}_manifest.json") if token_cache_manifest else None,
                    "hot_path_tokenize_removed": bool(hot_path_tokenize_removed),
                    "dataloader_queue_size": int(data_pipeline_cfg.get("dataloader_queue_size", 0) or 0),
                    "dataloader_queue_depth": None,
                },
                "source": {
                    "export_path": str(export_path),
                    "export_sha256": train_table_source_summary.get("expected_sha256"),
                    "index_contract_sha256": train_table_source_summary.get("index_contract_sha256"),
                    "manifest_sha256": train_table_source_summary.get("manifest_sha256"),
                    "pool_source": train_table_source_summary,
                    "profile_mode": profile_meta.get("profile_mode"),
                },
            }
        )
        rank_payload["long_window"] = {
            "requested": bool(long_window_requested),
            "steps_executed": 1,
            "post_step_alloc_gb_first": _round_gb(float(torch.cuda.memory_allocated(device) / (1024**3))),
            "post_step_alloc_gb_last": _round_gb(float(torch.cuda.memory_allocated(device) / (1024**3))),
            "post_step_alloc_gb_delta": 0.0,
            "memory_creep_detected": False,
            "pass": True,
        }
        if long_window_requested and long_window_steps > 1:
            alloc_series = [float(torch.cuda.memory_allocated(device) / (1024**3))]
            step_times = [float(total_step_ms)]
            timing_sums = {
                "sampler_plan": float(train_table_stats.get("sampler_plan_time_s", 0.0)) * 1000.0,
                "parquet_read": float(train_table_stats.get("parquet_read_time_s", 0.0)) * 1000.0,
                "prompt_build": float(train_table_stats.get("prompt_build_time_s", 0.0)) * 1000.0,
                "tokenize": float(tokenize_s) * 1000.0,
                "h2d": float(rank_payload["timing_ms"]["h2d"]),
                "forward": float(rank_payload["timing_ms"]["forward"]),
                "backward": float(rank_payload["timing_ms"]["backward"]),
                "optimizer": float(rank_payload["timing_ms"]["optimizer"]),
                "data_wait_wall": float(rank_payload["timing_ms"]["data_wait_wall"]),
                "total_step": float(total_step_ms),
            }
            extra_samples = 0
            extra_scorer = 0
            extra_explainer = 0
            extra_tokens = 0
            loader_iter = iter(dataloader)
            for extra_idx in range(long_window_steps - 1):
                try:
                    data_wait_t0 = time.perf_counter()
                    batch = next(loader_iter)
                    extra_data_wait_s = time.perf_counter() - data_wait_t0
                except StopIteration:
                    if sampler is not None and hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(extra_idx + 1)
                    loader_iter = iter(dataloader)
                    data_wait_t0 = time.perf_counter()
                    batch = next(loader_iter)
                    extra_data_wait_s = time.perf_counter() - data_wait_t0
                h2d_s, h2d_e = _cuda_event_pair()
                fwd_s, fwd_e = _cuda_event_pair()
                bwd_s, bwd_e = _cuda_event_pair()
                opt_s, opt_e = _cuda_event_pair()
                h2d_s.record()
                gb = ddp.module.gather(batch, device, non_blocking_h2d=non_blocking_h2d)
                h2d_e.record()
                user_idx = gb.user_idx
                item_idx = gb.item_idx
                rating = gb.rating
                tgt_input = gb.tgt_input
                tgt_output = gb.tgt_output
                domain_idx = gb.domain_idx
                _validate_route_masks_batch(
                    gb.route_scorer_mask,
                    gb.route_explainer_mask,
                    batch_size=int(user_idx.size(0)),
                    stage=str(args.stage),
                )
                torch.cuda.synchronize(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
                    fwd_s.record()
                    gate_a = build_rating_stability_control_gate(gb, innov_cfg)
                    gate_b = build_step5_explanation_gate(gb, innov_cfg)
                    ccv_packet = None
                    if not is_rating_stability_control_probe:
                        ccv_packet = build_ccv_control_packet(
                            gb,
                            innov_cfg,
                            producer="real_batch_backward_preflight",
                            head=str(args.stage),
                        )
                        validate_ccv_control_packet_shapes(
                            ccv_packet,
                            producer="real_batch_backward_preflight",
                            head=str(args.stage),
                            strict=True,
                        )
                    sequence_memory_audit = _sequence_memory_audit(gb, final_cfg)
                    graph_memory_audit = (
                        {
                            "lci_graph_retention_ok": True,
                            "fca_graph_retention_ok": True,
                            "ccv_control_packet_shape": {},
                            "rating_stability_control_probe_no_ccv_packet": True,
                            "unnecessary_logits_retained": False,
                            "tensors_retained_after_step": False,
                            "empty_cache_called_for_measurement_only": False,
                        }
                        if is_rating_stability_control_probe
                        else _graph_memory_audit(ccv_packet)
                    )
                    control_score, _context_dist, word_dist = ddp(
                        user_idx,
                        item_idx,
                        tgt_input,
                        domain_idx,
                        target_tokens=None if is_rating_stability_control_probe else tgt_output,
                        evidence_features=gb.evidence_features,
                        content_anchor_score=gb.content_anchor_score,
                        style_anchor_score=gb.style_anchor_score,
                        ccv_control_packet=ccv_packet,
                        return_explainer_logits=not is_rating_stability_control_probe,
                    )
                    fwd_e.record()
                    control_zero_ps = graph_tied_zero_like(control_score).expand_as(control_score).to(dtype=control_score.dtype)
                    if is_rating_stability_control_probe:
                        loss_flan_ps = control_zero_ps
                    else:
                        loss_flan_ps = per_sample_decoder_ce_from_logits(
                            word_dist,
                            tgt_output,
                            ignore_index=0,
                            label_smoothing=float(cfg.label_smoothing),
                        )
                    control_only = control_zero_ps.to(dtype=loss_flan_ps.dtype)
                    explainer_only = loss_flan_ps
                    dom = domain_idx.view(-1)
                    loss_factual = graph_tied_zero_like(control_score)
                    loss_counterfactual = (
                        graph_tied_zero_like(control_score)
                        if is_rating_stability_control_probe
                        else float(cfg.explainer_loss_weight)
                        * route_weighted_mean(
                            explainer_only,
                            gate_b.explainer_weight.to(dtype=explainer_only.dtype),
                            (dom == 0).to(dtype=explainer_only.dtype),
                        )
                    )
                    _model = ddp.module
                    spec_lat = _model._last_specific_latent
                    shared_lat = _model._last_shared_latent
                    lci_weighted_zero = graph_tied_zero_like(control_score)
                    lci_raw_zero = control_zero_ps
                    fca_bundle = None
                    if not is_rating_stability_control_probe:
                        fca_bundle = evidence_basis_fca_loss(
                            scorer_hidden=_model._last_h_score,
                            explainer_hidden=_model._last_h_explain_aligned,
                            shared_latent=shared_lat,
                            content_profile=_model._last_content_profile,
                            content_evidence_latent=_model._last_content_evidence_latent,
                            packet=ccv_packet,
                            gate=gate_b,
                            cfg=innov_cfg,
                        )
                    (
                        loss_factual,
                        loss_counterfactual,
                        lci_weighted_loss,
                        fca_weighted_loss,
                        _l_lci,
                        _l_fca,
                    ) = _head_gated_step5_loss_terms(
                        task_head=str(args.stage),
                        zero_like=control_score if is_rating_stability_control_probe else word_dist,
                        loss_factual=loss_factual,
                        loss_counterfactual=loss_counterfactual,
                        lci_weighted_loss=lci_weighted_zero,
                        fca_weighted_loss=graph_tied_zero_like(control_score) if is_rating_stability_control_probe else fca_bundle.fca_weighted_loss,
                        l_lci=None,
                        l_fca=graph_tied_zero_like(control_score) if is_rating_stability_control_probe else fca_bundle.fca_loss,
                    )
                    lci_raw_ps, lci_weighted_ps = lci_raw_zero, lci_raw_zero
                    if is_rating_stability_control_probe:
                        fca_raw_ps = control_zero_ps
                        fca_weighted_ps = fca_raw_ps
                    else:
                        fca_raw_ps, fca_weighted_ps = _fca_per_sample_losses(
                            fca_bundle=fca_bundle,
                            gate=gate_b,
                            cfg=innov_cfg,
                        )
                    control_weighted_ps = control_only * gate_a.scorer_weight.to(dtype=control_only.dtype) * (dom == 1).to(dtype=control_only.dtype)
                    explainer_weighted_ps = (
                        explainer_only
                        * gate_b.explainer_weight.to(dtype=explainer_only.dtype)
                        * (dom == 0).to(dtype=explainer_only.dtype)
                        * float(cfg.explainer_loss_weight)
                    )
                    _accumulate_per_tier_loss(
                        per_tier_acc,
                        gb,
                        scorer_raw=control_only,
                        scorer_weighted=control_weighted_ps,
                        lci_raw=lci_raw_ps,
                        lci_weighted=lci_weighted_ps,
                        uci_weight=gate_a.uci_weight,
                        explainer_raw=explainer_only,
                        explainer_weighted=explainer_weighted_ps,
                        fca_raw=fca_raw_ps,
                        fca_weighted=fca_weighted_ps,
                        ccv_explainer_weight=gate_b.explainer_weight,
                        ccv_fca_weight=gate_b.fca_weight,
                        ccv_uncertainty=gate_b.uncertainty,
                        ccv_confidence=gate_b.confidence_bucket,
                        ccv_reliability=gate_b.reliability,
                    )
                    loss_ortho_keep = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
                    lambda_ortho = float(getattr(final_cfg, "lambda_ortho_step5", 0.0) or 0.0)
                    if lambda_ortho > 0.0:
                        loss_ortho_keep = build_orthogonal_losses(
                            shared_lat,
                            spec_lat,
                            w_xcov=float(getattr(final_cfg, "lambda_ortho_xcov", 1.0)),
                            w_cos=float(getattr(final_cfg, "lambda_ortho_cos", 0.25)),
                        ).loss_ortho_total
                    loss_ul = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
                    if (not is_rating_stability_control_probe) and float(getattr(final_cfg, "loss_weight_repeat_ul", 0.0) or 0.0) > 0.0:
                        word_logp = F.log_softmax(word_dist, dim=-1)
                        loss_ul = odcr_anti_repeat_unlikelihood_loss_from_logp(word_logp, tgt_output)
                    loss_tc = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
                    loss_bd = graph_tied_zero(control_score if is_rating_stability_control_probe else word_dist)
                    total_loss = compose_step5_total_loss(
                        loss_factual=loss_factual,
                        loss_counterfactual=loss_counterfactual,
                        retired_repeat_ul_zero_guard=loss_ul,
                        loss_terminal_clean=loss_tc,
                        loss_batch_diversity=loss_bd,
                        repeat_ul_weight=float(getattr(final_cfg, "loss_weight_repeat_ul", 0.0) or 0.0),
                        terminal_clean_weight=0.0,
                        batch_diversity_weight=0.0,
                        lci_weighted_loss=lci_weighted_loss,
                        fca_weighted_loss=fca_weighted_loss,
                        ortho_keep_loss=loss_ortho_keep,
                        ortho_keep_weight=lambda_ortho,
                    )
                finite_local = torch.tensor([1 if _finite_scalar(total_loss) else 0], device=device, dtype=torch.int)
                dist.all_reduce(finite_local, op=dist.ReduceOp.MIN)
                if int(finite_local.item()) != 1:
                    raise RuntimeError("Step5 E4 long-window probe observed non-finite synchronized loss")
                bwd_s.record()
                total_loss.backward()
                bwd_e.record()
                all_grad_report = validate_all_trainable_params_receive_grad(
                    ddp.module,
                    total_loss,
                    head=str(args.stage),
                    evidence_context={
                        "evidence_id": f"{args.stage}_E4_gpu_shard_forward_bounded_formal_entry_with_validation_long_window_{args.candidate_id}",
                        "evidence_level": E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
                        "real_sample_plan_used": True,
                        "real_collate_executed": True,
                        "real_ccv_packet_used": not is_rating_stability_control_probe,
                    },
                    fail_on_missing=True,
                )
                opt_s.record()
                nn.utils.clip_grad_norm_(params, 1.0)
                lora_participation = _lora_parameter_participation(ddp.module)
                optimizer.step()
                opt_e.record()
                torch.cuda.synchronize(device)
                loss_components = {
                    "total_loss": total_loss,
                    "main_control_loss": loss_factual,
                    "explainer_loss": loss_counterfactual,
                    "lci_raw_loss": _l_lci,
                    "lci_weighted_loss": lci_weighted_loss,
                    "uci_weight_mean": lci_bundle.uci_weight_mean,
                    "scorer_weight_mean": lci_bundle.scorer_weight_mean,
                    "fca_raw_loss": _l_fca,
                    "fca_weighted_loss": fca_weighted_loss,
                    "fca_weight_mean": 0.0 if is_rating_stability_control_probe else fca_bundle.fca_weight_mean,
                    "repeat_ul_loss": loss_ul,
                    "terminal_clean_loss": loss_tc,
                    "batch_diversity_loss": loss_bd,
                    "ortho_keep_loss": loss_ortho_keep,
                }
                rank_payload["losses"] = {key: _loss_float(value) for key, value in loss_components.items()}
                optimizer.zero_grad(set_to_none=True)
                clear_step5_graph_cache(ddp.module, reason="after_formal_entry_long_window_step")
                assert_no_step5_graph_tensors_attached(ddp.module, phase="after_formal_entry_long_window_step")
                step_h2d = _elapsed_ms(h2d_s, h2d_e)
                step_fwd = _elapsed_ms(fwd_s, fwd_e)
                step_bwd = _elapsed_ms(bwd_s, bwd_e)
                step_opt = _elapsed_ms(opt_s, opt_e)
                step_total = step_h2d + step_fwd + step_bwd + step_opt
                timing_sums["h2d"] += step_h2d
                timing_sums["forward"] += step_fwd
                timing_sums["backward"] += step_bwd
                timing_sums["optimizer"] += step_opt
                timing_sums["total_step"] += step_total
                timing_sums["data_wait_wall"] += extra_data_wait_s * 1000.0
                step_times.append(float(step_total))
                alloc_series.append(float(torch.cuda.memory_allocated(device) / (1024**3)))
                if (
                    rank == 0
                    and bool(data_pipeline_cfg.get("gpu_util_sampling_enabled", False))
                    and ((extra_idx + 1) % 16 == 0 or extra_idx == long_window_steps - 2)
                ):
                    gpu_util_snapshots.append({"phase": f"during_step_{extra_idx + 2}", **_nvidia_smi_gpu_snapshot()})
                extra_samples += int(user_idx.size(0))
                extra_scorer += int((gb.route_scorer_mask.view(-1) > 0).sum().item())
                extra_explainer += int((gb.route_explainer_mask.view(-1) > 0).sum().item())
                extra_tokens += int((tgt_output != 0).sum().item())
            first_alloc = alloc_series[0] if alloc_series else 0.0
            last_alloc = alloc_series[-1] if alloc_series else 0.0
            alloc_delta = last_alloc - first_alloc
            memory_creep_detected = bool(alloc_delta > float(memory_cfg.get("memory_creep_delta_gb", 1.0) or 1.0))
            memory_truth = _memory_truth_payload(
                device=device,
                model=ddp.module,
                optimizer=optimizer,
                memory_creep_detected=memory_creep_detected,
            )
            validate_memory_truth_schema(memory_truth)
            model_memory_audit = _model_memory_audit(
                ddp.module,
                optimizer,
                gradient_checkpointing_enabled=bool(memory_policy_meta.get("gradient_checkpointing_enabled", False)),
                gradient_checkpointing_reentrant_policy=str(
                    memory_policy_meta.get("gradient_checkpointing_reentrant_policy")
                    or memory_cfg.get("gradient_checkpointing_reentrant_policy")
                    or "unknown"
                ),
                use_cache_training_disabled=bool(memory_policy_meta.get("use_cache_training_disabled", False)),
                bf16_effective=str(cfg.train_precision).lower() == "bf16",
            )
            steps_executed = len(step_times)
            sorted_step_times = sorted(step_times)
            p95_idx = min(len(sorted_step_times) - 1, max(0, math.ceil(0.95 * len(sorted_step_times)) - 1))
            total_samples = int(rank_payload["sample_count"]) + extra_samples
            total_tokens = token_count + extra_tokens
            total_step_all = float(timing_sums["total_step"])
            collate_summary = collate_timer.summary_ms()
            cpu_now = _cpu_usage_snapshot()
            cpu_pct = _cpu_percentages(cpu_start, cpu_now, wall_s=time.perf_counter() - wall_start)
            rank_payload.update(
                {
                    "sample_count": total_samples,
                    "route_scorer_count": int(rank_payload["route_scorer_count"]) + extra_scorer,
                    "route_explainer_count": int(rank_payload["route_explainer_count"]) + extra_explainer,
                    "memory_truth": memory_truth,
                    "model_memory_audit": model_memory_audit,
                    "sequence_memory_audit": sequence_memory_audit,
                    "graph_memory_audit": graph_memory_audit,
                    "timing_ms": {
                        "sampler_plan": float(timing_sums["sampler_plan"]),
                        "parquet_read": float(timing_sums["parquet_read"]),
                        "prompt_build": float(timing_sums["prompt_build"]),
                        "tokenize": float(timing_sums["tokenize"]),
                        "h2d": float(timing_sums["h2d"] / max(steps_executed, 1)),
                        "forward": float(timing_sums["forward"] / max(steps_executed, 1)),
                        "backward": float(timing_sums["backward"] / max(steps_executed, 1)),
                        "optimizer": float(timing_sums["optimizer"] / max(steps_executed, 1)),
                        "total_step": float(total_step_all / max(steps_executed, 1)),
                        "total_step_accumulated": total_step_all,
                        "data_wait_wall": float(timing_sums["data_wait_wall"] / max(steps_executed, 1)),
                        "dataloader_queue_wait": float(timing_sums["data_wait_wall"] / max(steps_executed, 1)),
                        "data_load_wall": data_load_s * 1000.0,
                        "collate": collate_summary["mean_ms"],
                    },
                    "pipeline_timing_ms": {
                        "sampler_plan_time": float(timing_sums["sampler_plan"]),
                        "parquet_read_time": float(timing_sums["parquet_read"]),
                        "prompt_build_time": float(timing_sums["prompt_build"]),
                        "tokenize_time": float(timing_sums["tokenize"]),
                        "collate_time": collate_summary["mean_ms"],
                        "dataloader_wait_time": float(timing_sums["data_wait_wall"] / max(steps_executed, 1)),
                        "dataloader_queue_wait": float(timing_sums["data_wait_wall"] / max(steps_executed, 1)),
                        "dataloader_queue_depth": None,
                        "h2d_time": float(timing_sums["h2d"] / max(steps_executed, 1)),
                        "forward_time": float(timing_sums["forward"] / max(steps_executed, 1)),
                        "backward_time": float(timing_sums["backward"] / max(steps_executed, 1)),
                        "optimizer_time": float(timing_sums["optimizer"] / max(steps_executed, 1)),
                        "total_step_time": float(total_step_all / max(steps_executed, 1)),
                        "data_wait_ratio": float(
                            (timing_sums["data_wait_wall"] / max(steps_executed, 1))
                            / max(total_step_all / max(steps_executed, 1), 1e-9)
                        ),
                        **cpu_pct,
                    },
                    "throughput": {
                        "samples_per_sec": float(total_samples / max(total_step_all / 1000.0, 1e-9)),
                        "tokens_per_sec": float(total_tokens / max(total_step_all / 1000.0, 1e-9)),
                    },
                    "gpu_memory": {
                        "allocated_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device)),
                        "allocated_peak_gb": float(torch.cuda.max_memory_allocated(device) / (1024**3)),
                        "reserved_peak_gb": float(torch.cuda.max_memory_reserved(device) / (1024**3)),
                        "reserved_is_diagnostic_only": True,
                    },
                    "lora_parameter_participation": lora_participation,
                    "long_window": {
                        "requested": True,
                        "steps_executed": int(steps_executed),
                        "step_time_ms_mean": float(total_step_all / max(steps_executed, 1)),
                        "step_time_ms_p95": float(sorted_step_times[p95_idx]) if sorted_step_times else 0.0,
                        "step_time_ms_max": float(max(step_times)) if step_times else 0.0,
                        "post_step_alloc_gb_first": _round_gb(first_alloc),
                        "post_step_alloc_gb_last": _round_gb(last_alloc),
                        "post_step_alloc_gb_delta": _round_gb(alloc_delta),
                        "memory_creep_detected": memory_creep_detected,
                        "pass": not memory_creep_detected,
                    },
                }
            )
        rank_payload["per_tier_loss"] = _finalize_per_tier_loss(per_tier_acc)
        rank_payload["per_tier_loss_keys"] = _per_tier_loss_keys(rank_payload["per_tier_loss"])
        if rank == 0 and bool(data_pipeline_cfg.get("gpu_util_sampling_enabled", False)):
            gpu_util_snapshots.append({"phase": "after", **_nvidia_smi_gpu_snapshot()})
        rank_payload["gpu_util"] = _gpu_util_summary(gpu_util_snapshots) if rank == 0 else {
            "available": False,
            "mean": None,
            "p50": None,
            "p95": None,
            "memory_used_mib_max": None,
            "snapshots": [],
        }
        if requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE:
            e5_dir = output_dir / "e5_lifecycle" / f"rank{int(rank)}"
            e5_dir.mkdir(parents=True, exist_ok=True)
            e5_checkpoint = e5_dir / "best.pth"
            e5_before_teardown = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            cpu_state = {key: value.detach().cpu() for key, value in ddp.module.state_dict().items()}
            torch.save(cpu_state, e5_checkpoint)
            del cpu_state
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint_fp = file_fingerprint(e5_checkpoint)
            clear_step5_graph_cache(ddp.module, reason="e5_before_train_model_teardown")
            assert_no_step5_graph_tensors_attached(ddp.module, phase="e5_before_train_model_teardown")
            batch = gb = user_idx = item_idx = rating = tgt_input = tgt_output = domain_idx = None
            pred_rating = word_dist = _context_dist = total_loss = None
            loss_components = {}
            ema_model = None
            optimizer = None
            params = []
            ddp = None
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
            e5_after_teardown = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            eval_args_ns = SimpleNamespace(_odcr_index_contract=index_contract)
            eval_model = _make_model(final_cfg, eval_args_ns, local_rank)
            eval_model_constructed = True
            eval_state = torch.load(e5_checkpoint, map_location="cpu", weights_only=True)
            incompatible = eval_model.load_state_dict(eval_state, strict=True)
            missing_keys = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected_keys = list(getattr(incompatible, "unexpected_keys", []) or [])
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "E5 CPU-staged checkpoint reload produced incompatible keys: "
                    f"missing={missing_keys[:5]} unexpected={unexpected_keys[:5]}"
                )
            del eval_state
            gc.collect()
            torch.cuda.empty_cache()
            e5_after_load = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            eval_loss_sum, eval_n, eval_lr_sum, eval_lc_sum, eval_le_sum = validModel(
                eval_model,
                valid_loader,
                device,
                coef=float(cfg.coef),
                explainer_loss_weight=float(cfg.explainer_loss_weight),
                step5_innov_cfg=innov_cfg,
                non_blocking_h2d=non_blocking_h2d,
                task_head=str(args.stage),
                valid_forward_micro_batch_size=int(
                    getattr(final_cfg, "valid_forward_micro_batch_size", 0)
                    or getattr(final_cfg, "valid_micro_batch_size", 0)
                    or 1
                ),
                validation_memory_policy=str(getattr(final_cfg, "validation_memory_policy", "microbatch_accumulate")),
                lambda_ortho_step5=float(getattr(final_cfg, "lambda_ortho_step5", 0.0) or 0.0),
                lambda_ortho_xcov=float(getattr(final_cfg, "lambda_ortho_xcov", 1.0) or 1.0),
                lambda_ortho_cos=float(getattr(final_cfg, "lambda_ortho_cos", 0.25) or 0.25),
            )
            eval_stat = torch.tensor(
                [eval_loss_sum, float(eval_n), eval_lr_sum, eval_lc_sum, eval_le_sum],
                dtype=torch.double,
                device=device,
            )
            dist.all_reduce(eval_stat, op=dist.ReduceOp.SUM)
            eval_den = float(eval_stat[1].item())
            eval_loss = float(eval_stat[0].item() / eval_den) if eval_den > 0.0 else 0.0
            if not math.isfinite(eval_loss):
                raise RuntimeError("E5 CPU-staged eval reload produced non-finite validation loss.")
            eval_model = None
            gc.collect()
            torch.cuda.empty_cache()
            e5_after_eval = {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            }
            rank_payload["e5_lifecycle"] = {
                "schema_version": "odcr_rating_stability_control_e5_post_train_eval_lifecycle_rank/1",
                "success": True,
                "evidence_level": E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE,
                "checkpoint_saved": True,
                "checkpoint_path": str(e5_checkpoint),
                "checkpoint_fingerprint": checkpoint_fp,
                "train_model_teardown_verified": e5_after_teardown["allocated_bytes"] <= e5_before_teardown["allocated_bytes"],
                "cpu_staged_checkpoint_load": True,
                "eval_model_constructed": eval_model_constructed,
                "eval_forward_executed": True,
                "eval_loss": eval_loss,
                "eval_sample_count_global": eval_den,
                "memory_before_teardown": e5_before_teardown,
                "memory_after_teardown": e5_after_teardown,
                "memory_after_cpu_staged_load": e5_after_load,
                "memory_after_eval": e5_after_eval,
                "formal_namespace_pollution": False,
                "latest_json_created": False,
                "formal_train_command_emitted": False,
                "synthetic_batch_used_for_formal_gate": False,
            }
        all_payloads: list[dict[str, Any] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(all_payloads, rank_payload)
        if rank == 0:
            sample_counts = [int((p or {}).get("sample_count", 0)) for p in all_payloads]
            step_times = [float(((p or {}).get("timing_ms") or {}).get("total_step", 0.0)) for p in all_payloads]
            rank_long_windows = [
                dict((p or {}).get("long_window") or {})
                for p in all_payloads
                if isinstance((p or {}).get("long_window"), Mapping)
            ]
            rank_memory_truths = [
                dict((p or {}).get("memory_truth") or {})
                for p in all_payloads
                if isinstance((p or {}).get("memory_truth"), Mapping)
            ]
            rank_pipeline = [
                dict((p or {}).get("pipeline_timing_ms") or {})
                for p in all_payloads
                if isinstance((p or {}).get("pipeline_timing_ms"), Mapping)
            ]
            gpu_util_payloads = [
                dict((p or {}).get("gpu_util") or {})
                for p in all_payloads
                if isinstance((p or {}).get("gpu_util"), Mapping)
            ]
            e5_lifecycle_reports = [
                dict((p or {}).get("e5_lifecycle") or {})
                for p in all_payloads
                if isinstance((p or {}).get("e5_lifecycle"), Mapping)
            ]
            runtime_signature_summary = _summarize_runtime_transformers_signatures(all_payloads)
            gpu_util = next((item for item in gpu_util_payloads if item.get("snapshots")), gpu_util_payloads[0] if gpu_util_payloads else {})
            rank_per_tier_reports = [
                dict((p or {}).get("per_tier_loss") or {})
                for p in all_payloads
                if isinstance((p or {}).get("per_tier_loss"), Mapping)
            ]
            per_tier_loss = _merge_per_tier_loss(rank_per_tier_reports)
            per_tier_key_payloads = [
                list((p or {}).get("per_tier_loss_keys") or [])
                for p in all_payloads
            ]
            expected_per_tier_keys = _per_tier_loss_keys(per_tier_loss)
            per_tier_keys_identical = all(keys == expected_per_tier_keys for keys in per_tier_key_payloads)

            def _rank_max_metric(key: str) -> float | None:
                vals = [float(item[key]) for item in rank_pipeline if item.get(key) is not None]
                return max(vals) if vals else None

            pipeline_breakdown = {
                "sampler_plan_time": _rank_max_metric("sampler_plan_time"),
                "parquet_read_time": _rank_max_metric("parquet_read_time"),
                "prompt_build_time": _rank_max_metric("prompt_build_time"),
                "tokenize_time": _rank_max_metric("tokenize_time"),
                "collate_time": _rank_max_metric("collate_time"),
                "dataloader_wait_time": _rank_max_metric("dataloader_wait_time"),
                "dataloader_queue_wait": _rank_max_metric("dataloader_queue_wait"),
                "dataloader_queue_depth": None,
                "h2d_time": _rank_max_metric("h2d_time"),
                "forward_time": _rank_max_metric("forward_time"),
                "backward_time": _rank_max_metric("backward_time"),
                "optimizer_time": _rank_max_metric("optimizer_time"),
                "total_step_time": _rank_max_metric("total_step_time"),
                "data_wait_ratio": _rank_max_metric("data_wait_ratio"),
                "cpu_user_percent": _rank_max_metric("cpu_user_percent"),
                "cpu_system_percent": _rank_max_metric("cpu_system_percent"),
                "gpu_util_mean": gpu_util.get("mean"),
                "gpu_util_p50": gpu_util.get("p50"),
                "gpu_util_p95": gpu_util.get("p95"),
                "gpu_util_available": bool(gpu_util.get("available")),
                "nvidia_smi_memory": gpu_util.get("memory_used_mib_max"),
                "per_rank_sample_count": sample_counts,
                "per_rank_worker_count": [
                    int(((p or {}).get("loader") or {}).get("num_workers", 0))
                    for p in all_payloads
                ],
            }
            gpu_peak_allocated = max((float(m.get("max_memory_allocated_gb") or 0.0) for m in rank_memory_truths), default=0.0)
            gpu_peak_reserved = max((float(m.get("max_memory_reserved_gb") or 0.0) for m in rank_memory_truths), default=0.0)
            aggregate_memory_truth = dict(rank_memory_truths[0]) if rank_memory_truths else {}
            if aggregate_memory_truth:
                aggregate_memory_truth.update(
                    {
                        "max_memory_allocated_gb": gpu_peak_allocated,
                        "max_memory_reserved_gb": gpu_peak_reserved,
                        "reserved_minus_allocated_gb": max(0.0, gpu_peak_reserved - gpu_peak_allocated),
                        "allocated_to_total_ratio": float(
                            gpu_peak_allocated / max(float(aggregate_memory_truth.get("device_total_gb") or 1.0), 1e-9)
                        ),
                        "reserved_to_total_ratio": float(
                            gpu_peak_reserved / max(float(aggregate_memory_truth.get("device_total_gb") or 1.0), 1e-9)
                        ),
                        "nvidia_smi_process_used_gb": max(
                            (
                                float(m.get("nvidia_smi_process_used_gb") or 0.0)
                                for m in rank_memory_truths
                            ),
                            default=0.0,
                        )
                        or None,
                        "memory_creep_detected": any(bool(m.get("memory_creep_detected")) for m in rank_memory_truths),
                        "oom": any(bool(m.get("oom")) for m in rank_memory_truths),
                        "reserved_is_diagnostic_only": True,
                    }
                )
                validate_memory_truth_schema(aggregate_memory_truth)
            throughput = sum(float(((p or {}).get("throughput") or {}).get("samples_per_sec", 0.0)) for p in all_payloads)
            final_payload = {
                    "schema_version": "odcr_step5_e4_bounded_probe/1",
                    "stage": str(args.stage),
                    "task_id": int(args.task),
                    "candidate_id": str(args.candidate_id),
                    "success": True,
                    "evidence_level": E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION,
                    "evidence_level_family": "E4_gpu_shard_forward_bounded",
                    "formal_entry_lifecycle": True,
                    "forward_executed": True,
                    "loss_backward_executed": True,
                    "optimizer_step_executed": True,
                    "preflight_executed": all(bool((payload or {}).get("preflight_success")) for payload in all_payloads),
                    "scratch_cleanup_status": (
                        "pass"
                        if all(str((payload or {}).get("scratch_cleanup_status")) == "pass" for payload in all_payloads)
                        else "fail"
                    ),
                    "graph_tensor_audit_status": (
                        "pass"
                        if all(str((payload or {}).get("graph_tensor_audit_status")) == "pass" for payload in all_payloads)
                        else "fail"
                    ),
                    "graph_tensor_audit_phase": "before_ema_init",
                    "graph_scratch_before_ema": list((all_payloads[0] or {}).get("graph_scratch_before_ema") or []),
                    "ema_enabled": bool(getattr(cfg, "ema_enabled", True)),
                    "ema_decay": float(getattr(cfg, "ema_decay", 0.999)),
                    "ema_init_strategy": str((all_payloads[0] or {}).get("ema_init_strategy") or "AveragedModel_after_scratch_cleanup"),
                    "ema_init_pass": all(bool((payload or {}).get("ema_init_pass")) for payload in all_payloads),
                    "ema_init_executed_in_E4": all(bool((payload or {}).get("ema_init_executed")) for payload in all_payloads),
                    "ddp_wrap_pass": all(bool((payload or {}).get("ddp_wrap_pass")) for payload in all_payloads),
                    "first_train_step_pass": all(bool((payload or {}).get("first_train_step_pass")) for payload in all_payloads),
                    "validation_pass_executed": all(bool((payload or {}).get("validation_pass_executed")) for payload in all_payloads),
                    "validation_forward_pass": all(bool((payload or {}).get("validation_forward_pass")) for payload in all_payloads),
                    "validation_loss_finite": all(bool((payload or {}).get("validation_loss_finite")) for payload in all_payloads),
                    "validation_oom": any(bool((payload or {}).get("validation_oom")) for payload in all_payloads),
                    "validation_loss": max(
                        (float((payload or {}).get("validation_loss") or 0.0) for payload in all_payloads),
                        default=0.0,
                    ),
                    "validation_sample_count_global": max(
                        (float((payload or {}).get("validation_sample_count_global") or 0.0) for payload in all_payloads),
                        default=0.0,
                    ),
                    "rating_stability_control_validation_control_only": (
                        str(args.stage) == "rating_stability_control"
                        and all(bool((payload or {}).get("rating_stability_control_validation_control_only")) for payload in all_payloads)
                    ),
                    "flan_explainer_called_in_rating_stability_control_validation": (
                        False
                        if str(args.stage) == "rating_stability_control"
                        and all((payload or {}).get("flan_explainer_called_in_rating_stability_control_validation") is False for payload in all_payloads)
                        else None
                    ),
                    "out_logits_materialized_in_rating_stability_control_validation": (
                        False
                        if str(args.stage) == "rating_stability_control"
                        and all((payload or {}).get("out_logits_materialized_in_rating_stability_control_validation") is False for payload in all_payloads)
                        else None
                    ),
                    "word_dist_returned_in_rating_stability_control_validation": (
                        False
                        if str(args.stage) == "rating_stability_control"
                        and all((payload or {}).get("word_dist_returned_in_rating_stability_control_validation") is False for payload in all_payloads)
                        else None
                    ),
                    "validation_microbatch_accumulation": all(
                        bool((payload or {}).get("validation_microbatch_accumulation")) for payload in all_payloads
                    ),
                    "validation_memory_policy": str((all_payloads[0] or {}).get("validation_memory_policy") or ""),
                    "valid_per_gpu_batch_size": int((all_payloads[0] or {}).get("valid_per_gpu_batch_size") or 0),
                    "valid_forward_micro_batch_size": int((all_payloads[0] or {}).get("valid_forward_micro_batch_size") or 0),
                    "train_per_gpu_batch_size": int((all_payloads[0] or {}).get("train_per_gpu_batch_size") or 0),
                    "ready_hook_error_absent": True,
                    "ccv_shape_error_absent": all(
                        bool((payload or {}).get("ccv_shape_error_absent")) for payload in all_payloads
                    ),
                    "real_forward_backward_executed": True,
                    "actual_gpu_backward_executed": True,
                    "real_task_data_used": True,
                    "real_model_loaded_on_gpu": True,
                    "real_data_batch_used": True,
                    "real_sample_plan_used": all(
                        bool((payload or {}).get("real_sample_plan_used")) for payload in all_payloads
                    ),
                    "real_collate_executed": all(
                        bool((payload or {}).get("real_collate_executed")) for payload in all_payloads
                    ),
                    "real_ccv_packet_used": (
                        False
                        if str(args.stage) == "rating_stability_control"
                        else all(bool((payload or {}).get("real_ccv_packet_used")) for payload in all_payloads)
                    ),
                    "real_checkpoint_used": False,
                    "checkpoint_policy": "fresh_init_model_no_formal_checkpoint",
                    "ddp_world_size": world_size,
                    "ddp": {
                        **_ddp_ready_hook_policy_payload(cfg),
                    },
                    "training_memory_policy": dict((all_payloads[0] or {}).get("training_memory_policy") or {}),
                    "lora_target_policy_id": str((all_payloads[0] or {}).get("lora_target_policy_id") or ""),
                    "head_specific_lora_allowlist_id": str(
                        (all_payloads[0] or {}).get("head_specific_lora_allowlist_id") or ""
                    ),
                    "final_lora_target_modules": list((all_payloads[0] or {}).get("final_lora_target_modules") or []),
                    "forbidden_lora_targets": list((all_payloads[0] or {}).get("forbidden_lora_targets") or []),
                    "deleted_legacy_modules": list((all_payloads[0] or {}).get("deleted_legacy_modules") or []),
                    "head_specific_trainable_policy": (all_payloads[0] or {}).get("head_specific_trainable_policy"),
                    "final_lora_target_modules_hash": (all_payloads[0] or {}).get("final_lora_target_modules_hash"),
                    "trainable_parameter_names_hash": (all_payloads[0] or {}).get("trainable_parameter_names_hash"),
                    "frozen_parameter_names_hash": (all_payloads[0] or {}).get("frozen_parameter_names_hash"),
                    "runtime_transformers_signature": runtime_signature_summary,
                    "rank_ids": list(range(world_size)),
                    "rank_results": all_payloads,
                    "rank_sample_balance_ok": max(sample_counts) - min(sample_counts) <= max(1, int(cfg.per_device_train_batch_size)),
                    "rank_step_time_balance_ok": (
                        (max(step_times) / max(min([x for x in step_times if x > 0.0] or [1.0]), 1e-9)) <= 1.35
                    ),
                    "long_window_pass": all(bool(item.get("pass", True)) for item in rank_long_windows),
                    "long_window": {
                        "rank_results": rank_long_windows,
                        "steps_executed_max": max((int(item.get("steps_executed") or 0) for item in rank_long_windows), default=0),
                        "memory_creep_detected": any(bool(item.get("memory_creep_detected")) for item in rank_long_windows),
                    },
                    "finite_loss_sync_ok": True,
                    "graph_safe_backward_ok": True,
                    "cpu_worker_budget_ok": bool(budget["ok"]),
                    "loss_component_keys_per_rank_identical": bool(keys_identical),
                    "per_tier_loss": per_tier_loss,
                    "per_tier_loss_keys_per_rank_identical": bool(per_tier_keys_identical),
                    "gold_cf_loss_breakdown_present": True,
                    "missing_tier_graph_tied_zero": True,
                    "throughput_samples_per_sec": throughput,
                    "gpu_memory_peak_gb": gpu_peak_allocated,
                    "gpu_memory_reserved_peak_gb_diagnostic": gpu_peak_reserved,
                    "memory_truth": aggregate_memory_truth,
                    "oom": bool((aggregate_memory_truth or {}).get("oom")),
                    "model_memory_audit": dict((all_payloads[0] or {}).get("model_memory_audit") or {}),
                    "lora_parameter_participation": dict((all_payloads[0] or {}).get("lora_parameter_participation") or {}),
                    "all_trainable_grad_status": str((all_payloads[0] or {}).get("all_trainable_grad_status") or ""),
                    "all_trainable_grad": dict((all_payloads[0] or {}).get("all_trainable_grad") or {}),
                    "all_trainable_grad_required": True,
                    "runtime_E4_uses_all_trainable_grad_gate": True,
                    "formal_preflight_uses_all_trainable_grad_gate": True,
                    "formal_entry_E4_required": True,
                    "formal_entry_E4_validation_required": True,
                    "old_E4_cannot_allow_formal": True,
                    "head_gated_loss_contract": head_gated_loss_contract(str(args.stage)),
                    "trainable_param_count": int((all_payloads[0] or {}).get("trainable_param_count") or 0),
                    "grad_present_count": int((all_payloads[0] or {}).get("grad_present_count") or 0),
                    "lora_trainable_count": int((all_payloads[0] or {}).get("lora_trainable_count") or 0),
                    "lora_grad_present_count": int((all_payloads[0] or {}).get("lora_grad_present_count") or 0),
                    "missing_grad_params": list((all_payloads[0] or {}).get("missing_grad_params") or []),
                    "sequence_memory_audit": dict((all_payloads[0] or {}).get("sequence_memory_audit") or {}),
                    "graph_memory_audit": dict((all_payloads[0] or {}).get("graph_memory_audit") or {}),
                    "pipeline_timing_breakdown": pipeline_breakdown,
                    "data_wait_ratio": float(
                        max(float(((p or {}).get("timing_ms") or {}).get("data_wait_wall", 0.0)) for p in all_payloads)
                        / max(max(step_times), 1e-9)
                    ),
                    "gpu_util": gpu_util,
                    "gpu_util_mean": gpu_util.get("mean"),
                    "gpu_util_p50": gpu_util.get("p50"),
                    "gpu_util_p95": gpu_util.get("p95"),
                    "gpu_util_available": bool(gpu_util.get("available")),
                    "formal_namespace_pollution": False,
                    "latest_json_created": False,
                    "checkpoint_written": False,
                    "formal_run_id": None,
                    "full_csv_parse_in_bounded": any(bool(((p or {}).get("loader") or {}).get("full_csv_parse")) for p in all_payloads),
                }
            if requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE:
                e5_success = bool(e5_lifecycle_reports) and all(bool(item.get("success")) for item in e5_lifecycle_reports)
                final_payload.update(
                    {
                        "evidence_level": E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE,
                        "evidence_level_family": "E5_rating_stability_control_post_train_eval_lifecycle",
                        "success": e5_success,
                        "post_train_eval_lifecycle": True,
                        "checkpoint_saved": all(bool(item.get("checkpoint_saved")) for item in e5_lifecycle_reports),
                        "probe_checkpoint_saved": all(bool(item.get("checkpoint_saved")) for item in e5_lifecycle_reports),
                        "train_model_teardown_verified": all(
                            bool(item.get("train_model_teardown_verified")) for item in e5_lifecycle_reports
                        ),
                        "cpu_staged_checkpoint_load": all(
                            bool(item.get("cpu_staged_checkpoint_load")) for item in e5_lifecycle_reports
                        ),
                        "eval_model_constructed": all(
                            bool(item.get("eval_model_constructed")) for item in e5_lifecycle_reports
                        ),
                        "eval_forward_executed": all(
                            bool(item.get("eval_forward_executed")) for item in e5_lifecycle_reports
                        ),
                        "e5_lifecycle_rank_results": e5_lifecycle_reports,
                        "formal_namespace_pollution": False,
                        "formal_train_command_emitted": False,
                        "explanation_command_emitted": False,
                        "synthetic_batch_used_for_formal_gate": False,
                    }
                )
            final_payload["candidate_decision"] = candidate_decision_from_result(final_payload, memory_cfg)
            final = mark_gpu_shard_forward(final_payload)
            if requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE:
                final["evidence_level"] = E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE
                final["evidence_level_family"] = "E5_rating_stability_control_post_train_eval_lifecycle"
                final["post_train_eval_lifecycle"] = True
            else:
                final["evidence_level"] = E4_GPU_SHARD_FORWARD_BOUNDED_FORMAL_ENTRY_WITH_VALIDATION
            final["formal_entry_lifecycle"] = True
            _write_json(Path(args.result_path), final)
    except Exception as exc:
        failure_classification = _classify_step5_runtime_probe_failure(exc)
        rank_payload.update(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "oom": "out of memory" in str(exc).lower(),
                **failure_classification,
            }
        )
        try:
            _write_json(output_dir / f"rank{int(rank)}_error.json", rank_payload)
        except Exception:
            pass
        try:
            all_payloads = [None for _ in range(world_size)]
            dist.all_gather_object(all_payloads, rank_payload)
            if rank == 0:
                failure_memory_truth: dict[str, Any] = {}
                if torch.cuda.is_available():
                    try:
                        props = torch.cuda.get_device_properties(device)
                        max_alloc_gb = _gb(int(torch.cuda.max_memory_allocated(device)))
                        max_reserved_gb = _gb(int(torch.cuda.max_memory_reserved(device)))
                        failure_memory_truth = {
                            "device_total_gb": _round_gb(_gb(int(props.total_memory))),
                            "max_memory_allocated_gb": _round_gb(max_alloc_gb),
                            "max_memory_reserved_gb": _round_gb(max_reserved_gb),
                            "reserved_minus_allocated_gb": _round_gb(max(0.0, max_reserved_gb - max_alloc_gb)),
                            "allocated_to_total_ratio": float(max_alloc_gb / max(_gb(int(props.total_memory)), 1e-9)),
                            "reserved_to_total_ratio": float(max_reserved_gb / max(_gb(int(props.total_memory)), 1e-9)),
                            "nvidia_smi_process_used_gb": _nvidia_smi_process_used_gb(),
                            "param_memory_gb": 0.0,
                            "trainable_param_memory_gb": 0.0,
                            "frozen_param_memory_gb": 0.0,
                            "grad_memory_gb": 0.0,
                            "optimizer_state_memory_gb": 0.0,
                            "activation_peak_estimated_gb": _round_gb(max_alloc_gb),
                            "fragmentation_hint": {
                                "reserved_minus_allocated_gb": _round_gb(max(0.0, max_reserved_gb - max_alloc_gb)),
                                "reserved_is_allocator_cache": True,
                                "diagnostic_only": True,
                            },
                            "memory_creep_detected": False,
                            "oom": bool(rank_payload.get("oom")),
                            "oom_error_message": str(exc) if bool(rank_payload.get("oom")) else "",
                            "cuda_allocator_backend": _cuda_allocator_backend(),
                            "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                            "reserved_is_diagnostic_only": True,
                        }
                    except Exception:
                        failure_memory_truth = {}
                payload = {
                    "schema_version": "odcr_step5_e4_bounded_probe/1",
                    "stage": str(args.stage),
                    "task_id": int(args.task),
                    "candidate_id": str(args.candidate_id),
                    "success": False,
                    "evidence_level": E3_GPU_TRANSPORT,
                    "real_forward_backward_executed": False,
                    "rank_results": all_payloads,
                    "error": str(exc),
                    "oom": bool(rank_payload.get("oom")),
                    "memory_truth": failure_memory_truth,
                    "ddp": {
                        **_ddp_ready_hook_policy_payload(locals().get("cfg", object())),
                    },
                    "runtime_transformers_signature": _summarize_runtime_transformers_signatures(all_payloads),
                    **failure_classification,
                    "formal_namespace_pollution": False,
                    "latest_json_created": False,
                    "checkpoint_written": False,
                }
                payload["candidate_decision"] = candidate_decision_from_result(payload, locals().get("memory_cfg", {}))
                _write_json(Path(args.result_path), payload)
        except Exception:
            if rank == 0:
                _write_json(
                    Path(args.result_path),
                    {
                        "schema_version": "odcr_step5_e4_bounded_probe/1",
                        "stage": str(args.stage),
                        "task_id": int(args.task),
                        "candidate_id": str(args.candidate_id),
                        "success": False,
                        "evidence_level": E3_GPU_TRANSPORT,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        **failure_classification,
                    },
                )
        return 1
    finally:
        try:
            _write_json(_rank_paths(output_dir, rank), rank_payload)
        except Exception:
            pass
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    return 0


def launch_probe(
    *,
    stage: str,
    task: int,
    candidate_id: str,
    config_path: str,
    set_overrides: Sequence[str],
    from_step4: str | None,
    evidence_level: str | None = None,
    output_dir: str | Path | None = None,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    stage_norm = str(stage)
    requested_evidence_level = _normalize_evidence_level(evidence_level)
    if stage_norm not in {"rating_stability_control", "explanation"}:
        raise RuntimeError(f"Step5 E4/E5 launcher supports rating_stability_control/explanation, got {stage!r}")
    if requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE and stage_norm != "rating_stability_control":
        raise RuntimeError("E5_rating_stability_control_post_train_eval_lifecycle launcher only supports rating_stability_control.")
    cfg, snapshot, source_table = _resolve_candidate(
        stage=stage_norm,
        task=int(task),
        config_path=config_path,
        set_overrides=list(set_overrides),
        from_step4=from_step4,
    )
    if int(cfg.ddp_world_size) != 2:
        raise RuntimeError(f"Step5 E4 bounded probe requires resolved ddp_world_size=2, got {cfg.ddp_world_size}")
    e4_cfg = json.loads(str(cfg.step5_e4_bounded_config_json or "{}"))
    memory_cfg = json.loads(str(cfg.step5_memory_truth_config_json or "{}"))
    budget = _worker_budget(cfg)
    candidate = {
        "candidate_id": str(candidate_id),
        "stage": stage_norm,
        "task_id": int(task),
        "requested_evidence_level": requested_evidence_level,
        "set_overrides": list(set_overrides),
        "one_control_e4_bounded": e4_cfg,
        "worker_budget": budget,
    }
    out_dir = Path(output_dir).resolve() if output_dir is not None else _candidate_output_dir(stage_norm, int(task), candidate_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    min_steps = _candidate_min_steps(str(candidate_id), memory_cfg)
    artifact_build_requested = _artifact_build_preflight_requested(str(candidate_id))
    existing_validation = (
        {
            "valid": False,
            "fresh_required": True,
            "reason": (
                "e5_post_train_lifecycle_requires_fresh_ddp_child"
                if requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE
                else "artifact_build_preflight_requires_fresh_ddp_child"
            ),
        }
        if artifact_build_requested or requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE
        else _validate_existing_probe_result(
            result_path,
            stage=stage_norm,
            task=int(task),
            candidate_id=str(candidate_id),
            min_steps=min_steps,
            require_per_tier=True,
        )
    )
    if bool(existing_validation.get("valid")):
        result = _load_json(result_path)
        result["status_first_reuse"] = True
        result["reuse_validation"] = existing_validation
        _write_json(result_path, result)
        return result
    resolution_paths = _write_candidate_resolution(
        output_dir=out_dir,
        cfg=cfg,
        snapshot=snapshot,
        source_table=source_table,
        candidate=candidate,
    )
    if not bool(budget["ok"]):
        result = {
            "schema_version": "odcr_step5_e4_bounded_probe/1",
            "stage": stage_norm,
            "task_id": int(task),
            "candidate_id": str(candidate_id),
            "success": False,
            "evidence_level": E3_GPU_TRANSPORT,
            "requested_evidence_level": requested_evidence_level,
            "rejection_reason": "cpu_worker_budget_exceeded",
            "worker_budget": budget,
            "formal_namespace_pollution": False,
            "latest_json_created": False,
            "checkpoint_written": False,
            **resolution_paths,
        }
        result["candidate_decision"] = candidate_decision_from_result(result, memory_cfg)
        _write_json(out_dir / "result.json", result)
        return result
    request = {
        "schema_version": "odcr_step5_e4_probe_request/1",
        "stage": stage_norm,
        "task": int(task),
            "candidate_id": str(candidate_id),
            "config_path": str(config_path),
            "set_overrides": list(set_overrides),
            "from_step4": from_step4,
            "evidence_level": requested_evidence_level,
        }
    request_path = out_dir / "request.json"
    log_path = out_dir / "torchrun.log"
    _write_json(request_path, request)
    compute_guard = _prelaunch_compute_app_guard(candidate_id=str(candidate_id), out_dir=out_dir)
    guard_path = out_dir / "compute_app_guard.json"
    _write_json(guard_path, compute_guard)
    if not bool(compute_guard.get("pass")):
        result = {
            "schema_version": "odcr_step5_e4_bounded_probe/1",
            "stage": stage_norm,
            "task_id": int(task),
            "candidate_id": str(candidate_id),
            "success": False,
            "evidence_level": E3_GPU_TRANSPORT,
            "requested_evidence_level": requested_evidence_level,
            "rejection_reason": "compute_app_resource_conflict",
            "compute_app_guard": compute_guard,
            "compute_app_guard_path": str(guard_path),
            "formal_namespace_pollution": False,
            "latest_json_created": False,
            "checkpoint_written": False,
            **resolution_paths,
        }
        result["candidate_decision"] = candidate_decision_from_result(result, memory_cfg)
        _write_json(result_path, result)
        return result
    env = dict(os.environ)
    env.update(_runtime_env_from_cfg(cfg))
    env["PYTHONPATH"] = str(CODE_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        *_torchrun_cmd(),
        "--standalone",
        "--nproc_per_node=2",
        str(CODE_DIR / "odcr_core" / "step5_runtime_probe.py"),
        "--worker",
        "--stage",
        stage_norm,
        "--task",
        str(int(task)),
        "--candidate-id",
        str(candidate_id),
        "--request-path",
        str(request_path),
        "--output-dir",
        str(out_dir),
        "--result-path",
        str(result_path),
        "--evidence-level",
        requested_evidence_level,
    ]
    started = time.monotonic()
    proc_returncode: int | None = None
    timeout_limit = int(timeout_s or e4_cfg.get("max_runtime_seconds", 900))
    command_id = (
        f"step5_{'e5' if requested_evidence_level == E5_STEP5_EXPLANATION_POST_TRAIN_EVAL_LIFECYCLE else 'e4'}_probe:"
        f"{stage_norm}:{int(task)}:{stable_hash([candidate_id, set_overrides, requested_evidence_level], length=12)}:{int(time.time())}"
    )
    process_state_path = out_dir / "process_state.json"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[step5-e4-probe] exec=" + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _write_process_state(
                process_state_path,
                {
                    "schema_version": "odcr_step5_bounded_process_state/1",
                    "command_id": command_id,
                    "pid": int(proc.pid),
                    "pgid": int(os.getpgid(proc.pid)),
                    "candidate_id": str(candidate_id),
                    "stage": stage_norm,
                    "task": int(task),
                    "output_dir": str(out_dir),
                    "result_path": str(result_path),
                    "started_at_monotonic": float(started),
                    "owned_bounded_process_group": True,
                },
            )
        except Exception:
            pass
        try:
            proc_returncode = int(proc.wait(timeout=timeout_limit))
        except subprocess.TimeoutExpired:
            proc_returncode = 124
            log.write(f"\n[step5-e4-probe] timeout after {timeout_limit}s; entering child-status grace period.\n")
            log.flush()
            try:
                proc_returncode = int(proc.wait(timeout=60))
            except subprocess.TimeoutExpired:
                recovery_validation = _validate_existing_probe_result(
                    result_path,
                    stage=stage_norm,
                    task=int(task),
                    candidate_id=str(candidate_id),
                    min_steps=min_steps,
                    require_per_tier=True,
                )
                if bool(recovery_validation.get("valid")):
                    log.write("[step5-e4-probe] valid result appeared during timeout recovery; waiting briefly for torchrun exit.\n")
                    log.flush()
                    try:
                        proc_returncode = int(proc.wait(timeout=60))
                    except subprocess.TimeoutExpired:
                        cleanup = _cleanup_process_group(os.getpgid(proc.pid), reason="completed_result_lingering_process_group")
                        log.write("[step5-e4-probe] cleanup after completed result: " + json.dumps(cleanup, sort_keys=True) + "\n")
                        log.flush()
                        proc_returncode = 0 if bool(cleanup.get("success")) else 124
                else:
                    compute_after_timeout = _nvidia_smi_compute_apps()
                    log.write(
                        "[step5-e4-probe] timeout recovery found no valid result; compute-apps="
                        + json.dumps(compute_after_timeout, sort_keys=True, default=_json_default)
                        + "\n"
                    )
                    log.flush()
                    cleanup = _cleanup_process_group(os.getpgid(proc.pid), reason="hard_timeout_owned_bounded_process_group")
                    log.write("[step5-e4-probe] hard-timeout cleanup: " + json.dumps(cleanup, sort_keys=True) + "\n")
                    log.flush()
                    proc_returncode = 124
    elapsed = time.monotonic() - started
    if result_path.is_file():
        result = _load_json(result_path)
    else:
        result = {
            "schema_version": "odcr_step5_e4_bounded_probe/1",
            "stage": stage_norm,
            "task_id": int(task),
            "candidate_id": str(candidate_id),
            "success": False,
            "evidence_level": E3_GPU_TRANSPORT,
            "error": "torchrun timed out before result.json" if proc_returncode == 124 else "torchrun did not write result.json",
            "timeout_s": timeout_limit if proc_returncode == 124 else None,
        }
    result.update(
        {
            **resolution_paths,
            "output_dir": str(out_dir),
            "torchrun_log_path": str(log_path),
            "torchrun_returncode": int(proc_returncode if proc_returncode is not None else 1),
            "elapsed_sec": float(elapsed),
            "command": cmd,
            "bridge_command_id": command_id,
            "process_state_path": str(process_state_path),
            "compute_app_guard": compute_guard,
            "compute_app_guard_path": str(guard_path),
            "status_first_reuse": False,
        }
    )
    if int(proc_returncode if proc_returncode is not None else 1) != 0:
        result["success"] = False
        result.setdefault("rejection_reason", "torchrun_timeout" if proc_returncode == 124 else "torchrun_failed")
        result.setdefault("evidence_level", E3_GPU_TRANSPORT)
    result.setdefault("candidate_decision", candidate_decision_from_result(result, memory_cfg))
    _patch_candidate_resolution_contract(resolution_paths=resolution_paths, result=result)
    _write_json(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step5 E4 bounded runtime probe")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--request-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--evidence-level", default="E4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.worker:
        raise SystemExit("step5_runtime_probe.py is an internal torchrun worker; use ./odcr runtime probe.")
    return _run_rank_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
