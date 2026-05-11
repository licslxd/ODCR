#!/usr/bin/env python3
"""Internal Step3 startup-only GPU validation.

This tool is not a user-visible ODCR entrypoint.  It is called only by the
controlled tmux GPU bridge in ``step3-startup-validation`` mode.  It resolves
the task2 Step3 One-Control payload, writes an isolated validation run
namespace, builds and reuses a tiny tokenizer cache through the live Step3
atomic cache functions, then launches two local NCCL ranks that load the
completed cache before initializing DDP.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402
from odcr_core import path_layout  # noqa: E402
from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.file_atomic import atomic_write_json  # noqa: E402
from odcr_core.manifests import (  # noqa: E402
    build_formal_source_table_snapshot,
    build_run_summary,
    formal_snapshot_view,
    write_resolved_config_artifacts,
    write_run_summary_json,
    write_training_runtime_config_artifact,
)
from odcr_core.runners import _odcr_layout_env, _torchrun_hardware_env  # noqa: E402
from odcr_core.step3_upstream_gate import validate_step3_preprocess_upstream_gate  # noqa: E402
from odcr_core.training_checkpoint import file_fingerprint, stable_hash  # noqa: E402


SCHEMA_VERSION = "odcr_step3_startup_validation/1"
DEFAULT_SLUG = "step3_tmux_gpu_bridge_startup_validation_closeout"
STATUS_OK = "completed_validation"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_component(value: str) -> str:
    import re

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("component must be non-empty")
    if raw in {".", ".."} or "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(f"unsafe path component: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", raw):
        raise ValueError(f"unsafe path component: {value!r}")
    return raw


class TeeLog:
    def __init__(self, *paths: Path) -> None:
        self.paths = tuple(path for path in paths if path is not None)
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def emit(self, *parts: Any) -> None:
        line = " ".join(str(part) for part in parts)
        print(line, flush=True)
        for path in self.paths:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _thread_trace(cfg: Any, formal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    formal = formal or {}
    hardware = formal.get("hardware") if isinstance(formal.get("hardware"), dict) else {}
    runtime_env = formal.get("runtime_env") if isinstance(formal.get("runtime_env"), dict) else {}
    thread_env = runtime_env.get("thread_env_effective")
    if not isinstance(thread_env, dict):
        thread_env = runtime_env.get("thread_env_requested")
    if not isinstance(thread_env, dict):
        thread_env = {}
    try:
        hardware_profile = json.loads(cfg.hardware_profile_json or "{}")
    except json.JSONDecodeError:
        hardware_profile = {}
    worker_budget = hardware.get("worker_budget_formula") if isinstance(hardware, dict) else {}
    if not isinstance(worker_budget, dict):
        worker_budget = hardware_profile.get("worker_budget_formula") if isinstance(hardware_profile, dict) else {}
    reserved_cpu = int(runtime_env.get("reserved_cpu") or worker_budget.get("reserved_cpu", 2))
    max_parallel_cpu = int(runtime_env.get("max_parallel_cpu") or hardware.get("max_parallel_cpu") or 0)
    train_workers = int(hardware.get("dataloader_num_workers_train") or getattr(cfg, "dataloader_num_workers_train", 0) or 0)
    ddp_world_size = int(hardware.get("ddp_world_size") or cfg.ddp_world_size)
    num_proc = int(runtime_env.get("num_proc") or cfg.num_proc)
    tokenization_formula = runtime_env.get("tokenization_formula") or (
        f"num_proc({num_proc}) + reserved_cpu({reserved_cpu}) <= max_parallel_cpu({max_parallel_cpu})"
    )
    worker_formula = runtime_env.get("worker_formula") or (
        f"dataloader_num_workers_train({train_workers}) * ddp_world_size({ddp_world_size}) "
        f"+ reserved_cpu({reserved_cpu}) <= max_parallel_cpu({max_parallel_cpu})"
    )
    return {
        "TOKENIZERS_PARALLELISM": str(thread_env.get("TOKENIZERS_PARALLELISM") or ("true" if bool(cfg.tokenizers_parallelism) else "false")),
        "OMP_NUM_THREADS": str(thread_env.get("OMP_NUM_THREADS") or int(cfg.omp_num_threads)),
        "MKL_NUM_THREADS": str(thread_env.get("MKL_NUM_THREADS") or int(cfg.mkl_num_threads)),
        "num_proc": num_proc,
        "max_parallel_cpu": max_parallel_cpu,
        "reserved_cpu": reserved_cpu,
        "tokenization_formula": tokenization_formula,
        "worker_formula": worker_formula,
    }


def _validation_paths(repo_root: Path, slug: str, run_id: str) -> dict[str, Path]:
    run_root = path_layout.get_step3_validation_run_root(repo_root, slug, run_id)
    meta = path_layout.get_step3_validation_meta_dir(repo_root, slug, run_id)
    evidence = path_layout.step3_validation_evidence_root(repo_root, slug, run_id)
    return {
        "run_root": run_root,
        "meta": meta,
        "evidence": evidence,
        "input": evidence / "input",
        "ranks": evidence / "ranks",
        "console_log": meta / "console.log",
        "full_log": meta / "full.log",
        "errors_log": meta / "errors.log",
        "result_json": evidence / "startup_validation_result.json",
    }


def _resolve_step3_task(task_id: int, paths: Mapping[str, Path]) -> tuple[Any, dict[str, Any]]:
    cfg, _sources, snapshot = resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=int(task_id),
        set_overrides=[],
        dry_run=True,
        run_id="auto",
        mode="full",
    )
    formal = formal_snapshot_view(snapshot)
    formal["validation"] = {
        "schema_version": SCHEMA_VERSION,
        "validation_mode": "startup-only",
        "namespace": "validation",
        "formal_parameters_modified": False,
        "formal_latest_updates_allowed": False,
        "formal_checkpoint_writes_allowed": False,
    }
    source_table = build_formal_source_table_snapshot(snapshot)
    source_table["validation"] = {
        "schema_version": SCHEMA_VERSION,
        "namespace": "validation",
        "formal_latest_updates_allowed": False,
    }
    write_resolved_config_artifacts(
        paths["meta"],
        formal,
        source_table=source_table,
        write_verbose_source_table=False,
    )
    return cfg, formal


def _patched_env(updates: Mapping[str, str]):
    @contextlib.contextmanager
    def _manager():
        old = {key: os.environ.get(key) for key in updates}
        os.environ.update({str(key): str(value) for key, value in updates.items()})
        try:
            yield
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return _manager()


def _runtime_env_for_validation(cfg: Any, paths: Mapping[str, Path]) -> dict[str, str]:
    env = dict(_odcr_layout_env(cfg))
    env.update(_torchrun_hardware_env(cfg))
    env["ODCR_STAGE_RUN_DIR"] = str(paths["run_root"])
    env["ODCR_MANIFEST_DIR"] = str(paths["meta"])
    env["ODCR_LOG_DIR"] = str(paths["meta"])
    env["ODCR_SUMMARY_LOG"] = str(paths["console_log"])
    env["ODCR_STEP3_TOKENIZER_CACHE_STARTUP_JSON"] = str(paths["meta"] / step3.STEP3_TOKENIZE_CACHE_STARTUP_FILENAME)
    return env


def _slice_validation_inputs(cfg: Any, paths: Mapping[str, Path], *, rows: int) -> tuple[Path, Path, dict[str, int]]:
    import pandas as pd

    src_root = Path(cfg.merged_dir) / str(int(cfg.task_id))
    train_src = src_root / "aug_train.csv"
    valid_src = src_root / "aug_valid.csv"
    if not train_src.is_file() or not valid_src.is_file():
        raise FileNotFoundError(f"Step3 validation requires merged task{cfg.task_id} CSVs: {train_src}, {valid_src}")
    train_df = pd.read_csv(train_src, nrows=max(1, int(rows)))
    valid_df = pd.read_csv(valid_src, nrows=max(1, min(int(rows), 4)))
    step3._require_step3_canonical_columns(train_df, csv_path=str(train_src), split="train")
    step3._require_step3_canonical_columns(valid_df, csv_path=str(valid_src), split="valid")
    train_df = train_df[train_df["explanation"].notna()].reset_index(drop=True)
    valid_df = valid_df[valid_df["explanation"].notna()].reset_index(drop=True)
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Step3 validation real-data slice is empty after explanation filter.")
    train_df["item"] = train_df["item"].astype(str)
    valid_df["item"] = valid_df["item"].astype(str)
    train_df["sample_id"] = range(len(train_df))
    valid_df["sample_id"] = range(len(valid_df))
    paths["input"].mkdir(parents=True, exist_ok=True)
    train_out = paths["input"] / "validation_aug_train_slice.csv"
    valid_out = paths["input"] / "validation_aug_valid_slice.csv"
    train_df.to_csv(train_out, index=False)
    valid_df.to_csv(valid_out, index=False)
    return train_out, valid_out, {"train": int(len(train_df)), "valid": int(len(valid_df))}


def _build_validation_dataset(train_path: Path, valid_path: Path) -> Any:
    import pandas as pd
    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_pandas(pd.read_csv(train_path)),
            "valid": Dataset.from_pandas(pd.read_csv(valid_path)),
        }
    )


def _build_cache_payload(
    cfg: Any,
    train_path: Path,
    valid_path: Path,
    split_counts: Mapping[str, int],
    upstream_evidence: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], str, Path]:
    tok = step3.get_odcr_text_tokenizer()
    processor = step3.Processor(
        cfg.auxiliary,
        cfg.target,
        max_length=int(cfg.tokenizer_max_length),
        evidence_length=int(cfg.evidence_max_length),
    )
    fp = step3._build_tokenize_cache_fingerprint(
        train_path=str(train_path),
        valid_path=str(valid_path),
        task_idx=int(cfg.task_id),
        source_domain=str(cfg.auxiliary),
        target_domain=str(cfg.target),
        mode="train_valid",
        split_row_counts=split_counts,
        upstream_evidence=upstream_evidence,
        tok=tok,
        max_length=int(cfg.tokenizer_max_length),
        evidence_length=int(cfg.evidence_max_length),
        cache_version=step3.ODCR_TOKENIZE_CACHE_VERSION,
    )
    compat_key = f"{step3.ODCR_TOKENIZE_CACHE_VERSION}_{str(fp['tokenizer_cache_compat_hash'])[:16]}"
    return processor, fp, compat_key, tok


def _cache_phase_guard() -> tuple[list[str], Any]:
    import torch.distributed as dist

    calls: list[str] = []
    originals = {
        "barrier": dist.barrier,
        "all_reduce": dist.all_reduce,
        "broadcast_object_list": dist.broadcast_object_list,
    }

    @contextlib.contextmanager
    def _manager():
        def _blocked(name: str):
            def _inner(*_args: Any, **_kwargs: Any) -> None:
                calls.append(name)
                raise RuntimeError(f"distributed collective called during cache phase: {name}")

            return _inner

        dist.barrier = _blocked("barrier")  # type: ignore[assignment]
        dist.all_reduce = _blocked("all_reduce")  # type: ignore[assignment]
        dist.broadcast_object_list = _blocked("broadcast_object_list")  # type: ignore[assignment]
        try:
            yield
        finally:
            dist.barrier = originals["barrier"]  # type: ignore[assignment]
            dist.all_reduce = originals["all_reduce"]  # type: ignore[assignment]
            dist.broadcast_object_list = originals["broadcast_object_list"]  # type: ignore[assignment]

    return calls, _manager()


def _rank_worker(
    rank: int,
    world_size: int,
    init_method: str,
    cache_dir: str,
    fingerprint: dict[str, Any],
    env_updates: dict[str, str],
    rank_dir: str,
    timeout_s: int,
) -> None:
    import torch
    import torch.distributed as dist

    os.environ.update(env_updates)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    rank_path = Path(rank_dir) / f"rank_{rank}.json"
    started = _utc_now()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rank": int(rank),
        "local_rank": int(rank),
        "world_size": int(world_size),
        "started_at": started,
        "phase": "rank_start",
        "cache_dir": str(cache_dir),
        "cache_loaded": False,
        "distributed_collective_calls_in_cache_phase": None,
        "nccl_initialized": False,
        "nccl_init_after_cache_ready": False,
        "training_loop_full_epoch_started": False,
        "checkpoint_created": False,
    }
    try:
        device_count = int(torch.cuda.device_count())
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_cuda_device_count"] = device_count
        if not torch.cuda.is_available() or device_count < world_size:
            raise RuntimeError(
                "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
                "in this same tmux to enter the GPU node, then rerun the probe."
            )
        torch.cuda.set_device(rank)
        payload["device_name"] = torch.cuda.get_device_name(rank)
        payload["phase"] = "cache_load"
        calls, guard = _cache_phase_guard()
        cache_ready_at = ""
        with guard:
            step3.load_completed_step3_tokenizer_cache_for_rank(
                cache_dir,
                expected_fingerprint=fingerprint,
                rank=rank,
                timing_sink=payload,
            )
            cache_ready_at = _utc_now()
        payload["cache_loaded"] = True
        payload["cache_ready_at"] = cache_ready_at
        payload["distributed_collective_calls_in_cache_phase"] = len(calls)
        payload["distributed_collective_call_names_in_cache_phase"] = calls
        payload["phase"] = "nccl_init"
        nccl_start = _utc_now()
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=int(rank),
            world_size=int(world_size),
            timeout=dt.timedelta(seconds=max(15, min(int(timeout_s), 120))),
        )
        payload["nccl_initialized"] = True
        payload["nccl_init_started_at"] = nccl_start
        payload["nccl_init_after_cache_ready"] = bool(cache_ready_at and nccl_start >= cache_ready_at)
        payload["phase"] = "post_init_barrier"
        dist.barrier()
        payload["phase"] = "completed"
        payload["status"] = "ok"
        payload["finished_at"] = _utc_now()
    except Exception as exc:
        payload["status"] = "failed"
        payload["fatal_signature"] = repr(exc)
        payload["traceback"] = traceback.format_exc()
        payload["finished_at"] = _utc_now()
        atomic_write_json(rank_path, payload)
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    atomic_write_json(rank_path, payload)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _formal_latest_hash(repo_root: Path, task_id: int) -> str | None:
    latest = path_layout.get_stage_task_root(repo_root, "step3", task_id) / "latest.json"
    if not latest.is_file():
        return None
    return file_fingerprint(latest, sample_only=True).get("sha256")


def _failed_latest_rejected(repo_root: Path, task_id: int) -> dict[str, Any]:
    from odcr_core.config_resolver import OneControlConfigError, _latest_run

    try:
        _latest_run(repo_root, int(task_id), "step3", dry_run=False)
    except OneControlConfigError as exc:
        return {"rejected": True, "signature": str(exc)}
    return {"rejected": False, "signature": "latest accepted"}


def _write_summary(
    *,
    cfg: Any,
    paths: Mapping[str, Path],
    run_id: str,
    status: str,
    started_at: str,
    finished_at: str,
    latest_error: str | None,
    result_payload: Mapping[str, Any],
) -> Path:
    base = build_run_summary(
        repo_root=REPO_ROOT,
        run_dir=paths["run_root"],
        meta_dir=paths["meta"],
        run_id=run_id,
        stage="step3_validation",
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        command="step3-startup-validation startup-only",
        task_id=int(cfg.task_id),
        source_domain=str(cfg.auxiliary),
        target_domain=str(cfg.target),
        console_log_path=paths["console_log"],
        full_log_path=paths["full_log"],
        errors_log_path=paths["errors_log"],
        key_artifacts={
            "startup_validation_result": paths["result_json"],
            "training_runtime_config": paths["meta"] / "training_runtime_config.json",
            "resolved_config": paths["meta"] / "resolved_config.json",
            "source_table": paths["meta"] / "source_table.json",
        },
        latest_error=latest_error,
        validation_status=status,
    )
    base.update(dict(result_payload))
    return write_run_summary_json(base, repo_root=REPO_ROOT, update_latest=False)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    slug = _safe_component(args.slug)
    run_id = _safe_component(args.run_id)
    if args.mode != "startup-only" or args.namespace != "validation":
        raise ValueError("Step3 startup validation only supports --mode startup-only --namespace validation")
    paths = _validation_paths(REPO_ROOT, slug, run_id)
    paths["meta"].mkdir(parents=True, exist_ok=True)
    paths["evidence"].mkdir(parents=True, exist_ok=True)
    paths["ranks"].mkdir(parents=True, exist_ok=True)
    log = TeeLog(paths["console_log"], paths["full_log"], Path(args.bridge_log_path) if args.bridge_log_path else None)
    started_at = _utc_now()
    before_latest_hash = _formal_latest_hash(REPO_ROOT, int(args.task))
    latest_error: str | None = None
    cfg: Any | None = None
    try:
        log.emit("ODCR_STEP3_STARTUP_VALIDATION_BEGIN", run_id)
        cfg, _formal = _resolve_step3_task(int(args.task), paths)
        env_updates = _runtime_env_for_validation(cfg, paths)
        os.environ.update(env_updates)
        thread_trace = _thread_trace(cfg, _formal)
        runtime_payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": "startup_validation_pre_cache",
            "status": "initial",
            "validation_mode": "startup-only",
            "namespace": "validation",
            "training_loop_started": False,
            "training_loop_full_epoch_started": False,
            "checkpoint_created": False,
            **thread_trace,
            "thread_env_effective": {
                "TOKENIZERS_PARALLELISM": thread_trace["TOKENIZERS_PARALLELISM"],
                "OMP_NUM_THREADS": thread_trace["OMP_NUM_THREADS"],
                "MKL_NUM_THREADS": thread_trace["MKL_NUM_THREADS"],
            },
            "runtime_env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            },
        }
        write_training_runtime_config_artifact(paths["meta"], runtime_payload)
        train_path, valid_path, split_counts = _slice_validation_inputs(cfg, paths, rows=int(args.rows))
        log.emit("validation_input", train_path, valid_path, split_counts)
        upstream = validate_step3_preprocess_upstream_gate(
            repo_root=REPO_ROOT,
            task_id=int(cfg.task_id),
            auxiliary_domain=str(cfg.auxiliary),
            target_domain=str(cfg.target),
            data_dir=str(cfg.data_dir),
            merged_dir=str(cfg.merged_dir),
            runs_dir=str(REPO_ROOT / "runs"),
            embed_dim=int(cfg.embed_dim),
        )
        processor, fp, compat_key, _tok = _build_cache_payload(cfg, train_path, valid_path, split_counts, upstream)
        cache_dir = path_layout.step3_validation_tokenizer_cache_entry_dir(
            REPO_ROOT,
            validation_slug=slug,
            run_id=run_id,
            task_id=int(cfg.task_id),
            source_domain=str(cfg.auxiliary),
            target_domain=str(cfg.target),
            compatibility_key=compat_key,
        )
        fp["validation_cache_namespace"] = str(cache_dir.parent)
        fp["validation_run_id"] = run_id
        fp["validation_slug"] = slug
        datasets = _build_validation_dataset(train_path, valid_path)
        cold_timing: dict[str, Any] = {}
        with _patched_env(env_updates):
            cold_manifest = step3.build_or_reuse_step3_tokenizer_cache_atomic(
                datasets=datasets,
                processor=processor,
                nproc=int(cfg.num_proc),
                cache_dir=str(cache_dir),
                cache_fingerprint=compat_key,
                cache_fingerprint_payload=fp,
                build_allowed=True,
                rank="validation-parent",
                show_datasets_progress=False,
                log_tokenize=True,
                phase="startup validation cold",
                log_file=str(paths["full_log"]),
                timing_sink=cold_timing,
            )
            warm_timing: dict[str, Any] = {}
            warm_manifest = step3.build_or_reuse_step3_tokenizer_cache_atomic(
                datasets=None,
                processor=processor,
                nproc=int(cfg.num_proc),
                cache_dir=str(cache_dir),
                cache_fingerprint=compat_key,
                cache_fingerprint_payload=fp,
                build_allowed=False,
                rank="validation-parent",
                show_datasets_progress=False,
                log_tokenize=True,
                phase="startup validation warm",
                log_file=str(paths["full_log"]),
                timing_sink=warm_timing,
            )
        if cold_timing.get("cache_status") != "miss_or_rebuild_completed":
            raise RuntimeError(f"cold cache did not complete as a rebuild: {cold_timing}")
        if warm_timing.get("cache_status") != "hit":
            raise RuntimeError(f"warm cache did not hit: {warm_timing}")
        import torch
        import torch.multiprocessing as mp

        device_count = int(torch.cuda.device_count())
        if not torch.cuda.is_available() or device_count < 2:
            raise RuntimeError(
                "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
                "in this same tmux to enter the GPU node, then rerun the probe."
            )
        port = _free_port()
        init_method = f"tcp://127.0.0.1:{port}"
        log.emit("launching_two_rank_startup_validation", init_method)
        mp.spawn(
            _rank_worker,
            args=(
                2,
                init_method,
                str(cache_dir),
                dict(fp),
                env_updates,
                str(paths["ranks"]),
                int(args.max_seconds),
            ),
            nprocs=2,
            join=True,
        )
        rank_payloads = [_read_json(paths["ranks"] / f"rank_{rank}.json") for rank in (0, 1)]
        ranks_seen = sorted(int(item["rank"]) for item in rank_payloads if item.get("status") == "ok")
        cache_collectives = sum(int(item.get("distributed_collective_calls_in_cache_phase") or 0) for item in rank_payloads)
        nccl_order_ok = all(bool(item.get("nccl_init_after_cache_ready")) for item in rank_payloads)
        after_latest_hash = _formal_latest_hash(REPO_ROOT, int(args.task))
        formal_latest_updated = before_latest_hash != after_latest_hash
        failed_latest_gate = _failed_latest_rejected(REPO_ROOT, int(args.task))
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_OK,
            "validation_mode": "startup-only",
            "validation_namespace": "validation",
            "validation_run_id": run_id,
            "validation_slug": slug,
            "run_dir": str(paths["run_root"]),
            "evidence_dir": str(paths["evidence"]),
            "cache_status": "cold_completed_and_warm_hit",
            "cold_cache_status": cold_timing.get("cache_status"),
            "warm_cache_status": warm_timing.get("cache_status"),
            "cache_dir": str(cache_dir),
            "cache_key": compat_key,
            "cache_manifest_path": str(Path(step3._step3_tokenize_cache_manifest_path(str(cache_dir)))),
            "cache_manifest_hash": stable_hash(cold_manifest),
            "warm_manifest_hash": stable_hash(warm_manifest),
            "nccl_init_after_cache_ready": bool(nccl_order_ok),
            "distributed_collective_calls_in_cache_phase": int(cache_collectives),
            "ranks_seen": ranks_seen,
            "torch_cuda_device_count": device_count,
            "hostname": socket.gethostname(),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "formal_latest_updated": bool(formal_latest_updated),
            "formal_namespace_polluted": bool(formal_latest_updated),
            "checkpoint_created": False,
            "training_loop_full_epoch_started": False,
            "training_loop_started": False,
            "failed_latest_rejection": failed_latest_gate,
            "thread_trace": thread_trace,
            "rank_evidence": [str(paths["ranks"] / f"rank_{rank}.json") for rank in (0, 1)],
        }
        if ranks_seen != [0, 1] or cache_collectives != 0 or not nccl_order_ok or formal_latest_updated:
            raise RuntimeError(f"Step3 startup validation invariant failed: {result}")
        atomic_write_json(paths["result_json"], result)
        write_training_runtime_config_artifact(
            paths["meta"],
            {
                **runtime_payload,
                "status": "completed",
                "phase": "startup_validation_completed",
                "cache_status": result["cache_status"],
                "cache_dir": result["cache_dir"],
                "cache_key": result["cache_key"],
                "ranks_seen": ranks_seen,
                "nccl_init_after_cache_ready": True,
                "distributed_collective_calls_in_cache_phase": 0,
            },
        )
        finished_at = _utc_now()
        summary_path = _write_summary(
            cfg=cfg,
            paths=paths,
            run_id=run_id,
            status=STATUS_OK,
            started_at=started_at,
            finished_at=finished_at,
            latest_error=None,
            result_payload=result,
        )
        result["run_summary_path"] = str(summary_path)
        atomic_write_json(paths["result_json"], result)
        log.emit("ODCR_STEP3_STARTUP_VALIDATION_END", run_id, "PASS")
        return result
    except KeyboardInterrupt:
        latest_error = "interrupted"
        raise
    except Exception as exc:
        latest_error = repr(exc)
        if paths["errors_log"]:
            _write_text(paths["errors_log"], traceback.format_exc())
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "validation_mode": "startup-only",
            "validation_namespace": "validation",
            "validation_run_id": run_id,
            "failure_phase": "startup_validation",
            "fatal_signature": repr(exc),
            "cache_status": "failed_or_incomplete",
            "whether_nccl_initialized": False,
            "whether_training_loop_started": False,
            "whether_checkpoint_created": False,
            "checkpoint_created": False,
            "formal_latest_updated": before_latest_hash != _formal_latest_hash(REPO_ROOT, int(args.task)),
            "formal_namespace_polluted": before_latest_hash != _formal_latest_hash(REPO_ROOT, int(args.task)),
            "root_cause_summary": str(exc),
        }
        atomic_write_json(paths["result_json"], result)
        if cfg is not None:
            _write_summary(
                cfg=cfg,
                paths=paths,
                run_id=run_id,
                status="failed",
                started_at=started_at,
                finished_at=_utc_now(),
                latest_error=latest_error,
                result_payload=result,
            )
        log.emit("ODCR_STEP3_STARTUP_VALIDATION_END", run_id, "FAIL", repr(exc))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal Step3 startup-only validation.")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--mode", choices=("startup-only",), default="startup-only")
    parser.add_argument("--namespace", choices=("validation",), default="validation")
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--max-seconds", type=int, default=170)
    parser.add_argument("--bridge-status-path")
    parser.add_argument("--bridge-log-path")
    parser.add_argument("--target-socket")
    parser.add_argument("--target-pane")
    parser.add_argument("--target-job-id")
    parser.add_argument("--target-node")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    try:
        result = run_validation(args)
        bridge_success = True
        exit_code = 0
        stop_reason = "step3_startup_validation_completed"
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "fatal_signature": repr(exc),
            "traceback": traceback.format_exc(),
        }
        bridge_success = False
        exit_code = 1
        stop_reason = "step3_startup_validation_failed"
    if args.bridge_status_path:
        status = {
            "schema_version": "odcr_tmux_gpu_bridge_status/1.0",
            "run_id": args.run_id,
            "kind": "step3-startup-validation",
            "success": bridge_success,
            "exit_code": exit_code,
            "elapsed_s": round(time.monotonic() - started, 3),
            "startup_timeout_s": min(20, int(args.max_seconds)),
            "first_result_timeout_s": int(args.max_seconds),
            "hard_timeout_s": int(args.max_seconds),
            "first_result_seen": True,
            "success_condition": "step3_startup_validation_completed",
            "stop_reason": stop_reason,
            "metrics": result,
        }
        atomic_write_json(Path(args.bridge_status_path), status)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
