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
from odcr_core.training_checkpoint import file_fingerprint, stable_hash


STEP4_PREFLIGHT_SCHEMA_VERSION = "odcr_step4_bounded_preflight/1"
STEP4_CACHE_PREPARE_SCHEMA_VERSION = "odcr_step4_prepare_cache/1"
STEP4_RUNTIME_ENV_KNOBS = (
    "ODCR_STEP4_DECODE_THREADS",
    "ODCR_STEP4_DECODE_CHUNK",
    "ODCR_STEP4_PARTIAL_FORMAT",
    "ODCR_STEP4_PERF_LOG_INTERVAL",
)


class Step4RuntimeError(RuntimeError):
    """Raised when Step4 runtime policy would be unsafe."""


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


def _layout_env(cfg: Any) -> dict[str, str]:
    out = {
        "ODCR_ROOT": str(Path(cfg.repo_root).resolve()),
        "ODCR_STAGE_RUN_DIR": str(Path(cfg.checkpoint_dir).resolve()),
        "ODCR_MANIFEST_DIR": str(Path(cfg.manifest_dir).resolve()),
        "ODCR_LOG_DIR": str(Path(cfg.manifest_dir).resolve()),
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
    if getattr(cfg, "step3_checkpoint_dir", None):
        out["ODCR_STEP3_RUN_DIR"] = str(Path(cfg.step3_checkpoint_dir).resolve())
    out.update(step4_runtime_env(cfg, mode="preflight"))
    return out


def _checkpoint_lineage_for_cache(cfg: Any) -> tuple[str, dict[str, Any]]:
    ckpt_dir = Path(str(cfg.step3_checkpoint_dir or "")).expanduser().resolve()
    checkpoint = ckpt_dir / "model" / "best.pth"
    sidecar = Path(str(checkpoint) + ".lineage.json")
    payload = _load_json(sidecar)
    lineage_hash = str(payload.get("lineage_hash") or "")
    if not lineage_hash:
        raise Step4RuntimeError(f"Step3 checkpoint lineage hash missing: {sidecar}")
    return lineage_hash, {
        "checkpoint_path": str(checkpoint),
        "checkpoint_hash": str(payload.get("checkpoint_file_hash") or ""),
        "checkpoint_lineage_path": str(sidecar),
        "checkpoint_lineage_hash": lineage_hash,
    }


def prepare_step4_encoded_cache(cfg: Any, *, dry_run: bool = False, build_allowed: bool = True) -> dict[str, Any]:
    """Build or validate the Step4 encoded cache before any DDP/NCCL setup."""
    with _patched_env(_layout_env(cfg)):
        from datasets import Dataset, load_from_disk
        from executors import step4_engine
        from paths_config import require_step5_text_model_dir

        task = int(cfg.task_id)
        aug_csv = Path(cfg.merged_dir).resolve() / str(task) / "aug_train.csv"
        if not aug_csv.is_file():
            raise Step4RuntimeError(f"missing Step4 source CSV: {aug_csv}")
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
        lineage_hash, lineage = _checkpoint_lineage_for_cache(cfg)
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
        if dry_run or valid:
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


def _tree_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "file_count": 0}
    if path.is_file():
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


def _formal_namespace_snapshot(cfg: Any) -> dict[str, Any]:
    root = Path(cfg.repo_root).resolve()
    task = int(cfg.task_id)
    step3_dir = Path(str(cfg.step3_checkpoint_dir or "")).resolve() if getattr(cfg, "step3_checkpoint_dir", None) else root / "runs" / "step3" / f"task{task}" / "2"
    return {
        "schema_version": "odcr_step4_formal_namespace_snapshot/1",
        "step4_latest": _tree_fingerprint(root / "runs" / "step4" / f"task{task}" / "latest.json"),
        "step4_formal_task_dir": _tree_fingerprint(root / "runs" / "step4" / f"task{task}"),
        "step5_latest": _tree_fingerprint(root / "runs" / "step5" / f"task{task}" / "latest.json"),
        "eval_latest": _tree_fingerprint(root / "runs" / "eval" / f"task{task}" / "latest.json"),
        "config": _tree_fingerprint(root / "configs" / "odcr.yaml"),
        "step3_stage_status": _tree_fingerprint(step3_dir / "meta" / "stage_status.json"),
        "step3_eval_handoff": _tree_fingerprint(step3_dir / "meta" / "eval_handoff.json"),
        "step3_selected_checkpoint": _tree_fingerprint(step3_dir / "model" / "best_observed.pth"),
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
            "checkpoint_path": str(Path(str(cfg.step3_checkpoint_dir)).resolve() / "model" / "best_observed.pth"),
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
]
