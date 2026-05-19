"""Validation-only Step4 GPU-shard preflight runner.

This module is launched by ``run_step4_bounded_preflight(..., preflight_mode=
"gpu-shard")`` under torchrun.  It writes only validation artifacts below the
caller-provided ``runs/step4_preflight`` namespace.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, TensorDataset

from odcr_core.file_atomic import atomic_write_json
from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    build_step4_export_lineage,
    step4_rcr_required_fields_hash,
)
from odcr_core.odcr_cf_routing import ODCFRoutingConfig, attach_odcr_cf_routing
from odcr_core.evidence_level import mark_gpu_shard_forward
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST
from odcr_core.training_checkpoint import file_fingerprint, stable_hash


SCHEMA_VERSION = "odcr_step4_gpu_shard_preflight/1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "peak": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "peak": float(np.max(arr)),
    }


class UtilizationMonitor:
    def __init__(self, *, interval_s: float = 0.5) -> None:
        self.interval_s = max(0.1, float(interval_s))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="step4-gpu-preflight-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def _run(self) -> None:
        try:
            import psutil
        except Exception:
            psutil = None
        while not self._stop.is_set():
            sample: dict[str, Any] = {"ts": time.time()}
            if psutil is not None:
                try:
                    sample["cpu_percent"] = float(psutil.cpu_percent(interval=None))
                    vm = psutil.virtual_memory()
                    sample["ram_used_gb"] = float(vm.used) / (1024**3)
                    sample["ram_percent"] = float(vm.percent)
                except Exception as exc:
                    sample["cpu_error"] = repr(exc)
            smi = shutil.which("nvidia-smi")
            if smi:
                try:
                    proc = subprocess.run(
                        [
                            smi,
                            "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=2,
                    )
                    gpus = []
                    if proc.returncode == 0:
                        for line in proc.stdout.splitlines():
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 3:
                                gpus.append(
                                    {
                                        "index": int(parts[0]),
                                        "util_percent": float(parts[1]),
                                        "mem_used_mb": float(parts[2]),
                                    }
                                )
                    sample["gpus"] = gpus
                except Exception as exc:
                    sample["nvidia_smi_error"] = repr(exc)
            self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        cpu = [float(s["cpu_percent"]) for s in self.samples if "cpu_percent" in s]
        ram = [float(s["ram_used_gb"]) for s in self.samples if "ram_used_gb" in s]
        gpu_values: dict[int, dict[str, list[float]]] = {}
        for sample in self.samples:
            for gpu in sample.get("gpus") or []:
                idx = int(gpu["index"])
                slot = gpu_values.setdefault(idx, {"util": [], "mem": []})
                slot["util"].append(float(gpu.get("util_percent") or 0.0))
                slot["mem"].append(float(gpu.get("mem_used_mb") or 0.0))
        gpu_summary = {
            str(idx): {
                "util_percent": _percentiles(values["util"]),
                "mem_used_mb": _percentiles(values["mem"]),
            }
            for idx, values in sorted(gpu_values.items())
        }
        return {
            "schema_version": "odcr_step4_gpu_cpu_utilization_monitor/1",
            "sample_count": len(self.samples),
            "cpu_percent": _percentiles(cpu),
            "ram_used_gb": _percentiles(ram),
            "gpus": gpu_summary,
        }


def _rcr_distribution(df: pd.DataFrame) -> dict[str, Any]:
    n = int(len(df))
    sw = pd.to_numeric(df["sample_weight_hint"], errors="coerce")
    bucket = pd.to_numeric(df["confidence_bucket"], errors="coerce").fillna(-1).astype(int)
    route_scorer = pd.to_numeric(df["route_scorer"], errors="coerce").fillna(0).astype(int)
    route_explainer = pd.to_numeric(df["route_explainer"], errors="coerce").fillna(0).astype(int)
    train_keep = pd.to_numeric(df["train_keep"], errors="coerce").fillna(0).astype(int)
    return {
        "sample_count": n,
        "route_scorer_count": int((route_scorer == 1).sum()),
        "route_explainer_count": int((route_explainer == 1).sum()),
        "train_keep_count": int((train_keep == 1).sum()),
        "route_scorer_ratio": float((route_scorer == 1).mean()) if n else 0.0,
        "route_explainer_ratio": float((route_explainer == 1).mean()) if n else 0.0,
        "train_keep_ratio": float((train_keep == 1).mean()) if n else 0.0,
        "neither_route_ratio": float(((route_scorer != 1) & (route_explainer != 1)).mean()) if n else 0.0,
        "confidence_bucket_distribution": {str(k): int(v) for k, v in bucket.value_counts().sort_index().items()},
        "sample_weight_hint": {
            "min": float(sw.min()) if len(sw) else None,
            "mean": float(sw.mean()) if len(sw) else None,
            "max": float(sw.max()) if len(sw) else None,
            "iqr": float(sw.quantile(0.75) - sw.quantile(0.25)) if len(sw) else None,
        },
    }


def _apply_train_keep_policy(df: pd.DataFrame, conf: ODCFRoutingConfig) -> pd.DataFrame:
    out = df.copy()
    scorer = pd.to_numeric(out["route_scorer"], errors="coerce").fillna(0).astype(int)
    explainer = pd.to_numeric(out["route_explainer"], errors="coerce").fillna(0).astype(int)
    keep = pd.to_numeric(out.get("train_keep", pd.Series([1] * len(out))), errors="coerce").fillna(1).astype(int)
    weight = pd.to_numeric(out.get("sample_weight_hint", pd.Series([1.0] * len(out))), errors="coerce").fillna(1.0)
    reject = (scorer == 0) & (explainer == 0) & bool(conf.train_keep.get("reject_when_both_routes_zero", True))
    keep.loc[reject] = 0
    weight.loc[reject] = 0.0
    if "train_drop_reason" not in out.columns:
        out["train_drop_reason"] = ""
    out.loc[reject, "train_drop_reason"] = str(conf.train_keep.get("reject_reason", "rcr_route_reject"))
    active = ~reject & (keep == 1)
    if active.any():
        sw_cfg = conf.sample_weight_hint
        route_mult = np.where(
            scorer.loc[active].to_numpy(dtype=int) == 1,
            float(sw_cfg["scorer_route_multiplier"]),
            float(sw_cfg["explainer_only_route_multiplier"]),
        )
        rel = pd.to_numeric(out.loc[active, "cf_reliability_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        unc = pd.to_numeric(out.loc[active, "uncertainty_score"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        posterior_mult = float(sw_cfg["reliability_floor"]) + float(sw_cfg["reliability_scale"]) * rel
        uncertainty_mult = float(sw_cfg["uncertainty_base"]) + float(sw_cfg["uncertainty_scale"]) * (1.0 - unc)
        weight.loc[active] = np.round(weight.loc[active].to_numpy(dtype=float) * route_mult * posterior_mult * uncertainty_mult, 6)
    out["train_keep"] = keep.astype(int)
    out["sample_weight_hint"] = weight.astype(float)
    return out


def _build_model(*, payload: Mapping[str, Any], device: str) -> torch.nn.Module:
    from executors.step3_train_core import Model, get_odcr_text_tokenizer
    from odcr_core.index_contract import load_profile_tensors_dual_first
    from odcr_core.training_checkpoint import (
        CheckpointLineageError,
        extract_checkpoint_model_architecture_payload,
        extract_checkpoint_state_dict_architecture_payload,
        read_checkpoint_lineage,
    )
    from config import get_odcr_embed_dim
    from paths_config import get_data_dir

    dc, ds, uc, us, ic, ist, _profile_meta = load_profile_tensors_dual_first(
        data_root=get_data_dir(),
        auxiliary_domain=str(payload["source"]),
        target_domain=str(payload["target"]),
        device_idx=device,
    )
    emsize = int(uc.shape[-1])
    if emsize != int(get_odcr_embed_dim()):
        raise RuntimeError(f"embed dim mismatch: profile={emsize} resolved={get_odcr_embed_dim()}")
    checkpoint_lineage = read_checkpoint_lineage(str(payload["checkpoint_path"]), expected_stage="step3")
    sidecar_arch = extract_checkpoint_model_architecture_payload(checkpoint_lineage)
    frozen_arch = extract_checkpoint_state_dict_architecture_payload(
        str(payload["checkpoint_path"]),
        fallback_payload=sidecar_arch,
    )
    hard_mismatches = []
    for key, observed in (("nuser", int(uc.shape[0])), ("nitem", int(ic.shape[0])), ("emsize", emsize)):
        if int(frozen_arch[key]) != int(observed):
            hard_mismatches.append(f"{key} checkpoint={frozen_arch[key]!r} observed={observed!r}")
    if hard_mismatches:
        raise CheckpointLineageError(
            "Step4 gpu-shard preflight refused Step3 checkpoint architecture: "
            + ", ".join(hard_mismatches)
        )
    model = Model(
        int(frozen_arch["nuser"]),
        int(frozen_arch["nitem"]),
        int(frozen_arch["ntoken"]),
        int(frozen_arch["emsize"]),
        int(frozen_arch["nhead"]),
        int(frozen_arch["nhid"]),
        int(frozen_arch["nlayers"]),
        float(frozen_arch["dropout"]),
        uc,
        us,
        ic,
        ist,
        dc,
        ds,
    ).to(device)
    raw_decode = (os.environ.get("ODCR_DECODE_PROFILE_JSON") or "").strip()
    if raw_decode:
        try:
            dp = json.loads(raw_decode)
        except json.JSONDecodeError:
            dp = {}
        if isinstance(dp, Mapping):
            model.decode_strategy = str(dp.get("decode_strategy", "greedy")).strip().lower()
            model.generate_temperature = float(dp.get("generate_temperature", 0.8))
            model.generate_top_p = float(dp.get("generate_top_p", 0.9))
            model.repetition_penalty = float(dp.get("repetition_penalty", 1.15))
            model.max_explanation_length = int(dp.get("max_explanation_length", 25))
            model.no_repeat_ngram_size = int(dp.get("no_repeat_ngram_size") or 0)
            model.min_len = int(dp.get("min_len") or 0)
            model.soft_max_len = int(dp.get("soft_max_len") or 0)
            model.hard_max_len = int(dp.get("hard_max_len") or model.max_explanation_length)
            model.eos_boost_start = int(dp.get("eos_boost_start", 9999))
            model.eos_boost_value = float(dp.get("eos_boost_value", 0.0))
            model.tail_temperature = float(dp.get("tail_temperature", -1.0))
            model.tail_top_p = float(dp.get("tail_top_p", -1.0))
            model.decode_token_repeat_window = int(dp.get("decode_token_repeat_window", 4))
            model.decode_token_repeat_max = int(dp.get("decode_token_repeat_max", 2))
            model.domain_fusion_mode = str(dp.get("domain_fusion_mode", "gate_cross_attn")).strip().lower()
            model.decoder_eos_id = int(getattr(get_odcr_text_tokenizer(), "eos_token_id", -1) or -1)
    return model


def _load_encoded_dataset(path: str):
    from datasets import load_from_disk

    encoded = load_from_disk(path)
    encoded.set_format("torch")
    return encoded


def _make_tensor_dataset(encoded_data) -> TensorDataset:
    return TensorDataset(
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


def _write_rank_summary(out_dir: Path, rank: int, payload: Mapping[str, Any]) -> None:
    rank_dir = out_dir / "rank_artifacts"
    rank_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(rank_dir / f"rank{rank}_summary.json", dict(payload))


def _run_rank(payload: dict[str, Any]) -> None:
    from base_utils import get_underlying_model
    from config import get_dataloader_num_workers, get_dataloader_prefetch_factor
    from executors.step4_engine import _decode_pred_token_rows, _step4_rcr_latent_diagnostics
    from odcr_core.gather_schema import require_gathered_batch
    from train_diagnostics import odcr_cuda_bf16_autocast

    if not torch.cuda.is_available():
        raise RuntimeError("Step4 gpu-shard preflight requires CUDA.")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    out_dir = Path(str(payload["out_dir"])).resolve()
    rank_dir = out_dir / "rank_artifacts"
    rank_dir.mkdir(parents=True, exist_ok=True)
    monitor = UtilizationMonitor(interval_s=0.5) if rank == 0 and bool(payload.get("profile_utilization")) else None
    if monitor is not None:
        monitor.start()
    wall0 = time.perf_counter()
    model_load0 = time.perf_counter()
    model = _build_model(payload=payload, device=device)
    checkpoint_path = str(payload["checkpoint_path"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model_load_s = time.perf_counter() - model_load0
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    model.eval()
    underlying = get_underlying_model(model)
    encoded = _load_encoded_dataset(str(payload["encoded_dataset_dir"]))
    n_samples = min(int(payload["sample_count"]), len(encoded))
    chunk = int(math.ceil(n_samples / float(world_size)))
    start = rank * chunk
    end = min(start + chunk, n_samples)
    indices = list(range(start, end))
    dataset = Subset(_make_tensor_dataset(encoded), indices)
    local_batch = int(payload.get("eval_per_gpu_batch_size") or 0)
    if local_batch <= 0:
        local_batch = max(1, int(payload.get("global_eval_batch_size") or 1) // max(1, world_size))
    workers = int(get_dataloader_num_workers("test"))
    prefetch = get_dataloader_prefetch_factor(workers, split="test")
    dataloader = DataLoader(
        dataset,
        batch_size=local_batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch if workers > 0 else None,
    )
    local_row_indices: list[int] = []
    local_pred_token_rows: list[list[int]] = []
    local_entropy_values: list[float] = []
    local_rating_target: list[float] = []
    local_rating_counterfactual: list[float] = []
    local_rating_delta: list[float] = []
    local_rating_stability: list[float] = []
    local_shared_latent_similarity: list[float] = []
    local_specific_latent_shift: list[float] = []
    h2d_s = 0.0
    forward_s = 0.0
    batches = 0
    first_batch_wait_s: float | None = None
    prev_end = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(local_rank)
    gpu0 = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.perf_counter()
            if first_batch_wait_s is None:
                first_batch_wait_s = batch_start - prev_end
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
            t_h2d = time.perf_counter()
            gb = require_gathered_batch(
                underlying.gather(
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
            h2d_s += time.perf_counter() - t_h2d
            t_fw = time.perf_counter()
            with odcr_cuda_bf16_autocast():
                pred_exps, entropy = underlying.generate(
                    gb.user_idx,
                    gb.item_idx,
                    gb.domain_idx,
                    content_anchor=gb.content_anchor_score,
                    style_anchor=gb.style_anchor_score,
                    content_evidence_ids=gb.content_evidence_ids,
                    style_evidence_ids=gb.style_evidence_ids,
                    domain_style_anchor_ids=gb.domain_style_anchor_ids,
                    local_style_hint_ids=gb.local_style_hint_ids,
                    polarity_ids=gb.polarity_ids,
                    evidence_quality_prior=gb.evidence_quality_prior,
                )
                rcr_diag = _step4_rcr_latent_diagnostics(
                    underlying,
                    user_idx=gb.user_idx,
                    item_idx=gb.item_idx,
                    content_anchor=gb.content_anchor_score,
                    style_anchor=gb.style_anchor_score,
                    content_evidence_ids=gb.content_evidence_ids,
                    style_evidence_ids=gb.style_evidence_ids,
                    domain_style_anchor_ids=gb.domain_style_anchor_ids,
                    local_style_hint_ids=gb.local_style_hint_ids,
                    polarity_ids=gb.polarity_ids,
                    evidence_quality_prior=gb.evidence_quality_prior,
                )
            torch.cuda.synchronize(local_rank)
            forward_s += time.perf_counter() - t_fw
            local_row_indices.extend(batch_sample_id.detach().cpu().reshape(-1).tolist())
            local_pred_token_rows.extend(pred_exps.detach().cpu().tolist())
            local_entropy_values.extend(entropy.detach().cpu().numpy().tolist())
            local_rating_target.extend(rcr_diag["rating_target"])
            local_rating_counterfactual.extend(rcr_diag["rating_counterfactual"])
            local_rating_delta.extend(rcr_diag["rating_delta"])
            local_rating_stability.extend(rcr_diag["rating_stability_score"])
            local_shared_latent_similarity.extend(rcr_diag["shared_latent_similarity"])
            local_specific_latent_shift.extend(rcr_diag["specific_latent_shift"])
            batches += 1
            prev_end = time.perf_counter()
    gpu_phase_s = time.perf_counter() - gpu0
    decode0 = time.perf_counter()
    local_explanations = _decode_pred_token_rows(local_pred_token_rows)
    decode_s = time.perf_counter() - decode0
    partial_df = pd.DataFrame(
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
    partial_path = rank_dir / f"rank{rank}_partial.csv"
    write0 = time.perf_counter()
    partial_df.to_csv(partial_path, index=False, encoding="utf-8")
    partial_write_s = time.perf_counter() - write0
    mem_peak_mb = float(torch.cuda.max_memory_allocated(local_rank) / (1024**2))
    rank_summary = {
        "schema_version": "odcr_step4_gpu_shard_rank_summary/1",
        "rank": int(rank),
        "world_size": int(world_size),
        "local_rank": int(local_rank),
        "device": device,
        "device_name": torch.cuda.get_device_name(local_rank),
        "model_loaded_on_gpu": next(underlying.parameters()).is_cuda,
        "checkpoint_loaded": True,
        "checkpoint_path": checkpoint_path,
        "sample_count": int(len(local_row_indices)),
        "shard_start": int(start),
        "shard_end": int(end),
        "batches": int(batches),
        "model_load_seconds": model_load_s,
        "gpu_phase_seconds": gpu_phase_s,
        "h2d_seconds": h2d_s,
        "forward_seconds": forward_s,
        "decode_seconds": decode_s,
        "partial_write_seconds": partial_write_s,
        "runtime_seconds": time.perf_counter() - wall0,
        "rows_per_sec_gpu_phase": float(len(local_row_indices) / gpu_phase_s) if gpu_phase_s > 0 else 0.0,
        "gpu_mem_peak_mb": mem_peak_mb,
        "first_batch_wait_seconds": first_batch_wait_s,
        "partial_path": str(partial_path),
    }
    _write_rank_summary(out_dir, rank, rank_summary)
    dist.barrier()
    pg_destroy0 = time.perf_counter()
    dist.destroy_process_group()
    pg_destroy_s = time.perf_counter() - pg_destroy0
    rank_summary["process_group_destroyed"] = True
    rank_summary["process_group_destroy_seconds"] = pg_destroy_s
    _write_rank_summary(out_dir, rank, rank_summary)
    if rank != 0:
        return
    export0 = time.perf_counter()
    target_df = pd.read_csv(str(payload["target_subset_csv"]))
    partials = []
    rank_summaries = []
    for r in range(world_size):
        partials.append(pd.read_csv(rank_dir / f"rank{r}_partial.csv"))
        rank_summaries.append(_load_json(rank_dir / f"rank{r}_summary.json"))
    merged = pd.concat(partials, ignore_index=True).sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if len(merged) != n_samples:
        raise RuntimeError(f"merged partial row count mismatch: {len(merged)} != {n_samples}")
    if not np.array_equal(merged["row_idx"].to_numpy(dtype=np.int64), np.arange(n_samples, dtype=np.int64)):
        raise RuntimeError("merged partial row_idx does not cover the bounded shard")
    cf_df = target_df.copy().reset_index(drop=True)
    cf_df["explanation"] = merged["explanation"].astype(str).to_numpy(copy=False)
    cf_df["entropy"] = merged["entropy"].astype(float).to_numpy(copy=False)
    for col in (
        "rating_target",
        "rating_counterfactual",
        "rating_delta",
        "rating_stability_score",
        "shared_latent_similarity",
        "specific_latent_shift",
    ):
        cf_df[col] = merged[col].astype(float).to_numpy(copy=False)
    rcr_config = ODCFRoutingConfig.from_mapping(payload["rcr_config"])
    routed = _apply_train_keep_policy(attach_odcr_cf_routing(target_df, cf_df, cfg=rcr_config), rcr_config)
    dist_payload = mark_gpu_shard_forward(_rcr_distribution(routed))
    required_fields = list((rcr_config.export or {}).get("required_fields") or [])
    missing_required = [name for name in required_fields if name not in routed.columns]
    required_check = mark_gpu_shard_forward({
        "schema_version": "odcr_step4_required_fields_check/1",
        "passed": not missing_required,
        "missing": missing_required,
        "required_fields": required_fields,
        "presence_ratio": 1.0 if not missing_required else 0.0,
    })
    step5_required_check = mark_gpu_shard_forward({
        "schema_version": "odcr_step4_step5_required_fields_check/1",
        "passed": not missing_required,
        "missing": missing_required,
        "required_fields": required_fields,
    })
    ckpt_fp = file_fingerprint(checkpoint_path)
    lineage = build_step4_export_lineage(
        task_id=int(payload["task"]),
        auxiliary_domain=str(payload["source"]),
        target_domain=str(payload["target"]),
        step3_checkpoint_lineage_hash=str(_load_json(Path(checkpoint_path + ".lineage.json")).get("lineage_hash") or ""),
        step4_rcr_config=rcr_config.to_dict(),
        step4_run=str(payload["validation_namespace"]),
        frozen_step3_lineage={
            "upstream_step3_run_id": str(payload.get("step3_run_id") or ""),
            "step3_checkpoint_path": checkpoint_path,
            "step3_checkpoint_hash": str(ckpt_fp.get("sha256") or ""),
            "step3_checkpoint_lineage_hash": str(_load_json(Path(checkpoint_path + ".lineage.json")).get("lineage_hash") or ""),
        },
    )
    manifest_preview = mark_gpu_shard_forward({
        "schema_version": "odcr_step4_manifest_preview/1",
        "export_manifest_name": STEP4_EXPORT_MANIFEST,
        "row_count": int(len(routed)),
        "rcr_required_fields_hash": step4_rcr_required_fields_hash(),
        "step4_export_lineage": lineage,
        "formal_export": False,
    })
    index_preview = mark_gpu_shard_forward({
        "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
        "file": INDEX_CONTRACT_FILENAME,
        "task_id": int(payload["task"]),
        "step4_run": str(payload["validation_namespace"]),
        "step4_export_lineage": lineage,
        "formal_export": False,
    })
    cpu_export_s = time.perf_counter() - export0
    monitor_summary = monitor.stop() if monitor is not None else UtilizationMonitor().summary()
    sample_counts = [int(item.get("sample_count", 0)) for item in rank_summaries]
    runtimes = [float(item.get("gpu_phase_seconds", 0.0) or 0.0) for item in rank_summaries]
    rank_imbalance = (max(sample_counts) - min(sample_counts)) / max(1, max(sample_counts)) if sample_counts else 1.0
    runtime_imbalance = (max(runtimes) - min(runtimes)) / max(1e-9, max(runtimes)) if runtimes else 1.0
    total_wall_s = time.perf_counter() - wall0
    gpu_phase_s_max = max(runtimes) if runtimes else 0.0
    per_rank = {
        "schema_version": "odcr_step4_gpu_shard_per_rank_summary/1",
        "world_size": int(world_size),
        "ranks": rank_summaries,
        "rank_sample_imbalance_ratio": float(rank_imbalance),
        "rank_runtime_imbalance_ratio": float(runtime_imbalance),
    }
    timing = {
        "schema_version": "odcr_step4_gpu_shard_timing_breakdown/1",
        "total_wall_seconds": total_wall_s,
        "gpu_shard_phase_seconds": gpu_phase_s_max,
        "cpu_export_phase_seconds": cpu_export_s,
        "cpu_export_after_pg_destroy": True,
        "rank_gpu_phase_seconds": runtimes,
        "rank_decode_seconds": [float(item.get("decode_seconds", 0.0) or 0.0) for item in rank_summaries],
        "rank_partial_write_seconds": [float(item.get("partial_write_seconds", 0.0) or 0.0) for item in rank_summaries],
    }
    proof = mark_gpu_shard_forward({
        "schema_version": "odcr_step4_gpu_shard_path_proof/1",
        "preflight_mode": "gpu-shard",
        "force_gpu_forward": True,
        "actual_gpu_forward_executed": bool(sum(sample_counts) > 0 and all(item.get("batches", 0) > 0 for item in rank_summaries)),
        "actual_model_loaded_on_gpu": bool(all(item.get("model_loaded_on_gpu") for item in rank_summaries)),
        "two_gpus_visible": torch.cuda.device_count() >= 2,
        "two_gpus_used": bool(world_size >= 2 and len({int(item.get("local_rank", -1)) for item in rank_summaries}) >= 2),
        "world_size": int(world_size),
        "per_rank_shards_balanced": rank_imbalance < 0.10,
        "rank0_rank1_runtime_balanced": runtime_imbalance < 0.20,
        "process_group_destroyed_before_cpu_export": True,
        "formal_latest_write": False,
        "formal_export_write": False,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": str(ckpt_fp.get("sha256") or ""),
        "candidate": payload.get("candidate"),
    })
    gpu_snapshot = mark_gpu_shard_forward({
        "schema_version": "odcr_step4_cpu_gpu_utilization_snapshot/2",
        "cuda_available": True,
        "device_count": int(torch.cuda.device_count()),
        "gpu_runtime_evidence": True,
        "runtime_mode": "gpu_shard_forward",
        "monitor": monitor_summary,
        "rank_gpu_memory_peak_mb": {
            str(item["rank"]): float(item.get("gpu_mem_peak_mb", 0.0) or 0.0) for item in rank_summaries
        },
        "cpu_util_mean": float(monitor_summary.get("cpu_percent", {}).get("mean", 0.0)),
        "cpu_util_peak": float(monitor_summary.get("cpu_percent", {}).get("peak", 0.0)),
        "ram_peak_gb": float(monitor_summary.get("ram_used_gb", {}).get("peak", 0.0)),
    })
    summary = mark_gpu_shard_forward({
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "task": int(payload["task"]),
        "validation_namespace": str(payload["validation_namespace"]),
        "output_dir": str(out_dir),
        "max_samples": int(payload["max_samples"]),
        "sample_count": int(n_samples),
        "uses_real_task_data": True,
        "uses_real_run2_checkpoint": str(payload.get("step3_run_id")) == "2",
        "upstream_step3_run_id": str(payload.get("step3_run_id") or ""),
        "preflight_mode": "gpu-shard",
        "candidate": payload.get("candidate"),
        "checkpoint": {
            "checkpoint_path": checkpoint_path,
            "checkpoint_hash": str(ckpt_fp.get("sha256") or ""),
        },
        "rcr_distribution": dist_payload,
        "confidence_bucket_distribution": dist_payload["confidence_bucket_distribution"],
        "sample_weight_hint_stats": dist_payload["sample_weight_hint"],
        "required_fields_check": required_check,
        "step5_required_fields_check": step5_required_check,
        "formal_latest_write": False,
        "formal_export_write": False,
        "gpu_runtime_evidence": True,
        "actual_gpu_forward_executed": proof["actual_gpu_forward_executed"],
        "actual_model_loaded_on_gpu": proof["actual_model_loaded_on_gpu"],
        "two_gpus_used": proof["two_gpus_used"],
        "rows_per_sec": float(n_samples / total_wall_s) if total_wall_s > 0 else 0.0,
        "gpu_phase_rows_per_sec": float(n_samples / gpu_phase_s_max) if gpu_phase_s_max > 0 else 0.0,
        "cpu_export_after_pg_destroy": True,
    })
    atomic_write_json(out_dir / "rcr_distribution.json", dist_payload)
    atomic_write_json(out_dir / "required_fields_check.json", required_check)
    atomic_write_json(out_dir / "step5_required_fields_check.json", step5_required_check)
    atomic_write_json(out_dir / "manifest_preview.json", manifest_preview)
    atomic_write_json(out_dir / "index_contract_preview.json", index_preview)
    atomic_write_json(out_dir / "lineage_preview.json", lineage)
    atomic_write_json(out_dir / "cpu_gpu_utilization_snapshot.json", gpu_snapshot)
    atomic_write_json(out_dir / "gpu_shard_path_proof.json", proof)
    atomic_write_json(out_dir / "per_rank_summary.json", per_rank)
    atomic_write_json(out_dir / "timing_breakdown.json", timing)
    atomic_write_json(out_dir / "preflight_summary.json", summary)
    routed.head(min(256, len(routed))).to_csv(out_dir / "routed_preview_head.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="ODCR Step4 validation-only GPU-shard preflight runner")
    parser.add_argument("--payload", required=True)
    args = parser.parse_args()
    payload = _load_json(Path(args.payload).resolve())
    _run_rank(payload)


if __name__ == "__main__":
    main()
