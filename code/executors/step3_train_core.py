# -*- coding: utf-8 -*-
"""
INTERNAL EXECUTOR — Step3 shared/specific 结构化解耦训练（非用户首选入口）。

- MAINLINE ENTRY：``python code/odcr.py step3|eval|…``（仓库根目录）。
- 本文件由 ``odcr.py`` 经 torchrun 分发给 **step3 runner** 调用；勿作为日常手工入口。
- Step4 引擎 ``executors.step4_engine`` 自本模块 ``import *`` 复用符号。

训练语义：shared/content 与 specific/style 两条物理主线先独立聚合，
再通过 evidence-guided disentangler 产生结构化 latent；训练期仅保留
rating/explainer + orthogonal + invariance + separation + evidence/prototype geometry 主图。
"""
import os
import sys
import copy
import json
import time
import hashlib
import shutil
import warnings
import argparse
import logging
import math
import subprocess
import uuid
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

from paths_config import get_nltk_data_dir

# 以下为「环境侧」离线偏好；**不能**替代各 `from_pretrained(..., local_files_only=True)` 与 `require_*` 目录校验。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")
_ROOT = os.path.dirname(os.path.abspath(__file__))
# 在 import base_utils（会 import nltk）之前指定离线 NLTK 语料，避免 METEOR 触发 nltk.download
_nltk_data = get_nltk_data_dir()
if os.path.isdir(_nltk_data):
    os.environ.setdefault("NLTK_DATA", os.path.abspath(_nltk_data))
sys.path.insert(0, _ROOT)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*no current CUDA context.*", category=UserWarning)

from base_utils import *
from paths_config import (
    get_data_dir,
    get_merged_data_dir,
    get_stage_run_dir,
    get_sentence_embed_model_dir,
    require_step5_text_model_dir,
)
from odcr_core import path_layout
from odcr_core.runtime_env_pack import runtime_env_dict_for_config_resolved
from odcr_core.config_schema import SAFE_DECODE_PLACEHOLDER
from odcr_core.manifests import write_training_runtime_config_artifact
import torch

# transformers 在 modeling_utils.load_state_dict 里用 torch.load(..., map_location=...) 未传 weights_only，
# PyTorch 2.4+ 会 FutureWarning。在 from_pretrained 前默认 weights_only=True，与官方推荐一致。
def _patch_torch_load_default_weights_only() -> None:
    _orig = torch.load

    def _wrapped(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = True
        try:
            return _orig(*args, **kwargs)
        except TypeError as e:
            if "weights_only" not in str(e).lower():
                raise
            kwargs.pop("weights_only", None)
            return _orig(*args, **kwargs)

    torch.load = _wrapped  # type: ignore[assignment]


_patch_torch_load_default_weights_only()

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch import optim
from torch.optim import lr_scheduler as lr_sched
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.distributed as dist
from transformers import T5Tokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from perf_monitor import PerfMonitor, gather_ddp_gpu_stats_for_epoch_log
from datasets import Dataset, DatasetDict, load_from_disk
from config import (
    FinalTrainingConfig,
    apply_ddp_fast_torch_backends,
    build_resolved_training_config,
    get_eval_batch_size,
    get_dataloader_num_workers,
    get_dataloader_prefetch_factor,
    get_num_proc,
    get_odcr_embed_dim,
    hf_datasets_progress_bar,
    resolve_task_idx_from_aux_target,
)
from data_contract import CANONICAL_PREPROCESS_ASSET_COLUMNS, PREPROCESS_CONTRACT_VERSION
from training_hardware_inputs import collect_training_hardware_overrides_from_args
from odcr_core.training_diagnostics import training_diagnostics_snapshot
from odcr_core.file_atomic import atomic_torch_save, atomic_write_json
from odcr_core.training_checkpoint import (
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    STEP3_CHECKPOINT_MIN_ACCEPTED_SCHEMA_VERSION,
    current_effective_payload,
    current_resolved_config_lineage,
    current_one_control_resolved_config_hash,
    current_training_runtime_config_lineage,
    current_source_table_lineage,
    file_fingerprint,
    model_artifact_fingerprint,
    checkpoint_file_sha256,
    step3_resolved_config_compatibility_payload,
    step3_source_table_compatibility_payload,
    stable_hash,
    state_dict_for_canonical_best_pth,
    write_checkpoint_lineage,
)
from odcr_core.step3_quality import (
    STEP3_QUALITY_GATE_VERSION,
    build_best_event_payload,
    build_step3_quality_audit,
    checkpoint_event_from_sidecar,
    checkpoint_filename_for_metric,
    checkpoint_sidecar_payload,
    collapse_stats_from_predictions,
    diagnostic_sample_record,
    inspect_gradients,
    metric_improved,
    sync_grad_finite_decision,
    timing_row_with_closure,
    write_step3_quality_audit,
)
from odcr_core.step3_eval_protocol import (
    FULL_PIPELINE_FINAL_EVAL,
    MINIMAL_EVAL,
    ODCR_STEP3_DIAGNOSTIC,
    PAPER_TARGET_ONLY_EVAL,
    PREDICTION_SHARD_REQUIRED_FIELDS,
    STEP3_BATCH_INVARIANCE_SCHEMA_VERSION,
    STEP3_EVAL_PROTOCOL_SCHEMA_VERSION,
    STEP3_PAPER_METRIC_IMPLEMENTATION_VERSION,
    STEP3_PREDICTION_SHARD_SCHEMA_VERSION,
    STEP3_TRAINING_EFFECTIVENESS_SCHEMA_VERSION,
    build_training_effectiveness_record,
    compare_eval_batch_outputs,
    explain_lr_floor,
    metrics_from_prediction_rows,
    normalize_eval_protocol,
    sample_integrity_report,
    scheduler_semantics,
    select_largest_safe_eval_batch,
    sort_prediction_rows,
    stable_step3_sample_id,
    step3_eval_protocol_spec,
    summarize_loss_component_rows,
)
from odcr_core.step3_upstream_gate import validate_step3_preprocess_upstream_gate
from lr_schedule_utils import resolve_warmup_steps, warmup_cosine_multiplier_lambda
from odcr_core.gather_schema import GatheredBatch, require_gathered_batch
from executors.decode_controller import (
    GenerateConfig,
    apply_eos_boost,
    apply_min_len_eos_mask,
    apply_no_repeat_ngram_logits,
    apply_repetition_penalty_logits,
    apply_sampling_schedule,
    apply_token_repeat_suppression,
    prepare_logits,
    sample_next_token,
)
from odcr_core.odcr_representation import ODCRDisentangler
from odcr_core.odcr_losses import (
    anchor_score_alignment_loss,
    build_orthogonal_losses,
    cosine_pull_loss,
    domain_style_prototype_separation,
    residual_l2_penalty,
    shared_prototype_pull_loss,
    shared_invariance_loss,
    specific_separation_loss,
    variance_floor_loss,
)
from odcr_core.index_contract import load_profile_tensors_dual_first
from train_logging import (
    setup_train_logging,
    log_run_header,
    log_config_snapshot,
    flush_preset_load_events,
    format_epoch_training_block,
    log_epoch_training_block,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    append_eval_run_summaries,
    LOGGER_NAME,
    logger_has_file_handler,
    log_route_extra,
    ROUTE_DETAIL,
    ROUTE_SUMMARY,
)
from train_diagnostics import (
    collect_distributed_env_for_meta,
    odcr_cuda_bf16_autocast,
    odcr_cuda_bf16_autocast_enabled,
    odcr_ddp_epoch_end_barrier,
    odcr_grad_topk,
    odcr_log_grad_interval,
    odcr_log_step_interval,
    odcr_log_step_loss_parts,
    odcr_save_checkpoint,
    odcr_timing_phase,
    ddp_heartbeat,
    grad_norm_total,
    log_bf16_amp_note,
    log_grad_monitor,
    log_step_sample,
    log_training_crash,
    maybe_log_grad_norm_diff_ddp,
    parse_odcr_finite_check_mode,
    run_training_finite_checks,
    warn_empty_batch,
)
from train_logging import (
    append_step3_damping_events_jsonl,
    append_step3_epoch_summary_csv,
    append_step3_gpu_profile_jsonl,
    append_step3_loss_breakdown_jsonl,
    append_step3_objective_drift_jsonl,
    append_step3_recovery_events_jsonl,
    append_step3_samples_jsonl,
    append_step3_scheduler_events_jsonl,
    append_step3_timing_profile_jsonl,
    append_step3_training_effectiveness_jsonl,
    append_train_epoch_metrics_jsonl,
    write_step3_component_contribution_summary_md,
    write_step3_collapse_stats_json,
    write_step3_loss_component_epoch_summary_csv,
    write_step3_loss_component_trends_json,
    write_step3_training_effectiveness_summary_json,
)
from odcr_core.step3_v3_policy import (
    build_recovery_plan,
    detect_objective_drift,
    resolve_phase_for_epoch,
    safe_damping_v2_decision,
)

def _nonfinite_loss_abort_threshold() -> int:
    raw = os.environ.get("ODCR_NONFINITE_LOSS_ABORT_AFTER", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


STEP3_TOTAL_LOSS_COMPONENT_KEYS: tuple[str, ...] = (
    "L_rating_shared",
    "L_light_explainer",
    "L_orthogonal",
    "L_variance",
    "L_shared_invariance",
    "L_specific_separation",
    "L_anchor_content",
    "L_anchor_style",
    "L_content_alignment",
    "L_style_alignment",
    "L_shared_proto",
    "L_domain_style_alignment",
    "L_local_style_alignment",
    "L_polarity_alignment",
    "L_residual_specific",
    "L_prototype_separation",
)

STEP3_DIAGNOSTIC_LOSS_KEYS: tuple[str, ...] = (
    "L_orthogonal_xcov",
    "L_orthogonal_cosine",
    "L_shared_variance",
    "L_specific_variance",
)

STEP3_STRUCTURED_LOSS_INPUT_KEYS: tuple[str, ...] = (
    "shared_latent",
    "specific_latent",
    "shared_prototype",
    "domain_style_proto",
    "domain_style_component",
    "content_evidence_target",
    "style_evidence_target",
    "domain_style_target",
    "local_style_target",
    "polarity_target",
    "residual_local",
    "anchor_pred_content",
    "anchor_pred_style",
)


class Step3ForwardOutput(NamedTuple):
    rating: torch.Tensor
    context_dist: torch.Tensor
    word_dist: torch.Tensor
    shared_proj: torch.Tensor
    specific_proj: torch.Tensor
    odcr_latents: Any
    structured_loss_inputs: dict[str, torch.Tensor]
    diagnostics: dict[str, Any]

    def legacy_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.rating, self.context_dist, self.word_dist, self.shared_proj, self.specific_proj)


@dataclass(frozen=True)
class Step3LossSemantics:
    specific_separation_margin: float
    variance_target_std: float
    variance_eps: float
    orthogonal_eps: float
    cosine_eps: float
    sample_weight_eps: float
    quality_evidence_base: float
    quality_evidence_scale: float
    quality_anchor_base: float
    quality_anchor_scale: float
    prototype_separation_eps: float


@dataclass(frozen=True)
class Step3LossBundle:
    total_loss: torch.Tensor
    components: dict[str, torch.Tensor]
    weights: dict[str, float]
    weighted_components: dict[str, torch.Tensor]
    participates_in_total: dict[str, bool]
    finite_status: dict[str, bool]
    graph_tied_zero_status: dict[str, bool]
    duplicate_loss_check_summary: dict[str, Any]
    logging_summary: dict[str, Any]
    diagnostics: dict[str, torch.Tensor]


_STEP3_CROSS_RANK_ALLOWED_CONTEXT_KEYS = frozenset(
    {
        "shared_repr",
        "specific_repr",
        "domain_ids",
        "style_ids",
        "structured_masks",
        "quality_weights",
    }
)
_STEP3_CROSS_RANK_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "raw_text",
        "token_ids",
        "profile_matrices",
        "domain_profile_matrices",
        "full_profile_matrices",
        "large_vocab_logits",
        "raw_vocab_logits",
        "item_user_full_profile_matrix",
        "checkpoint_state_tensors",
    }
)


def _parse_step3_cross_rank_gather_config(final_cfg: FinalTrainingConfig) -> dict[str, Any]:
    raw = str(getattr(final_cfg, "cross_rank_structured_gather_config_json", "") or "{}")
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Step3 cross-rank gather config is invalid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise RuntimeError("Step3 cross-rank gather config must be a JSON object.")
    enabled = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode") or "local_gradient_context")
    if enabled and mode != "local_gradient_context":
        raise RuntimeError("Step3 cross-rank gather currently supports only local_gradient_context.")
    forbidden = set(str(x) for x in cfg.get("forbidden_tensors") or [])
    if enabled and not _STEP3_CROSS_RANK_FORBIDDEN_CONTEXT_KEYS.issubset(forbidden):
        raise RuntimeError("Step3 cross-rank gather config must explicitly forbid raw/token/profile/logit tensors.")
    if not enabled:
        cfg.setdefault("mode", "local_gradient_context")
        cfg.setdefault("allowed_tensors", sorted(_STEP3_CROSS_RANK_ALLOWED_CONTEXT_KEYS))
        cfg.setdefault("forbidden_tensors", sorted(_STEP3_CROSS_RANK_FORBIDDEN_CONTEXT_KEYS))
    return cfg


def _all_gather_local_gradient_tensor(tensor: torch.Tensor, *, world_size: int, rank: int) -> torch.Tensor:
    if world_size <= 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[int(rank)] = tensor
    return torch.cat(gathered, dim=0)


def gather_step3_structured_context_local_gradient(
    *,
    shared_repr: torch.Tensor,
    specific_repr: torch.Tensor,
    domain_ids: torch.Tensor,
    quality_weights: torch.Tensor | None = None,
    world_size: int,
    rank: int,
    requested_keys: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    requested = set(str(x) for x in (requested_keys or _STEP3_CROSS_RANK_ALLOWED_CONTEXT_KEYS))
    forbidden_requested = sorted(requested & _STEP3_CROSS_RANK_FORBIDDEN_CONTEXT_KEYS)
    if forbidden_requested:
        raise RuntimeError(f"Step3 cross-rank gather forbids raw/profile tensors: {forbidden_requested}")
    if not {"shared_repr", "specific_repr", "domain_ids"}.issubset(requested):
        raise RuntimeError("Step3 cross-rank gather requires shared_repr, specific_repr, and domain_ids.")
    if shared_repr.ndim != 2 or specific_repr.ndim != 2:
        raise RuntimeError("Step3 cross-rank gather requires compact [B,H] shared/specific tensors.")
    if shared_repr.shape != specific_repr.shape:
        raise RuntimeError("Step3 cross-rank gather shared/specific shapes must match.")
    local_per_gpu = int(shared_repr.shape[0])
    if local_per_gpu != int(domain_ids.reshape(-1).shape[0]):
        raise RuntimeError("Step3 cross-rank gather domain_ids length must match local per-GPU batch.")
    t0 = time.perf_counter()
    context = {
        "shared_repr": _all_gather_local_gradient_tensor(shared_repr, world_size=world_size, rank=rank),
        "specific_repr": _all_gather_local_gradient_tensor(specific_repr, world_size=world_size, rank=rank),
    }
    domain_detached = _all_gather_local_gradient_tensor(
        domain_ids.reshape(-1).to(device=shared_repr.device),
        world_size=world_size,
        rank=rank,
    )
    context["domain_ids"] = domain_detached.detach()
    if quality_weights is not None and "quality_weights" in requested:
        q = quality_weights.reshape(-1).to(device=shared_repr.device, dtype=shared_repr.dtype)
        context["quality_weights"] = _all_gather_local_gradient_tensor(q, world_size=world_size, rank=rank).detach()
    elapsed = time.perf_counter() - t0
    gathered_bytes = {
        key: int(value.numel() * value.element_size())
        for key, value in context.items()
        if torch.is_tensor(value)
    }
    summary = {
        "cross_rank_gather_enabled": bool(world_size > 1),
        "gather_mode": "local_gradient_context",
        "compact_gather_only": True,
        "local_per_gpu_batch": local_per_gpu,
        "structured_effective_pool_size": int(context["shared_repr"].shape[0]),
        "effective_structured_pool": int(context["shared_repr"].shape[0]),
        "gathered_tensor_names": sorted(context.keys()),
        "gathered_tensor_shapes": {key: list(value.shape) for key, value in context.items()},
        "structured_gather_tensor_shapes": {key: list(value.shape) for key, value in context.items()},
        "structured_gather_total_bytes": int(sum(gathered_bytes.values())),
        "structured_gather_bytes_by_tensor": gathered_bytes,
        "structured_gather_dtype": {key: str(value.dtype).replace("torch.", "") for key, value in context.items()},
        "structured_gather_ms": float(elapsed) * 1000.0,
        "communication_time": float(elapsed),
        "remote_tensors_detached": True,
        "forbidden_tensor_policy": "raw_text/token_ids/profile_matrices/large_vocab_logits_not_gathered",
    }
    return context, summary


def _step3_cross_rank_loss_context(
    *,
    structured_inputs: Mapping[str, torch.Tensor],
    batch: GatheredBatch,
    final_cfg: FinalTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    cfg = _parse_step3_cross_rank_gather_config(final_cfg)
    enabled = bool(cfg.get("enabled", False))
    world_size = int(getattr(final_cfg, "ddp_world_size", None) or getattr(final_cfg, "world_size", 1) or 1)
    if not enabled or world_size <= 1:
        local_per_gpu = int(structured_inputs["shared_latent"].shape[0])
        return (
            structured_inputs["shared_latent"],
            structured_inputs["specific_latent"],
            batch.domain_idx,
            {
                "cross_rank_gather_enabled": False,
                "gather_mode": "off",
                "compact_gather_only": True,
                "local_per_gpu_batch": local_per_gpu,
                "structured_effective_pool_size": local_per_gpu,
                "effective_structured_pool": local_per_gpu,
                "gathered_tensor_names": [],
                "gathered_tensor_shapes": {},
                "structured_gather_tensor_shapes": {},
                "structured_gather_total_bytes": 0,
                "structured_gather_dtype": {},
                "structured_gather_ms": 0.0,
                "finite_sync_ms": 0.0,
                "communication_time": 0.0,
                "remote_tensors_detached": False,
            },
        )
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Step3 cross-rank structured gather requires initialized DDP process group.")
    rank = dist.get_rank()
    requested = list(cfg.get("allowed_tensors") or _STEP3_CROSS_RANK_ALLOWED_CONTEXT_KEYS)
    context, summary = gather_step3_structured_context_local_gradient(
        shared_repr=structured_inputs["shared_latent"],
        specific_repr=structured_inputs["specific_latent"],
        domain_ids=batch.domain_idx,
        quality_weights=batch.evidence_quality_prior,
        world_size=world_size,
        rank=rank,
        requested_keys=requested,
    )
    return context["shared_repr"], context["specific_repr"], context["domain_ids"], summary


def _detach_odcr_latent_bundle(latents: Any) -> Any:
    values = {}
    for item in fields(latents):
        value = getattr(latents, item.name)
        values[item.name] = value.detach() if torch.is_tensor(value) else value
    return type(latents)(**values)


def _structured_loss_inputs_from_latents(latents: Any) -> dict[str, torch.Tensor]:
    return {
        "shared_latent": latents.shared_latent,
        "specific_latent": latents.specific_latent,
        "shared_prototype": latents.shared_prototype,
        "domain_style_proto": latents.domain_style_proto,
        "domain_style_component": latents.domain_style_component,
        "content_evidence_target": latents.content_evidence_target,
        "style_evidence_target": latents.style_evidence_target,
        "domain_style_target": latents.domain_style_target,
        "local_style_target": latents.local_style_target,
        "polarity_target": latents.polarity_target,
        "residual_local": latents.residual_local,
        "anchor_pred_content": latents.anchor_pred_content,
        "anchor_pred_style": latents.anchor_pred_style,
    }


def _tensor_is_graph_tied_zero(value: torch.Tensor) -> bool:
    if not torch.is_tensor(value):
        return False
    try:
        is_zero = bool((value.detach().abs() <= 0).all().item())
    except Exception:
        is_zero = False
    return bool(is_zero and value.requires_grad)


def duplicate_step3_loss_check(component_keys: Sequence[str]) -> dict[str, Any]:
    seen: set[str] = set()
    dupes: list[str] = []
    for key in component_keys:
        if key in seen and key not in dupes:
            dupes.append(str(key))
        seen.add(str(key))
    return {
        "status": "duplicate_semantic_components" if dupes else "unique_semantic_components",
        "duplicates": dupes,
        "component_count": len(component_keys),
        "unique_count": len(seen),
    }


def _validate_step3_loss_bundle(
    losses: Mapping[str, torch.Tensor] | Step3LossBundle,
    *,
    ctx: str,
    check_finite: bool = True,
) -> None:
    component_map = losses.components if isinstance(losses, Step3LossBundle) else losses
    for k in STEP3_TOTAL_LOSS_COMPONENT_KEYS:
        if k not in component_map:
            raise RuntimeError(f"{ctx} 缺少 Step3 ODCR 损失项: {k}")
        v = component_map[k]
        if check_finite and not bool(torch.isfinite(v).all().item()):
            raise RuntimeError(f"{ctx} 出现非有限 Step3 ODCR 损失项: {k}={v}")
    if isinstance(losses, Step3LossBundle):
        dup = losses.duplicate_loss_check_summary
        if dup.get("duplicates"):
            raise RuntimeError(f"{ctx} Step3 loss duplicate semantic components: {dup['duplicates']}")


def step3_global_finite_decision_from_local(
    local_finite: bool,
    *,
    world_size: int = 1,
    reduce_min: Optional[Callable[[int], int]] = None,
) -> bool:
    """Pure helper for Step3 finite-loss policy: every rank must agree before backward."""
    local_flag = 1 if bool(local_finite) else 0
    if int(world_size) <= 1:
        return bool(local_flag)
    if reduce_min is None:
        raise RuntimeError("Step3 finite-loss DDP decision requires a global min reducer.")
    return bool(int(reduce_min(local_flag)))


def step3_sync_loss_bundle_finite_status(
    loss_bundle: Step3LossBundle,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Aggregate total/component finite flags in one DDP all-reduce."""
    flags = [torch.isfinite(loss_bundle.total_loss.detach()).all()]
    flags.extend(torch.isfinite(loss_bundle.components[key].detach()).all() for key in STEP3_TOTAL_LOSS_COMPONENT_KEYS)
    local_vec = torch.stack([flag.to(dtype=torch.int32) for flag in flags]).to(device=loss_bundle.total_loss.device)
    local_total_finite = bool(int(local_vec[0].item()))
    if int(world_size) > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Step3 DDP loss-bundle finite sync requested before torch.distributed init.")
        dist.all_reduce(local_vec, op=dist.ReduceOp.MIN)
    global_vec = [bool(int(value.item())) for value in local_vec]
    component_status = {
        key: bool(global_vec[idx + 1])
        for idx, key in enumerate(STEP3_TOTAL_LOSS_COMPONENT_KEYS)
    }
    summary = {
        "local_total_finite": local_total_finite,
        "global_total_finite": bool(global_vec[0]),
        "global_component_finite_status": component_status,
        "sync_method": "single_all_reduce_min_vector",
        "component_count": len(STEP3_TOTAL_LOSS_COMPONENT_KEYS),
    }
    loss_bundle.logging_summary["global_finite_status"] = summary
    return summary


@dataclass(frozen=True)
class Step3StructuredLossWeights:
    orthogonal_weight: float
    orthogonal_xcov_weight: float
    orthogonal_cosine_weight: float
    variance_weight: float
    shared_invariance_weight: float
    specific_separation_weight: float
    anchor_alignment_weight: float
    content_alignment_weight: float
    style_alignment_weight: float
    shared_prototype_weight: float
    domain_style_alignment_weight: float
    local_style_alignment_weight: float
    polarity_alignment_weight: float
    residual_specific_weight: float
    prototype_separation_weight: float
    light_explainer_weight: float


def _loss_weight(raw: Mapping[str, Any], key: str, ctx: str) -> float:
    if key not in raw:
        raise RuntimeError(f"{ctx} missing required Step3 structured loss weight: {key}")
    try:
        out = float(raw[key])
    except Exception as exc:
        raise RuntimeError(f"{ctx}.{key} must be numeric") from exc
    if not math.isfinite(out) or out < 0.0:
        raise RuntimeError(f"{ctx}.{key} must be finite and non-negative")
    return out


def parse_step3_structured_loss_weights(raw: str | Mapping[str, Any]) -> Step3StructuredLossWeights:
    if isinstance(raw, str):
        if not raw.strip():
            raise RuntimeError("Step3 structured loss weights JSON is required from configs/odcr.yaml.")
        obj = json.loads(raw)
    else:
        obj = dict(raw)
    if not isinstance(obj, Mapping):
        raise RuntimeError("Step3 structured loss weights root must be an object.")
    orth = obj.get("orthogonal")
    if not isinstance(orth, Mapping):
        raise RuntimeError("step3.structured_losses.orthogonal must be an object.")
    return Step3StructuredLossWeights(
        orthogonal_weight=_loss_weight(orth, "weight", "step3.structured_losses.orthogonal"),
        orthogonal_xcov_weight=_loss_weight(orth, "xcov_weight", "step3.structured_losses.orthogonal"),
        orthogonal_cosine_weight=_loss_weight(orth, "cosine_weight", "step3.structured_losses.orthogonal"),
        variance_weight=_loss_weight(obj, "variance_weight", "step3.structured_losses"),
        shared_invariance_weight=_loss_weight(obj, "shared_invariance_weight", "step3.structured_losses"),
        specific_separation_weight=_loss_weight(obj, "specific_separation_weight", "step3.structured_losses"),
        anchor_alignment_weight=_loss_weight(obj, "anchor_alignment_weight", "step3.structured_losses"),
        content_alignment_weight=_loss_weight(obj, "content_alignment_weight", "step3.structured_losses"),
        style_alignment_weight=_loss_weight(obj, "style_alignment_weight", "step3.structured_losses"),
        shared_prototype_weight=_loss_weight(obj, "shared_prototype_weight", "step3.structured_losses"),
        domain_style_alignment_weight=_loss_weight(obj, "domain_style_alignment_weight", "step3.structured_losses"),
        local_style_alignment_weight=_loss_weight(obj, "local_style_alignment_weight", "step3.structured_losses"),
        polarity_alignment_weight=_loss_weight(obj, "polarity_alignment_weight", "step3.structured_losses"),
        residual_specific_weight=_loss_weight(obj, "residual_specific_weight", "step3.structured_losses"),
        prototype_separation_weight=_loss_weight(obj, "prototype_separation_weight", "step3.structured_losses"),
        light_explainer_weight=_loss_weight(obj, "light_explainer_weight", "step3.structured_losses"),
    )


def step3_structured_loss_weights_from_config(final_cfg: FinalTrainingConfig) -> Step3StructuredLossWeights:
    return parse_step3_structured_loss_weights(
        str(getattr(final_cfg, "step3_structured_loss_weights_json", "") or "")
    )


def _loss_semantic_float(raw: Mapping[str, Any], key: str, ctx: str, *, min_value: float | None = None) -> float:
    if key not in raw:
        raise RuntimeError(f"{ctx} missing required Step3 loss semantic parameter: {key}")
    try:
        out = float(raw[key])
    except Exception as exc:
        raise RuntimeError(f"{ctx}.{key} must be numeric") from exc
    if not math.isfinite(out):
        raise RuntimeError(f"{ctx}.{key} must be finite")
    if min_value is not None and out < min_value:
        raise RuntimeError(f"{ctx}.{key} must be >= {min_value}")
    return out


def parse_step3_loss_semantics(raw: str | Mapping[str, Any]) -> Step3LossSemantics:
    if isinstance(raw, str):
        if not raw.strip():
            raise RuntimeError("Step3 loss semantics JSON is required from configs/odcr.yaml.")
        obj = json.loads(raw)
    else:
        obj = dict(raw)
    if not isinstance(obj, Mapping):
        raise RuntimeError("Step3 loss semantics root must be an object.")
    quality = obj.get("quality_weight")
    if not isinstance(quality, Mapping):
        raise RuntimeError("step3.loss_semantics.quality_weight must be an object.")
    return Step3LossSemantics(
        specific_separation_margin=_loss_semantic_float(
            obj, "specific_separation_margin", "step3.loss_semantics", min_value=0.0
        ),
        variance_target_std=_loss_semantic_float(
            obj, "variance_target_std", "step3.loss_semantics", min_value=0.0
        ),
        variance_eps=_loss_semantic_float(obj, "variance_eps", "step3.loss_semantics", min_value=0.0),
        orthogonal_eps=_loss_semantic_float(obj, "orthogonal_eps", "step3.loss_semantics", min_value=0.0),
        cosine_eps=_loss_semantic_float(obj, "cosine_eps", "step3.loss_semantics", min_value=0.0),
        sample_weight_eps=_loss_semantic_float(
            obj, "sample_weight_eps", "step3.loss_semantics", min_value=0.0
        ),
        quality_evidence_base=_loss_semantic_float(
            quality, "evidence_base", "step3.loss_semantics.quality_weight", min_value=0.0
        ),
        quality_evidence_scale=_loss_semantic_float(
            quality, "evidence_scale", "step3.loss_semantics.quality_weight", min_value=0.0
        ),
        quality_anchor_base=_loss_semantic_float(
            quality, "anchor_base", "step3.loss_semantics.quality_weight", min_value=0.0
        ),
        quality_anchor_scale=_loss_semantic_float(
            quality, "anchor_scale", "step3.loss_semantics.quality_weight", min_value=0.0
        ),
        prototype_separation_eps=_loss_semantic_float(
            obj, "prototype_separation_eps", "step3.loss_semantics", min_value=0.0
        ),
    )


def step3_loss_semantics_from_config(final_cfg: FinalTrainingConfig) -> Step3LossSemantics:
    return parse_step3_loss_semantics(str(getattr(final_cfg, "step3_loss_semantics_json", "") or ""))


def _step3_component_weights(weights: Step3StructuredLossWeights) -> dict[str, float]:
    return {
        "L_rating_shared": 1.0,
        "L_light_explainer": float(weights.light_explainer_weight),
        "L_orthogonal": float(weights.orthogonal_weight),
        "L_variance": float(weights.variance_weight),
        "L_shared_invariance": float(weights.shared_invariance_weight),
        "L_specific_separation": float(weights.specific_separation_weight),
        "L_anchor_content": float(weights.anchor_alignment_weight),
        "L_anchor_style": float(weights.anchor_alignment_weight),
        "L_content_alignment": float(weights.content_alignment_weight),
        "L_style_alignment": float(weights.style_alignment_weight),
        "L_shared_proto": float(weights.shared_prototype_weight),
        "L_domain_style_alignment": float(weights.domain_style_alignment_weight),
        "L_local_style_alignment": float(weights.local_style_alignment_weight),
        "L_polarity_alignment": float(weights.polarity_alignment_weight),
        "L_residual_specific": float(weights.residual_specific_weight),
        "L_prototype_separation": float(weights.prototype_separation_weight),
    }


_COMPONENT_TO_WEIGHT_FIELD = {
    "L_orthogonal": "orthogonal_weight",
    "L_variance": "variance_weight",
    "L_shared_invariance": "shared_invariance_weight",
    "L_specific_separation": "specific_separation_weight",
    "L_anchor_content": "anchor_alignment_weight",
    "L_anchor_style": "anchor_alignment_weight",
    "L_content_alignment": "content_alignment_weight",
    "L_style_alignment": "style_alignment_weight",
    "L_shared_proto": "shared_prototype_weight",
    "L_domain_style_alignment": "domain_style_alignment_weight",
    "L_local_style_alignment": "local_style_alignment_weight",
    "L_polarity_alignment": "polarity_alignment_weight",
    "L_residual_specific": "residual_specific_weight",
    "L_prototype_separation": "prototype_separation_weight",
    "L_light_explainer": "light_explainer_weight",
}


def apply_step3_phase_loss_multipliers(
    weights: Step3StructuredLossWeights,
    multipliers: Mapping[str, Any] | None,
) -> Step3StructuredLossWeights:
    if not multipliers:
        return weights
    values = {field.name: getattr(weights, field.name) for field in fields(Step3StructuredLossWeights)}
    for component, raw_multiplier in dict(multipliers).items():
        field_name = _COMPONENT_TO_WEIGHT_FIELD.get(str(component))
        if not field_name:
            continue
        values[field_name] = float(values[field_name]) * float(raw_multiplier)
    return Step3StructuredLossWeights(**values)


def compose_step3_loss_from_forward_output(
    *,
    forward_output: Step3ForwardOutput,
    batch: GatheredBatch,
    final_cfg: FinalTrainingConfig,
    weights: Step3StructuredLossWeights | None = None,
    semantics: Step3LossSemantics | None = None,
) -> Step3LossBundle:
    if not isinstance(forward_output, Step3ForwardOutput):
        raise RuntimeError("Step3 loss builder requires Step3ForwardOutput from model.forward.")
    s = forward_output.structured_loss_inputs
    missing_inputs = [key for key in STEP3_STRUCTURED_LOSS_INPUT_KEYS if key not in s]
    if missing_inputs:
        raise RuntimeError(f"Step3ForwardOutput.structured_loss_inputs missing keys: {missing_inputs}")
    weights = weights or step3_structured_loss_weights_from_config(final_cfg)
    semantics = semantics or step3_loss_semantics_from_config(final_cfg)
    c_a = batch.content_anchor_score
    s_a = batch.style_anchor_score
    eq = batch.evidence_quality_prior
    if c_a is None or s_a is None or eq is None:
        raise RuntimeError("Step3 loss builder requires anchor and evidence-quality tensors from GatheredBatch.")
    q_w = (
        semantics.quality_evidence_base
        + semantics.quality_evidence_scale * eq.view(-1).to(dtype=torch.float32).clamp(0.0, 1.0)
    ).detach()
    q_content = q_w * (
        semantics.quality_anchor_base
        + semantics.quality_anchor_scale * c_a.view(-1).to(dtype=torch.float32).clamp(0.0, 1.0)
    )
    q_style = q_w * (
        semantics.quality_anchor_base
        + semantics.quality_anchor_scale * s_a.view(-1).to(dtype=torch.float32).clamp(0.0, 1.0)
    )
    structured_shared, structured_specific, structured_domain_idx, gather_summary = _step3_cross_rank_loss_context(
        structured_inputs=s,
        batch=batch,
        final_cfg=final_cfg,
    )
    orth = build_orthogonal_losses(
        structured_shared,
        structured_specific,
        eps=float(semantics.orthogonal_eps),
        w_xcov=float(weights.orthogonal_xcov_weight),
        w_cos=float(weights.orthogonal_cosine_weight),
    )
    var = variance_floor_loss(
        structured_shared,
        structured_specific,
        target_std=float(semantics.variance_target_std),
        eps=float(semantics.variance_eps),
    )
    components = {
        "L_rating_shared": F.mse_loss(forward_output.rating, batch.rating, reduction="mean"),
        "L_light_explainer": F.cross_entropy(
            forward_output.word_dist.view(-1, forward_output.word_dist.size(-1)),
            batch.tgt_output.reshape(-1),
            ignore_index=0,
        ),
        "L_orthogonal": orth.loss_ortho_total,
        "L_variance": var.loss_var_total,
        "L_shared_invariance": shared_invariance_loss(structured_shared, structured_domain_idx),
        "L_specific_separation": specific_separation_loss(
            structured_specific,
            structured_domain_idx,
            margin=float(semantics.specific_separation_margin),
        ),
        "L_anchor_content": anchor_score_alignment_loss(
            s["anchor_pred_content"],
            c_a.view(-1),
            sample_weight=q_content,
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_anchor_style": anchor_score_alignment_loss(
            s["anchor_pred_style"],
            s_a.view(-1),
            sample_weight=q_style,
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_content_alignment": cosine_pull_loss(
            s["shared_latent"],
            s["content_evidence_target"],
            sample_weight=q_content,
            eps=float(semantics.cosine_eps),
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_style_alignment": cosine_pull_loss(
            s["specific_latent"],
            s["style_evidence_target"],
            sample_weight=q_style,
            eps=float(semantics.cosine_eps),
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_shared_proto": shared_prototype_pull_loss(
            s["shared_latent"],
            s["shared_prototype"],
            sample_weight=q_content,
            eps=float(semantics.cosine_eps),
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_domain_style_alignment": cosine_pull_loss(
            s["domain_style_component"],
            s["domain_style_target"],
            sample_weight=q_style,
            eps=float(semantics.cosine_eps),
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_local_style_alignment": cosine_pull_loss(
            s["residual_local"],
            s["local_style_target"],
            sample_weight=q_style,
            eps=float(semantics.cosine_eps),
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_polarity_alignment": cosine_pull_loss(
            s["specific_latent"],
            s["polarity_target"],
            sample_weight=q_style,
            eps=float(semantics.cosine_eps),
            sample_weight_eps=float(semantics.sample_weight_eps),
        ),
        "L_residual_specific": residual_l2_penalty(s["residual_local"]),
        "L_prototype_separation": domain_style_prototype_separation(
            s["domain_style_proto"],
            eps=float(semantics.prototype_separation_eps),
        ),
    }
    component_weights = _step3_component_weights(weights)
    duplicate_summary = duplicate_step3_loss_check(list(STEP3_TOTAL_LOSS_COMPONENT_KEYS))
    weighted = {key: components[key] * float(component_weights[key]) for key in STEP3_TOTAL_LOSS_COMPONENT_KEYS}
    total = sum((weighted[key] for key in STEP3_TOTAL_LOSS_COMPONENT_KEYS), components["L_rating_shared"].sum() * 0.0)
    participates = {key: True for key in STEP3_TOTAL_LOSS_COMPONENT_KEYS}
    finite_status = {key: bool(torch.isfinite(value.detach()).all().item()) for key, value in components.items()}
    graph_zero = {key: _tensor_is_graph_tied_zero(value) for key, value in components.items()}
    diagnostics = {
        "L_orthogonal_xcov": orth.loss_ortho_xcov,
        "L_orthogonal_cosine": orth.loss_ortho_cos,
        "L_shared_variance": var.loss_shared_var,
        "L_specific_variance": var.loss_specific_var,
        "shared_std_mean": var.shared_std_mean,
        "specific_std_mean": var.specific_std_mean,
        "shared_std_min": var.shared_std_min,
        "specific_std_min": var.specific_std_min,
    }
    logging_summary = {
        "components": {
            key: {
                "raw": float(components[key].detach().item()),
                "weight": float(component_weights[key]),
                "weighted": float(weighted[key].detach().item()),
                "participates_in_total": bool(participates[key]),
                "finite": bool(finite_status[key]),
                "graph_tied_zero": bool(graph_zero[key]),
            }
            for key in STEP3_TOTAL_LOSS_COMPONENT_KEYS
        },
        "diagnostics": {key: float(value.detach().item()) for key, value in diagnostics.items()},
        "total_loss": float(total.detach().item()),
        "duplicate_loss_check_summary": duplicate_summary,
        "loss_builder_contract": "Step3ForwardOutput + GatheredBatch + resolved_config",
        "cross_rank_gather": gather_summary,
    }
    bundle = Step3LossBundle(
        total_loss=total,
        components=components,
        weights=component_weights,
        weighted_components=weighted,
        participates_in_total=participates,
        finite_status=finite_status,
        graph_tied_zero_status=graph_zero,
        duplicate_loss_check_summary=duplicate_summary,
        logging_summary=logging_summary,
        diagnostics=diagnostics,
    )
    _validate_step3_loss_bundle(bundle, ctx="step3/loss-builder", check_finite=False)
    return bundle


def validate_step3_graph_safety_preflight(
    *,
    forward_output: Step3ForwardOutput,
    loss_bundle: Step3LossBundle,
    underlying_model: nn.Module | None = None,
    ctx: str,
) -> dict[str, Any]:
    if not isinstance(forward_output, Step3ForwardOutput):
        raise RuntimeError(f"{ctx}: model.forward must return Step3ForwardOutput.")
    missing_inputs = [key for key in STEP3_STRUCTURED_LOSS_INPUT_KEYS if key not in forward_output.structured_loss_inputs]
    if missing_inputs:
        raise RuntimeError(f"{ctx}: missing structured_loss_inputs {missing_inputs}.")
    _validate_step3_loss_bundle(loss_bundle, ctx=ctx, check_finite=False)
    detached_last = None
    if underlying_model is not None and hasattr(underlying_model, "last_odcr_latents"):
        last = getattr(underlying_model, "last_odcr_latents")
        if last is not None:
            detached_last = all(
                (not torch.is_tensor(getattr(last, item.name))) or (not getattr(last, item.name).requires_grad)
                for item in fields(last)
            )
            if not detached_last:
                raise RuntimeError(f"{ctx}: last_odcr_latents must be detached debug state, not loss state.")
    summary = {
        "status": "pass",
        "forward_output_type": "Step3ForwardOutput",
        "structured_loss_input_count": len(forward_output.structured_loss_inputs),
        "required_loss_keys_present": True,
        "last_odcr_latents_detached": detached_last,
        "loss_builder_contract": "no module side-channel parameter reads",
        "duplicate_loss_check_summary": dict(loss_bundle.duplicate_loss_check_summary),
    }
    return summary


def _require_step3_canonical_columns(df: pd.DataFrame, *, csv_path: str, split: str) -> None:
    required = (
        "content_anchor_score",
        "style_anchor_score",
        *CANONICAL_PREPROCESS_ASSET_COLUMNS,
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Step3 {split} CSV 缺少 canonical preprocess 字段: {missing} | csv={csv_path}. "
            "Step3 已移除旧弱字段兼容，请先完成 Phase1 canonical preprocess rerun。"
        )


_EVAL_REQUIRES_TORCHRUN_MSG = (
    "step3 runner 的 eval 仅支持 torchrun / python -m torch.distributed.run 下的 DDP。\n"
    "用户日常：在项目根执行  python code/odcr.py step3 --eval-only …\n"
    "请勿使用 `python <薄壳>.py eval` 在非 torchrun 环境下直接启动。\n"
    "高级排障（须 torchrun，在 code/ 目录）见 docs/ODCR_Scripts_and_Runtime_Guide.md 附录。\n"
    "多卡请设置 CUDA_VISIBLE_DEVICES 并使 nproc_per_node 与可见 GPU 数一致。"
)

_odcr_text_tok: Optional[Any] = None
_odcr_text_tok_override: Optional[Any] = None


def set_odcr_text_tokenizer_override(tok: Optional[Any]) -> None:
    """测试注入：非 None 时 ``get_odcr_text_tokenizer()`` 直接返回 tok，不读本地目录。"""
    global _odcr_text_tok_override, _odcr_text_tok
    _odcr_text_tok_override = tok
    _odcr_text_tok = None


def get_odcr_text_tokenizer() -> Any:
    """懒加载 T5Tokenizer（google flan-t5-xl 本地目录）；禁止 Hub 回退。"""
    global _odcr_text_tok
    if _odcr_text_tok_override is not None:
        return _odcr_text_tok_override
    if _odcr_text_tok is None:
        _odcr_text_tok = T5Tokenizer.from_pretrained(
            require_step5_text_model_dir(), legacy=True, local_files_only=True
        )
    return _odcr_text_tok


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _step3_model_architecture_lineage(final_cfg: FinalTrainingConfig) -> Dict[str, Any]:
    return {
        "nuser": int(final_cfg.nuser),
        "nitem": int(final_cfg.nitem),
        "ntoken": int(final_cfg.ntoken),
        "emsize": int(final_cfg.emsize),
        "nlayers": int(final_cfg.nlayers),
        "nhead": int(final_cfg.nhead),
        "nhid": int(final_cfg.nhid),
        "dropout": float(final_cfg.dropout),
    }


_STEP3_CHECKPOINT_CRITICAL_CODE_FILES = (
    "code/executors/step3_train_core.py",
    "code/executors/step3_entry.py",
    "code/odcr_core/training_checkpoint.py",
    "code/odcr_core/step3_upstream_gate.py",
    "code/odcr_core/config_resolver.py",
    "code/odcr_core/config_schema.py",
    "code/odcr_core/index_contract.py",
    "code/config.py",
    "code/data_contract.py",
    "configs/odcr.yaml",
)


def _repo_root_for_step3_lineage() -> Path:
    return Path(os.environ.get("ODCR_ROOT") or Path(__file__).resolve().parents[2]).expanduser().resolve()


def _git_code_fingerprint_for_step3() -> Dict[str, Any]:
    repo_root = _repo_root_for_step3_lineage()

    def _git(args: list[str]) -> str:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:
            return f"unavailable:{type(exc).__name__}:{exc}"
        if proc.returncode != 0:
            return f"unavailable:returncode={proc.returncode}:{(proc.stderr or '').strip()}"
        return (proc.stdout or "").strip()

    code_files = []
    for rel in _STEP3_CHECKPOINT_CRITICAL_CODE_FILES:
        path = (repo_root / rel).resolve()
        code_files.append(
            {
                "path": rel,
                "fingerprint": file_fingerprint(path, sample_only=False),
            }
        )
    payload = {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty_status": _git(["status", "--short", "--", *_STEP3_CHECKPOINT_CRITICAL_CODE_FILES]),
        "critical_code_files": code_files,
    }
    payload["critical_code_files_hash"] = stable_hash(code_files)
    payload["fingerprint_hash"] = stable_hash(payload)
    return payload


def _step3_preprocess_checkpoint_sections(upstream_evidence: Mapping[str, Any]) -> Dict[str, Any]:
    preprocess = upstream_evidence.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise RuntimeError("Step3 checkpoint lineage requires upstream preprocess gate evidence.")
    latest_run_ids: Dict[str, str] = {}
    run_summary_fps: Dict[str, Any] = {}
    stage_status_fps: Dict[str, Any] = {}
    stage_manifest_fps: Dict[str, Any] = {}
    source_table_fps: Dict[str, Any] = {}
    metrics_fps: Dict[str, Any] = {}
    verify_report_fps: Dict[str, Any] = {}
    run_summary_fingerprints: Dict[str, Any] = {}
    stage_fingerprints: Dict[str, Any] = {}
    for unit in ("a", "b", "c"):
        item = preprocess.get(unit)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Step3 checkpoint lineage missing preprocess_{unit} evidence.")
        run_id = str(item.get("run_id") or "").strip()
        if not run_id:
            raise RuntimeError(f"Step3 checkpoint lineage missing preprocess_{unit} run_id.")
        latest_run_ids[unit] = run_id
        run_summary_fps[unit] = item.get("run_summary_fingerprint")
        stage_status_fps[unit] = item.get("stage_status_fingerprint")
        stage_manifest_fps[unit] = item.get("stage_manifest_fingerprint")
        source_table_fps[unit] = item.get("source_table_fingerprint")
        metrics_fps[unit] = item.get("metrics_fingerprint")
        verify_report_fps[unit] = item.get("verify_report_fingerprint")
        run_summary_fingerprints[unit] = item.get("fingerprint_hash")
        stage_fingerprints[unit] = {
            "run_fingerprint_hash": item.get("fingerprint_hash"),
            "run_summary": item.get("run_summary_fingerprint"),
            "stage_status": item.get("stage_status_fingerprint"),
            "stage_manifest": item.get("stage_manifest_fingerprint"),
            "source_table": item.get("source_table_fingerprint"),
            "metrics": item.get("metrics_fingerprint"),
            "verify_report": item.get("verify_report_fingerprint"),
        }
    return {
        "preprocess_latest_run_ids": latest_run_ids,
        "preprocess_a_latest_run_id": latest_run_ids["a"],
        "preprocess_b_latest_run_id": latest_run_ids["b"],
        "preprocess_c_latest_run_id": latest_run_ids["c"],
        "preprocess_run_summary_fingerprints": run_summary_fps,
        "preprocess_run_summary_lineage_fingerprints": run_summary_fingerprints,
        "preprocess_stage_status_fingerprints": stage_status_fps,
        "preprocess_stage_manifest_fingerprints": stage_manifest_fps,
        "preprocess_source_table_fingerprints": source_table_fps,
        "preprocess_metrics_fingerprints": metrics_fps,
        "preprocess_verify_report_fingerprints": verify_report_fps,
        "preprocess_stage_fingerprints": stage_fingerprints,
        "preprocess_run_summary_fingerprints_hash": stable_hash(run_summary_fps),
        "preprocess_stage_status_fingerprints_hash": stable_hash(stage_status_fps),
        "preprocess_stage_manifest_fingerprints_hash": stable_hash(stage_manifest_fps),
        "preprocess_source_table_fingerprints_hash": stable_hash(source_table_fps),
        "preprocess_metrics_fingerprints_hash": stable_hash(metrics_fps),
        "preprocess_verify_report_fingerprints_hash": stable_hash(verify_report_fps),
    }


def _step3_tokenizer_cache_manifest_lineage(final_cfg: FinalTrainingConfig) -> Dict[str, Any]:
    raw = str(getattr(final_cfg, "step3_tokenizer_cache_manifest_json", "") or "").strip()
    if not raw:
        raise RuntimeError("Step3 checkpoint lineage requires step3_tokenizer_cache_manifest_json.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Step3 tokenizer cache manifest summary is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Step3 tokenizer cache manifest summary must be a JSON object.")
    for key in (
        "manifest_path",
        "manifest_fingerprint",
        "tokenizer_cache_compat_hash",
        "fingerprint_hash",
        "cache_content_hash",
    ):
        if not payload.get(key):
            raise RuntimeError(f"Step3 tokenizer cache manifest summary missing {key}.")
    payload["manifest_hash"] = stable_hash(payload.get("manifest_fingerprint") or {})
    return payload


def _step3_batch_semantics(final_cfg: FinalTrainingConfig) -> Dict[str, Any]:
    batch_size = int(final_cfg.train_batch_size)
    per_gpu_batch_size = int(final_cfg.per_device_train_batch_size)
    ddp_world_size = int(final_cfg.ddp_world_size or final_cfg.world_size)
    rhs = per_gpu_batch_size * ddp_world_size
    return {
        "batch_semantics_version": "odcr_no_accum/1",
        "step3_batch_semantics": "odcr_no_accum/1",
        "batch_size": batch_size,
        "global_batch_size": batch_size,
        "per_gpu_batch_size": per_gpu_batch_size,
        "micro_batch_size_alias": per_gpu_batch_size,
        "ddp_world_size": ddp_world_size,
        "formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
        "formula_proof": {
            "lhs": batch_size,
            "rhs": rhs,
            "matches": batch_size == rhs,
        },
        "valid_batch_size": int(final_cfg.valid_batch_size),
        "valid_micro_batch_size": int(final_cfg.valid_micro_batch_size),
        "valid_batch_source": "step3.eval resolver-derived or explicit One-Control",
        "cross_rank_structured_gather": json.loads(
            str(getattr(final_cfg, "cross_rank_structured_gather_config_json", "") or "{}")
        ),
        "effective_structured_pool": {
            "local_per_gpu_batch": per_gpu_batch_size,
            "local_micro_batch_alias": per_gpu_batch_size,
            "effective_pool": rhs,
            "gathered_tensor_names": (
                json.loads(str(getattr(final_cfg, "cross_rank_structured_gather_config_json", "") or "{}")).get(
                    "allowed_tensors", []
                )
            ),
            "remote_tensors_detached": True,
        },
    }


def _step3_ddp_config(final_cfg: FinalTrainingConfig) -> Dict[str, Any]:
    return {
        "ddp_world_size": int(final_cfg.ddp_world_size or final_cfg.world_size),
        "world_size": int(final_cfg.world_size),
        "device_ids": [int(x) for x in tuple(final_cfg.device_ids or ())],
        "ddp_find_unused_parameters": bool(final_cfg.ddp_find_unused_parameters),
        "ddp_find_unused_false_preflight": str(final_cfg.ddp_find_unused_false_preflight),
        "ddp_static_graph": bool(getattr(final_cfg, "ddp_static_graph", False)),
        "ddp_graph_safety_preflight": bool(getattr(final_cfg, "ddp_graph_safety_preflight", True)),
        "num_proc": int(final_cfg.num_proc),
        "max_parallel_cpu": int(final_cfg.max_parallel_cpu),
        "dataloader_num_workers_train": int(final_cfg.dataloader_num_workers_train),
        "dataloader_num_workers_valid": int(final_cfg.dataloader_num_workers_valid),
        "dataloader_prefetch_factor_train": final_cfg.dataloader_prefetch_factor_train,
        "dataloader_prefetch_factor_valid": final_cfg.dataloader_prefetch_factor_valid,
        "pin_memory": bool(final_cfg.pin_memory),
        "persistent_workers": bool(final_cfg.persistent_workers),
        "non_blocking_h2d": bool(final_cfg.non_blocking_h2d),
    }


def _step3_sentence_embed_model_identity() -> Dict[str, Any]:
    path = os.environ.get("ODCR_RESOLVED_SENTENCE_EMBED_MODEL") or get_sentence_embed_model_dir()
    return {
        "identity": os.path.abspath(str(path)),
        "resolved_env_key": "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
        "model_artifact_fingerprint": model_artifact_fingerprint(path),
    }


def _step3_checkpoint_run_id(checkpoint_path: str) -> str:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if ckpt.parent.name == "model":
        return ckpt.parent.parent.name
    return str(ckpt.parent.parent.name if ckpt.parent.parent.name else ckpt.parent.name)


def _step3_checkpoint_compatibility_metadata() -> Dict[str, Any]:
    return {
        "minimum_accepted_schema_version": STEP3_CHECKPOINT_MIN_ACCEPTED_SCHEMA_VERSION,
        "downstream_consumers": ["step4", "step5_indirect_via_step4_export", "eval_rerank_indirect_via_step5_checkpoint"],
        "downstream_compare_fields": [
            "sidecar_schema_version",
            "checkpoint_file_hash",
            "task_id",
            "source_domain",
            "target_domain",
            "artifact_lineage_hash",
            "semantic_model_compat_hash",
            "data_contract_hash",
            "tokenizer_cache_compat_hash",
            "preprocess_latest_run_ids",
            "preprocess_run_summary_fingerprints_hash",
            "preprocess_stage_status_fingerprints_hash",
            "preprocess_stage_manifest_fingerprints_hash",
            "preprocess_source_table_fingerprints_hash",
            "preprocess_metrics_fingerprints_hash",
            "preprocess_verify_report_fingerprints_hash",
            "profile_artifact_fingerprints_hash",
            "domain_artifact_fingerprints_hash",
            "source_csv_fingerprints_hash",
            "merged_csv_fingerprints_hash",
            "embed_dim",
            "model_architecture_config_hash",
            "step3_structured_losses_config_hash",
            "step3_loss_semantics_config_hash",
            "step3_tokenizer_config_hash",
            "step3_evidence_config_hash",
            "step3_scenario_profile_hash",
            "step3_tokenizer_cache_manifest_hash",
        ],
        "record_only_fields": [
            "full_run_config_hash",
            "one_control_resolved_config_hash",
            "source_table_hash",
            "train_runtime_config_hash",
            "optimizer_config_hash",
            "performance_profile_hash",
            "step3_optimizer_config_hash",
            "step3_scheduler_config_hash",
            "step3_valid_batch_config_hash",
            "ddp_config_hash",
            "precision_config_hash",
            "batch_semantics_hash",
        ],
        "incompatible_reason_codes": [
            "missing_sidecar",
            "unsupported_schema",
            "checkpoint_file_hash_mismatch",
            "task_or_domain_mismatch",
            "preprocess_latest_mismatch",
            "preprocess_artifact_fingerprint_mismatch",
            "profile_domain_fingerprint_mismatch",
            "embed_dim_mismatch",
            "architecture_or_loss_mismatch",
            "tokenizer_cache_manifest_mismatch",
        ],
    }


def _build_step3_checkpoint_lineage(
    final_cfg: FinalTrainingConfig,
    *,
    checkpoint_path: str,
    checkpoint_context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = current_effective_payload(required=True)
    task_idx = int(final_cfg.task_idx)
    checkpoint_abs = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
    ctx = dict(checkpoint_context or {})
    merged_root = get_merged_data_dir()
    train_path = os.path.join(merged_root, str(task_idx), "aug_train.csv")
    valid_path = os.path.join(merged_root, str(task_idx), "aug_valid.csv")
    data_fps = {
        "aug_train_csv": file_fingerprint(train_path),
        "aug_valid_csv": file_fingerprint(valid_path),
    }
    structured = payload.get("step3_structured_losses") or json.loads(
        str(final_cfg.step3_structured_loss_weights_json or "{}")
    )
    loss_semantics = payload.get("step3_loss_semantics") or json.loads(
        str(getattr(final_cfg, "step3_loss_semantics_json", "") or "{}")
    )
    optimizer_config = payload.get("step3_optimizer") or json.loads(str(final_cfg.optimizer_config_json or "{}"))
    precision_config = payload.get("step3_precision") or json.loads(str(final_cfg.precision_config_json or "{}"))
    tokenizer_config = payload.get("step3_tokenizer") or json.loads(str(final_cfg.tokenizer_config_json or "{}"))
    evidence_config = payload.get("step3_evidence") or json.loads(str(final_cfg.evidence_config_json or "{}"))
    scheduler_config = payload.get("step3_scheduler") or json.loads(str(final_cfg.scheduler_config_json or "{}"))
    valid_batch_config = payload.get("step3_eval") or json.loads(str(final_cfg.valid_batch_config_json or "{}"))
    scenario_profile = payload.get("step3_scenario_profile") or json.loads(str(final_cfg.scenario_profile_json or "{}"))
    arch = _step3_model_architecture_lineage(final_cfg)
    upstream_raw = str(getattr(final_cfg, "step3_upstream_evidence_json", "") or "").strip()
    if not upstream_raw:
        raise RuntimeError("Step3 checkpoint lineage requires step3_upstream_evidence_json from the preprocess hard gate.")
    try:
        upstream_evidence = json.loads(upstream_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Step3 upstream evidence summary is not valid JSON: {exc}") from exc
    if not isinstance(upstream_evidence, dict) or upstream_evidence.get("status") != "ok":
        raise RuntimeError("Step3 upstream evidence summary must be an ok JSON object.")
    resolved_config = current_resolved_config_lineage(
        stage="step3",
        task_id=task_idx,
        artifact="step3_checkpoint",
        required_file=bool((os.environ.get("ODCR_MANIFEST_DIR") or "").strip()),
    )
    source_table = current_source_table_lineage(required_file=bool((os.environ.get("ODCR_MANIFEST_DIR") or "").strip()))
    training_runtime_config = current_training_runtime_config_lineage(
        required_file=bool((os.environ.get("ODCR_MANIFEST_DIR") or "").strip())
    )
    preprocess_sections = _step3_preprocess_checkpoint_sections(upstream_evidence)
    tokenizer_cache_manifest = _step3_tokenizer_cache_manifest_lineage(final_cfg)
    resolved_compatibility = step3_resolved_config_compatibility_payload(
        payload=payload,
        task_id=task_idx,
        source_domain=str(final_cfg.auxiliary),
        target_domain=str(final_cfg.target),
        embed_dim=int(get_odcr_embed_dim()),
        structured_losses=structured,
        loss_semantics=loss_semantics,
        architecture_hash=stable_hash(arch),
    )
    source_table_compatibility = step3_source_table_compatibility_payload(source_table)
    tokenizer_cache_compat_hash = str(
        tokenizer_cache_manifest.get("tokenizer_cache_compat_hash")
        or tokenizer_cache_manifest.get("compatibility_key")
        or ""
    )
    data_contract_payload = {
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "source_task": {
            "task_id": task_idx,
            "auxiliary": str(final_cfg.auxiliary),
            "target": str(final_cfg.target),
        },
        "source_csv_fingerprints": upstream_evidence.get("source_csv_artifacts"),
        "merged_csv_fingerprints": upstream_evidence.get("merged_artifacts") or data_fps,
    }
    artifact_lineage_payload = {
        "data_merged_artifact_fingerprint": stable_hash(data_fps),
        "preprocess": preprocess_sections,
        "profile_artifact_fingerprints": upstream_evidence.get("profile_artifact_fingerprints"),
        "domain_artifact_fingerprints": upstream_evidence.get("domain_artifact_fingerprints"),
        "sentence_embed_model_identity": _step3_sentence_embed_model_identity(),
    }
    semantic_model_payload = {
        "resolved_config_compatibility": resolved_compatibility,
        "source_table_compatibility": source_table_compatibility,
        "embed_dim": int(final_cfg.emsize),
        "model_architecture_config_hash": stable_hash(arch),
        "representation_output_contract_hash": stable_hash(
            {
                "Step3ForwardOutput": "odcr_step3_forward_output/structured_shared_specific_v1",
                "Step3LossBundle": "odcr_step3_loss_bundle/structured_shared_specific_v1",
            }
        ),
        "structured_losses_hash": stable_hash(structured),
        "loss_semantics_hash": stable_hash(loss_semantics),
        "profile_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("profile_artifact_fingerprints") or {}),
        "domain_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("domain_artifact_fingerprints") or {}),
    }
    loss_config_payload = {
        "step3_structured_losses": structured,
        "step3_loss_semantics": loss_semantics,
        "cross_rank_structured_gather": payload.get("step3_cross_rank_structured_gather"),
        "effective_structured_pool": {
            "local_per_gpu_batch": (payload.get("training_row") or {}).get("local_per_gpu_batch"),
            "effective_pool": (payload.get("training_row") or {}).get("effective_structured_pool"),
            "gathered_tensor_names": (payload.get("training_row") or {}).get("gathered_tensor_names"),
            "remote_tensors_detached": (payload.get("training_row") or {}).get("remote_tensors_detached"),
        },
    }
    train_runtime_payload = {
        "batch_semantics": _step3_batch_semantics(final_cfg),
        "ddp_config": _step3_ddp_config(final_cfg),
        "precision_config": {
            "train_precision": str(final_cfg.train_precision),
            "allow_tf32": bool(final_cfg.allow_tf32),
            "amp_autocast": bool(final_cfg.amp_autocast),
            "grad_scaler": bool(final_cfg.grad_scaler),
            "runtime_precision_env": os.environ.get("ODCR_RUNTIME_PRECISION_MODE", ""),
            "runtime_allow_tf32_env": os.environ.get("ODCR_RUNTIME_ALLOW_TF32", ""),
        },
        "cross_rank_structured_gather": payload.get("step3_cross_rank_structured_gather"),
        "effective_structured_pool": loss_config_payload["effective_structured_pool"],
    }
    optimizer_payload = {
        "step3_optimizer_config": optimizer_config,
        "step3_scheduler_config": scheduler_config,
        "learning_rate": float(final_cfg.learning_rate),
        "max_grad_norm": float(final_cfg.max_grad_norm),
    }
    performance_profile_payload = {
        "task_profile_id": payload.get("task_profile_id"),
        "task_profile_key": payload.get("task_profile_key"),
        "profile_isolation_hash": payload.get("profile_isolation_hash"),
        "step3_task_profile": payload.get("step3_task_profile"),
        "step3_backup_profiles": payload.get("step3_backup_profiles"),
        "step3_exploration_profiles": payload.get("step3_exploration_profiles"),
        "step3_worker_profiles": payload.get("step3_worker_profiles"),
        "step3_prefetcher": payload.get("step3_prefetcher"),
        "step3_cross_rank_structured_gather": payload.get("step3_cross_rank_structured_gather"),
        "step3_memory": payload.get("step3_memory"),
        "step3_timing": payload.get("step3_timing"),
        "step3_batch_semantics": (payload.get("training_row") or {}).get("step3_batch_semantics"),
    }
    lineage: Dict[str, Any] = {
        "sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "stage": "step3",
        "run_id": str(final_cfg.run_id or _step3_checkpoint_run_id(checkpoint_abs)),
        "task_id": task_idx,
        "source_domain": str(final_cfg.auxiliary),
        "target_domain": str(final_cfg.target),
        "task_profile_id": str(getattr(final_cfg, "task_profile_id", "") or payload.get("task_profile_id") or ""),
        "task_profile_key": str(getattr(final_cfg, "task_profile_key", "") or payload.get("task_profile_key") or ""),
        "profile_isolation_hash": str(
            getattr(final_cfg, "profile_isolation_hash", "") or payload.get("profile_isolation_hash") or ""
        ),
        "scenario": str(getattr(final_cfg, "scenario", "") or payload.get("scenario") or ""),
        "direction": str(getattr(final_cfg, "direction", "") or payload.get("direction") or ""),
        "checkpoint_path": checkpoint_abs,
        "checkpoint_file_hash": checkpoint_file_sha256(checkpoint_abs),
        "checkpoint_file": file_fingerprint(checkpoint_abs),
        "checkpoint_epoch": int(ctx.get("checkpoint_epoch", 0) or 0),
        "selection_metric": str(ctx.get("selection_metric") or "valid_loss"),
        "selection_metric_value": float(ctx.get("selection_metric_value", 0.0) or 0.0),
        "selection_direction": str(ctx.get("selection_direction") or "min"),
        "selection_scope": str(ctx.get("selection_scope") or "latest"),
        "reason": str(ctx.get("reason") or ""),
        "replaced_previous": bool(ctx.get("replaced_previous", False)),
        "global_best_epoch": ctx.get("global_best_epoch"),
        "global_best_metric": ctx.get("global_best_metric"),
        "after_min_epochs_best_epoch": ctx.get("after_min_epochs_best_epoch"),
        "after_min_epochs_best_metric": ctx.get("after_min_epochs_best_metric"),
        "epoch_summary_hash": str(ctx.get("epoch_summary_hash") or ""),
        "metrics_jsonl_hash": str(ctx.get("metrics_jsonl_hash") or ""),
        "training_runtime_config_hash": str(ctx.get("training_runtime_config_hash") or ""),
        "quality_status_at_save": str(ctx.get("quality_status_at_save") or "not_evaluated"),
        "quality_status": str(ctx.get("quality_status") or ctx.get("quality_status_at_save") or "not_evaluated"),
        "downstream_ready": bool(ctx.get("downstream_ready", False)),
        "grad_inf_count_until_epoch": int(ctx.get("grad_inf_count_until_epoch", 0) or 0),
        "model_file_hash": checkpoint_file_sha256(checkpoint_abs),
        "optimizer_state_hash": str(ctx.get("optimizer_state_hash") or ""),
        "code_commit": str(ctx.get("code_commit") or _git_code_fingerprint_for_step3().get("git_commit", "")),
        "created_at": _utc_now(),
        "git_code_fingerprint": _git_code_fingerprint_for_step3(),
        "one_control_resolved_config_path": resolved_config.get("resolved_config_path"),
        "one_control_resolved_config_hash": resolved_config["hash"],
        "resolved_config_hash": resolved_config["hash"],
        "resolved_config": resolved_config,
        "resolved_config_compatibility": resolved_compatibility,
        "resolved_config_compatibility_hash": stable_hash(resolved_compatibility),
        "source_table_path": source_table.get("source_table_path"),
        "source_table_hash": source_table["hash"],
        "source_table": source_table,
        "source_table_compatibility": source_table_compatibility,
        "source_table_compatibility_hash": stable_hash(source_table_compatibility),
        "source_table_payload_summary": source_table.get("source_table_payload_summary"),
        "full_run_config_hash": resolved_config["hash"],
        "train_runtime_config_hash": stable_hash(train_runtime_payload),
        "training_runtime_config_path": training_runtime_config.get("training_runtime_config_path"),
        "training_runtime_config_hash": training_runtime_config["hash"],
        "training_runtime_config": training_runtime_config,
        "optimizer_config_hash": stable_hash(optimizer_payload),
        "performance_profile_hash": stable_hash(performance_profile_payload),
        "loss_config_hash": stable_hash(loss_config_payload),
        "loss_config": loss_config_payload,
        "semantic_model_compat_hash": stable_hash(semantic_model_payload),
        "data_contract_hash": stable_hash(data_contract_payload),
        "artifact_lineage_hash": stable_hash(artifact_lineage_payload),
        "tokenizer_cache_compat_hash": tokenizer_cache_compat_hash,
        "semantic_model_compatibility": semantic_model_payload,
        "data_contract": data_contract_payload,
        "artifact_lineage_contract": artifact_lineage_payload,
        "train_runtime_config": train_runtime_payload,
        "optimizer_runtime_config": optimizer_payload,
        "performance_profile": performance_profile_payload,
        "task_profile": payload.get("step3_task_profile"),
        "training_semantic_fingerprint": os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT", ""),
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "artifact_lineage": data_fps,
        "data_merged_artifact_fingerprint": stable_hash(data_fps),
        "step3_upstream_preprocess_gate": upstream_evidence,
        "step3_upstream_preprocess_gate_hash": upstream_evidence.get("fingerprint_hash"),
        **preprocess_sections,
        "profile_artifact_fingerprints": upstream_evidence.get("profile_artifact_fingerprints"),
        "profile_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("profile_artifact_fingerprints") or {}),
        "domain_artifact_fingerprints": upstream_evidence.get("domain_artifact_fingerprints"),
        "domain_artifact_fingerprints_hash": stable_hash(upstream_evidence.get("domain_artifact_fingerprints") or {}),
        "source_csv_fingerprints": upstream_evidence.get("source_csv_artifacts"),
        "source_csv_fingerprints_hash": stable_hash(upstream_evidence.get("source_csv_artifacts") or {}),
        "merged_csv_fingerprints": upstream_evidence.get("merged_artifacts") or data_fps,
        "merged_csv_fingerprints_hash": stable_hash(upstream_evidence.get("merged_artifacts") or data_fps),
        "sentence_embed_model_identity": _step3_sentence_embed_model_identity(),
        "step3_tokenizer_cache_manifest": tokenizer_cache_manifest,
        "step3_tokenizer_cache_manifest_hash": stable_hash(tokenizer_cache_manifest),
        "schema_contract_versions": {
            "lineage_gate_schema_version": "odcr_lineage_gate/4A",
            "step3_checkpoint_sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
            "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
            "step3_upstream_gate_schema_version": upstream_evidence.get("schema_version"),
            "step3_upstream_contract_schema_version": upstream_evidence.get("contract_schema_version"),
            "step3_tokenizer_cache_schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        },
        "embed_dim": int(final_cfg.emsize),
        "env": {
            "embed_dim": int(get_odcr_embed_dim()),
            "resolved_embed_dim_env_key": "ODCR_RESOLVED_EMBED_DIM",
        },
        "profile_dims": {
            "user_count": int(final_cfg.nuser),
            "item_count": int(final_cfg.nitem),
            "emsize": int(final_cfg.emsize),
        },
        "step3_structured_losses_config_hash": stable_hash(structured),
        "step3_structured_losses_config": structured,
        "step3_loss_semantics_config_hash": stable_hash(loss_semantics),
        "step3_loss_semantics_config": loss_semantics,
        "step3_optimizer_config": optimizer_config,
        "step3_optimizer_config_hash": stable_hash(optimizer_config),
        "step3_tokenizer_config": tokenizer_config,
        "step3_tokenizer_config_hash": stable_hash(tokenizer_config),
        "step3_evidence_config": evidence_config,
        "step3_evidence_config_hash": stable_hash(evidence_config),
        "step3_scheduler_config": scheduler_config,
        "step3_scheduler_config_hash": stable_hash(scheduler_config),
        "step3_valid_batch_config": valid_batch_config,
        "step3_valid_batch_config_hash": stable_hash(valid_batch_config),
        "step3_scenario_profile": scenario_profile,
        "step3_scenario_profile_hash": stable_hash(scenario_profile),
        "step3_task_profile": payload.get("step3_task_profile"),
        "step3_task_profile_hash": stable_hash(payload.get("step3_task_profile") or {}),
        "step3_cross_rank_structured_gather": payload.get("step3_cross_rank_structured_gather"),
        "step3_cross_rank_structured_gather_hash": stable_hash(payload.get("step3_cross_rank_structured_gather") or {}),
        "model_architecture_config": arch,
        "model_architecture_config_hash": stable_hash(arch),
        "ddp_config": _step3_ddp_config(final_cfg),
        "precision_config": {
            "train_precision": str(final_cfg.train_precision),
            "allow_tf32": bool(final_cfg.allow_tf32),
            "amp_autocast": bool(final_cfg.amp_autocast),
            "grad_scaler": bool(final_cfg.grad_scaler),
            "runtime_precision_env": os.environ.get("ODCR_RUNTIME_PRECISION_MODE", ""),
            "runtime_allow_tf32_env": os.environ.get("ODCR_RUNTIME_ALLOW_TF32", ""),
        },
        "batch_semantics": _step3_batch_semantics(final_cfg),
        "source_task": {
            "task_id": task_idx,
            "auxiliary": str(final_cfg.auxiliary),
            "target": str(final_cfg.target),
            "scenario": str(getattr(final_cfg, "scenario", "") or payload.get("scenario") or ""),
            "direction": str(getattr(final_cfg, "direction", "") or payload.get("direction") or ""),
            "task_profile_id": str(getattr(final_cfg, "task_profile_id", "") or payload.get("task_profile_id") or ""),
        },
        "compatibility_metadata": _step3_checkpoint_compatibility_metadata(),
        "metrics_summary": {
            "status": "placeholder",
            "reason": "checkpoint sidecar is written during training; epoch metrics remain in run meta when available",
        },
        "effective_payload_schema_version": payload.get("schema_version"),
    }
    lineage["ddp_config_hash"] = stable_hash(lineage["ddp_config"])
    lineage["precision_config_hash"] = stable_hash(lineage["precision_config"])
    lineage["batch_semantics_hash"] = stable_hash(lineage["batch_semantics"])
    lineage["checkpoint_compatibility_hash"] = stable_hash(lineage)
    return lineage

# HuggingFace tokenize 磁盘缓存：修改 Processor/tokenize 语义或需强制失效时与 step5 引擎同步递增
ODCR_TOKENIZE_CACHE_VERSION = "v8_step3_tokenizer_cache_v2"

tasks = [
    ("AM_Electronics", "AM_CDs"),
    ("AM_Movies", "AM_CDs"),
    ("AM_CDs", "AM_Electronics"),
    ("AM_Movies", "AM_Electronics"),
    ("AM_CDs", "AM_Movies"),
    ("AM_Electronics", "AM_Movies"),
    ("Yelp", "TripAdvisor"),
    ("TripAdvisor", "Yelp")
]


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        hidden_states = []
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            hidden_states.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return output, hidden_states


_POLARITY_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


@dataclass
class Step3EvidenceBatch:
    content_evidence_ids: torch.Tensor
    style_evidence_ids: torch.Tensor
    domain_style_anchor_ids: torch.Tensor
    local_style_hint_ids: torch.Tensor
    polarity_ids: torch.Tensor
    evidence_quality_prior: torch.Tensor


class Processor:
    def __init__(
        self,
        auxiliary,
        target,
        *,
        max_length: int,
        evidence_length: int,
        length_protocol: str = ODCR_STEP3_DIAGNOSTIC,
    ):
        self.max_length = int(max_length)
        self.evidence_length = int(evidence_length)
        self.length_protocol = normalize_eval_protocol(length_protocol) if str(length_protocol) else ODCR_STEP3_DIAGNOSTIC
        if self.max_length <= 0 or self.evidence_length <= 0:
            raise ValueError("Step3 Processor max_length/evidence_length must be positive resolved values.")
        paper_25_allowed = self.length_protocol == PAPER_TARGET_ONLY_EVAL and self.max_length == 25
        if (self.max_length in (24, 25) and not paper_25_allowed) or self.evidence_length in (24, 25):
            raise ValueError("Step3 Processor refused retired 24/25 legacy lengths.")
        self.auxiliary = auxiliary
        self.target = target

    def _encode_text(self, text: str, *, max_length: int) -> torch.Tensor:
        ids = get_odcr_text_tokenizer()(
            str(text),
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def _required_text(self, sample, key: str) -> str:
        if key not in sample:
            raise KeyError(f"Step3 sample 缺少 canonical 字段 {key!r}")
        value = sample[key]
        return "" if value is None else str(value)

    def _required_float(self, sample, key: str) -> torch.Tensor:
        if key not in sample:
            raise KeyError(f"Step3 sample 缺少 canonical 字段 {key!r}")
        return torch.tensor(float(sample[key]), dtype=torch.float32)

    def __call__(self, sample):
        user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
        item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
        rating = torch.tensor(sample["rating"], dtype=torch.float)
        explanation_idx = self._encode_text(sample["explanation"], max_length=self.max_length)
        content_evidence_ids = self._encode_text(
            self._required_text(sample, "content_evidence"),
            max_length=self.evidence_length,
        )
        style_evidence_ids = self._encode_text(
            self._required_text(sample, "style_evidence"),
            max_length=self.evidence_length,
        )
        domain_style_anchor_ids = self._encode_text(
            self._required_text(sample, "domain_style_anchor"),
            max_length=self.evidence_length,
        )
        local_style_hint_ids = self._encode_text(
            self._required_text(sample, "local_style_residual_hint"),
            max_length=self.evidence_length,
        )
        polarity_text = self._required_text(sample, "polarity_anchor").strip().lower()
        if polarity_text not in _POLARITY_TO_ID:
            raise ValueError(f"Step3 sample polarity_anchor 非法: {polarity_text!r}")
        polarity_ids = torch.tensor(_POLARITY_TO_ID[polarity_text], dtype=torch.long)

        if sample["domain"] == "auxiliary":
            domain_val = 0
        elif sample["domain"] == "target":
            domain_val = 1
        else:
            raise ValueError("Unknown domain!")

        domain_idx = torch.tensor(domain_val, dtype=torch.long)
        sample_id = torch.tensor(int(sample["sample_id"]), dtype=torch.long)
        c_anchor = self._required_float(sample, "content_anchor_score")
        s_anchor = self._required_float(sample, "style_anchor_score")
        evidence_quality_prior = self._required_float(sample, "evidence_quality_prior")
        return {
            "user_idx": user_idx,
            "item_idx": item_idx,
            "rating": rating,
            "explanation_idx": explanation_idx,
            "domain_idx": domain_idx,
            "sample_id": sample_id,
            "content_anchor_score": c_anchor,
            "style_anchor_score": s_anchor,
            "content_evidence_ids": content_evidence_ids,
            "style_evidence_ids": style_evidence_ids,
            "domain_style_anchor_ids": domain_style_anchor_ids,
            "local_style_hint_ids": local_style_hint_ids,
            "polarity_ids": polarity_ids,
            "evidence_quality_prior": evidence_quality_prior,
        }


class PETER_MLP(nn.Module):
    def __init__(self, emsize=512):
        super().__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):
        mlp_vector = self.sigmoid(self.linear1(hidden))
        rating = self.linear2(mlp_vector).view(-1)
        return rating


def _domain_fusion_causal_mask(tgt_len: int, device: torch.device, prefix_len: int = 10) -> torch.Tensor:
    total_len = prefix_len + tgt_len
    mask = torch.triu(torch.ones((total_len, total_len), device=device, dtype=torch.bool), diagonal=1)
    mask[:prefix_len, :prefix_len] = False
    return mask


class Model(nn.Module):
    def __init__(
        self,
        nuser,
        nitem,
        ntoken,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout,
        user_content_profiles,
        user_style_profiles,
        item_content_profiles,
        item_style_profiles,
        domain_content_profiles,
        domain_style_profiles,
    ):
        super().__init__()
        self.register_buffer("domain_content_profiles", domain_content_profiles.detach(), persistent=False)
        self.register_buffer("domain_style_profiles", domain_style_profiles.detach(), persistent=False)
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.register_buffer("user_content_profiles", user_content_profiles.detach(), persistent=False)
        self.register_buffer("user_style_profiles", user_style_profiles.detach(), persistent=False)
        self.register_buffer("item_content_profiles", item_content_profiles.detach(), persistent=False)
        self.register_buffer("item_style_profiles", item_style_profiles.detach(), persistent=False)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.recommender = PETER_MLP(emsize)
        self.odcr_disentangler = ODCRDisentangler(hidden_size=emsize, proj_size=emsize, dropout=dropout)
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.shared_id_adapter = nn.Linear(emsize, emsize)
        self.specific_id_adapter = nn.Linear(emsize, emsize)
        self.shared_stream_attn = nn.MultiheadAttention(
            embed_dim=emsize, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.specific_stream_attn = nn.MultiheadAttention(
            embed_dim=emsize, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.shared_stream_norm = nn.LayerNorm(emsize)
        self.specific_stream_norm = nn.LayerNorm(emsize)
        self.evidence_pool_norm = nn.LayerNorm(emsize)
        self.polarity_embedding = nn.Embedding(3, emsize)
        self.decoder_eos_id = -1
        safe_decode = SAFE_DECODE_PLACEHOLDER
        self.decode_strategy = str(safe_decode["decode_strategy"])
        self.generate_temperature = float(safe_decode["generate_temperature"])
        self.generate_top_p = float(safe_decode["generate_top_p"])
        self.repetition_penalty = float(safe_decode["repetition_penalty"])
        self.max_explanation_length = int(safe_decode["max_explanation_length"])
        self.no_repeat_ngram_size = 0
        self.min_len = 0
        self.soft_max_len = 0
        self.hard_max_len = 25
        self.eos_boost_start = 9999
        self.eos_boost_value = 0.0
        self.tail_temperature = -1.0
        self.tail_top_p = -1.0
        self.decode_token_repeat_window = 4
        self.decode_token_repeat_max = 2
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layers, nlayers)
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.emsize = emsize
        self.ntoken = int(ntoken)
        self.evidence_length = 0
        self.rating_loss_fn = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.profile_buffer_policy = "gpu_resident"
        self.activation_checkpointing_policy = {"enabled": False, "policy": "selective", "modules": []}
        self.last_odcr_latents = None
        self.last_shared_proj: torch.Tensor | None = None
        self.last_specific_proj: torch.Tensor | None = None
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.08, 0.08)

    def apply_runtime_config(self, cfg: FinalTrainingConfig, tok) -> None:
        self.decode_strategy = str(cfg.decode_strategy).strip().lower()
        self.generate_temperature = max(float(cfg.generate_temperature), 1e-8)
        self.generate_top_p = float(cfg.generate_top_p)
        self.repetition_penalty = float(cfg.repetition_penalty)
        self.max_explanation_length = int(cfg.max_explanation_length)
        self.no_repeat_ngram_size = int(getattr(cfg, "no_repeat_ngram_size", 0) or 0)
        self.min_len = int(getattr(cfg, "min_len", 0) or 0)
        self.soft_max_len = int(getattr(cfg, "soft_max_len", 0) or 0)
        self.hard_max_len = int(getattr(cfg, "hard_max_len", self.max_explanation_length) or self.max_explanation_length)
        self.eos_boost_start = int(getattr(cfg, "eos_boost_start", 9999))
        self.eos_boost_value = float(getattr(cfg, "eos_boost_value", 0.0))
        self.tail_temperature = float(getattr(cfg, "tail_temperature", -1.0))
        self.tail_top_p = float(getattr(cfg, "tail_top_p", -1.0))
        self.decode_token_repeat_window = int(getattr(cfg, "decode_token_repeat_window", 4))
        self.decode_token_repeat_max = int(getattr(cfg, "decode_token_repeat_max", 2))
        self.evidence_length = int(getattr(cfg, "evidence_max_length", 0) or 0)
        if self.evidence_length <= 0:
            raise RuntimeError("Step3 model requires resolved evidence_max_length.")
        eid = getattr(tok, "eos_token_id", None)
        self.decoder_eos_id = int(eid) if eid is not None else -1

    def _profile_lookup(self, name: str, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
        value = getattr(self, name)
        if value.device == device:
            return value[indices]
        cpu_indices = indices.detach().to(device=value.device, non_blocking=False)
        selected = value[cpu_indices]
        return selected.to(device=device, non_blocking=True)

    def _pool_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        tok = token_ids.long().clamp_min(0).clamp_max(self.ntoken - 1)
        emb = self.word_embeddings(tok)
        mask = tok.ne(0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = (emb * mask).sum(dim=1) / denom
        return self.evidence_pool_norm(pooled)

    def _default_evidence_batch(self, batch_size: int, device: torch.device) -> Step3EvidenceBatch:
        zeros = torch.zeros((batch_size, self.evidence_length), dtype=torch.long, device=device)
        polarity = torch.full((batch_size,), _POLARITY_TO_ID["neutral"], dtype=torch.long, device=device)
        quality = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)
        return Step3EvidenceBatch(
            content_evidence_ids=zeros,
            style_evidence_ids=zeros,
            domain_style_anchor_ids=zeros.clone(),
            local_style_hint_ids=zeros.clone(),
            polarity_ids=polarity,
            evidence_quality_prior=quality,
        )

    def _resolve_evidence_batch(
        self,
        *,
        batch_size: int,
        device: torch.device,
        content_evidence_ids: torch.Tensor | None,
        style_evidence_ids: torch.Tensor | None,
        domain_style_anchor_ids: torch.Tensor | None,
        local_style_hint_ids: torch.Tensor | None,
        polarity_ids: torch.Tensor | None,
        evidence_quality_prior: torch.Tensor | None,
    ) -> Step3EvidenceBatch:
        default = self._default_evidence_batch(batch_size, device)
        return Step3EvidenceBatch(
            content_evidence_ids=default.content_evidence_ids if content_evidence_ids is None else content_evidence_ids.long(),
            style_evidence_ids=default.style_evidence_ids if style_evidence_ids is None else style_evidence_ids.long(),
            domain_style_anchor_ids=(
                default.domain_style_anchor_ids if domain_style_anchor_ids is None else domain_style_anchor_ids.long()
            ),
            local_style_hint_ids=default.local_style_hint_ids if local_style_hint_ids is None else local_style_hint_ids.long(),
            polarity_ids=default.polarity_ids if polarity_ids is None else polarity_ids.long(),
            evidence_quality_prior=(
                default.evidence_quality_prior
                if evidence_quality_prior is None
                else evidence_quality_prior.to(dtype=torch.float32)
            ),
        )

    def _build_stream_seeds(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        domain_idx: torch.Tensor,
        evidence: Step3EvidenceBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        device = user_emb.device
        domain_content = self._profile_lookup("domain_content_profiles", domain_idx, device)
        domain_style = self._profile_lookup("domain_style_profiles", domain_idx, device)
        user_content = self._profile_lookup("user_content_profiles", user, device)
        user_style = self._profile_lookup("user_style_profiles", user, device)
        item_content = self._profile_lookup("item_content_profiles", item, device)
        item_style = self._profile_lookup("item_style_profiles", item, device)

        content_guide = self._pool_token_embeddings(evidence.content_evidence_ids)
        style_guide = self._pool_token_embeddings(evidence.style_evidence_ids)
        domain_style_guide = self._pool_token_embeddings(evidence.domain_style_anchor_ids)
        local_style_guide = self._pool_token_embeddings(evidence.local_style_hint_ids)
        polarity_guide = self.polarity_embedding(evidence.polarity_ids)

        shared_tokens = torch.stack(
            [
                domain_content,
                user_content,
                item_content,
                self.shared_id_adapter(user_emb),
                self.shared_id_adapter(item_emb),
                content_guide,
            ],
            dim=1,
        )
        specific_tokens = torch.stack(
            [
                domain_style,
                user_style,
                item_style,
                self.specific_id_adapter(user_emb),
                self.specific_id_adapter(item_emb),
                style_guide,
                domain_style_guide,
                local_style_guide,
                polarity_guide,
            ],
            dim=1,
        )
        shared_ctx, _ = self.shared_stream_attn(
            content_guide.unsqueeze(1),
            shared_tokens,
            shared_tokens,
            need_weights=False,
        )
        specific_query = (style_guide + domain_style_guide + local_style_guide + polarity_guide).unsqueeze(1)
        specific_ctx, _ = self.specific_stream_attn(
            specific_query,
            specific_tokens,
            specific_tokens,
            need_weights=False,
        )
        shared_seed = self.shared_stream_norm(shared_ctx.squeeze(1) + shared_tokens.mean(dim=1))
        specific_seed = self.specific_stream_norm(specific_ctx.squeeze(1) + specific_tokens.mean(dim=1))
        guides = {
            "content_guide": content_guide,
            "style_guide": style_guide,
            "domain_style_guide": domain_style_guide,
            "local_style_guide": local_style_guide,
            "polarity_guide": polarity_guide,
            "domain_content": domain_content,
            "user_content": user_content,
            "item_content": item_content,
            "user_style": user_style,
            "item_style": item_style,
        }
        return shared_seed, specific_seed, guides

    def _build_prefix(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        domain_idx: torch.Tensor,
        latents,
        guides: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.stack(
            [
                latents.shared,
                latents.specific,
                guides["domain_content"],
                latents.domain_style_component,
                guides["user_content"],
                guides["item_content"],
                guides["user_style"],
                guides["item_style"],
                latents.content_evidence_target,
                latents.style_evidence_target,
            ],
            dim=1,
        )

    def _prefix_len(self) -> int:
        return 10

    def _make_generate_config(self) -> GenerateConfig:
        return GenerateConfig(
            strategy=str(self.decode_strategy).lower(),
            temperature=float(self.generate_temperature),
            top_p=float(self.generate_top_p),
            repetition_penalty=float(self.repetition_penalty),
            no_repeat_ngram_size=max(0, int(self.no_repeat_ngram_size)),
            min_len=max(0, int(self.min_len)),
            soft_max_len=max(0, int(self.soft_max_len)),
            hard_max_len=max(1, int(self.hard_max_len)),
            eos_boost_start=int(self.eos_boost_start),
            eos_boost_value=float(self.eos_boost_value),
            tail_temperature=float(self.tail_temperature),
            tail_top_p=float(self.tail_top_p),
            token_repeat_window=int(self.decode_token_repeat_window),
            token_repeat_max=int(self.decode_token_repeat_max),
        )

    def _compute_latents(
        self,
        user: torch.Tensor,
        item: torch.Tensor,
        domain_idx: torch.Tensor,
        *,
        content_anchor: torch.Tensor | None,
        style_anchor: torch.Tensor | None,
        content_evidence_ids: torch.Tensor | None,
        style_evidence_ids: torch.Tensor | None,
        domain_style_anchor_ids: torch.Tensor | None,
        local_style_hint_ids: torch.Tensor | None,
        polarity_ids: torch.Tensor | None,
        evidence_quality_prior: torch.Tensor | None,
    ) -> tuple[Any, Dict[str, torch.Tensor], Step3EvidenceBatch]:
        device = user.device
        batch_size = int(user.size(0))
        if content_anchor is None:
            content_anchor = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
        if style_anchor is None:
            style_anchor = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
        evidence = self._resolve_evidence_batch(
            batch_size=batch_size,
            device=device,
            content_evidence_ids=content_evidence_ids,
            style_evidence_ids=style_evidence_ids,
            domain_style_anchor_ids=domain_style_anchor_ids,
            local_style_hint_ids=local_style_hint_ids,
            polarity_ids=polarity_ids,
            evidence_quality_prior=evidence_quality_prior,
        )
        shared_seed, specific_seed, guides = self._build_stream_seeds(user, item, domain_idx, evidence)
        latents = self.odcr_disentangler(
            shared_seed,
            specific_seed,
            domain_idx.view(-1),
            content_guide=guides["content_guide"],
            style_guide=guides["style_guide"],
            domain_style_guide=guides["domain_style_guide"],
            local_style_guide=guides["local_style_guide"],
            polarity_guide=guides["polarity_guide"],
            content_anchor_score=content_anchor.view(-1),
            style_anchor_score=style_anchor.view(-1),
        )
        return latents, guides, evidence

    def forward(
        self,
        user,
        item,
        tgt_input,
        domain_idx,
        *,
        content_anchor=None,
        style_anchor=None,
        content_evidence_ids=None,
        style_evidence_ids=None,
        domain_style_anchor_ids=None,
        local_style_hint_ids=None,
        polarity_ids=None,
        evidence_quality_prior=None,
    ):
        device = user.device
        latents, guides, evidence = self._compute_latents(
            user,
            item,
            domain_idx,
            content_anchor=content_anchor,
            style_anchor=style_anchor,
            content_evidence_ids=content_evidence_ids,
            style_evidence_ids=style_evidence_ids,
            domain_style_anchor_ids=domain_style_anchor_ids,
            local_style_hint_ids=local_style_hint_ids,
            polarity_ids=polarity_ids,
            evidence_quality_prior=evidence_quality_prior,
        )
        prefix = self._build_prefix(user, item, domain_idx, latents, guides)
        word_feature = self.word_embeddings(tgt_input)
        src = torch.cat([prefix, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = _domain_fusion_causal_mask(tgt_input.shape[1], device, prefix_len=self._prefix_len())
        hidden, _ = self.transformer_encoder(src=src, mask=attn_mask)
        self.last_odcr_latents = _detach_odcr_latent_bundle(latents)
        self.last_shared_proj = latents.shared_proj.detach()
        self.last_specific_proj = latents.specific_proj.detach()
        rating = self.recommender(latents.shared)
        context_logits = self.hidden2token(latents.specific).unsqueeze(1)
        context_dist = context_logits.repeat(1, tgt_input.shape[1], 1)
        word_dist = self.hidden2token(hidden[:, self._prefix_len():])
        return Step3ForwardOutput(
            rating=rating,
            context_dist=context_dist,
            word_dist=word_dist,
            shared_proj=latents.shared_proj,
            specific_proj=latents.specific_proj,
            odcr_latents=latents,
            structured_loss_inputs=_structured_loss_inputs_from_latents(latents),
            diagnostics={
                "prefix_len": self._prefix_len(),
                "structured_loss_input_keys": STEP3_STRUCTURED_LOSS_INPUT_KEYS,
            },
        )

    def gather(self, batch, device):
        (
            user_idx,
            item_idx,
            rating,
            tgt_output,
            domain_idx,
            sample_id,
            content_anchor_score,
            style_anchor_score,
            content_evidence_ids,
            style_evidence_ids,
            domain_style_anchor_ids,
            local_style_hint_ids,
            polarity_ids,
            evidence_quality_prior,
        ) = batch
        non_blocking = bool(getattr(self, "non_blocking_h2d", True))
        user_idx = user_idx.to(device, non_blocking=non_blocking)
        item_idx = item_idx.to(device, non_blocking=non_blocking)
        domain_idx = domain_idx.to(device, non_blocking=non_blocking)
        rating = rating.to(device, non_blocking=non_blocking).float()
        tgt_output = tgt_output.to(device, non_blocking=non_blocking)
        sample_id = sample_id.to(device, non_blocking=non_blocking)
        content_anchor_score = content_anchor_score.to(device, non_blocking=non_blocking).float()
        style_anchor_score = style_anchor_score.to(device, non_blocking=non_blocking).float()
        content_evidence_ids = content_evidence_ids.to(device, non_blocking=non_blocking).long()
        style_evidence_ids = style_evidence_ids.to(device, non_blocking=non_blocking).long()
        domain_style_anchor_ids = domain_style_anchor_ids.to(device, non_blocking=non_blocking).long()
        local_style_hint_ids = local_style_hint_ids.to(device, non_blocking=non_blocking).long()
        polarity_ids = polarity_ids.to(device, non_blocking=non_blocking).long()
        evidence_quality_prior = evidence_quality_prior.to(device, non_blocking=non_blocking).float()
        tgt_input = T5_shift_right(tgt_output)
        return GatheredBatch(
            user_idx=user_idx,
            item_idx=item_idx,
            rating=rating,
            tgt_input=tgt_input,
            tgt_output=tgt_output,
            domain_idx=domain_idx,
            sample_id=sample_id,
            exp_sample_weight=None,
            content_anchor_score=content_anchor_score,
            style_anchor_score=style_anchor_score,
            content_evidence_ids=content_evidence_ids,
            style_evidence_ids=style_evidence_ids,
            domain_style_anchor_ids=domain_style_anchor_ids,
            local_style_hint_ids=local_style_hint_ids,
            polarity_ids=polarity_ids,
            evidence_quality_prior=evidence_quality_prior,
        )

    def recommend(
        self,
        user,
        item,
        domain,
        *,
        content_anchor=None,
        style_anchor=None,
        content_evidence_ids=None,
        style_evidence_ids=None,
        domain_style_anchor_ids=None,
        local_style_hint_ids=None,
        polarity_ids=None,
        evidence_quality_prior=None,
    ):
        latents, _, _ = self._compute_latents(
            user,
            item,
            domain,
            content_anchor=content_anchor,
            style_anchor=style_anchor,
            content_evidence_ids=content_evidence_ids,
            style_evidence_ids=style_evidence_ids,
            domain_style_anchor_ids=domain_style_anchor_ids,
            local_style_hint_ids=local_style_hint_ids,
            polarity_ids=polarity_ids,
            evidence_quality_prior=evidence_quality_prior,
        )
        rating = self.recommender(latents.shared)
        return rating

    def generate(
        self,
        user,
        item,
        domain,
        *,
        content_anchor=None,
        style_anchor=None,
        content_evidence_ids=None,
        style_evidence_ids=None,
        domain_style_anchor_ids=None,
        local_style_hint_ids=None,
        polarity_ids=None,
        evidence_quality_prior=None,
    ):
        total_entropies = []
        gc = self._make_generate_config()
        max_len = int(gc.hard_max_len)
        bos_idx = 0
        device = user.device
        batch_size = user.shape[0]
        latents, guides, _ = self._compute_latents(
            user,
            item,
            domain,
            content_anchor=content_anchor,
            style_anchor=style_anchor,
            content_evidence_ids=content_evidence_ids,
            style_evidence_ids=style_evidence_ids,
            domain_style_anchor_ids=domain_style_anchor_ids,
            local_style_hint_ids=local_style_hint_ids,
            polarity_ids=polarity_ids,
            evidence_quality_prior=evidence_quality_prior,
        )
        prefix = self._build_prefix(user, item, domain, latents, guides)
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)
        eos_id = int(self.decoder_eos_id)
        recent = [[] for _ in range(batch_size)]
        for i in range(max_len):
            gen_so_far = int(decoder_input_ids.shape[1]) - 1
            word_feature = self.word_embeddings(decoder_input_ids)
            src = torch.cat([prefix, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)
            attn_mask = _domain_fusion_causal_mask(decoder_input_ids.shape[1], device, prefix_len=self._prefix_len())
            hidden, _ = self.transformer_encoder(src=src, mask=attn_mask)
            logits = prepare_logits(hidden[:, -1, :], self.hidden2token)
            logits = apply_repetition_penalty_logits(logits, decoder_input_ids, float(gc.repetition_penalty))
            if gc.no_repeat_ngram_size > 0:
                apply_no_repeat_ngram_logits(logits, decoder_input_ids, int(gc.no_repeat_ngram_size))
            apply_token_repeat_suppression(
                logits,
                recent,
                window=int(gc.token_repeat_window),
                max_same=int(gc.token_repeat_max),
            )
            apply_min_len_eos_mask(logits, eos_id=eos_id, gen_so_far=gen_so_far, min_len=int(gc.min_len))
            eff_t, eff_p = apply_sampling_schedule(gc, gen_so_far)
            apply_eos_boost(logits, eos_id=eos_id, step=gen_so_far, cfg=gc)
            output_id, entropies, _, _ = sample_next_token(
                logits,
                strategy=gc.strategy,
                temperature=eff_t,
                top_p=eff_p,
                generator=None,
            )
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
            for b in range(batch_size):
                recent[b].append(int(output_id[b, 0].item()))
            total_entropies.append(entropies)
            if eos_id >= 0 and bool((output_id.squeeze(-1) == eos_id).all()):
                break
        total_entropies = torch.stack(total_entropies).mean(dim=0)
        return decoder_input_ids[:,1:], total_entropies


def validModel(model, valid_dataloader, device):
    _model = get_underlying_model(model)
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            g = require_gathered_batch(_model.gather(batch, device))
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = (
                g.user_idx,
                g.item_idx,
                g.rating,
                g.tgt_input,
                g.tgt_output,
                g.domain_idx,
            )
            ca = g.content_anchor_score
            sa = g.style_anchor_score
            ce = g.content_evidence_ids
            se = g.style_evidence_ids
            dsa = g.domain_style_anchor_ids
            lsh = g.local_style_hint_ids
            pol = g.polarity_ids
            eq = g.evidence_quality_prior
            if any(value is None for value in (ca, sa, ce, se, dsa, lsh, pol, eq)):
                raise RuntimeError("validModel gather 缺少 Step3 canonical evidence 张量。")
            with odcr_cuda_bf16_autocast():
                out = model(
                    user_idx,
                    item_idx,
                    tgt_input,
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
            loss_r = F.mse_loss(out.rating, rating, reduction="mean")
            loss_e = F.cross_entropy(out.word_dist.view(-1, out.word_dist.size(-1)), tgt_output.reshape(-1), ignore_index=0)
            loss = loss_r + loss_e
            avg_loss += loss.item()
        avg_loss /= len(valid_dataloader)
        return avg_loss


def validModel_sum_batches(model, valid_dataloader, device):
    """
    DDP 训练时用于验证聚合（与 train loss、Step5 valid 一致：样本加权）：
    - 返回 (loss_sum, n_samples)，其中 loss_sum = Σ (batch 标量 loss × batch 内样本数)。
    - 外层对两维做 all_reduce(SUM) 后，current_valid_loss = 全局 loss_sum / 全局 n_samples。
    - 验证集在各 rank 上为无重叠划分（见 build_config_and_data_ddp 中 Subset + 索引），
      避免 DistributedSampler 补重复样本导致全局均值偏差。
    """
    _model = get_underlying_model(model)
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        n_samples = 0
        for batch in valid_dataloader:
            g = require_gathered_batch(_model.gather(batch, device))
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = (
                g.user_idx,
                g.item_idx,
                g.rating,
                g.tgt_input,
                g.tgt_output,
                g.domain_idx,
            )
            ca = g.content_anchor_score
            sa = g.style_anchor_score
            ce = g.content_evidence_ids
            se = g.style_evidence_ids
            dsa = g.domain_style_anchor_ids
            lsh = g.local_style_hint_ids
            pol = g.polarity_ids
            eq = g.evidence_quality_prior
            if any(value is None for value in (ca, sa, ce, se, dsa, lsh, pol, eq)):
                raise RuntimeError("validModel_sum_batches gather 缺少 Step3 canonical evidence 张量。")
            bsz = int(user_idx.size(0))
            with odcr_cuda_bf16_autocast():
                out = model(
                    user_idx,
                    item_idx,
                    tgt_input,
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
            loss_r = F.mse_loss(out.rating, rating, reduction="mean")
            loss_e = F.cross_entropy(out.word_dist.view(-1, out.word_dist.size(-1)), tgt_output.reshape(-1), ignore_index=0)
            loss = loss_r + loss_e
            loss_sum += loss.detach().double().item() * bsz
            n_samples += bsz
        return loss_sum, n_samples


def _log_tokenize_done(
    phase: str,
    nproc: int,
    elapsed_s: float,
    log_file: Optional[str],
    *,
    also_print: bool = True,
) -> None:
    """datasets.map（desc=Tokenize）结束后的显式耗时与 num_proc，便于在日志中检索。"""
    msg = f"[Tokenize] {phase} 完成 | num_proc={nproc} | wall_time={elapsed_s:.2f}s"
    lg = logging.getLogger(LOGGER_NAME)
    # 已由 FileHandler 写文件时不再 print，避免与 shell 重定向双份
    if also_print and not logger_has_file_handler(lg):
        print(msg, flush=True)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


_STEP3_PREFETCH_TIMING_FIELDS = (
    "dataloader_next_wait",
    "h2d_prefetch_time",
    "compute_wait_for_prefetch",
    "forward_time",
    "loss_time",
    "backward_time",
    "structured_gather_ms",
    "finite_sync_ms",
    "duplicate_loss_check_ms",
    "ddp_backward_sync_ms",
    "backward_compute_ms",
    "grad_check_ms",
    "grad_norm_compute_ms",
    "grad_clip_ms",
    "grad_monitor_ms",
    "nonfinite_detect_ms",
    "optimizer_time",
    "optimizer_ms",
    "ema_ms",
    "zero_grad_ms",
    "scheduler_time",
    "scheduler_ms",
    "metrics_io_ms",
    "sync_time",
    "logging_time",
    "checkpoint_io_ms",
    "cuda_sync_ms",
    "total_step_time",
    "loader_next_wait_ms",
    "cpu_collate_ms",
    "h2d_submit_ms",
    "h2d_wait_ms",
    "prefetch_wait_ms",
    "optimizer_step_executed",
    "scheduler_step_executed",
    "grad_finite",
    "skipped_step_reason",
    "nonfinite_param_count",
    "nonfinite_param_topk",
    "nonfinite_param_group_topk",
    "continuous_nonfinite_steps",
)


def _step3_loss_breakdown_row(
    loss_bundle: Step3LossBundle,
    *,
    final_cfg: FinalTrainingConfig,
    rank: int,
    global_step: int,
    epoch: int,
) -> list[dict[str, Any]]:
    summary = dict(loss_bundle.logging_summary or {})
    components = summary.get("components")
    if not isinstance(components, Mapping):
        components = {}
    phase = summary.get("phase") if isinstance(summary.get("phase"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for loss_name in sorted(loss_bundle.components):
        raw_value = float(loss_bundle.components[loss_name].detach().item())
        weight = float(loss_bundle.weights.get(loss_name, 0.0))
        weighted_value = float(loss_bundle.weighted_components[loss_name].detach().item())
        component_detail = components.get(loss_name) if isinstance(components, Mapping) else None
        finite = bool(loss_bundle.finite_status.get(loss_name, True))
        if isinstance(component_detail, Mapping) and "finite" in component_detail:
            finite = bool(component_detail.get("finite"))
        rows.append(
            {
                "run_id": str(getattr(final_cfg, "run_id", "") or ""),
                "task_id": int(getattr(final_cfg, "task_idx", 0) or 0),
                "profile_id": str(getattr(final_cfg, "task_profile_id", "") or ""),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "rank": int(rank),
                "loss_name": str(loss_name),
                "loss_phase": str(phase.get("phase") or ""),
                "raw_value": raw_value,
                "weight": weight,
                "weighted_value": weighted_value,
                "finite": finite,
                "timestamp": ts,
            }
        )
    return rows


def _step3_timing_profile_row(
    step_timing: Mapping[str, Any],
    *,
    final_cfg: FinalTrainingConfig,
    rank: int,
    global_step: int,
    epoch: int,
    samples_per_sec: float,
) -> dict[str, Any]:
    base = {
        "run_id": str(getattr(final_cfg, "run_id", "") or ""),
        "task_id": int(getattr(final_cfg, "task_idx", 0) or 0),
        "profile_id": str(getattr(final_cfg, "task_profile_id", "") or ""),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "rank": int(rank),
        "samples_per_sec": float(samples_per_sec),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    row = timing_row_with_closure(step_timing, base=base)
    row["data_time"] = (
        float(step_timing.get("dataloader_next_wait", 0.0) or 0.0)
        + float(step_timing.get("h2d_prefetch_time", 0.0) or 0.0)
        + float(step_timing.get("compute_wait_for_prefetch", 0.0) or 0.0)
    )
    row["forward_time"] = float(step_timing.get("forward_time", 0.0) or step_timing.get("forward_ms", 0.0) / 1000.0 or 0.0)
    row["backward_time"] = float(step_timing.get("backward_time", 0.0) or step_timing.get("backward_compute_ms", 0.0) / 1000.0 or 0.0)
    row["optimizer_time"] = float(step_timing.get("optimizer_time", 0.0) or step_timing.get("optimizer_ms", 0.0) / 1000.0 or 0.0)
    row["step_time"] = float(step_timing.get("total_step_time", 0.0) or step_timing.get("step_total_ms", 0.0) / 1000.0 or 0.0)
    return row


def _step3_gpu_profile_row(
    *,
    final_cfg: FinalTrainingConfig,
    rank: int,
    device: int | str | torch.device,
    global_step: int | None,
    epoch: int,
    phase: str = "step",
) -> dict[str, Any]:
    dev_text = str(device)
    row: dict[str, Any] = {
        "run_id": str(getattr(final_cfg, "run_id", "") or ""),
        "task_id": int(getattr(final_cfg, "task_idx", 0) or 0),
        "profile_id": str(getattr(final_cfg, "task_profile_id", "") or ""),
        "rank": int(rank),
        "device": dev_text,
        "phase": str(phase),
        "global_step": None if global_step is None else int(global_step),
        "epoch": int(epoch),
        "allocated_gib": 0.0,
        "reserved_gib": 0.0,
        "max_allocated_gib": 0.0,
        "max_reserved_gib": 0.0,
        "reserved_minus_allocated_gib": 0.0,
        "inactive_split_gib": 0.0,
        "non_releasable_gib": 0.0,
        "cuda_malloc_retry_count": 0,
        "cuda_oom_count": 0,
        "largest_free_block": None,
        "after_empty_cache_allocated": None,
        "after_empty_cache_reserved": None,
        "memory_snapshot_path": "",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if torch.cuda.is_available():
        try:
            dev = int(device) if isinstance(device, int) else torch.device(device).index
            if dev is None:
                dev = torch.cuda.current_device()
            allocated = float(torch.cuda.memory_allocated(dev)) / (1024**3)
            reserved = float(torch.cuda.memory_reserved(dev)) / (1024**3)
            max_allocated = float(torch.cuda.max_memory_allocated(dev)) / (1024**3)
            max_reserved = float(torch.cuda.max_memory_reserved(dev)) / (1024**3)
            stats = torch.cuda.memory_stats(dev)
            inactive_split = float(stats.get("inactive_split_bytes.all.current", 0.0) or 0.0) / (1024**3)
            non_releasable = float(stats.get("inactive_split_bytes.all.current", 0.0) or 0.0) / (1024**3)
            row.update(
                {
                    "device": f"cuda:{dev}",
                    "allocated_gib": round(allocated, 6),
                    "reserved_gib": round(reserved, 6),
                    "max_allocated_gib": round(max_allocated, 6),
                    "max_reserved_gib": round(max_reserved, 6),
                    "reserved_minus_allocated_gib": round(max(reserved - allocated, 0.0), 6),
                    "inactive_split_gib": round(inactive_split, 6),
                    "non_releasable_gib": round(non_releasable, 6),
                    "cuda_malloc_retry_count": int(stats.get("num_alloc_retries", 0) or 0),
                    "cuda_oom_count": int(stats.get("num_ooms", 0) or 0),
                    "largest_free_block": stats.get("largest_free_block.all.current"),
                }
            )
        except Exception:
            pass
    return row


class Step3CUDAPrefetcher:
    """Double-buffer Step3 batches with an explicit CUDA H2D prefetch stream."""

    timing_fields = _STEP3_PREFETCH_TIMING_FIELDS

    def __init__(
        self,
        loader,
        *,
        device: int | str | torch.device,
        non_blocking: bool = True,
        enabled: bool = True,
        diagnostic_cpu_mode: bool = False,
        double_buffer: bool = True,
        fallback_policy: str = "fail_fast",
    ) -> None:
        self.loader = loader
        self.device = torch.device(device if isinstance(device, str) else f"cuda:{int(device)}" if isinstance(device, int) else device)
        self.non_blocking = bool(non_blocking)
        self.enabled = bool(enabled)
        self.diagnostic_cpu_mode = bool(diagnostic_cpu_mode)
        self.double_buffer_configured = bool(double_buffer)
        self.fallback_policy = str(fallback_policy or "fail_fast")
        self._record_stream_tensor_count = 0
        self._compute_wait_stream_count = 0
        self.last_timing: dict[str, float] = {field: 0.0 for field in self.timing_fields}
        self.last_evidence: dict[str, Any] = self._base_evidence(fallback_used=False, fallback_reason="")
        if self.enabled and (self.device.type != "cuda" or not torch.cuda.is_available()):
            if not self.diagnostic_cpu_mode:
                raise RuntimeError(
                    "Step3 CUDA prefetcher requires visible CUDA; CPU/no-CUDA is diagnostic-only and must be explicit."
                )
            self.enabled = False
        self.stream = torch.cuda.Stream(device=self.device) if self.enabled else None
        if not self.enabled:
            self.last_evidence = self._base_evidence(
                fallback_used=True,
                fallback_reason="diagnostic_cpu_mode" if self.diagnostic_cpu_mode else "disabled",
            )

    def _base_evidence(self, *, fallback_used: bool, fallback_reason: str) -> dict[str, Any]:
        active = bool(self.enabled and not fallback_used)
        return {
            "prefetcher_code_present": True,
            "prefetcher_active_in_formal_loop": active,
            "h2d_stream_created": bool(active),
            "double_buffer_configured": bool(self.double_buffer_configured),
            "double_buffer_active": bool(active and self.double_buffer_configured),
            "num_device_buffers": 2 if active and self.double_buffer_configured else (1 if active else 0),
            "record_stream_tensor_count": int(self._record_stream_tensor_count),
            "compute_wait_stream_count": int(self._compute_wait_stream_count),
            "h2d_event_elapsed_ms": 0.0,
            "h2d_wait_ms": 0.0,
            "prefetch_wait_ms": 0.0,
            "h2d_hidden_by_compute_ratio": 0.0,
            "overlap_verified": False,
            "runtime_verified": False,
            "formal_verified": False,
            "fallback_used": bool(fallback_used),
            "fallback_reason": str(fallback_reason),
        }

    def __len__(self) -> int:
        return len(self.loader)

    def _move_to_device(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=self.non_blocking)
        if isinstance(value, dict):
            return {key: self._move_to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_to_device(item) for item in value]
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*(self._move_to_device(item) for item in value))
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item) for item in value)
        if hasattr(value, "__dataclass_fields__"):
            return replace(
                value,
                **{field.name: self._move_to_device(getattr(value, field.name)) for field in fields(value)},
            )
        return value

    def _record_stream(self, value: Any, stream) -> None:
        if torch.is_tensor(value) and value.is_cuda:
            value.record_stream(stream)
            self._record_stream_tensor_count += 1
            return
        if isinstance(value, Mapping):
            for item in value.values():
                self._record_stream(item, stream)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._record_stream(item, stream)
            return
        if hasattr(value, "__dataclass_fields__"):
            for field in fields(value):
                self._record_stream(getattr(value, field.name), stream)

    def __iter__(self):
        if not self.enabled:
            for batch in self.loader:
                self.last_timing = {field: 0.0 for field in self.timing_fields}
                self.last_evidence = self._base_evidence(
                    fallback_used=True,
                    fallback_reason="diagnostic_cpu_mode" if self.diagnostic_cpu_mode else "disabled",
                )
                yield batch
            return
        iterator = iter(self.loader)
        current_stream = torch.cuda.current_stream(device=self.device)

        def _preload():
            timing = {field: 0.0 for field in self.timing_fields}
            t0 = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                return None, timing
            timing["dataloader_next_wait"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            assert self.stream is not None
            with torch.cuda.stream(self.stream):
                gpu_batch = self._move_to_device(cpu_batch)
            timing["h2d_prefetch_time"] = time.perf_counter() - t1
            timing["h2d_submit_ms"] = float(timing["h2d_prefetch_time"]) * 1000.0
            timing["prefetch_wait_ms"] = float(timing["dataloader_next_wait"]) * 1000.0
            return gpu_batch, timing

        next_batch, next_timing = _preload()
        while next_batch is not None:
            assert self.stream is not None
            wait_t0 = time.perf_counter()
            current_stream.wait_stream(self.stream)
            self._compute_wait_stream_count += 1
            next_timing["compute_wait_for_prefetch"] = time.perf_counter() - wait_t0
            next_timing["h2d_wait_ms"] = float(next_timing["compute_wait_for_prefetch"]) * 1000.0
            self._record_stream(next_batch, current_stream)
            batch = next_batch
            self.last_timing = dict(next_timing)
            self.last_evidence = self._base_evidence(fallback_used=False, fallback_reason="")
            self.last_evidence.update(
                {
                    "record_stream_tensor_count": int(self._record_stream_tensor_count),
                    "compute_wait_stream_count": int(self._compute_wait_stream_count),
                    "h2d_event_elapsed_ms": float(next_timing.get("h2d_prefetch_time", 0.0) or 0.0) * 1000.0,
                    "h2d_wait_ms": float(next_timing.get("compute_wait_for_prefetch", 0.0) or 0.0) * 1000.0,
                    "prefetch_wait_ms": float(next_timing.get("dataloader_next_wait", 0.0) or 0.0) * 1000.0,
                }
            )
            next_batch, next_timing = _preload()
            yield batch


def _tokenizer_cache_identity(tok) -> str:
    nop = getattr(tok, "name_or_path", None) or getattr(tok, "name", None)
    if nop:
        return str(nop)
    return type(tok).__name__


STEP3_TOKENIZE_CACHE_SCHEMA_VERSION = "odcr_step3_tokenizer_cache/2"
STEP3_TOKENIZE_CACHE_MANIFEST = "cache_manifest.json"
STEP3_TOKENIZE_CACHE_COMPLETED_MARKER = "completed.marker"
STEP3_TOKENIZE_CACHE_FAILED_MARKER = "failed.marker"
STEP3_TOKENIZE_CACHE_STARTUP_FILENAME = "step3_tokenizer_cache_startup.json"
STEP3_TOKENIZE_CACHE_ROLE = "step3_tokenizer_cache"
STEP3_TOKENIZE_CACHE_PRODUCER_CODE_VERSION = "executors.step3_train_core.tokenizer_cache/2"
STEP3_TOKENIZE_CACHE_CONTENT_FINGERPRINT_VERSION = "odcr_step3_tokenizer_cache_content/2"
_STEP3_TOKENIZE_INPUT_COLUMNS = (
    "user_idx",
    "item_idx",
    "rating",
    "explanation",
    "domain",
    "sample_id",
    "content_anchor_score",
    "style_anchor_score",
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "evidence_quality_prior",
)
_STEP3_TOKENIZE_SEQUENCE_FIELDS = (
    "explanation_idx",
    "content_evidence_ids",
    "style_evidence_ids",
    "domain_style_anchor_ids",
    "local_style_hint_ids",
)
_STEP3_TOKENIZE_OUTPUT_FIELDS = (
    "user_idx",
    "item_idx",
    "rating",
    "explanation_idx",
    "domain_idx",
    "sample_id",
    "content_anchor_score",
    "style_anchor_score",
    "content_evidence_ids",
    "style_evidence_ids",
    "domain_style_anchor_ids",
    "local_style_hint_ids",
    "polarity_ids",
    "evidence_quality_prior",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json_env_object(var_name: str, *, context: str) -> dict[str, Any]:
    raw = (os.environ.get(var_name) or "").strip()
    if not raw:
        raise RuntimeError(f"{context} requires resolver-injected {var_name}; refusing ungated Step3 tokenizer cache reuse.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{var_name} must be valid JSON for Step3 tokenizer cache lineage: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{var_name} JSON root must be an object for Step3 tokenizer cache lineage.")
    return payload


def _resolver_thread_env_effective() -> dict[str, Any]:
    raw = (os.environ.get("ODCR_THREAD_ENV_EFFECTIVE_JSON") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _step3_source_table_lineage() -> dict[str, Any]:
    return current_source_table_lineage(required_file=bool((os.environ.get("ODCR_MANIFEST_DIR") or "").strip()))


def _step3_resolved_config_lineage(*, mode: str, task_idx: int) -> dict[str, Any]:
    return current_resolved_config_lineage(
        stage="step3",
        task_id=int(task_idx),
        artifact=STEP3_TOKENIZE_CACHE_ROLE,
        mode=str(mode),
        required_file=bool((os.environ.get("ODCR_MANIFEST_DIR") or "").strip()),
    )


def _tokenizer_lineage(tok) -> dict[str, Any]:
    model_path = require_step5_text_model_dir()
    return {
        "identity": _tokenizer_cache_identity(tok),
        "class": type(tok).__name__,
        "model_path": os.path.abspath(model_path),
        "model_artifact_fingerprint": model_artifact_fingerprint(model_path),
    }


def _step3_preprocess_lineage_sections(upstream_evidence: Mapping[str, Any]) -> dict[str, Any]:
    preprocess = upstream_evidence.get("preprocess")
    if not isinstance(preprocess, Mapping):
        raise RuntimeError("Step3 tokenizer cache requires upstream preprocess gate evidence.")
    run_ids: dict[str, str] = {}
    manifest_fps: dict[str, Any] = {}
    source_table_fps: dict[str, Any] = {}
    metrics_verify_fps: dict[str, Any] = {}
    for unit in ("a", "b", "c"):
        item = preprocess.get(unit)
        if not isinstance(item, Mapping):
            raise RuntimeError(f"Step3 tokenizer cache missing preprocess_{unit} gate evidence.")
        run_id = str(item.get("run_id") or "").strip()
        if not run_id:
            raise RuntimeError(f"Step3 tokenizer cache missing preprocess_{unit} latest run_id.")
        run_ids[unit] = run_id
        manifest_fps[unit] = {
            "run_fingerprint_hash": item.get("fingerprint_hash"),
            "run_summary": item.get("run_summary_fingerprint"),
            "stage_status": item.get("stage_status_fingerprint"),
            "stage_manifest": item.get("stage_manifest_fingerprint"),
        }
        source_table_fps[unit] = item.get("source_table_fingerprint")
        metrics_verify_fps[unit] = {
            "metrics": item.get("metrics_fingerprint"),
            "verify_report": item.get("verify_report_fingerprint"),
        }
    return {
        "preprocess_latest_run_ids": run_ids,
        "preprocess_manifest_fingerprints": manifest_fps,
        "preprocess_source_table_fingerprints": source_table_fps,
        "preprocess_metrics_verify_fingerprints": metrics_verify_fps,
    }


def _build_tokenize_cache_fingerprint(
    *,
    train_path: str,
    valid_path: str | None,
    task_idx: int,
    source_domain: str,
    target_domain: str,
    mode: str,
    split_row_counts: Mapping[str, int],
    upstream_evidence: Mapping[str, Any],
    tok,
    max_length: int,
    evidence_length: int,
    cache_version: str,
) -> dict[str, Any]:
    """Step3 tokenizer cache compatibility payload.

    This intentionally binds cache reuse to content fingerprints and upstream
    lineage.  Path/mtime/dataset_dict-only reuse is not a valid contract.
    """
    preprocess_sections = _step3_preprocess_lineage_sections(upstream_evidence)
    tokenizer_config = {
        "tokenizer": _tokenizer_lineage(tok),
        "processor": {
            "name": "executors.step3_train_core.Processor",
            "max_length": int(max_length),
            "evidence_length": int(evidence_length),
            "padding": "max_length",
            "truncation": True,
            "tokenize_columns": list(_STEP3_TOKENIZE_INPUT_COLUMNS),
            "sequence_fields": list(_STEP3_TOKENIZE_SEQUENCE_FIELDS),
            "output_fields": list(_STEP3_TOKENIZE_OUTPUT_FIELDS),
        },
    }
    merged_fps: dict[str, Any] = {}
    if str(mode) == "eval_valid":
        merged_fps["valid"] = file_fingerprint(train_path)
    else:
        merged_fps["train"] = file_fingerprint(train_path)
    if valid_path:
        merged_fps["valid"] = file_fingerprint(valid_path)
    split_info = {
        "mode": str(mode),
        "splits": {str(k): {"row_count": int(v)} for k, v in sorted(split_row_counts.items())},
    }
    if train_path and str(mode) == "eval_valid":
        split_info["splits"].setdefault("valid", {})["path"] = os.path.abspath(train_path)
    elif train_path:
        split_info["splits"].setdefault("train", {})["path"] = os.path.abspath(train_path)
    if valid_path:
        split_info["splits"].setdefault("valid", {})["path"] = os.path.abspath(valid_path)
    schema_contract = {
        "cache_manifest_schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "step3_upstream_gate_schema_version": upstream_evidence.get("schema_version"),
        "step3_upstream_contract_schema_version": upstream_evidence.get("contract_schema_version"),
        "dataset_format": "huggingface_dataset_dict_save_to_disk",
        "text_normalization_version": "odcr_step3_text_normalization/1",
    }
    resolved_config = _step3_resolved_config_lineage(mode=str(mode), task_idx=int(task_idx))
    source_table = _step3_source_table_lineage()
    effective_payload = current_effective_payload(required=False)
    data_contract_payload = {
        "schema_contract": schema_contract,
        "tokenize_input_columns": list(_STEP3_TOKENIZE_INPUT_COLUMNS),
        "tokenize_sequence_fields": list(_STEP3_TOKENIZE_SEQUENCE_FIELDS),
        "tokenize_output_fields": list(_STEP3_TOKENIZE_OUTPUT_FIELDS),
    }
    preprocessing_artifact_payload = {
        "source_csv_fingerprints": {
            "raw_source_csvs": upstream_evidence.get("source_csv_artifacts"),
            "merged_csvs": merged_fps,
            "upstream_merged_csvs": upstream_evidence.get("merged_artifacts"),
        },
        **preprocess_sections,
        "upstream_gate_hash": upstream_evidence.get("fingerprint_hash"),
    }
    train_runtime_payload = {
        "training_row": {
            key: (effective_payload.get("training_row") or {}).get(key)
            for key in (
                "train_batch_size",
                "per_device_train_batch_size",
                "step3_batch_semantics",
                "step3_batch_formula",
                "train_precision",
                "ddp_world_size",
                "cross_rank_structured_gather_enabled",
                "gather_mode",
                "local_per_gpu_batch",
                "effective_structured_pool",
                "gathered_tensor_names",
                "remote_tensors_detached",
            )
        },
        "step3_ddp": effective_payload.get("step3_ddp"),
        "step3_precision": effective_payload.get("step3_precision"),
        "step3_cross_rank_structured_gather": effective_payload.get("step3_cross_rank_structured_gather"),
    }
    optimizer_payload = {
        "step3_optimizer": effective_payload.get("step3_optimizer"),
        "step3_scheduler": effective_payload.get("step3_scheduler"),
        "learning_rate": (effective_payload.get("training_row") or {}).get("lr"),
        "max_grad_norm": (effective_payload.get("training_row") or {}).get("max_grad_norm"),
    }
    performance_profile_payload = {
        "task_profile_id": effective_payload.get("task_profile_id"),
        "task_profile_key": effective_payload.get("task_profile_key"),
        "profile_isolation_hash": effective_payload.get("profile_isolation_hash"),
        "step3_task_profile": effective_payload.get("step3_task_profile"),
        "step3_backup_profiles": effective_payload.get("step3_backup_profiles"),
        "step3_exploration_profiles": effective_payload.get("step3_exploration_profiles"),
        "step3_worker_profiles": effective_payload.get("step3_worker_profiles"),
        "step3_prefetcher": effective_payload.get("step3_prefetcher"),
        "step3_cross_rank_structured_gather": effective_payload.get("step3_cross_rank_structured_gather"),
        "step3_memory": effective_payload.get("step3_memory"),
        "step3_timing": effective_payload.get("step3_timing"),
        "hardware": effective_payload.get("hardware"),
        "runtime_threads": {
            "num_proc": (effective_payload.get("training_row") or {}).get("num_proc"),
            **_resolver_thread_env_effective(),
        },
    }
    cache_policy = effective_payload.get("step3_cache_policy") if isinstance(effective_payload, Mapping) else {}
    if not isinstance(cache_policy, Mapping):
        cache_policy = {}
    formal_cache_namespace = str(cache_policy.get("formal_cache_namespace") or "cache/step3/tokenizer")
    tokenizer_compat_payload = {
        "manifest_schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "cache_role": STEP3_TOKENIZE_CACHE_ROLE,
        "stage": "step3",
        "cache_version": str(cache_version),
        "task_id": int(task_idx),
        "source_domain": str(source_domain),
        "target_domain": str(target_domain),
        "tokenizer_cache_namespace": {
            "formal_cache_namespace": formal_cache_namespace,
            "task_id": int(task_idx),
            "source_domain": str(source_domain),
            "target_domain": str(target_domain),
            "profile_training_parameters_excluded": True,
        },
        "mode": str(mode),
        "writer_code_version": STEP3_TOKENIZE_CACHE_PRODUCER_CODE_VERSION,
        "step3_tokenizer_config": tokenizer_config,
        "dataset_split_info": split_info,
        "data_contract": data_contract_payload,
        "preprocessing_artifacts": preprocessing_artifact_payload,
    }
    tokenizer_cache_compat_hash = stable_hash(tokenizer_compat_payload)
    record_only_lineage = {
        "resolved_config": resolved_config,
        "source_table": source_table,
        "full_run_config_hash": resolved_config["hash"],
        "source_table_hash": source_table["hash"],
        "train_runtime_config": train_runtime_payload,
        "optimizer_config": optimizer_payload,
        "performance_profile": performance_profile_payload,
        "task_profile_id": effective_payload.get("task_profile_id"),
        "profile_isolation_hash": effective_payload.get("profile_isolation_hash"),
    }
    run_lineage_hash = stable_hash(record_only_lineage)
    payload: dict[str, Any] = {
        "manifest_schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "cache_role": STEP3_TOKENIZE_CACHE_ROLE,
        "stage": "step3",
        "cache_version": str(cache_version),
        "task_id": int(task_idx),
        "source_domain": str(source_domain),
        "target_domain": str(target_domain),
        "mode": str(mode),
        "writer_code_version": STEP3_TOKENIZE_CACHE_PRODUCER_CODE_VERSION,
        "tokenizer_cache_compat_hash": tokenizer_cache_compat_hash,
        "tokenization_compat_hash": tokenizer_cache_compat_hash,
        "run_lineage_hash": run_lineage_hash,
        "formal_cache_namespace": formal_cache_namespace,
        "data_contract_hash": stable_hash(data_contract_payload),
        "preprocessing_artifact_hash": stable_hash(preprocessing_artifact_payload),
        "full_run_config_hash": resolved_config["hash"],
        "train_runtime_config_hash": stable_hash(train_runtime_payload),
        "optimizer_config_hash": stable_hash(optimizer_payload),
        "performance_profile_hash": stable_hash(performance_profile_payload),
        "task_profile_id": effective_payload.get("task_profile_id"),
        "task_profile_key": effective_payload.get("task_profile_key"),
        "profile_isolation_hash": effective_payload.get("profile_isolation_hash"),
        "source_table_hash": source_table["hash"],
        "record_only_lineage": record_only_lineage,
        "tokenizer_cache_compat_payload": tokenizer_compat_payload,
        "formal_cache_namespace": formal_cache_namespace,
        "step3_tokenizer_config": tokenizer_config,
        "dataset_split_info": split_info,
        "source_csv_fingerprints": preprocessing_artifact_payload["source_csv_fingerprints"],
        **preprocess_sections,
        "profile_artifact_fingerprints": upstream_evidence.get("profile_artifact_fingerprints"),
        "domain_artifact_fingerprints": upstream_evidence.get("domain_artifact_fingerprints"),
        "env": {"embed_dim": int(get_odcr_embed_dim())},
        "schema_contract": schema_contract,
        "upstream_gate_hash": upstream_evidence.get("fingerprint_hash"),
    }
    payload["compatibility_key"] = tokenizer_cache_compat_hash
    payload["fingerprint_hash"] = payload["compatibility_key"]
    return payload


def _build_step3_cache_dir(
    task_idx: int,
    train_path: str,
    valid_path: str,
    processor,
    tok,
    *,
    source_domain: str,
    target_domain: str,
    split_row_counts: Mapping[str, int],
    upstream_evidence: Mapping[str, Any],
    cache_version: str = ODCR_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str, dict[str, Any]]:
    fp_payload = _build_tokenize_cache_fingerprint(
        train_path=train_path,
        valid_path=valid_path,
        task_idx=task_idx,
        source_domain=source_domain,
        target_domain=target_domain,
        mode="train_valid",
        split_row_counts=split_row_counts,
        upstream_evidence=upstream_evidence,
        tok=tok,
        max_length=int(processor.max_length),
        evidence_length=int(processor.evidence_length),
        cache_version=cache_version,
    )
    fp = f"{cache_version}_{str(fp_payload['tokenizer_cache_compat_hash'])[:16]}"
    repo_root = Path(os.environ.get("ODCR_ROOT") or Path(__file__).resolve().parents[2]).resolve()
    cache_dir = path_layout.step3_tokenizer_cache_entry_dir(
        repo_root,
        formal_cache_namespace=str(fp_payload.get("formal_cache_namespace") or "cache/step3/tokenizer"),
        task_id=int(task_idx),
        source_domain=str(source_domain),
        target_domain=str(target_domain),
        compatibility_key=fp,
    )
    return str(cache_dir), fp, fp_payload


def _build_step3_eval_cache_dir(
    task_idx: int,
    eval_data_path: str,
    processor,
    tok,
    *,
    source_domain: str,
    target_domain: str,
    split_row_counts: Mapping[str, int],
    upstream_evidence: Mapping[str, Any],
    cache_version: str = ODCR_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str, dict[str, Any]]:
    fp_payload = _build_tokenize_cache_fingerprint(
        train_path=eval_data_path,
        valid_path=None,
        task_idx=task_idx,
        source_domain=source_domain,
        target_domain=target_domain,
        mode="eval_valid",
        split_row_counts=split_row_counts,
        upstream_evidence=upstream_evidence,
        tok=tok,
        max_length=int(processor.max_length),
        evidence_length=int(processor.evidence_length),
        cache_version=cache_version,
    )
    fp = f"{cache_version}_{str(fp_payload['tokenizer_cache_compat_hash'])[:16]}"
    repo_root = Path(os.environ.get("ODCR_ROOT") or Path(__file__).resolve().parents[2]).resolve()
    cache_dir = path_layout.step3_tokenizer_cache_entry_dir(
        repo_root,
        formal_cache_namespace=str(fp_payload.get("formal_cache_namespace") or "cache/step3/tokenizer"),
        task_id=int(task_idx),
        source_domain=str(source_domain),
        target_domain=str(target_domain),
        compatibility_key=fp,
    )
    return str(cache_dir), fp, fp_payload


def _hf_dataset_cache_ready(cache_dir: str) -> bool:
    return os.path.isdir(cache_dir) and os.path.isfile(os.path.join(cache_dir, "dataset_dict.json"))


def _step3_tokenize_cache_manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, STEP3_TOKENIZE_CACHE_MANIFEST)


def _load_step3_tokenize_cache_manifest(cache_dir: str) -> dict[str, Any] | None:
    path = _step3_tokenize_cache_manifest_path(cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fingerprint_step3_hf_cache_content(cache_dir: str) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    if not _hf_dataset_cache_ready(str(root)):
        raise RuntimeError(f"Step3 tokenizer cache content fingerprint requires dataset_dict.json: {root}")
    files: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in {"__pycache__"})
        for name in sorted(filenames):
            if name in {
                STEP3_TOKENIZE_CACHE_MANIFEST,
                STEP3_TOKENIZE_CACHE_COMPLETED_MARKER,
                STEP3_TOKENIZE_CACHE_FAILED_MARKER,
            } or name.endswith(".tmp"):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            files.append({"relative_path": rel, "fingerprint": file_fingerprint(path)})
    dataset_dict_path = root / "dataset_dict.json"
    dataset_dict_fp = file_fingerprint(dataset_dict_path)
    payload = {
        "fingerprint_version": STEP3_TOKENIZE_CACHE_CONTENT_FINGERPRINT_VERSION,
        "cache_dir": str(root),
        "dataset_dict_fingerprint": dataset_dict_fp,
        "files": files,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _step3_tokenize_cache_manifest_gate_fields(
    fingerprint: Mapping[str, Any],
    *,
    cache_dir: str,
) -> dict[str, Any]:
    return {
        "manifest_schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "cache_schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "cache_role": STEP3_TOKENIZE_CACHE_ROLE,
        "stage": "step3",
        "cache_version": str(fingerprint.get("cache_version") or ODCR_TOKENIZE_CACHE_VERSION),
        "task_id": int(fingerprint.get("task_id", -1)),
        "source_domain": str(fingerprint.get("source_domain") or ""),
        "target_domain": str(fingerprint.get("target_domain") or ""),
        "mode": str(fingerprint.get("mode") or ""),
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "writer_code_version": STEP3_TOKENIZE_CACHE_PRODUCER_CODE_VERSION,
        "tokenizer_cache_compat_hash": str(fingerprint.get("tokenizer_cache_compat_hash") or ""),
        "data_contract_hash": str(fingerprint.get("data_contract_hash") or ""),
        "preprocessing_artifact_hash": str(fingerprint.get("preprocessing_artifact_hash") or ""),
        "tokenizer_config_hash": stable_hash(fingerprint.get("step3_tokenizer_config") or {}),
        "dataset_split_hash": stable_hash(fingerprint.get("dataset_split_info") or {}),
        "source_csv_fingerprint_hash": stable_hash(fingerprint.get("source_csv_fingerprints") or {}),
        "preprocess_latest_run_ids_hash": stable_hash(fingerprint.get("preprocess_latest_run_ids") or {}),
        "preprocess_manifest_fingerprints_hash": stable_hash(fingerprint.get("preprocess_manifest_fingerprints") or {}),
        "preprocess_metrics_verify_fingerprints_hash": stable_hash(
            fingerprint.get("preprocess_metrics_verify_fingerprints") or {}
        ),
        "schema_contract_hash": stable_hash(fingerprint.get("schema_contract") or {}),
        "upstream_gate_hash": str(fingerprint.get("upstream_gate_hash") or ""),
        "tokenizer_cache_compat_payload_hash": stable_hash(fingerprint.get("tokenizer_cache_compat_payload") or {}),
        "compatibility_key": str(fingerprint.get("compatibility_key") or ""),
        "fingerprint_hash": str(fingerprint.get("fingerprint_hash") or ""),
    }


def _step3_tokenize_cache_manifest_sections(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    section_keys = (
        "tokenizer_cache_compat_payload",
        "step3_tokenizer_config",
        "dataset_split_info",
        "source_csv_fingerprints",
        "preprocess_latest_run_ids",
        "preprocess_manifest_fingerprints",
        "preprocess_source_table_fingerprints",
        "preprocess_metrics_verify_fingerprints",
        "schema_contract",
    )
    return {key: fingerprint.get(key) for key in section_keys}


def _step3_tokenize_cache_manifest_record_only_sections(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full_run_config_hash": str(fingerprint.get("full_run_config_hash") or ""),
        "source_table_hash": str(fingerprint.get("source_table_hash") or ""),
        "train_runtime_config_hash": str(fingerprint.get("train_runtime_config_hash") or ""),
        "optimizer_config_hash": str(fingerprint.get("optimizer_config_hash") or ""),
        "performance_profile_hash": str(fingerprint.get("performance_profile_hash") or ""),
        "run_lineage_hash": str(fingerprint.get("run_lineage_hash") or ""),
        "record_only_lineage": fingerprint.get("record_only_lineage"),
        "profile_artifact_fingerprints": fingerprint.get("profile_artifact_fingerprints"),
        "domain_artifact_fingerprints": fingerprint.get("domain_artifact_fingerprints"),
        "env": fingerprint.get("env"),
    }


def _step3_tokenize_cache_completed_marker_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, STEP3_TOKENIZE_CACHE_COMPLETED_MARKER)


def _step3_tokenize_cache_failed_marker_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, STEP3_TOKENIZE_CACHE_FAILED_MARKER)


def _fsync_directory(path: str | Path) -> None:
    try:
        fd = os.open(str(Path(path).resolve()), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _step3_tokenize_cache_manifest_decision(
    cache_dir: str,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    cache_dir_abs = str(Path(cache_dir).expanduser().resolve())
    decision: dict[str, Any] = {
        "cache_status": "miss",
        "cache_dir": cache_dir_abs,
        "tokenization_compat_hash": str(
            expected_fingerprint.get("tokenization_compat_hash")
            or expected_fingerprint.get("tokenizer_cache_compat_hash")
            or ""
        ),
        "run_lineage_hash": str(expected_fingerprint.get("run_lineage_hash") or ""),
        "manifest_exists": False,
        "completed": False,
        "hard_gate_match": False,
        "miss_reason": "not_checked",
        "rejected_fields": [],
        "record_only_mismatches": [],
        "would_hit_cache": False,
    }
    if os.path.exists(_step3_tokenize_cache_failed_marker_path(cache_dir)):
        decision["miss_reason"] = "failed_marker_present"
        return decision
    if not _hf_dataset_cache_ready(cache_dir):
        final = Path(cache_dir).expanduser().resolve()
        if final.parent.exists() and any(final.parent.glob(f"{final.name}.partial.*")):
            decision["miss_reason"] = "partial_dir_only"
            return decision
        decision["miss_reason"] = "missing_dataset"
        return decision
    manifest = _load_step3_tokenize_cache_manifest(cache_dir)
    decision["manifest_exists"] = manifest is not None
    if manifest is None:
        decision["miss_reason"] = "missing_manifest"
        return decision
    decision["completed"] = manifest.get("completed") is True
    decision["tokenization_compat_hash"] = str(
        manifest.get("tokenization_compat_hash")
        or manifest.get("tokenizer_cache_compat_hash")
        or decision["tokenization_compat_hash"]
    )
    decision["run_lineage_hash"] = str(manifest.get("run_lineage_hash") or decision["run_lineage_hash"])
    manifest_schema = str(manifest.get("manifest_schema_version") or manifest.get("schema_version") or "")
    if manifest_schema == "odcr_step3_tokenizer_cache/1":
        decision["miss_reason"] = "retired_v1_schema_rebuild_required"
        return decision
    if str(manifest.get("manifest_schema_version")) != STEP3_TOKENIZE_CACHE_SCHEMA_VERSION:
        decision["miss_reason"] = "manifest_schema_mismatch"
        decision["rejected_fields"] = ["manifest_schema_version"]
        return decision
    if str(manifest.get("schema_version")) != STEP3_TOKENIZE_CACHE_SCHEMA_VERSION:
        decision["miss_reason"] = "schema_mismatch"
        decision["rejected_fields"] = ["schema_version"]
        return decision
    if manifest.get("completed") is not True:
        decision["miss_reason"] = "completed_false"
        return decision
    if not os.path.isfile(_step3_tokenize_cache_completed_marker_path(cache_dir)):
        decision["miss_reason"] = "completed_marker_missing"
        return decision
    if str(manifest.get("cache_version")) != str(expected_fingerprint.get("cache_version") or ODCR_TOKENIZE_CACHE_VERSION):
        decision["miss_reason"] = "version_mismatch"
        decision["rejected_fields"] = ["cache_version"]
        return decision
    expected_gate = _step3_tokenize_cache_manifest_gate_fields(
        expected_fingerprint,
        cache_dir=cache_dir,
    )
    rejected_fields: list[str] = []
    for key, expected_value in expected_gate.items():
        if manifest.get(key) != expected_value:
            rejected_fields.append(key)
    for key, expected_value in _step3_tokenize_cache_manifest_sections(expected_fingerprint).items():
        if stable_hash(manifest.get(key)) != stable_hash(expected_value):
            rejected_fields.append(key)
    if rejected_fields:
        decision["miss_reason"] = f"{rejected_fields[0]}_mismatch"
        decision["rejected_fields"] = rejected_fields
        return decision
    record_only_mismatches: list[str] = []
    for key, expected_value in _step3_tokenize_cache_manifest_record_only_sections(expected_fingerprint).items():
        if stable_hash(manifest.get(key)) != stable_hash(expected_value):
            record_only_mismatches.append(key)
    try:
        actual_content = _fingerprint_step3_hf_cache_content(cache_dir)
    except RuntimeError:
        decision["miss_reason"] = "missing_dataset"
        return decision
    if stable_hash(manifest.get("dataset_dict_fingerprint")) != stable_hash(actual_content.get("dataset_dict_fingerprint")):
        decision["miss_reason"] = "dataset_dict_fingerprint_mismatch"
        decision["rejected_fields"] = ["dataset_dict_fingerprint"]
        return decision
    if str(manifest.get("cache_content_hash") or "") != str(actual_content.get("content_hash") or ""):
        decision["miss_reason"] = "cache_content_fingerprint_mismatch"
        decision["rejected_fields"] = ["cache_content_hash"]
        return decision
    if stable_hash(manifest.get("cache_content_fingerprint")) != stable_hash(actual_content):
        decision["miss_reason"] = "cache_content_fingerprint_mismatch"
        decision["rejected_fields"] = ["cache_content_fingerprint"]
        return decision
    decision.update(
        {
            "cache_status": "hit",
            "hard_gate_match": True,
            "miss_reason": "",
            "record_only_mismatches": record_only_mismatches,
            "would_hit_cache": True,
        }
    )
    if record_only_mismatches:
        decision["cache_status"] = "hit_record_only_mismatch"
    return decision


def _write_step3_cache_failed_marker(
    cache_dir: str,
    *,
    phase: str,
    reason: str,
    rank: int | str,
) -> None:
    payload = {
        "schema_version": "odcr_step3_tokenizer_cache_failed/1",
        "status": "failed",
        "phase": str(phase),
        "reason": str(reason),
        "rank": str(rank),
        "pid": os.getpid(),
        "created_at": _utc_now(),
    }
    atomic_write_json(_step3_tokenize_cache_failed_marker_path(cache_dir), payload)


def _step3_tokenize_cache_manifest_matches(
    cache_dir: str,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> tuple[bool, str]:
    decision = _step3_tokenize_cache_manifest_decision(cache_dir, expected_fingerprint=expected_fingerprint)
    if bool(decision.get("would_hit_cache")):
        mismatches = decision.get("record_only_mismatches") or []
        if mismatches:
            return True, "hit_record_only_mismatch"
        return True, "hit"
    return False, str(decision.get("miss_reason") or "miss")


def _write_step3_tokenize_cache_manifest(
    cache_dir: str,
    *,
    fingerprint: Mapping[str, Any],
    completed: bool = True,
    manifest_cache_dir: str | None = None,
) -> None:
    content_fp = _fingerprint_step3_hf_cache_content(cache_dir)
    cache_dir_for_manifest = str(Path(manifest_cache_dir or cache_dir).expanduser().resolve())
    payload = {
        **_step3_tokenize_cache_manifest_gate_fields(fingerprint, cache_dir=cache_dir_for_manifest),
        "tokenization_compat_hash": str(
            fingerprint.get("tokenization_compat_hash")
            or fingerprint.get("tokenizer_cache_compat_hash")
            or ""
        ),
        **_step3_tokenize_cache_manifest_sections(fingerprint),
        **_step3_tokenize_cache_manifest_record_only_sections(fingerprint),
        "fingerprint": dict(fingerprint),
        "completed": bool(completed),
        "dataset_format": "huggingface_dataset_dict_save_to_disk",
        "dataset_dict_fingerprint": content_fp["dataset_dict_fingerprint"],
        "cache_content_fingerprint": content_fp,
        "cache_content_hash": content_fp["content_hash"],
        "created_at": _utc_now(),
    }
    atomic_write_json(_step3_tokenize_cache_manifest_path(cache_dir), payload)
    if completed:
        atomic_write_json(
            _step3_tokenize_cache_completed_marker_path(cache_dir),
            {
                "schema_version": "odcr_step3_tokenizer_cache_completed/1",
                "status": "completed",
                "cache_manifest": STEP3_TOKENIZE_CACHE_MANIFEST,
                "created_at": _utc_now(),
            },
        )
    _fsync_directory(cache_dir)


def _log_tokenize_cache_line(msg: str, log_file: Optional[str]) -> None:
    lg = logging.getLogger(LOGGER_NAME)
    if not logger_has_file_handler(lg):
        print(msg, flush=True)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


def validate_completed_step3_tokenizer_cache(
    cache_dir: str,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    ok, reason = _step3_tokenize_cache_manifest_matches(
        cache_dir,
        expected_fingerprint=expected_fingerprint,
    )
    if not ok:
        raise RuntimeError(f"Step3 tokenizer cache is not reusable: reason={reason} dir={cache_dir}")
    manifest = _load_step3_tokenize_cache_manifest(cache_dir)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Step3 tokenizer cache manifest missing after validation: {cache_dir}")
    return manifest


def wait_for_completed_cache_manifest_file_polling(
    cache_dir: str,
    *,
    expected_fingerprint: Mapping[str, Any],
    timeout_s: float = 7200.0,
    poll_interval_s: float = 2.0,
    log_file: Optional[str] = None,
) -> dict[str, Any]:
    """Wait for cache readiness with plain filesystem polling only.

    This function is intentionally free of torch.distributed calls. It is used
    only before NCCL/DDP startup or in the eval cache cold path.
    """
    start = time.monotonic()
    last_reason = "not_checked"
    while True:
        ok, reason = _step3_tokenize_cache_manifest_matches(
            cache_dir,
            expected_fingerprint=expected_fingerprint,
        )
        if ok:
            manifest = _load_step3_tokenize_cache_manifest(cache_dir)
            if isinstance(manifest, dict):
                return manifest
        last_reason = reason
        if reason == "failed_marker_present":
            raise RuntimeError(f"Step3 tokenizer cache build failed in another process: {cache_dir}")
        if time.monotonic() - start > float(timeout_s):
            raise TimeoutError(
                f"Timed out waiting for completed Step3 tokenizer cache manifest: "
                f"dir={cache_dir} last_reason={last_reason}"
            )
        if log_file and int(time.monotonic() - start) % 60 == 0:
            _log_tokenize_cache_line(
                f"[Tokenize] waiting for completed cache manifest | cache_dir={cache_dir} | last_reason={last_reason}",
                log_file,
            )
        time.sleep(max(0.1, float(poll_interval_s)))


def _make_step3_cache_partial_dir(cache_dir: str) -> str:
    final = Path(cache_dir).expanduser().resolve()
    partial = final.parent / f"{final.name}.partial.{os.getpid()}.{uuid.uuid4().hex[:12]}"
    partial.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        partial / "build_started.json",
        {
            "schema_version": "odcr_step3_tokenizer_cache_build/1",
            "status": "building",
            "final_cache_dir": str(final),
            "pid": os.getpid(),
            "created_at": _utc_now(),
        },
    )
    _fsync_directory(partial)
    return str(partial)


def build_or_reuse_step3_tokenizer_cache_atomic(
    *,
    datasets: DatasetDict | None,
    processor,
    nproc: int,
    cache_dir: str,
    cache_fingerprint: str,
    cache_fingerprint_payload: Mapping[str, Any],
    build_allowed: bool,
    rank: int | str,
    show_datasets_progress: bool,
    log_tokenize: bool,
    phase: str,
    log_file: Optional[str],
    timing_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cache_dir = str(Path(cache_dir).expanduser().resolve())
    cache_decision = _step3_tokenize_cache_manifest_decision(
        cache_dir,
        expected_fingerprint=cache_fingerprint_payload,
    )
    cache_valid = bool(cache_decision.get("would_hit_cache"))
    cache_reason = "hit_record_only_mismatch" if cache_decision.get("record_only_mismatches") else (
        "hit" if cache_valid else str(cache_decision.get("miss_reason") or "miss")
    )
    if timing_sink is not None:
        timing_sink["cache_decision"] = dict(cache_decision)
        timing_sink["cache_status"] = str(cache_decision.get("cache_status") or ("hit" if cache_valid else "miss"))
        timing_sink["cache_gate_reason"] = str(cache_reason)
        timing_sink["record_only_mismatches"] = list(cache_decision.get("record_only_mismatches") or [])
        timing_sink["rejected_fields"] = list(cache_decision.get("rejected_fields") or [])
    if cache_valid:
        if log_tokenize:
            _log_tokenize_cache_line(
                f"[Tokenize] {phase} cache hit | fingerprint={cache_fingerprint} | cache_dir={cache_dir}",
                log_file,
            )
        return validate_completed_step3_tokenizer_cache(
            cache_dir,
            expected_fingerprint=cache_fingerprint_payload,
        )

    if not build_allowed:
        wait0 = time.perf_counter()
        manifest = wait_for_completed_cache_manifest_file_polling(
            cache_dir,
            expected_fingerprint=cache_fingerprint_payload,
            log_file=log_file,
        )
        if timing_sink is not None:
            timing_sink["cache_status"] = "waited_for_completed_manifest"
            timing_sink["non_rank_wait_time_s"] = round(time.perf_counter() - wait0, 6)
            timing_sink["cache_gate_reason"] = "completed_after_file_poll"
        return manifest

    if datasets is None:
        raise RuntimeError("Step3 tokenizer cache build requires in-memory datasets for the lock owner.")

    final = Path(cache_dir).expanduser().resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = _make_step3_cache_partial_dir(cache_dir)
    if log_tokenize:
        _log_tokenize_cache_line(
            f"[Tokenize] {phase} cache build atomic start | fingerprint={cache_fingerprint} | "
            f"reason={cache_reason} | rejected_fields={cache_decision.get('rejected_fields') or []} | "
            f"record_only_mismatches={cache_decision.get('record_only_mismatches') or []} | "
            f"cache_dir={cache_dir} | partial_dir={partial_dir}",
            log_file,
        )
    t0 = time.perf_counter()
    try:
        with hf_datasets_progress_bar(show_datasets_progress):
            encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
        encoded_data.save_to_disk(partial_dir)
        _write_step3_tokenize_cache_manifest(
            partial_dir,
            fingerprint=cache_fingerprint_payload,
            completed=False,
            manifest_cache_dir=cache_dir,
        )
        if not _hf_dataset_cache_ready(partial_dir):
            raise RuntimeError(f"partial Step3 tokenizer cache is missing dataset_dict.json: {partial_dir}")
        _write_step3_tokenize_cache_manifest(
            partial_dir,
            fingerprint=cache_fingerprint_payload,
            completed=True,
            manifest_cache_dir=cache_dir,
        )
        if final.exists():
            old_valid, old_reason = _step3_tokenize_cache_manifest_matches(
                str(final),
                expected_fingerprint=cache_fingerprint_payload,
            )
            if old_valid:
                shutil.rmtree(partial_dir, ignore_errors=True)
                return validate_completed_step3_tokenizer_cache(
                    str(final),
                    expected_fingerprint=cache_fingerprint_payload,
                )
            retired = final.parent / f"{final.name}.retired.{os.getpid()}.{uuid.uuid4().hex[:12]}"
            os.replace(str(final), str(retired))
            shutil.rmtree(retired, ignore_errors=True)
            if log_tokenize:
                _log_tokenize_cache_line(
                    f"[Tokenize] {phase} retired invalid cache before atomic publish | "
                    f"reason={old_reason} | cache_dir={cache_dir}",
                    log_file,
                )
        os.replace(partial_dir, cache_dir)
        _fsync_directory(final.parent)
        _write_step3_tokenize_cache_manifest(
            cache_dir,
            fingerprint=cache_fingerprint_payload,
            completed=True,
            manifest_cache_dir=cache_dir,
        )
        manifest = validate_completed_step3_tokenizer_cache(
            cache_dir,
            expected_fingerprint=cache_fingerprint_payload,
        )
    except Exception as exc:
        try:
            _write_step3_cache_failed_marker(
                partial_dir,
                phase=phase,
                reason=str(exc),
                rank=rank,
            )
        except Exception:
            pass
        if timing_sink is not None:
            timing_sink["cache_status"] = "failed"
            timing_sink["cache_failure_reason"] = str(exc)
        raise
    elapsed = time.perf_counter() - t0
    if timing_sink is not None:
        timing_sink["cache_status"] = "miss_or_rebuild_completed"
        timing_sink["rank0_cache_build_time_s"] = round(elapsed, 6)
        timing_sink["cache_gate_reason"] = "completed_after_atomic_publish"
    if log_tokenize:
        _log_tokenize_done(phase, nproc, elapsed, log_file)
        _log_tokenize_cache_line(
            f"[Tokenize] {phase} cache completed | fingerprint={cache_fingerprint} | "
            f"cache_dir={cache_dir} | build_wall_time={elapsed:.2f}s",
            log_file,
        )
    return manifest


def load_completed_step3_tokenizer_cache_for_rank(
    cache_dir: str,
    *,
    expected_fingerprint: Mapping[str, Any],
    rank: int | str,
    timing_sink: dict[str, Any] | None = None,
) -> DatasetDict:
    _ = rank
    validate_completed_step3_tokenizer_cache(
        cache_dir,
        expected_fingerprint=expected_fingerprint,
    )
    load0 = time.perf_counter()
    encoded_data = load_from_disk(cache_dir)
    if timing_sink is not None:
        timing_sink["cache_load_time_s"] = round(time.perf_counter() - load0, 6)
    return encoded_data


def _map_tokenize_train_valid_to_hf_cache(
    *,
    datasets: DatasetDict | None,
    processor,
    nproc: int,
    cache_dir: str,
    cache_fingerprint: str,
    cache_fingerprint_payload: Mapping[str, Any],
    rank: int,
    show_datasets_progress: bool,
    log_tokenize: bool,
    phase: str,
    log_file: Optional[str],
    timing_sink: dict[str, Any] | None = None,
) -> DatasetDict:
    """Build/reuse a completed tokenizer cache without any distributed collective."""
    build_or_reuse_step3_tokenizer_cache_atomic(
        datasets=datasets,
        processor=processor,
        nproc=nproc,
        cache_dir=cache_dir,
        cache_fingerprint=cache_fingerprint,
        cache_fingerprint_payload=cache_fingerprint_payload,
        build_allowed=(int(rank) == 0),
        rank=rank,
        show_datasets_progress=show_datasets_progress,
        log_tokenize=bool(log_tokenize and int(rank) == 0),
        phase=phase,
        log_file=log_file,
        timing_sink=timing_sink,
    )
    return load_completed_step3_tokenizer_cache_for_rank(
        cache_dir,
        expected_fingerprint=cache_fingerprint_payload,
        rank=rank,
        timing_sink=timing_sink,
    )


def _step3_tokenizer_cache_startup_path() -> Path | None:
    raw = (os.environ.get("ODCR_STEP3_TOKENIZER_CACHE_STARTUP_JSON") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    meta = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if meta:
        return Path(meta).expanduser().resolve() / STEP3_TOKENIZE_CACHE_STARTUP_FILENAME
    return None


def _write_step3_tokenizer_cache_startup_payload(payload: Mapping[str, Any]) -> Path | None:
    path = _step3_tokenizer_cache_startup_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(payload))
    return path


def _read_step3_tokenizer_cache_startup_payload() -> dict[str, Any] | None:
    # Internal-only transport; _load_step3_artefacts validates status and cache manifest fail-fast before DDP startup.
    path = _step3_tokenizer_cache_startup_path()
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_manifest_summary(cache_dir: str, startup_timing: Mapping[str, Any]) -> dict[str, Any]:
    cache_manifest = _load_step3_tokenize_cache_manifest(cache_dir)
    if not isinstance(cache_manifest, dict):
        raise RuntimeError(f"Step3 tokenizer cache manifest missing after gate validation: {cache_dir}")
    cache_manifest_path = _step3_tokenize_cache_manifest_path(cache_dir)
    return {
        "schema_version": STEP3_TOKENIZE_CACHE_SCHEMA_VERSION,
        "manifest_path": os.path.abspath(cache_manifest_path),
        "manifest_fingerprint": file_fingerprint(cache_manifest_path, sample_only=True),
        "cache_dir": os.path.abspath(cache_dir),
        "cache_version": cache_manifest.get("cache_version"),
        "completed": bool(cache_manifest.get("completed")),
        "tokenizer_cache_compat_hash": cache_manifest.get("tokenizer_cache_compat_hash"),
        "tokenization_compat_hash": cache_manifest.get("tokenization_compat_hash") or cache_manifest.get("tokenizer_cache_compat_hash"),
        "run_lineage_hash": cache_manifest.get("run_lineage_hash"),
        "data_contract_hash": cache_manifest.get("data_contract_hash"),
        "preprocessing_artifact_hash": cache_manifest.get("preprocessing_artifact_hash"),
        "full_run_config_hash": cache_manifest.get("full_run_config_hash"),
        "source_table_hash": cache_manifest.get("source_table_hash"),
        "train_runtime_config_hash": cache_manifest.get("train_runtime_config_hash"),
        "optimizer_config_hash": cache_manifest.get("optimizer_config_hash"),
        "performance_profile_hash": cache_manifest.get("performance_profile_hash"),
        "compatibility_key": cache_manifest.get("compatibility_key"),
        "fingerprint_hash": cache_manifest.get("fingerprint_hash"),
        "cache_content_hash": (cache_manifest.get("cache_content_fingerprint") or {}).get("content_hash"),
        "upstream_gate_hash": cache_manifest.get("upstream_gate_hash"),
        "startup_timing": dict(startup_timing),
        "cache_decision": dict(startup_timing.get("cache_decision") or {}),
        "pre_ddp_cache_ready": True,
        "rank0_only_cache_build": False,
        "rank0_only_csv_tokenizer_build": False,
    }


def _write_step3_initial_training_runtime_config(
    args,
    resolved: FinalTrainingConfig,
    *,
    rank: int | str,
    cache_status: str,
) -> None:
    meta_dir = (os.environ.get("ODCR_MANIFEST_DIR") or "").strip()
    if not meta_dir:
        return
    out_path = Path(meta_dir).expanduser().resolve() / "training_runtime_config.json"
    if out_path.is_file() and str(rank) != "parent":
        return
    thread_env = _resolver_thread_env_effective()
    reserved_cpu = 2
    payload = {
        "phase": "pre_ddp_tokenizer_cache",
        "status": "initial",
        "rank": str(rank),
        "task_id": int(getattr(resolved, "task_idx", 0) or 0),
        "source_domain": str(getattr(args, "auxiliary", "") or ""),
        "target_domain": str(getattr(args, "target", "") or ""),
        "training_loop_started": False,
        "checkpoint_created": False,
        "cache_status": str(cache_status),
        "num_proc": int(getattr(resolved, "num_proc", 1) or 1),
        "max_parallel_cpu": int(getattr(resolved, "max_parallel_cpu", 0) or 0),
        "reserved_cpu": reserved_cpu,
        "tokenization_formula": (
            f"num_proc({int(getattr(resolved, 'num_proc', 1) or 1)}) + reserved_cpu({reserved_cpu}) "
            f"<= max_parallel_cpu({int(getattr(resolved, 'max_parallel_cpu', 0) or 0)})"
        ),
        "worker_formula": (
            f"dataloader_num_workers_train({int(getattr(resolved, 'dataloader_num_workers_train', 0) or 0)}) "
            f"* ddp_world_size({int(getattr(resolved, 'ddp_world_size', 1) or 1)}) + reserved_cpu({reserved_cpu}) "
            f"<= max_parallel_cpu({int(getattr(resolved, 'max_parallel_cpu', 0) or 0)})"
        ),
        "omp_num_threads": int(thread_env.get("OMP_NUM_THREADS") or 1),
        "mkl_num_threads": int(thread_env.get("MKL_NUM_THREADS") or 1),
        "tokenizers_parallelism": thread_env.get("TOKENIZERS_PARALLELISM", ""),
        "runtime_env": runtime_env_dict_for_config_resolved(),
        "effective_training_payload_json_present": bool((os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()),
        "training_semantic_fingerprint": (os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or None,
        "generation_semantic_fingerprint": (os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or None,
        "runtime_diagnostics_fingerprint": (os.environ.get("ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip() or None,
    }
    write_training_runtime_config_artifact(meta_dir, payload)


def ensure_step3_tokenizer_cache_ready_pre_ddp(
    args,
    *,
    resolved: FinalTrainingConfig | None = None,
    rank: int | str = "parent",
    world_size: int = 1,
    build_allowed: bool = True,
    log_tokenize: bool = True,
    show_datasets_progress: bool = True,
) -> dict[str, Any]:
    """Build or validate the Step3 tokenizer cache before NCCL/DDP exists."""
    task_idx = resolve_task_idx_from_aux_target(args.auxiliary, args.target)
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")
    _ro = collect_training_hardware_overrides_from_args(args)
    resolved_cfg = resolved or build_resolved_training_config(
        args,
        task_idx=int(task_idx),
        world_size=int(world_size),
        hardware_overrides=_ro,
    )
    _write_step3_initial_training_runtime_config(
        args,
        resolved_cfg,
        rank=rank,
        cache_status="starting",
    )
    repo_root = os.path.abspath(os.environ.get("ODCR_ROOT") or os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    upstream_evidence = validate_step3_preprocess_upstream_gate(
        repo_root=repo_root,
        task_id=int(task_idx),
        auxiliary_domain=args.auxiliary,
        target_domain=args.target,
        data_dir=get_data_dir(),
        merged_dir=get_merged_data_dir(),
        runs_dir=os.environ.get("ODCR_RESOLVED_RUNS_DIR") or os.path.join(repo_root, "runs"),
        embed_dim=int(get_odcr_embed_dim()),
    )
    path = os.path.join(get_merged_data_dir(), str(task_idx))
    train_path = os.path.join(path, "aug_train.csv")
    valid_path = os.path.join(path, "aug_valid.csv")
    save_file = args.save_file or os.path.join(get_stage_run_dir(task_idx), "model", "best.pth")
    startup_timing: dict[str, Any] = {
        "rank": rank,
        "world_size": int(world_size),
        "rank0_csv_load_time_s": None,
        "cache_status": "not_run",
        "rank0_cache_build_time_s": None,
        "non_rank_wait_time_s": None,
        "cache_load_time_s": None,
        "pre_ddp": True,
    }
    csv0 = time.perf_counter()
    train_df = pd.read_csv(train_path)
    _require_step3_canonical_columns(train_df, csv_path=train_path, split="train")
    nuser = int(train_df["user_idx"].max()) + 1
    nitem = int(train_df["item_idx"].max()) + 1
    valid_df = pd.read_csv(valid_path)
    _require_step3_canonical_columns(valid_df, csv_path=valid_path, split="valid")
    train_df["item"] = train_df["item"].astype(str)
    valid_df["item"] = valid_df["item"].astype(str)
    train_df = train_df[train_df["explanation"].notna()].reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    train_df["sample_id"] = np.arange(len(train_df), dtype=np.int64)
    valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "valid": Dataset.from_pandas(valid_df),
    })
    tok_runtime = get_odcr_text_tokenizer()
    processor = Processor(
        args.auxiliary,
        args.target,
        max_length=int(resolved_cfg.tokenizer_max_length),
        evidence_length=int(resolved_cfg.evidence_max_length),
    )
    split_row_counts = {name: len(datasets[name]) for name in datasets.keys()}
    cache_dir, cache_fp, cache_fp_payload = _build_step3_cache_dir(
        int(task_idx),
        train_path,
        valid_path,
        processor,
        tok_runtime,
        source_domain=args.auxiliary,
        target_domain=args.target,
        split_row_counts=split_row_counts,
        upstream_evidence=upstream_evidence,
    )
    startup_timing["rank0_csv_load_time_s"] = round(time.perf_counter() - csv0, 6)
    if log_tokenize:
        _log_tokenize_cache_line(
            f"[Tokenize] step3 pre-DDP cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
            getattr(args, "log_file", None),
        )
    _tok_lg = logging.getLogger(LOGGER_NAME)
    try:
        with odcr_timing_phase(
            _tok_lg,
            "pre_ddp_tokenize_pipeline_step3_train_valid",
            route=ROUTE_SUMMARY,
            rank=0 if str(rank) == "parent" else int(rank),
        ):
            manifest = build_or_reuse_step3_tokenizer_cache_atomic(
                datasets=datasets if build_allowed else None,
                processor=processor,
                nproc=int(resolved_cfg.num_proc),
                cache_dir=cache_dir,
                cache_fingerprint=cache_fp,
                cache_fingerprint_payload=cache_fp_payload,
                build_allowed=bool(build_allowed),
                rank=rank,
                show_datasets_progress=show_datasets_progress,
                log_tokenize=log_tokenize,
                phase="train+valid pre-DDP",
                log_file=getattr(args, "log_file", None),
                timing_sink=startup_timing,
            )
    except Exception:
        startup_timing["cache_status"] = "failed"
        payload = {
            "schema_version": "odcr_step3_tokenizer_cache_startup/1",
            "status": "failed",
            "failure_phase": "tokenization_cache",
            "task_id": int(task_idx),
            "source_domain": str(args.auxiliary),
            "target_domain": str(args.target),
            "cache_dir": str(locals().get("cache_dir", "")),
            "cache_key": str(locals().get("cache_fp", "")),
            "startup_timing": startup_timing,
            "training_loop_started": False,
            "checkpoint_created": False,
            "created_at": _utc_now(),
        }
        _write_step3_tokenizer_cache_startup_payload(payload)
        _write_step3_initial_training_runtime_config(
            args,
            resolved_cfg,
            rank=rank,
            cache_status="failed",
        )
        raise
    cache_manifest_summary = _cache_manifest_summary(cache_dir, startup_timing)
    payload = {
        "schema_version": "odcr_step3_tokenizer_cache_startup/1",
        "status": "completed",
        "stage": "step3",
        "phase": "pre_ddp_tokenizer_cache",
        "task_id": int(task_idx),
        "source_domain": str(args.auxiliary),
        "target_domain": str(args.target),
        "nuser": int(nuser),
        "nitem": int(nitem),
        "save_file": str(save_file),
        "cache_dir": str(cache_dir),
        "cache_key": str(cache_fp),
        "cache_fp": str(cache_fp),
        "cache_fp_payload": dict(cache_fp_payload),
        "split_row_counts": split_row_counts,
        "tokenizer_len": int(len(tok_runtime)),
        "tokenizer_eos_token_id": getattr(tok_runtime, "eos_token_id", None),
        "upstream_evidence": upstream_evidence,
        "cache_manifest": manifest,
        "cache_manifest_summary": cache_manifest_summary,
        "startup_timing": startup_timing,
        "training_loop_started": False,
        "checkpoint_created": False,
        "created_at": _utc_now(),
    }
    _write_step3_tokenizer_cache_startup_payload(payload)
    _write_step3_initial_training_runtime_config(
        args,
        resolved_cfg,
        rank=rank,
        cache_status="completed",
    )
    return payload


def _load_step3_artefacts(
    args,
    device: int,
    resolved: FinalTrainingConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    log_tokenize: bool = True,
    show_datasets_progress: bool = True,
):
    task_idx = int(resolved.task_idx)
    _resolved_lengths = (resolved.tokenizer_max_length, resolved.evidence_max_length)
    _ = _resolved_lengths
    os.makedirs(get_stage_run_dir(task_idx), exist_ok=True)
    startup_timing: dict[str, Any] = {
        "rank": int(rank),
        "rank0_csv_load_time_s": None,
        "cache_status": "not_run",
        "rank0_cache_build_time_s": None,
        "non_rank_wait_time_s": None,
        "cache_load_time_s": None,
        "pre_ddp": True,
    }
    startup_payload = _read_step3_tokenizer_cache_startup_payload()
    if not isinstance(startup_payload, dict) or startup_payload.get("status") != "completed":
        startup_payload = ensure_step3_tokenizer_cache_ready_pre_ddp(
            args,
            resolved=resolved,
            rank=rank,
            world_size=world_size,
            build_allowed=(int(rank) == 0),
            log_tokenize=(rank == 0 and log_tokenize),
            show_datasets_progress=(rank == 0 and show_datasets_progress),
        )
    if startup_payload.get("status") != "completed":
        raise RuntimeError("Step3 tokenizer cache startup payload is not completed; refusing DDP startup.")
    nuser = int(startup_payload["nuser"])
    nitem = int(startup_payload["nitem"])
    save_file = str(startup_payload["save_file"])
    cache_dir = str(startup_payload["cache_dir"])
    cache_fp = str(startup_payload["cache_fp"])
    cache_fp_payload = dict(startup_payload["cache_fp_payload"])
    startup_timing.update(dict(startup_payload.get("startup_timing") or {}))
    tok_runtime = _TokenizerRuntimeView(
        vocab_size=int(startup_payload["tokenizer_len"]),
        eos_token_id=startup_payload.get("tokenizer_eos_token_id"),
    )
    if rank == 0 and log_tokenize:
        _log_tokenize_cache_line(
            f"[Tokenize] step3 cache ready before DDP | fingerprint={cache_fp} | cache_dir={cache_dir}",
            getattr(args, "log_file", None),
        )
    encoded_data = load_completed_step3_tokenizer_cache_for_rank(
        cache_dir,
        expected_fingerprint=cache_fp_payload,
        rank=rank,
        timing_sink=startup_timing,
    )
    cache_manifest_summary = dict(startup_payload.get("cache_manifest_summary") or _cache_manifest_summary(cache_dir, startup_timing))
    upstream_evidence = dict(startup_payload.get("upstream_evidence") or {})
    if not upstream_evidence:
        raise RuntimeError("Step3 tokenizer cache startup payload is missing upstream evidence.")
    encoded_data.set_format("torch")
    train_dataset = TensorDataset(
        encoded_data["train"]["user_idx"],
        encoded_data["train"]["item_idx"],
        encoded_data["train"]["rating"],
        encoded_data["train"]["explanation_idx"],
        encoded_data["train"]["domain_idx"],
        encoded_data["train"]["sample_id"],
        encoded_data["train"]["content_anchor_score"],
        encoded_data["train"]["style_anchor_score"],
        encoded_data["train"]["content_evidence_ids"],
        encoded_data["train"]["style_evidence_ids"],
        encoded_data["train"]["domain_style_anchor_ids"],
        encoded_data["train"]["local_style_hint_ids"],
        encoded_data["train"]["polarity_ids"],
        encoded_data["train"]["evidence_quality_prior"],
    )
    valid_dataset = TensorDataset(
        encoded_data["valid"]["user_idx"],
        encoded_data["valid"]["item_idx"],
        encoded_data["valid"]["rating"],
        encoded_data["valid"]["explanation_idx"],
        encoded_data["valid"]["domain_idx"],
        encoded_data["valid"]["sample_id"],
        encoded_data["valid"]["content_anchor_score"],
        encoded_data["valid"]["style_anchor_score"],
        encoded_data["valid"]["content_evidence_ids"],
        encoded_data["valid"]["style_evidence_ids"],
        encoded_data["valid"]["domain_style_anchor_ids"],
        encoded_data["valid"]["local_style_hint_ids"],
        encoded_data["valid"]["polarity_ids"],
        encoded_data["valid"]["evidence_quality_prior"],
    )

    dc, ds, uc, us, ic, ist, profile_meta = load_profile_tensors_dual_first(
        data_root=get_data_dir(),
        auxiliary_domain=args.auxiliary,
        target_domain=args.target,
        device_idx=device,
        step3_upstream_preflight_summary=upstream_evidence,
    )
    if rank == 0:
        print(
            f"[Step3] ODCR dual-channel profiles loaded ({profile_meta.get('profile_mode')}).",
            flush=True,
        )

    _em_prof = int(uc.shape[-1])
    if _em_prof != int(get_odcr_embed_dim()):
        raise ValueError(
            f"Step3 加载的 profile 隐层维度={_em_prof} 与 ODCR_EMBED_DIM={get_odcr_embed_dim()} 不一致；"
            "请重算 embeddings 或调整 ODCR_EMBED_DIM。"
        )
    resolved_profiles = replace(
        resolved,
        emsize=_em_prof,
        step3_upstream_evidence_json=json.dumps(upstream_evidence, ensure_ascii=False, sort_keys=True),
        step3_tokenizer_cache_manifest_json=json.dumps(cache_manifest_summary, ensure_ascii=False, sort_keys=True),
    )

    model = Model(
        nuser,
        nitem,
        int(len(tok_runtime)),
        resolved_profiles.emsize,
        resolved_profiles.nhead,
        resolved_profiles.nhid,
        args.nlayers,
        resolved_profiles.dropout,
        uc,
        us,
        ic,
        ist,
        dc,
        ds,
    ).to(device)
    model.apply_runtime_config(resolved_profiles, tok_runtime)
    memory_summary = apply_step3_memory_controls(model, resolved_profiles)
    if rank == 0:
        print(
            "[Step3] memory controls: "
            + json.dumps(memory_summary, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    return train_dataset, valid_dataset, model, nuser, nitem, save_file, resolved_profiles


def _distributed_valid_sample_indices(n_valid: int, world_size: int, rank: int) -> list[int]:
    """
    验证集按样本无重叠划分到各 rank（不补重复样本）。
    与 DistributedSampler(drop_last=False) 不同：后者为对齐 total_size 会复制索引，导致
    valid loss 的 all_reduce 加权平均偏离「全验证集各样本恰好一次」的真实均值。
    分片为连续区间，与 bleu4_explanation_full_valid_ddp 的按 rank 切块一致（仅按余数平衡各卡条数）。
    """
    if n_valid <= 0:
        return []
    if world_size <= 1:
        return list(range(n_valid))
    base = n_valid // world_size
    rem = n_valid % world_size
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 0)
    return list(range(start, start + size))


class _TokenizerRuntimeView:
    def __init__(self, *, vocab_size: int, eos_token_id: int | None) -> None:
        self._vocab_size = int(vocab_size)
        self.eos_token_id = eos_token_id

    def __len__(self) -> int:
        return self._vocab_size


def step3_profile_buffer_names() -> tuple[str, ...]:
    return (
        "domain_content_profiles",
        "domain_style_profiles",
        "user_content_profiles",
        "user_style_profiles",
        "item_content_profiles",
        "item_style_profiles",
    )


def _parse_step3_memory_config(final_cfg: FinalTrainingConfig) -> dict[str, Any]:
    raw = str(getattr(final_cfg, "memory_config_json", "") or "{}")
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Step3 memory config is invalid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise RuntimeError("Step3 memory config must be a JSON object.")
    ckpt = cfg.get("activation_checkpointing") if isinstance(cfg.get("activation_checkpointing"), Mapping) else {}
    policy = str(cfg.get("profile_buffer_policy") or "gpu_resident")
    if policy not in ("gpu_resident", "cpu_pinned_batch_gather"):
        raise RuntimeError("Step3 profile_buffer_policy must be gpu_resident or cpu_pinned_batch_gather.")
    return {
        "activation_checkpointing": {
            "enabled": bool(ckpt.get("enabled", False)),
            "policy": str(ckpt.get("policy") or "selective"),
            "modules": [str(x) for x in (ckpt.get("modules") or [])],
        },
        "profile_buffer_policy": policy,
    }


def apply_step3_memory_controls(model: nn.Module, final_cfg: FinalTrainingConfig) -> dict[str, Any]:
    underlying = get_underlying_model(model)
    cfg = _parse_step3_memory_config(final_cfg)
    policy = str(cfg["profile_buffer_policy"])
    moved_to_cpu: list[str] = []
    if policy == "cpu_pinned_batch_gather":
        for name in ("user_content_profiles", "user_style_profiles", "item_content_profiles", "item_style_profiles"):
            tensor = getattr(underlying, name, None)
            if torch.is_tensor(tensor):
                cpu_tensor = tensor.detach().cpu()
                if torch.cuda.is_available():
                    try:
                        cpu_tensor = cpu_tensor.pin_memory()
                    except RuntimeError:
                        pass
                setattr(underlying, name, cpu_tensor)
                moved_to_cpu.append(name)
    setattr(underlying, "profile_buffer_policy", policy)
    setattr(underlying, "activation_checkpointing_policy", dict(cfg["activation_checkpointing"]))
    return {
        "activation_checkpointing": dict(cfg["activation_checkpointing"]),
        "profile_buffer_policy": policy,
        "cpu_pinned_profile_buffers": moved_to_cpu,
        "silent_fallback": False,
    }


def step3_trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def step3_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for _, param in step3_trainable_named_parameters(model)]


def _step3_optimizer_config(final_cfg: FinalTrainingConfig) -> dict[str, Any]:
    raw = str(getattr(final_cfg, "optimizer_config_json", "") or "").strip()
    if not raw:
        raise RuntimeError("Step3 optimizer_config_json is required.")
    cfg = json.loads(raw)
    if not isinstance(cfg, dict):
        raise RuntimeError("Step3 optimizer_config_json must decode to an object.")
    if str(cfg.get("name") or "").strip().lower() != "adamw":
        raise RuntimeError("Step3 active optimizer must be AdamW.")
    return cfg


def _step3_optimizer_group_for_name(name: str) -> str:
    low = name.lower()
    if low.endswith(".bias") or ".bias" in low or "norm" in low or "layernorm" in low:
        return "no_decay"
    special_markers = (
        "proto",
        "prototype",
        "disentangler",
        "anchor_gate",
        "domain_style",
        "local_style",
        "polarity",
    )
    if any(marker in low for marker in special_markers):
        return "special"
    return "dense"


def build_step3_optimizer_param_groups(
    model: nn.Module,
    final_cfg: FinalTrainingConfig,
) -> list[dict[str, Any]]:
    cfg = _step3_optimizer_config(final_cfg)
    group_cfg = cfg.get("param_groups")
    if not isinstance(group_cfg, Mapping):
        raise RuntimeError("Step3 optimizer param_groups must be present in resolved config.")
    decay_by_group = {
        "dense": float(group_cfg["dense_weight_decay"]),
        "special": float(group_cfg["special_weight_decay"]),
        "no_decay": float(group_cfg["no_decay"]),
    }
    named = step3_trainable_named_parameters(get_underlying_model(model))
    grouped: dict[str, list[nn.Parameter]] = {"dense": [], "special": [], "no_decay": []}
    names: dict[str, list[str]] = {"dense": [], "special": [], "no_decay": []}
    for name, param in named:
        if not param.requires_grad:
            continue
        group = _step3_optimizer_group_for_name(name)
        grouped[group].append(param)
        names[group].append(name)
    out: list[dict[str, Any]] = []
    for group_name in ("dense", "special", "no_decay"):
        params = grouped[group_name]
        if not params:
            continue
        out.append(
            {
                "params": params,
                "weight_decay": decay_by_group[group_name],
                "odcr_group": group_name,
                "odcr_param_names": names[group_name],
            }
        )
    if not out:
        raise RuntimeError("Step3 optimizer received no trainable parameters.")
    return out


def build_step3_optimizer(model: nn.Module, final_cfg: FinalTrainingConfig) -> optim.Optimizer:
    cfg = _step3_optimizer_config(final_cfg)
    betas_raw = cfg.get("betas")
    if not isinstance(betas_raw, Sequence) or len(betas_raw) != 2:
        raise RuntimeError("Step3 optimizer betas must be a two-item sequence.")
    param_groups = build_step3_optimizer_param_groups(model, final_cfg)
    return optim.AdamW(
        param_groups,
        lr=float(final_cfg.scheduler_initial_lr),
        betas=(float(betas_raw[0]), float(betas_raw[1])),
        eps=float(cfg.get("eps", 1e-8)),
    )


def apply_step3_precision_backend(final_cfg: FinalTrainingConfig) -> None:
    if str(final_cfg.train_precision).strip().lower() != "bf16":
        raise RuntimeError("Step3 v0 requires train_precision=bf16.")
    torch.backends.cuda.matmul.allow_tf32 = bool(final_cfg.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(final_cfg.allow_tf32)
    if bool(final_cfg.allow_tf32):
        torch.backends.cudnn.benchmark = True
    os.environ["ODCR_RUNTIME_PRECISION_MODE"] = "bf16"
    os.environ["ODCR_RUNTIME_ALLOW_TF32"] = "1" if bool(final_cfg.allow_tf32) else "0"
    os.environ["ODCR_RUNTIME_AMP_AUTOCAST"] = "1" if bool(final_cfg.amp_autocast) else "0"
    os.environ["ODCR_RUNTIME_GRAD_SCALER"] = "1" if bool(final_cfg.grad_scaler) else "0"
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Step3 v0 bf16 requested but current CUDA device does not report bf16 support.")


def summarize_step3_profile_buffers(model: nn.Module, optimizer: optim.Optimizer | None = None) -> dict[str, Any]:
    underlying = get_underlying_model(model)
    buffer_summary: dict[str, Any] = {}
    total_profile_bytes = 0
    named_buffers = dict(underlying.named_buffers())
    profile_names = step3_profile_buffer_names()
    for name in profile_names:
        value = named_buffers.get(name)
        if value is None:
            buffer_summary[name] = {"present": False}
            continue
        total_profile_bytes += int(value.numel() * value.element_size())
        buffer_summary[name] = {
            "present": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "bytes": int(value.numel() * value.element_size()),
            "requires_grad": bool(value.requires_grad),
            "persistent_state_dict": name in underlying.state_dict(),
        }
    trainable_named = step3_trainable_named_parameters(underlying)
    trainable_param_ids = {id(param) for _, param in trainable_named}
    profile_param_names = [name for name, _ in trainable_named if name in profile_names]
    optimizer_contains_profile = False
    optimizer_param_count = None
    if optimizer is not None:
        opt_ids = {id(param) for group in optimizer.param_groups for param in group.get("params", [])}
        optimizer_param_count = len(opt_ids)
        optimizer_contains_profile = bool(opt_ids & {id(getattr(underlying, name, None)) for name in profile_names})
    return {
        "trainable_param_count": int(sum(param.numel() for _, param in trainable_named)),
        "trainable_parameter_tensor_count": len(trainable_named),
        "frozen_buffer_tensor_count": len([name for name in profile_names if name in named_buffers]),
        "profile_domain_memory_bytes": int(total_profile_bytes),
        "profile_domain_memory_gib": round(total_profile_bytes / (1024 ** 3), 6),
        "profile_buffer_policy": str(getattr(underlying, "profile_buffer_policy", "gpu_resident")),
        "activation_checkpointing": dict(getattr(underlying, "activation_checkpointing_policy", {}) or {}),
        "profile_buffers": buffer_summary,
        "profile_domain_requires_grad_false": all(
            bool(buffer_summary[name].get("present")) and not bool(buffer_summary[name].get("requires_grad"))
            for name in profile_names
        ),
        "profile_domain_not_trainable_parameters": not profile_param_names,
        "profile_domain_trainable_parameter_names": profile_param_names,
        "optimizer_param_count": optimizer_param_count,
        "optimizer_contains_profile_domain_artifacts": optimizer_contains_profile,
        "optimizer_all_params_require_grad": (
            all(id(param) in trainable_param_ids for group in optimizer.param_groups for param in group.get("params", []))
            if optimizer is not None
            else None
        ),
    }


def init_step3_ddp_after_cache_ready(*, local_rank: int, rank: int, world_size: int) -> None:
    """Initialize NCCL only after every rank has loaded a completed tokenizer cache."""
    _ = local_rank
    if not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend="nccl")
    if rank == 0:
        logging.getLogger(LOGGER_NAME).info(
            "[DDP startup] NCCL process group initialized after completed tokenizer cache load.",
            extra=log_route_extra(logging.getLogger(LOGGER_NAME), ROUTE_SUMMARY),
        )


def build_config_and_data_ddp(args, rank: int, world_size: int, local_rank: int) -> tuple:
    _tid = resolve_task_idx_from_aux_target(args.auxiliary, args.target)
    if _tid is None:
        raise ValueError("未知的 auxiliary/target 组合")
    _ro = collect_training_hardware_overrides_from_args(args)
    resolved = build_resolved_training_config(
        args,
        task_idx=_tid,
        world_size=world_size,
        hardware_overrides=_ro,
    )
    G = int(resolved.train_batch_size)
    P = int(resolved.per_device_train_batch_size)

    train_dataset, valid_dataset, model, nuser, nitem, save_file, resolved_profiles = (
        _load_step3_artefacts(
            args,
            local_rank,
            resolved,
            rank=rank,
            world_size=world_size,
            log_tokenize=(rank == 0),
            show_datasets_progress=(rank == 0),
        )
    )

    train_drop_last = False
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=train_drop_last,
    )
    pin_memory = bool(resolved.pin_memory)
    persistent_workers = bool(resolved.persistent_workers)
    nw_train = int(resolved.dataloader_num_workers_train)
    nw_valid = int(resolved.dataloader_num_workers_valid)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=P,
        sampler=sampler,
        shuffle=False,
        num_workers=nw_train,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and nw_train > 0,
        prefetch_factor=resolved.dataloader_prefetch_factor_train,
        drop_last=train_drop_last,
    )
    n_train_batches = len(train_dataloader)
    if rank == 0:
        _acc_lg = logging.getLogger(LOGGER_NAME)
        _acc_msg = (
            f"[Train/no_accum] n_optimizer_steps={n_train_batches} "
            "batch_semantics_version=odcr_no_accum/1 "
            "formula=global_batch_size = per_gpu_batch_size * ddp_world_size"
        )
        if _acc_lg.handlers:
            _acc_lg.info(_acc_msg)
        else:
            print(_acc_msg, flush=True)
    if G % world_size != 0:
        raise ValueError(
            f"global_batch_size={G} 与 world_size={world_size} 不整除，无法得到每卡 batch。"
            "请修改 configs/odcr.yaml 中 step3.train.batch_size / per_gpu_batch_size 或 hardware.profiles.*.ddp_world_size。"
        )
    valid_per_rank = int(resolved.valid_micro_batch_size)
    n_valid = len(valid_dataset)
    if world_size > 1:
        _v_idx = _distributed_valid_sample_indices(n_valid, world_size, rank)
        # 即使某 rank 分片为空也必须用 Subset(..., [])，不能用「if _v_idx」回退全量（[] 在 Python 中为假值）
        valid_shard: TensorDataset | Subset = Subset(valid_dataset, _v_idx)
    else:
        valid_shard = valid_dataset
    valid_dataloader = DataLoader(
        valid_shard,
        batch_size=valid_per_rank,
        shuffle=False,
        num_workers=nw_valid,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and nw_valid > 0,
        prefetch_factor=resolved.dataloader_prefetch_factor_valid,
    )

    init_step3_ddp_after_cache_ready(local_rank=local_rank, rank=rank, world_size=world_size)

    _ddp_find_unused = bool(resolved.ddp_find_unused_parameters)
    _ddp_static_graph = bool(getattr(resolved, "ddp_static_graph", False))
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=_ddp_find_unused,
        static_graph=_ddp_static_graph,
        broadcast_buffers=False,
    )
    final_cfg = replace(
        resolved_profiles,
        nuser=nuser,
        nitem=nitem,
        save_file=save_file,
        device=local_rank,
        device_ids=tuple(range(world_size)),
        ddp_world_size=world_size,
        nlayers=args.nlayers,
        valid_dataset=valid_dataset,
        ddp_find_unused_parameters=_ddp_find_unused,
        ddp_static_graph=_ddp_static_graph,
        ddp_graph_safety_preflight=bool(getattr(resolved, "ddp_graph_safety_preflight", True)),
        rank0_only_logging=True,
    )
    setattr(get_underlying_model(model), "non_blocking_h2d", bool(final_cfg.non_blocking_h2d))

    return final_cfg, train_dataloader, valid_dataloader, model, sampler


def build_step3_training_components(args, rank: int, world_size: int, local_rank: int) -> tuple:
    """Build the live Step3 DDP components for formal training or validation windows."""

    return build_config_and_data_ddp(args, rank=rank, world_size=world_size, local_rank=local_rank)


def run_step3_measured_steps(**kwargs: Any) -> dict[str, Any]:
    """Run a bounded validation-only Step3 measured window."""

    from odcr_core.step3_runtime_probe import run_step3_validation_window as _run_step3_validation_window

    return _run_step3_validation_window(**kwargs)


def run_step3_validation_window(**kwargs: Any) -> dict[str, Any]:
    """Validation namespace API for bounded Step3 runtime probes."""

    return run_step3_measured_steps(**kwargs)


def trainModel_ddp(
    model,
    train_dataloader,
    valid_dataloader,
    sampler,
    final_cfg: FinalTrainingConfig,
    rank,
    world_size,
    max_steps=None,
    save_final_checkpoint: bool = False,
):
    epochs = final_cfg.epochs
    G = int(final_cfg.train_batch_size)
    P = int(final_cfg.per_device_train_batch_size)
    eff = int(final_cfg.effective_global_batch_size)
    initial_lr = float(final_cfg.scheduler_initial_lr)
    learning_rate = initial_lr
    _model = get_underlying_model(model)
    device = final_cfg.device
    apply_step3_precision_backend(final_cfg)
    use_bf16 = odcr_cuda_bf16_autocast_enabled()
    n_batches = len(train_dataloader)
    n_steps = max(1, n_batches)
    train_info = (
        f"[Train] global_batch_size={G} effective_global_batch_size={eff} "
        f"per_gpu_batch_size={P} world_size={world_size} "
        "batch_semantics_version=odcr_no_accum/1 "
        f"batches_per_epoch={n_batches} optimizer_steps_per_epoch={n_steps} epochs={epochs} "
        f"validate_every_epochs={int(final_cfg.validate_every_epochs)} max_grad_norm={float(final_cfg.max_grad_norm)}"
    )
    _lg = final_cfg.logger
    min_epochs = int(final_cfg.min_epochs)
    early_stop_patience = int(final_cfg.early_stop_patience)
    checkpoint_metric = str(final_cfg.checkpoint_metric)
    lr_scheduler_type = str(final_cfg.lr_scheduler)
    warmup_epochs = float(final_cfg.warmup_epochs)
    min_lr_ratio = float(final_cfg.min_lr_ratio)
    scheduler_cfg = json.loads(str(getattr(final_cfg, "scheduler_config_json", "") or "{}"))
    damping_cfg_for_semantics = (
        scheduler_cfg.get("validation_aware_lr_damping")
        if isinstance(scheduler_cfg.get("validation_aware_lr_damping"), Mapping)
        else {}
    )
    damping_enabled_for_semantics = bool(scheduler_cfg.get("damping_enabled", False))
    if lr_scheduler_type == "warmup_cosine" and damping_enabled_for_semantics:
        raise RuntimeError("hidden Step3 LR damping is forbidden for scheduler_type=warmup_cosine.")
    if lr_scheduler_type == "warmup_cosine_with_damping":
        raise RuntimeError("warmup_cosine_with_damping is retired from formal Step3; use pure warmup_cosine or probe-only safe_damping_v2.")
    if lr_scheduler_type == "safe_damping_v2" and not damping_enabled_for_semantics:
        raise RuntimeError("scheduler_type=safe_damping_v2 requires damping_enabled=true.")
    warmup_steps_env = final_cfg.odcr_warmup_steps
    warmup_ratio_env = final_cfg.odcr_warmup_ratio
    validate_every_epochs = max(1, int(final_cfg.validate_every_epochs))
    total_steps_plan = max(1, int(epochs * n_steps))
    enduration = 0
    prev_valid_loss = float("inf")
    if rank == 0:
        if _lg:
            _lg.info(train_info, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(train_info, flush=True)
        if use_bf16:
            _bf16_msg = "[Train] bf16 autocast: ON (runtime precision_mode=bf16)"
        elif os.environ.get("ODCR_RUNTIME_PRECISION_MODE", "").strip().lower() not in ("", "bf16"):
            _bf16_msg = "[Train] bf16 autocast: OFF (runtime precision_mode!=bf16)"
        elif not torch.cuda.is_available():
            _bf16_msg = "[Train] bf16 autocast: OFF (CUDA not available)"
        else:
            _bf16_msg = "[Train] bf16 autocast: OFF (torch.cuda.is_bf16_supported() is False)"
        if _lg:
            _lg.info(_bf16_msg, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_bf16_msg, flush=True)
        log_bf16_amp_note(_lg, use_bf16, has_grad_scaler=False)
        _es = (
            f"Early stop: min_epochs={min_epochs}, patience={early_stop_patience} "
            f"(valid_loss 连续变差), checkpoint_metric={checkpoint_metric}"
        )
        if _lg:
            _lg.info(_es, extra=log_route_extra(_lg, ROUTE_SUMMARY))
        else:
            print(_es, flush=True)
        if _lg:
            _lg.info(
                "Train profile: lr_scheduler=%s warmup_epochs=%g %s",
                lr_scheduler_type,
                warmup_epochs,
                "train-loop-metrics=valid_loss_only",
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )

    optimizer = build_step3_optimizer(model, final_cfg)
    if rank == 0 and _lg:
        _lg.info(
            "[Step3][Artifacts] frozen profile/domain summary: %s",
            json.dumps(summarize_step3_profile_buffers(model, optimizer), ensure_ascii=False, sort_keys=True),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
    ema_enabled = bool(getattr(final_cfg, "ema_enabled", True))
    ema_decay = float(getattr(final_cfg, "ema_decay", 0.999))
    ema_model: Optional[AveragedModel] = None
    if ema_enabled:
        ema_model = AveragedModel(_model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    sched = None
    ws_resolved = None
    warmup_ratio_logged = 0.0
    base_min_lr = initial_lr * min_lr_ratio
    damping_factor_cumulative = 1.0
    effective_min_lr = base_min_lr
    effective_min_lr_policy = str(
        damping_cfg_for_semantics.get("effective_min_lr_policy")
        or scheduler_cfg.get("effective_min_lr_policy")
        or "base_floor"
    )
    scheduler_state = scheduler_semantics(
        scheduler_type=lr_scheduler_type,
        damping_enabled=damping_enabled_for_semantics,
        base_min_lr=base_min_lr,
        damping_factor_cumulative=damping_factor_cumulative,
        effective_min_lr_policy=effective_min_lr_policy,
    )
    if lr_scheduler_type in ("warmup_cosine", "safe_damping_v2"):
        ws_resolved, warmup_ratio_logged = resolve_warmup_steps(
            total_steps_plan,
            n_steps,
            explicit_steps=warmup_steps_env,
            explicit_ratio=warmup_ratio_env,
            warmup_epochs_fallback=warmup_epochs,
        )
        lr_lambda = warmup_cosine_multiplier_lambda(ws_resolved, total_steps_plan, min_lr_ratio)
        sched = lr_sched.LambdaLR(optimizer, lr_lambda)
        if rank == 0 and _lg:
            _lg.info(
                "LR schedule resolved: scheduler_type=%s base_scheduler=%s damping_enabled=%s "
                "initial_lr=%s current_lr=%s (equals initial before first step) "
                "base_min_lr=%s effective_min_lr=%s min_lr_ratio=%s damping_factor_cumulative=%s "
                "warmup_steps=%d total_steps=%d warmup_ratio=%s | "
                "LambdaLR: one scheduler.step() immediately after each optimizer.step() (global_step aligned)",
                scheduler_state["scheduler_type"],
                scheduler_state["base_scheduler"],
                str(scheduler_state["damping_enabled"]).lower(),
                initial_lr,
                initial_lr,
                scheduler_state["base_min_lr"],
                scheduler_state["effective_min_lr"],
                min_lr_ratio,
                scheduler_state["damping_factor_cumulative"],
                ws_resolved,
                total_steps_plan,
                warmup_ratio_logged,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
    structured_weights = step3_structured_loss_weights_from_config(final_cfg)
    loss_semantics = step3_loss_semantics_from_config(final_cfg)
    objective_drift_cfg = json.loads(str(getattr(final_cfg, "objective_drift_config_json", "") or "{}"))
    recovery_cfg = json.loads(str(getattr(final_cfg, "recovery_config_json", "") or "{}"))
    phase_schedule_cfg = json.loads(str(getattr(final_cfg, "phase_loss_schedule_config_json", "") or "{}"))
    conflict_aware_cfg = json.loads(str(getattr(final_cfg, "conflict_aware_config_json", "") or "{}"))
    adapter_gating_cfg = json.loads(str(getattr(final_cfg, "adapter_gating_config_json", "") or "{}"))
    orth_w_xcov = structured_weights.orthogonal_xcov_weight
    orth_w_cos = structured_weights.orthogonal_cosine_weight
    graph_preflight_enabled = bool(getattr(final_cfg, "ddp_graph_safety_preflight", True))
    graph_preflight_summary: dict[str, Any] | None = None
    if rank == 0:
        if _lg:
            _lg.info(
                "[Step3][Orth] lambda_ortho=%g lambda_ortho_xcov=%g lambda_ortho_cos=%g "
                "(build_orthogonal_losses: total = w_xcov*xcov + w_cos*cos)",
                structured_weights.orthogonal_weight,
                orth_w_xcov,
                orth_w_cos,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
            _lg.info(
                "[Step3][Structured] variance=%g shared_inv=%g specific_sep=%g anchor=%g "
                "content_align=%g style_align=%g shared_proto=%g domain_style=%g local_style=%g polarity=%g proto_sep=%g residual=%g light_explainer=%g",
                structured_weights.variance_weight,
                structured_weights.shared_invariance_weight,
                structured_weights.specific_separation_weight,
                structured_weights.anchor_alignment_weight,
                structured_weights.content_alignment_weight,
                structured_weights.style_alignment_weight,
                structured_weights.shared_prototype_weight,
                structured_weights.domain_style_alignment_weight,
                structured_weights.local_style_alignment_weight,
                structured_weights.polarity_alignment_weight,
                structured_weights.prototype_separation_weight,
                structured_weights.residual_specific_weight,
                structured_weights.light_explainer_weight,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
    device_ids = list(final_cfg.device_ids) if final_cfg.device_ids else [device]
    train_nw = getattr(train_dataloader, "num_workers", 0)
    valid_nw = getattr(valid_dataloader, "num_workers", 0) if valid_dataloader is not None else None
    perf = None
    if rank == 0:
        perf = PerfMonitor(
            device=final_cfg.device,
            log_file=final_cfg.log_file,
            num_proc=final_cfg.num_proc,
            device_ids=device_ids,
            train_num_workers=train_nw,
            valid_num_workers=valid_nw,
            training_logger=_lg,
        )
        perf.start()
    step_count = 0
    global_step = 0
    best_version = 0
    step_iv = max(1, odcr_log_step_interval())
    grad_iv = max(1, odcr_log_grad_interval())
    checkpoint_policy_cfg = json.loads(str(getattr(final_cfg, "checkpoint_policy_config_json", "") or "{}"))
    quality_gate_cfg = json.loads(str(getattr(final_cfg, "quality_gate_config_json", "") or "{}"))
    grad_finite_cfg = json.loads(str(getattr(final_cfg, "grad_finite_config_json", "") or "{}"))
    diagnostic_eval_cfg = json.loads(str(getattr(final_cfg, "diagnostic_eval_config_json", "") or "{}"))
    checkpoint_metric_name = str(checkpoint_policy_cfg.get("selection_metric") or checkpoint_metric or "valid_loss")
    checkpoint_direction = str(checkpoint_policy_cfg.get("selection_direction") or "min")
    checkpoint_top_k = max(3, int(checkpoint_policy_cfg.get("top_k", 3) or 3))
    per_epoch_policy = checkpoint_policy_cfg.get("per_epoch") if isinstance(checkpoint_policy_cfg.get("per_epoch"), Mapping) else {}
    model_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(final_cfg.save_file))))
    run_dir = os.path.dirname(model_dir)
    state_dir = os.path.join(run_dir, "state")
    meta_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(getattr(final_cfg, "log_file", "") or "")))) or os.path.join(run_dir, "meta")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    best_observed_path = os.path.join(model_dir, "best_observed.pth")
    best_after_min_epochs_path = os.path.join(model_dir, "best_after_min_epochs.pth")
    latest_checkpoint_path = os.path.join(model_dir, "latest.pth")
    topk_dir = os.path.join(model_dir, "topk")
    per_epoch_dir = os.path.join(model_dir, "per_epoch")
    os.makedirs(topk_dir, exist_ok=True)
    if bool(per_epoch_policy.get("enabled", False)):
        os.makedirs(per_epoch_dir, exist_ok=True)
    best_observed_metric: float | None = None
    best_observed_epoch: int | None = None
    best_observed_event: Dict[str, Any] | None = None
    best_after_min_metric: float | None = None
    best_after_min_epoch: int | None = None
    best_after_min_event: Dict[str, Any] | None = None
    latest_event: Dict[str, Any] | None = None
    topk_candidates: list[tuple[float, int, Dict[str, Any]]] = []
    grad_inf_count_until_epoch = 0
    continuous_nonfinite_steps = 0
    post_clip_zero_count = 0
    optimizer_step_attempts = 0
    scheduler_events: list[dict[str, Any]] = []
    damping_events: list[dict[str, Any]] = []
    training_effectiveness_records: list[dict[str, Any]] = []
    objective_drift_records: list[dict[str, Any]] = []
    recovery_events: list[dict[str, Any]] = []
    phase_history: list[dict[str, Any]] = []
    loss_component_records: list[dict[str, Any]] = []
    damping_state = {"worsen_count": 0, "cooldown_remaining": 0, "event_count": 0}
    recovery_state = {"active": False, "remaining_epochs": 0, "count": 0}
    latest_objective_drift_status = "none"
    _finite_mode, _finite_warn = parse_odcr_finite_check_mode()
    _nonfinite_skips = 0
    _nonfinite_abort_th = _nonfinite_loss_abort_threshold()
    if rank == 0 and _lg:
        if _finite_warn:
            _lg.warning(
                "[Diag] %s",
                _finite_warn,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
        _lg.info(
            "[Diag] finite_check_mode=%s（环境变量 ODCR_FINITE_CHECK_MODE；默认 loss_only）",
            _finite_mode,
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[Step3][Quality] checkpoint_policy=%s quality_gate=%s grad_finite=%s evidence_levels=code_present/active_path/runtime_verified/formal_verified/downstream_eligible",
            json.dumps(checkpoint_policy_cfg, ensure_ascii=False, sort_keys=True),
            json.dumps(quality_gate_cfg, ensure_ascii=False, sort_keys=True),
            json.dumps(grad_finite_cfg, ensure_ascii=False, sort_keys=True),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )

    def _hash_meta_file(name: str) -> str:
        path = os.path.join(meta_dir, name)
        if not os.path.isfile(path):
            return ""
        fp = file_fingerprint(path)
        return str(fp.get("sha256") or fp.get("sample_sha256") or "")

    def _checkpoint_context(*, epoch: int, metric: float, scope: str, reason: str, replaced_previous: bool) -> dict[str, Any]:
        optimizer_state_path = os.path.join(state_dir, "optimizer.pt")
        optimizer_hash = _sha256_file(optimizer_state_path) if os.path.isfile(optimizer_state_path) else ""
        return {
            "checkpoint_epoch": int(epoch),
            "selection_metric": checkpoint_metric_name,
            "selection_metric_value": float(metric),
            "selection_direction": checkpoint_direction,
            "selection_scope": scope,
            "reason": str(reason),
            "replaced_previous": bool(replaced_previous),
            "global_best_epoch": best_observed_epoch if best_observed_epoch is not None else epoch,
            "global_best_metric": best_observed_metric if best_observed_metric is not None else metric,
            "after_min_epochs_best_epoch": best_after_min_epoch,
            "after_min_epochs_best_metric": best_after_min_metric,
            "epoch_summary_hash": _hash_meta_file("epoch_summary.csv"),
            "metrics_jsonl_hash": _hash_meta_file("metrics.jsonl"),
            "quality_status_at_save": "not_evaluated",
            "quality_status": "not_evaluated",
            "downstream_ready": False,
            "grad_inf_count_until_epoch": int(grad_inf_count_until_epoch),
            "optimizer_state_hash": optimizer_hash,
            "code_commit": _git_code_fingerprint_for_step3().get("git_commit", ""),
        }

    def _save_step3_checkpoint(
        path: str,
        *,
        epoch: int,
        metric: float,
        scope: str,
        reason: str,
        replaced_previous: bool,
    ) -> Dict[str, Any]:
        state_to_save = state_dict_for_canonical_best_pth(
            ema_enabled=ema_enabled,
            ema_model=ema_model,
            ddp_module=model,
            underlying_model_fn=get_underlying_model,
        )
        ckpt_t0 = time.perf_counter()
        atomic_torch_save(path, state_to_save)
        if bool(checkpoint_policy_cfg.get("save_optimizer_state", True)):
            atomic_torch_save(os.path.join(state_dir, "optimizer.pt"), optimizer.state_dict())
        lineage = _build_step3_checkpoint_lineage(
            final_cfg,
            checkpoint_path=str(path),
            checkpoint_context=_checkpoint_context(
                epoch=epoch,
                metric=metric,
                scope=scope,
                reason=reason,
                replaced_previous=replaced_previous,
            ),
        )
        lineage["reason"] = str(reason)
        lineage["replaced_previous"] = bool(replaced_previous)
        lineage_path = write_checkpoint_lineage(str(path), lineage)
        atomic_write_json(
            os.path.join(state_dir, "trainer_state.json"),
            {"epoch": int(epoch), "global_step": int(global_step), "continuous_nonfinite_steps": int(continuous_nonfinite_steps)},
        )
        event = checkpoint_event_from_sidecar(lineage, reason=reason, replaced_previous=replaced_previous)
        event["lineage_path"] = str(lineage_path)
        event["checkpoint_io_ms"] = (time.perf_counter() - ckpt_t0) * 1000.0
        append_step3_gpu_profile_jsonl(
            log_file=getattr(final_cfg, "log_file", None),
            row=_step3_gpu_profile_row(
                final_cfg=final_cfg,
                rank=rank,
                device=device,
                global_step=global_step,
                epoch=epoch,
                phase="after_checkpoint_save",
            ),
        )
        return event

    def _apply_validation_aware_lr_damping(epoch: int, valid_loss_value: float, *, emit: bool) -> None:
        nonlocal damping_factor_cumulative, effective_min_lr
        if lr_scheduler_type != "safe_damping_v2":
            return
        damping_cfg = scheduler_cfg.get("validation_aware_lr_damping")
        if not isinstance(damping_cfg, Mapping) or not bool(damping_cfg.get("enabled", False)):
            return
        if best_observed_metric is None:
            return
        decision = safe_damping_v2_decision(
            epoch=epoch,
            valid_loss=valid_loss_value,
            best_valid_loss=float(best_observed_metric),
            previous_valid_loss=None if not math.isfinite(prev_valid_loss) else prev_valid_loss,
            current_lr=float(optimizer.param_groups[0].get("lr", 0.0) or 0.0),
            base_min_lr=float(base_min_lr),
            event_count=int(damping_state.get("event_count", 0) or 0),
            cooldown_remaining=int(damping_state.get("cooldown_remaining", 0) or 0),
            config=damping_cfg,
        )
        damping_state["cooldown_remaining"] = int(decision.get("cooldown_remaining", damping_state.get("cooldown_remaining", 0)) or 0)
        if not bool(decision.get("apply", False)):
            if decision.get("action_gate") and emit:
                scheduler_events.append(
                    {
                        "event": "safe_damping_v2_action_gate",
                        "scheduler_type": "safe_damping_v2",
                        "epoch": int(epoch),
                        "reason": str(decision.get("reason") or ""),
                        "action_gate": str(decision.get("action_gate") or ""),
                        "event_count": int(damping_state.get("event_count", 0) or 0),
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                )
            return
        factor = float(decision.get("lr_decay_factor", damping_cfg.get("lr_decay_factor", 0.5)) or 0.5)
        new_damping_factor_cumulative = float(damping_factor_cumulative) * factor
        new_effective_min_lr = float(decision.get("effective_lr_floor", float(base_min_lr) * 0.25))
        before_lrs = [float(group.get("lr", 0.0) or 0.0) for group in optimizer.param_groups]
        before_base_lrs = [
            float(x)
            for x in (
                getattr(sched, "base_lrs", None)
                if sched is not None and hasattr(sched, "base_lrs")
                else [group.get("initial_lr", initial_lr) for group in optimizer.param_groups]
            )
        ]
        after_lrs: list[float] = []
        for group in optimizer.param_groups:
            old = float(group.get("lr", 0.0) or 0.0)
            old_initial = float(group.get("initial_lr", initial_lr) or initial_lr)
            new_initial = max(old_initial * factor, new_effective_min_lr)
            new = max(old * factor, new_effective_min_lr)
            group["lr"] = new
            group["initial_lr"] = new_initial
            after_lrs.append(float(new))
        if sched is not None and hasattr(sched, "base_lrs"):
            sched.base_lrs = [max(float(x) * factor, new_effective_min_lr) for x in getattr(sched, "base_lrs")]
        damping_factor_cumulative = new_damping_factor_cumulative
        effective_min_lr = new_effective_min_lr
        damping_state["cooldown_remaining"] = int(decision.get("cooldown_remaining", damping_cfg.get("cooldown_epochs", 3)) or 3)
        damping_state["event_count"] = int(damping_state.get("event_count", 0) or 0) + 1
        damping_state["worsen_count"] = 0
        if not emit:
            return
        event = {
            "event": "validation_aware_lr_damping",
            "scheduler_type": "safe_damping_v2",
            "base_scheduler": "warmup_cosine",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "monitor_metric": "valid_loss",
            "damping_reason": str(decision.get("reason") or "safe_damping_v2_recent_trend_worsened"),
            "current_valid_loss": float(valid_loss_value),
            "best_observed_metric": float(best_observed_metric),
            "worsen_abs": float(decision.get("worsen_abs", float(valid_loss_value) - float(best_observed_metric))),
            "worsen_ratio": float(decision.get("worsen_ratio", 0.0)),
            "base_lr_before_damping": before_base_lrs,
            "lr_before": before_lrs,
            "lr_after": after_lrs,
            "effective_lr_after_damping": after_lrs,
            "damping_factor": factor,
            "damping_factor_cumulative": float(damping_factor_cumulative),
            "event_count": int(damping_state.get("event_count", 0) or 0),
            "max_damping_events": int(damping_cfg.get("max_damping_events", 2) or 2),
            "base_min_lr": float(base_min_lr),
            "effective_min_lr": float(effective_min_lr),
            "effective_min_lr_policy": effective_min_lr_policy,
            "cooldown_epochs": int(damping_cfg.get("cooldown_epochs", 3) or 3),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        scheduler_events.append(event)
        damping_events.append(event)
        append_step3_scheduler_events_jsonl(log_file=getattr(final_cfg, "log_file", None), row=event)
        append_step3_damping_events_jsonl(log_file=getattr(final_cfg, "log_file", None), row=event)
        if _lg:
            _lg.warning(
                "[Step3][LRDamping] scheduler_type=safe_damping_v2 epoch=%d "
                "valid_loss=%.6f best=%.6f base_min_lr=%.6g effective_min_lr=%.6g "
                "damping_factor_cumulative=%.6g lr_before=%s lr_after=%s",
                epoch,
                valid_loss_value,
                float(best_observed_metric),
                float(base_min_lr),
                float(effective_min_lr),
                float(damping_factor_cumulative),
                before_lrs,
                after_lrs,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )

    def _component_weighted_means_for_epoch(epoch: int) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for row in loss_component_records:
            try:
                if int(row.get("epoch", -1)) != int(epoch):
                    continue
                name = str(row.get("loss_name") or "")
                if not name:
                    continue
                values.setdefault(name, []).append(float(row.get("weighted_value", 0.0) or 0.0))
            except Exception:
                continue
        return {key: sum(items) / len(items) for key, items in values.items() if items}

    def _component_weighted_deltas(epoch: int, best_epoch: int | None) -> dict[str, float]:
        if best_epoch is None:
            return {}
        cur = _component_weighted_means_for_epoch(epoch)
        base = _component_weighted_means_for_epoch(best_epoch)
        return {key: float(cur[key]) - float(base.get(key, cur[key])) for key in cur}

    def _activate_recovery_from_best(epoch: int, drift_record: Mapping[str, Any]) -> None:
        nonlocal optimizer, sched, ws_resolved, total_steps_plan, warmup_ratio_logged, latest_objective_drift_status
        if not bool(recovery_cfg.get("enabled", True)):
            return
        if int(recovery_state.get("count", 0) or 0) >= int(recovery_cfg.get("max_recoveries", 1) or 1):
            return
        recovery_state["count"] = int(recovery_state.get("count", 0) or 0) + 1
        recovery_state["active"] = True
        recovery_state["remaining_epochs"] = int(recovery_cfg.get("recovery_epochs", 8) or 8)
        plan = build_recovery_plan(
            epoch=epoch,
            drift_record=drift_record,
            config=recovery_cfg,
            best_observed_checkpoint=str(best_observed_path),
            latest_checkpoint=str(latest_checkpoint_path),
            recovery_index=int(recovery_state["count"]),
        )
        if rank == 0:
            recovery_events.append(plan)
            append_step3_recovery_events_jsonl(log_file=getattr(final_cfg, "log_file", None), row=plan)
            if bool(recovery_cfg.get("save_drift_checkpoint", True)):
                diag_dir = os.path.join(model_dir, "drift_diagnostics")
                os.makedirs(diag_dir, exist_ok=True)
                _save_step3_checkpoint(
                    os.path.join(diag_dir, f"epoch_{int(epoch):03d}_drift_diagnostic.pth"),
                    epoch=epoch,
                    metric=float(drift_record.get("valid_loss", 0.0) or 0.0),
                    scope="drift_diagnostic",
                    reason="objective_drift_diagnostic_before_recovery",
                    replaced_previous=False,
                )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        state = torch.load(best_observed_path, map_location=device)
        get_underlying_model(model).load_state_dict(state, strict=False)
        optimizer = build_step3_optimizer(model, final_cfg)
        restart_lr = float(initial_lr) * float(recovery_cfg.get("restart_lr_ratio", 0.25) or 0.25)
        for group in optimizer.param_groups:
            group["lr"] = restart_lr
            group["initial_lr"] = restart_lr
        recovery_steps = max(1, int(recovery_state["remaining_epochs"]) * n_steps)
        total_steps_plan = recovery_steps
        ws_resolved = max(1, int(0.10 * recovery_steps))
        warmup_ratio_logged = float(ws_resolved) / float(recovery_steps)
        sched = lr_sched.LambdaLR(optimizer, warmup_cosine_multiplier_lambda(ws_resolved, recovery_steps, min_lr_ratio))
        latest_objective_drift_status = "recovery_active"
    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            epoch_1 = epoch + 1
            if rank == 0:
                perf.epoch_start()
            model.train()
            recovery_active = bool(recovery_state.get("active", False))
            phase_record = resolve_phase_for_epoch(
                epoch=epoch_1,
                config=phase_schedule_cfg,
                objective_drift_status=latest_objective_drift_status,
                recovery_active=recovery_active,
            )
            phase_weights = apply_step3_phase_loss_multipliers(
                structured_weights,
                phase_record.get("loss_multipliers") if bool(phase_record.get("enabled", True)) else {},
            )
            if rank == 0:
                phase_history.append(dict(phase_record))
            loss_sum = torch.zeros((), dtype=torch.double, device=device)
            n_samples = torch.zeros((), dtype=torch.double, device=device)
            optimizer.zero_grad(set_to_none=True)
            prefetch_cfg = json.loads(str(getattr(final_cfg, "prefetcher_config_json", "") or "{}"))
            timing_cfg = json.loads(str(getattr(final_cfg, "timing_config_json", "") or "{}"))
            prefetcher = None
            iterator = train_dataloader
            if bool(prefetch_cfg.get("enabled", True)):
                prefetcher = Step3CUDAPrefetcher(
                    train_dataloader,
                    device=device,
                    non_blocking=bool(getattr(final_cfg, "non_blocking_h2d", True)),
                    enabled=True,
                    diagnostic_cpu_mode=bool(prefetch_cfg.get("diagnostic_cpu_mode", False)),
                    double_buffer=bool(prefetch_cfg.get("double_buffer", True)),
                    fallback_policy=str(prefetch_cfg.get("fallback_policy") or "fail_fast"),
                )
                iterator = prefetcher
            if rank == 0:
                iterator = tqdm(iterator, total=len(train_dataloader))
            for step_idx, batch in enumerate(iterator):
                _step_t0 = time.perf_counter()
                step_timing: Dict[str, Any] = {field: 0.0 for field in _STEP3_PREFETCH_TIMING_FIELDS}
                if prefetcher is not None:
                    step_timing.update(prefetcher.last_timing)
                    step_timing.update(prefetcher.last_evidence)
                gn_pre_m = None
                gn_post_m = None
                step_count += 1
                _gather_t0 = time.perf_counter()
                g = require_gathered_batch(_model.gather(batch, device))
                step_timing["structured_gather_ms"] += (time.perf_counter() - _gather_t0) * 1000.0
                user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = (
                    g.user_idx,
                    g.item_idx,
                    g.rating,
                    g.tgt_input,
                    g.tgt_output,
                    g.domain_idx,
                )
                c_a = g.content_anchor_score
                s_a = g.style_anchor_score
                ce = g.content_evidence_ids
                se = g.style_evidence_ids
                dsa = g.domain_style_anchor_ids
                lsh = g.local_style_hint_ids
                pol = g.polarity_ids
                eq = g.evidence_quality_prior
                if any(value is None for value in (c_a, s_a, ce, se, dsa, lsh, pol, eq)):
                    raise RuntimeError("Step3 gather 缺少 canonical evidence 张量。")
                bsz = int(user_idx.size(0))
                warn_empty_batch(_lg, global_step=global_step, epoch=epoch_1, bsz=bsz)
                _sync_t0 = time.perf_counter()
                with odcr_cuda_bf16_autocast():
                    _forward_t0 = time.perf_counter()
                    forward_out = model(
                        user_idx,
                        item_idx,
                        tgt_input,
                        domain_idx,
                        content_anchor=c_a,
                        style_anchor=s_a,
                        content_evidence_ids=ce,
                        style_evidence_ids=se,
                        domain_style_anchor_ids=dsa,
                        local_style_hint_ids=lsh,
                        polarity_ids=pol,
                        evidence_quality_prior=eq,
                    )
                    step_timing["forward_time"] += time.perf_counter() - _forward_t0
                    _loss_t0 = time.perf_counter()
                    loss_bundle = compose_step3_loss_from_forward_output(
                        forward_output=forward_out,
                        batch=g,
                        final_cfg=final_cfg,
                        weights=phase_weights,
                        semantics=loss_semantics,
                    )
                    loss_bundle.logging_summary["phase"] = phase_record
                    loss = loss_bundle.total_loss
                    step_timing["loss_time"] += time.perf_counter() - _loss_t0
                    if graph_preflight_enabled and graph_preflight_summary is None:
                        graph_preflight_summary = validate_step3_graph_safety_preflight(
                            forward_output=forward_out,
                            loss_bundle=loss_bundle,
                            underlying_model=_model,
                            ctx="step3/train",
                        )
                        if rank == 0 and _lg:
                            _lg.info(
                                "[DDP preflight] Step3 graph-safety preflight passed: %s",
                                json.dumps(graph_preflight_summary, ensure_ascii=False, sort_keys=True),
                                extra=log_route_extra(_lg, ROUTE_SUMMARY),
                            )
                    _finite_sync_t0 = time.perf_counter()
                    finite_sync = step3_sync_loss_bundle_finite_status(loss_bundle, world_size=world_size)
                    step_timing["finite_sync_ms"] += (time.perf_counter() - _finite_sync_t0) * 1000.0
                    local_loss_finite = bool(finite_sync["local_total_finite"])
                    global_loss_finite = bool(finite_sync["global_total_finite"])
                    if global_loss_finite:
                        _backward_t0 = time.perf_counter()
                        loss.backward()
                        step_timing["backward_time"] += time.perf_counter() - _backward_t0
                    else:
                        _nonfinite_skips += 1
                        optimizer.zero_grad(set_to_none=True)
                        if rank == 0 and _lg:
                            _lg.warning(
                                "[Train] non-finite loss synchronized skip optimizer: "
                                "epoch=%d optimizer_step=%d local_finite=%s",
                                epoch_1,
                                global_step + 1,
                                str(local_loss_finite).lower(),
                                extra=log_route_extra(_lg, ROUTE_SUMMARY),
                            )
                        if _nonfinite_abort_th > 0 and _nonfinite_skips >= _nonfinite_abort_th:
                            raise RuntimeError(
                                f"非有限 loss 累计同步跳过 {_nonfinite_skips} 次 >= "
                                f"ODCR_NONFINITE_LOSS_ABORT_AFTER={_nonfinite_abort_th}"
                            )
                step_timing["sync_time"] += time.perf_counter() - _sync_t0
                if global_loss_finite:
                    grad_check_t0 = time.perf_counter()
                    grad_inspection = inspect_gradients(
                        step3_trainable_named_parameters(model),
                        topk=int(grad_finite_cfg.get("anomaly_topk", 5) or 5),
                    )
                    step_timing["grad_check_ms"] += (time.perf_counter() - grad_check_t0) * 1000.0
                    step_timing["grad_norm_compute_ms"] += float(step_timing["grad_check_ms"])
                    gn_pre_m = float(grad_inspection.grad_norm_pre_clip)
                    global_grad_finite = sync_grad_finite_decision(
                        bool(grad_inspection.grad_finite),
                        device=torch.device(f"cuda:{int(device)}" if isinstance(device, int) else device),
                        world_size=world_size,
                    )
                    step_timing["grad_finite"] = bool(global_grad_finite)
                    step_timing["grad_norm_pre_clip"] = gn_pre_m
                    step_timing["nonfinite_param_count"] = int(grad_inspection.nonfinite_param_count)
                    step_timing["nonfinite_param_topk"] = list(grad_inspection.nonfinite_param_topk)
                    step_timing["nonfinite_param_group_topk"] = list(grad_inspection.nonfinite_param_group_topk)
                    optimizer_step_executed = False
                    scheduler_step_executed = False
                    skipped_step_reason = ""
                    if not global_grad_finite and bool(grad_finite_cfg.get("skip_optimizer_on_nonfinite", True)):
                        grad_inf_count_until_epoch += 1
                        continuous_nonfinite_steps += 1
                        skipped_step_reason = "nonfinite_grad"
                        step_timing["nonfinite_detect_ms"] += float(step_timing["grad_check_ms"])
                        _zero_t0 = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        step_timing["zero_grad_ms"] += (time.perf_counter() - _zero_t0) * 1000.0
                        if sched is not None and bool(grad_finite_cfg.get("scheduler_step_on_skipped_optimizer", False)):
                            _scheduler_t0 = time.perf_counter()
                            sched.step()
                            scheduler_step_executed = True
                            step_timing["scheduler_ms"] += (time.perf_counter() - _scheduler_t0) * 1000.0
                            step_timing["scheduler_time"] += float(step_timing["scheduler_ms"]) / 1000.0
                        if rank == 0 and _lg:
                            _lg.warning(
                                "[Step3][GradFinite] nonfinite grad skip optimizer: epoch=%d step=%d nonfinite_params=%d topk=%s continuous=%d",
                                epoch_1,
                                global_step + 1,
                                int(grad_inspection.nonfinite_param_count),
                                json.dumps(list(grad_inspection.nonfinite_param_topk), ensure_ascii=False),
                                int(continuous_nonfinite_steps),
                                extra=log_route_extra(_lg, ROUTE_SUMMARY),
                            )
                        abort_th = int(grad_finite_cfg.get("continuous_nonfinite_abort_threshold", 0) or 0)
                        if abort_th > 0 and continuous_nonfinite_steps >= abort_th:
                            raise RuntimeError(
                                f"Step3 nonfinite gradient gate aborted after {continuous_nonfinite_steps} continuous skipped steps."
                            )
                    else:
                        continuous_nonfinite_steps = 0
                        _clip_t0 = time.perf_counter()
                        nn.utils.clip_grad_norm_(
                            step3_trainable_parameters(model),
                            float(final_cfg.max_grad_norm),
                        )
                        step_timing["grad_clip_ms"] += (time.perf_counter() - _clip_t0) * 1000.0
                        _grad_monitor_t0 = time.perf_counter()
                        if rank == 0 and _lg:
                            gn_post_m = grad_norm_total(step3_trainable_parameters(model))
                            step_timing["grad_norm_post_clip"] = float(gn_post_m)
                            if float(gn_post_m) == 0.0:
                                post_clip_zero_count += 1
                            log_grad_monitor(
                                _lg,
                                model,
                                global_step=global_step + 1,
                                epoch=epoch_1,
                                route_detail=ROUTE_DETAIL,
                                grad_norm_pre_clip=gn_pre_m,
                                grad_norm_post_clip=gn_post_m,
                                skip_param_topk=not ((global_step + 1) % max(1, int(grad_finite_cfg.get("monitor_interval_steps", grad_iv) or grad_iv)) == 0),
                            )
                        step_timing["grad_monitor_ms"] += (time.perf_counter() - _grad_monitor_t0) * 1000.0
                        _optimizer_t0 = time.perf_counter()
                        optimizer.step()
                        optimizer_step_executed = True
                        step_timing["optimizer_ms"] += (time.perf_counter() - _optimizer_t0) * 1000.0
                        if ema_model is not None:
                            _ema_t0 = time.perf_counter()
                            ema_model.update_parameters(_model)
                            step_timing["ema_ms"] += (time.perf_counter() - _ema_t0) * 1000.0
                        _zero_t0 = time.perf_counter()
                        optimizer.zero_grad(set_to_none=True)
                        step_timing["zero_grad_ms"] += (time.perf_counter() - _zero_t0) * 1000.0
                        step_timing["optimizer_time"] += (
                            float(step_timing["optimizer_ms"])
                            + float(step_timing["ema_ms"])
                            + float(step_timing["zero_grad_ms"])
                        ) / 1000.0
                        if sched is not None:
                            _scheduler_t0 = time.perf_counter()
                            sched.step()
                            scheduler_step_executed = True
                            step_timing["scheduler_ms"] += (time.perf_counter() - _scheduler_t0) * 1000.0
                            step_timing["scheduler_time"] += float(step_timing["scheduler_ms"]) / 1000.0
                        global_step += 1
                    step_timing["optimizer_step_executed"] = bool(optimizer_step_executed)
                    step_timing["scheduler_step_executed"] = bool(scheduler_step_executed)
                    step_timing["skipped_step_reason"] = skipped_step_reason
                    step_timing["continuous_nonfinite_steps"] = int(continuous_nonfinite_steps)
                    optimizer_step_attempts += 1
                    maybe_log_grad_norm_diff_ddp(
                        model,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        global_step=global_step,
                        logger=_lg,
                        route_detail=ROUTE_DETAIL,
                    )
                    if rank == 0 and _lg and global_step > 0 and global_step % step_iv == 0:
                        _logging_t0 = time.perf_counter()
                        _lr_now = optimizer.param_groups[0]["lr"]
                        _extra_na: Dict[str, Any] = {
                            "loss_ortho": float(loss_bundle.components["L_orthogonal"].detach().item()),
                            "loss_ortho_xcov": float(loss_bundle.diagnostics["L_orthogonal_xcov"].detach().item()),
                            "loss_ortho_cos": float(loss_bundle.diagnostics["L_orthogonal_cosine"].detach().item()),
                            "loss_var_total": float(loss_bundle.components["L_variance"].detach().item()),
                        }
                        if odcr_log_step_loss_parts():
                            _extra_na.update(
                                {
                                    key: float(value.detach().item())
                                    for key, value in sorted(loss_bundle.components.items())
                                }
                            )
                            _extra_na.update(
                                {
                                    "shared_std_mean": float(
                                        loss_bundle.diagnostics["shared_std_mean"].detach().item()
                                    ),
                                    "specific_std_mean": float(
                                        loss_bundle.diagnostics["specific_std_mean"].detach().item()
                                    ),
                                }
                            )
                        _extra_na["batch_semantics_version"] = "odcr_no_accum/1"
                        if gn_pre_m is not None and gn_post_m is not None:
                            _extra_na["grad_norm_pre_clip"] = float(gn_pre_m)
                            _extra_na["grad_norm_post_clip"] = float(gn_post_m)
                        if bool(timing_cfg.get("enabled", True)):
                            step_timing["total_step_time"] = time.perf_counter() - _step_t0
                            _extra_na["startup_steady_state_timing"] = {
                                key: (
                                    float(step_timing.get(key, 0.0))
                                    if isinstance(step_timing.get(key, 0.0), (int, float))
                                    else step_timing.get(key)
                                )
                                for key in _STEP3_PREFETCH_TIMING_FIELDS
                            }
                        log_step_sample(
                            _lg,
                            global_step=global_step,
                            epoch=epoch_1,
                            lr=float(_lr_now),
                            train_loss_batch=float(loss.detach().item()),
                            extra=_extra_na or None,
                        )
                        step_timing["logging_time"] += time.perf_counter() - _logging_t0
                        step_timing["total_step_time"] = time.perf_counter() - _step_t0
                        _throughput = 0.0
                        if float(step_timing["total_step_time"]) > 0.0:
                            _throughput = float(bsz * world_size) / float(step_timing["total_step_time"])
                        append_train_epoch_metrics_jsonl(
                            log_file=getattr(final_cfg, "log_file", None),
                            row={
                                "run_id": str(getattr(final_cfg, "run_id", "") or ""),
                                "task_id": int(getattr(final_cfg, "task_idx", 0) or 0),
                                "profile_id": str(getattr(final_cfg, "task_profile_id", "") or ""),
                                "epoch": int(epoch_1),
                                "global_step": int(global_step),
                                "rank": int(rank),
                                "split": "train",
                                "loss_total": float(loss.detach().item()),
                                "lr": float(_lr_now),
                                "finite": bool(global_loss_finite),
                                "grad_finite": bool(step_timing.get("grad_finite", True)),
                                "optimizer_step_executed": bool(step_timing.get("optimizer_step_executed", False)),
                                "scheduler_step_executed": bool(step_timing.get("scheduler_step_executed", False)),
                                "skipped_step_reason": str(step_timing.get("skipped_step_reason") or ""),
                                "continuous_nonfinite_steps": int(step_timing.get("continuous_nonfinite_steps", 0) or 0),
                                "nonfinite_param_count": int(step_timing.get("nonfinite_param_count", 0) or 0),
                                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                            },
                        )
                        _loss_rows = _step3_loss_breakdown_row(
                            loss_bundle,
                            final_cfg=final_cfg,
                            rank=rank,
                            global_step=global_step,
                            epoch=epoch_1,
                        )
                        loss_component_records.extend(_loss_rows)
                        for _loss_row in _loss_rows:
                            append_step3_loss_breakdown_jsonl(
                                log_file=getattr(final_cfg, "log_file", None),
                                row=_loss_row,
                            )
                        append_step3_timing_profile_jsonl(
                            log_file=getattr(final_cfg, "log_file", None),
                            row=_step3_timing_profile_row(
                                step_timing,
                                final_cfg=final_cfg,
                                rank=rank,
                                global_step=global_step,
                                epoch=epoch_1,
                                samples_per_sec=float(_throughput),
                            ),
                        )
                        append_step3_gpu_profile_jsonl(
                            log_file=getattr(final_cfg, "log_file", None),
                            row=_step3_gpu_profile_row(
                                final_cfg=final_cfg,
                                rank=rank,
                                device=device,
                                global_step=global_step,
                                epoch=epoch_1,
                            ),
                        )

                if global_loss_finite:
                    loss_sum = loss_sum + loss.detach().double() * bsz
                    n_samples += bsz
                if global_loss_finite and step_count % step_iv == 0 and rank == 0:
                    run_training_finite_checks(
                        _finite_mode,
                        loss,
                        forward_out.word_dist,
                        _lg,
                        global_step=global_step,
                        epoch=epoch_1,
                        route_detail=ROUTE_DETAIL,
                    )

                # 用于快速验证：跑到指定 steps 后直接退出，观察是否触发 DDP reduction 错误
                if max_steps is not None and step_count >= max_steps:
                    return

            ddp_heartbeat(_lg, "before_train_loss_allreduce", rank=rank, epoch=epoch_1)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
            ddp_heartbeat(_lg, "after_train_loss_allreduce", rank=rank, epoch=epoch_1)
            _ns_tot = float(n_samples.item())
            avg_loss = (loss_sum / n_samples).item() if _ns_tot > 0 else float("nan")

            lr_epoch = optimizer.param_groups[0]["lr"]
            ddp_heartbeat(_lg, "before_gpu_stats_allgather", rank=rank, epoch=epoch_1)
            _gu, _gm, _gpeak = gather_ddp_gpu_stats_for_epoch_log(rank, world_size, int(device))
            ddp_heartbeat(_lg, "after_gpu_stats_allgather", rank=rank, epoch=epoch_1)
            if rank == 0:
                rec = perf.epoch_end(epoch + 1, len(train_dataloader), emit_log=False)
                rec["gpu_util"] = _gu
                rec["gpu_mem"] = _gm
                if _gpeak is not None:
                    rec["gpu_mem_bytes"] = _gpeak
            else:
                rec = None

            # 各 rank valid 分片上样本加权求和，一次 all_reduce 得全局 avg valid loss（与 train / Step5 一致）。
            _t_valid0 = time.perf_counter()
            valid_loss_sum, valid_n_samples = validModel_sum_batches(model, valid_dataloader, device)
            if rank == 0 and _lg:
                _lg.info(
                    "[Timing] valid_loss_forward end epoch=%d elapsed_s=%.3f",
                    epoch_1,
                    time.perf_counter() - _t_valid0,
                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                )
            v_stat = torch.tensor(
                [valid_loss_sum, float(valid_n_samples)],
                dtype=torch.double,
                device=device,
            )
            dist.all_reduce(v_stat, op=dist.ReduceOp.SUM)
            current_valid_loss = float(v_stat[0] / v_stat[1]) if v_stat[1] > 0 else 0.0
            is_best_observed_epoch = metric_improved(
                current_valid_loss,
                best_observed_metric,
                direction=checkpoint_direction,
            )
            is_best_after_min_epoch = bool(
                epoch_1 >= min_epochs
                and metric_improved(current_valid_loss, best_after_min_metric, direction=checkpoint_direction)
            )
            best_valid_for_effectiveness = float(
                min(
                    current_valid_loss,
                    best_observed_metric if best_observed_metric is not None else current_valid_loss,
                )
                if checkpoint_direction == "min"
                else max(
                    current_valid_loss,
                    best_observed_metric if best_observed_metric is not None else current_valid_loss,
                )
            )
            lr_floor_state = explain_lr_floor(
                current_lr=lr_epoch,
                base_min_lr=base_min_lr,
                scheduler_type=lr_scheduler_type,
                damping_enabled=damping_enabled_for_semantics,
                effective_min_lr=effective_min_lr,
            )
            latest_damping_event = damping_events[-1] if damping_events and int(damping_events[-1].get("epoch", -1)) == int(epoch_1) else None
            effectiveness_record = build_training_effectiveness_record(
                epoch=epoch_1,
                valid_loss=current_valid_loss,
                best_valid_loss=best_valid_for_effectiveness,
                previous_valid_loss=None if not math.isfinite(prev_valid_loss) else prev_valid_loss,
                lr_base=base_min_lr,
                lr_effective=lr_epoch,
                base_min_lr=base_min_lr,
                effective_min_lr=effective_min_lr,
                damping_event=latest_damping_event,
                checkpoint_improved=bool(is_best_observed_epoch),
                grad_finite=True,
            )
            effectiveness_record["lr_floor"] = lr_floor_state
            objective_record = detect_objective_drift(
                epoch=epoch_1,
                valid_loss=current_valid_loss,
                best_valid_loss=best_observed_metric if best_observed_metric is not None else current_valid_loss,
                previous_valid_loss=None if not math.isfinite(prev_valid_loss) else prev_valid_loss,
                component_deltas=_component_weighted_deltas(epoch_1, best_observed_epoch),
                config=objective_drift_cfg,
                training_effectiveness=effectiveness_record,
            )
            latest_objective_drift_status = str(objective_record.get("status") or "none")
            effectiveness_record["objective_drift_status"] = latest_objective_drift_status
            effectiveness_record["objective_drift_action"] = str(objective_record.get("action") or "")
            recovery_requested = bool(
                latest_objective_drift_status == str(recovery_cfg.get("trigger") or "severe_objective_drift")
                and bool(recovery_cfg.get("enabled", True))
                and int(recovery_state.get("count", 0) or 0) < int(recovery_cfg.get("max_recoveries", 1) or 1)
            )
            training_effectiveness_records.append(effectiveness_record)
            if rank == 0:
                objective_drift_records.append(objective_record)
                append_step3_objective_drift_jsonl(
                    log_file=getattr(final_cfg, "log_file", None),
                    row=objective_record,
                )

            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bleu_line = None
                lr_sched_line = None
                if sched is not None and ws_resolved is not None:
                    scheduler_state = scheduler_semantics(
                        scheduler_type=lr_scheduler_type,
                        damping_enabled=damping_enabled_for_semantics,
                        base_min_lr=base_min_lr,
                        damping_factor_cumulative=damping_factor_cumulative,
                        effective_min_lr_policy=effective_min_lr_policy,
                    )
                    lr_sched_line = (
                        f"scheduler_type={scheduler_state['scheduler_type']} "
                        f"base_scheduler={scheduler_state['base_scheduler']} "
                        f"damping_enabled={str(scheduler_state['damping_enabled']).lower()} "
                        f"initial_lr={initial_lr:.6g} current_lr={lr_epoch:.6g} "
                        f"base_min_lr={base_min_lr:.6g} effective_min_lr={effective_min_lr:.6g} "
                        f"min_lr_ratio={min_lr_ratio:.6g} damping_factor_cumulative={damping_factor_cumulative:.6g} "
                        f"warmup_steps={ws_resolved} total_steps={total_steps_plan} "
                        f"scheduler_steps_end_of_epoch={global_step} warmup_ratio={warmup_ratio_logged:.6g} "
                        f"lr_floor_explained={str(lr_floor_state['floor_explained']).lower()}"
                    )
                block = format_epoch_training_block(
                    time_str=current_time,
                    epoch=epoch + 1,
                    epoch_time_s=rec["epoch_time"],
                    total_time_s=rec["total_time"],
                    step_time_s=rec["step_time"],
                    gpu_util=rec["gpu_util"],
                    gpu_mem=rec["gpu_mem"],
                    cpu_used=rec["cpu_used"],
                    cpu_total=rec["cpu_total"],
                    cpu_util=rec["cpu_util"],
                    lr=lr_epoch,
                    train_loss=avg_loss,
                    valid_loss=current_valid_loss,
                    bleu_line=bleu_line,
                    lr_schedule_detail=lr_sched_line,
                )
                log_epoch_training_block(_lg, block)
                _epoch_time = float(rec.get("epoch_time", 0.0) or 0.0)
                _throughput_epoch = float(_ns_tot / _epoch_time) if _epoch_time > 0 else 0.0
                _peak_memory = 0
                if isinstance(_gpeak, (list, tuple)):
                    try:
                        _peak_memory = int(max(int(x or 0) for x in _gpeak))
                    except Exception:
                        _peak_memory = 0
                elif _gpeak is not None:
                    try:
                        _peak_memory = int(_gpeak)
                    except Exception:
                        _peak_memory = 0
                append_train_epoch_metrics_jsonl(
                    log_file=getattr(final_cfg, "log_file", None),
                    row={
                        "run_id": str(getattr(final_cfg, "run_id", "") or ""),
                        "task_id": int(getattr(final_cfg, "task_idx", 0) or 0),
                        "profile_id": str(getattr(final_cfg, "task_profile_id", "") or ""),
                        "epoch": int(epoch_1),
                        "global_step": int(global_step),
                        "rank": int(rank),
                        "split": "valid",
                        "loss_total": float(current_valid_loss),
                        "lr": float(lr_epoch),
                        "finite": bool(math.isfinite(float(current_valid_loss))),
                        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    },
                )
                append_step3_gpu_profile_jsonl(
                    log_file=getattr(final_cfg, "log_file", None),
                    row={
                        **_step3_gpu_profile_row(
                            final_cfg=final_cfg,
                            rank=rank,
                            device=device,
                            global_step=global_step,
                            epoch=epoch_1,
                        ),
                        "phase": "epoch",
                    },
                )
                append_step3_epoch_summary_csv(
                    log_file=getattr(final_cfg, "log_file", None),
                    row={
                        "epoch": int(epoch_1),
                        "train_loss": float(avg_loss),
                        "valid_loss": float(current_valid_loss),
                        "best_metric": float(current_valid_loss),
                        "delta_from_best": effectiveness_record["delta_from_best"],
                        "delta_recent": effectiveness_record["delta_recent"],
                        "lr_base": float(base_min_lr),
                        "lr_effective": float(lr_epoch),
                        "base_min_lr": float(base_min_lr),
                        "effective_min_lr": float(effective_min_lr),
                        "damping_event": json.dumps(latest_damping_event or {}, ensure_ascii=False, sort_keys=True),
                        "objective_drift_status": latest_objective_drift_status,
                        "loss_phase": str(phase_record.get("phase") or ""),
                        "checkpoint_improved": bool(is_best_observed_epoch),
                        "effective_improvement_status": effectiveness_record["effective_improvement_status"],
                        "recommended_action": effectiveness_record["recommended_action"],
                        "elapsed_s": float(rec.get("total_time", 0.0) or 0.0),
                        "samples_per_sec": float(_throughput_epoch),
                        "checkpoint_path": str(best_observed_path) if bool(is_best_observed_epoch) else "",
                        "status": "best_observed" if bool(is_best_observed_epoch) else ("best_after_min_epochs" if bool(is_best_after_min_epoch) else "ok"),
                    },
                )
                append_step3_training_effectiveness_jsonl(
                    log_file=getattr(final_cfg, "log_file", None),
                    row=effectiveness_record,
                )

            if current_valid_loss > prev_valid_loss:
                enduration += 1
                if lr_scheduler_type not in ("warmup_cosine", "safe_damping_v2"):
                    learning_rate /= 2.0
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate
            else:
                enduration = 0
            best_observed_replaced_previous = best_observed_epoch is not None
            best_after_min_replaced_previous = best_after_min_epoch is not None
            latest_replaced_previous = latest_event is not None or os.path.isfile(latest_checkpoint_path)
            if bool(is_best_observed_epoch):
                best_observed_metric = float(current_valid_loss)
                best_observed_epoch = int(epoch_1)
            if bool(is_best_after_min_epoch):
                best_after_min_metric = float(current_valid_loss)
                best_after_min_epoch = int(epoch_1)
            if str(checkpoint_metric).strip().lower() in ("valid_loss", "loss") and rank == 0:
                latest_event = _save_step3_checkpoint(
                    latest_checkpoint_path,
                    epoch=epoch_1,
                    metric=current_valid_loss,
                    scope="latest",
                    reason="latest_epoch_snapshot",
                    replaced_previous=latest_replaced_previous,
                )
                reverse_topk = checkpoint_direction == "max"
                sorted_topk = sorted(topk_candidates, key=lambda item: item[0], reverse=reverse_topk)
                topk_has_room = len(sorted_topk) < checkpoint_top_k
                worst_topk_metric = None if topk_has_room or not sorted_topk else float(sorted_topk[-1][0])
                topk_should_save = bool(topk_has_room or metric_improved(current_valid_loss, worst_topk_metric, direction=checkpoint_direction))
                if topk_should_save:
                    topk_path = os.path.join(
                        topk_dir,
                        checkpoint_filename_for_metric(epoch_1, checkpoint_metric_name, current_valid_loss),
                    )
                    topk_replaced_previous = not topk_has_room
                    topk_event = _save_step3_checkpoint(
                        topk_path,
                        epoch=epoch_1,
                        metric=current_valid_loss,
                        scope="topk",
                        reason="topk_replaced" if topk_replaced_previous else "topk_retained",
                        replaced_previous=topk_replaced_previous,
                    )
                    topk_candidates.append((float(current_valid_loss), int(epoch_1), topk_event))
                    sorted_topk = sorted(topk_candidates, key=lambda item: item[0], reverse=reverse_topk)
                    evicted_topk = sorted_topk[checkpoint_top_k:]
                    topk_candidates = sorted_topk[:checkpoint_top_k]
                    for _metric, _epoch, evicted_event in evicted_topk:
                        evicted_path = str(evicted_event.get("path") or "")
                        if evicted_path:
                            try:
                                os.remove(evicted_path)
                            except FileNotFoundError:
                                pass
                            except OSError:
                                pass
                            try:
                                os.remove(evicted_path + ".lineage.json")
                            except FileNotFoundError:
                                pass
                            except OSError:
                                pass
                if bool(per_epoch_policy.get("enabled", False)) and epoch_1 % max(1, int(per_epoch_policy.get("keep_interval", 1) or 1)) == 0:
                    _save_step3_checkpoint(
                        os.path.join(per_epoch_dir, f"epoch_{epoch_1:03d}.pth"),
                        epoch=epoch_1,
                        metric=current_valid_loss,
                        scope="per_epoch",
                        reason="configured_per_epoch_retention",
                        replaced_previous=False,
                    )
                if bool(is_best_observed_epoch):
                    best_version += 1
                    best_observed_event = _save_step3_checkpoint(
                        best_observed_path,
                        epoch=epoch_1,
                        metric=current_valid_loss,
                        scope="best_observed",
                        reason="global_best_improved",
                        replaced_previous=best_observed_replaced_previous,
                    )
                    if bool(checkpoint_policy_cfg.get("keep_best_pth_alias", True)):
                        best_alias_replaced_previous = os.path.isfile(str(final_cfg.save_file))
                        shutil.copy2(best_observed_path, str(final_cfg.save_file))
                        alias_lineage = _build_step3_checkpoint_lineage(
                            final_cfg,
                            checkpoint_path=str(final_cfg.save_file),
                            checkpoint_context=_checkpoint_context(
                                epoch=epoch_1,
                                metric=current_valid_loss,
                                scope="best_observed",
                                reason="best_observed_alias_for_legacy_best_pth",
                                replaced_previous=best_alias_replaced_previous,
                            ),
                        )
                        alias_lineage["best_pth_alias_of"] = str(best_observed_path)
                        alias_lineage["reason"] = "best_observed_alias_for_legacy_best_pth"
                        alias_lineage["replaced_previous"] = bool(best_alias_replaced_previous)
                        alias_lineage_path = write_checkpoint_lineage(str(final_cfg.save_file), alias_lineage)
                        if best_observed_event is not None:
                            best_observed_event["best_pth_alias"] = str(final_cfg.save_file)
                            best_observed_event["best_pth_alias_lineage_path"] = str(alias_lineage_path)
                if bool(is_best_after_min_epoch):
                    best_after_min_event = _save_step3_checkpoint(
                        best_after_min_epochs_path,
                        epoch=epoch_1,
                        metric=current_valid_loss,
                        scope="best_after_min_epochs",
                        reason="after_min_epochs_best_improved",
                        replaced_previous=best_after_min_replaced_previous,
                    )
                atomic_write_json(
                    os.path.join(state_dir, "best_event.json"),
                    build_best_event_payload(
                        best_observed_event=best_observed_event,
                        best_after_min_epochs_event=best_after_min_event,
                        latest_event=latest_event,
                        topk_events=[item[2] for item in topk_candidates],
                    ),
                )
            if recovery_requested:
                _activate_recovery_from_best(epoch_1, objective_record)
            elif bool(recovery_state.get("active", False)):
                recovery_state["remaining_epochs"] = max(0, int(recovery_state.get("remaining_epochs", 0) or 0) - 1)
                if int(recovery_state["remaining_epochs"]) <= 0:
                    recovery_state["active"] = False
            if not recovery_requested:
                _apply_validation_aware_lr_damping(epoch_1, current_valid_loss, emit=bool(rank == 0))
            prev_valid_loss = current_valid_loss

            if odcr_ddp_epoch_end_barrier():
                ddp_heartbeat(_lg, "before_epoch_end_barrier", rank=rank, epoch=epoch_1)
                dist.barrier()
                ddp_heartbeat(_lg, "after_epoch_end_barrier", rank=rank, epoch=epoch_1)

            if epoch + 1 >= min_epochs and enduration >= early_stop_patience:
                break
    finally:
        if rank == 0 and perf is not None:
            perf.finish()
        if save_final_checkpoint and rank == 0:
            _sf = os.path.abspath(os.path.expanduser(str(latest_checkpoint_path)))
            _final_sd = state_dict_for_canonical_best_pth(
                ema_enabled=ema_enabled,
                ema_model=ema_model,
                ddp_module=model,
                underlying_model_fn=get_underlying_model,
            )
            odcr_save_checkpoint(
                _final_sd,
                _sf,
                epoch=int(epochs),
                reason="save_final_checkpoint",
                logger=_lg,
                is_last=True,
            )
            _final_lineage = _build_step3_checkpoint_lineage(
                final_cfg,
                checkpoint_path=_sf,
                checkpoint_context=_checkpoint_context(
                    epoch=int(epochs),
                    metric=float(best_observed_metric if best_observed_metric is not None else 0.0),
                    scope="latest",
                    reason="latest_epoch_snapshot",
                    replaced_previous=os.path.isfile(_sf),
                ),
            )
            write_checkpoint_lineage(_sf, _final_lineage)
        if rank == 0:
            try:
                effectiveness_summary = {
                    "schema_version": STEP3_TRAINING_EFFECTIVENESS_SCHEMA_VERSION,
                    "records": training_effectiveness_records,
                    "latest": training_effectiveness_records[-1] if training_effectiveness_records else {},
                    "status_counts": {
                        str(status): sum(
                            1
                            for row in training_effectiveness_records
                            if str(row.get("effective_improvement_status") or "") == str(status)
                        )
                        for status in sorted(
                            {str(row.get("effective_improvement_status") or "") for row in training_effectiveness_records}
                        )
                    },
                    "scheduler": scheduler_semantics(
                        scheduler_type=lr_scheduler_type,
                        damping_enabled=damping_enabled_for_semantics,
                        base_min_lr=base_min_lr,
                        damping_factor_cumulative=damping_factor_cumulative,
                        effective_min_lr_policy=effective_min_lr_policy,
                    ),
                    "damping_events_count": len(damping_events),
                    "objective_drift": {
                        "schema_version": "odcr_step3_objective_drift/1",
                        "records": objective_drift_records,
                        "latest": objective_drift_records[-1] if objective_drift_records else {},
                        "status_counts": {
                            str(status): sum(
                                1 for row in objective_drift_records if str(row.get("status") or "") == str(status)
                            )
                            for status in sorted({str(row.get("status") or "") for row in objective_drift_records})
                        },
                    },
                    "recovery": {
                        "config": recovery_cfg,
                        "events": recovery_events,
                        "count": int(recovery_state.get("count", 0) or 0),
                    },
                    "phase_history": phase_history,
                    "conflict_aware": conflict_aware_cfg,
                    "adapter_gating": adapter_gating_cfg,
                }
                write_step3_training_effectiveness_summary_json(
                    log_file=getattr(final_cfg, "log_file", None),
                    payload=effectiveness_summary,
                )
                loss_dashboard = summarize_loss_component_rows(loss_component_records)
                write_step3_loss_component_epoch_summary_csv(
                    log_file=getattr(final_cfg, "log_file", None),
                    rows=list(loss_dashboard.get("epoch_rows") or []),
                )
                write_step3_loss_component_trends_json(
                    log_file=getattr(final_cfg, "log_file", None),
                    payload={
                        "schema_version": loss_dashboard.get("schema_version"),
                        "component_trends": loss_dashboard.get("component_trends") or {},
                    },
                )
                trend_count = len(loss_dashboard.get("component_trends") or {})
                write_step3_component_contribution_summary_md(
                    log_file=getattr(final_cfg, "log_file", None),
                    text=(
                        "# Step3 Loss Component Dashboard\n\n"
                        f"- schema_version: {loss_dashboard.get('schema_version')}\n"
                        f"- component_count: {trend_count}\n"
                        f"- record_count: {len(loss_component_records)}\n"
                        "- purpose: explain post-epoch plateau, saturation, and loss rebalance signals.\n"
                    ),
                )
            except Exception as exc:
                if _lg:
                    _lg.warning(
                        "[Step3][Effectiveness] failed to write effectiveness/loss dashboard sidecars: %s",
                        exc,
                        extra=log_route_extra(_lg, ROUTE_SUMMARY),
                    )
            try:
                audit = build_step3_quality_audit(run_dir, thresholds=quality_gate_cfg)
                audit["quality_gate_inputs"] = dict(quality_gate_cfg)
                audit["runtime_evidence"] = {
                    "code_present": True,
                    "active_path": True,
                    "runtime_verified": False,
                    "formal_verified": False,
                    "note": "Generated by Step3 train loop; runtime/formal pass requires controlled validation/full-run artifacts.",
                }
                write_step3_quality_audit(run_dir, audit)
            except Exception as exc:
                if _lg:
                    _lg.warning(
                        "[Step3][Quality] failed to write quality_audit sidecar: %s",
                        exc,
                        extra=log_route_extra(_lg, ROUTE_SUMMARY),
                    )


def _eval_collect_shard_predictions(
    model,
    test_dataloader,
    device,
    *,
    protocol_spec: Mapping[str, Any],
    task_idx: int,
    split: str,
    rank: int,
    source_domain: str,
    target_domain: str,
    dataset_name: str,
):
    _model = get_underlying_model(model)
    model = model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    compute_text_metrics = bool(protocol_spec.get("compute_text_metrics", False))
    tok = get_odcr_text_tokenizer()
    start_time = time.perf_counter()
    with torch.inference_mode():
        for batch in test_dataloader:
            g = require_gathered_batch(_model.gather(batch, device))
            user_idx, item_idx, rating, tgt_output, domain_idx, sample_id_tensor = (
                g.user_idx,
                g.item_idx,
                g.rating,
                g.tgt_output,
                g.domain_idx,
                g.sample_id,
            )
            ca = g.content_anchor_score
            sa = g.style_anchor_score
            ce = g.content_evidence_ids
            se = g.style_evidence_ids
            dsa = g.domain_style_anchor_ids
            lsh = g.local_style_hint_ids
            pol = g.polarity_ids
            eq = g.evidence_quality_prior
            if any(value is None for value in (ca, sa, ce, se, dsa, lsh, pol, eq)):
                raise RuntimeError("_eval_collect_shard_predictions 缺少 Step3 canonical evidence 张量。")
            with odcr_cuda_bf16_autocast():
                pred_ratings = _model.recommend(
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
                if compute_text_metrics:
                    pred_exps, _ = _model.generate(
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
                else:
                    pred_exps = None
            pred_texts = tok.batch_decode(pred_exps, skip_special_tokens=True) if pred_exps is not None else [""] * int(rating.size(0))
            ref_texts = tok.batch_decode(tgt_output, skip_special_tokens=True) if compute_text_metrics else [""] * int(rating.size(0))
            source_rows = sample_id_tensor.detach().cpu().tolist()
            users = user_idx.detach().cpu().tolist()
            items = item_idx.detach().cpu().tolist()
            domains = domain_idx.detach().cpu().tolist()
            pred_rating_values = pred_ratings.detach().cpu().tolist()
            gold_rating_values = rating.detach().cpu().tolist()
            for i, source_row_index in enumerate(source_rows):
                domain_name = "target" if int(domains[i]) == 1 else "auxiliary"
                sample_id = stable_step3_sample_id(
                    dataset_name=f"task{int(task_idx)}:{dataset_name}",
                    split=split,
                    source_row_index=int(source_row_index),
                    user_id=int(users[i]),
                    item_id=int(items[i]),
                )
                rows.append(
                    {
                        "schema_version": STEP3_PREDICTION_SHARD_SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "row_id": int(source_row_index),
                        "split": str(split),
                        "domain": domain_name,
                        "user_id": int(users[i]),
                        "item_id": int(items[i]),
                        "rating_gold": float(gold_rating_values[i]),
                        "rating_pred": float(pred_rating_values[i]),
                        "pred_text": str(pred_texts[i]),
                        "ref_text": str(ref_texts[i]),
                        "decode_status": "decoded" if compute_text_metrics else "not_requested",
                        "source_row_index": int(source_row_index),
                        "rank": int(rank),
                        "source_domain": str(source_domain),
                        "target_domain": str(target_domain),
                        "protocol": str(protocol_spec.get("protocol")),
                    }
                )
    elapsed = max(time.perf_counter() - start_time, 1.0e-12)
    return rows, {"elapsed_s": elapsed, "rows": len(rows), "rows_per_second": float(len(rows) / elapsed)}


def metrics_from_eval_lists(prediction_ratings, ground_truth_ratings, prediction_exps, reference_exps):
    prediction_ratings = np.array(prediction_ratings)
    ground_truth_ratings = np.array(ground_truth_ratings)
    rating_diffs = prediction_ratings - ground_truth_ratings
    mae = round(np.mean(np.abs(rating_diffs)), 4)
    rmse = round(np.sqrt(np.mean(np.square(rating_diffs))), 4)
    text_results = evaluate_text(prediction_exps, reference_exps)
    return {"recommendation": {"mae": mae, "rmse": rmse}, "explanation": text_results}


def step3_target_only_diagnostic_protocol() -> dict[str, Any]:
    """Declare the Step3 target-only diagnostic protocol without running Step4/5/eval/rerank."""
    return {
        "evaluator_protocol": "code1_target_only_comparable",
        "diagnostic_only": True,
        "not_final_paper_metric": True,
        "target_only": True,
        "stage_boundary": "Step3 internal valid/test reader only",
        "does_not_start": ["step4", "step5", "eval", "rerank"],
    }


def evalModel(model, test_dataloader, device):
    rows, _stats = _eval_collect_shard_predictions(
        model,
        test_dataloader,
        device,
        protocol_spec=step3_eval_protocol_spec(ODCR_STEP3_DIAGNOSTIC),
        task_idx=0,
        split="valid",
        rank=0,
        source_domain="",
        target_domain="",
        dataset_name="legacy_eval_model",
    )
    pr = [row["rating_pred"] for row in rows]
    gt = [row["rating_gold"] for row in rows]
    pe = [row["pred_text"] for row in rows]
    re = [row["ref_text"] for row in rows]
    return metrics_from_eval_lists(pr, gt, pe, re)


def _eval_artifact_root(log_file: str, *, protocol: str, split: str) -> str:
    meta_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(log_file))))
    path = os.path.join(meta_dir, f"eval_{protocol}_{split}")
    os.makedirs(path, exist_ok=True)
    return path


def _write_json_file(path: str, payload: Mapping[str, Any]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl_file(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _read_prediction_shards(shard_dir: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(shard_dir).glob("rank_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _write_eval_status_sidecar(log_file: str, payload: Mapping[str, Any]) -> None:
    meta_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(log_file))))
    _write_json_file(os.path.join(meta_dir, "step3_eval_status.json"), payload)


def build_config_and_dataloader(args, ddp_rank: int, ddp_world_size: int, local_rank: int):
    primary_device = local_rank

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    repo_root = os.path.abspath(os.environ.get("ODCR_ROOT") or os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    upstream_evidence = validate_step3_preprocess_upstream_gate(
        repo_root=repo_root,
        task_id=int(task_idx),
        auxiliary_domain=args.auxiliary,
        target_domain=args.target,
        data_dir=get_data_dir(),
        merged_dir=get_merged_data_dir(),
        runs_dir=os.environ.get("ODCR_RESOLVED_RUNS_DIR") or os.path.join(repo_root, "runs"),
        embed_dim=int(get_odcr_embed_dim()),
    )
    path = os.path.join(get_merged_data_dir(), str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    nuser = train_df['user_idx'].max() + 1
    nitem = train_df['item_idx'].max() + 1

    eval_protocol = normalize_eval_protocol(getattr(args, "eval_protocol", None) or MINIMAL_EVAL)
    eval_split = str(getattr(args, "eval_split", None) or "valid").strip().lower()
    protocol_spec = step3_eval_protocol_spec(eval_protocol, split=eval_split)
    if bool(protocol_spec.get("interface_only", False)):
        raise RuntimeError("full_pipeline_final_eval is an interface contract only in Step3; run Step4/Step5 final eval later.")

    batch_size = args.batch_size if args.batch_size is not None else get_eval_batch_size()
    resolved_nproc = get_num_proc()
    if args.num_proc is not None and int(args.num_proc) != int(resolved_nproc):
        raise RuntimeError(
            f"step3 eval child argparse conflict: --num-proc={args.num_proc} conflicts with "
            f"ODCR_HARDWARE_PROFILE_JSON.num_proc={resolved_nproc}. "
            "Only public ./odcr --set may alter configs/odcr.yaml; torchrun children must use the resolved hardware payload."
        )
    nproc = resolved_nproc

    if batch_size % ddp_world_size != 0:
        raise ValueError(
            f"eval_batch_size={batch_size} 与 world_size={ddp_world_size} 不整除，DDP 评测非法。"
            "请到 configs/odcr.yaml 修改 eval.profiles.*.eval_batch_size，或调整 hardware.profiles.*.ddp_world_size。"
        )
    loader_batch_size = batch_size // ddp_world_size

    config = {
        "task_idx": task_idx,
        "device": primary_device if torch.cuda.is_available() else args.device,
        "log_file": args.log_file,
        "save_file": args.save_file
        or os.path.join(get_stage_run_dir(task_idx), "model", "best.pth"),
        "batch_size": loader_batch_size,
        "emsize": int(get_odcr_embed_dim()),
        "nlayers": args.nlayers,
        "nhid": 2048,
        "ntoken": len(get_odcr_text_tokenizer()),
        "dropout": 0.2,
        "nuser": nuser,
        "nitem": nitem,
        "nhead": 2
    }
    config["batch_size_global"] = batch_size
    config["eval_protocol"] = eval_protocol
    config["eval_split"] = eval_split
    config["eval_protocol_spec"] = protocol_spec

    if eval_protocol == ODCR_STEP3_DIAGNOSTIC:
        valid_path = os.path.join(path, f"aug_{eval_split}.csv")
        dataset_name = f"merged_auxiliary_target:{args.auxiliary}->{args.target}"
    else:
        valid_path = os.path.join(get_data_dir(), args.target, f"{eval_split}.csv")
        dataset_name = f"target_only:{args.target}"
    valid_df = pd.read_csv(valid_path)
    if eval_protocol != ODCR_STEP3_DIAGNOSTIC:
        valid_df["domain"] = "target"
    _require_step3_canonical_columns(valid_df, csv_path=valid_path, split=f"eval-{eval_split}-{eval_protocol}")
    valid_df['item'] = valid_df['item'].astype(str)
    valid_df = valid_df.reset_index(drop=True)
    valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
    datasets = DatasetDict({
        "valid": Dataset.from_pandas(valid_df)
    })
    eval_tokenizer_length = int(os.environ.get("ODCR_STEP3_TOKENIZER_MAX_LENGTH", "0") or "0")
    eval_evidence_length = int(os.environ.get("ODCR_STEP3_EVIDENCE_MAX_LENGTH", "0") or "0")
    if eval_tokenizer_length <= 0 or eval_evidence_length <= 0:
        payload_raw = (os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()
        payload = json.loads(payload_raw) if payload_raw else {}
        row = payload.get("training_row") if isinstance(payload, Mapping) else {}
        if isinstance(row, Mapping):
            eval_tokenizer_length = int(row.get("tokenizer_max_length") or eval_tokenizer_length or 0)
            eval_evidence_length = int(row.get("evidence_max_length") or eval_evidence_length or 0)
    if protocol_spec.get("max_ref_len"):
        eval_tokenizer_length = int(protocol_spec["max_ref_len"])
    processor = Processor(
        args.auxiliary,
        args.target,
        max_length=eval_tokenizer_length,
        evidence_length=eval_evidence_length,
        length_protocol=eval_protocol,
    )
    cache_dir, cache_fp, cache_fp_payload = _build_step3_eval_cache_dir(
        task_idx=int(task_idx),
        eval_data_path=os.path.abspath(valid_path),
        processor=processor,
        tok=get_odcr_text_tokenizer(),
        source_domain=args.auxiliary,
        target_domain=args.target,
        split_row_counts={f"{eval_protocol}:{name}": len(datasets[name]) for name in datasets.keys()},
        upstream_evidence=upstream_evidence,
    )
    if ddp_rank == 0:
        _log_tokenize_cache_line(
            f"[Tokenize] eval valid cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
            getattr(args, "log_file", None),
        )
    encoded_data = _map_tokenize_train_valid_to_hf_cache(
        datasets=datasets,
        processor=processor,
        nproc=nproc,
        cache_dir=cache_dir,
        cache_fingerprint=cache_fp,
        cache_fingerprint_payload=cache_fp_payload,
        rank=ddp_rank,
        show_datasets_progress=(ddp_rank == 0),
        log_tokenize=(ddp_rank == 0),
        phase="eval valid",
        log_file=getattr(args, "log_file", None),
    )
    encoded_data.set_format("torch")
    valid_dataset = TensorDataset(
        encoded_data["valid"]["user_idx"],
        encoded_data["valid"]["item_idx"],
        encoded_data["valid"]["rating"],
        encoded_data["valid"]["explanation_idx"],
        encoded_data["valid"]["domain_idx"],
        encoded_data["valid"]["sample_id"],
        encoded_data["valid"]["content_anchor_score"],
        encoded_data["valid"]["style_anchor_score"],
        encoded_data["valid"]["content_evidence_ids"],
        encoded_data["valid"]["style_evidence_ids"],
        encoded_data["valid"]["domain_style_anchor_ids"],
        encoded_data["valid"]["local_style_hint_ids"],
        encoded_data["valid"]["polarity_ids"],
        encoded_data["valid"]["evidence_quality_prior"],
    )
    n_samples = len(valid_dataset)
    shard_idx = list(range(ddp_rank, n_samples, ddp_world_size))
    valid_dataset = Subset(valid_dataset, shard_idx)
    config["eval_dataset_path"] = os.path.abspath(valid_path)
    config["eval_dataset_name"] = dataset_name
    config["eval_expected_sample_count"] = int(n_samples)
    config["eval_expected_sample_ids"] = [
        stable_step3_sample_id(
            dataset_name=f"task{int(task_idx)}:{dataset_name}",
            split=eval_split,
            source_row_index=int(i),
            user_id=int(valid_df.iloc[int(i)]["user_idx"]),
            item_id=int(valid_df.iloc[int(i)]["item_idx"]),
        )
        for i in range(len(valid_df))
    ]
    eval_world_size = max(ddp_world_size, 1)
    dl_valid = get_dataloader_num_workers("valid")
    num_workers = min(max(1, dl_valid // eval_world_size), 8)
    _hw_eval = json.loads(os.environ.get("ODCR_HARDWARE_PROFILE_JSON", "{}") or "{}")
    pin_memory = bool(_hw_eval.get("pin_memory", True))
    persistent_workers = bool(_hw_eval.get("persistent_workers", True))
    _pf_ev = get_dataloader_prefetch_factor(num_workers, split="valid")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=_pf_ev,
    )

    dc, ds, uc, us, ic, ist, profile_meta = load_profile_tensors_dual_first(
        data_root=get_data_dir(),
        auxiliary_domain=args.auxiliary,
        target_domain=args.target,
        device_idx="cpu",
        step3_upstream_preflight_summary=upstream_evidence,
    )
    if ddp_rank == 0:
        print(
            f"[Step3 eval] ODCR dual-channel profiles loaded ({profile_meta.get('profile_mode')}).",
            flush=True,
        )

    _em_ev = int(uc.shape[-1])
    if _em_ev != int(get_odcr_embed_dim()):
        raise ValueError(
            f"Step3 eval 加载的 profile 隐层维度={_em_ev} 与 ODCR_EMBED_DIM={get_odcr_embed_dim()} 不一致。"
        )
    if int(config.get("emsize", -1)) != _em_ev:
        config = {**config, "emsize": _em_ev}

    model = Model(
        config.get("nuser"), config.get("nitem"), config.get("ntoken"),
        config.get("emsize"), config.get("nhead"), config.get("nhid"),
        config.get("nlayers"), config.get("dropout"),
        uc, us, ic, ist, dc, ds,
    )
    _map = f"cuda:{config.get('device')}" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(config.get("save_file"), map_location=_map, weights_only=True))
    model = model.to(config.get("device"))
    decode_len = int(protocol_spec.get("max_decode_len") or eval_tokenizer_length)
    model.max_explanation_length = decode_len
    model.hard_max_len = decode_len
    try:
        model.decoder_eos_id = int(get_odcr_text_tokenizer().eos_token_id)
    except Exception:
        pass

    return config, valid_dataloader, model


def _run_train_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    ddp_fast_backends = apply_ddp_fast_torch_backends()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    if not getattr(args, "log_file", None):
        raise RuntimeError("Step3 DDP train requires parent-provided --log_file under runs/.../meta/full.log.")
    log_path = str(args.log_file)
    run_id = Path(log_path).expanduser().resolve().parents[1].name
    args.log_file = log_path

    _setup = setup_train_logging(
        log_file=log_path,
        task_idx=task_idx,
        rank=rank,
        world_size=world_size,
        run_id=run_id,
    )
    train_logger = _setup["logger"]

    try:
        base_final, train_dataloader, valid_dataloader, model, sampler = build_config_and_data_ddp(
            args, rank, world_size, local_rank,
        )
        final_cfg = replace(
            base_final,
            run_id=run_id,
            logger=train_logger,
            log_file=log_path,
            ddp_fast_backends=ddp_fast_backends,
        )
        if rank == 0:
            _meta = {
                "run_id": run_id,
                "task_idx": task_idx,
                "rank": rank,
                "world_size": world_size,
                "cuda_available": bool(torch.cuda.is_available()),
                "local_rank": local_rank,
                "learning_rate": final_cfg.learning_rate,
                "batch_size": final_cfg.train_batch_size,
                "batch_size_global": final_cfg.batch_size_global,
                "per_gpu_batch_size": final_cfg.per_device_train_batch_size,
                "batch_semantics_version": "odcr_no_accum/1",
                "grad_accum_removed": True,
                "effective_global_batch_size": final_cfg.effective_global_batch_size,
                "epochs": final_cfg.epochs,
                "save_file": os.path.abspath(str(final_cfg.save_file)),
                "log_file": os.path.abspath(log_path),
                "auxiliary": args.auxiliary,
                "target": args.target,
                "distributed_env": collect_distributed_env_for_meta(),
            }
            log_run_header(train_logger, _meta)
            _cfg_snap = dict(final_cfg.to_log_dict())
            _cfg_snap["training_diagnostics"] = training_diagnostics_snapshot(
                diagnostics_scope="child",
                effective_training_payload_json=os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""),
                ddp_find_unused_parameters_effective=bool(final_cfg.ddp_find_unused_parameters),
            )
            _cfg_snap["training_semantic_fingerprint"] = (
                (os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or None
            )
            _cfg_snap["generation_semantic_fingerprint"] = (
                (os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or None
            )
            _cfg_snap["runtime_diagnostics_fingerprint"] = (
                (os.environ.get("ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip() or None
            )
            _cfg_snap["runtime_env"] = runtime_env_dict_for_config_resolved()
            log_config_snapshot(train_logger, _cfg_snap)
            train_logger.info(
                "[Fingerprints] training_semantic=%s generation_semantic=%s runtime_diag=%s",
                (os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or "n/a",
                (os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or "n/a",
                (os.environ.get("ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip() or "n/a",
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            flush_preset_load_events(train_logger)
            _cfg_runtime_path = write_training_runtime_config_artifact(os.path.dirname(log_path), _cfg_snap)
            train_logger.info(
                "[Training runtime config] wrote %s",
                _cfg_runtime_path,
                extra=log_route_extra(train_logger, ROUTE_DETAIL),
            )

        try:
            trainModel_ddp(
                model,
                train_dataloader,
                valid_dataloader,
                sampler,
                final_cfg,
                rank,
                world_size,
                max_steps=getattr(args, "max_steps", None),
                save_final_checkpoint=bool(getattr(args, "save_final_checkpoint", False)),
            )
        except Exception as exc:
            if rank == 0:
                log_training_crash(train_logger, exc)
            raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _eval_torchrun_env() -> bool:
    return (
        "LOCAL_RANK" in os.environ
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )


def _write_eval_results_log(
    config,
    final,
    *,
    task_description: Optional[str] = None,
    pipeline: str = "Step3_structured_eval",
    domain_from: str = "",
    domain_to: str = "",
    start_time: Optional[str] = None,
    eval_elapsed: Optional[float] = None,
):
    lines = format_final_results_lines(final, task_description=task_description, start_time=start_time)
    if eval_elapsed is not None:
        _m, _s = divmod(int(eval_elapsed), 60)
        lines.append(f"Eval elapsed: {_m}m {_s}s ({eval_elapsed:.1f}s)")
    log_path = config.get("log_file")
    lg = config.get("logger")
    log_final_results_block(lg, lines)
    finalize_run_log(lg)
    append_eval_run_summaries(
        final,
        task_idx=int(config.get("task_idx") or 0),
        run_id=str(config.get("run_id") or ""),
        pipeline=pipeline,
        domain_from=domain_from,
        domain_to=domain_to,
        log_file=log_path if isinstance(log_path, str) else None,
        save_file=config.get("save_file"),
        task_description=task_description,
        start_time=start_time,
        eval_elapsed=eval_elapsed,
    )
    if lg is not None:
        lg.info("(eval 指标已写入 %s)", os.path.abspath(log_path) if log_path else log_path)
    else:
        logging.info("(eval 指标已写入 %s)", os.path.abspath(log_path) if log_path else log_path)


def _run_eval_ddp(args):
    if not torch.cuda.is_available():
        raise RuntimeError("step3 runner eval 的 torchrun DDP 需要 CUDA + NCCL。")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    if not getattr(args, "log_file", None):
        raise RuntimeError("Step3 DDP eval requires parent-provided --log_file under runs/.../meta/full.log.")
    log_path = str(args.log_file)
    run_id = Path(log_path).expanduser().resolve().parents[1].name
    args.log_file = log_path

    _setup = setup_train_logging(
        log_file=log_path,
        task_idx=task_idx,
        rank=rank,
        world_size=world_size,
        run_id=run_id,
    )
    ev_logger = _setup["logger"]

    config: dict[str, Any] | None = None
    protocol = normalize_eval_protocol(getattr(args, "eval_protocol", None) or MINIMAL_EVAL)
    split = str(getattr(args, "eval_split", None) or "valid").strip().lower()
    protocol_spec = step3_eval_protocol_spec(protocol, split=split)
    artifact_root = _eval_artifact_root(log_path, protocol=protocol, split=split)
    shard_dir = os.path.join(artifact_root, "prediction_shards")
    os.makedirs(shard_dir, exist_ok=True)
    try:
        config, valid_dataloader, model = build_config_and_dataloader(
            args, ddp_rank=rank, ddp_world_size=world_size, local_rank=local_rank,
        )
        config["logger"] = ev_logger
        config["run_id"] = run_id
        protocol_spec = dict(config.get("eval_protocol_spec") or protocol_spec)
        init_step3_ddp_after_cache_ready(local_rank=local_rank, rank=rank, world_size=world_size)
        if rank == 0:
            log_run_header(
                ev_logger,
                {
                    "run_id": run_id,
                    "task_idx": task_idx,
                    "rank": rank,
                    "world_size": world_size,
                    "mode": "eval_ddp_gpu_inference_phase",
                    "cuda_available": True,
                    "local_rank": local_rank,
                    "batch_size": config.get("batch_size_global", config.get("batch_size")),
                    "save_file": os.path.abspath(str(config.get("save_file", ""))),
                    "log_file": os.path.abspath(log_path),
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                    "eval_protocol": protocol_spec,
                    "prediction_shards": os.path.abspath(shard_dir),
                },
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(local_rank)
        _eval_t0 = time.time()
        _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        shard_rows, shard_stats = _eval_collect_shard_predictions(
            model,
            valid_dataloader,
            local_rank,
            protocol_spec=protocol_spec,
            task_idx=int(task_idx),
            split=split,
            rank=rank,
            source_domain=args.auxiliary,
            target_domain=args.target,
            dataset_name=str(config.get("eval_dataset_name") or f"task{task_idx}"),
        )
        shard_path = os.path.join(shard_dir, f"rank_{rank:02d}.jsonl")
        _write_jsonl_file(shard_path, shard_rows)
        shard_meta = {
            "schema_version": STEP3_PREDICTION_SHARD_SCHEMA_VERSION,
            "rank": int(rank),
            "world_size": int(world_size),
            "rows": len(shard_rows),
            "path": os.path.abspath(shard_path),
            "eval_batch_global": int(config.get("batch_size_global") or 0),
            "eval_batch_per_rank": int(config.get("batch_size") or 0),
            "eval_steps_per_rank": len(valid_dataloader),
            "gpu_memory_peak": int(torch.cuda.max_memory_allocated(local_rank)) if torch.cuda.is_available() else 0,
            "throughput": float(shard_stats.get("rows_per_second") or 0.0),
        }
        _write_json_file(os.path.join(shard_dir, f"rank_{rank:02d}.meta.json"), shard_meta)

        dist.barrier()
        dist.destroy_process_group()

        if rank == 0:
            all_rows = sort_prediction_rows(_read_prediction_shards(shard_dir))
            integrity = sample_integrity_report(
                all_rows,
                expected_count=int(config.get("eval_expected_sample_count") or 0),
                expected_sample_ids=config.get("eval_expected_sample_ids") or None,
            )
            _write_json_file(os.path.join(artifact_root, "sample_integrity_report.json"), integrity)
            _write_json_file(os.path.join(artifact_root, "eval_protocol.json"), protocol_spec)
            if integrity.get("status") != "PASS":
                status_payload = {
                    "train_status": "completed",
                    "eval_status": "failed",
                    "quality_status": "not_evaluated",
                    "downstream_ready": False,
                    "failure_phase": "post_train_eval",
                    "eval_protocol": protocol,
                    "eval_scope": split,
                    "paper_comparable": bool(protocol_spec.get("paper_comparable", False)),
                    "selected_checkpoint": os.path.abspath(str(config.get("save_file", ""))),
                    "selected_checkpoint_scope": "eval_only_checkpoint",
                    "error": "prediction shard sample integrity failed",
                    "artifact_root": os.path.abspath(artifact_root),
                }
                _write_eval_status_sidecar(log_path, status_payload)
                raise RuntimeError("Step3 eval prediction shard sample integrity failed; see sample_integrity_report.json")

            final = metrics_from_prediction_rows(
                all_rows,
                compute_text_metrics=bool(protocol_spec.get("compute_text_metrics", False)),
                text_metric_fn=evaluate_text,
            )
            final["protocol"] = protocol
            final["split"] = split
            final["paper_comparable"] = bool(protocol_spec.get("paper_comparable", False))
            final["target_only"] = bool(protocol_spec.get("target_only", False))
            final["bertscore_enabled"] = False
            _write_jsonl_file(os.path.join(artifact_root, "samples.jsonl"), all_rows)
            if bool(protocol_spec.get("compute_text_metrics", False)):
                collapse_payload = collapse_stats_from_predictions(
                    [str(row.get("pred_text") or "") for row in all_rows],
                    [str(row.get("ref_text") or "") for row in all_rows],
                )
                collapse_payload.update(
                    {
                        "protocol_schema_version": STEP3_EVAL_PROTOCOL_SCHEMA_VERSION,
                        "evaluator_protocol": protocol,
                        "diagnostic_only": bool(protocol_spec.get("diagnostic_only", False)),
                        "not_paper_comparable": bool(protocol_spec.get("not_paper_comparable", False)),
                        "paper_comparable": bool(protocol_spec.get("paper_comparable", False)),
                        "bertscore_enabled": False,
                    }
                )
                _write_json_file(os.path.join(artifact_root, "collapse_stats.json"), collapse_payload)
            _eval_elapsed = time.time() - _eval_t0
            shard_metas = []
            for path_obj in sorted(Path(shard_dir).glob("rank_*.meta.json")):
                shard_metas.append(json.loads(path_obj.read_text(encoding="utf-8")))
            summary = {
                "schema_version": STEP3_EVAL_PROTOCOL_SCHEMA_VERSION,
                "eval_status": "completed",
                "eval_protocol": protocol,
                "split": split,
                "paper_comparable": bool(protocol_spec.get("paper_comparable", False)),
                "target_only": bool(protocol_spec.get("target_only", False)),
                "bertscore_enabled": False,
                "max_ref_len": protocol_spec.get("max_ref_len"),
                "max_decode_len": protocol_spec.get("max_decode_len"),
                "sample_count": len(all_rows),
                "expected_sample_count": int(config.get("eval_expected_sample_count") or 0),
                "metrics": final,
                "sample_integrity": integrity,
                "artifact_root": os.path.abspath(artifact_root),
                "prediction_shards": os.path.abspath(shard_dir),
                "eval_batch_global": int(config.get("batch_size_global") or 0),
                "eval_batch_per_rank": int(config.get("batch_size") or 0),
                "eval_steps_per_rank": max((int(row.get("eval_steps_per_rank") or 0) for row in shard_metas), default=0),
                "gpu_memory_peak": max((int(row.get("gpu_memory_peak") or 0) for row in shard_metas), default=0),
                "throughput": sum(float(row.get("throughput") or 0.0) for row in shard_metas),
                "selected_eval_batch": int(config.get("batch_size_global") or 0),
                "invariance_status": "NOT_RUN",
                "metric_implementation_version": STEP3_PAPER_METRIC_IMPLEMENTATION_VERSION,
            }
            _write_json_file(os.path.join(artifact_root, "eval_summary.json"), summary)
            status_payload = {
                "train_status": "completed",
                "eval_status": "completed",
                "quality_status": "paper_evaluated" if protocol == PAPER_TARGET_ONLY_EVAL else "minimally_evaluated",
                "downstream_ready": bool(protocol == PAPER_TARGET_ONLY_EVAL and integrity.get("status") == "PASS"),
                "failure_phase": "",
                "eval_protocol": protocol,
                "eval_scope": split,
                "paper_comparable": bool(protocol_spec.get("paper_comparable", False)),
                "selected_checkpoint": os.path.abspath(str(config.get("save_file", ""))),
                "selected_checkpoint_scope": "eval_only_checkpoint",
                "artifact_root": os.path.abspath(artifact_root),
                "eval_summary": os.path.join(os.path.abspath(artifact_root), "eval_summary.json"),
            }
            _write_eval_status_sidecar(log_path, status_payload)
            _td = (
                f"Step 3 two-phase eval Task {task_idx} protocol={protocol} split={split} "
                f"(nproc={world_size}): {args.auxiliary} -> {args.target}"
            )
            _write_eval_results_log(
                config,
                final,
                task_description=_td,
                pipeline=f"Step3_two_phase_{protocol}",
                domain_from=args.auxiliary,
                domain_to=args.target,
                start_time=_eval_start_str,
                eval_elapsed=_eval_elapsed,
            )
            ev_logger.info("DONE.")
    except Exception as exc:
        if rank == 0 and config is not None:
            _write_eval_status_sidecar(
                log_path,
                {
                    "train_status": "completed",
                    "eval_status": "failed",
                    "quality_status": "not_evaluated",
                    "downstream_ready": False,
                    "failure_phase": "post_train_eval",
                    "eval_protocol": protocol,
                    "eval_scope": split,
                    "paper_comparable": bool(protocol_spec.get("paper_comparable", False)),
                    "selected_checkpoint": os.path.abspath(str(config.get("save_file", ""))),
                    "selected_checkpoint_scope": "eval_only_checkpoint",
                    "artifact_root": os.path.abspath(artifact_root),
                    "error": repr(exc),
                },
            )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _dispatch_eval(args):
    if not _eval_torchrun_env():
        raise RuntimeError(_EVAL_REQUIRES_TORCHRUN_MSG)
    if not torch.cuda.is_available():
        raise RuntimeError("检测到分布式启动环境但无 CUDA，无法使用 eval DDP（需要 CUDA + NCCL）。")
    _run_eval_ddp(args)


def _add_train_args(p):
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志路径；mainline 默认 runs/.../meta/full.log；内部直跑不应作为用户入口",
    )
    p.add_argument("--device", type=int, default=0, help="DDP 下以 LOCAL_RANK 为准，可忽略")
    p.add_argument("--auxiliary", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率；不传则由 build_resolved_training_config 按 BASE→TASK→预设→ENV→CLI 解析",
    )
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None, help="不传则用 config.epochs")
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="训练全局有效 batch（每优化步跨所有 rank 的样本总数）",
    )
    p.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="单卡 DataLoader 微批；mainline 由父进程 resolved payload 注入",
    )
    p.add_argument("--num-proc", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="快速验证用：最多训练到 N 个 batch 就退出")
    p.add_argument(
        "--save-final-checkpoint",
        action="store_true",
        help="训练进程退出时（含 --max-steps 提前结束）由 rank0 无条件写入 save_file；"
        "不改变 loss/BLEU 选模逻辑。适用于 smoke / debug；正式训练默认关闭。",
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help="早停生效前最少训练的 epoch 数；默认 TRAIN_MIN_EPOCHS / config",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="验证损失相对上一轮变差的连续次数上限（改进时会清零）；默认 TRAIN_EARLY_STOP_PATIENCE",
    )
    p.add_argument(
        "--early-stop-patience-full",
        type=int,
        default=None,
        help="dual_bleu 时：连续多少次 full BLEU eval 未刷新 best 则早停（quick 不参与）；"
        "默认 TRAIN_EARLY_STOP_PATIENCE_FULL 或与 --early-stop-patience 相同",
    )
    p.add_argument(
        "--early-stop-patience-loss",
        type=int,
        default=None,
        help="dual_bleu：valid_loss 连续变差早停次数，与 patience_full 独立；"
        "默认 TRAIN_EARLY_STOP_PATIENCE_LOSS 或同 --early-stop-patience",
    )
    p.add_argument(
        "--checkpoint-metric",
        type=str,
        choices=["valid_loss", "loss"],
        default="valid_loss",
        help="保存 canonical best.pth 的依据：仅 valid_loss（与训练 preset checkpoint_metric 一致）",
    )
    p.add_argument(
        "--bleu4-max-samples",
        type=int,
        default=None,
        help="按 BLEU-4 选模时验证集最多采样条数；默认 TRAIN_BLEU4_MAX_SAMPLES",
    )
    p.add_argument(
        "--scheduler-initial-lr",
        type=float,
        default=None,
        help="优化器初始 LR；若设置则覆盖 --learning_rate（均在 resolve 内处理）",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="warmup 步数；等价 ODCR_WARMUP_STEPS",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="warmup 占计划总步数比例；等价 ODCR_WARMUP_RATIO",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="cosine 末端 LR / initial_lr；等价 ODCR_MIN_LR_RATIO",
    )
    p.add_argument(
        "--quick-eval-max-samples",
        type=int,
        default=None,
        help="每 epoch quick BLEU 子集；等价 ODCR_QUICK_EVAL_MAX_SAMPLES",
    )
    p.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DDP 是否检测未参与 loss 的参数；默认真（避免多分支/对抗训练报错），可用 --no-ddp-find-unused-parameters 换略快训练",
    )


def _add_eval_args(p):
    p.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志路径；mainline 默认 runs/.../meta/full.log；内部直跑不应作为用户入口",
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--auxiliary", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-proc", type=int, default=None)
    p.add_argument("--eval-protocol", type=str, default=None)
    p.add_argument("--eval-split", type=str, default=None)
