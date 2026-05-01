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
import warnings
import argparse
import contextlib
import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

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
from paths_config import get_data_dir, get_hf_cache_root, get_merged_data_dir, get_stage_run_dir, require_step5_text_model_dir
from odcr_core.runtime_env_pack import runtime_env_dict_for_config_resolved
from odcr_core.config_schema import SAFE_DECODE_PLACEHOLDER
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
    current_effective_payload,
    current_one_control_resolved_config_hash,
    file_fingerprint,
    stable_hash,
    state_dict_for_canonical_best_pth,
    write_checkpoint_lineage,
)
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
    create_run_paths,
    setup_train_logging,
    log_run_header,
    log_config_snapshot,
    flush_preset_load_events,
    format_epoch_training_block,
    log_epoch_training_block,
    broadcast_run_paths_ddp,
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

def _nonfinite_loss_abort_threshold() -> int:
    raw = os.environ.get("ODCR_NONFINITE_LOSS_ABORT_AFTER", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _validate_step3_loss_bundle(
    losses: Dict[str, torch.Tensor],
    *,
    ctx: str,
    check_finite: bool = True,
) -> None:
    required = (
        "L_rating_shared",
        "L_ortho_xcov",
        "L_ortho_cos",
        "L_ortho_total",
        "L_shared_invariance",
        "L_specific_separation",
        "L_shared_var",
        "L_specific_var",
        "L_var_total",
        "L_anchor_sh",
        "L_anchor_sp",
        "L_content_align",
        "L_style_align",
        "L_shared_proto",
        "L_domain_style_align",
        "L_local_style_align",
        "L_polarity_align",
        "L_residual_sp",
        "L_proto_sep",
    )
    for k in required:
        if k not in losses:
            raise RuntimeError(f"{ctx} 缺少 Step3 ODCR 损失项: {k}")
        v = losses[k]
        if check_finite and not bool(torch.isfinite(v).all().item()):
            raise RuntimeError(f"{ctx} 出现非有限 Step3 ODCR 损失项: {k}={v}")


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


def step3_global_finite_loss_decision(loss: torch.Tensor, *, world_size: int) -> tuple[bool, bool]:
    """Return ``(local_finite, global_finite)`` with DDP all-reduce synchronization."""
    local_finite = bool(torch.isfinite(loss.detach()).all().item())
    if int(world_size) <= 1:
        return local_finite, local_finite
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("Step3 DDP finite-loss sync requested before torch.distributed init.")
    flag = torch.tensor(1 if local_finite else 0, dtype=torch.int32, device=loss.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return local_finite, bool(int(flag.item()))


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


def compose_step3_structured_loss(
    *,
    weights: Step3StructuredLossWeights,
    rating_shared: torch.Tensor,
    light_explainer: torch.Tensor,
    orthogonal_total: torch.Tensor,
    variance_total: torch.Tensor,
    shared_invariance: torch.Tensor,
    specific_separation: torch.Tensor,
    anchor_shared: torch.Tensor,
    anchor_specific: torch.Tensor,
    content_alignment: torch.Tensor,
    style_alignment: torch.Tensor,
    shared_prototype: torch.Tensor,
    domain_style_alignment: torch.Tensor,
    local_style_alignment: torch.Tensor,
    polarity_alignment: torch.Tensor,
    residual_specific: torch.Tensor,
    prototype_separation: torch.Tensor,
) -> torch.Tensor:
    return (
        rating_shared
        + weights.light_explainer_weight * light_explainer
        + weights.orthogonal_weight * orthogonal_total
        + weights.variance_weight * variance_total
        + weights.shared_invariance_weight * shared_invariance
        + weights.specific_separation_weight * specific_separation
        + weights.anchor_alignment_weight * anchor_shared
        + weights.anchor_alignment_weight * anchor_specific
        + weights.content_alignment_weight * content_alignment
        + weights.style_alignment_weight * style_alignment
        + weights.shared_prototype_weight * shared_prototype
        + weights.domain_style_alignment_weight * domain_style_alignment
        + weights.local_style_alignment_weight * local_style_alignment
        + weights.polarity_alignment_weight * polarity_alignment
        + weights.residual_specific_weight * residual_specific
        + weights.prototype_separation_weight * prototype_separation
    )


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


def _build_step3_checkpoint_lineage(final_cfg: FinalTrainingConfig) -> Dict[str, Any]:
    payload = current_effective_payload(required=True)
    task_idx = int(final_cfg.task_idx)
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
    arch = _step3_model_architecture_lineage(final_cfg)
    lineage: Dict[str, Any] = {
        "stage": "step3",
        "one_control_resolved_config_hash": current_one_control_resolved_config_hash(
            extra={"stage": "step3", "task_idx": task_idx}
        ),
        "training_semantic_fingerprint": os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT", ""),
        "preprocess_contract_version": PREPROCESS_CONTRACT_VERSION,
        "artifact_lineage": data_fps,
        "data_merged_artifact_fingerprint": stable_hash(data_fps),
        "embed_dim": int(final_cfg.emsize),
        "profile_dims": {
            "user_count": int(final_cfg.nuser),
            "item_count": int(final_cfg.nitem),
            "emsize": int(final_cfg.emsize),
        },
        "step3_structured_losses_config_hash": stable_hash(structured),
        "model_architecture_config": arch,
        "model_architecture_config_hash": stable_hash(arch),
        "source_task": {
            "task_id": task_idx,
            "auxiliary": str(final_cfg.auxiliary),
            "target": str(final_cfg.target),
        },
    }
    lineage["checkpoint_compatibility_hash"] = stable_hash(lineage)
    return lineage

# HuggingFace tokenize 磁盘缓存：修改 Processor/tokenize 语义或需强制失效时与 step5 引擎同步递增
ODCR_TOKENIZE_CACHE_VERSION = "v6_structured_evidence_paths"

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
    def __init__(self, auxiliary, target):
        self.max_length = 25
        self.evidence_length = 24
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
        self.domain_content_profiles = nn.Parameter(domain_content_profiles)
        self.domain_style_profiles = nn.Parameter(domain_style_profiles)
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.user_content_profiles = nn.Parameter(user_content_profiles)
        self.user_style_profiles = nn.Parameter(user_style_profiles)
        self.item_content_profiles = nn.Parameter(item_content_profiles)
        self.item_style_profiles = nn.Parameter(item_style_profiles)
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
        self.evidence_length = 24
        self.rating_loss_fn = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
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
        eid = getattr(tok, "eos_token_id", None)
        self.decoder_eos_id = int(eid) if eid is not None else -1

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
        domain_content = self.domain_content_profiles[domain_idx]
        domain_style = self.domain_style_profiles[domain_idx]
        user_content = self.user_content_profiles[user]
        user_style = self.user_style_profiles[user]
        item_content = self.item_content_profiles[item]
        item_style = self.item_style_profiles[item]

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
        self.last_odcr_latents = latents
        self.last_shared_proj = latents.shared_proj
        self.last_specific_proj = latents.specific_proj
        rating = self.recommender(latents.shared)
        context_logits = self.hidden2token(latents.specific).unsqueeze(1)
        context_dist = context_logits.repeat(1, tgt_input.shape[1], 1)
        word_dist = self.hidden2token(hidden[:, self._prefix_len():])
        return rating, context_dist, word_dist, latents.shared_proj, latents.specific_proj

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
        # 配合 DataLoader(pin_memory=True) 使用 non_blocking=True，减少同步拷贝等待
        user_idx = user_idx.to(device, non_blocking=True)
        item_idx = item_idx.to(device, non_blocking=True)
        domain_idx = domain_idx.to(device, non_blocking=True)
        rating = rating.to(device, non_blocking=True).float()
        tgt_output = tgt_output.to(device, non_blocking=True)
        sample_id = sample_id.to(device, non_blocking=True)
        content_anchor_score = content_anchor_score.to(device, non_blocking=True).float()
        style_anchor_score = style_anchor_score.to(device, non_blocking=True).float()
        content_evidence_ids = content_evidence_ids.to(device, non_blocking=True).long()
        style_evidence_ids = style_evidence_ids.to(device, non_blocking=True).long()
        domain_style_anchor_ids = domain_style_anchor_ids.to(device, non_blocking=True).long()
        local_style_hint_ids = local_style_hint_ids.to(device, non_blocking=True).long()
        polarity_ids = polarity_ids.to(device, non_blocking=True).long()
        evidence_quality_prior = evidence_quality_prior.to(device, non_blocking=True).float()
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
            if None in (ca, sa, ce, se, dsa, lsh, pol, eq):
                raise RuntimeError("validModel gather 缺少 Step3 canonical evidence 张量。")
            with odcr_cuda_bf16_autocast():
                pred_rating, _, word_dist, _, _, = model(
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
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
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
            if None in (ca, sa, ce, se, dsa, lsh, pol, eq):
                raise RuntimeError("validModel_sum_batches gather 缺少 Step3 canonical evidence 张量。")
            bsz = int(user_idx.size(0))
            with odcr_cuda_bf16_autocast():
                pred_rating, _, word_dist, _, _, = model(
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
            loss_r = _model.rating_loss_fn(pred_rating, rating)
            loss_e = _model.exp_loss_fn(word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1))
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


def _dist_barrier_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _safe_file_mtime(path: str) -> str:
    try:
        if not os.path.isfile(path):
            return "missing"
        return str(int(os.path.getmtime(path)))
    except OSError:
        return "na"


def _tokenizer_cache_identity(tok) -> str:
    nop = getattr(tok, "name_or_path", None) or getattr(tok, "name", None)
    if nop:
        return str(nop)
    return type(tok).__name__


def _build_tokenize_cache_fingerprint(
    *,
    train_path: str,
    valid_path: str,
    tok,
    max_length: int,
    cache_version: str,
) -> str:
    """稳定、简洁的 cache key 段（含可读版本前缀 + 12 位 sha1 截断）。"""
    parts = [
        f"train={os.path.abspath(train_path)}",
        f"train_mtime={_safe_file_mtime(train_path)}",
        f"valid={os.path.abspath(valid_path)}",
        f"valid_mtime={_safe_file_mtime(valid_path)}",
        f"tok={_tokenizer_cache_identity(tok)}",
        f"maxlen={int(max_length)}",
        f"ver={cache_version}",
    ]
    raw = "|".join(parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{cache_version}_{h}"


def _build_step3_cache_dir(
    task_idx: int,
    train_path: str,
    valid_path: str,
    processor,
    tok,
    cache_version: str = ODCR_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str]:
    fp = _build_tokenize_cache_fingerprint(
        train_path=train_path,
        valid_path=valid_path,
        tok=tok,
        max_length=int(getattr(processor, "max_length", 25)),
        cache_version=cache_version,
    )
    cache_dir = os.path.join(get_hf_cache_root(task_idx), f"hf_cache_step3_{fp}")
    return cache_dir, fp


def _build_eval_tokenize_cache_fingerprint(
    *,
    eval_data_path: str,
    tok,
    max_length: int,
    cache_version: str,
) -> str:
    """
    Step3 eval 子命令专用 tokenize 缓存 key（与 train 的 train+valid 缓存独立）。
    字段需覆盖：eval 数据路径、mtime、tokenizer、max_length、版本、mode=eval。
    """
    parts = [
        "mode=eval",
        f"data={os.path.abspath(eval_data_path)}",
        f"data_mtime={_safe_file_mtime(eval_data_path)}",
        f"tok={_tokenizer_cache_identity(tok)}",
        f"maxlen={int(max_length)}",
        f"ver={cache_version}",
    ]
    raw = "|".join(parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{cache_version}_{h}"


def _build_step3_eval_cache_dir(
    task_idx: int,
    eval_data_path: str,
    processor,
    tok,
    cache_version: str = ODCR_TOKENIZE_CACHE_VERSION,
) -> Tuple[str, str]:
    fp = _build_eval_tokenize_cache_fingerprint(
        eval_data_path=eval_data_path,
        tok=tok,
        max_length=int(getattr(processor, "max_length", 25)),
        cache_version=cache_version,
    )
    cache_dir = os.path.join(get_hf_cache_root(task_idx), f"hf_cache_step3_eval_{fp}")
    return cache_dir, fp


def _hf_dataset_cache_ready(cache_dir: str) -> bool:
    return os.path.isdir(cache_dir) and os.path.isfile(os.path.join(cache_dir, "dataset_dict.json"))


def _log_tokenize_cache_line(msg: str, log_file: Optional[str]) -> None:
    lg = logging.getLogger(LOGGER_NAME)
    if not logger_has_file_handler(lg):
        print(msg, flush=True)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


def _map_tokenize_train_valid_to_hf_cache(
    *,
    datasets: DatasetDict,
    processor,
    nproc: int,
    cache_dir: str,
    cache_fingerprint: str,
    rank: int,
    show_datasets_progress: bool,
    log_tokenize: bool,
    phase: str,
    log_file: Optional[str],
) -> DatasetDict:
    """
    rank0 负责 map + save_to_disk；barrier 后所有 rank load_from_disk。
    cache 已存在时各 rank 直接 load（跳过 map）。
    """
    if _hf_dataset_cache_ready(cache_dir):
        t_hit0 = time.perf_counter()
        encoded_data = load_from_disk(cache_dir)
        elapsed_hit = time.perf_counter() - t_hit0
        if rank == 0 and log_tokenize:
            msg = (
                f"[Tokenize] {phase} cache hit | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"load_wall_time={elapsed_hit:.2f}s"
            )
            _log_tokenize_cache_line(msg, log_file)
        return encoded_data

    if rank == 0:
        t0 = time.perf_counter()
        with hf_datasets_progress_bar(show_datasets_progress):
            encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
        encoded_data.save_to_disk(cache_dir)
        elapsed = time.perf_counter() - t0
        if log_tokenize:
            _log_tokenize_done(phase, nproc, elapsed, log_file)
            msg = (
                f"[Tokenize] {phase} cache miss | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"build_wall_time={elapsed:.2f}s"
            )
            _log_tokenize_cache_line(msg, log_file)

    _dist_barrier_if_initialized()
    encoded_data = load_from_disk(cache_dir)
    return encoded_data


def _load_step3_artefacts(
    args,
    device: int,
    resolved: FinalTrainingConfig,
    *,
    rank: int = 0,
    log_tokenize: bool = True,
    show_datasets_progress: bool = True,
):
    task_idx = int(resolved.task_idx)
    path = os.path.join(get_merged_data_dir(), str(task_idx))
    train_path = os.path.join(path, "aug_train.csv")
    valid_path = os.path.join(path, "aug_valid.csv")
    train_df = pd.read_csv(train_path)
    _require_step3_canonical_columns(train_df, csv_path=train_path, split="train")
    nuser = int(train_df["user_idx"].max()) + 1
    nitem = int(train_df["item_idx"].max()) + 1

    os.makedirs(get_stage_run_dir(task_idx), exist_ok=True)
    nproc = int(resolved.num_proc)
    save_file = args.save_file or os.path.join(get_stage_run_dir(task_idx), "model", "best.pth")

    valid_df = pd.read_csv(valid_path)
    _require_step3_canonical_columns(valid_df, csv_path=valid_path, split="valid")
    train_df["item"] = train_df["item"].astype(str)
    valid_df["item"] = valid_df["item"].astype(str)
    # 与 step5 一致：仅保留有 explanation 的训练行（nuser/nitem 已在上方用过滤前 train_df 计算）
    train_df = train_df[train_df["explanation"].notna()].reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    train_df["sample_id"] = np.arange(len(train_df), dtype=np.int64)
    valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "valid": Dataset.from_pandas(valid_df),
    })
    processor = Processor(args.auxiliary, args.target)
    cache_dir, cache_fp = _build_step3_cache_dir(
        task_idx, train_path, valid_path, processor, get_odcr_text_tokenizer(),
    )
    if rank == 0 and log_tokenize:
        _log_tokenize_cache_line(
            f"[Tokenize] step3 cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
            getattr(args, "log_file", None),
        )
    _tok_lg = logging.getLogger(LOGGER_NAME)
    with odcr_timing_phase(
        _tok_lg,
        "tokenize_pipeline_step3_train_valid",
        route=ROUTE_SUMMARY,
        rank=rank,
    ):
        encoded_data = _map_tokenize_train_valid_to_hf_cache(
            datasets=datasets,
            processor=processor,
            nproc=nproc,
            cache_dir=cache_dir,
            cache_fingerprint=cache_fp,
            rank=rank,
            show_datasets_progress=show_datasets_progress,
            log_tokenize=log_tokenize,
            phase="train+valid",
            log_file=getattr(args, "log_file", None),
        )
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
    resolved_profiles = replace(resolved, emsize=_em_prof)

    model = Model(
        nuser,
        nitem,
        int(len(get_odcr_text_tokenizer())),
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
    model.apply_runtime_config(resolved_profiles, get_odcr_text_tokenizer())
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
    A = int(resolved.gradient_accumulation_steps)

    train_dataset, valid_dataset, model, nuser, nitem, save_file, resolved_profiles = (
        _load_step3_artefacts(
            args,
            local_rank,
            resolved,
            rank=rank,
            log_tokenize=(rank == 0),
            show_datasets_progress=(rank == 0),
        )
    )

    train_drop_last = A > 1
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=train_drop_last,
    )
    pin_memory = torch.cuda.is_available()
    nw_train = int(resolved.dataloader_num_workers_train)
    nw_valid = int(resolved.dataloader_num_workers_valid)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=P,
        sampler=sampler,
        shuffle=False,
        num_workers=nw_train,
        pin_memory=pin_memory,
        persistent_workers=nw_train > 0,
        prefetch_factor=resolved.dataloader_prefetch_factor_train,
        drop_last=train_drop_last,
    )
    n_train_micro = len(train_dataloader)
    A_accum = max(1, int(A))
    tail_size = int(n_train_micro % A_accum)
    if rank == 0:
        _acc_lg = logging.getLogger(LOGGER_NAME)
        _acc_msg = (
            f"[Train/accum] n_train_micro={n_train_micro} gradient_accumulation_steps={A_accum} "
            f"tail_size={tail_size}"
        )
        if _acc_lg.handlers:
            _acc_lg.info(_acc_msg)
        else:
            print(_acc_msg, flush=True)
    if G % world_size != 0:
        raise ValueError(
            f"train_batch_size={G} 与 world_size={world_size} 不整除，无法得到每卡 batch。"
            "请修改 training preset（train_batch_size / per_device_train_batch_size / gradient_accumulation_steps）或 hardware preset（ddp_world_size）。"
        )
    valid_per_rank = G // world_size
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
        persistent_workers=nw_valid > 0,
        prefetch_factor=resolved.dataloader_prefetch_factor_valid,
    )

    _ddp_find_unused = bool(resolved.ddp_find_unused_parameters)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=_ddp_find_unused,
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
        rank0_only_logging=True,
    )

    return final_cfg, train_dataloader, valid_dataloader, model, sampler


@contextlib.contextmanager
def _ddp_no_sync_model(model, world_size: int, sync_gradients: bool):
    """在梯度累积的非边界微批上使用 DDP no_sync，边界步上规约梯度。"""
    if world_size <= 1 or sync_gradients:
        yield
    else:
        with model.no_sync():
            yield


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
    A = max(1, int(final_cfg.gradient_accumulation_steps))
    eff = int(final_cfg.effective_global_batch_size)
    initial_lr = float(final_cfg.scheduler_initial_lr)
    learning_rate = initial_lr
    _model = get_underlying_model(model)
    device = final_cfg.device
    use_bf16 = odcr_cuda_bf16_autocast_enabled()
    n_micro = len(train_dataloader)
    n_steps = max(1, (n_micro + A - 1) // A)
    train_info = (
        f"[Train] global_batch_size={G} effective_global_batch_size={eff} "
        f"per_device_batch_size={P} gradient_accumulation_steps={A} world_size={world_size} "
        f"micro_batches_per_epoch={n_micro} optimizer_steps_per_epoch={n_steps} epochs={epochs}"
    )
    _lg = final_cfg.logger
    min_epochs = int(final_cfg.min_epochs)
    early_stop_patience = int(final_cfg.early_stop_patience)
    checkpoint_metric = str(final_cfg.checkpoint_metric)
    lr_scheduler_type = str(final_cfg.lr_scheduler)
    warmup_epochs = float(final_cfg.warmup_epochs)
    min_lr_ratio = float(final_cfg.min_lr_ratio)
    warmup_steps_env = final_cfg.odcr_warmup_steps
    warmup_ratio_env = final_cfg.odcr_warmup_ratio
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
        elif os.environ.get("ODCR_RUNTIME_PRECISION_MODE", "").strip().lower() in ("fp32", "fp16"):
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

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    ema_enabled = bool(getattr(final_cfg, "ema_enabled", True))
    ema_decay = float(getattr(final_cfg, "ema_decay", 0.999))
    ema_model: Optional[AveragedModel] = None
    if ema_enabled:
        ema_model = AveragedModel(_model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
    sched = None
    ws_resolved = None
    warmup_ratio_logged = 0.0
    min_lr_effective = initial_lr * min_lr_ratio
    if lr_scheduler_type == "warmup_cosine":
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
                "LR schedule resolved: scheduler_type=warmup_cosine "
                "initial_lr=%s current_lr=%s (equals initial before first step) min_lr=%s min_lr_ratio=%s "
                "warmup_steps=%d total_steps=%d warmup_ratio=%s | "
                "LambdaLR: one scheduler.step() immediately after each optimizer.step() (global_step aligned)",
                initial_lr,
                initial_lr,
                min_lr_effective,
                min_lr_ratio,
                ws_resolved,
                total_steps_plan,
                warmup_ratio_logged,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
    structured_weights = step3_structured_loss_weights_from_config(final_cfg)
    orth_w_xcov = structured_weights.orthogonal_xcov_weight
    orth_w_cos = structured_weights.orthogonal_cosine_weight
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
            "[DDP] ddp_find_unused_parameters=%s（true=安全默认，扫描未用参数；false=吞吐向，须图稳定）",
            bool(final_cfg.ddp_find_unused_parameters),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[DDP] epoch_end_barrier=%s（ODCR_DDP_EPOCH_END_BARRIER；默认关闭）",
            str(odcr_ddp_epoch_end_barrier()).lower(),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[Diag] ODCR_GRAD_TOPK=%d（仅 >0 时 log_grad_monitor 打印 top 参数 grad norm）",
            odcr_grad_topk(),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
    try:
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            epoch_1 = epoch + 1
            if rank == 0:
                perf.epoch_start()
            model.train()
            loss_sum = torch.zeros((), dtype=torch.double, device=device)
            n_samples = torch.zeros((), dtype=torch.double, device=device)
            micro_step_epoch = 0
            window_step = 0
            window_nonfinite_loss = False
            optimizer.zero_grad(set_to_none=True)
            iterator = train_dataloader
            if rank == 0:
                iterator = tqdm(train_dataloader, total=len(train_dataloader))
            for step_idx, batch in enumerate(iterator):
                gn_pre_m = None
                gn_post_m = None
                step_count += 1
                micro_step_epoch += 1
                window_step += 1
                is_last_batch = step_idx == n_micro - 1
                sync = (micro_step_epoch % A == 0) or is_last_batch
                if sync:
                    current_accum = window_step
                else:
                    current_accum = A
                inv_accum = 1.0 / float(current_accum)
                is_tail_window = bool(sync and window_step < A)
                sync_ctx = _ddp_no_sync_model(model, world_size, sync)
                g = require_gathered_batch(_model.gather(batch, device))
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
                if None in (c_a, s_a, ce, se, dsa, lsh, pol, eq):
                    raise RuntimeError("Step3 gather 缺少 canonical evidence 张量。")
                bsz = int(user_idx.size(0))
                warn_empty_batch(_lg, global_step=global_step, epoch=epoch_1, bsz=bsz)
                q_w = (0.55 + 0.45 * eq.view(-1).to(dtype=torch.float32).clamp(0.0, 1.0)).detach()
                q_content = q_w * (0.40 + 0.60 * c_a.view(-1).to(dtype=torch.float32).clamp(0.0, 1.0))
                q_style = q_w * (0.40 + 0.60 * s_a.view(-1).to(dtype=torch.float32).clamp(0.0, 1.0))
                with sync_ctx:
                    with odcr_cuda_bf16_autocast():
                        pred_rating, _, word_dist, _, _ = model(
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
                        lat = _model.last_odcr_latents
                        l_anchor_sh = anchor_score_alignment_loss(
                            lat.anchor_pred_content,
                            c_a.view(-1),
                            sample_weight=q_content,
                        )
                        l_anchor_sp = anchor_score_alignment_loss(
                            lat.anchor_pred_style,
                            s_a.view(-1),
                            sample_weight=q_style,
                        )
                        l_content_align = cosine_pull_loss(
                            lat.shared_latent,
                            lat.content_evidence_target,
                            sample_weight=q_content,
                        )
                        l_style_align = cosine_pull_loss(
                            lat.specific_latent,
                            lat.style_evidence_target,
                            sample_weight=q_style,
                        )
                        l_shared_proto = shared_prototype_pull_loss(
                            lat.shared_latent,
                            lat.shared_prototype,
                            sample_weight=q_content,
                        )
                        l_domain_style_align = cosine_pull_loss(
                            lat.domain_style_component,
                            lat.domain_style_target,
                            sample_weight=q_style,
                        )
                        l_local_style_align = cosine_pull_loss(
                            lat.residual_local,
                            lat.local_style_target,
                            sample_weight=q_style,
                        )
                        l_polarity_align = cosine_pull_loss(
                            lat.specific_latent,
                            lat.polarity_target,
                            sample_weight=q_style,
                        )
                        l_res_sp = residual_l2_penalty(lat.residual_local)
                        l_proto = domain_style_prototype_separation(
                            _model.odcr_disentangler.domain_style_proto.weight
                        )
                        l_rating_shared = _model.rating_loss_fn(pred_rating, rating)
                        light_explainer = _model.exp_loss_fn(
                            word_dist.view(-1, _model.ntoken), tgt_output.reshape(-1)
                        )
                        _orth = build_orthogonal_losses(
                            lat.shared_latent,
                            lat.specific_latent,
                            w_xcov=orth_w_xcov,
                            w_cos=orth_w_cos,
                        )
                        l_ortho_xcov = _orth.loss_ortho_xcov
                        l_ortho_cos = _orth.loss_ortho_cos
                        l_ortho_total = _orth.loss_ortho_total
                        _var = variance_floor_loss(lat.shared_latent, lat.specific_latent)
                        l_shared_var = _var.loss_shared_var
                        l_specific_var = _var.loss_specific_var
                        l_var_total = _var.loss_var_total
                        l_shared_inv = shared_invariance_loss(lat.shared_latent, domain_idx)
                        l_specific_sep = specific_separation_loss(lat.specific_latent, domain_idx)
                        loss = compose_step3_structured_loss(
                            weights=structured_weights,
                            rating_shared=l_rating_shared,
                            light_explainer=light_explainer,
                            orthogonal_total=l_ortho_total,
                            variance_total=l_var_total,
                            shared_invariance=l_shared_inv,
                            specific_separation=l_specific_sep,
                            anchor_shared=l_anchor_sh,
                            anchor_specific=l_anchor_sp,
                            content_alignment=l_content_align,
                            style_alignment=l_style_align,
                            shared_prototype=l_shared_proto,
                            domain_style_alignment=l_domain_style_align,
                            local_style_alignment=l_local_style_align,
                            polarity_alignment=l_polarity_align,
                            residual_specific=l_res_sp,
                            prototype_separation=l_proto,
                        )
                        _validate_step3_loss_bundle(
                            {
                                "L_rating_shared": l_rating_shared,
                                "L_ortho_xcov": l_ortho_xcov,
                                "L_ortho_cos": l_ortho_cos,
                                "L_ortho_total": l_ortho_total,
                                "L_shared_var": l_shared_var,
                                "L_specific_var": l_specific_var,
                                "L_var_total": l_var_total,
                                "L_shared_invariance": l_shared_inv,
                                "L_specific_separation": l_specific_sep,
                                "L_anchor_sh": l_anchor_sh,
                                "L_anchor_sp": l_anchor_sp,
                                "L_content_align": l_content_align,
                                "L_style_align": l_style_align,
                                "L_shared_proto": l_shared_proto,
                                "L_domain_style_align": l_domain_style_align,
                                "L_local_style_align": l_local_style_align,
                                "L_polarity_align": l_polarity_align,
                                "L_residual_sp": l_res_sp,
                                "L_proto_sep": l_proto,
                            },
                            ctx="step3/train",
                            check_finite=False,
                        )
                        local_loss_finite, global_loss_finite = step3_global_finite_loss_decision(
                            loss,
                            world_size=world_size,
                        )
                        if global_loss_finite:
                            (loss * inv_accum).backward()
                        else:
                            window_nonfinite_loss = True
                            _nonfinite_skips += 1
                            optimizer.zero_grad(set_to_none=True)
                            if rank == 0 and _lg:
                                _lg.warning(
                                    "[Train] non-finite loss synchronized skip: "
                                    "epoch=%d micro_step=%d next_global_step=%d local_finite=%s",
                                    epoch_1,
                                    micro_step_epoch,
                                    global_step + 1,
                                    str(local_loss_finite).lower(),
                                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                                )
                            if _nonfinite_abort_th > 0 and _nonfinite_skips >= _nonfinite_abort_th:
                                raise RuntimeError(
                                    f"非有限 loss 累计同步跳过 {_nonfinite_skips} 次 >= "
                                    f"ODCR_NONFINITE_LOSS_ABORT_AFTER={_nonfinite_abort_th}"
                                )
                if sync:
                    if window_nonfinite_loss:
                        if rank == 0 and _lg:
                            _lg.warning(
                                "[Train] non-finite loss synchronized window skip optimizer: "
                                "epoch=%d next_global_step=%d",
                                epoch_1,
                                global_step + 1,
                                extra=log_route_extra(_lg, ROUTE_SUMMARY),
                            )
                        optimizer.zero_grad(set_to_none=True)
                        window_nonfinite_loss = False
                    else:
                        if rank == 0 and _lg:
                            gn_pre_m = grad_norm_total(model.parameters())
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if rank == 0 and _lg:
                            gn_post_m = grad_norm_total(model.parameters())
                            log_grad_monitor(
                                _lg,
                                model,
                                global_step=global_step + 1,
                                epoch=epoch_1,
                                route_detail=ROUTE_DETAIL,
                                grad_norm_pre_clip=gn_pre_m,
                                grad_norm_post_clip=gn_post_m,
                                current_accum=current_accum,
                                is_tail_window=is_tail_window,
                                skip_param_topk=not ((global_step + 1) % grad_iv == 0),
                            )
                        optimizer.step()
                        if ema_model is not None:
                            ema_model.update_parameters(_model)
                        optimizer.zero_grad(set_to_none=True)
                        if sched is not None:
                            sched.step()
                        global_step += 1
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
                            _lr_now = optimizer.param_groups[0]["lr"]
                            _extra_na: Dict[str, Any] = {
                                "loss_ortho": float(l_ortho_total.detach().item()),
                                "loss_ortho_xcov": float(l_ortho_xcov.detach().item()),
                                "loss_ortho_cos": float(l_ortho_cos.detach().item()),
                                "loss_var_total": float(l_var_total.detach().item()),
                            }
                            if odcr_log_step_loss_parts():
                                _extra_na.update({
                                    "L_rating_shared": float(l_rating_shared.detach().item()),
                                    "L_shared_var": float(l_shared_var.detach().item()),
                                    "L_specific_var": float(l_specific_var.detach().item()),
                                    "L_shared_invariance": float(l_shared_inv.detach().item()),
                                    "L_specific_separation": float(l_specific_sep.detach().item()),
                                    "L_content_align": float(l_content_align.detach().item()),
                                    "L_style_align": float(l_style_align.detach().item()),
                                    "L_shared_proto": float(l_shared_proto.detach().item()),
                                    "L_domain_style_align": float(l_domain_style_align.detach().item()),
                                    "L_local_style_align": float(l_local_style_align.detach().item()),
                                    "L_polarity_align": float(l_polarity_align.detach().item()),
                                    "shared_std_mean": float(_var.shared_std_mean.detach().item()),
                                    "specific_std_mean": float(_var.specific_std_mean.detach().item()),
                                })
                            _extra_na["current_accum"] = int(current_accum)
                            _extra_na["is_tail_window"] = bool(is_tail_window)
                            if gn_pre_m is not None and gn_post_m is not None:
                                _extra_na["grad_norm_pre_clip"] = float(gn_pre_m)
                                _extra_na["grad_norm_post_clip"] = float(gn_post_m)
                            log_step_sample(
                                _lg,
                                global_step=global_step,
                                epoch=epoch_1,
                                lr=float(_lr_now),
                                train_loss_batch=float(loss.detach().item()),
                                extra=_extra_na or None,
                            )
                    window_step = 0

                if global_loss_finite:
                    loss_sum = loss_sum + loss.detach().double() * bsz
                    n_samples += bsz
                if global_loss_finite and step_count % step_iv == 0 and rank == 0:
                    run_training_finite_checks(
                        _finite_mode,
                        loss,
                        word_dist,
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

            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bleu_line = None
                lr_sched_line = None
                if sched is not None and ws_resolved is not None:
                    lr_sched_line = (
                        f"scheduler_type=warmup_cosine "
                        f"initial_lr={initial_lr:.6g} current_lr={lr_epoch:.6g} min_lr={min_lr_effective:.6g} "
                        f"min_lr_ratio={min_lr_ratio:.6g} warmup_steps={ws_resolved} total_steps={total_steps_plan} "
                        f"scheduler_steps_end_of_epoch={global_step} warmup_ratio={warmup_ratio_logged:.6g}"
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

            if current_valid_loss > prev_valid_loss:
                enduration += 1
                if lr_scheduler_type != "warmup_cosine":
                    learning_rate /= 2.0
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate
            else:
                enduration = 0
                if str(checkpoint_metric).strip().lower() in ("valid_loss", "loss") and rank == 0:
                    best_version += 1
                    state_to_save = state_dict_for_canonical_best_pth(
                        ema_enabled=ema_enabled,
                        ema_model=ema_model,
                        ddp_module=model,
                        underlying_model_fn=get_underlying_model,
                    )
                    atomic_torch_save(str(final_cfg.save_file), state_to_save)
                    _ckpt_lineage = _build_step3_checkpoint_lineage(final_cfg)
                    _ckpt_lineage["checkpoint_file"] = file_fingerprint(str(final_cfg.save_file))
                    _ckpt_lineage_path = write_checkpoint_lineage(str(final_cfg.save_file), _ckpt_lineage)
                    _run_dir = os.path.dirname(os.path.dirname(os.path.abspath(str(final_cfg.save_file))))
                    _state_dir = os.path.join(_run_dir, "state")
                    os.makedirs(_state_dir, exist_ok=True)
                    atomic_torch_save(os.path.join(_state_dir, "optimizer.pt"), optimizer.state_dict())
                    atomic_write_json(
                        os.path.join(_state_dir, "trainer_state.json"),
                        {"epoch": int(epoch_1), "global_step": int(global_step)},
                    )
                    atomic_write_json(
                        os.path.join(_state_dir, "best_event.json"),
                        {
                            "best_version": int(best_version),
                            "epoch": int(epoch_1),
                            "global_step": int(global_step),
                            "valid_loss": float(current_valid_loss),
                            "model_sha256": _sha256_file(str(final_cfg.save_file)),
                            "checkpoint_lineage_hash": _ckpt_lineage.get("checkpoint_compatibility_hash"),
                            "checkpoint_lineage_path": str(_ckpt_lineage_path),
                            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "ema_enabled": bool(ema_enabled),
                            "weight_kind": "ema" if ema_enabled else "raw",
                        },
                    )
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
            _sf = os.path.abspath(os.path.expanduser(str(final_cfg.save_file)))
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
            _final_lineage = _build_step3_checkpoint_lineage(final_cfg)
            _final_lineage["checkpoint_file"] = file_fingerprint(_sf)
            write_checkpoint_lineage(_sf, _final_lineage)


def _eval_collect_shard_predictions(model, test_dataloader, device):
    _model = get_underlying_model(model)
    model = model.to(device)
    model.eval()
    prediction_ratings = []
    ground_truth_ratings = []
    prediction_exps = []
    reference_exps = []
    # inference_mode 与 no_grad 数值一致，略少开销，适合纯推理
    with torch.inference_mode():
        for batch in test_dataloader:
            g = require_gathered_batch(_model.gather(batch, device))
            user_idx, item_idx, rating, tgt_output, domain_idx = (
                g.user_idx,
                g.item_idx,
                g.rating,
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
            if None in (ca, sa, ce, se, dsa, lsh, pol, eq):
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
            prediction_ratings.extend(pred_ratings.tolist())
            ground_truth_ratings.extend(rating.tolist())
            prediction_exps.extend(get_odcr_text_tokenizer().batch_decode(pred_exps, skip_special_tokens=True))
            reference_exps.extend(get_odcr_text_tokenizer().batch_decode(tgt_output, skip_special_tokens=True))
    return prediction_ratings, ground_truth_ratings, prediction_exps, reference_exps


def metrics_from_eval_lists(prediction_ratings, ground_truth_ratings, prediction_exps, reference_exps):
    prediction_ratings = np.array(prediction_ratings)
    ground_truth_ratings = np.array(ground_truth_ratings)
    rating_diffs = prediction_ratings - ground_truth_ratings
    mae = round(np.mean(np.abs(rating_diffs)), 4)
    rmse = round(np.sqrt(np.mean(np.square(rating_diffs))), 4)
    text_results = evaluate_text(prediction_exps, reference_exps)
    return {"recommendation": {"mae": mae, "rmse": rmse}, "explanation": text_results}


def evalModel(model, test_dataloader, device):
    pr, gt, pe, re = _eval_collect_shard_predictions(model, test_dataloader, device)
    return metrics_from_eval_lists(pr, gt, pe, re)


def build_config_and_dataloader(args, ddp_rank: int, ddp_world_size: int, local_rank: int):
    primary_device = local_rank

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    path = os.path.join(get_merged_data_dir(), str(task_idx))
    train_df = pd.read_csv(os.path.join(path, "aug_train.csv"))
    nuser = train_df['user_idx'].max() + 1
    nitem = train_df['item_idx'].max() + 1

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

    valid_path = os.path.join(path, "aug_valid.csv")
    valid_df = pd.read_csv(valid_path)
    _require_step3_canonical_columns(valid_df, csv_path=valid_path, split="eval-valid")
    valid_df['item'] = valid_df['item'].astype(str)
    valid_df = valid_df.reset_index(drop=True)
    valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
    datasets = DatasetDict({
        "valid": Dataset.from_pandas(valid_df)
    })
    processor = Processor(args.auxiliary, args.target)
    cache_dir, cache_fp = _build_step3_eval_cache_dir(
        task_idx=int(task_idx),
        eval_data_path=os.path.abspath(valid_path),
        processor=processor,
        tok=get_odcr_text_tokenizer(),
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
    eval_world_size = max(ddp_world_size, 1)
    dl_valid = get_dataloader_num_workers("valid")
    num_workers = min(max(1, dl_valid // eval_world_size), 8)
    pin_memory = torch.cuda.is_available()
    _pf_ev = get_dataloader_prefetch_factor(num_workers, split="valid")
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=_pf_ev,
    )

    dc, ds, uc, us, ic, ist, profile_meta = load_profile_tensors_dual_first(
        data_root=get_data_dir(),
        auxiliary_domain=args.auxiliary,
        target_domain=args.target,
        device_idx="cpu",
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

    return config, valid_dataloader, model


def _run_train_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    ddp_fast_backends = apply_ddp_fast_torch_backends()
    dist.init_process_group(backend="nccl")

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

    if rank == 0:
        log_path, run_id = create_run_paths(task_idx, args.log_file)
    else:
        log_path, run_id = None, None
    log_path, run_id = broadcast_run_paths_ddp(log_path, run_id, rank)
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
                "per_device_batch_size": final_cfg.per_device_train_batch_size,
                "gradient_accumulation_steps": final_cfg.gradient_accumulation_steps,
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
            _cfg_resolved_path = os.path.join(os.path.dirname(log_path), "resolved_config.json")
            with open(_cfg_resolved_path, "w", encoding="utf-8") as _cf:
                json.dump(_cfg_snap, _cf, ensure_ascii=False, indent=2, default=str)
                _cf.write("\n")
            train_logger.info(
                "[Config resolved] wrote %s",
                _cfg_resolved_path,
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
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
    dist.init_process_group(backend="nccl")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")

    if rank == 0:
        log_path, run_id = create_run_paths(task_idx, args.log_file)
    else:
        log_path, run_id = None, None
    log_path, run_id = broadcast_run_paths_ddp(log_path, run_id, rank)
    args.log_file = log_path

    _setup = setup_train_logging(
        log_file=log_path,
        task_idx=task_idx,
        rank=rank,
        world_size=world_size,
        run_id=run_id,
    )
    ev_logger = _setup["logger"]

    try:
        config, valid_dataloader, model = build_config_and_dataloader(
            args, ddp_rank=rank, ddp_world_size=world_size, local_rank=local_rank,
        )
        config["logger"] = ev_logger
        config["run_id"] = run_id
        if rank == 0:
            log_run_header(
                ev_logger,
                {
                    "run_id": run_id,
                    "task_idx": task_idx,
                    "rank": rank,
                    "world_size": world_size,
                    "mode": "eval_ddp",
                    "cuda_available": True,
                    "local_rank": local_rank,
                    "batch_size": config.get("batch_size_global", config.get("batch_size")),
                    "save_file": os.path.abspath(str(config.get("save_file", ""))),
                    "log_file": os.path.abspath(log_path),
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                },
            )
        _eval_t0 = time.time()
        _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pr, gt, pe, re = _eval_collect_shard_predictions(model, valid_dataloader, local_rank)
        payload = {
            "prediction_ratings": pr,
            "ground_truth_ratings": gt,
            "prediction_exps": pe,
            "reference_exps": re,
        }
        if rank == 0:
            gathered = [None] * world_size
        else:
            gathered = None
        dist.gather_object(payload, gathered, dst=0)

        if rank == 0:
            all_pr, all_gt, all_pe, all_re = [], [], [], []
            for shard in gathered:
                all_pr.extend(shard["prediction_ratings"])
                all_gt.extend(shard["ground_truth_ratings"])
                all_pe.extend(shard["prediction_exps"])
                all_re.extend(shard["reference_exps"])
            final = metrics_from_eval_lists(all_pr, all_gt, all_pe, all_re)
            _eval_elapsed = time.time() - _eval_t0
            _td = (
                f"Step 3 structured DDP eval Task {task_idx} (nproc={world_size}): "
                f"{args.auxiliary} -> {args.target}"
            )
            _write_eval_results_log(
                config,
                final,
                task_description=_td,
                pipeline="Step3_structured_eval_ddp",
                domain_from=args.auxiliary,
                domain_to=args.target,
                start_time=_eval_start_str,
                eval_elapsed=_eval_elapsed,
            )
            ev_logger.info("DONE.")
        dist.barrier()
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
    p.add_argument(
        "--coef",
        type=float,
        default=None,
        help="不传则由 resolve：TASK_DEFAULTS / 命名预设 / ENV / CLI 覆盖链决定",
    )
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="训练全局有效 batch（每优化步跨所有 rank 的样本总数）",
    )
    p.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="梯度累积步数；默认 config / ODCR_GRADIENT_ACCUMULATION_STEPS；满足 G=P×world_size×A",
    )
    p.add_argument(
        "--per-device-batch-size",
        type=int,
        default=None,
        help="单卡 DataLoader 微批；显存不足时减小并配合全局 G 推出 accum；或 ODCR_PER_DEVICE_BATCH_SIZE",
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
