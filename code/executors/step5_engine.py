"""
Step5 执行体核心（ENGINE）：主模型 train/eval/test。

由 ``executors.step5_entry`` 在 torchrun 下调用（code/ 下历史薄壳名保持不变）。
用户入口请使用 ``python code/odcr.py step5|eval|pipeline …``。
"""
import os
import sys
import time
import hashlib
import logging
import shutil
from datetime import datetime, timezone
# 离线模式：禁止从网络加载
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
from base_utils import *
from paths_config import get_data_dir, get_hf_cache_root, get_stage_run_dir, require_step5_text_model_dir
from config import (
    FinalTrainingConfig,
    apply_ddp_fast_torch_backends,
    build_full_bleu_monitor_cfg_override,
    build_resolved_training_config,
    format_full_bleu_eval_epoch_decision_log_line,
    format_full_bleu_eval_resolved_log_line,
    format_full_bleu_monitor_log_line,
    hf_datasets_progress_bar,
    resolve_task_idx_from_aux_target,
    should_run_full_bleu_eval_epoch,
)
from training_hardware_inputs import collect_training_hardware_overrides_from_args
from odcr_core.runtime_env_pack import runtime_env_dict_for_config_resolved
from odcr_core.training_diagnostics import training_diagnostics_snapshot
from lr_schedule_utils import resolve_warmup_steps, warmup_cosine_multiplier_lambda
from bleu_valid_ddp import bleu4_explanation_full_valid_ddp
from odcr_core.bleu_runtime import explanation_bleu4_quick_score, mainline_monitor_full_valid_ddp
from odcr_core.mainline_monitor import mainline_selection_gate
from odcr_core.gather_schema import GatheredBatch, require_gathered_batch
from odcr_core.index_contract import (
    GLOBAL_COL_ITEM,
    GLOBAL_COL_USER,
    IndexContractError,
    ODCR_ROUTING_TRAIN_CSV,
    load_index_contract,
    load_profile_tensors_from_contract,
    normalize_split_indices_to_global,
    resolve_index_contract_path,
    validate_first_batch_indices,
    validate_index_contract_against_profiles,
    validate_split_indices,
    validate_step4_export_lineage,
    write_index_contract_audit,
)
import torch

# transformers 在 modeling_utils.load_state_dict 里用 torch.load 未传 weights_only，
# PyTorch 2.4+ 会 FutureWarning；在 from_pretrained 前默认 weights_only=True。
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

import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch import nn, optim
from torch.nn.modules.transformer import _get_activation_fn
import argparse
import gzip
import json
import contextlib
import math
from functools import partial
from dataclasses import replace
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from tqdm import tqdm
from torch.optim import lr_scheduler as lr_sched
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datasets import Dataset, DatasetDict, load_from_disk
from perf_monitor import PerfMonitor, gather_ddp_gpu_stats_for_epoch_log
import numpy as np
import copy
import torch.nn.functional as F
from train_logging import (
    append_train_epoch_metrics_jsonl,
    create_run_paths,
    setup_train_logging,
    log_run_snapshot,
    flush_preset_load_events,
    format_epoch_summary_lines,
    format_epoch_training_block,
    log_epoch_training_block,
    log_epoch_summary_compact,
    broadcast_run_paths_ddp,
    format_final_results_lines,
    log_final_results_block,
    finalize_run_log,
    flush_odcr_file_handlers,
    LOGGER_NAME,
    logger_has_file_handler,
    log_route_extra,
    ROUTE_DETAIL,
    ROUTE_SUMMARY,
)
from odcr_core import path_layout
from odcr_eval_dirty_text import compute_dirty_text_stats
from odcr_core.generation_semantics import build_generation_semantic_resolved_and_fingerprint
from odcr_core.file_atomic import atomic_torch_save, atomic_write_json
from odcr_core.training_checkpoint import (
    CheckpointLineageError,
    STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    STEP5_EVAL_OUTPUT_SCHEMA_VERSION,
    STEP5_TRAIN_SCHEMA_VERSION,
    current_effective_payload,
    current_one_control_resolved_config_hash,
    file_fingerprint,
    model_artifact_fingerprint,
    read_checkpoint_lineage,
    stable_hash,
    state_dict_for_canonical_best_pth,
    write_checkpoint_lineage,
)
from odcr_core.rerank import (
    build_rerank_weights_dict,
    compute_lp_norm,
    extract_rerank_features,
    extract_rerank_features_for_v3,
    keywords_from_source_text,
    merge_rerank_v3_profile,
    rouge_l_proxy,
    score_candidates_rule_v1,
    score_candidates_rule_v2,
    score_candidates_rule_v3,
)
from odcr_eval_metrics import (
    compute_collapse_stats,
    eval_decode_tag,
    extended_text_metrics_bundle,
    log_sample_id_alignment_snippet,
    merge_eval_rows_by_sample_id,
    write_predictions_csv,
    write_predictions_jsonl,
    write_eval_digest_log,
)
from odcr_core.step5_word_losses import (
    odcr_anti_repeat_unlikelihood_loss_from_logp,
    per_sample_mean_ce_from_logp,
    route_weighted_mean,
)
from odcr_core.odcr_losses import build_orthogonal_losses
from odcr_core.odcr_scorer import ODCRScorer
from odcr_core.step5_innovation import (
    STEP5_EVIDENCE_FEATURE_DIM,
    CF_RELIABILITY,
    CONTENT_RETENTION,
    CCVControlPacket,
    EVIDENCE_QUALITY_PRIOR,
    RATING_STABILITY,
    STYLE_SHIFT,
    TEXT_QUALITY,
    UNCERTAINTY,
    build_ccv_control_packet,
    build_step5a_scorer_gate,
    build_step5b_explainer_gate,
    evidence_basis_fca_loss,
    lci_score_invariance_loss,
    parse_step5_innovation_config_json,
)
from odcr_core.step5_native_lora import apply_native_lora_to_step5_model, discover_step5_text_linear_targets
from odcr_core.step5b_flan_bridge import (
    discover_flan_explainer_lora_targets,
    per_sample_decoder_ce_from_logits,
)
from executors.decode_controller import (
    DECODE_BACKEND_FALLBACK_RAISE,
    DECODE_BACKEND_FALLBACK_SYNC_THEN_FALLBACK,
    DECODE_BACKEND_KV_FAST,
    DECODE_BACKEND_KV_SAFE,
    GenerateConfig,
    coerce_generate_cfg_override,
    build_candidate_generation_specs,
    apply_eos_boost,
    apply_min_len_eos_mask,
    apply_no_repeat_ngram_logits,
    apply_repetition_penalty_logits,
    apply_sampling_schedule,
    apply_token_repeat_suppression,
    apply_unbalanced_delimiter_eos_mask,
    build_generate_kwargs_effective_v2,
    decode_backend_uses_kv_cache,
    decode_exception_blocks_fallback,
    forbid_eos_if_bad_tail_token,
    prepare_logits,
    resolve_decode_backend_fallback_policy,
    resolve_decode_backend_name,
    sample_next_token,
)
from odcr_core.generation.decoder_kv import DecoderKVBackend
from odcr_core.generation.cache_types import PastKeyValues
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
    grad_topk_param_norms,
    log_bf16_amp_note,
    log_step_sample,
    log_training_crash,
    maybe_log_grad_norm_diff_ddp,
    parse_odcr_finite_check_mode,
    run_training_finite_checks,
    warn_empty_batch,
)

# T5 tokenizer：不在模块 import 时加载（测试环境可仅 import；训练/评测在首次 get_step5_tokenizer() 时加载）。
_step5_tokenizer_obj: Optional[Any] = None
_step5_tokenizer_override: Optional[Any] = None


def set_step5_tokenizer_override(tok: Optional[Any]) -> None:
    """测试/宿主注入：非 None 时 get_step5_tokenizer() 直接返回 tok，不触发磁盘/HF 加载。"""
    global _step5_tokenizer_override
    _step5_tokenizer_override = tok


def get_step5_tokenizer() -> Any:
    """懒加载 T5Tokenizer；import 本模块时不读取模型目录。"""
    global _step5_tokenizer_obj
    if _step5_tokenizer_override is not None:
        return _step5_tokenizer_override
    if _step5_tokenizer_obj is None:
        _t5_path = require_step5_text_model_dir()
        _step5_tokenizer_obj = T5Tokenizer.from_pretrained(
            _t5_path, legacy=True, local_files_only=True
        )
    return _step5_tokenizer_obj

# HuggingFace tokenize 磁盘缓存：与 Step3 在共享 tokenize 语义变更时同步递增
ODCR_TOKENIZE_CACHE_VERSION = "v7_step5_lineage_manifest"
STEP5_TOKENIZE_CACHE_SCHEMA_VERSION = "odcr_step5_tokenize_cache/1"
STEP5_TOKENIZE_CACHE_MANIFEST = "cache_manifest.json"
STEP5_TOKENIZE_CACHE_PRODUCER_CODE_VERSION = "executors.step5_engine.tokenize_cache/2"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_paths_from_save_file(save_file: str) -> Dict[str, str]:
    model_path = os.path.abspath(os.path.expanduser(save_file))
    run_dir = os.path.dirname(os.path.dirname(model_path))
    state_dir = os.path.join(run_dir, "state")
    return {
        "run_dir": run_dir,
        "state_dir": state_dir,
        "trainer_state": os.path.join(state_dir, "trainer_state.json"),
        "optimizer_pt": os.path.join(state_dir, "optimizer.pt"),
        "best_event": os.path.join(state_dir, "best_event.json"),
    }


def _step5_model_architecture_lineage(final_cfg: FinalTrainingConfig, model: nn.Module | None = None) -> Dict[str, Any]:
    underlying = get_underlying_model(model) if model is not None else None
    return {
        "nuser": int(final_cfg.nuser),
        "nitem": int(final_cfg.nitem),
        "ntoken": int(final_cfg.ntoken),
        "emsize": int(final_cfg.emsize),
        "nlayers": int(final_cfg.nlayers),
        "nhead": int(final_cfg.nhead),
        "nhid": int(final_cfg.nhid),
        "dropout": float(final_cfg.dropout),
        "flan_hidden_dim": int(getattr(underlying, "flan_d_model", 0) or 0),
        "ccv_numeric_control_dim": int(getattr(underlying, "ccv_numeric_control_dim", 0) or 0),
        "ccv_control_adapter_input_blocks": int(getattr(underlying, "ccv_control_adapter_input_blocks", 0) or 0),
        "train_mode": str(getattr(final_cfg, "train_mode", "")),
        "native_lora": {
            "r": int(getattr(final_cfg, "lora_r", 0) or 0),
            "alpha": float(getattr(final_cfg, "lora_alpha", 0.0) or 0.0),
            "dropout": float(getattr(final_cfg, "lora_dropout", 0.0) or 0.0),
            "target_modules": list(getattr(final_cfg, "lora_target_modules", ()) or ()),
        },
    }


def _step5_config_hashes(final_cfg: FinalTrainingConfig) -> Dict[str, str]:
    raw = str(final_cfg.step5_innovation_config_json or "").strip()
    if not raw:
        raise CheckpointLineageError("Step5 checkpoint lineage requires resolved step5_innovation_config_json.")
    obj = json.loads(raw)
    model_cfg = {
        "nlayers": int(final_cfg.nlayers),
        "nhead": int(final_cfg.nhead),
        "nhid": int(final_cfg.nhid),
        "dropout": float(final_cfg.dropout),
    }
    return {
        "step5_lci": stable_hash(obj.get("lci") or {}),
        "step5_uci": stable_hash(obj.get("uci") or {}),
        "step5_ccv": stable_hash(obj.get("ccv") or {}),
        "step5_fca": stable_hash(obj.get("fca") or {}),
        "step5_explainer_gate": stable_hash(obj.get("explainer_gate") or {}),
        "step5_model": stable_hash(model_cfg),
        "step5_innovation_full": stable_hash(obj),
    }


def _build_step5_checkpoint_lineage(final_cfg: FinalTrainingConfig, model: nn.Module) -> Dict[str, Any]:
    payload = current_effective_payload(required=True)
    step4_lineage = json.loads(str(final_cfg.step4_export_lineage_json or "{}"))
    if not isinstance(step4_lineage, dict) or not step4_lineage.get("lineage_hash"):
        raise CheckpointLineageError("Step5 checkpoint lineage requires Step4 export/index_contract lineage.")
    arch = _step5_model_architecture_lineage(final_cfg, model)
    lineage: Dict[str, Any] = {
        "stage": "step5",
        "compat_schema_version": STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "train_schema_version": STEP5_TRAIN_SCHEMA_VERSION,
        "step4_export_lineage_hash": str(step4_lineage["lineage_hash"]),
        "step5_config_hashes": _step5_config_hashes(final_cfg),
        "one_control_resolved_config_hash": current_one_control_resolved_config_hash(
            extra={"stage": "step5", "task_idx": int(final_cfg.task_idx)}
        ),
        "training_semantic_fingerprint": os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT", ""),
        "tokenizer_model_path": os.path.abspath(require_step5_text_model_dir()),
        "tokenizer_model_artifact_fingerprint": model_artifact_fingerprint(require_step5_text_model_dir()),
        "architecture": arch,
        "architecture_hash": stable_hash(arch),
        "task": {
            "task_id": int(final_cfg.task_idx),
            "auxiliary": str(final_cfg.auxiliary),
            "target": str(final_cfg.target),
        },
        "effective_payload_schema_version": payload.get("schema_version"),
    }
    lineage["checkpoint_compatibility_hash"] = stable_hash(lineage)
    return lineage


def _current_step5_checkpoint_expectation(final_cfg: FinalTrainingConfig, model: nn.Module | None = None) -> Dict[str, Any]:
    step4_lineage = json.loads(str(final_cfg.step4_export_lineage_json or "{}"))
    if not isinstance(step4_lineage, dict) or not step4_lineage.get("lineage_hash"):
        raise CheckpointLineageError("Eval/rerank requires current Step4 export lineage before checkpoint load.")
    arch = _step5_model_architecture_lineage(final_cfg, model)
    return {
        "compat_schema_version": STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION,
        "train_schema_version": STEP5_TRAIN_SCHEMA_VERSION,
        "step4_export_lineage_hash": str(step4_lineage["lineage_hash"]),
        "step5_config_hashes": _step5_config_hashes(final_cfg),
        "tokenizer_model_path": os.path.abspath(require_step5_text_model_dir()),
        "tokenizer_model_artifact_fingerprint": model_artifact_fingerprint(require_step5_text_model_dir()),
        "architecture_hash": stable_hash(arch),
        "task": {
            "task_id": int(final_cfg.task_idx),
            "auxiliary": str(final_cfg.auxiliary),
            "target": str(final_cfg.target),
        },
    }

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


_STEP5_PROCESSOR_REQUIRED_POSTERIOR_FIELDS = (
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "cf_reliability_score",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "uncertainty_score",
    "confidence_bucket",
    "text_quality_score",
)
_STEP5_PROCESSOR_REQUIRED_CCV_FIELDS = (
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
)
_STEP5_TOKENIZE_REQUIRED_FIELDS = (
    GLOBAL_COL_USER,
    GLOBAL_COL_ITEM,
    "rating",
    "clean_text",
    "domain",
    "sample_id",
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "cf_reliability_score",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "uncertainty_score",
    "confidence_bucket",
    "text_quality_score",
    "content_evidence",
    "style_evidence",
    "domain_style_anchor",
    "local_style_residual_hint",
    "polarity_anchor",
    "content_anchor_score",
    "style_anchor_score",
    "evidence_quality_prior",
)
STEP5_FACTUAL_EVAL_CONTROL_SCHEMA_VERSION = "odcr_step5_factual_eval_control/1.0"
STEP5_CONTROL_MODE_FACTUAL_EVAL_DEFAULT = "factual_eval_default"
STEP5_CONTROL_MODE_RCR_POSTERIOR = "rcr_posterior"
_STEP5_CONTROL_MODE_COLUMN = "step5_control_mode"
_STEP5_CONTROL_SOURCE_COLUMN = "step5_control_source"
_STEP5_CONTROL_CONTRACT_COLUMN = "step5_control_contract_version"


def step5_factual_eval_control_contract(split_label: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": STEP5_FACTUAL_EVAL_CONTROL_SCHEMA_VERSION,
        "mode": STEP5_CONTROL_MODE_FACTUAL_EVAL_DEFAULT,
        "split": str(split_label or ""),
        "route_scorer": 1,
        "route_explainer": 1,
        "sample_weight_hint": 1.0,
        "is_rcr_posterior": False,
        "is_train_route": False,
        "is_step4_export_posterior": False,
        "description": (
            "Step5 valid/test target factual rows receive neutral eval controls so the "
            "CCV/LCI packet can be built; these controls are not Step4 RCR posterior decisions."
        ),
    }


def _require_step5_rcr_posterior_controls(df: pd.DataFrame, *, ctx: str) -> None:
    try:
        _require_step5_train_csv_columns(df)
    except ValueError as exc:
        raise ValueError(
            f"{ctx} requires canonical Step4 RCR posterior/control columns. "
            "Rerun Step4 to produce odcr_routing_train.csv; factual_eval_default controls are valid/test only."
        ) from exc
    if _STEP5_CONTROL_MODE_COLUMN in df.columns:
        modes = set(df[_STEP5_CONTROL_MODE_COLUMN].dropna().astype(str))
        if STEP5_CONTROL_MODE_FACTUAL_EVAL_DEFAULT in modes:
            raise ValueError(
                f"{ctx} received factual_eval_default rows in the Step4 training export path; "
                "Step5 train input must use RCR posterior controls from Step4."
            )


def _apply_step5_factual_eval_default_controls(df: pd.DataFrame, *, split_label: str) -> pd.DataFrame:
    out = df.copy()
    contract = step5_factual_eval_control_contract(split_label)
    out["sample_weight_hint"] = 1.0
    out["route_scorer"] = int(contract["route_scorer"])
    out["route_explainer"] = int(contract["route_explainer"])
    out["route_reason_scorer"] = STEP5_CONTROL_MODE_FACTUAL_EVAL_DEFAULT
    out["route_reason_explainer"] = STEP5_CONTROL_MODE_FACTUAL_EVAL_DEFAULT
    out[_STEP5_CONTROL_MODE_COLUMN] = STEP5_CONTROL_MODE_FACTUAL_EVAL_DEFAULT
    out[_STEP5_CONTROL_SOURCE_COLUMN] = "step5_valid_test_target_split"
    out[_STEP5_CONTROL_CONTRACT_COLUMN] = STEP5_FACTUAL_EVAL_CONTROL_SCHEMA_VERSION
    out["step5_control_is_rcr_posterior"] = False
    out["step5_control_is_train_route"] = False  # internal-only eval contract label
    out["step5_control_is_step4_export_posterior"] = False
    out["content_evidence"] = out["clean_text"].fillna("").astype(str)
    out["style_evidence"] = ""
    out["domain_style_anchor"] = "target"
    out["local_style_residual_hint"] = ""
    out["polarity_anchor"] = np.where(out["rating"].astype(float) >= 3.0, "positive", "negative")
    for _col, _dv in (
        ("entropy_score", 0.25),
        ("uncertainty_score", 0.25),
        ("confidence_bucket", 1.0),
        ("content_anchor_score", 0.5),
        ("style_anchor_score", 0.5),
        ("evidence_quality_prior", 0.5),
        ("cf_reliability_score", 1.0),
        ("content_retention_score", 1.0),
        ("style_shift_score", 0.0),
        ("rating_stability_score", 1.0),
        ("text_quality_score", 1.0),
    ):
        if _col not in out.columns:
            out[_col] = _dv
    return out


def _is_missing_sample_value(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _required_sample_value(sample: Mapping[str, Any], key: str) -> Any:
    if key not in sample or _is_missing_sample_value(sample[key]):
        raise KeyError(
            f"Step5 training sample missing required Step4 posterior/control field {key!r}; "
            "use the canonical Step4 odcr_routing_train.csv and do not fall back to legacy fields."
        )
    return sample[key]


def _required_sample_float(sample: Mapping[str, Any], key: str) -> float:
    return float(_required_sample_value(sample, key))


def _control_text_to_ids(text: Any, *, max_length: int) -> torch.Tensor:
    raw = "" if _is_missing_sample_value(text) else str(text)
    ids = get_step5_tokenizer()(
        raw,
        padding=False,
        max_length=max(1, int(max_length)),
        truncation=True,
    )["input_ids"]
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask, src_key_padding_mask):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attns = []
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)  
        if self.norm is not None:
            output = self.norm(output)
        return output, attns

class Processor():
    def __init__(self, auxiliary, target, max_length: int = 25):
        self.max_length = int(max_length)
        self.auxiliary = auxiliary
        self.target = target

    def __call__(self, sample):
        if GLOBAL_COL_USER not in sample or GLOBAL_COL_ITEM not in sample:
            raise KeyError(
                f"样本须含 Step4 全局索引列 {GLOBAL_COL_USER}/{GLOBAL_COL_ITEM}（禁止混用未标注语义的 user_idx 列）。"
            )
        user_idx = torch.tensor(int(sample[GLOBAL_COL_USER]), dtype=torch.long)
        item_idx = torch.tensor(int(sample[GLOBAL_COL_ITEM]), dtype=torch.long)
        raitng = torch.tensor(sample["rating"], dtype=torch.float)
        if "clean_text" not in sample:
            raise KeyError(
                "训练样本缺少 clean_text（须使用新版 Step4 CSV）；禁止静默回退到 explanation 原文。"
            )
        explanation = sample["clean_text"]
        explanation_idx = get_step5_tokenizer()(
            explanation, padding=False, max_length=self.max_length, truncation=True
        )["input_ids"]
        explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)

        if sample["domain"] == "auxiliary":
            domain_val = 0  # Auxiliary domain
        elif sample["domain"] == "target":
            domain_val = 1  # Target domain
        else:
            raise ValueError("Unknown domain!")

        domain_idx = torch.tensor(domain_val, dtype=torch.long)
        sample_id = torch.tensor(int(sample["sample_id"]), dtype=torch.long)
        for _field in (*_STEP5_PROCESSOR_REQUIRED_POSTERIOR_FIELDS, *_STEP5_PROCESSOR_REQUIRED_CCV_FIELDS):
            _required_sample_value(sample, _field)
        sw = _required_sample_float(sample, "sample_weight_hint")
        exp_sample_weight = torch.tensor(float(sw), dtype=torch.float32)
        route_scorer_mask = torch.tensor(int(_required_sample_float(sample, "route_scorer")), dtype=torch.float32)
        route_explainer_mask = torch.tensor(int(_required_sample_float(sample, "route_explainer")), dtype=torch.float32)
        entropy_score = torch.tensor(float(sample.get("entropy_score", 0.25)), dtype=torch.float32)
        uncertainty_score = torch.tensor(_required_sample_float(sample, "uncertainty_score"), dtype=torch.float32)
        confidence_bucket = torch.tensor(_required_sample_float(sample, "confidence_bucket"), dtype=torch.float32)
        content_anchor_score = torch.tensor(_required_sample_float(sample, "content_anchor_score"), dtype=torch.float32)
        style_anchor_score = torch.tensor(_required_sample_float(sample, "style_anchor_score"), dtype=torch.float32)
        evidence_quality_prior = _required_sample_float(sample, "evidence_quality_prior")
        cf_reliability_score = _required_sample_float(sample, "cf_reliability_score")
        style_shift_score = _required_sample_float(sample, "style_shift_score")
        rating_stability_score = _required_sample_float(sample, "rating_stability_score")
        content_retention_score = _required_sample_float(sample, "content_retention_score")
        text_quality_score = _required_sample_float(sample, "text_quality_score")
        evf = torch.tensor(
            [
                evidence_quality_prior,
                cf_reliability_score,
                style_shift_score,
                rating_stability_score,
                content_retention_score,
                text_quality_score,
                float(uncertainty_score.item()),
                float(raitng.item() - 3.0) / 2.0,
            ],
            dtype=torch.float32,
        )
        control_max_len = max(4, min(32, int(self.max_length)))
        return {
            "user_idx": user_idx,
            "item_idx": item_idx,
            "rating": raitng,
            "explanation_idx": explanation_idx,
            "domain_idx": domain_idx,
            "sample_id": sample_id,
            "exp_sample_weight": exp_sample_weight,
            "route_scorer_mask": route_scorer_mask,
            "route_explainer_mask": route_explainer_mask,
            "entropy_score": entropy_score,
            "uncertainty_score": uncertainty_score,
            "confidence_bucket": confidence_bucket,
            "content_anchor_score": content_anchor_score,
            "style_anchor_score": style_anchor_score,
            "evidence_features": evf,
            "content_evidence_ids": _control_text_to_ids(sample["content_evidence"], max_length=control_max_len),
            "style_evidence_ids": _control_text_to_ids(sample["style_evidence"], max_length=control_max_len),
            "domain_style_anchor_ids": _control_text_to_ids(sample["domain_style_anchor"], max_length=control_max_len),
            "local_style_hint_ids": _control_text_to_ids(sample["local_style_residual_hint"], max_length=control_max_len),
            "polarity_ids": _control_text_to_ids(sample["polarity_anchor"], max_length=control_max_len),
        }


def _step5_collate_dynamic(
    batch: List[Dict[str, torch.Tensor]],
    *,
    dynamic_padding: bool,
    fixed_max_length: int,
):
    if not batch:
        raise ValueError("step5 collate 收到空 batch。")
    user_idx = torch.stack([torch.as_tensor(x["user_idx"], dtype=torch.long) for x in batch], dim=0)
    item_idx = torch.stack([torch.as_tensor(x["item_idx"], dtype=torch.long) for x in batch], dim=0)
    rating = torch.stack([torch.as_tensor(x["rating"], dtype=torch.float32) for x in batch], dim=0)
    domain_idx = torch.stack([torch.as_tensor(x["domain_idx"], dtype=torch.long) for x in batch], dim=0)
    sample_id = torch.stack([torch.as_tensor(x["sample_id"], dtype=torch.long) for x in batch], dim=0)
    exp_sample_weight = torch.stack(
        [torch.as_tensor(x["exp_sample_weight"], dtype=torch.float32) for x in batch], dim=0
    )
    route_scorer_mask = torch.stack(
        [torch.as_tensor(x.get("route_scorer_mask", 1.0), dtype=torch.float32) for x in batch], dim=0
    )
    route_explainer_mask = torch.stack(
        [torch.as_tensor(x.get("route_explainer_mask", 1.0), dtype=torch.float32) for x in batch], dim=0
    )
    entropy_score = torch.stack([torch.as_tensor(x["entropy_score"], dtype=torch.float32) for x in batch], dim=0)
    uncertainty_score = torch.stack(
        [torch.as_tensor(x["uncertainty_score"], dtype=torch.float32) for x in batch], dim=0
    )
    confidence_bucket = torch.stack(
        [torch.as_tensor(x["confidence_bucket"], dtype=torch.float32) for x in batch], dim=0
    )
    content_anchor_score = torch.stack(
        [torch.as_tensor(x["content_anchor_score"], dtype=torch.float32) for x in batch], dim=0
    )
    style_anchor_score = torch.stack(
        [torch.as_tensor(x["style_anchor_score"], dtype=torch.float32) for x in batch], dim=0
    )
    evidence_features = torch.stack([torch.as_tensor(x["evidence_features"], dtype=torch.float32) for x in batch], dim=0)

    def _pad_ids(name: str) -> torch.Tensor:
        seq_list = [torch.as_tensor(x[name], dtype=torch.long).view(-1) for x in batch]
        max_control_len = max(1, max(int(s.numel()) for s in seq_list))
        out = torch.zeros((len(seq_list), max_control_len), dtype=torch.long)
        for j, seq in enumerate(seq_list):
            L = min(max_control_len, int(seq.numel()))
            out[j, :L] = seq[:L]
        return out

    content_evidence_ids = _pad_ids("content_evidence_ids")
    style_evidence_ids = _pad_ids("style_evidence_ids")
    domain_style_anchor_ids = _pad_ids("domain_style_anchor_ids")
    local_style_hint_ids = _pad_ids("local_style_hint_ids")
    polarity_ids = _pad_ids("polarity_ids")
    seqs = [torch.as_tensor(x["explanation_idx"], dtype=torch.long).view(-1) for x in batch]
    if dynamic_padding:
        max_len = max(int(s.numel()) for s in seqs)
    else:
        max_len = max(1, int(fixed_max_length))
    padded = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = min(max_len, int(s.numel()))
        padded[i, :L] = s[:L]
    return (
        user_idx,
        item_idx,
        rating,
        padded,
        domain_idx,
        sample_id,
        exp_sample_weight,
        route_scorer_mask,
        route_explainer_mask,
        entropy_score,
        uncertainty_score,
        confidence_bucket,
        content_anchor_score,
        style_anchor_score,
        evidence_features,
        content_evidence_ids,
        style_evidence_ids,
        domain_style_anchor_ids,
        local_style_hint_ids,
        polarity_ids,
    )

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

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear1(hidden))  # (batch_size, emsize)
        rating = self.linear2(mlp_vector).view(-1)  # (batch_size,)
        return rating


def _domain_fusion_causal_mask(tgt_len: int, device: torch.device, prefix_len: int = 2) -> torch.Tensor:
    """前缀 prefix_len 个 token 全互见，其后为因果掩码；prefix_len 须与 Model._prefix_len() 一致。"""
    total_len = prefix_len + tgt_len
    mask = torch.triu(torch.ones((total_len, total_len), device=device, dtype=torch.bool), diagonal=1)
    mask[:prefix_len, :prefix_len] = False
    return mask


def _fca_explain_pool_from_encoder_hidden(
    encoder_last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Explainer encoder 末层 → FCA 用向量：仅对 attention_mask 标为有效的 time 步求均值。
    hidden ``[B, T, H]``，mask ``[B, T]``（1=有效，0=padding）；mask 广播为 ``[B, T, 1]`` 做加权和，
    有效计数 ``clamp(min=1)`` 避免除零；某行全零 mask 时分子为 0，得到零向量（图安全退化）。
    """
    hidden = encoder_last_hidden_state
    m = attention_mask.to(dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(-1)
    num = (hidden.float() * m).sum(dim=1)
    den = m.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1.0)
    return (num / den).float()


class _FlanT5ExplainerStub(nn.Module):
    """
    单测占位：与 ``ntoken``/``emsize`` 对齐的极小 seq2seq 头。
    仅当环境变量 ``ODCR_STEP5_INIT_FLAN_STUB=1`` 时由 ``Model`` 使用；训练 runner 禁止设置。
    """

    def __init__(self, *, vocab_size: int, d_model: int):
        super().__init__()
        self.config = type(
            "Cfg",
            (),
            {
                "vocab_size": int(vocab_size),
                "d_model": int(d_model),
            },
        )()
        self.dec = nn.Linear(int(d_model), int(vocab_size))

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        del attention_mask, output_hidden_states, kwargs
        if decoder_input_ids is None:
            raise RuntimeError("stub 需要 decoder_input_ids")
        b, td = decoder_input_ids.shape
        d = int(inputs_embeds.size(-1))
        enc_pool = inputs_embeds.mean(dim=1, keepdim=True).expand(b, td, d)
        logits = self.dec(enc_pool.float())
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits.to(dtype=inputs_embeds.dtype),
            encoder_last_hidden_state=inputs_embeds,
        )

    def generate(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 4,
        **kwargs: Any,
    ):
        del attention_mask, kwargs
        if inputs_embeds is None:
            raise RuntimeError("stub generate 需要 inputs_embeds")
        b = int(inputs_embeds.size(0))
        dev = inputs_embeds.device
        return torch.ones(b, max(1, int(max_new_tokens)), dtype=torch.long, device=dev)


class Model(nn.Module):
    """DMPF 主线：domain gate 残差调制 + domain cross-attn prefix fusion + decode controller。"""

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
        label_smoothing: float = 0.1,
        step5_innovation_config_json: str | None = None,
    ):
        super().__init__()
        if not str(step5_innovation_config_json or "").strip():
            raise RuntimeError("Step5 Model requires resolved step5_innovation_config_json from configs/odcr.yaml.")
        _model_step5_cfg = parse_step5_innovation_config_json(step5_innovation_config_json)
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
        self.odcr_scorer = ODCRScorer(emsize)
        if os.environ.get("ODCR_STEP5_INIT_FLAN_STUB", "").strip() == "1":
            self.flan_explainer = _FlanT5ExplainerStub(vocab_size=int(ntoken), d_model=int(emsize))
            self.flan_d_model = int(emsize)
        else:
            _t5p = require_step5_text_model_dir()
            _fd = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            # local_files_only=True + require_*：离线 fail-fast，不依赖 HF_HUB_OFFLINE 环境变量。
            self.flan_explainer = T5ForConditionalGeneration.from_pretrained(
                _t5p, local_files_only=True, torch_dtype=_fd
            )
            self.flan_d_model = int(self.flan_explainer.config.d_model)
        for _p in self.flan_explainer.parameters():
            _p.requires_grad_(False)
        self.ccv_enabled = bool(_model_step5_cfg.ccv.enabled)
        self.ccv_control_packet_field_policy = str(_model_step5_cfg.ccv.control_packet_field_policy)
        self.ccv_verbalizer_adapter_policy = str(_model_step5_cfg.ccv.verbalizer_adapter_policy)
        self.flan_soft_len = int(_model_step5_cfg.ccv.soft_prompt_len)
        self.ccv_numeric_control_dim = int(_model_step5_cfg.ccv.numeric_control_dim)
        self.ccv_control_adapter_input_blocks = int(_model_step5_cfg.ccv.control_adapter_input_blocks)
        self.flan_soft_prompt_stack = nn.Sequential(
            nn.Linear(int(emsize) * 2 + STEP5_EVIDENCE_FEATURE_DIM, self.flan_d_model * self.flan_soft_len),
            nn.GELU(),
        )
        self.ccv_numeric_adapter = nn.Linear(self.ccv_numeric_control_dim, emsize)
        self.ccv_control_adapter = nn.Sequential(
            nn.Linear(int(emsize) * self.ccv_control_adapter_input_blocks, self.flan_d_model * self.flan_soft_len),
            nn.GELU(),
        )
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.fca_score_align = nn.Linear(emsize, emsize)
        self.fca_explain_align = nn.Linear(self.flan_d_model, emsize)
        self.ntoken = int(ntoken)
        self.domain_cross_attn = nn.MultiheadAttention(
            embed_dim=emsize, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.domain_gate = nn.Linear(emsize, emsize)
        self.domain_fusion_mode = "gate_cross_attn"
        self.last_gate_stats: Dict[str, float] = {}
        self._last_uncertainty_decode_stats: Optional[Dict[str, Any]] = None
        self.register_buffer(
            "batch_diversity_ema_mean_probs",
            torch.zeros(int(ntoken), dtype=torch.float32),
        )
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = CustomTransformerEncoder(encoder_layers, nlayers)
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.emsize = emsize
        self.repetition_penalty = 1.15
        self.generate_temperature = 0.8
        self.generate_top_p = 0.9
        self.max_explanation_length = 25
        self.decode_strategy = "greedy"
        self.decode_seed = None  # type: Optional[int]
        self.no_repeat_ngram_size = None  # type: Optional[int]
        self.min_len = None  # type: Optional[int]
        self.soft_max_len = None  # type: Optional[int]
        self.hard_max_len = None  # type: Optional[int]
        self.eos_boost_start = 9999
        self.eos_boost_value = 0.0
        self.tail_temperature = -1.0
        self.tail_top_p = -1.0
        self.forbid_eos_after_open_quote = False
        self.forbid_eos_after_open_bracket = False
        self.forbid_bad_terminal_tokens = True
        self.bad_terminal_token_ids_resolved: Tuple[int, ...] = ()
        self.decode_token_repeat_window = 4
        self.decode_token_repeat_max = 2
        self.gap_threshold = 0.35
        self.uncertainty_entropy_eps = 1e-8
        self.prefix_greedy_steps = 4
        self.decode_top_k = 5
        self.candidate_family = "balanced"
        self.candidate_mixed_include_diverse = True
        self.decode_backend = DECODE_BACKEND_KV_FAST
        self.decode_backend_fallback_policy = DECODE_BACKEND_FALLBACK_RAISE
        self.loss_weight_repeat_ul = 0.0
        self.loss_weight_terminal_clean = 0.0
        self.terminal_clean_span = 3
        self.ccv_numeric_control_weight = 1.0
        self.decoder_eos_id = -1
        self.rating_loss_fn = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=float(label_smoothing))
        self.init_weights()

    def _flan_autocast_dtype(self) -> torch.dtype:
        try:
            return self.flan_explainer.dtype  # type: ignore[attr-defined]
        except Exception:
            return next(self.flan_explainer.parameters()).dtype

    def init_weights(self):
        # 仅初始化自有 Linear / Embedding / Parameter；跳过 TransformerEncoder 内 self_attn 子树，
        # 避免 uniform_ 覆盖 nn.MultiheadAttention 的 in_proj（PyTorch 默认 xavier 更合理）。
        initrange = 0.1

        def _init_linear(m: nn.Linear) -> None:
            nn.init.uniform_(m.weight.data, -initrange, initrange)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

        _init_linear(self.hidden2token)
        nn.init.uniform_(self.user_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.item_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.word_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.domain_content_profiles.data, -initrange, initrange)
        nn.init.uniform_(self.domain_style_profiles.data, -initrange, initrange)
        nn.init.uniform_(self.user_content_profiles.data, -initrange, initrange)
        nn.init.uniform_(self.user_style_profiles.data, -initrange, initrange)
        nn.init.uniform_(self.item_content_profiles.data, -initrange, initrange)
        nn.init.uniform_(self.item_style_profiles.data, -initrange, initrange)
        self.recommender.init_weights()
        for enc_layer in self.transformer_encoder.layers:
            for name, mod in enc_layer.named_modules():
                if ".self_attn" in name or name.endswith("self_attn"):
                    continue
                if isinstance(mod, nn.Linear):
                    _init_linear(mod)
        # Zero-destruction warm-start:
        # domain_gate.weight=bias=0 => gate=2*sigmoid(0)=1，调制恒等，不破坏已有训练主线。
        nn.init.zeros_(self.domain_gate.weight)
        nn.init.zeros_(self.domain_gate.bias)
        for mod in self.flan_soft_prompt_stack.modules():
            if isinstance(mod, nn.Linear):
                _init_linear(mod)
        _init_linear(self.ccv_numeric_adapter)
        for mod in self.ccv_control_adapter.modules():
            if isinstance(mod, nn.Linear):
                _init_linear(mod)
        _init_linear(self.fca_score_align)
        _init_linear(self.fca_explain_align)

    def apply_runtime_config(self, cfg: FinalTrainingConfig, tok) -> None:
        """从 FinalTrainingConfig + tokenizer 同步解码与解释头超参（训练/评测共用）。"""
        self.repetition_penalty = float(cfg.repetition_penalty)
        self.generate_temperature = max(float(cfg.generate_temperature), 1e-8)
        self.generate_top_p = float(cfg.generate_top_p)
        self.max_explanation_length = int(cfg.max_explanation_length)
        self.decode_strategy = str(cfg.decode_strategy).strip().lower()
        self.decode_seed = int(cfg.decode_seed) if cfg.decode_seed is not None else None
        _nr = getattr(cfg, "no_repeat_ngram_size", None)
        self.no_repeat_ngram_size = int(_nr) if _nr is not None and int(_nr) > 0 else None
        _mn = getattr(cfg, "min_len", None)
        self.min_len = int(_mn) if _mn is not None and int(_mn) > 0 else None
        _sm = getattr(cfg, "soft_max_len", None)
        self.soft_max_len = int(_sm) if _sm is not None and int(_sm) > 0 else None
        _hm = getattr(cfg, "hard_max_len", None)
        self.hard_max_len = int(_hm) if _hm is not None and int(_hm) > 0 else None
        self.eos_boost_start = int(getattr(cfg, "eos_boost_start", 9999))
        self.eos_boost_value = float(getattr(cfg, "eos_boost_value", 0.0))
        self.tail_temperature = float(getattr(cfg, "tail_temperature", -1.0))
        self.tail_top_p = float(getattr(cfg, "tail_top_p", -1.0))
        self.forbid_eos_after_open_quote = bool(getattr(cfg, "forbid_eos_after_open_quote", True))
        self.forbid_eos_after_open_bracket = bool(getattr(cfg, "forbid_eos_after_open_bracket", True))
        self.forbid_bad_terminal_tokens = bool(getattr(cfg, "forbid_bad_terminal_tokens", True))
        self.decode_token_repeat_window = int(getattr(cfg, "decode_token_repeat_window", 4))
        self.decode_token_repeat_max = int(getattr(cfg, "decode_token_repeat_max", 2))
        self.gap_threshold = float(getattr(cfg, "gap_threshold", 0.35))
        self.uncertainty_entropy_eps = float(getattr(cfg, "uncertainty_entropy_eps", 1e-8))
        self.prefix_greedy_steps = int(getattr(cfg, "prefix_greedy_steps", 4))
        self.decode_top_k = max(1, int(getattr(cfg, "decode_top_k", 5)))
        self.candidate_family = str(getattr(cfg, "candidate_family", "balanced")).strip().lower()
        self.candidate_mixed_include_diverse = bool(getattr(cfg, "candidate_mixed_include_diverse", True))
        self.loss_weight_repeat_ul = float(getattr(cfg, "loss_weight_repeat_ul", 0.0))
        self.loss_weight_terminal_clean = float(getattr(cfg, "loss_weight_terminal_clean", 0.0))
        self.terminal_clean_span = int(getattr(cfg, "terminal_clean_span", 3))
        _st5_json = str(getattr(cfg, "step5_innovation_config_json", "") or "").strip()
        if not _st5_json:
            raise RuntimeError("Step5 runtime config missing step5_innovation_config_json from One-Control.")
        _st5_innov = parse_step5_innovation_config_json(_st5_json)
        self.ccv_numeric_control_weight = float(_st5_innov.ccv.numeric_control_weight)
        self.domain_fusion_mode = str(getattr(cfg, "domain_fusion_mode", "gate_cross_attn")).strip().lower()
        self.decode_backend = resolve_decode_backend_name(getattr(cfg, "decode_backend", DECODE_BACKEND_KV_FAST))
        self.decode_backend_fallback_policy = resolve_decode_backend_fallback_policy(
            getattr(cfg, "decode_backend_fallback_policy", DECODE_BACKEND_FALLBACK_RAISE)
        )
        eid = getattr(tok, "eos_token_id", None)
        self.decoder_eos_id = int(eid) if eid is not None else -1
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=float(cfg.label_smoothing))
        _bt = getattr(cfg, "bad_terminal_token_ids", None)
        if _bt is not None and len(_bt) > 0:
            self.bad_terminal_token_ids_resolved = tuple(int(x) for x in _bt)
        else:
            self.bad_terminal_token_ids_resolved = Model._default_bad_terminal_token_ids(tok)

    @staticmethod
    def _default_bad_terminal_token_ids(tok) -> Tuple[int, ...]:
        """未配置时的坏尾 token：常见未闭合起笔符号的首子词 id。"""
        ids: List[int] = []
        for piece in ("(", "[", "{", "<", "``", "''"):
            try:
                enc = tok.encode(piece, add_special_tokens=False)
            except Exception:
                enc = []
            if enc:
                ids.append(int(enc[0]))
        return tuple(sorted(set(ids)))

    def _build_context_tokens(self, user, item):
        uc = self.user_content_profiles[user].unsqueeze(dim=1)
        us = self.user_style_profiles[user].unsqueeze(dim=1)
        ic = self.item_content_profiles[item].unsqueeze(dim=1)
        ist = self.item_style_profiles[item].unsqueeze(dim=1)
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        return torch.cat([uc, us, ic, ist, user_embeddings, item_embeddings], dim=1)

    def _compute_domain_gate(self, domain_embedding):
        return 2.0 * torch.sigmoid(self.domain_gate(domain_embedding))

    def _apply_domain_modulation(self, context_tokens, gate):
        # 残差写法显式保留 zero-destruction warm-start 语义
        return context_tokens + context_tokens * (gate - 1.0)

    def _build_prefix(self, domain_idx, user, item):
        dc = self.domain_content_profiles[domain_idx].unsqueeze(dim=1)
        ds = self.domain_style_profiles[domain_idx].unsqueeze(dim=1)
        domain_stack = torch.cat([dc, ds], dim=1)
        context_tokens = self._build_context_tokens(user, item)
        gate = self._compute_domain_gate(dc)
        modulated_context = self._apply_domain_modulation(context_tokens, gate)
        if self.domain_fusion_mode == "gate_only":
            domain_enhanced = domain_stack
        else:
            domain_enhanced, _ = self.domain_cross_attn(
                domain_stack, modulated_context, modulated_context, need_weights=False
            )
            if self.domain_fusion_mode == "cross_attn_only":
                modulated_context = context_tokens
        self.last_gate_stats = {
            "domain_gate_mean": float(gate.detach().mean().item()),
            "domain_gate_std": float(gate.detach().std(unbiased=False).item()),
            "domain_gate_min": float(gate.detach().amin().item()),
            "domain_gate_max": float(gate.detach().amax().item()),
            "gate_user_profile_mean": float(gate[:, :, :].detach().mean().item()),
            "gate_item_profile_mean": float(gate[:, :, :].detach().mean().item()),
            "gate_user_emb_mean": float(gate[:, :, :].detach().mean().item()),
            "gate_item_emb_mean": float(gate[:, :, :].detach().mean().item()),
        }
        prefix = torch.cat([domain_enhanced, domain_stack, modulated_context], dim=1)
        return prefix

    def get_domain_gate_stats(self) -> Dict[str, float]:
        return dict(self.last_gate_stats)

    @staticmethod
    def _prefix_len() -> int:
        # [domain_enhanced×2, domain_content/style×2, user_c/s, item_c/s, user_emb, item_emb]
        return 10

    def _mean_control_embedding(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2:
            raise RuntimeError(f"CCV control ids must be [B,T], got {tuple(ids.shape)}")
        safe_ids = ids.clamp(min=0, max=int(self.ntoken) - 1)
        emb = self.word_embeddings(safe_ids)
        mask = (safe_ids != 0).to(dtype=emb.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (emb * mask).sum(dim=1) / denom

    def _build_ccv_soft_prompt(
        self,
        shared_latent: torch.Tensor,
        specific_latent: torch.Tensor,
        evidence_features: torch.Tensor,
        control_packet: Optional[CCVControlPacket],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]], Optional[torch.Tensor]]:
        if control_packet is None:
            if bool(getattr(self, "ccv_enabled", True)):
                raise RuntimeError(
                    "Step5B CCV is enabled but no CCVControlPacket was provided; "
                    "prompt-only soft-prompt fallback is not an active path."
                )
            soft_in = torch.cat([shared_latent, specific_latent, evidence_features.to(dtype=shared_latent.dtype)], dim=-1)
            soft_flat = self.flan_soft_prompt_stack(soft_in)
            return soft_flat, None, None
        content_lat = self._mean_control_embedding(control_packet.content_evidence_ids)
        style_lat = self._mean_control_embedding(control_packet.style_evidence_ids)
        domain_style_lat = self._mean_control_embedding(control_packet.domain_style_anchor_ids)
        local_style_lat = self._mean_control_embedding(control_packet.local_style_hint_ids)
        polarity_lat = self._mean_control_embedding(control_packet.polarity_ids)
        style_basis = 0.50 * style_lat + 0.30 * domain_style_lat + 0.20 * local_style_lat
        numeric_controls = control_packet.numeric_controls().to(device=shared_latent.device, dtype=shared_latent.dtype)
        if int(numeric_controls.shape[-1]) != int(self.ccv_numeric_control_dim):
            raise RuntimeError(
                f"CCV numeric_controls dim={int(numeric_controls.shape[-1])} "
                f"does not match step5.ccv.numeric_control_dim={int(self.ccv_numeric_control_dim)}."
            )
        numeric_basis = self.ccv_numeric_adapter(numeric_controls) * float(self.ccv_numeric_control_weight)
        # CCV interface: content_lat controls what must be said, style_basis controls how it is phrased,
        # numeric_basis carries route/reliability/UCI tone without text prompt concatenation.
        ccv_in = torch.cat(
            [
                shared_latent,
                specific_latent,
                content_lat.to(dtype=shared_latent.dtype),
                style_basis.to(dtype=shared_latent.dtype),
                polarity_lat.to(dtype=shared_latent.dtype),
                numeric_basis,
            ],
            dim=-1,
        )
        soft_flat = self.ccv_control_adapter(ccv_in)
        stats = {
            "ccv_route_scorer_mean": float(control_packet.route_scorer_mask.detach().float().mean().item()),
            "ccv_route_explainer_mean": float(control_packet.route_explainer_mask.detach().float().mean().item()),
            "ccv_uncertainty_mean": float(control_packet.uncertainty_score.detach().float().mean().item()),
            "ccv_confidence_mean": float(control_packet.confidence_bucket.detach().float().mean().item()),
            "ccv_content_anchor_mean": float(control_packet.content_anchor_score.detach().float().mean().item()),
            "ccv_style_anchor_mean": float(control_packet.style_anchor_score.detach().float().mean().item()),
        }
        return soft_flat, stats, content_lat.to(dtype=shared_latent.dtype)

    def _make_generate_config(self) -> GenerateConfig:
        hard = int(self.hard_max_len) if getattr(self, "hard_max_len", None) else int(self.max_explanation_length)
        soft = int(self.soft_max_len) if getattr(self, "soft_max_len", None) not in (None, 0) else 0
        _nr = self.no_repeat_ngram_size
        nrs = int(_nr) if _nr is not None and int(_nr) > 0 else 0
        _mn = self.min_len
        min_l = int(_mn) if _mn is not None and int(_mn) > 0 else 0
        bad = tuple(getattr(self, "bad_terminal_token_ids_resolved", ()) or ())
        return GenerateConfig(
            strategy=str(self.decode_strategy).lower(),
            temperature=float(self.generate_temperature),
            top_p=float(self.generate_top_p),
            gap_threshold=float(getattr(self, "gap_threshold", 0.35)),
            prefix_greedy_steps=int(getattr(self, "prefix_greedy_steps", 4)),
            top_k=int(getattr(self, "decode_top_k", 5)),
            repetition_penalty=float(self.repetition_penalty),
            no_repeat_ngram_size=nrs,
            min_len=min_l,
            soft_max_len=soft,
            hard_max_len=max(1, hard),
            eos_boost_start=int(getattr(self, "eos_boost_start", 9999)),
            eos_boost_value=float(getattr(self, "eos_boost_value", 0.0)),
            tail_temperature=float(getattr(self, "tail_temperature", -1.0)),
            tail_top_p=float(getattr(self, "tail_top_p", -1.0)),
            forbid_eos_after_open_quote=bool(getattr(self, "forbid_eos_after_open_quote", True)),
            forbid_eos_after_open_bracket=bool(getattr(self, "forbid_eos_after_open_bracket", True)),
            forbid_bad_terminal_tokens=bool(getattr(self, "forbid_bad_terminal_tokens", True)),
            bad_terminal_token_ids=bad,
            token_repeat_window=int(getattr(self, "decode_token_repeat_window", 4)),
            token_repeat_max=int(getattr(self, "decode_token_repeat_max", 2)),
            decode_seed=self.decode_seed,
            uncertainty_entropy_eps=float(getattr(self, "uncertainty_entropy_eps", 1e-8)),
            decode_backend=str(getattr(self, "decode_backend", DECODE_BACKEND_KV_FAST)),
            decode_backend_fallback_policy=str(
                getattr(self, "decode_backend_fallback_policy", DECODE_BACKEND_FALLBACK_RAISE)
            ),
            decode_run_context=None,
        )

    def get_generate_kwargs_effective(self) -> Dict[str, Any]:
        """实际参与本模型手写 decode 循环的参数字典（供 metrics / 日志核对）。"""
        out: Dict[str, Any] = {
            "decode_strategy": self.decode_strategy,
            "max_explanation_length": self.max_explanation_length,
            "repetition_penalty": self.repetition_penalty,
            "generate_temperature": self.generate_temperature,
            "generate_top_p": self.generate_top_p,
        }
        if self.decode_seed is not None:
            out["decode_seed"] = self.decode_seed
        if self.decoder_eos_id >= 0:
            out["eos_token_id"] = self.decoder_eos_id
        if self.no_repeat_ngram_size is not None:
            out["no_repeat_ngram_size"] = self.no_repeat_ngram_size
        if self.min_len is not None:
            out["min_length"] = self.min_len
        return out

    def get_generate_kwargs_effective_v2(self) -> Dict[str, Any]:
        return build_generate_kwargs_effective_v2(
            self._make_generate_config(),
            eos_token_id=int(self.decoder_eos_id),
        )

    def _position_encode_with_offset(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        pe = self.pos_encoder.pe[start_pos : start_pos + int(x.size(1))].transpose(0, 1).to(x.device, dtype=x.dtype)
        out = x + pe
        return self.pos_encoder.dropout(out)

    def _decode_with_controller_legacy(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator],
        *,
        track_logprobs: bool,
        cfg_override: Optional[GenerateConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
        gc = cfg_override if cfg_override is not None else self._make_generate_config()
        _policy = resolve_decode_backend_fallback_policy(
            gc.decode_backend_fallback_policy
            if gc.decode_backend_fallback_policy is not None
            else getattr(self, "decode_backend_fallback_policy", DECODE_BACKEND_FALLBACK_RAISE)
        )
        _backend = resolve_decode_backend_name(
            gc.decode_backend if gc.decode_backend is not None else getattr(self, "decode_backend", DECODE_BACKEND_KV_FAST)
        )
        _tte = str(getattr(gc, "decode_run_context", "") or "") == "train_time_eval"
        _sampling_ctx = {
            "train_time_eval": str(_tte).lower(),
            "backend": str(_backend),
            "policy": str(_policy),
        }
        bos_idx = 0
        device = user.device
        batch_size = int(user.shape[0])
        prefix = self._build_prefix(domain, user, item)
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device).fill_(bos_idx)
        eos_id = int(self.decoder_eos_id)
        gen = generator
        _stoch = str(gc.strategy).lower() in ("nucleus", "uncertainty_low_temp_top_k")
        if gen is None and self.decode_seed is not None and _stoch:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.decode_seed))
        total_entropies: List[torch.Tensor] = []
        max_steps = int(gc.hard_max_len)
        token_logprob_sum = torch.zeros(batch_size, device=device, dtype=torch.float32)
        token_count = torch.zeros(batch_size, device=device, dtype=torch.float32)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        recent: List[List[int]] = [[] for _ in range(batch_size)]
        attention_scores = None
        self._last_uncertainty_decode_stats = None
        u_post_prefix_decisions = 0
        u_trigger_count = 0
        u_trigger_entropy_sum = 0.0
        u_trigger_entropy_count = 0
        u_trigger_entropy_values: List[float] = []
        first_trig_step = [-1] * batch_size
        for _step in range(max_steps):
            if not bool(active.any()):
                break
            gen_so_far = int(decoder_input_ids.shape[1]) - 1
            word_feature = self.word_embeddings(decoder_input_ids)
            src = torch.cat([prefix, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)
            attn_mask = _domain_fusion_causal_mask(decoder_input_ids.shape[1], device, prefix_len=self._prefix_len())
            hidden, attention_scores = self.transformer_encoder(src=src, mask=attn_mask)
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
            if gc.forbid_eos_after_open_quote or gc.forbid_eos_after_open_bracket:
                texts = get_step5_tokenizer().batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)
                apply_unbalanced_delimiter_eos_mask(logits, eos_id=eos_id, decoded_texts=texts, cfg=gc)
            tail_ids = decoder_input_ids[:, -1]
            if gc.forbid_bad_terminal_tokens:
                forbid_eos_if_bad_tail_token(
                    logits,
                    eos_id=eos_id,
                    tail_token_ids=tail_ids,
                    bad_ids=gc.bad_terminal_token_ids,
                )
            eff_t, eff_p = apply_sampling_schedule(gc, gen_so_far)
            apply_eos_boost(logits, eos_id=eos_id, step=gen_so_far, cfg=gc)
            output_id, ent_step, log_probs, udiag = sample_next_token(
                logits,
                strategy=gc.strategy,
                temperature=eff_t,
                top_p=eff_p,
                generator=gen,
                gen_so_far=gen_so_far,
                gap_threshold=float(gc.gap_threshold),
                prefix_greedy_steps=int(gc.prefix_greedy_steps),
                top_k=int(gc.top_k),
                row_active=active,
                uncertainty_entropy_eps=float(gc.uncertainty_entropy_eps),
                sampling_diag_context=_sampling_ctx,
            )
            if udiag is not None:
                u_post_prefix_decisions += int(udiag["post_prefix_decisions"])
                u_trigger_count += int(udiag["trigger_count"])
                tm = udiag["trigger_mask"]
                if bool(tm.any()):
                    ev = ent_step[tm].detach().float().reshape(-1).tolist()
                    for vx in ev:
                        fv = float(vx)
                        u_trigger_entropy_values.append(fv)
                        u_trigger_entropy_sum += fv
                    u_trigger_entropy_count += len(ev)
                for b in range(batch_size):
                    if bool(tm[b].item()) and first_trig_step[b] < 0:
                        first_trig_step[b] = int(gen_so_far)
            total_entropies.append(ent_step)
            if track_logprobs:
                chosen_lp = log_probs.gather(1, output_id).squeeze(-1)
                token_logprob_sum = token_logprob_sum + chosen_lp * active.to(dtype=torch.float32)
                token_count = token_count + active.to(dtype=torch.float32)
            for b in range(batch_size):
                recent[b].append(int(output_id[b, 0].item()))
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
            if eos_id >= 0:
                active = active & (output_id.squeeze(-1) != eos_id)
                if not bool(active.any()):
                    break
        stacked = (
            torch.stack(total_entropies).mean(dim=0)
            if total_entropies
            else torch.zeros(batch_size, device=device)
        )
        avg_lp: Optional[torch.Tensor]
        if track_logprobs:
            avg_lp = token_logprob_sum / token_count.clamp(min=1.0)
        else:
            avg_lp = None
        if str(gc.strategy).lower() == "uncertainty_low_temp_top_k":
            fts = [first_trig_step[b] for b in range(batch_size) if first_trig_step[b] >= 0]
            self._last_uncertainty_decode_stats = {
                "total_decision_count": u_post_prefix_decisions,
                "trigger_count": u_trigger_count,
                "first_trigger_steps": fts,
                "trigger_entropy_sum": float(u_trigger_entropy_sum),
                "trigger_entropy_count": int(u_trigger_entropy_count),
                "trigger_entropy_values": u_trigger_entropy_values,
            }
        else:
            self._last_uncertainty_decode_stats = None
        return decoder_input_ids[:, 1:], stacked, attention_scores, avg_lp

    def _decode_with_controller_kv(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator],
        *,
        track_logprobs: bool,
        sdpa_variant: str,
        resolved_backend: str,
        train_time_eval: bool,
        cfg_override: Optional[GenerateConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
        gc = cfg_override if cfg_override is not None else self._make_generate_config()
        _policy_kv = resolve_decode_backend_fallback_policy(
            gc.decode_backend_fallback_policy
            if gc.decode_backend_fallback_policy is not None
            else getattr(self, "decode_backend_fallback_policy", DECODE_BACKEND_FALLBACK_RAISE)
        )
        _sampling_ctx = {
            "train_time_eval": str(train_time_eval).lower(),
            "backend": str(resolved_backend),
            "policy": str(_policy_kv),
        }
        _lg_kv = logging.getLogger(LOGGER_NAME)
        _lg_kv.info(
            "[DecodeBackend] kv_decode_start backend=%s sdpa_variant=%s train_time_eval=%s policy=%s",
            resolved_backend,
            str(sdpa_variant),
            str(train_time_eval).lower(),
            _policy_kv,
        )
        bos_idx = 0
        device = user.device
        batch_size = int(user.shape[0])
        prefix = self._build_prefix(domain, user, item)
        prefix = self._position_encode_with_offset(prefix * math.sqrt(self.emsize), start_pos=0)
        kv_backend = DecoderKVBackend(self, sdpa_variant=str(sdpa_variant))
        cache: PastKeyValues = kv_backend.prefill(prefix)
        decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device).fill_(bos_idx)
        eos_id = int(self.decoder_eos_id)
        gen = generator
        _stoch = str(gc.strategy).lower() in ("nucleus", "uncertainty_low_temp_top_k")
        if gen is None and self.decode_seed is not None and _stoch:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.decode_seed))
        total_entropies: List[torch.Tensor] = []
        max_steps = int(gc.hard_max_len)
        token_logprob_sum = torch.zeros(batch_size, device=device, dtype=torch.float32)
        token_count = torch.zeros(batch_size, device=device, dtype=torch.float32)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        recent: List[List[int]] = [[] for _ in range(batch_size)]
        self._last_uncertainty_decode_stats = None
        u_post_prefix_decisions = 0
        u_trigger_count = 0
        u_trigger_entropy_sum = 0.0
        u_trigger_entropy_count = 0
        u_trigger_entropy_values: List[float] = []
        first_trig_step = [-1] * batch_size
        for _step in range(max_steps):
            if not bool(active.any()):
                break
            gen_so_far = int(decoder_input_ids.shape[1]) - 1
            last_token = decoder_input_ids[:, -1:]
            pos_idx = self._prefix_len() + gen_so_far

            def _embed_token_fn(tok: torch.Tensor) -> torch.Tensor:
                raw = self.word_embeddings(tok) * math.sqrt(self.emsize)
                return self._position_encode_with_offset(raw, start_pos=pos_idx)

            def _hidden_to_logits_fn(h: torch.Tensor) -> torch.Tensor:
                return prepare_logits(h[:, -1, :], self.hidden2token)

            step_out = kv_backend.decode_step(
                last_token,
                cache,
                embed_token_fn=_embed_token_fn,
                hidden_to_logits_fn=_hidden_to_logits_fn,
            )
            cache = step_out.past_key_values if step_out.past_key_values is not None else cache
            logits = step_out.logits
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
            if gc.forbid_eos_after_open_quote or gc.forbid_eos_after_open_bracket:
                texts = get_step5_tokenizer().batch_decode(decoder_input_ids[:, 1:], skip_special_tokens=True)
                apply_unbalanced_delimiter_eos_mask(logits, eos_id=eos_id, decoded_texts=texts, cfg=gc)
            tail_ids = decoder_input_ids[:, -1]
            if gc.forbid_bad_terminal_tokens:
                forbid_eos_if_bad_tail_token(
                    logits,
                    eos_id=eos_id,
                    tail_token_ids=tail_ids,
                    bad_ids=gc.bad_terminal_token_ids,
                )
            eff_t, eff_p = apply_sampling_schedule(gc, gen_so_far)
            apply_eos_boost(logits, eos_id=eos_id, step=gen_so_far, cfg=gc)
            output_id, ent_step, log_probs, udiag = sample_next_token(
                logits,
                strategy=gc.strategy,
                temperature=eff_t,
                top_p=eff_p,
                generator=gen,
                gen_so_far=gen_so_far,
                gap_threshold=float(gc.gap_threshold),
                prefix_greedy_steps=int(gc.prefix_greedy_steps),
                top_k=int(gc.top_k),
                row_active=active,
                uncertainty_entropy_eps=float(gc.uncertainty_entropy_eps),
                sampling_diag_context=_sampling_ctx,
            )
            if udiag is not None:
                u_post_prefix_decisions += int(udiag["post_prefix_decisions"])
                u_trigger_count += int(udiag["trigger_count"])
                tm = udiag["trigger_mask"]
                if bool(tm.any()):
                    ev = ent_step[tm].detach().float().reshape(-1).tolist()
                    for vx in ev:
                        fv = float(vx)
                        u_trigger_entropy_values.append(fv)
                        u_trigger_entropy_sum += fv
                    u_trigger_entropy_count += len(ev)
                for b in range(batch_size):
                    if bool(tm[b].item()) and first_trig_step[b] < 0:
                        first_trig_step[b] = int(gen_so_far)
            total_entropies.append(ent_step)
            if track_logprobs:
                chosen_lp = log_probs.gather(1, output_id).squeeze(-1)
                token_logprob_sum = token_logprob_sum + chosen_lp * active.to(dtype=torch.float32)
                token_count = token_count + active.to(dtype=torch.float32)
            for b in range(batch_size):
                recent[b].append(int(output_id[b, 0].item()))
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
            if eos_id >= 0:
                active = active & (output_id.squeeze(-1) != eos_id)
                if not bool(active.any()):
                    break
        stacked = (
            torch.stack(total_entropies).mean(dim=0)
            if total_entropies
            else torch.zeros(batch_size, device=device)
        )
        avg_lp: Optional[torch.Tensor]
        if track_logprobs:
            avg_lp = token_logprob_sum / token_count.clamp(min=1.0)
        else:
            avg_lp = None
        if str(gc.strategy).lower() == "uncertainty_low_temp_top_k":
            fts = [first_trig_step[b] for b in range(batch_size) if first_trig_step[b] >= 0]
            self._last_uncertainty_decode_stats = {
                "total_decision_count": u_post_prefix_decisions,
                "trigger_count": u_trigger_count,
                "first_trigger_steps": fts,
                "trigger_entropy_sum": float(u_trigger_entropy_sum),
                "trigger_entropy_count": int(u_trigger_entropy_count),
                "trigger_entropy_values": u_trigger_entropy_values,
            }
        else:
            self._last_uncertainty_decode_stats = None
        return decoder_input_ids[:, 1:], stacked, None, avg_lp

    def _decode_with_controller(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator],
        *,
        track_logprobs: bool,
        cfg_override: Optional[GenerateConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
        gc = cfg_override if cfg_override is not None else self._make_generate_config()
        policy = resolve_decode_backend_fallback_policy(
            gc.decode_backend_fallback_policy
            if gc.decode_backend_fallback_policy is not None
            else getattr(self, "decode_backend_fallback_policy", DECODE_BACKEND_FALLBACK_RAISE)
        )
        backend = resolve_decode_backend_name(
            gc.decode_backend if gc.decode_backend is not None else getattr(self, "decode_backend", DECODE_BACKEND_KV_FAST)
        )
        train_time_eval = str(getattr(gc, "decode_run_context", "") or "") == "train_time_eval"
        lg = logging.getLogger(LOGGER_NAME)
        _ref = user
        _dev = getattr(_ref, "device", None)
        _dtype = getattr(_ref, "dtype", None)

        if decode_backend_uses_kv_cache(backend):
            sdpa_variant = "safe" if backend == DECODE_BACKEND_KV_SAFE else "fast"
            try:
                return self._decode_with_controller_kv(
                    user,
                    item,
                    domain,
                    generator,
                    track_logprobs=track_logprobs,
                    sdpa_variant=sdpa_variant,
                    resolved_backend=backend,
                    train_time_eval=train_time_eval,
                    cfg_override=cfg_override,
                )
            except Exception as e:
                blocked = decode_exception_blocks_fallback(e)
                if policy == DECODE_BACKEND_FALLBACK_RAISE or blocked:
                    lg.error(
                        "[DecodeBackend] kv decode failed: backend=%s sdpa_variant=%s policy=%s "
                        "train_time_eval=%s ref_dtype=%s ref_device=%s err=%s",
                        backend,
                        sdpa_variant,
                        policy,
                        str(train_time_eval).lower(),
                        str(_dtype),
                        str(_dev),
                        str(e),
                        exc_info=True,
                    )
                    raise
                if policy == DECODE_BACKEND_FALLBACK_SYNC_THEN_FALLBACK:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if decode_exception_blocks_fallback(e):
                        lg.error(
                            "[DecodeBackend] kv decode failed after sync (fatal class): backend=%s policy=%s err=%s",
                            backend,
                            policy,
                            str(e),
                            exc_info=True,
                        )
                        raise
                    lg.error(
                        "[DecodeBackend] kv decode failed -> sync_then_fallback legacy: backend=%s sdpa_variant=%s "
                        "train_time_eval=%s err=%s",
                        backend,
                        sdpa_variant,
                        str(train_time_eval).lower(),
                        str(e),
                        exc_info=True,
                    )
                    return self._decode_with_controller_legacy(
                        user,
                        item,
                        domain,
                        generator,
                        track_logprobs=track_logprobs,
                        cfg_override=cfg_override,
                    )
                raise

        return self._decode_with_controller_legacy(
            user,
            item,
            domain,
            generator,
            track_logprobs=track_logprobs,
            cfg_override=cfg_override,
        )

    def forward(
        self,
        user,
        item,
        tgt_input,
        domain_idx,
        *,
        target_tokens: Optional[torch.Tensor] = None,
        evidence_features: Optional[torch.Tensor] = None,
        content_anchor_score: Optional[torch.Tensor] = None,
        style_anchor_score: Optional[torch.Tensor] = None,
        ccv_control_packet: Optional[CCVControlPacket] = None,
    ):
        device = user.device
        bsz = int(user.shape[0])
        if evidence_features is None:
            evidence_features = torch.zeros(bsz, STEP5_EVIDENCE_FEATURE_DIM, device=device, dtype=torch.float32)
        if content_anchor_score is None:
            content_anchor_score = torch.full((bsz,), 0.5, device=device, dtype=torch.float32)
        if style_anchor_score is None:
            style_anchor_score = torch.full((bsz,), 0.5, device=device, dtype=torch.float32)
        if target_tokens is None:
            raise RuntimeError("Step5B Flan 主链须传入 target_tokens（非 shift 的 target token ids）。")
        if tuple(target_tokens.shape) != tuple(tgt_input.shape):
            raise RuntimeError(
                "target_tokens 与 tgt_input 形状须一致；Flan decoder_input_ids 应为 shift_right(target_tokens)。"
            )
        prefix = self._build_prefix(domain_idx, user, item)
        word_feature = self.word_embeddings(tgt_input)
        src = torch.cat([prefix, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = _domain_fusion_causal_mask(tgt_input.shape[1], device, prefix_len=self._prefix_len())
        hidden, _ = self.transformer_encoder(src=src, mask=attn_mask)
        shared_latent = hidden[:, : self._prefix_len(), :].mean(dim=1)
        specific_latent = hidden[:, self._prefix_len() :, :].mean(dim=1)
        content_profile = self.user_content_profiles[user]
        rating = self.odcr_scorer(shared_latent, content_profile, specific_latent)
        ft = evidence_features.to(dtype=shared_latent.dtype)
        soft_flat, ccv_stats, content_evidence_latent = self._build_ccv_soft_prompt(
            shared_latent, specific_latent, ft, ccv_control_packet
        )
        soft_embeds = soft_flat.view(bsz, self.flan_soft_len, self.flan_d_model).to(
            dtype=self._flan_autocast_dtype(), device=device
        )
        enc_mask = torch.ones(bsz, self.flan_soft_len, device=device, dtype=torch.long)
        # 不传 labels：避免 HF T5 内部再算一遍 CE（显存/算力）；训练 CE 仅走外层 per_sample_decoder_ce_from_logits。
        out = self.flan_explainer(
            inputs_embeds=soft_embeds,
            attention_mask=enc_mask,
            decoder_input_ids=tgt_input,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits.float()
        enc_last = out.encoder_last_hidden_state
        h_explain = _fca_explain_pool_from_encoder_hidden(enc_last, enc_mask)
        h_score_raw = self.odcr_scorer.last_hidden
        if h_score_raw is None:
            raise RuntimeError("ODCRScorer.last_hidden 为空。")
        self._last_h_explain = h_explain
        self._last_h_score = self.fca_score_align(h_score_raw)
        self._last_h_explain_aligned = self.fca_explain_align(h_explain)
        self._last_specific_latent = specific_latent
        self._last_shared_latent = shared_latent
        self._last_content_profile = content_profile
        self._last_content_evidence_latent = (
            content_evidence_latent if content_evidence_latent is not None else shared_latent * 0.0
        )
        self._last_ccv_control_stats = ccv_stats or {
            "ccv_route_scorer_mean": 0.0,
            "ccv_route_explainer_mean": 0.0,
            "ccv_uncertainty_mean": 0.0,
            "ccv_confidence_mean": 0.0,
            "ccv_content_anchor_mean": 0.0,
            "ccv_style_anchor_mean": 0.0,
        }
        word_dist = logits
        context_dist = logits[:, 0, :] * 0.0
        return rating, context_dist, word_dist
    
    def gather(self, batch, device):
        if len(batch) != 20:
            raise ValueError(
                f"batch 须含 20 张量（含 UCI/anchor/evidence/CCV control），当前 {len(batch)}。"
                "请确认 DataLoader 与新版 Processor 一致。"
            )
        (
            user_idx,
            item_idx,
            rating,
            tgt_output,
            domain_idx,
            sample_id,
            exp_sample_weight,
            route_scorer_mask,
            route_explainer_mask,
            entropy_score,
            uncertainty_score,
            confidence_bucket,
            content_anchor_score,
            style_anchor_score,
            evidence_features,
            content_evidence_ids,
            style_evidence_ids,
            domain_style_anchor_ids,
            local_style_hint_ids,
            polarity_ids,
        ) = batch
        # 配合 DataLoader(pin_memory=True) 使用 non_blocking=True，减少同步拷贝等待
        user_idx = user_idx.to(device, non_blocking=True)
        item_idx = item_idx.to(device, non_blocking=True)
        domain_idx = domain_idx.to(device, non_blocking=True)
        rating = rating.to(device, non_blocking=True).float()
        tgt_output = tgt_output.to(device, non_blocking=True)
        sample_id = sample_id.to(device, non_blocking=True)
        exp_sample_weight = exp_sample_weight.to(device, non_blocking=True).float()
        route_scorer_mask = route_scorer_mask.to(device, non_blocking=True).float()
        route_explainer_mask = route_explainer_mask.to(device, non_blocking=True).float()
        entropy_score = entropy_score.to(device, non_blocking=True).float()
        uncertainty_score = uncertainty_score.to(device, non_blocking=True).float()
        confidence_bucket = confidence_bucket.to(device, non_blocking=True).float()
        content_anchor_score = content_anchor_score.to(device, non_blocking=True).float()
        style_anchor_score = style_anchor_score.to(device, non_blocking=True).float()
        evidence_features = evidence_features.to(device, non_blocking=True).float()
        content_evidence_ids = content_evidence_ids.to(device, non_blocking=True).long()
        style_evidence_ids = style_evidence_ids.to(device, non_blocking=True).long()
        domain_style_anchor_ids = domain_style_anchor_ids.to(device, non_blocking=True).long()
        local_style_hint_ids = local_style_hint_ids.to(device, non_blocking=True).long()
        polarity_ids = polarity_ids.to(device, non_blocking=True).long()
        tgt_input = T5_shift_right(tgt_output)
        return GatheredBatch(
            user_idx=user_idx,
            item_idx=item_idx,
            rating=rating,
            tgt_input=tgt_input,
            tgt_output=tgt_output,
            domain_idx=domain_idx,
            sample_id=sample_id,
            exp_sample_weight=exp_sample_weight,
            route_scorer_mask=route_scorer_mask,
            route_explainer_mask=route_explainer_mask,
            entropy_score=entropy_score,
            uncertainty_score=uncertainty_score,
            confidence_bucket=confidence_bucket,
            content_anchor_score=content_anchor_score,
            style_anchor_score=style_anchor_score,
            evidence_features=evidence_features,
            content_evidence_ids=content_evidence_ids,
            style_evidence_ids=style_evidence_ids,
            domain_style_anchor_ids=domain_style_anchor_ids,
            local_style_hint_ids=local_style_hint_ids,
            polarity_ids=polarity_ids,
        )

    def recommend(self, user, item, domain):
        src = self._build_prefix(domain, user, item)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, _ = self.transformer_encoder(src=src, mask=None)
        shared_latent = hidden[:, : self._prefix_len(), :].mean(dim=1)
        pl = int(hidden.size(1)) - int(self._prefix_len())
        if pl > 0:
            specific_latent = hidden[:, self._prefix_len() :, :].mean(dim=1)
        else:
            specific_latent = torch.zeros_like(shared_latent)
        content_profile = self.user_content_profiles[user]
        rating = self.odcr_scorer(shared_latent, content_profile, specific_latent)
        return rating

    def generate(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator] = None,
        *,
        cfg_override: Optional[Mapping[str, Any]] = None,
        ccv_control_packet: Optional[CCVControlPacket] = None,
    ):
        """Flan-T5-XL 解码主链：encoder 仅 soft prompt；与 scorer 前缀 Transformer 物理隔离。"""
        _base_gc = self._make_generate_config()
        gc = coerce_generate_cfg_override(_base_gc, cfg_override) or _base_gc
        device = user.device
        bsz = int(user.shape[0])
        prefix = self._build_prefix(domain, user, item)
        src = prefix * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden, _ = self.transformer_encoder(src=src, mask=None)
        shared_latent = hidden[:, : self._prefix_len(), :].mean(dim=1)
        ctx_tail = 6
        specific_latent = hidden[:, self._prefix_len() - ctx_tail : self._prefix_len(), :].mean(dim=1)
        ft = torch.zeros(bsz, STEP5_EVIDENCE_FEATURE_DIM, device=device, dtype=torch.float32)
        soft_flat, ccv_stats, _ = self._build_ccv_soft_prompt(
            shared_latent, specific_latent, ft, ccv_control_packet
        )
        self._last_ccv_control_stats = ccv_stats or {
            "ccv_route_scorer_mean": 0.0,
            "ccv_route_explainer_mean": 0.0,
            "ccv_uncertainty_mean": 0.0,
            "ccv_confidence_mean": 0.0,
            "ccv_content_anchor_mean": 0.0,
            "ccv_style_anchor_mean": 0.0,
        }
        soft_embeds = soft_flat.view(bsz, self.flan_soft_len, self.flan_d_model).to(
            dtype=self._flan_autocast_dtype(), device=device
        )
        enc_mask = torch.ones(bsz, self.flan_soft_len, device=device, dtype=torch.long)
        max_new = max(1, int(gc.hard_max_len))
        strat = str(gc.strategy).lower()
        do_sample = strat not in ("greedy",)
        gen_kwargs: Dict[str, Any] = {
            "inputs_embeds": soft_embeds,
            "attention_mask": enc_mask,
            "max_new_tokens": max_new,
            "repetition_penalty": float(gc.repetition_penalty),
            "use_cache": True,
        }
        if int(self.decoder_eos_id) >= 0:
            gen_kwargs["eos_token_id"] = int(self.decoder_eos_id)
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(gc.temperature)
            gen_kwargs["top_p"] = float(gc.top_p)
        else:
            gen_kwargs["do_sample"] = False
        gen = self.flan_explainer.generate(**gen_kwargs)
        ent = torch.zeros(bsz, device=device, dtype=torch.float32)
        return gen, ent, None

    def generate_with_token_logprobs(
        self,
        user,
        item,
        domain,
        generator: Optional[torch.Generator] = None,
        *,
        cfg_override: Optional[Union[GenerateConfig, Mapping[str, Any]]] = None,
        ccv_control_packet: Optional[CCVControlPacket] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        解码 + 每 token 选中类 logprob；平均得 avg_logprob（长度归一见 rerank v3 lp_norm）。
        cfg_override 可为 GenerateConfig 或 dict，仅本调用有效。
        """
        out, ent, attn = self.generate(
            user,
            item,
            domain,
            generator,
            cfg_override=cfg_override,
            ccv_control_packet=ccv_control_packet,
        )
        avg_lp = torch.zeros(int(out.size(0)), device=out.device, dtype=torch.float32)
        return out, ent, attn, avg_lp


def _per_sample_mean_ce(
    logits_bt: torch.Tensor,
    tgt: torch.Tensor,
    *,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    """(B,T,V) 与 (B,T) → 每样本对非 padding 位置的平均 CE（与全局 CE 的 label_smoothing 语义一致）。"""
    B, T, V = logits_bt.shape
    ce = F.cross_entropy(
        logits_bt.reshape(-1, V),
        tgt.reshape(-1).long(),
        ignore_index=ignore_index,
        label_smoothing=float(label_smoothing),
        reduction="none",
    ).view(B, T)
    mask = (tgt != ignore_index).to(dtype=ce.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (ce * mask).sum(dim=1) / denom


def _per_sample_mean_ce_bv_to_bt(
    logits_bv: torch.Tensor,
    tgt: torch.Tensor,
    *,
    ignore_index: int,
    label_smoothing: float,
) -> torch.Tensor:
    """
    context 分支 CE：输入 (B,V) + target (B,T)。
    通过广播 gather 计算每个 token 位点的 CE，并按非 pad 位点求样本均值；
    避免构造 (B,T,V) repeat 张量。
    """
    B, V = logits_bv.shape
    T = int(tgt.shape[1])
    logp = F.log_softmax(logits_bv, dim=-1)
    tg = tgt.long()
    gather_idx = tg.clamp(min=0).unsqueeze(-1)
    nll = -logp.unsqueeze(1).expand(B, T, V).gather(-1, gather_idx).squeeze(-1)
    if float(label_smoothing) > 0.0:
        smooth = -logp.mean(dim=-1, keepdim=True).expand(B, T)
        ce = (1.0 - float(label_smoothing)) * nll + float(label_smoothing) * smooth
    else:
        ce = nll
    ce = torch.where(tg == int(ignore_index), torch.zeros_like(ce), ce)
    mask = (tg != int(ignore_index)).to(dtype=ce.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (ce * mask).sum(dim=1) / denom


def graph_tied_zero(ref: torch.Tensor) -> torch.Tensor:
    """Scalar zero that keeps DDP's autograd graph connected to ``ref``."""
    return ref.sum() * 0.0


def graph_tied_zero_like(ref: torch.Tensor) -> torch.Tensor:
    """Elementwise zero with the same shape as ``ref`` and an autograd edge."""
    return ref * 0.0


def compose_step5_total_loss(
    *,
    loss_factual: torch.Tensor,
    loss_counterfactual: torch.Tensor,
    loss_repeat_ul: torch.Tensor,
    loss_terminal_clean: torch.Tensor,
    loss_batch_diversity: torch.Tensor,
    repeat_ul_weight: float,
    terminal_clean_weight: float,
    batch_diversity_weight: float,
    lci_weighted_loss: torch.Tensor,
    fca_weighted_loss: torch.Tensor,
    ortho_keep_loss: torch.Tensor,
    ortho_keep_weight: float,
) -> torch.Tensor:
    """Single Step5 train loss composer; LCI/FCA weighted terms enter exactly once."""
    loss = (
        loss_factual
        + loss_counterfactual
        + float(repeat_ul_weight) * loss_repeat_ul
        + float(terminal_clean_weight) * loss_terminal_clean
        + float(batch_diversity_weight) * loss_batch_diversity
        + lci_weighted_loss
        + fca_weighted_loss
    )
    if float(ortho_keep_weight) > 0.0:
        loss = loss + float(ortho_keep_weight) * ortho_keep_loss
    return loss


def _validate_route_masks_batch(
    route_scorer_mask: torch.Tensor,
    route_explainer_mask: torch.Tensor,
    *,
    batch_size: int,
    stage: str,
) -> None:
    rs = route_scorer_mask.view(-1)
    re = route_explainer_mask.view(-1)
    if int(rs.numel()) != int(batch_size) or int(re.numel()) != int(batch_size):
        raise RuntimeError(
            f"{stage} route 掩码维度不匹配: batch_size={batch_size}, "
            f"route_scorer={tuple(route_scorer_mask.shape)}, route_explainer={tuple(route_explainer_mask.shape)}"
        )
    if not bool(torch.isfinite(rs).all().item()) or not bool(torch.isfinite(re).all().item()):
        raise RuntimeError(f"{stage} route 掩码存在非有限值（NaN/Inf）。")


def _step5_synthetic_preflight_batch(*, model: nn.Module, final_cfg: FinalTrainingConfig) -> GatheredBatch:
    _model = get_underlying_model(model)
    device = next(_model.parameters()).device
    bsz = 4
    seq_len = max(2, min(6, int(getattr(final_cfg, "train_label_max_length", 6))))
    vocab_hi = max(2, min(int(getattr(_model, "ntoken", 32)), 32))
    target_tokens = (torch.arange(bsz * seq_len, device=device).view(bsz, seq_len) % (vocab_hi - 1)) + 1
    tgt_input = T5_shift_right(target_tokens)
    evidence = torch.zeros(bsz, STEP5_EVIDENCE_FEATURE_DIM, device=device, dtype=torch.float32)
    evidence[:, CF_RELIABILITY] = torch.tensor([0.95, 0.90, 0.65, 0.85], device=device)
    evidence[:, STYLE_SHIFT] = torch.tensor([0.10, 0.60, 0.20, 0.75], device=device)
    evidence[:, RATING_STABILITY] = torch.tensor([0.95, 0.85, 0.70, 0.90], device=device)
    evidence[:, CONTENT_RETENTION] = torch.tensor([0.95, 0.80, 0.70, 0.90], device=device)
    evidence[:, TEXT_QUALITY] = torch.ones(bsz, device=device)
    evidence[:, UNCERTAINTY] = torch.tensor([0.05, 0.15, 0.30, 0.10], device=device)
    evidence[:, EVIDENCE_QUALITY_PRIOR] = torch.ones(bsz, device=device)
    return GatheredBatch(
        user_idx=(torch.arange(bsz, device=device) % max(1, int(final_cfg.nuser))).long(),
        item_idx=(torch.arange(bsz, device=device) % max(1, int(final_cfg.nitem))).long(),
        rating=torch.tensor([4.5, 3.0, 2.0, 5.0], device=device, dtype=torch.float32),
        tgt_input=tgt_input.long(),
        tgt_output=target_tokens.long(),
        domain_idx=torch.tensor([1, 0, 1, 0], device=device, dtype=torch.long),
        sample_id=torch.arange(bsz, device=device, dtype=torch.long),
        exp_sample_weight=torch.ones(bsz, device=device, dtype=torch.float32),
        route_scorer_mask=torch.tensor([1.0, 0.0, 0.0, 1.0], device=device),
        route_explainer_mask=torch.tensor([0.0, 1.0, 0.0, 1.0], device=device),
        entropy_score=torch.zeros(bsz, device=device),
        uncertainty_score=evidence[:, UNCERTAINTY],
        confidence_bucket=torch.tensor([2.0, 2.0, 0.0, 1.0], device=device),
        content_anchor_score=torch.tensor([0.90, 0.80, 0.60, 0.85], device=device),
        style_anchor_score=torch.tensor([0.20, 0.75, 0.35, 0.80], device=device),
        evidence_features=evidence,
        content_evidence_ids=target_tokens[:, :seq_len].long(),
        style_evidence_ids=target_tokens.flip(1).long(),
        domain_style_anchor_ids=target_tokens.roll(shifts=1, dims=1).long(),
        local_style_hint_ids=target_tokens.roll(shifts=2, dims=1).long(),
        polarity_ids=torch.tensor([2, 1, 0, 2], device=device, dtype=torch.long),
    )


def run_step5_find_unused_parameters_preflight(
    model: nn.Module,
    final_cfg: FinalTrainingConfig,
    *,
    step5_innov_cfg: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Synthetic graph participation preflight required before disabling DDP unused-param scan."""
    if bool(getattr(final_cfg, "ddp_find_unused_parameters", True)):
        return
    policy = str(getattr(final_cfg, "ddp_find_unused_false_preflight", "") or "").strip().lower()
    if policy != "synthetic_one_batch":
        raise RuntimeError(
            "step5.ddp.find_unused_parameters=false requires "
            "step5.ddp.find_unused_false_preflight=synthetic_one_batch."
        )
    _model = get_underlying_model(model)
    was_training = bool(_model.training)
    _model.train()
    _model.zero_grad(set_to_none=True)
    batch = _step5_synthetic_preflight_batch(model=_model, final_cfg=final_cfg)
    gate_a = build_step5a_scorer_gate(batch, step5_innov_cfg)
    gate_b = build_step5b_explainer_gate(batch, step5_innov_cfg)
    packet = build_ccv_control_packet(batch, step5_innov_cfg)
    with torch.enable_grad():
        pred_rating, _context_dist, word_dist = _model(
            batch.user_idx,
            batch.item_idx,
            batch.tgt_input,
            batch.domain_idx,
            target_tokens=batch.tgt_output,
            evidence_features=batch.evidence_features,
            content_anchor_score=batch.content_anchor_score,
            style_anchor_score=batch.style_anchor_score,
            ccv_control_packet=packet,
        )
        loss_r_ps = F.mse_loss(pred_rating, batch.rating, reduction="none")
        loss_flan_ps = per_sample_decoder_ce_from_logits(
            word_dist,
            batch.tgt_output,
            ignore_index=0,
            label_smoothing=float(final_cfg.label_smoothing),
        )
        loss_c_ps = graph_tied_zero_like(pred_rating).to(dtype=loss_flan_ps.dtype)
        scorer_only = loss_r_ps
        explainer_only = float(final_cfg.coef) * loss_c_ps + loss_flan_ps
        dom = batch.domain_idx.view(-1)
        loss_factual = route_weighted_mean(
            scorer_only,
            gate_a.scorer_weight.to(dtype=scorer_only.dtype),
            (dom == 1).to(dtype=scorer_only.dtype),
        )
        loss_counterfactual = float(final_cfg.explainer_loss_weight) * route_weighted_mean(
            explainer_only,
            gate_b.explainer_weight.to(dtype=explainer_only.dtype),
            (dom == 0).to(dtype=explainer_only.dtype),
        )
        shared_lat = _model._last_shared_latent
        spec_lat = _model._last_specific_latent
        noise = float(step5_innov_cfg.lci.perturb_std) * torch.randn_like(spec_lat)
        score_pert = _model.odcr_scorer(shared_lat, _model.user_content_profiles[batch.user_idx], spec_lat + noise)
        score_robust = _model.odcr_scorer(
            shared_lat.detach() + 0.0 * shared_lat,
            _model.user_content_profiles[batch.user_idx],
            spec_lat + noise.flip(0),
        )
        lci_bundle = lci_score_invariance_loss(
            factual_score=pred_rating,
            cf_score=score_pert,
            robust_score=score_robust,
            target_rating=batch.rating,
            gate=gate_a,
            cfg=step5_innov_cfg,
        )
        fca_bundle = evidence_basis_fca_loss(
            scorer_hidden=_model._last_h_score,
            explainer_hidden=_model._last_h_explain_aligned,
            shared_latent=shared_lat,
            content_profile=_model._last_content_profile,
            content_evidence_latent=_model._last_content_evidence_latent,
            packet=packet,
            gate=gate_b,
            cfg=step5_innov_cfg,
        )
        loss_ortho_keep = graph_tied_zero(word_dist)
        if float(final_cfg.lambda_ortho_step5) > 0.0:
            loss_ortho_keep = build_orthogonal_losses(
                shared_lat,
                spec_lat,
                w_xcov=float(final_cfg.lambda_ortho_xcov),
                w_cos=float(final_cfg.lambda_ortho_cos),
            ).loss_ortho_total
        loss = compose_step5_total_loss(
            loss_factual=loss_factual,
            loss_counterfactual=loss_counterfactual,
            loss_repeat_ul=graph_tied_zero(word_dist),
            loss_terminal_clean=graph_tied_zero(word_dist),
            loss_batch_diversity=graph_tied_zero(word_dist),
            repeat_ul_weight=0.0,
            terminal_clean_weight=0.0,
            batch_diversity_weight=0.0,
            lci_weighted_loss=lci_bundle.lci_weighted_loss,
            fca_weighted_loss=fca_bundle.fca_weighted_loss,
            ortho_keep_loss=loss_ortho_keep,
            ortho_keep_weight=float(final_cfg.lambda_ortho_step5),
        )
        loss.backward()
    unused = [
        name
        for name, param in _model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    _model.zero_grad(set_to_none=True)
    _model.train(was_training)
    if unused:
        preview = ", ".join(unused[:20])
        more = "" if len(unused) <= 20 else f" ... (+{len(unused) - 20} more)"
        raise RuntimeError(
            "Step5 find_unused_parameters=false preflight failed; trainable params without grad: "
            f"{preview}{more}"
        )
    if logger is not None:
        logger.info(
            "[DDP preflight] find_unused_parameters=false synthetic graph participation passed",
            extra=log_route_extra(logger, ROUTE_SUMMARY),
        )


@contextlib.contextmanager
def _ddp_no_sync_model(model, world_size: int, sync_gradients: bool):
    """梯度累积非边界微批上使用 DDP no_sync。"""
    if world_size <= 1 or sync_gradients:
        yield
    else:
        with model.no_sync():
            yield


def odcr_profile_step_components_enabled() -> bool:
    """ODCR_PROFILE_STEP_COMPONENTS=1 时开启 Step5 微批热点计时（仅用于验证优化，默认关闭）。"""
    v = os.environ.get("ODCR_PROFILE_STEP_COMPONENTS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _checkpoint_composite_weights_dict(cfg: FinalTrainingConfig) -> Dict[str, float]:
    return {
        "w_bleu4": float(cfg.checkpoint_composite_w_bleu4),
        "w_rouge_l": float(cfg.checkpoint_composite_w_rouge_l),
        "w_meteor": float(cfg.checkpoint_composite_w_meteor),
        "w_dist1": float(cfg.checkpoint_composite_w_dist1),
        "w_dist2": float(cfg.checkpoint_composite_w_dist2),
        "w_dirty": float(cfg.checkpoint_composite_w_dirty),
    }


class _StepComponentCudaTimer:
    """rank0 调试：可选 CUDA Event 计时；关闭时方法为空操作，无额外同步。"""

    def __init__(self, active: bool, *, use_cuda_events: bool):
        self.active = active
        self.use_cuda = bool(active and use_cuda_events)
        self._starts: Dict[str, Any] = {}
        self._ends: Dict[str, Any] = {}

    def start(self, name: str) -> None:
        if not self.active:
            return
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._starts[name] = ev
        else:
            self._starts[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not self.active:
            return
        if self.use_cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._ends[name] = ev
        else:
            self._ends[name] = time.perf_counter()

    def ms(self, name: str) -> float:
        if not self.active:
            return 0.0
        s, e = self._starts.get(name), self._ends.get(name)
        if s is None or e is None:
            return 0.0
        if self.use_cuda:
            e.synchronize()
            return float(s.elapsed_time(e))
        return float((e - s) * 1000.0)


def _fmt_opt_float(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(fv):
        return "nan"
    return f"{fv:.6g}"


def trainModel_ddp(
    model,
    train_dataloader,
    valid_dataloader,
    sampler,
    valid_sampler,
    final_cfg: FinalTrainingConfig,
    rank,
    world_size,
    step5_collate_fn=None,
):
    epochs = final_cfg.epochs
    G = int(final_cfg.train_batch_size)
    P = int(final_cfg.per_device_train_batch_size)
    A = max(1, int(final_cfg.gradient_accumulation_steps))
    eff = int(final_cfg.effective_global_batch_size)
    initial_lr = float(final_cfg.scheduler_initial_lr)
    learning_rate = initial_lr
    coef = float(final_cfg.coef)
    explainer_loss_weight = float(final_cfg.explainer_loss_weight)
    _model = get_underlying_model(model)
    device = final_cfg.device
    use_bf16 = odcr_cuda_bf16_autocast_enabled()
    n_micro = len(train_dataloader)
    n_steps = max(1, n_micro // A)
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
        _ckm = str(getattr(final_cfg, "checkpoint_selection_mode", "guarded_composite")).strip().lower()
        if _lg:
            _lg.info(
                "Train profile: lr_scheduler=%s warmup_epochs=%g checkpoint_selection_mode=%s",
                lr_scheduler_type,
                warmup_epochs,
                _ckm,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
            _lg.info(
                "[CheckpointPolicy] checkpoint_metric=%s | selection_mode=%s",
                checkpoint_metric,
                _ckm,
                extra=log_route_extra(_lg, ROUTE_SUMMARY),
            )
            _lg.info(
                "[ValidWeighting] aligned_with_train_main_loss=True (sample_weight/exp_sample_weight enabled)",
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
    scheduler_steps = 0
    global_step = 0
    step_iv = max(1, odcr_log_step_interval())
    profile_step_iv = max(100, step_iv * 20)
    grad_iv = max(1, odcr_log_grad_interval())
    _finite_mode, _finite_warn = parse_odcr_finite_check_mode()
    micro_step_count = 0
    lambda_ortho_step5 = float(getattr(final_cfg, "lambda_ortho_step5", 0.2))
    orth5_w_xcov = float(getattr(final_cfg, "lambda_ortho_xcov", 1.0))
    orth5_w_cos = float(getattr(final_cfg, "lambda_ortho_cos", 0.25))
    _st5_final_json = str(getattr(final_cfg, "step5_innovation_config_json", "") or "").strip()
    if not _st5_final_json:
        raise RuntimeError("Step5 training requires resolved step5_innovation_config_json from One-Control.")
    step5_innov_cfg = parse_step5_innovation_config_json(_st5_final_json)
    w_lci = float(step5_innov_cfg.lci.weight)
    w_fca = float(step5_innov_cfg.fca.weight)
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
            "[DDP] ddp_find_unused_parameters=%s（true=安全默认；false=吞吐向，须图稳定）",
            bool(final_cfg.ddp_find_unused_parameters),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[DDP] epoch_end_barrier=%s（ODCR_DDP_EPOCH_END_BARRIER；默认关闭）",
            str(odcr_ddp_epoch_end_barrier()).lower(),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[Diag] ODCR_GRAD_TOPK=%d（仅 >0 时在 GradClip 路径打印 top 参数 grad norm）",
            odcr_grad_topk(),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
        _lg.info(
            "[Step5][A/B] lambda_ortho_step5=%g lambda_ortho_xcov=%g lambda_ortho_cos=%g "
            "lci_enabled=%s lci_weight=%g uci_enabled=%s ccv_enabled=%s fca_enabled=%s fca_weight=%g "
            "explainer_loss_weight=%g explainer_only_multiplier=%g",
            lambda_ortho_step5,
            orth5_w_xcov,
            orth5_w_cos,
            bool(step5_innov_cfg.lci.enabled),
            w_lci,
            bool(step5_innov_cfg.uci.enabled),
            bool(step5_innov_cfg.ccv.enabled),
            bool(step5_innov_cfg.fca.enabled),
            w_fca,
            explainer_loss_weight,
            float(step5_innov_cfg.explainer_gate.explainer_only_multiplier),
            extra=log_route_extra(_lg, ROUTE_SUMMARY),
        )
    save_last_checkpoint = False
    last_epoch_completed = 0
    last_ckpt_meta: Dict[str, Any] = {
        "valid_loss_total": float("nan"),
        "valid_loss_r": float("nan"),
        "valid_loss_c": float("nan"),
        "valid_loss_e": float("nan"),
        "quick_bleu4": None,
        "full_bleu4": None,
        "mainline_composite": None,
    }
    best_version = 0
    ckpt_mode = str(getattr(final_cfg, "checkpoint_selection_mode", "guarded_composite")).strip().lower()
    best_vl_ever = float("inf")
    best_composite_saved = -1e9
    best_monitor_bundle_gate: Optional[Dict[str, Any]] = None
    shadow_strict_best_vl = float("inf")
    shadow_strict_best_epoch = 0
    try:
        for epoch in range(epochs):
            epoch_1 = epoch + 1
            last_epoch_completed = epoch_1
            _epoch_loss_bd_sum = 0.0
            _epoch_loss_bd_wsum = 0.0
            _epoch_loss_bd_batches = 0
            _epoch_loss_bd_raw_sum = 0.0
            _epoch_bd_skip_batches = 0
            _epoch_bd_ema_batches = 0
            _epoch_orth_sum = 0.0
            _epoch_orth_batches = 0
            epoch_metrics_row: Optional[Dict[str, Any]] = None
            _bd_warm = int(getattr(final_cfg, "batch_diversity_warmup_epochs", 0))
            _bd_ramp_ep = int(getattr(final_cfg, "batch_diversity_ramp_epochs", 2))
            _bd_ramp_tgt = float(getattr(final_cfg, "batch_diversity_ramp_target_scale", 1.0))
            _w_bd_base = float(getattr(final_cfg, "loss_weight_batch_diversity", 0.0))
            if epoch_1 <= _bd_warm:
                _epoch_bd_ramp_factor = 0.0
            else:
                _epoch_bd_ramp_factor = min(
                    1.0,
                    max(0.0, float(epoch_1 - _bd_warm - 1) / float(max(1, _bd_ramp_ep))),
                )
            _epoch_bd_eff_weight = _w_bd_base * _epoch_bd_ramp_factor * _bd_ramp_tgt
            sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            if rank == 0:
                perf.epoch_start()
            model.train()
            loss_sum = torch.zeros((), dtype=torch.double, device=device)
            loss_r_sum = torch.zeros((), dtype=torch.double, device=device)
            loss_c_sum = torch.zeros((), dtype=torch.double, device=device)
            loss_e_sum = torch.zeros((), dtype=torch.double, device=device)
            n_samples = torch.zeros((), dtype=torch.double, device=device)
            micro_step_epoch = 0
            optimizer.zero_grad(set_to_none=True)
            inv_accum = 1.0 / float(A)
            iterator = train_dataloader
            if rank == 0:
                iterator = tqdm(train_dataloader, total=len(train_dataloader))
            for batch in iterator:
                micro_step_epoch += 1
                micro_step_count += 1
                sync = micro_step_epoch % A == 0
                sync_ctx = _ddp_no_sync_model(model, world_size, sync)
                gb = require_gathered_batch(_model.gather(batch, device))
                user_idx = gb.user_idx
                item_idx = gb.item_idx
                rating = gb.rating
                tgt_input = gb.tgt_input
                tgt_output = gb.tgt_output
                domain_idx = gb.domain_idx
                exp_w = gb.exp_sample_weight
                route_scorer_mask = gb.route_scorer_mask
                route_explainer_mask = gb.route_explainer_mask
                if exp_w is None:
                    raise RuntimeError(
                        "Step5 训练 batch 缺少 exp_sample_weight：请确认 DataLoader 与 Processor 输出完整张量。"
                    )
                if route_scorer_mask is None or route_explainer_mask is None:
                    raise RuntimeError("Step5 训练 batch 缺少 route_scorer/route_explainer 掩码。")
                entropy_s = gb.entropy_score
                uncertainty_s = gb.uncertainty_score
                conf_bucket = gb.confidence_bucket
                evidence_f = gb.evidence_features
                c_anchor = gb.content_anchor_score
                s_anchor = gb.style_anchor_score
                step5a_gate = build_step5a_scorer_gate(gb, step5_innov_cfg)
                step5b_gate = build_step5b_explainer_gate(gb, step5_innov_cfg)
                ccv_packet = build_ccv_control_packet(gb, step5_innov_cfg)
                if (
                    entropy_s is None
                    or uncertainty_s is None
                    or conf_bucket is None
                    or evidence_f is None
                    or c_anchor is None
                    or s_anchor is None
                ):
                    raise RuntimeError("Step5 batch 缺少 UCI / evidence / anchor 张量。")
                bsz = int(user_idx.size(0))
                _validate_route_masks_batch(
                    route_scorer_mask,
                    route_explainer_mask,
                    batch_size=bsz,
                    stage="train",
                )
                warn_empty_batch(_lg, global_step=global_step, epoch=epoch_1, bsz=bsz)
                _do_step_profile = (
                    rank == 0
                    and odcr_profile_step_components_enabled()
                    and (micro_step_count % profile_step_iv == 0)
                )
                _use_cuda_prof = _do_step_profile and torch.cuda.is_available()
                _step_timer = _StepComponentCudaTimer(_do_step_profile, use_cuda_events=_use_cuda_prof)
                with sync_ctx:
                    with odcr_cuda_bf16_autocast():
                        _step_timer.start("forward")
                        pred_rating, context_dist, word_dist = model(
                            user_idx,
                            item_idx,
                            tgt_input,
                            domain_idx,
                            target_tokens=tgt_output,
                            evidence_features=evidence_f,
                            content_anchor_score=c_anchor,
                            style_anchor_score=s_anchor,
                            ccv_control_packet=ccv_packet,
                        )
                        _step_timer.end("forward")
                        word_logp = F.log_softmax(word_dist, dim=-1)
                        _step_timer.start("exp_ce")
                        ls = float(final_cfg.label_smoothing)
                        loss_r_ps = F.mse_loss(pred_rating, rating, reduction="none")
                        loss_flan_ps = per_sample_decoder_ce_from_logits(
                            word_dist, tgt_output, ignore_index=0, label_smoothing=ls
                        )
                        loss_c_ps = graph_tied_zero_like(pred_rating).to(dtype=loss_flan_ps.dtype)
                        loss_e_ps = loss_flan_ps
                        scorer_only = loss_r_ps
                        explainer_only = coef * loss_c_ps + loss_e_ps

                        dom = domain_idx.view(-1)
                        w = exp_w.view(-1)
                        rs = route_scorer_mask.view(-1)
                        re = route_explainer_mask.view(-1)
                        f_mask = (dom == 1).to(dtype=scorer_only.dtype)
                        c_mask = (dom == 0).to(dtype=scorer_only.dtype)
                        scorer_w = step5a_gate.scorer_weight.to(dtype=w.dtype)
                        explainer_w = step5b_gate.explainer_weight.to(dtype=w.dtype)
                        loss_factual = route_weighted_mean(scorer_only, scorer_w, f_mask)
                        loss_counterfactual = explainer_loss_weight * route_weighted_mean(
                            explainer_only,
                            explainer_w,
                            c_mask,
                        )
                        spec_lat = _model._last_specific_latent
                        shared_lat = _model._last_shared_latent
                        noise = float(step5_innov_cfg.lci.perturb_std) * torch.randn_like(spec_lat)
                        score_pert = _model.odcr_scorer(
                            shared_lat, _model.user_content_profiles[user_idx], spec_lat + noise
                        )
                        score_robust = _model.odcr_scorer(
                            shared_lat.detach() + 0.0 * shared_lat,
                            _model.user_content_profiles[user_idx],
                            spec_lat + noise.flip(0),
                        )
                        lci_bundle = lci_score_invariance_loss(
                            factual_score=pred_rating,
                            cf_score=score_pert,
                            robust_score=score_robust,
                            target_rating=rating,
                            gate=step5a_gate,
                            cfg=step5_innov_cfg,
                        )
                        fca_bundle = evidence_basis_fca_loss(
                            scorer_hidden=_model._last_h_score,
                            explainer_hidden=_model._last_h_explain_aligned,
                            shared_latent=shared_lat,
                            content_profile=_model._last_content_profile,
                            content_evidence_latent=_model._last_content_evidence_latent,
                            packet=ccv_packet,
                            gate=step5b_gate,
                            cfg=step5_innov_cfg,
                        )
                        l_lci = lci_bundle.lci_loss
                        l_fca = fca_bundle.fca_loss
                        loss_ortho_keep = word_dist.sum() * 0.0
                        loss_ortho_xcov_log = 0.0
                        loss_ortho_cos_log = 0.0
                        if lambda_ortho_step5 > 0.0 and bsz > 0:
                            _ob = build_orthogonal_losses(
                                shared_lat,
                                spec_lat,
                                w_xcov=orth5_w_xcov,
                                w_cos=orth5_w_cos,
                            )
                            loss_ortho_keep = _ob.loss_ortho_total
                            loss_ortho_xcov_log = float(_ob.loss_ortho_xcov.detach().item())
                            loss_ortho_cos_log = float(_ob.loss_ortho_cos.detach().item())
                            _epoch_orth_sum += float(_ob.loss_ortho_total.detach().item())
                            _epoch_orth_batches += 1
                        elif lambda_ortho_step5 > 0.0 and bsz == 0:
                            loss_ortho_keep = word_dist.sum() * 0.0
                        _step_timer.end("exp_ce")
                        w_ul = float(final_cfg.loss_weight_repeat_ul)
                        w_tc = float(final_cfg.loss_weight_terminal_clean)
                        loss_ul = graph_tied_zero(word_dist)
                        loss_tc = graph_tied_zero(word_dist)
                        _step_timer.start("repeat_ul")
                        if w_ul > 0:
                            loss_ul = odcr_anti_repeat_unlikelihood_loss_from_logp(
                                word_logp, tgt_output
                            )
                        _step_timer.end("repeat_ul")
                        _step_timer.start("terminal_clean")
                        if w_tc > 0:
                            loss_tc = odcr_terminal_cleanliness_loss(
                                word_dist,
                                tgt_output,
                                list(_model.bad_terminal_token_ids_resolved),
                                int(final_cfg.terminal_clean_span),
                            )
                        _step_timer.end("terminal_clean")
                        loss_bd_raw_preclamp = graph_tied_zero(word_dist)
                        batch_div_skipped = False
                        batch_div_valid_tokens = 0
                        w_bd_eff = float(_epoch_bd_eff_weight)
                        warm_bd = int(_bd_warm)
                        mode_bd = str(getattr(final_cfg, "batch_diversity_mode", "mean_prob_neg_entropy")).strip().lower()
                        eps_bd = float(getattr(final_cfg, "batch_diversity_eps", 1e-8))
                        use_ema_bd = bool(getattr(final_cfg, "batch_diversity_use_ema", True))
                        ema_dec_bd = float(getattr(final_cfg, "batch_diversity_ema_decay", 0.9))
                        ema_init_mode = str(
                            getattr(final_cfg, "batch_diversity_ema_init_mode", "uniform")
                        ).strip().lower()
                        min_tok_bd = int(getattr(final_cfg, "batch_diversity_min_valid_tokens", 64))
                        clamp_bd = float(getattr(final_cfg, "batch_diversity_loss_clamp_abs", 0.2))
                        loss_bd = graph_tied_zero(word_dist)
                        if (
                            _w_bd_base > 0.0
                            and float(_epoch_bd_ramp_factor) > 0.0
                            and int(epoch_1) > warm_bd
                            and mode_bd == "mean_prob_neg_entropy"
                        ):
                            pad_id = 0
                            m = (tgt_output != pad_id).float().unsqueeze(-1)
                            batch_div_valid_tokens = int(m.sum().item())
                            if batch_div_valid_tokens < min_tok_bd:
                                batch_div_skipped = True
                            else:
                                probs = F.softmax(word_dist, dim=-1)
                                denom = m.sum(dim=(0, 1)).clamp(min=1.0)
                                mean_probs = (probs * m).sum(dim=(0, 1)) / denom
                                mean_probs = torch.nan_to_num(
                                    mean_probs, nan=0.0, posinf=0.0, neginf=0.0
                                )
                                if use_ema_bd:
                                    ema_buf = _model.batch_diversity_ema_mean_probs
                                    if not getattr(_model, "_batch_div_ema_seeded", False):
                                        if ema_init_mode == "uniform":
                                            with torch.no_grad():
                                                _vn = max(int(ema_buf.numel()), 1)
                                                ema_buf.fill_(1.0 / float(_vn))
                                        _model._batch_div_ema_seeded = True
                                    blend = ema_dec_bd * ema_buf.detach() + (1.0 - ema_dec_bd) * mean_probs
                                    blend = torch.nan_to_num(
                                        blend, nan=0.0, posinf=0.0, neginf=0.0
                                    )
                                    loss_bd_raw_preclamp = torch.sum(blend * torch.log(blend + eps_bd))
                                    with torch.no_grad():
                                        ema_buf.mul_(ema_dec_bd).add_(
                                            mean_probs.detach(), alpha=(1.0 - ema_dec_bd)
                                        )
                                else:
                                    loss_bd_raw_preclamp = torch.sum(
                                        mean_probs * torch.log(mean_probs + eps_bd)
                                    )
                                loss_bd_raw_preclamp = torch.nan_to_num(
                                    loss_bd_raw_preclamp,
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=0.0,
                                )
                                loss_bd = torch.clamp(
                                    loss_bd_raw_preclamp, min=-clamp_bd, max=clamp_bd
                                )
                                if not getattr(_model, "_batch_div_loss_warned", False):
                                    _raw_dbg = float(loss_bd_raw_preclamp.detach().item())
                                    _wtd_dbg = float((w_bd_eff * loss_bd).detach().item())
                                    if (
                                        not math.isfinite(_raw_dbg)
                                        or not math.isfinite(_wtd_dbg)
                                        or abs(_raw_dbg) > 1e5
                                        or abs(_wtd_dbg) > 1e5
                                    ):
                                        _model._batch_div_loss_warned = True
                                        if _lg is not None:
                                            _lg.warning(
                                                "[BatchDiversity] abnormal loss_batch_div_raw=%.6g "
                                                "loss_batch_div_weighted=%.6g (one-shot)",
                                                _raw_dbg,
                                                _wtd_dbg,
                                                extra=log_route_extra(_lg, ROUTE_SUMMARY),
                                            )
                        loss = compose_step5_total_loss(
                            loss_factual=loss_factual,
                            loss_counterfactual=loss_counterfactual,
                            loss_repeat_ul=loss_ul,
                            loss_terminal_clean=loss_tc,
                            loss_batch_diversity=loss_bd,
                            repeat_ul_weight=w_ul,
                            terminal_clean_weight=w_tc,
                            batch_diversity_weight=w_bd_eff,
                            lci_weighted_loss=lci_bundle.lci_weighted_loss,
                            fca_weighted_loss=fca_bundle.fca_weighted_loss,
                            ortho_keep_loss=loss_ortho_keep,
                            ortho_keep_weight=lambda_ortho_step5,
                        )
                    with torch.no_grad():
                        wsum = w.sum().clamp(min=1e-8)
                        _tr = (loss_r_ps * w).sum() / wsum
                        _tc = (loss_c_ps * w).sum() / wsum
                        _te = (loss_e_ps * w).sum() / wsum
                    _step_timer.start("backward")
                    (loss * inv_accum).backward()
                    _step_timer.end("backward")
                if sync:
                    _step_timer.start("optim")
                    _log_grad = rank == 0 and _lg is not None and (global_step + 1) % grad_iv == 0
                    _pre_gn = None
                    _tops = None
                    if _log_grad:
                        _pre_gn = grad_norm_total(model.parameters())
                        _tops = grad_topk_param_norms(model, odcr_grad_topk())
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    if _log_grad:
                        _post_gn = grad_norm_total(model.parameters())
                        _tp = (
                            " top_params=" + json.dumps(_tops, ensure_ascii=False)
                            if _tops
                            else ""
                        )
                        _lg.info(
                            "[GradClip] global_step=%d epoch=%d grad_norm_pre_clip=%.6g grad_norm_post_clip=%.6g%s",
                            global_step + 1,
                            epoch_1,
                            float(_pre_gn),
                            float(_post_gn),
                            _tp,
                            extra=log_route_extra(_lg, ROUTE_DETAIL),
                        )
                    optimizer.step()
                    if ema_model is not None:
                        ema_model.update_parameters(_model)
                    optimizer.zero_grad(set_to_none=True)
                    # LambdaLR：必须在 optimizer.step() 之后调用，使内部 step 与全局优化步一致
                    if sched is not None:
                        sched.step()
                        scheduler_steps += 1
                    global_step += 1
                    _wbd = float(_epoch_bd_eff_weight)
                    if (
                        _w_bd_base > 0.0
                        and float(_epoch_bd_ramp_factor) > 0.0
                        and int(epoch_1) > int(_bd_warm)
                    ):
                        if not batch_div_skipped:
                            _epoch_loss_bd_sum += float(loss_bd.detach().item())
                            _epoch_loss_bd_wsum += float((_wbd * loss_bd).detach().item())
                            _epoch_loss_bd_batches += 1
                            _epoch_loss_bd_raw_sum += float(loss_bd_raw_preclamp.detach().item())
                            if use_ema_bd:
                                _epoch_bd_ema_batches += 1
                        else:
                            _epoch_bd_skip_batches += 1
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
                        _extra = None
                        with torch.no_grad():
                            _wsum = w.sum().clamp(min=1e-8)
                            _lr_h = (loss_r_ps * w).sum() / _wsum
                            _lc_h = (loss_c_ps * w).sum() / _wsum
                            _le_h = (loss_e_ps * w).sum() / _wsum
                        w_ul = float(final_cfg.loss_weight_repeat_ul)
                        w_tc = float(final_cfg.loss_weight_terminal_clean)
                        _lul = float(loss_ul.detach().item()) if w_ul > 0 else 0.0
                        _ltc = float(loss_tc.detach().item()) if w_tc > 0 else 0.0
                        _brk = {
                            "main": float((loss_factual + loss_counterfactual).detach().item()),
                            "weighted_repeat_ul": w_ul * _lul,
                            "weighted_terminal_clean": w_tc * _ltc,
                        }
                        if odcr_log_step_loss_parts():
                            _extra = {
                                "loss_factual": float(loss_factual.detach().item()),
                                "loss_counterfactual": float(loss_counterfactual.detach().item()),
                                "loss_r": float(_lr_h.item()),
                                "loss_c": float(_lc_h.item()),
                                "loss_e": float(_le_h.item()),
                                "loss_repeat_ul": _lul,
                                "loss_terminal_clean": _ltc,
                                "total_loss_breakdown": _brk,
                            }
                        else:
                            _extra = {
                                "loss_r": float(_lr_h.item()),
                                "loss_c": float(_lc_h.item()),
                                "loss_e": float(_le_h.item()),
                                "loss_repeat_ul": _lul,
                                "loss_terminal_clean": _ltc,
                                "total_loss_breakdown": _brk,
                            }
                        _extra["loss_lci"] = float(l_lci.detach().item())
                        _extra["loss_lci_weighted"] = float(lci_bundle.lci_weighted_loss.detach().item())
                        _extra["loss_lci_consistency"] = float(lci_bundle.lci_consistency_loss.detach().item())
                        _extra["loss_lci_cf_score"] = float(lci_bundle.lci_cf_score_loss.detach().item())
                        _extra["loss_lci_robustness"] = float(lci_bundle.lci_robustness_loss.detach().item())
                        _extra["uci_weight"] = float(lci_bundle.uci_weight_mean.detach().item())
                        _extra["step5a_scorer_weight"] = float(lci_bundle.scorer_weight_mean.detach().item())
                        _extra["loss_fca"] = float(l_fca.detach().item())
                        _extra["loss_fca_weighted"] = float(fca_bundle.fca_weighted_loss.detach().item())
                        _extra["step5b_fca_weight"] = float(fca_bundle.fca_weight_mean.detach().item())
                        _extra["step5b_explainer_weight"] = float(step5b_gate.explainer_weight.detach().mean().item())
                        _extra.update(getattr(_model, "_last_ccv_control_stats", {}) or {})
                        _extra["loss_ortho"] = float(loss_ortho_keep.detach().item())
                        _extra["loss_ortho_xcov"] = float(loss_ortho_xcov_log)
                        _extra["loss_ortho_cos"] = float(loss_ortho_cos_log)
                        _extra["loss_ortho_keep_weighted"] = float(
                            (lambda_ortho_step5 * loss_ortho_keep).detach().item()
                        )
                        _wbd_log = float(_epoch_bd_eff_weight)
                        if (
                            _w_bd_base > 0.0
                            and float(_epoch_bd_ramp_factor) > 0.0
                            and int(epoch_1) > int(_bd_warm)
                        ):
                            _extra["loss_batch_div_raw"] = float(loss_bd_raw_preclamp.detach().item())
                            _extra["loss_batch_div_weighted"] = float(
                                (_wbd_log * loss_bd).detach().item()
                            )
                            _extra["batch_div_weight_effective"] = float(_wbd_log)
                            _extra["batch_div_ramp_factor"] = float(_epoch_bd_ramp_factor)
                            _extra["batch_div_ema_initialized"] = bool(
                                getattr(_model, "_batch_div_ema_seeded", False)
                            )
                            _extra["batch_div_valid_tokens"] = int(batch_div_valid_tokens)
                            _extra["batch_div_skipped_low_tokens"] = bool(batch_div_skipped)
                            _extra["batch_div_use_ema"] = bool(use_ema_bd)
                        log_step_sample(
                            _lg,
                            global_step=global_step,
                            epoch=epoch_1,
                            lr=float(_lr_now),
                            train_loss_batch=float(loss.detach().item()),
                            extra=_extra,
                        )
                    _step_timer.end("optim")
                if _do_step_profile:
                    _pf = (
                        "[StepProfile] micro_step=%d sync=%s forward_ms=%.3f exp_ce_ms=%.3f "
                        "repeat_ul_ms=%.3f terminal_clean_ms=%.3f backward_ms=%.3f optim_ms=%.3f"
                    ) % (
                        micro_step_count,
                        str(bool(sync)),
                        _step_timer.ms("forward"),
                        _step_timer.ms("exp_ce"),
                        _step_timer.ms("repeat_ul"),
                        _step_timer.ms("terminal_clean"),
                        _step_timer.ms("backward"),
                        _step_timer.ms("optim"),
                    )
                    if _lg:
                        _lg.info(_pf, extra=log_route_extra(_lg, ROUTE_SUMMARY))
                    else:
                        print(_pf, flush=True)
                loss_sum = loss_sum + loss.detach().double() * bsz
                loss_r_sum = loss_r_sum + _tr.double() * bsz
                loss_c_sum = loss_c_sum + _tc.double() * bsz
                loss_e_sum = loss_e_sum + _te.double() * bsz
                n_samples += bsz
                if micro_step_count % step_iv == 0 and rank == 0:
                    run_training_finite_checks(
                        _finite_mode,
                        loss,
                        word_dist,
                        _lg,
                        global_step=global_step,
                        epoch=epoch_1,
                        route_detail=ROUTE_DETAIL,
                    )
            ddp_heartbeat(_lg, "before_train_loss_allreduce", rank=rank, epoch=epoch_1)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_r_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_c_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_e_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
            ddp_heartbeat(_lg, "after_train_loss_allreduce", rank=rank, epoch=epoch_1)
            _ns = float(n_samples.item()) if n_samples.numel() else 0.0
            avg_loss = (loss_sum / n_samples).item() if _ns > 0 else 0.0
            train_loss_r_epoch = (loss_r_sum / n_samples).item() if _ns > 0 else 0.0
            train_loss_c_epoch = (loss_c_sum / n_samples).item() if _ns > 0 else 0.0
            train_loss_e_epoch = (loss_e_sum / n_samples).item() if _ns > 0 else 0.0
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

            # valid：各 rank 只跑本地分片（DistributedSampler），样本加权 sum + all_reduce 得全局 avg（与 train loss 聚合方式一致）
            _t_valid0 = time.perf_counter()
            v_loss_sum, v_n, v_lr_sum, v_lc_sum, v_le_sum = validModel(
                model,
                valid_dataloader,
                device,
                coef=coef,
                explainer_loss_weight=explainer_loss_weight,
                step5_innov_cfg=step5_innov_cfg,
            )
            if rank == 0 and _lg:
                _lg.info(
                    "[Timing] valid_loss_forward end epoch=%d elapsed_s=%.3f",
                    epoch_1,
                    time.perf_counter() - _t_valid0,
                    extra=log_route_extra(_lg, ROUTE_SUMMARY),
                )
            v_stat = torch.tensor(
                [v_loss_sum, float(v_n), v_lr_sum, v_lc_sum, v_le_sum],
                dtype=torch.double,
                device=device,
            )
            dist.all_reduce(v_stat, op=dist.ReduceOp.SUM)
            _vn = float(v_stat[1].item())
            current_valid_loss = float(v_stat[0] / v_stat[1]) if _vn > 0 else 0.0
            valid_loss_r_epoch = float(v_stat[2] / v_stat[1]) if _vn > 0 else 0.0
            valid_loss_c_epoch = float(v_stat[3] / v_stat[1]) if _vn > 0 else 0.0
            valid_loss_e_epoch = float(v_stat[4] / v_stat[1]) if _vn > 0 else 0.0
            strict_would_save = bool(current_valid_loss <= prev_valid_loss + 1e-12)
            if current_valid_loss < shadow_strict_best_vl - 1e-12:
                shadow_strict_best_vl = float(current_valid_loss)
                shadow_strict_best_epoch = int(epoch_1)
            quick_bleu4 = None
            full_bleu4_val = None
            mainline_composite_val = None
            mainline_monitor_bundle: Optional[Dict[str, Any]] = None
            if (
                ckpt_mode == "guarded_composite"
                and str(checkpoint_metric).strip().lower() in ("valid_loss", "loss")
            ):
                _vd_mon = getattr(final_cfg, "valid_dataset", None)
                if _vd_mon is None:
                    raise RuntimeError("guarded_composite 选模需要 final_cfg.valid_dataset")
                _ov_mon = build_full_bleu_monitor_cfg_override(final_cfg)
                mainline_composite_val, mainline_monitor_bundle = mainline_monitor_full_valid_ddp(
                    model,
                    _vd_mon,
                    tokenizer=get_step5_tokenizer(),
                    device=int(device),
                    rank=int(rank),
                    world_size=int(world_size),
                    batch_size=int(final_cfg.eval_batch_size),
                    dataloader_num_workers=int(final_cfg.dataloader_num_workers_valid),
                    dataloader_prefetch_factor=final_cfg.dataloader_prefetch_factor_valid,
                    logger=_lg,
                    collate_fn=step5_collate_fn,
                    cfg_override=_ov_mon,
                    composite_weights=_checkpoint_composite_weights_dict(final_cfg),
                    uncertainty_high_entropy_threshold=float(
                        getattr(final_cfg, "uncertainty_high_entropy_threshold", 1.0)
                    ),
                )
            if rank == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                bleu_line = None
                if mainline_composite_val is not None:
                    _lg.info(
                        "[MainlineMonitor] epoch=%d composite=%.6f decode_override=%s",
                        epoch + 1,
                        float(mainline_composite_val),
                        json.dumps(build_full_bleu_monitor_cfg_override(final_cfg), ensure_ascii=False, sort_keys=True),
                        extra=log_route_extra(_lg, ROUTE_SUMMARY),
                    )
                _wbd_ep = float(_w_bd_base)
                if (
                    _wbd_ep > 0.0
                    and int(epoch + 1) > int(_bd_warm)
                    and float(_epoch_bd_ramp_factor) > 0.0
                ):
                    if _epoch_loss_bd_batches > 0:
                        _lg.info(
                            "[BatchDiversity] epoch=%d batch_div_ramp_factor=%.6g batch_div_weight_effective=%.6g "
                            "batch_div_ema_initialized=%s loss_batch_div_mean=%.6g loss_batch_div_raw_mean=%.6g "
                            "loss_batch_div_weighted_mean=%.6g batches_applied=%d batches_skipped_low_tokens=%d "
                            "batches_with_ema=%d",
                            epoch + 1,
                            float(_epoch_bd_ramp_factor),
                            float(_epoch_bd_eff_weight),
                            str(bool(getattr(_model, "_batch_div_ema_seeded", False))).lower(),
                            _epoch_loss_bd_sum / max(1, _epoch_loss_bd_batches),
                            _epoch_loss_bd_raw_sum / max(1, _epoch_loss_bd_batches),
                            _epoch_loss_bd_wsum / max(1, _epoch_loss_bd_batches),
                            int(_epoch_loss_bd_batches),
                            int(_epoch_bd_skip_batches),
                            int(_epoch_bd_ema_batches),
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    elif _epoch_bd_skip_batches > 0:
                        _lg.info(
                            "[BatchDiversity] epoch=%d all batches skipped (low valid tokens) skipped=%d",
                            epoch + 1,
                            int(_epoch_bd_skip_batches),
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                _ud_sum = (mainline_monitor_bundle or {}).get("uncertainty_decode_summary")
                if _ud_sum is not None and _lg:
                    _tr_raw = _ud_sum.get("uncertainty_trigger_rate")
                    tr_u = float(_tr_raw) if _tr_raw is not None else float("nan")
                    _mte = _ud_sum.get("mean_trigger_entropy")
                    _lg.info(
                        "[MainlineUncertainty] epoch=%d Mainline_Uncertainty_Trigger_Rate=%s "
                        "decisions=%d triggers=%d mean_trigger_entropy=%s trigger_entropy_p50=%s "
                        "trigger_entropy_p90=%s first_trig_mean=%s first_trig_p50=%s first_trig_p90=%s "
                        "first_trig_seqs=%d",
                        epoch + 1,
                        _fmt_opt_float(_tr_raw) if _tr_raw is not None else "n/a",
                        int(_ud_sum.get("uncertainty_total_decision_count", 0)),
                        int(_ud_sum.get("uncertainty_trigger_count", 0)),
                        _fmt_opt_float(_mte) if _mte is not None else "n/a",
                        _fmt_opt_float(_ud_sum.get("trigger_entropy_p50"))
                        if _ud_sum.get("trigger_entropy_p50") is not None
                        else "n/a",
                        _fmt_opt_float(_ud_sum.get("trigger_entropy_p90"))
                        if _ud_sum.get("trigger_entropy_p90") is not None
                        else "n/a",
                        _fmt_opt_float(_ud_sum.get("first_trigger_step_mean")),
                        _fmt_opt_float(_ud_sum.get("first_trigger_step_p50")),
                        _fmt_opt_float(_ud_sum.get("first_trigger_step_p90")),
                        int(_ud_sum.get("first_trigger_sequences", 0)),
                        extra=log_route_extra(_lg, ROUTE_SUMMARY),
                    )
                    if _tr_raw is not None and tr_u < 0.03:
                        _lg.warning(
                            "[MainlineUncertainty] low-trigger warning: trigger_rate=%.6f < 0.03 "
                            "(可考虑略降 gap_threshold 或缩短 prefix_greedy_steps；或检查 top_k/temperature 是否过保守)",
                            tr_u,
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    if _tr_raw is not None and tr_u > 0.30:
                        _lg.warning(
                            "[MainlineUncertainty] high-trigger warning: trigger_rate=%.6f > 0.30 "
                            "(可考虑提高 gap_threshold 或延长 prefix_greedy_steps；或降低 top_k / 采样温度激进程度)",
                            tr_u,
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    if (
                        _tr_raw is not None
                        and tr_u > 0.15
                        and _mte is not None
                        and float(_mte) < 0.25
                    ):
                        _lg.warning(
                            "[MainlineUncertainty] low-value-trigger: trigger_rate=%.6f 偏高但 mean_trigger_entropy=%.6f "
                            "偏低，多为 top 概率仍尖锐的「假触发」；可提高 gap_threshold 或检查 prefix_greedy_steps",
                            tr_u,
                            float(_mte),
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    if (
                        _tr_raw is not None
                        and tr_u > 0.15
                        and _mte is not None
                        and float(_mte) > 1.0
                    ):
                        _lg.warning(
                            "[MainlineUncertainty] aggressive-sampling: trigger_rate=%.6f 且 mean_trigger_entropy=%.6f "
                            "偏高，低温 top-k 分布较平；可降低 top_k / 略降温度或提高 gap_threshold",
                            tr_u,
                            float(_mte),
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                lr_sched_line = None
                if sched is not None and ws_resolved is not None:
                    lr_sched_line = (
                        f"scheduler_type=warmup_cosine "
                        f"initial_lr={initial_lr:.6g} current_lr={lr_epoch:.6g} min_lr={min_lr_effective:.6g} "
                        f"min_lr_ratio={min_lr_ratio:.6g} warmup_steps={ws_resolved} total_steps={total_steps_plan} "
                        f"scheduler_steps_cumulative={scheduler_steps} warmup_ratio={warmup_ratio_logged:.6g}"
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
                _gate_stats = getattr(_model, "get_domain_gate_stats", lambda: {})()
                if _gate_stats:
                    _lg.info(
                        "[DMPF gate][epoch=%d] mode=%s stats=%s",
                        epoch + 1,
                        str(getattr(_model, "domain_fusion_mode", "gate_cross_attn")),
                        json.dumps(_gate_stats, ensure_ascii=False, default=str),
                        extra=log_route_extra(_lg, ROUTE_SUMMARY),
                    )
                summ = format_epoch_summary_lines(
                    epoch=epoch + 1,
                    train_loss_total_epoch=avg_loss,
                    train_loss_r_epoch=train_loss_r_epoch,
                    train_loss_c_epoch=train_loss_c_epoch,
                    train_loss_e_epoch=train_loss_e_epoch,
                    valid_loss_total_epoch=current_valid_loss,
                    valid_loss_r_epoch=valid_loss_r_epoch,
                    valid_loss_c_epoch=valid_loss_c_epoch,
                    valid_loss_e_epoch=valid_loss_e_epoch,
                    lr=lr_epoch,
                    quick_bleu4=quick_bleu4,
                    full_bleu_monitor_bleu4=full_bleu4_val,
                    meteor=None,
                )
                log_epoch_summary_compact(_lg, summ)
                _row_metrics: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "train_loss_total_epoch": avg_loss,
                    "train_loss_r_epoch": train_loss_r_epoch,
                    "train_loss_c_epoch": train_loss_c_epoch,
                    "train_loss_e_epoch": train_loss_e_epoch,
                    "valid_loss_total_epoch": current_valid_loss,
                    "valid_loss_r_epoch": valid_loss_r_epoch,
                    "valid_loss_c_epoch": valid_loss_c_epoch,
                    "valid_loss_e_epoch": valid_loss_e_epoch,
                    "lr": lr_epoch,
                    "quick_bleu4": quick_bleu4,
                    "full_bleu_monitor_bleu4": full_bleu4_val,
                    "mainline_composite_score": mainline_composite_val,
                    "checkpoint_selection_mode": ckpt_mode,
                    "loss_weight_batch_diversity": float(final_cfg.loss_weight_batch_diversity),
                    "batch_diversity_warmup_epochs": int(final_cfg.batch_diversity_warmup_epochs),
                    "batch_diversity_ramp_epochs": int(getattr(final_cfg, "batch_diversity_ramp_epochs", 2)),
                    "batch_diversity_ramp_target_scale": float(
                        getattr(final_cfg, "batch_diversity_ramp_target_scale", 1.0)
                    ),
                    "batch_diversity_ema_init_mode": str(
                        getattr(final_cfg, "batch_diversity_ema_init_mode", "uniform")
                    ),
                    "batch_div_ramp_factor_epoch": float(_epoch_bd_ramp_factor),
                    "batch_div_weight_effective_epoch": float(_epoch_bd_eff_weight),
                    "lambda_ortho_step5": float(lambda_ortho_step5),
                    "lambda_ortho_xcov": float(orth5_w_xcov),
                    "lambda_ortho_cos": float(orth5_w_cos),
                    "step5_lci_weight": float(w_lci),
                    "step5_fca_weight": float(w_fca),
                }
                if _epoch_orth_batches > 0:
                    _row_metrics["loss_ortho_epoch_mean"] = _epoch_orth_sum / float(_epoch_orth_batches)
                if _epoch_loss_bd_batches > 0:
                    _row_metrics["loss_batch_div_epoch_mean"] = _epoch_loss_bd_sum / max(1, _epoch_loss_bd_batches)
                    _row_metrics["loss_batch_div_raw_epoch_mean"] = _epoch_loss_bd_raw_sum / max(
                        1, _epoch_loss_bd_batches
                    )
                    _row_metrics["loss_batch_div_weighted_epoch_mean"] = _epoch_loss_bd_wsum / max(
                        1, _epoch_loss_bd_batches
                    )
                if _epoch_bd_skip_batches > 0:
                    _row_metrics["batch_div_batches_skipped_low_tokens"] = int(_epoch_bd_skip_batches)
                _uds = (mainline_monitor_bundle or {}).get("uncertainty_decode_summary")
                if _uds is not None:
                    _row_metrics["uncertainty_decode_summary"] = dict(_uds)
                epoch_metrics_row = _row_metrics
            last_ckpt_meta["valid_loss_total"] = current_valid_loss
            last_ckpt_meta["valid_loss_r"] = valid_loss_r_epoch
            last_ckpt_meta["valid_loss_c"] = valid_loss_c_epoch
            last_ckpt_meta["valid_loss_e"] = valid_loss_e_epoch
            if rank == 0:
                last_ckpt_meta["quick_bleu4"] = quick_bleu4
                last_ckpt_meta["full_bleu4"] = full_bleu4_val
                last_ckpt_meta["mainline_composite"] = mainline_composite_val
            if current_valid_loss > prev_valid_loss:
                enduration += 1
                if lr_scheduler_type != "warmup_cosine":
                    learning_rate /= 2.0
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = learning_rate
            else:
                enduration = 0

            do_save = False
            save_reason = ""
            if str(checkpoint_metric).strip().lower() in ("valid_loss", "loss") and rank == 0:
                if ckpt_mode == "valid_loss_only":
                    do_save = current_valid_loss <= prev_valid_loss
                    save_reason = "valid_loss_not_worse_than_prev_epoch"
                elif ckpt_mode == "guarded_composite":
                    best_vl_gate_ref = float(best_vl_ever)
                    rel_tol = float(final_cfg.checkpoint_guard_valid_loss_rel_tol)
                    abs_tol = float(final_cfg.checkpoint_guard_valid_loss_abs_tol)
                    tol_line = best_vl_gate_ref * (1.0 + rel_tol) + abs_tol
                    strict_vl = current_valid_loss < best_vl_gate_ref - 1e-12
                    within_tol = current_valid_loss <= tol_line + 1e-12
                    comp = float(mainline_composite_val if mainline_composite_val is not None else 0.0)
                    comp_better = comp > best_composite_saved + 1e-9
                    gate_ok = True
                    if mainline_monitor_bundle is not None:
                        gate_ok, _ = mainline_selection_gate(
                            mainline_monitor_bundle, best_monitor_bundle_gate
                        )
                    if strict_vl:
                        do_save = True
                        save_reason = "valid_loss_strict_improve_vs_best_ever"
                    elif within_tol and comp_better and gate_ok:
                        do_save = True
                        save_reason = "composite_tiebreak_within_loss_tolerance"
                    reject_reason = ""
                    if not do_save:
                        if not strict_vl and not within_tol:
                            reject_reason = "valid_loss_outside_tolerance_band"
                        elif within_tol and not comp_better:
                            reject_reason = "composite_not_improved_within_tolerance_band"
                        elif within_tol and comp_better and not gate_ok:
                            reject_reason = "mainline_monitor_gate_failed"
                        else:
                            reject_reason = "no_strict_improvement_and_no_tiebreak_eligible"
                    if _lg is not None:
                        _lg.info(
                            "[CheckpointGuard] epoch=%d valid_loss=%.6f best_valid_loss_ever=%.6f "
                            "rel_tol=%.6g abs_tol=%.6g tol_ceiling=%.6f strict_better=%s within_tol=%s "
                            "composite=%.6f best_composite_saved=%.6f composite_better=%s mainline_gate_ok=%s "
                            "will_save=%s accepted_path=%s reject_reason=%s",
                            int(epoch_1),
                            float(current_valid_loss),
                            float(best_vl_gate_ref),
                            float(rel_tol),
                            float(abs_tol),
                            float(tol_line),
                            str(bool(strict_vl)).lower(),
                            str(bool(within_tol)).lower(),
                            float(comp),
                            float(best_composite_saved),
                            str(bool(comp_better)).lower(),
                            str(bool(gate_ok)).lower(),
                            str(bool(do_save)).lower(),
                            save_reason if do_save else "",
                            reject_reason,
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    best_vl_ever = min(best_vl_ever, current_valid_loss)
                    strict_vs_guarded_diverged = bool(strict_would_save != do_save)
                    if _lg is not None:
                        _lg.info(
                            "[CheckpointShadowStrict] epoch=%d strict_would_save=%s guarded_will_save=%s "
                            "strict_best_valid_loss=%.6f shadow_strict_best_epoch=%d "
                            "guarded_best_valid_loss_ever=%.6f strict_vs_guarded_diverged=%s",
                            int(epoch_1),
                            str(strict_would_save).lower(),
                            str(bool(do_save)).lower(),
                            float(shadow_strict_best_vl),
                            int(shadow_strict_best_epoch),
                            float(best_vl_ever),
                            str(strict_vs_guarded_diverged).lower(),
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                    if (not do_save) and strict_would_save and _lg is not None:
                        _lg.warning(
                            "[CheckpointShadowStrict] guarded_rejected_but_strict_would_save epoch=%d "
                            "valid_loss=%.6f",
                            int(epoch_1),
                            float(current_valid_loss),
                            extra=log_route_extra(_lg, ROUTE_SUMMARY),
                        )
                else:
                    do_save = current_valid_loss <= prev_valid_loss
                    save_reason = "unknown_mode_fallback_prev_epoch"

            if str(checkpoint_metric).strip().lower() in ("valid_loss", "loss") and rank == 0 and do_save:
                best_version += 1
                if ckpt_mode == "guarded_composite" and mainline_monitor_bundle is not None:
                    best_composite_saved = float(mainline_composite_val or 0.0)
                    best_monitor_bundle_gate = dict(mainline_monitor_bundle)
                model_to_save = state_dict_for_canonical_best_pth(
                    ema_enabled=ema_enabled,
                    ema_model=ema_model,
                    ddp_module=model,
                    underlying_model_fn=get_underlying_model,
                )
                atomic_torch_save(str(final_cfg.save_file), model_to_save)
                model_sha = _sha256_file(str(final_cfg.save_file))
                _ckpt_lineage = _build_step5_checkpoint_lineage(final_cfg, model)
                _ckpt_lineage["checkpoint_file"] = file_fingerprint(str(final_cfg.save_file))
                _ckpt_lineage_path = write_checkpoint_lineage(str(final_cfg.save_file), _ckpt_lineage)
                _sp = _state_paths_from_save_file(str(final_cfg.save_file))
                os.makedirs(_sp["state_dir"], exist_ok=True)
                atomic_torch_save(_sp["optimizer_pt"], optimizer.state_dict())
                atomic_write_json(
                    _sp["trainer_state"],
                    {
                        "epoch": int(epoch_1),
                        "global_step": int(global_step),
                        "scheduler_steps": int(scheduler_steps),
                    },
                )
                _acc_vs_shadow = "n/a"
                if ckpt_mode == "guarded_composite":
                    if save_reason == "valid_loss_strict_improve_vs_best_ever":
                        if (
                            abs(float(current_valid_loss) - float(shadow_strict_best_vl)) < 1e-8
                            and int(epoch_1) == int(shadow_strict_best_epoch)
                        ):
                            _acc_vs_shadow = "same_epoch_as_shadow_strict_best"
                        else:
                            _acc_vs_shadow = "strict_primary_accepted"
                    elif save_reason == "composite_tiebreak_within_loss_tolerance":
                        if current_valid_loss > shadow_strict_best_vl + 1e-8:
                            _acc_vs_shadow = "guarded_later_composite_tiebreak_vs_shadow_best_vl"
                        else:
                            _acc_vs_shadow = "composite_tiebreak_within_tolerance"
                    else:
                        _acc_vs_shadow = str(save_reason)
                atomic_write_json(
                    _sp["best_event"],
                    {
                        "best_version": int(best_version),
                        "epoch": int(epoch_1),
                        "global_step": int(global_step),
                        "valid_loss": float(current_valid_loss),
                        "model_sha256": model_sha,
                        "checkpoint_lineage_hash": _ckpt_lineage.get("checkpoint_compatibility_hash"),
                        "checkpoint_lineage_path": str(_ckpt_lineage_path),
                        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ema_enabled": bool(ema_enabled),
                        "weight_kind": "ema" if ema_enabled else "raw",
                        "checkpoint_selection_mode": ckpt_mode,
                        "checkpoint_save_reason": save_reason,
                        "mainline_composite_score": mainline_composite_val,
                        "checkpoint_guard_valid_loss_rel_tol": float(
                            final_cfg.checkpoint_guard_valid_loss_rel_tol
                        ),
                        "checkpoint_guard_valid_loss_abs_tol": float(
                            final_cfg.checkpoint_guard_valid_loss_abs_tol
                        ),
                        "checkpoint_acceptance": (
                            "strict_valid_loss"
                            if save_reason == "valid_loss_strict_improve_vs_best_ever"
                            else (
                                "composite_tiebreak_within_tolerance"
                                if save_reason == "composite_tiebreak_within_loss_tolerance"
                                else str(save_reason)
                            )
                        ),
                        "shadow_strict_best_epoch": int(shadow_strict_best_epoch),
                        "shadow_strict_best_valid_loss": float(shadow_strict_best_vl),
                        "accepted_vs_shadow_strict_relation": _acc_vs_shadow,
                    },
                )
                if _lg is not None:
                    _lg.info(
                        "[Checkpoint] updated canonical best=%s epoch=%d valid_loss=%.6f best_version=%d "
                        "mode=%s reason=%s composite=%s ema=%s",
                        str(final_cfg.save_file),
                        int(epoch_1),
                        float(current_valid_loss),
                        int(best_version),
                        ckpt_mode,
                        save_reason,
                        str(mainline_composite_val),
                        str(bool(ema_model is not None)).lower(),
                        extra=log_route_extra(_lg, ROUTE_SUMMARY),
                    )
            prev_valid_loss = current_valid_loss
            if rank == 0 and epoch_metrics_row is not None:
                if str(checkpoint_metric).strip().lower() in ("valid_loss", "loss"):
                    epoch_metrics_row["strict_would_save"] = bool(strict_would_save)
                    epoch_metrics_row["guarded_will_save"] = bool(do_save)
                    if ckpt_mode == "guarded_composite":
                        epoch_metrics_row["strict_vs_guarded_diverged"] = bool(
                            strict_would_save != do_save
                        )
                        epoch_metrics_row["shadow_strict_best_valid_loss"] = float(shadow_strict_best_vl)
                        epoch_metrics_row["shadow_strict_best_epoch"] = int(shadow_strict_best_epoch)
                        epoch_metrics_row["guarded_best_valid_loss_ever"] = float(best_vl_ever)
                append_train_epoch_metrics_jsonl(log_file=final_cfg.log_file, row=epoch_metrics_row)

            if odcr_ddp_epoch_end_barrier():
                ddp_heartbeat(_lg, "before_epoch_end_barrier", rank=rank, epoch=epoch_1)
                dist.barrier()
                ddp_heartbeat(_lg, "after_epoch_end_barrier", rank=rank, epoch=epoch_1)

            if epoch + 1 >= min_epochs and enduration >= early_stop_patience:
                save_last_checkpoint = True
                break
        else:
            save_last_checkpoint = True
    finally:
        if rank == 0 and perf is not None:
            perf.finish()
        # canonical checkpoint-only: 训练结束不再写 second model file。


def _step5_checkpoint_metadata(
    final_cfg: FinalTrainingConfig,
    *,
    valid_loss_total: float,
    valid_loss_r: float,
    valid_loss_c: float,
    valid_loss_e: float,
    quick_bleu4: Optional[float] = None,
    full_bleu4: Optional[float] = None,
    mainline_composite: Optional[float] = None,
    mainline_bundle: Optional[Dict[str, Any]] = None,
    checkpoint_policy: str = "best",
) -> Dict[str, Any]:
    """与 canonical 权重同 stem 的 .meta.json。"""
    _mon = json.dumps(
        build_full_bleu_monitor_cfg_override(final_cfg),
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    _cksel = str(getattr(final_cfg, "checkpoint_selection_mode", "guarded_composite")).strip().lower()
    _sem = (
        "guarded_valid_loss_plus_mainline_text_composite"
        if _cksel == "guarded_composite"
        else "valid_loss_prev_epoch_monotone"
    )
    out: Dict[str, Any] = {
        "checkpoint_policy": str(checkpoint_policy),
        "checkpoint_selection_metric": str(final_cfg.checkpoint_metric),
        "checkpoint_selection_mode": _cksel,
        "checkpoint_selection_decode_semantics": _sem,
        "checkpoint_guard_valid_loss_rel_tol": float(final_cfg.checkpoint_guard_valid_loss_rel_tol),
        "checkpoint_guard_valid_loss_abs_tol": float(final_cfg.checkpoint_guard_valid_loss_abs_tol),
        "checkpoint_composite_weights": _checkpoint_composite_weights_dict(final_cfg),
        "loss_weight_batch_diversity": float(getattr(final_cfg, "loss_weight_batch_diversity", 0.0)),
        "batch_diversity_warmup_epochs": int(getattr(final_cfg, "batch_diversity_warmup_epochs", 0)),
        "batch_diversity_mode": str(getattr(final_cfg, "batch_diversity_mode", "")),
        "batch_diversity_use_ema": bool(getattr(final_cfg, "batch_diversity_use_ema", True)),
        "batch_diversity_ema_decay": float(getattr(final_cfg, "batch_diversity_ema_decay", 0.9)),
        "batch_diversity_min_valid_tokens": int(
            getattr(final_cfg, "batch_diversity_min_valid_tokens", 64)
        ),
        "batch_diversity_loss_clamp_abs": float(
            getattr(final_cfg, "batch_diversity_loss_clamp_abs", 0.2)
        ),
        "batch_diversity_ramp_epochs": int(getattr(final_cfg, "batch_diversity_ramp_epochs", 2)),
        "batch_diversity_ramp_target_scale": float(
            getattr(final_cfg, "batch_diversity_ramp_target_scale", 1.0)
        ),
        "batch_diversity_ema_init_mode": str(
            getattr(final_cfg, "batch_diversity_ema_init_mode", "uniform")
        ),
        "uncertainty_entropy_eps": float(getattr(final_cfg, "uncertainty_entropy_eps", 1e-8)),
        "uncertainty_high_entropy_threshold": float(
            getattr(final_cfg, "uncertainty_high_entropy_threshold", 1.0)
        ),
        "decode_monitor_fingerprint_sha1": hashlib.sha1(_mon.encode("utf-8")).hexdigest()[:16],
        "valid_loss_total_epoch": float(valid_loss_total),
        "valid_loss_r_epoch": float(valid_loss_r),
        "valid_loss_c_epoch": float(valid_loss_c),
        "valid_loss_e_epoch": float(valid_loss_e),
        "quick_bleu4": None if quick_bleu4 is None else float(quick_bleu4),
        "full_bleu_monitor_bleu4": None if full_bleu4 is None else float(full_bleu4),
        "mainline_composite_score": None if mainline_composite is None else float(mainline_composite),
        "full_bleu_decode_strategy": str(final_cfg.full_bleu_decode_strategy),
        "decode_strategy": str(final_cfg.decode_strategy),
        "generate_temperature": float(final_cfg.generate_temperature),
        "generate_top_p": float(final_cfg.generate_top_p),
        "repetition_penalty": float(final_cfg.repetition_penalty),
        "decode_seed": final_cfg.decode_seed,
        "max_explanation_length": int(final_cfg.max_explanation_length),
        "train_label_max_length": int(getattr(final_cfg, "train_label_max_length", 128)),
        "hard_max_len": getattr(final_cfg, "hard_max_len", None),
        "soft_max_len": getattr(final_cfg, "soft_max_len", None),
    }
    if mainline_bundle is not None:
        out["mainline_monitor_snapshot"] = {
            "bleu": mainline_bundle.get("bleu"),
            "rouge": mainline_bundle.get("rouge"),
            "meteor": mainline_bundle.get("meteor"),
            "dist": mainline_bundle.get("dist"),
            "dirty_hit_rate": mainline_bundle.get("dirty_hit_rate"),
            "rmse_rating": mainline_bundle.get("rmse_rating"),
            "mae_rating": mainline_bundle.get("mae_rating"),
            "mainline_composite_score": mainline_bundle.get("mainline_composite_score"),
            "checkpoint_composite_weights": mainline_bundle.get("checkpoint_composite_weights"),
            "uncertainty_decode_summary": mainline_bundle.get("uncertainty_decode_summary"),
        }
    return out


def odcr_terminal_cleanliness_loss(
    word_logits: torch.Tensor,
    tgt: torch.Tensor,
    bad_ids: Sequence[int],
    span: int,
    *,
    pad_id: int = 0,
) -> torch.Tensor:
    """尾部若干步上压低「坏尾」词 id 的 softmax 质量（轻量）。"""
    if not bad_ids:
        return graph_tied_zero(word_logits)
    B, T, _V = word_logits.shape
    span = max(1, min(int(span), T))
    t0 = max(0, T - span)
    sl = word_logits[:, t0:, :]
    m = (tgt[:, t0:] != pad_id).float()
    probs = F.softmax(sl, dim=-1)
    bid = torch.tensor(list(bad_ids), device=probs.device, dtype=torch.long)
    mass = probs.index_select(-1, bid).sum(-1)
    den = m.sum().clamp(min=1.0)
    return (mass * m).sum() / den


def validModel(model, valid_dataloader, device, *, coef: float, explainer_loss_weight: float, step5_innov_cfg=None):
    _model = get_underlying_model(model)
    # forward 必须用底层 _model，避免 DDP 包装后的 model(...) 在局部执行时触发额外 NCCL collective。
    _model.eval()
    loss_sum = 0.0
    loss_r_sum = 0.0
    loss_c_sum = 0.0
    loss_e_sum = 0.0
    n_samples = 0
    c = float(coef)
    explainer_w_scalar = float(explainer_loss_weight)
    if step5_innov_cfg is None:
        raise RuntimeError("validModel requires resolved step5_innov_cfg from configs/odcr.yaml.")
    with torch.no_grad():
        for batch in valid_dataloader:
            gb = require_gathered_batch(_model.gather(batch, device))
            user_idx = gb.user_idx
            item_idx = gb.item_idx
            rating = gb.rating
            tgt_input = gb.tgt_input
            tgt_output = gb.tgt_output
            domain_idx = gb.domain_idx
            exp_w = gb.exp_sample_weight
            route_scorer_mask = gb.route_scorer_mask
            route_explainer_mask = gb.route_explainer_mask
            if exp_w is None:
                raise RuntimeError("validModel 缺少 exp_sample_weight，无法与训练损失口径对齐。")
            if route_scorer_mask is None or route_explainer_mask is None:
                raise RuntimeError("validModel 缺少 route_scorer/route_explainer 掩码。")
            entropy_s = gb.entropy_score
            uncertainty_s = gb.uncertainty_score
            conf_bucket = gb.confidence_bucket
            evidence_f = gb.evidence_features
            c_anchor = gb.content_anchor_score
            s_anchor = gb.style_anchor_score
            step5a_gate = build_step5a_scorer_gate(gb, step5_innov_cfg)
            step5b_gate = build_step5b_explainer_gate(gb, step5_innov_cfg)
            ccv_packet = build_ccv_control_packet(gb, step5_innov_cfg)
            if (
                entropy_s is None
                or uncertainty_s is None
                or conf_bucket is None
                or evidence_f is None
                or c_anchor is None
                or s_anchor is None
            ):
                raise RuntimeError("validModel 缺少 UCI / evidence / anchor 张量。")
            bsz = int(user_idx.size(0))
            _validate_route_masks_batch(
                route_scorer_mask,
                route_explainer_mask,
                batch_size=bsz,
                stage="valid",
            )
            with odcr_cuda_bf16_autocast():
                pred_rating, context_dist, word_dist = _model(
                    user_idx,
                    item_idx,
                    tgt_input,
                    domain_idx,
                    target_tokens=tgt_output,
                    evidence_features=evidence_f,
                    content_anchor_score=c_anchor,
                    style_anchor_score=s_anchor,
                    ccv_control_packet=ccv_packet,
                )
            ls = float(getattr(_model.exp_loss_fn, "label_smoothing", 0.0) or 0.0)
            loss_r_ps = F.mse_loss(pred_rating, rating, reduction="none")
            loss_flan_ps = per_sample_decoder_ce_from_logits(
                word_dist, tgt_output, ignore_index=0, label_smoothing=ls
            )
            loss_c_ps = graph_tied_zero_like(pred_rating).to(dtype=loss_flan_ps.dtype)
            loss_e_ps = loss_flan_ps
            scorer_only = loss_r_ps
            explainer_only = c * loss_c_ps + loss_e_ps
            dom = domain_idx.view(-1)
            w = exp_w.view(-1)
            rs = route_scorer_mask.view(-1)
            re = route_explainer_mask.view(-1)
            f_mask = (dom == 1).to(dtype=scorer_only.dtype)
            c_mask = (dom == 0).to(dtype=scorer_only.dtype)
            scorer_w = step5a_gate.scorer_weight.to(dtype=w.dtype)
            explainer_w = step5b_gate.explainer_weight.to(dtype=w.dtype)
            loss = route_weighted_mean(scorer_only, scorer_w, f_mask) + explainer_w_scalar * route_weighted_mean(
                explainer_only, explainer_w, c_mask
            )
            wsum = w.sum().clamp(min=1e-8)
            loss_r = (loss_r_ps * w).sum() / wsum
            loss_c = (loss_c_ps * w).sum() / wsum
            loss_e = (loss_e_ps * w).sum() / wsum
            loss_sum += float(loss.detach().item()) * bsz
            loss_r_sum += float(loss_r.detach().item()) * bsz
            loss_c_sum += float(loss_c.detach().item()) * bsz
            loss_e_sum += float(loss_e.detach().item()) * bsz
            n_samples += bsz
    return loss_sum, n_samples, loss_r_sum, loss_c_sum, loss_e_sum


def evalModel(model, test_dataloader, device, *, step5_innov_cfg):
    """逐 batch 推理，返回带 sample_id 的行列表（用于 DDP gather 后按 id 排序）。"""
    import time as _time_perf

    _model = get_underlying_model(model).to(device)
    _model.eval()
    rows: List[dict] = []
    decode_wall = 0.0
    with torch.no_grad():
        for batch in test_dataloader:
            _t0 = _time_perf.perf_counter()
            gb = require_gathered_batch(_model.gather(batch, device))
            user_idx = gb.user_idx
            item_idx = gb.item_idx
            rating = gb.rating
            tgt_output = gb.tgt_output
            domain_idx = gb.domain_idx
            sample_id = gb.sample_id
            ccv_packet = build_ccv_control_packet(gb, step5_innov_cfg)
            with odcr_cuda_bf16_autocast():
                pred_ratings = _model.recommend(user_idx, item_idx, domain_idx)
                pred_exps, *_ = _model.generate(
                    user_idx, item_idx, domain_idx, ccv_control_packet=ccv_packet
                )
            pred_texts = get_step5_tokenizer().batch_decode(pred_exps, skip_special_tokens=True)
            ref_texts = get_step5_tokenizer().batch_decode(tgt_output, skip_special_tokens=True)
            decode_wall += _time_perf.perf_counter() - _t0
            pr = pred_ratings.detach().cpu().tolist()
            gr = rating.detach().cpu().tolist()
            sids = sample_id.detach().cpu().tolist()
            for i in range(len(sids)):
                rows.append(
                    {
                        "sample_id": int(sids[i]),
                        "pred_rating": float(pr[i]),
                        "gt_rating": float(gr[i]),
                        "pred_text": pred_texts[i],
                        "ref_text": ref_texts[i],
                        "pred_token_ids": pred_exps[i].detach().cpu().tolist(),
                        "ref_token_ids": tgt_output[i].detach().cpu().tolist(),
                    }
                )
    return {"rows": rows, "timings": {"decode_time": float(decode_wall)}}


def _load_review_by_sample_id(csv_path: str) -> Tuple[List[str], Dict[str, Any]]:
    if not (csv_path or "").strip() or not os.path.isfile(csv_path):
        return [], {"fast_path": "missing_path", "review_rows_count": 0}
    try:
        df = pd.read_csv(csv_path, usecols=["review"])
    except Exception:
        return [], {"fast_path": "read_failed", "review_rows_count": 0}
    if "review" not in df.columns:
        return [], {"fast_path": "missing_review_col", "review_rows_count": 0}
    vals = df["review"].fillna("").astype(str).tolist()
    return vals, {"fast_path": "loaded", "review_rows_count": int(len(vals))}


def _count_tokens_before_eos(ids_list: List[int], eos_id: int) -> int:
    if eos_id is not None and eos_id >= 0 and eos_id in ids_list:
        return int(ids_list.index(eos_id))
    return len(ids_list)


def evalModelWithRerank(
    model,
    test_dataloader,
    device,
    *,
    num_return_sequences: int,
    rerank_method: str,
    rerank_top_k: int,
    rerank_weight_logprob: float,
    rerank_weight_length: float,
    rerank_weight_repeat: float,
    rerank_weight_dirty: float,
    rerank_target_len_ratio: float,
    rerank_malformed_tail_penalty: float,
    rerank_malformed_token_penalty: float,
    cli_seed: int,
    step5_innov_cfg,
    review_by_sample_id: Optional[Sequence[str]] = None,
    rerank_v3_profile: Optional[Dict[str, Any]] = None,
):
    """混合候选池 + rule_v1/v2/v3 rerank；最终 pred_text 与 evalModel 对齐。"""
    import time as _time_perf

    rm = (rerank_method or "").strip().lower().replace("-", "_")
    if rm not in ("rule_v1", "rule_v2", "rule_v3"):
        raise ValueError(
            f"不支持的 rerank_method={rerank_method!r}（支持 rule_v1 / rule_v2 / rule_v3）"
        )
    v3_prof = merge_rerank_v3_profile(rerank_v3_profile)
    _m = get_underlying_model(model).to(device)
    _m.eval()
    K = max(1, int(num_return_sequences))
    strategy = str(_m.decode_strategy).lower()
    _tok_eos = get_step5_tokenizer()
    eos_id = int(_tok_eos.eos_token_id) if _tok_eos.eos_token_id is not None else -1
    base_gc = _m._make_generate_config()
    fam = str(getattr(_m, "candidate_family", "balanced")).strip().lower()
    specs = build_candidate_generation_specs(
        base_gc,
        fam,
        k_cli=K,
        include_diverse=bool(getattr(_m, "candidate_mixed_include_diverse", True)),
    )
    cand_slot = 0
    rows: List[dict] = []
    batch_idx = 0
    decode_wall = 0.0
    feature_wall = 0.0
    score_wall = 0.0
    base_decode_seed = _m.decode_seed if _m.decode_seed is not None else int(cli_seed)
    review_rows = list(review_by_sample_id or [])
    with torch.no_grad():
        for batch in test_dataloader:
            gb = require_gathered_batch(_m.gather(batch, device))
            user_idx = gb.user_idx
            item_idx = gb.item_idx
            rating = gb.rating
            tgt_output = gb.tgt_output
            domain_idx = gb.domain_idx
            sample_id = gb.sample_id
            ccv_packet = build_ccv_control_packet(gb, step5_innov_cfg)
            ref_texts = get_step5_tokenizer().batch_decode(tgt_output, skip_special_tokens=True)
            B = int(user_idx.size(0))
            candidates_per_row: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
            _tdecode0 = _time_perf.perf_counter()
            with odcr_cuda_bf16_autocast():
                pred_ratings = _m.recommend(user_idx, item_idx, domain_idx)
                for fam_tag, cfg_ov in specs:
                    gen = None
                    if strategy == "nucleus":
                        gen = torch.Generator(device=device)
                        gen.manual_seed(
                            int(base_decode_seed) + cand_slot * 1_000_003 + batch_idx * 97
                        )
                    gen_ids, _, _, avg_lp = _m.generate_with_token_logprobs(
                        user_idx,
                        item_idx,
                        domain_idx,
                        generator=gen,
                        cfg_override=cfg_ov,
                        ccv_control_packet=ccv_packet,
                    )
                    pred_texts_k = get_step5_tokenizer().batch_decode(gen_ids, skip_special_tokens=True)
                    eff_cfg = cfg_ov if cfg_ov is not None else base_gc
                    gen_ids_cpu = gen_ids.detach().cpu()
                    for i in range(B):
                        ids_i = gen_ids_cpu[i].tolist()
                        n_tok = _count_tokens_before_eos(ids_i, eos_id)
                        alp = float(avg_lp[i].detach().item())
                        candidates_per_row[i].append(
                            {
                                "text": pred_texts_k[i],
                                "avg_logprob": alp,
                                "lp_norm": float(compute_lp_norm(alp, n_tok)),
                                "token_len": int(n_tok),
                                "candidate_family": fam_tag,
                                "effective_temperature": float(eff_cfg.temperature),
                                "effective_top_p": float(eff_cfg.top_p),
                            }
                        )
                    cand_slot += 1
            decode_wall += _time_perf.perf_counter() - _tdecode0
            pr = pred_ratings.detach().cpu().tolist()
            gr = rating.detach().cpu().tolist()
            sids = sample_id.detach().cpu().tolist()
            _tfeat0 = _time_perf.perf_counter()
            feat_cache: List[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = [[] for _ in range(B)]
            for i in range(B):
                ref = ref_texts[i]
                ref_w = max(len(ref.split()), 1)
                ref_mean_dirty = float(ref_w)
                sid = int(sids[i])
                review_txt = review_rows[sid] if 0 <= sid < len(review_rows) else ""
                review_kw = keywords_from_source_text(review_txt)
                for rank_before, c in enumerate(candidates_per_row[i]):
                    if rm == "rule_v3":
                        feats = extract_rerank_features_for_v3(
                            c["text"],
                            avg_logprob=c["avg_logprob"],
                            token_len=int(c["token_len"]),
                            ref_mean_len_words=18.0,
                        )
                        feats_out = dict(feats)
                        ok_hard, hard_rs, sc, bkd = score_candidates_rule_v3(
                            c["text"],
                            feats,
                            review_keywords=review_kw,
                            lp_norm=float(c["lp_norm"]),
                            profile=v3_prof,
                        )
                        feats_out["rerank_score"] = round(sc, 6)
                        feats_out["v3_hard_pass"] = bool(ok_hard)
                        feats_out["v3_hard_filter_reasons"] = list(hard_rs)
                        feats_out["v3_score_breakdown"] = {k: round(float(v), 6) for k, v in bkd.items()}
                        feats_out["length_deviation_penalty"] = float(
                            bkd.get("length_penalty_v2", 0.0) or 0.0
                        )
                        feats_out["malformed_tail_penalty"] = float(
                            bkd.get("malformed_tail_penalty", 0.0) or 0.0
                        )
                        feats_out["malformed_token_penalty"] = float(
                            bkd.get("malformed_token_penalty", 0.0) or 0.0
                        )
                        len_pen = float(feats_out["length_deviation_penalty"])
                    else:
                        feats = extract_rerank_features(
                            c["text"],
                            ref,
                            avg_logprob=c["avg_logprob"],
                            ref_mean_len_words=ref_mean_dirty,
                        )
                        feats_out = dict(feats)
                        if rm == "rule_v2":
                            sc, len_pen, bkd = score_candidates_rule_v2(
                                feats,
                                weight_logprob=rerank_weight_logprob,
                                weight_length=rerank_weight_length,
                                weight_repeat=rerank_weight_repeat,
                                weight_dirty=rerank_weight_dirty,
                                target_len_ratio=rerank_target_len_ratio,
                                coef_malformed_tail=float(rerank_malformed_tail_penalty),
                                coef_malformed_token=float(rerank_malformed_token_penalty),
                            )
                            tail_ap = (
                                float(rerank_malformed_tail_penalty)
                                if feats_out.get("malformed_tail_hit")
                                else 0.0
                            )
                            tok_ap = (
                                float(rerank_malformed_token_penalty)
                                if feats_out.get("malformed_token_hit")
                                else 0.0
                            )
                            feats_out["malformed_tail_penalty"] = round(tail_ap, 6)
                            feats_out["malformed_token_penalty"] = round(tok_ap, 6)
                            feats_out["rule_v2_score_breakdown"] = {
                                k: round(float(v), 6) for k, v in bkd.items()
                            }
                        else:
                            sc, len_pen = score_candidates_rule_v1(
                                feats,
                                weight_logprob=rerank_weight_logprob,
                                weight_length=rerank_weight_length,
                                weight_repeat=rerank_weight_repeat,
                                weight_dirty=rerank_weight_dirty,
                                target_len_ratio=rerank_target_len_ratio,
                            )
                            feats_out["malformed_tail_penalty"] = 0.0
                            feats_out["malformed_token_penalty"] = 0.0
                        feats_out["length_deviation_penalty"] = round(len_pen, 6)
                        feats_out["rerank_score"] = round(sc, 6)
                    feat_cache[i].append((dict(c), feats_out))
            feature_wall += _time_perf.perf_counter() - _tfeat0
            _tsc0 = _time_perf.perf_counter()
            for i in range(B):
                sid = int(sids[i])
                scored: List[Tuple[int, Dict[str, Any], float, float, Dict[str, Any]]] = []
                for rank_before, (c, feats_out) in enumerate(feat_cache[i]):
                    _rs = feats_out.get("rerank_score", 0.0)
                    sc = float(0.0 if _rs is None else _rs)
                    _lp = feats_out.get("length_deviation_penalty", 0.0)
                    len_pen = float(0.0 if _lp is None else _lp)
                    scored.append((rank_before, feats_out, sc, len_pen, dict(c)))
                scored.sort(key=lambda x: -x[2])
                take_k = max(1, int(rerank_top_k))
                top = scored[:take_k]
                best = top[0]
                best_lp_rank = max(
                    range(len(candidates_per_row[i])),
                    key=lambda j: candidates_per_row[i][j]["avg_logprob"],
                )
                sel_text = best[4]["text"]
                cand_payload = []
                selected_rank = int(best[0])
                for rank_before, feats_out, sc, _len_pen, cdict in scored:
                    entry = {
                        "candidate_rank_before_rerank": rank_before,
                        "candidate_text": cdict["text"] if os.environ.get("ODCR_RERANK_DEBUG", "0") == "1" else "",
                        "avg_logprob": cdict["avg_logprob"],
                        "lp_norm": cdict.get("lp_norm"),
                        "candidate_family": cdict.get("candidate_family"),
                        "effective_temperature": cdict.get("effective_temperature"),
                        "effective_top_p": cdict.get("effective_top_p"),
                        "token_len": cdict.get("token_len"),
                        "features": feats_out,
                        "rerank_score": feats_out["rerank_score"],
                        "selected_as_final": rank_before == selected_rank,
                    }
                    if rm == "rule_v3":
                        entry["v3_hard_pass"] = feats_out.get("v3_hard_pass")
                        entry["v3_hard_filter_reasons"] = feats_out.get("v3_hard_filter_reasons")
                        entry["v3_score_breakdown"] = feats_out.get("v3_score_breakdown")
                    cand_payload.append(entry)
                sel = best[1]
                rows.append(
                    {
                        "sample_id": sid,
                        "pred_rating": float(pr[i]),
                        "gt_rating": float(gr[i]),
                        "pred_text": sel_text,
                        "ref_text": ref,
                        "pred_token_ids": [],
                        "ref_token_ids": tgt_output[i].detach().cpu().tolist(),
                        "candidate_family": best[4].get("candidate_family"),
                        "lp_norm": best[4].get("lp_norm"),
                        "completion_ok": bool(
                            (sel.get("v3_score_breakdown") or {}).get("completion_score", 0) >= 0.9
                        )
                        if rm == "rule_v3"
                        else None,
                        "_rerank": {
                            "candidates": cand_payload,
                            "selected_rank_before_rerank": selected_rank,
                            "best_logprob_rank_before": int(best_lp_rank),
                            "rerank_method_effective": rm,
                            "selected": {
                                "text": sel_text,
                                "avg_logprob": candidates_per_row[i][selected_rank]["avg_logprob"],
                                "lp_norm": candidates_per_row[i][selected_rank].get("lp_norm"),
                                "candidate_family": candidates_per_row[i][selected_rank].get(
                                    "candidate_family"
                                ),
                                "rerank_score": sel["rerank_score"],
                                "repeat_penalty": sel.get("repeat_penalty"),
                                "dirty_penalty_diagnostic_only": sel.get(
                                    "dirty_penalty_diagnostic_only", sel.get("dirty_penalty")
                                ),
                                "length_deviation_penalty": sel.get("length_deviation_penalty"),
                                "pred_len_ratio": sel.get("pred_len_ratio"),
                                "malformed_tail_hit": bool(sel.get("malformed_tail_hit")),
                                "malformed_token_hit": bool(sel.get("malformed_token_hit")),
                                "malformed_tail_penalty": float(
                                    sel.get("malformed_tail_penalty", 0.0) or 0.0
                                ),
                                "malformed_token_penalty": float(
                                    sel.get("malformed_token_penalty", 0.0) or 0.0
                                ),
                                "v3_hard_pass": sel.get("v3_hard_pass"),
                                "v3_hard_filter_reasons": sel.get("v3_hard_filter_reasons"),
                                "v3_score_breakdown": sel.get("v3_score_breakdown"),
                            },
                            "best_by_logprob": {
                                "rank_before": int(best_lp_rank),
                                "text": candidates_per_row[i][best_lp_rank]["text"],
                                "avg_logprob": candidates_per_row[i][best_lp_rank]["avg_logprob"],
                            },
                        },
                    }
                )
            score_wall += _time_perf.perf_counter() - _tsc0
            batch_idx += 1
    return {
        "rows": rows,
        "timings": {
            "decode_time": float(decode_wall),
            "rerank_feature_time": float(feature_wall),
            "rerank_scoring_time": float(score_wall),
        },
    }


def _aggregate_rerank_summary(
    merged: List[dict],
    *,
    export_examples_mode: str,
    rerank_method: str,
) -> Dict[str, Any]:
    n = len(merged)
    if n <= 0:
        return {
            "num_samples": 0,
            "avg_candidate_count": 0.0,
            "mean_best_logprob_score": float("nan"),
            "mean_selected_rerank_score": float("nan"),
            "selected_not_best_logprob_rate": float("nan"),
            "mean_selected_len_ratio": float("nan"),
            "mean_selected_repeat_penalty": float("nan"),
            "mean_selected_dirty_penalty": float("nan"),
            "mean_selected_avg_logprob": float("nan"),
            "mean_selected_length_deviation_penalty": float("nan"),
            "mean_candidate_rouge_proxy": float("nan"),
            "mean_selected_malformed_tail_penalty": float("nan"),
            "mean_selected_malformed_token_penalty": float("nan"),
            "selected_malformed_tail_hit_rate": float("nan"),
            "selected_malformed_token_hit_rate": float("nan"),
            "export_examples_mode": export_examples_mode,
            "rerank_method": rerank_method,
            "completion_pass_rate": float("nan"),
            "well_formed_pass_rate": float("nan"),
            "source_coverage_mean": float("nan"),
            "entity_drift_hit_rate": float("nan"),
            "generic_template_hit_rate": float("nan"),
            "hard_filter_drop_rate": float("nan"),
        }

    def _float_field(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
        if key not in d:
            return default
        v = d[key]
        if v is None:
            return default
        return float(v)

    cand_counts: List[int] = []
    best_lps: List[float] = []
    sel_scores: List[float] = []
    not_best_lp = 0
    sel_lr: List[float] = []
    sel_rep: List[float] = []
    sel_dty: List[float] = []
    sel_alp: List[float] = []
    sel_ldp: List[float] = []
    sel_mtail: List[float] = []
    sel_mtok: List[float] = []
    hit_mtail = 0
    hit_mtok = 0
    rouge_proxies: List[float] = []
    rm_lc = (rerank_method or "").strip().lower().replace("-", "_")
    comp_pass = 0
    wf_pass = 0
    src_covs: List[float] = []
    drift_hits = 0
    gen_tmpl_hits = 0
    total_cands_v3 = 0
    hard_drop_cands = 0
    for r in merged:
        rr = r.get("_rerank") or {}
        cands = rr.get("candidates") or []
        cand_counts.append(len(cands))
        if cands:
            best_lps.append(
                max(
                    float(c["avg_logprob"]) if c.get("avg_logprob") is not None else float("-inf")
                    for c in cands
                )
            )
            rx = [rouge_l_proxy(str(c.get("candidate_text", "")), str(r.get("ref_text", ""))) for c in cands]
            rouge_proxies.append(sum(rx) / max(len(rx), 1))
        sel = rr.get("selected") or {}
        if sel:
            sel_scores.append(_float_field(sel, "rerank_score"))
            sel_lr.append(_float_field(sel, "pred_len_ratio"))
            sel_rep.append(_float_field(sel, "repeat_penalty"))
            dty_v = _float_field(sel, "dirty_penalty_diagnostic_only")
            if dty_v != dty_v:
                dty_v = _float_field(sel, "dirty_penalty")
            sel_dty.append(dty_v)
            sel_alp.append(_float_field(sel, "avg_logprob"))
            sel_ldp.append(_float_field(sel, "length_deviation_penalty"))
            mtp = float(sel.get("malformed_tail_penalty", 0.0) or 0.0)
            mkp = float(sel.get("malformed_token_penalty", 0.0) or 0.0)
            sel_mtail.append(mtp)
            sel_mtok.append(mkp)
            if bool(sel.get("malformed_tail_hit")) or mtp > 0:
                hit_mtail += 1
            if bool(sel.get("malformed_token_hit")) or mkp > 0:
                hit_mtok += 1
        br = int(rr.get("best_logprob_rank_before", -1))
        sr = int(rr.get("selected_rank_before_rerank", -2))
        if br >= 0 and sr >= 0 and br != sr:
            not_best_lp += 1
        if rm_lc == "rule_v3" and cands:
            for c in cands:
                total_cands_v3 += 1
                if c.get("v3_hard_pass") is False:
                    hard_drop_cands += 1
        if sel and rm_lc == "rule_v3":
            bd_sel = sel.get("v3_score_breakdown") or {}
            if bd_sel:
                if float(bd_sel.get("completion_score", 0) or 0) >= 0.9:
                    comp_pass += 1
                if float(bd_sel.get("well_formed_score", 0) or 0) >= 0.65:
                    wf_pass += 1
                scv = _float_field(bd_sel, "source_coverage_score")
                if scv == scv:
                    src_covs.append(scv)
                if float(bd_sel.get("entity_drift_penalty", 0) or 0) >= 0.55:
                    drift_hits += 1
                _gt_raw = bd_sel.get("generic_template_penalty_raw", bd_sel.get("generic_template_penalty"))
                if _gt_raw is not None and float(_gt_raw) >= 0.99:
                    gen_tmpl_hits += 1

    def _mean(xs: List[float]) -> float:
        v = [x for x in xs if x == x]
        return float(sum(v) / max(len(v), 1)) if v else float("nan")

    return {
        "num_samples": n,
        "avg_candidate_count": float(sum(cand_counts) / max(n, 1)),
        "mean_best_logprob_score": _mean(best_lps),
        "mean_selected_rerank_score": _mean(sel_scores),
        "selected_not_best_logprob_rate": float(not_best_lp) / float(n),
        "mean_selected_len_ratio": _mean(sel_lr),
        "mean_selected_repeat_penalty": _mean(sel_rep),
        "mean_selected_dirty_penalty": _mean(sel_dty),
        "mean_selected_avg_logprob": _mean(sel_alp),
        "mean_selected_length_deviation_penalty": _mean(sel_ldp),
        "mean_candidate_rouge_proxy": _mean(rouge_proxies),
        "mean_selected_malformed_tail_penalty": _mean(sel_mtail),
        "mean_selected_malformed_token_penalty": _mean(sel_mtok),
        "selected_malformed_tail_hit_rate": float(hit_mtail) / float(n),
        "selected_malformed_token_hit_rate": float(hit_mtok) / float(n),
        "export_examples_mode": export_examples_mode,
        "rerank_method": rerank_method,
        "completion_pass_rate": float(comp_pass) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "well_formed_pass_rate": float(wf_pass) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "source_coverage_mean": _mean(src_covs) if rm_lc == "rule_v3" else float("nan"),
        "entity_drift_hit_rate": float(drift_hits) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "generic_template_hit_rate": float(gen_tmpl_hits) / float(n) if rm_lc == "rule_v3" else float("nan"),
        "hard_filter_drop_rate": float(hard_drop_cands) / float(max(total_cands_v3, 1))
        if rm_lc == "rule_v3" and total_cands_v3 > 0
        else float("nan"),
    }


def _rerank_eval_cli_resolved(args: Any) -> Dict[str, Any]:
    """Return rerank settings transported by the One-Control resolver.

    Direct/helper eval-rerank invocations used to fall back to literals here.
    Phase 4A makes that a hard gate: every value must be injected by
    ``code/odcr.py`` from ``configs/odcr.yaml`` and the resolved profile JSON.
    """
    required_attrs = (
        "num_return_sequences",
        "rerank_method",
        "rerank_top_k",
        "rerank_weight_logprob",
        "rerank_weight_length",
        "rerank_weight_repeat",
        "rerank_weight_dirty",
        "rerank_target_len_ratio",
        "export_examples_mode",
        "rerank_malformed_tail_penalty",
        "rerank_malformed_token_penalty",
    )
    missing = [
        name
        for name in required_attrs
        if getattr(args, name, None) is None or str(getattr(args, name, "")).strip() == ""
    ]
    raw_profile = (os.environ.get("ODCR_RERANK_PROFILE_JSON") or "").strip()
    if missing or not raw_profile:
        raise RuntimeError(
            "eval-rerank requires resolver-owned One-Control rerank config; "
            f"missing_cli_transport={missing}, has_ODCR_RERANK_PROFILE_JSON={bool(raw_profile)}. "
            "Use `python code/odcr.py eval --profile <profile-with-rerank>`."
        )
    try:
        profile_obj = json.loads(raw_profile)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ODCR_RERANK_PROFILE_JSON must be valid JSON: {exc}") from exc
    if not isinstance(profile_obj, dict) or not profile_obj:
        raise RuntimeError("ODCR_RERANK_PROFILE_JSON must be a non-empty object from One-Control resolver.")

    num_ret = max(1, int(getattr(args, "num_return_sequences")))
    rm_s = str(getattr(args, "rerank_method")).strip()
    top_k = max(1, int(getattr(args, "rerank_top_k")))
    w_lp = float(getattr(args, "rerank_weight_logprob"))
    w_len = float(getattr(args, "rerank_weight_length"))
    w_rep = float(getattr(args, "rerank_weight_repeat"))
    w_drt = float(getattr(args, "rerank_weight_dirty"))
    tlr = float(getattr(args, "rerank_target_len_ratio"))
    ex_mode = str(getattr(args, "export_examples_mode")).strip().lower()
    mtail = float(getattr(args, "rerank_malformed_tail_penalty"))
    mtok = float(getattr(args, "rerank_malformed_token_penalty"))
    return {
        "num_return_sequences": num_ret,
        "rerank_method": rm_s,
        "rerank_top_k": top_k,
        "rerank_weight_logprob": w_lp,
        "rerank_weight_length": w_len,
        "rerank_weight_repeat": w_rep,
        "rerank_weight_dirty": w_drt,
        "rerank_target_len_ratio": tlr,
        "export_examples_mode": ex_mode,
        "rerank_malformed_tail_penalty": mtail,
        "rerank_malformed_token_penalty": mtok,
        "rerank_profile": profile_obj,
        "rerank_source_table": {
            "ODCR_RERANK_PROFILE_JSON": "configs/odcr.yaml: eval.profiles.*.rerank -> eval.rerank.*",
            "cli_transport": "odcr_core.config_resolver.ResolvedConfig -> odcr_core.runners._rerank_runner_cli_args",
        },
    }


def _eval_rows_local(
    model,
    dl,
    device,
    args,
    *,
    review_rows: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], Dict[str, float]]:
    _st5_raw = str(getattr(args, "_odcr_step5_innovation_config_json", "") or "").strip()
    if not _st5_raw:
        raise RuntimeError("eval requires resolved Step5 LCI/UCI/CCV/FCA config JSON from One-Control.")
    step5_innov_cfg = parse_step5_innovation_config_json(_st5_raw)
    if str(args.command) == "eval-rerank":
        _rr = _rerank_eval_cli_resolved(args)
        v3p = dict(_rr.get("rerank_profile") or {})
        out = evalModelWithRerank(
            model,
            dl,
            device,
            num_return_sequences=int(_rr["num_return_sequences"]),
            rerank_method=str(_rr["rerank_method"]),
            rerank_top_k=int(_rr["rerank_top_k"]),
            rerank_weight_logprob=float(_rr["rerank_weight_logprob"]),
            rerank_weight_length=float(_rr["rerank_weight_length"]),
            rerank_weight_repeat=float(_rr["rerank_weight_repeat"]),
            rerank_weight_dirty=float(_rr["rerank_weight_dirty"]),
            rerank_target_len_ratio=float(_rr["rerank_target_len_ratio"]),
            rerank_malformed_tail_penalty=float(_rr["rerank_malformed_tail_penalty"]),
            rerank_malformed_token_penalty=float(_rr["rerank_malformed_token_penalty"]),
            cli_seed=int(args.seed),
            step5_innov_cfg=step5_innov_cfg,
            review_by_sample_id=review_rows,
            rerank_v3_profile=v3p,
        )
        return out["rows"], dict(out.get("timings") or {})
    out_m = evalModel(model, dl, device, step5_innov_cfg=step5_innov_cfg)
    return out_m["rows"], dict(out_m.get("timings") or {})


_RERANK_TOP2_GAP = 0.025


def _rerank_compact_cand(c: Dict[str, Any], *, text_max: int = 200) -> Dict[str, Any]:
    fe = c.get("features") or {}
    t = str(c.get("candidate_text") or "")
    if len(t) > text_max:
        t = t[:text_max] + "…"
    out: Dict[str, Any] = {
        "candidate_rank_before_rerank": c.get("candidate_rank_before_rerank"),
        "avg_logprob": c.get("avg_logprob"),
        "rerank_score": c.get("rerank_score"),
        "pred_len_ratio": fe.get("pred_len_ratio"),
        "dirty_penalty_diagnostic_only": fe.get(
            "dirty_penalty_diagnostic_only", fe.get("dirty_penalty")
        ),
        "malformed_tail_penalty": fe.get("malformed_tail_penalty", 0),
        "malformed_token_penalty": fe.get("malformed_token_penalty", 0),
        "text_preview": t,
    }
    bd = fe.get("rule_v2_score_breakdown")
    if bd:
        out["rule_v2_score_breakdown"] = bd
    dr = fe.get("dirty_detail_v2")
    if isinstance(dr, dict):
        out["dirty_detail_v2"] = {"active_rules": dr.get("active_rules", [])}
    return out


def _rerank_build_changed_only_row(r: dict) -> Tuple[bool, Dict[str, Any]]:
    rr = r.get("_rerank") or {}
    cands = list(rr.get("candidates") or [])
    sid = int(r["sample_id"])
    sel = rr.get("selected") or {}
    best_lp = rr.get("best_by_logprob") or {}
    br = int(rr.get("best_logprob_rank_before", -1))
    sr = int(rr.get("selected_rank_before_rerank", -2))
    selected_not_best_lp = bool(br >= 0 and sr >= 0 and br != sr)

    scores = sorted(
        [float((c.get("rerank_score") if c.get("rerank_score") is not None else 0.0) or 0.0) for c in cands],
        reverse=True,
    )
    top12_gap = (scores[0] - scores[1]) if len(scores) >= 2 else 1.0
    tight_top2 = bool(len(scores) >= 2 and top12_gap < _RERANK_TOP2_GAP)

    if cands:
        lp_winner = max(
            cands,
            key=lambda x: float(x.get("avg_logprob"))
            if x.get("avg_logprob") is not None
            else float("-inf"),
        )
        lp_rank = int(lp_winner.get("candidate_rank_before_rerank", -1))
    else:
        lp_rank = -1
    rerank_differs_from_lp_winner = bool(lp_rank >= 0 and sr >= 0 and lp_rank != sr)

    dirty = float(sel.get("dirty_penalty_diagnostic_only", sel.get("dirty_penalty", 0.0)) or 0.0)
    ldp = float(sel.get("length_deviation_penalty", 0.0) or 0.0)
    mtail_ap = float(sel.get("malformed_tail_penalty", 0.0) or 0.0)
    mtok_ap = float(sel.get("malformed_token_penalty", 0.0) or 0.0)
    mtail_hit = bool(sel.get("malformed_tail_hit"))
    mtok_hit = bool(sel.get("malformed_token_hit"))

    veto_clean_lp = False
    if dirty >= 0.15 and cands:
        lp_text = str(best_lp.get("text") or "")
        lp_c = next((c for c in cands if str(c.get("candidate_text") or "") == lp_text), None)
        if lp_c:
            lp_d = float(
                (
                    (lp_c.get("features") or {}).get(
                        "dirty_penalty_diagnostic_only",
                        (lp_c.get("features") or {}).get("dirty_penalty", 1.0),
                    )
                )
                or 1.0
            )
            if lp_d < 0.05:
                veto_clean_lp = True

    flags: List[str] = []
    if selected_not_best_lp:
        flags.append("selected_not_best_logprob")
    if dirty > 0:
        flags.append("selected_dirty")
    if mtail_ap > 0 or mtail_hit:
        flags.append("malformed_tail")
    if mtok_ap > 0 or mtok_hit:
        flags.append("malformed_token")
    if ldp > 0.30:
        flags.append("length_dev_high")
    if rerank_differs_from_lp_winner:
        flags.append("rerank_not_logprob_winner")
    if tight_top2:
        flags.append("tight_top2_scores")
    if veto_clean_lp:
        flags.append("veto_dirty_selected_vs_clean_logprob")

    include = bool(
        selected_not_best_lp
        or dirty > 0
        or mtail_ap > 0
        or mtok_ap > 0
        or mtail_hit
        or mtok_hit
        or ldp > 0.30
        or rerank_differs_from_lp_winner
        or tight_top2
        or veto_clean_lp
    )
    scored_c = sorted(
        cands,
        key=lambda x: -float(x["rerank_score"])
        if x.get("rerank_score") is not None
        else float("inf"),
    )
    top_compact = [_rerank_compact_cand(x) for x in scored_c[:3]]
    rec: Dict[str, Any] = {
        "sample_id": sid,
        "reference": r.get("ref_text"),
        "selected": {
            "text": sel.get("text"),
            "avg_logprob": sel.get("avg_logprob"),
            "rerank_score": sel.get("rerank_score"),
            "dirty_penalty_diagnostic_only": sel.get(
                "dirty_penalty_diagnostic_only", sel.get("dirty_penalty")
            ),
            "malformed_tail_penalty": mtail_ap,
            "malformed_token_penalty": mtok_ap,
            "length_deviation_penalty": ldp,
        },
        "best_by_logprob": best_lp,
        "comparison": {
            "rerank_selected_text": sel.get("text"),
            "logprob_winner_text": best_lp.get("text"),
            "selected_not_best_logprob": selected_not_best_lp,
            "top1_top2_rerank_gap": round(top12_gap, 6),
        },
        "top_candidates_compact": top_compact,
        "analysis_flags": flags,
    }
    return include, rec


def _rerank_head50_sort_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    flags = set(rec.get("analysis_flags") or [])
    sel = rec.get("selected") or {}
    _g = (rec.get("comparison") or {}).get("top1_top2_rerank_gap", 1.0)
    gap = float(1.0 if _g is None else _g)
    dirty = float(sel.get("dirty_penalty_diagnostic_only", sel.get("dirty_penalty", 0.0)) or 0.0)
    return (
        0 if "malformed_token" in flags else 1,
        0 if "malformed_tail" in flags else 1,
        0 if "selected_not_best_logprob" in flags else 1,
        -dirty,
        gap,
    )


def _write_rerank_artifacts(
    eval_sub: str,
    merged: List[dict],
    *,
    rerank_cfg: Dict[str, Any],
    rerank_summary: Dict[str, Any],
    export_examples_mode: str,
    export_full_rerank_examples: bool,
) -> None:
    import csv as _csv

    mode = (export_examples_mode or "head50").strip().lower()
    cand_path = os.path.join(eval_sub, "rerank_candidates.csv")
    fieldnames = [
        "sample_id",
        "candidate_rank_before_rerank",
        "candidate_text",
        "avg_logprob",
        "pred_len_words",
        "pred_len_ratio",
        "repeat_penalty",
        "dirty_penalty_diagnostic_only",
        "length_deviation_penalty",
        "malformed_tail_penalty",
        "malformed_token_penalty",
        "rerank_score",
        "selected_as_final",
    ]
    full_samples: List[Dict[str, Any]] = []
    with open(cand_path, "w", newline="", encoding="utf-8") as cf:
        w = _csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in merged:
            sid = int(r["sample_id"])
            rr = r.get("_rerank") or {}
            for c in rr.get("candidates") or []:
                fe = c.get("features") or {}
                w.writerow(
                    {
                        "sample_id": sid,
                        "candidate_rank_before_rerank": c.get("candidate_rank_before_rerank"),
                        "candidate_text": c.get("candidate_text"),
                        "avg_logprob": fe.get("avg_logprob"),
                        "pred_len_words": fe.get("pred_len_words"),
                        "pred_len_ratio": fe.get("pred_len_ratio"),
                        "repeat_penalty": fe.get("repeat_penalty"),
                        "dirty_penalty_diagnostic_only": fe.get(
                            "dirty_penalty_diagnostic_only", fe.get("dirty_penalty")
                        ),
                        "length_deviation_penalty": fe.get("length_deviation_penalty"),
                        "malformed_tail_penalty": fe.get("malformed_tail_penalty", 0),
                        "malformed_token_penalty": fe.get("malformed_token_penalty", 0),
                        "rerank_score": c.get("rerank_score"),
                        "selected_as_final": c.get("selected_as_final"),
                    }
                )
            full_samples.append(
                {
                    "sample_id": sid,
                    "reference": r.get("ref_text"),
                    "candidates": rr.get("candidates"),
                    "selected": rr.get("selected"),
                    "best_by_logprob": rr.get("best_by_logprob"),
                    "comparison": {
                        "rerank_selected_text": (rr.get("selected") or {}).get("text"),
                        "logprob_winner_text": (rr.get("best_by_logprob") or {}).get("text"),
                    },
                }
            )

    want_light = mode in ("changed_only", "head20", "head50")
    changed_records: List[Dict[str, Any]] = []
    if want_light:
        for r in merged:
            inc, rec = _rerank_build_changed_only_row(r)
            if inc:
                changed_records.append(rec)
        jlp = os.path.join(eval_sub, "rerank_examples_changed_only.jsonl")
        with open(jlp, "w", encoding="utf-8") as jlf:
            for rec in changed_records:
                jlf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        idx_p = os.path.join(eval_sub, "rerank_examples_changed_only_index.csv")
        with open(idx_p, "w", newline="", encoding="utf-8") as ixf:
            iw = _csv.DictWriter(
                ixf,
                fieldnames=[
                    "sample_id",
                    "selected_not_best_logprob",
                    "dirty_penalty_diagnostic_only",
                    "malformed_tail_penalty",
                    "malformed_token_penalty",
                    "flags_summary",
                ],
            )
            iw.writeheader()
            for rec in changed_records:
                comp = rec.get("comparison") or {}
                sel = rec.get("selected") or {}
                iw.writerow(
                    {
                        "sample_id": rec.get("sample_id"),
                        "selected_not_best_logprob": comp.get("selected_not_best_logprob"),
                        "dirty_penalty_diagnostic_only": sel.get(
                            "dirty_penalty_diagnostic_only", sel.get("dirty_penalty")
                        ),
                        "malformed_tail_penalty": sel.get("malformed_tail_penalty"),
                        "malformed_token_penalty": sel.get("malformed_token_penalty"),
                        "flags_summary": ";".join(rec.get("analysis_flags") or []),
                    }
                )
        _head_limits = {"head20": 20, "head50": 50}
        if mode in _head_limits:
            n = _head_limits[mode]
            head = sorted(changed_records, key=_rerank_head50_sort_key)[:n]
            with open(
                os.path.join(eval_sub, f"rerank_examples_{mode}.json"),
                "w",
                encoding="utf-8",
            ) as hf:
                json.dump(head, hf, ensure_ascii=False, indent=2, default=str)

    if export_full_rerank_examples:
        gz_path = os.path.join(eval_sub, "rerank_examples.json.gz")
        blob = json.dumps(
            {
                "rerank_cfg": rerank_cfg,
                "rerank_summary": rerank_summary,
                "samples": full_samples,
            },
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        with gzip.open(gz_path, "wb", compresslevel=6) as gzf:
            gzf.write(blob)


def _resolve_odcr_profile_paths(index_contract: Mapping[str, Any]) -> Dict[str, str]:
    def _req_path(key: str) -> str:
        p = index_contract.get(key)
        if not p or not str(p).strip():
            raise IndexContractError(f"index_contract 缺少双通道路径字段 {key!r}")
        return os.path.abspath(str(p))

    target_dir = os.path.dirname(_req_path("target_user_content_profiles_path"))
    aux_dir = os.path.dirname(_req_path("aux_user_content_profiles_path"))
    return {
        "target_user_content": os.path.join(target_dir, "user_content_profiles.npy"),
        "target_user_style": os.path.join(target_dir, "user_style_profiles.npy"),
        "aux_user_content": os.path.join(aux_dir, "user_content_profiles.npy"),
        "aux_user_style": os.path.join(aux_dir, "user_style_profiles.npy"),
        "target_item_content": os.path.join(target_dir, "item_content_profiles.npy"),
        "target_item_style": os.path.join(target_dir, "item_style_profiles.npy"),
        "aux_item_content": os.path.join(aux_dir, "item_content_profiles.npy"),
        "aux_item_style": os.path.join(aux_dir, "item_style_profiles.npy"),
        "target_domain_content": os.path.join(target_dir, "domain_content.npy"),
        "target_domain_style": os.path.join(target_dir, "domain_style.npy"),
        "aux_domain_content": os.path.join(aux_dir, "domain_content.npy"),
        "aux_domain_style": os.path.join(aux_dir, "domain_style.npy"),
    }


def _load_odcr_profile_tensors_from_contract(
    index_contract: Mapping[str, Any],
    device_idx: int | str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    paths = _resolve_odcr_profile_paths(index_contract)
    required_paths = list(paths.values())
    missing = [p for p in required_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "ODCR Step5 需要完整的双通道 profile/domain 文件，以下路径缺失：\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    dc, ds, uc, us, ic, ist = load_profile_tensors_from_contract(index_contract, device_idx)
    if int(dc.shape[-1]) != int(uc.shape[-1]) or int(ic.shape[-1]) != int(uc.shape[-1]):
        raise ValueError(
            "ODCR dual-channel profile hidden_size 不一致："
            f"domain={tuple(dc.shape)} user={tuple(uc.shape)} item={tuple(ic.shape)}"
        )
    meta = {
        "profile_mode": "odcr_dual_channel",
        "profile_consumption": "physical_separate",
        "selected_paths": {
            "user_content": [paths["target_user_content"], paths["aux_user_content"]],
            "user_style": [paths["target_user_style"], paths["aux_user_style"]],
            "item_content": [paths["target_item_content"], paths["aux_item_content"]],
            "item_style": [paths["target_item_style"], paths["aux_item_style"]],
            "domain_content_style": [
                paths["target_domain_content"],
                paths["target_domain_style"],
                paths["aux_domain_content"],
                paths["aux_domain_style"],
            ],
        },
    }
    return dc, ds, uc, us, ic, ist, meta


def _make_model(final_cfg: FinalTrainingConfig, args, device_idx):
    ic = getattr(args, "_odcr_index_contract", None)
    if ic is None:
        raise RuntimeError(
            "缺少 args._odcr_index_contract：Step5 模型须由 index_contract.json 驱动 profile 加载；"
            "请确认已先执行 build_odcr_ddp_artefacts 并成功读取契约。"
        )
    dc, ds, uc, us, ic, ist, _meta = _load_odcr_profile_tensors_from_contract(ic, device_idx)
    m = Model(
        final_cfg.nuser,
        final_cfg.nitem,
        final_cfg.ntoken,
        final_cfg.emsize,
        final_cfg.nhead,
        final_cfg.nhid,
        final_cfg.nlayers,
        final_cfg.dropout,
        uc,
        us,
        ic,
        ist,
        dc,
        ds,
        label_smoothing=float(final_cfg.label_smoothing),
        step5_innovation_config_json=str(getattr(final_cfg, "step5_innovation_config_json", "") or ""),
    ).to(device_idx)
    m.apply_runtime_config(final_cfg, get_step5_tokenizer())
    _tm = str(getattr(final_cfg, "train_mode", "full")).strip().lower()
    if _tm == "lora":
        _lt = tuple(getattr(final_cfg, "lora_target_modules", ()) or ())
        _flt = discover_flan_explainer_lora_targets(m, parent="flan_explainer")
        if len(_lt) > 0:
            _ov = list(dict.fromkeys(list(_lt) + [x for x in _flt if x not in set(_lt)]))
        else:
            _disc = discover_step5_text_linear_targets(m)
            _ov = list(dict.fromkeys(list(_disc) + _flt))
        _peft_meta = apply_native_lora_to_step5_model(
            m,
            r=int(getattr(final_cfg, "lora_r", 16)),
            alpha=float(getattr(final_cfg, "lora_alpha", 32.0)),
            dropout=float(getattr(final_cfg, "lora_dropout", 0.05)),
            target_modules_override=_ov,
        )
        setattr(args, "_odcr_step5_peft_meta", _peft_meta)
    elif _tm == "full":
        setattr(
            args,
            "_odcr_step5_peft_meta",
            {
                "enabled": False,
                "type": "full",
                "implementation": "",
                "r": int(getattr(final_cfg, "lora_r", 16)),
                "alpha": float(getattr(final_cfg, "lora_alpha", 32.0)),
                "dropout": float(getattr(final_cfg, "lora_dropout", 0.05)),
                "target_modules": None,
            },
        )
    else:
        raise RuntimeError(
            f"Step5 train_mode={_tm!r} 非法；须为 lora 或 full（禁止静默回退）。"
        )
    return m


def _load_step5_checkpoint_fail_fast(model: nn.Module, checkpoint_path: str, final_cfg: FinalTrainingConfig, device_idx: int) -> None:
    lineage = read_checkpoint_lineage(checkpoint_path, expected_stage="step5")
    expected = _current_step5_checkpoint_expectation(final_cfg, model)
    for key, expected_value in expected.items():
        if lineage.get(key) != expected_value:
            raise CheckpointLineageError(
                f"Step5 eval/rerank refused checkpoint due to compatibility mismatch for {key}: "
                f"checkpoint={lineage.get(key)!r} current={expected_value!r}"
            )
    state = torch.load(
        checkpoint_path,
        map_location=f"cuda:{device_idx}",
        weights_only=True,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        arch = {
            "emsize": int(final_cfg.emsize),
            "nlayers": int(final_cfg.nlayers),
            "nhead": int(final_cfg.nhead),
            "nhid": int(final_cfg.nhid),
            "dropout": float(final_cfg.dropout),
        }
        raise RuntimeError(
            "Step5 checkpoint load failed fast; checkpoint tensors do not match "
            f"resolved step5.model architecture {arch}. checkpoint={checkpoint_path}"
        ) from exc


def _tokenizer_cache_identity(tok) -> str:
    nop = getattr(tok, "name_or_path", None) or getattr(tok, "name", None)
    if nop:
        return str(nop)
    return type(tok).__name__


def _build_tokenize_cache_fingerprint(
    *,
    train_path: str,
    eval_split_path: str,
    task_idx: int,
    split_label: str,
    tok,
    max_length: int,
    cache_version: str,
    eval_only: bool,
    index_contract_path: str | None = None,
    step4_export_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_files: dict[str, Any] = {
        "eval_split": file_fingerprint(eval_split_path),
    }
    if not eval_only:
        source_files["train"] = file_fingerprint(train_path)
    primary_source_fp = source_files["eval_split"] if eval_only else source_files["train"]
    index_contract_fp = file_fingerprint(index_contract_path) if index_contract_path else None
    tokenizer_path = require_step5_text_model_dir()
    tokenizer_fp = model_artifact_fingerprint(tokenizer_path)
    effective_payload = current_effective_payload(required=True)
    resolved_step5_config_hash = current_one_control_resolved_config_hash(
        extra={
            "stage": "step5",
            "task_id": int(task_idx),
            "artifact": "tokenize_cache",
            "split_label": str(split_label),
            "eval_only": bool(eval_only),
        }
    )
    payload: dict[str, Any] = {
        "schema_version": STEP5_TOKENIZE_CACHE_SCHEMA_VERSION,
        "cache_version": str(cache_version),
        "stage": "step5",
        "task_id": int(task_idx),
        "split_label": str(split_label),
        "eval_only": bool(eval_only),
        "source_files": source_files,
        "source_step4_export_path": str(primary_source_fp.get("path") or os.path.abspath(eval_split_path if eval_only else train_path)),
        "source_step4_export_sha256": str(primary_source_fp.get("sha256") or ""),
        "index_contract": index_contract_fp,
        "index_contract_hash": str((index_contract_fp or {}).get("sha256") or ""),
        "step4_export_lineage_hash": str((step4_export_lineage or {}).get("lineage_hash") or ""),
        "tokenizer": {
            "identity": _tokenizer_cache_identity(tok),
            "model_path": os.path.abspath(tokenizer_path),
            "artifact_fingerprint": tokenizer_fp,
        },
        "tokenizer_path_or_id": os.path.abspath(tokenizer_path),
        "tokenizer_fingerprint": tokenizer_fp,
        "processor": {
            "name": "executors.step5_engine.Processor",
            "max_length": int(max_length),
            "dynamic_padding": "collate_time",
            "required_fields": list(_STEP5_TOKENIZE_REQUIRED_FIELDS),
        },
        "max_length": int(max_length),
        "resolved_step5_config_hash": resolved_step5_config_hash,
        "one_control_resolved_config_hash": resolved_step5_config_hash,
        "step5_innovation_config_hash": stable_hash(effective_payload.get("step5_innovation") or {}),
        "required_fields_hash": stable_hash(_STEP5_TOKENIZE_REQUIRED_FIELDS),
        "producer_code_version": STEP5_TOKENIZE_CACHE_PRODUCER_CODE_VERSION,
        "training_semantic_fingerprint": os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT", ""),
        "generation_semantic_fingerprint": os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT", ""),
    }
    if eval_only:
        _eval_control_contract = step5_factual_eval_control_contract(split_label)
        payload["eval_control_contract"] = _eval_control_contract
        payload["eval_control_contract_hash"] = stable_hash(_eval_control_contract)
    else:
        payload["eval_control_contract"] = None
        payload["eval_control_contract_hash"] = ""
    payload["fingerprint_hash"] = stable_hash(payload)
    return payload


def _build_step5_cache_dir(
    ckpt_task_dir: str,
    train_path: str,
    eval_split_path: str,
    processor,
    tok,
    *,
    task_idx: int,
    split_label: str,
    eval_only: bool,
    cache_version: str = ODCR_TOKENIZE_CACHE_VERSION,
    index_contract_path: str | None = None,
    step4_export_lineage: Mapping[str, Any] | None = None,
) -> Tuple[str, str, dict[str, Any]]:
    fp_payload = _build_tokenize_cache_fingerprint(
        train_path=train_path,
        eval_split_path=eval_split_path,
        task_idx=task_idx,
        split_label=split_label,
        tok=tok,
        max_length=int(getattr(processor, "max_length", 25)),
        cache_version=cache_version,
        eval_only=eval_only,
        index_contract_path=index_contract_path,
        step4_export_lineage=step4_export_lineage,
    )
    prefix = "hf_cache_step5_eval" if eval_only else "hf_cache_step5"
    fp = f"{cache_version}_{stable_hash(fp_payload, length=16)}"
    cache_dir = os.path.join(ckpt_task_dir, f"{prefix}_{fp}")
    return cache_dir, fp, fp_payload


def _dist_barrier_if_initialized() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _hf_dataset_cache_ready(cache_dir: str) -> bool:
    return os.path.isdir(cache_dir) and os.path.isfile(os.path.join(cache_dir, "dataset_dict.json"))


def _step5_tokenize_cache_manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, STEP5_TOKENIZE_CACHE_MANIFEST)


def _load_step5_tokenize_cache_manifest(cache_dir: str) -> dict[str, Any] | None:
    path = _step5_tokenize_cache_manifest_path(cache_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _step5_tokenize_cache_manifest_gate_fields(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cache_schema_version": STEP5_TOKENIZE_CACHE_SCHEMA_VERSION,
        "schema_version": STEP5_TOKENIZE_CACHE_SCHEMA_VERSION,
        "cache_version": ODCR_TOKENIZE_CACHE_VERSION,
        "stage": "step5",
        "task_id": int(fingerprint.get("task_id", -1)),
        "source_step4_export_path": str(fingerprint.get("source_step4_export_path") or ""),
        "source_step4_export_sha256": str(fingerprint.get("source_step4_export_sha256") or ""),
        "step4_export_lineage_hash": str(fingerprint.get("step4_export_lineage_hash") or ""),
        "index_contract_hash": str(fingerprint.get("index_contract_hash") or ""),
        "tokenizer_path_or_id": str(fingerprint.get("tokenizer_path_or_id") or ""),
        "tokenizer_fingerprint": fingerprint.get("tokenizer_fingerprint"),
        "max_length": int(fingerprint.get("max_length", -1)),
        "resolved_step5_config_hash": str(fingerprint.get("resolved_step5_config_hash") or ""),
        "step5_innovation_config_hash": str(fingerprint.get("step5_innovation_config_hash") or ""),
        "required_fields_hash": str(fingerprint.get("required_fields_hash") or ""),
        "eval_control_contract_hash": str(fingerprint.get("eval_control_contract_hash") or ""),
        "producer_code_version": STEP5_TOKENIZE_CACHE_PRODUCER_CODE_VERSION,
    }


def _step5_tokenize_cache_manifest_matches(
    cache_dir: str,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> tuple[bool, str]:
    if not _hf_dataset_cache_ready(cache_dir):
        return False, "missing_dataset"
    manifest = _load_step5_tokenize_cache_manifest(cache_dir)
    if manifest is None:
        return False, "missing_manifest"
    if str(manifest.get("schema_version")) != STEP5_TOKENIZE_CACHE_SCHEMA_VERSION:
        return False, "schema_mismatch"
    if str(manifest.get("cache_version")) != ODCR_TOKENIZE_CACHE_VERSION:
        return False, "version_mismatch"
    expected_hash = str(expected_fingerprint.get("fingerprint_hash") or "")
    if str(manifest.get("fingerprint_hash") or "") != expected_hash:
        return False, "fingerprint_mismatch"
    expected_gate = _step5_tokenize_cache_manifest_gate_fields(expected_fingerprint)
    for key, expected_value in expected_gate.items():
        if manifest.get(key) != expected_value:
            return False, f"{key}_mismatch"
    return True, "hit"


def _write_step5_tokenize_cache_manifest(
    cache_dir: str,
    *,
    fingerprint: Mapping[str, Any],
    splits: Mapping[str, int],
) -> None:
    atomic_write_json(
        _step5_tokenize_cache_manifest_path(cache_dir),
        {
            **_step5_tokenize_cache_manifest_gate_fields(fingerprint),
            "fingerprint_hash": str(fingerprint.get("fingerprint_hash") or ""),
            "fingerprint": dict(fingerprint),
            "splits": {str(k): int(v) for k, v in splits.items()},
            "dataset_format": "huggingface_dataset_dict_save_to_disk",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )


def _log_step5_tokenize_line(msg: str) -> None:
    lg = logging.getLogger(LOGGER_NAME)
    if not logger_has_file_handler(lg):
        print(msg, flush=True)
    if lg.handlers:
        lg.info(msg)
    else:
        logging.info(msg)


def _log_tokenize_map_done_step5(phase: str, nproc: int, elapsed_s: float) -> None:
    msg = f"[Tokenize] {phase} 完成 | num_proc={nproc} | wall_time={elapsed_s:.2f}s"
    _log_step5_tokenize_line(msg)


def _step5_map_or_load_tokenize_cache(
    *,
    datasets: DatasetDict,
    processor,
    nproc: int,
    cache_dir: str,
    cache_fingerprint: str,
    cache_fingerprint_payload: Mapping[str, Any],
    rank: int,
    show_datasets_progress: bool,
    log_tokenize: bool,
    phase: str,
) -> DatasetDict:
    """rank0 map + save；barrier 后各 rank load_from_disk；cache 命中则直接 load。"""
    cache_valid, cache_reason = _step5_tokenize_cache_manifest_matches(
        cache_dir,
        expected_fingerprint=cache_fingerprint_payload,
    )
    if cache_valid:
        t_hit0 = time.perf_counter()
        encoded_data = load_from_disk(cache_dir)
        elapsed_hit = time.perf_counter() - t_hit0
        if rank == 0 and log_tokenize:
            msg = (
                f"[Tokenize] {phase} cache hit | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"load_wall_time={elapsed_hit:.2f}s"
            )
            _log_step5_tokenize_line(msg)
        return encoded_data

    if rank == 0:
        if os.path.exists(cache_dir):
            _log_step5_tokenize_line(
                f"[Tokenize] {phase} cache rebuild | fingerprint={cache_fingerprint} | "
                f"reason={cache_reason} | cache_dir={cache_dir}"
            )
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            else:
                os.unlink(cache_dir)
        t0 = time.perf_counter()
        with hf_datasets_progress_bar(show_datasets_progress):
            encoded_data = datasets.map(lambda sample: processor(sample), num_proc=nproc, desc="Tokenize")
        encoded_data.save_to_disk(cache_dir)
        _write_step5_tokenize_cache_manifest(
            cache_dir,
            fingerprint=cache_fingerprint_payload,
            splits={name: len(encoded_data[name]) for name in encoded_data.keys()},
        )
        elapsed = time.perf_counter() - t0
        if log_tokenize:
            _log_tokenize_map_done_step5(phase, nproc, elapsed)
            msg = (
                f"[Tokenize] {phase} cache miss | fingerprint={cache_fingerprint} | cache_dir={cache_dir} | "
                f"build_wall_time={elapsed:.2f}s"
            )
            _log_step5_tokenize_line(msg)

    _dist_barrier_if_initialized()
    cache_valid, cache_reason = _step5_tokenize_cache_manifest_matches(
        cache_dir,
        expected_fingerprint=cache_fingerprint_payload,
    )
    if not cache_valid:
        raise RuntimeError(
            f"Step5 tokenize cache lineage gate failed: reason={cache_reason} dir={cache_dir}"
        )
    encoded_data = load_from_disk(cache_dir)
    return encoded_data


def _decode_profile_to_training_patch(blob: Dict[str, Any]) -> Dict[str, Any]:
    """将 decode profile 的解码控制字段映射到 FinalTrainingConfig.replace 补丁。"""
    out: Dict[str, Any] = {}
    if not blob:
        return out

    def _opt_int(k: str) -> Optional[int]:
        v = blob.get(k)
        if v is None or v == "":
            return None
        try:
            i = int(v)
            return i if i > 0 else None
        except Exception:
            return None

    sm = _opt_int("soft_max_len")
    if sm is not None:
        out["soft_max_len"] = sm
    hm = _opt_int("hard_max_len")
    if hm is not None:
        out["hard_max_len"] = hm
    if "eos_boost_start" in blob:
        try:
            out["eos_boost_start"] = int(blob["eos_boost_start"])
        except Exception:
            pass
    if "eos_boost_value" in blob:
        try:
            out["eos_boost_value"] = float(blob["eos_boost_value"])
        except Exception:
            pass
    for fk in ("tail_temperature", "tail_top_p"):
        if fk in blob:
            try:
                out[fk] = float(blob[fk])
            except Exception:
                pass
    for bk in (
        "forbid_eos_after_open_quote",
        "forbid_eos_after_open_bracket",
        "forbid_bad_terminal_tokens",
        "candidate_mixed_include_diverse",
    ):
        if bk in blob:
            out[bk] = bool(blob[bk])
    if "decode_token_repeat_window" in blob:
        try:
            out["decode_token_repeat_window"] = max(1, int(blob["decode_token_repeat_window"]))
        except Exception:
            pass
    if "decode_token_repeat_max" in blob:
        try:
            out["decode_token_repeat_max"] = max(1, int(blob["decode_token_repeat_max"]))
        except Exception:
            pass
    cf = blob.get("candidate_family")
    if cf is not None and str(cf).strip():
        out["candidate_family"] = str(cf).strip().lower()
    if "gap_threshold" in blob:
        try:
            out["gap_threshold"] = float(blob["gap_threshold"])
        except Exception:
            pass
    if "prefix_greedy_steps" in blob:
        try:
            out["prefix_greedy_steps"] = max(0, int(blob["prefix_greedy_steps"]))
        except Exception:
            pass
    if "top_k" in blob:
        try:
            out["decode_top_k"] = max(1, int(blob["top_k"]))
        except Exception:
            pass
    bt = blob.get("bad_terminal_token_ids")
    if isinstance(bt, list) and bt:
        try:
            out["bad_terminal_token_ids"] = tuple(int(x) for x in bt)
        except Exception:
            pass
    fm = blob.get("domain_fusion_mode")
    if fm is not None and str(fm).strip():
        v = str(fm).strip().lower()
        if v in ("cross_attn_only", "gate_only", "gate_cross_attn"):
            out["domain_fusion_mode"] = v
    return out


_STEP5_TRAIN_REQUIRED_COLS = (
    "clean_text",
    "sample_origin",
    "train_keep",
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "entropy_score",
    "uncertainty_score",
    "confidence_bucket",
)
_STEP5_ODCR_ANCHOR_COLS = (
    "content_evidence",
    "content_anchor_score",
    "polarity_anchor",
    "domain_style_anchor",
    "local_style_residual_hint",
    "style_evidence",
    "style_anchor_score",
    "evidence_quality_prior",
)
_STEP5_ODCR_ROUTING_RELIABILITY_COLS = (
    "route_reason_scorer",
    "route_reason_explainer",
    "cf_reliability_score",
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "text_quality_score",
)
_STEP5_IGNORED_FIELDS = (
    "adversarial_coef",
    "adversarial_alpha",
    "adversarial_beta",
    "adversarial_schedule_enabled",
    "adversarial_start_epoch",
    "adversarial_warmup_epochs",
    "adversarial_coef_target",
)


def _require_step5_train_csv_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _STEP5_TRAIN_REQUIRED_COLS if c not in df.columns]
    missing += [c for c in _STEP5_ODCR_ANCHOR_COLS if c not in df.columns]
    missing += [c for c in _STEP5_ODCR_ROUTING_RELIABILITY_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Step5 训练 CSV 缺少必需列: "
            + ", ".join(missing)
            + "。请使用新版 Step4 导出的 odcr_routing_train.csv（禁止静默回退 explanation）。"
        )


def _rank0_step5_train_data_audit(
    raw_df: pd.DataFrame,
    filt_df: pd.DataFrame,
    *,
    train_label_max_length: int,
    train_dynamic_padding: bool,
    train_padding_strategy: str,
    log_path: Optional[str],
    ddp_find_unused_parameters_effective: Optional[bool] = None,
) -> None:
    """Console plus canonical run-meta data audit JSON/CSV."""
    n_raw = len(raw_df)
    n_filt = len(filt_df)
    msg_lines = [
        f"[Step5 数据审计] 过滤后训练行数={n_filt}（过滤前={n_raw}，条件 train_keep==1 且 clean_text 非空）",
        f"  train_label_max_length={train_label_max_length}",
        f"  train_dynamic_padding={1 if train_dynamic_padding else 0}",
        f"  train_padding_strategy={train_padding_strategy}",
    ]
    if "sample_origin" in raw_df.columns and n_raw > 0:
        vc = raw_df["sample_origin"].value_counts()
        for k, v in vc.items():
            msg_lines.append(f"  过滤前 sample_origin {k}: {int(v)} ({100.0 * float(v) / n_raw:.2f}%)")
    if "sample_origin" in filt_df.columns and n_filt > 0:
        vc2 = filt_df["sample_origin"].value_counts()
        msg_lines.append("  过滤后按来源:")
        for k, v in vc2.items():
            msg_lines.append(f"    {k}: {int(v)} ({100.0 * float(v) / n_filt:.2f}%)")
        if "sample_weight_hint" in filt_df.columns:
            for org in sorted(vc2.keys(), key=lambda x: str(x)):
                sub = filt_df[filt_df["sample_origin"] == org]
                mw = float(sub["sample_weight_hint"].astype(float).mean())
                msg_lines.append(f"    {org} sample_weight_hint 均值={mw:.6f}")

    for flag, label in (
        ("html_entity_hit", "HTML 实体命中"),
        ("bad_tail_hit", "bad_tail 命中"),
        ("template_hit", "template 命中"),
    ):
        if flag in raw_df.columns:
            msg_lines.append(f"  过滤前 {label}: {int(raw_df[flag].sum())}")
    if "template_hit" in raw_df.columns:
        t_total = int((raw_df["template_hit"] == 1).sum())
        t_kept = int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 1)).sum())
        t_drop = int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 0)).sum())
        t_dw = int(
            ((raw_df["template_hit"] == 1) & (raw_df.get("template_downweighted", 0) == 1)).sum()
        )
        msg_lines.append(
            "  template_hit 审计: "
            f"total={t_total}, kept={t_kept}, downweighted={t_dw}, dropped={t_drop}"
        )

    if "train_keep" in raw_df.columns:
        msg_lines.append(f"  train_keep==0 行数: {int((raw_df['train_keep'] == 0).sum())}")

    topn: List[tuple[Any, int]] = []
    if "train_drop_reason" in raw_df.columns:
        dr = raw_df.loc[raw_df["train_keep"] == 0, "train_drop_reason"].fillna("").astype(str)
        ctr = Counter([x for x in dr.tolist() if x])
        topn = ctr.most_common(15)
        msg_lines.append(
            "  train_drop_reason Top-N: " + ", ".join(f"{a}:{b}" for a, b in topn) if topn else "  train_drop_reason Top-N: (无)"
        )

    tok_lens: List[int] = []
    truncated = 0
    word_lens: List[int] = []
    # 审计需统计「未按 train_label 截断」的原始 token 数；若超过 T5 默认 model_max_length（512），
    # HF 会对 truncation=False 打告警。训练路径中 Processor 已 truncation=True，不会把超长序列喂进模型。
    _tok_audit = get_step5_tokenizer()
    _prev_mml = int(getattr(_tok_audit, "model_max_length", 512) or 512)
    try:
        _tok_audit.model_max_length = 1_000_000
        for _, row in filt_df.iterrows():
            ct = str(row.get("clean_text", "") or "")
            word_lens.append(len(ct.split()))
            ids = _tok_audit(ct, add_special_tokens=True, truncation=False)["input_ids"]
            L = len(ids)
            tok_lens.append(L)
            if L > train_label_max_length:
                truncated += 1
    finally:
        _tok_audit.model_max_length = _prev_mml

    if tok_lens:
        arr = np.asarray(tok_lens, dtype=np.int64)
        warr = np.asarray(word_lens, dtype=np.int64)
        p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99]).astype(int)
        msg_lines.append(f"  clean_text token 长度 p50={p50} p90={p90} p95={p95} p99={p99}")
        msg_lines.append(
            f"  超过 train_label_max_length 将被截断: {truncated} / {len(tok_lens)} "
            f"({100.0 * truncated / max(len(tok_lens), 1):.2f}%)"
        )
        w50, w90 = np.percentile(warr, [50, 90]).astype(int)
        msg_lines.append(f"  clean_text 词数 p50={w50} p90={w90}")

    text_block = "\n".join(msg_lines)
    print(text_block, flush=True)
    _lg_a = logging.getLogger(LOGGER_NAME)
    if _lg_a.handlers:
        _lg_a.info(text_block)
    else:
        logging.info(text_block)

    _tg_raw = int((raw_df["sample_origin"] == "target_gold").sum()) if "sample_origin" in raw_df.columns else 0
    _tg_filt = int((filt_df["sample_origin"] == "target_gold").sum()) if "sample_origin" in filt_df.columns else 0
    _cf_raw = int((raw_df["sample_origin"] == "aux_cf").sum()) if "sample_origin" in raw_df.columns else 0
    _cf_filt = int((filt_df["sample_origin"] == "aux_cf").sum()) if "sample_origin" in filt_df.columns else 0
    _t_sev_drop = (
        int((raw_df["train_drop_reason"].astype(str) == "severe_template").sum())
        if "train_drop_reason" in raw_df.columns
        else 0
    )
    _t_med_dw = (
        int(
            ((raw_df.get("template_downweighted", 0) == 1) & (raw_df["train_keep"] == 1)).sum()
        )
        if "train_keep" in raw_df.columns
        else 0
    )
    _nt_dw = (
        int((raw_df.get("noisy_tail_downweighted", 0) == 1).sum())
        if "noisy_tail_downweighted" in raw_df.columns
        else 0
    )
    _nt_drop = (
        int(
            (
                (raw_df.get("repeat_tail_hit", 0) == 1)
                & (raw_df["train_keep"] == 0)
                & (raw_df["train_drop_reason"].astype(str) != "severe_template")
            ).sum()
        )
        if "train_keep" in raw_df.columns and "train_drop_reason" in raw_df.columns
        else 0
    )
    msg_lines.append(
        f"  治理统计 template_severe_dropped={_t_sev_drop} template_medium_downweighted={_t_med_dw} "
        f"noisy_tail_downweighted={_nt_dw} noisy_tail_dropped_other={_nt_drop}"
    )
    msg_lines.append(
        f"  target_gold_kept_ratio={(_tg_filt / max(_tg_raw, 1)):.4f} "
        f"aux_cf_kept_ratio={(_cf_filt / max(_cf_raw, 1)):.4f}"
    )

    audit_obj: Dict[str, Any] = {
        "schema_version": "odcr_step5_train_audit/1.1",
        "n_rows_before_filter": n_raw,
        "n_rows_after_filter": n_filt,
        "train_label_max_length": int(train_label_max_length),
        "train_dynamic_padding": bool(train_dynamic_padding),
        "train_padding_strategy": str(train_padding_strategy),
        "drop_reason_top": {str(k): int(v) for k, v in topn},
        "template_severe_dropped": _t_sev_drop,
        "template_medium_downweighted": _t_med_dw,
        "noisy_tail_downweighted": _nt_dw,
        "noisy_tail_dropped": _nt_drop,
        "target_gold_kept_ratio": float(_tg_filt / max(_tg_raw, 1)),
        "aux_cf_kept_ratio": float(_cf_filt / max(_cf_raw, 1)),
        "training_diagnostics": training_diagnostics_snapshot(
            diagnostics_scope="child",
            effective_training_payload_json=os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""),
            ddp_find_unused_parameters_effective=ddp_find_unused_parameters_effective,
        ),
    }
    if "template_hit" in raw_df.columns:
        audit_obj["template_hit_audit"] = {
            "template_hit_total": int((raw_df["template_hit"] == 1).sum()),
            "template_hit_kept": int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 1)).sum()),
            "template_hit_dropped": int(((raw_df["template_hit"] == 1) & (raw_df["train_keep"] == 0)).sum()),
            "template_hit_downweighted": int(
                ((raw_df["template_hit"] == 1) & (raw_df.get("template_downweighted", 0) == 1)).sum()
            ),
        }
    if tok_lens:
        arr = np.asarray(tok_lens, dtype=np.int64)
        warr = np.asarray(word_lens, dtype=np.int64)
        audit_obj["truncation_over_max"] = {
            "count": int(truncated),
            "frac": float(truncated / max(len(tok_lens), 1)),
        }
        audit_obj["token_len_quantiles"] = {
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
            "p95": int(np.percentile(arr, 95)),
            "p99": int(np.percentile(arr, 99)),
        }
        audit_obj["word_len_quantiles"] = {
            "p50": int(np.percentile(warr, 50)),
            "p90": int(np.percentile(warr, 90)),
        }

    if log_path:
        log_dir = os.path.dirname(os.path.abspath(os.path.expanduser(log_path)))
        os.makedirs(log_dir, exist_ok=True)
        jpath = os.path.join(log_dir, path_layout.metrics_filename("data_audit"))
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(audit_obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        csv_path = os.path.join(log_dir, path_layout.metrics_filename("data_audit_summary"))
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write(f"n_rows_before_filter,{n_raw}\n")
            f.write(f"n_rows_after_filter,{n_filt}\n")
            f.write(f"truncation_over_max_count,{truncated}\n")
            f.write(f"template_severe_dropped,{_t_sev_drop}\n")
            f.write(f"template_medium_downweighted,{_t_med_dw}\n")
            f.write(f"noisy_tail_downweighted,{_nt_dw}\n")
            f.write(f"noisy_tail_dropped,{_nt_drop}\n")
            f.write(f"target_gold_kept_ratio,{_tg_filt / max(_tg_raw, 1):.6f}\n")
            f.write(f"aux_cf_kept_ratio,{_cf_filt / max(_cf_raw, 1):.6f}\n")


def _step5_audit_first_collate_batches(
    *,
    collate_fn,
    train_dataset,
    valid_dataset,
    contract: Mapping[str, Any],
    ctx: Mapping[str, Any],
    strict: bool,
    valid_split_label: str,
) -> None:
    if not strict:
        return
    cap = 8
    if train_dataset is not None and len(train_dataset) > 0:
        n = min(cap, len(train_dataset))
        batch = collate_fn([train_dataset[i] for i in range(n)])
        validate_first_batch_indices(batch, contract, "train", ctx=dict(ctx))
    if valid_dataset is not None and len(valid_dataset) > 0:
        n = min(cap, len(valid_dataset))
        batch = collate_fn([valid_dataset[i] for i in range(n)])
        validate_first_batch_indices(batch, contract, valid_split_label, ctx=dict(ctx))


def build_odcr_ddp_artefacts(
    args,
    world_size,
    local_rank,
    rank,
    *,
    command: str,
    show_datasets_progress: bool = True,
):
    _dataset_build_t0 = time.perf_counter()
    task_idx = resolve_task_idx_from_aux_target(args.auxiliary, args.target)
    if task_idx is None:
        raise ValueError("未知的 auxiliary/target 组合")
    eval_only = command != "train"
    _ro = collect_training_hardware_overrides_from_args(args)
    resolved = build_resolved_training_config(
        args,
        task_idx=task_idx,
        world_size=world_size,
        hardware_overrides=_ro,
    )
    path = os.path.join(get_data_dir(), args.target)
    _ckpt_task = get_stage_run_dir(task_idx)
    os.makedirs(_ckpt_task, exist_ok=True)
    train_path = os.path.join(_ckpt_task, ODCR_ROUTING_TRAIN_CSV)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"缺少 Step4 训练 CSV: {train_path}")
    valid_path = os.path.join(path, "valid.csv")
    test_path = os.path.join(path, "test.csv")
    if command == "test":
        if not os.path.isfile(test_path):
            raise FileNotFoundError(f"缺少测试集 CSV（--command test）: {test_path}")
        eval_data_path = test_path
        split_label = "test"
    else:
        eval_data_path = valid_path
        split_label = "valid"

    _contract_path = resolve_index_contract_path(train_path)
    try:
        index_contract = load_index_contract(_contract_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Step5 要求与 {ODCR_ROUTING_TRAIN_CSV}（解析后真实路径）同目录下的 index_contract.json；"
            f"请重跑 Step4。contract_path={_contract_path}"
        ) from e
    if str(index_contract.get("aux_domain")) != str(args.auxiliary) or str(
        index_contract.get("target_domain")
    ) != str(args.target):
        raise IndexContractError(
            f"index_contract 域与 CLI 不一致: contract aux={index_contract.get('aux_domain')!r} "
            f"tgt={index_contract.get('target_domain')!r} args aux={args.auxiliary!r} tgt={args.target!r} "
            f"path={_contract_path}"
        )
    _payload_for_lineage = current_effective_payload(required=True)
    _step4_export_lineage = validate_step4_export_lineage(
        index_contract,
        current_step4_rcr_config=dict(_payload_for_lineage.get("step4_rcr") or {}),
        task_id=int(task_idx),
        auxiliary_domain=str(args.auxiliary),
        target_domain=str(args.target),
    )
    nuser = int(index_contract["nuser_global"])
    nitem = int(index_contract["nitem_global"])
    _prof_dc, _prof_ds, _prof_uc, _prof_us, _prof_ic, _prof_is, _prof_meta = _load_odcr_profile_tensors_from_contract(
        index_contract, "cpu"
    )
    _ = (_prof_dc, _prof_ds, _prof_us, _prof_is)
    _ictx = {
        "task_id": int(task_idx),
        "iteration_id": index_contract.get("iteration_id"),
        "step4_run": index_contract.get("step4_run"),
        "step5_run": os.path.basename(os.path.abspath(_ckpt_task)),
        "contract_path": _contract_path,
        "profile_mode": _prof_meta.get("profile_mode"),
        "profile_path": (_prof_meta.get("selected_paths", {}) or {}).get("user_content", [None])[0],
        "csv_path": train_path,
    }
    validate_index_contract_against_profiles(
        index_contract, _prof_uc, _prof_ic, ctx=_ictx
    )
    from config import get_odcr_embed_dim

    _prof_dim = int(_prof_uc.shape[-1])
    if _prof_dim != int(get_odcr_embed_dim()):
        raise ValueError(
            f"已加载双通道 profile 隐层维度={_prof_dim} 与 ODCR_EMBED_DIM={get_odcr_embed_dim()} 不一致；"
            "请用与 ODCR_EMBED_DIM 一致的句向量模型重跑 compute_embeddings.py / infer_domain_semantics.py。"
        )
    if int(index_contract["embed_dim"]) != _prof_dim:
        raise IndexContractError(
            f"index_contract.embed_dim={index_contract.get('embed_dim')!r} 与磁盘 profile 最后一维={_prof_dim} 不一致。"
        )
    resolved = replace(resolved, emsize=_prof_dim)
    setattr(args, "_odcr_index_contract", index_contract)
    setattr(args, "_odcr_profile_meta", _prof_meta)

    train_df = pd.read_csv(train_path)
    if GLOBAL_COL_USER not in train_df.columns or GLOBAL_COL_ITEM not in train_df.columns:
        raise ValueError(
            f"训练 CSV 须含 {GLOBAL_COL_USER}/{GLOBAL_COL_ITEM}（Step4 index_contract 管线）。path={train_path}"
        )
    if command == "train":
        _require_step5_rcr_posterior_controls(train_df, ctx=f"Step5 training CSV {train_path}")
    validate_split_indices(train_df, index_contract, "train", ctx={**_ictx, "csv_path": train_path})

    def _idx_mm(d: pd.DataFrame):
        if d is None or len(d) == 0 or GLOBAL_COL_USER not in d.columns:
            return None
        return {
            "user_idx_global": [int(d[GLOBAL_COL_USER].min()), int(d[GLOBAL_COL_USER].max())],
            "item_idx_global": [int(d[GLOBAL_COL_ITEM].min()), int(d[GLOBAL_COL_ITEM].max())],
        }

    _mm_train_file = _idx_mm(train_df)
    _mm_train_after_filter: Optional[Dict[str, Any]] = None
    _mm_eval_split: Optional[Dict[str, Any]] = None

    _model_dir = os.path.join(_ckpt_task, "model")
    os.makedirs(_model_dir, exist_ok=True)
    _best = os.path.join(_model_dir, "best.pth")
    _last = os.path.join(_model_dir, "best.pth")
    save_file = args.save_file or _best
    save_file = os.path.abspath(os.path.expanduser(save_file))
    nproc = int(resolved.num_proc)
    _raw_dp = (os.environ.get("ODCR_DECODE_PROFILE_JSON") or "").strip()
    if not _raw_dp:
        raise RuntimeError(
            "缺少 ODCR_DECODE_PROFILE_JSON：须由父进程 `python code/odcr.py …` 经 torchrun 注入完整 decode 预设 JSON；"
            "勿裸调 executors/step5_entry。"
        )
    try:
        _dp_full = json.loads(_raw_dp)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ODCR_DECODE_PROFILE_JSON 非法 JSON: {e}") from e
    if not isinstance(_dp_full, dict):
        raise TypeError("ODCR_DECODE_PROFILE_JSON 根须为 object")
    _prof_patch = _decode_profile_to_training_patch(_dp_full)
    _ds = str(_dp_full.get("decode_strategy", "greedy")).strip().lower()
    _dseed = _dp_full.get("decode_seed")
    if _dseed in ("", "null", "None", None):
        _dseed_p = None
    else:
        _dseed_p = int(_dseed)
    _nr = _dp_full.get("no_repeat_ngram_size")
    _nr_p = None if _nr in (None, "", "null", "None") else int(_nr)
    _mn = _dp_full.get("min_len")
    _mn_p = None if _mn in (None, "", "null", "None") else int(_mn)
    ml_eff = int(_dp_full.get("max_explanation_length", int(args.max_explanation_length)))
    tlm = int(resolved.train_label_max_length)
    processor = Processor(args.auxiliary, args.target, max_length=tlm)
    base_final = replace(
        resolved,
        **_prof_patch,
        nuser=nuser,
        nitem=nitem,
        ntoken=len(get_step5_tokenizer()),
        save_file=save_file,
        last_checkpoint_path=_last,
        device=local_rank,
        device_ids=tuple(range(world_size)),
        ddp_world_size=world_size,
        nlayers=int(resolved.nlayers),
        nhead=int(resolved.nhead),
        nhid=int(resolved.nhid),
        dropout=float(resolved.dropout),
        label_smoothing=float(_dp_full["label_smoothing"]),
        repetition_penalty=float(_dp_full["repetition_penalty"]),
        generate_temperature=float(_dp_full["generate_temperature"]),
        generate_top_p=float(_dp_full["generate_top_p"]),
        max_explanation_length=ml_eff,
        decode_strategy=_ds,
        decode_seed=_dseed_p,
        no_repeat_ngram_size=_nr_p,
        min_len=_mn_p,
        step4_export_lineage_json=json.dumps(_step4_export_lineage, ensure_ascii=False, sort_keys=True),
    )
    setattr(args, "_odcr_eval_split_label", split_label)
    setattr(args, "_odcr_eval_data_path", eval_data_path)

    if eval_only:
        ev_df = pd.read_csv(eval_data_path)
        ev_df = normalize_split_indices_to_global(
            ev_df, index_contract, split_label, ctx={**_ictx, "csv_path": eval_data_path}
        )
        validate_split_indices(ev_df, index_contract, split_label, ctx={**_ictx, "csv_path": eval_data_path})
        _mm_eval_split = _idx_mm(ev_df)
        _mm_train_after_filter = _mm_train_file
        ev_df["domain"] = "target"
        ev_df = ev_df.reset_index(drop=True)
        ev_df["sample_id"] = np.arange(len(ev_df), dtype=np.int64)
        ev_df["clean_text"] = ev_df["explanation"].fillna("").astype(str)
        ev_df = _apply_step5_factual_eval_default_controls(ev_df, split_label=split_label)
        cache_dir, cache_fp, cache_fp_payload = _build_step5_cache_dir(
            get_hf_cache_root(task_idx),
            train_path,
            eval_data_path,
            processor,
            get_step5_tokenizer(),
            task_idx=task_idx,
            split_label=split_label,
            eval_only=True,
            index_contract_path=_contract_path,
            step4_export_lineage=_step4_export_lineage,
        )
        if rank == 0:
            _log_step5_tokenize_line(
                f"[Tokenize] step5 cache key | split={split_label} | fingerprint={cache_fp} | cache_dir={cache_dir}",
            )
        datasets = DatasetDict({"valid": Dataset.from_pandas(ev_df)})
        _st5 = logging.getLogger(LOGGER_NAME)
        _tok_wall0 = time.perf_counter()
        with odcr_timing_phase(
            _st5,
            f"tokenize_pipeline_step5_{split_label}",
            route=ROUTE_SUMMARY,
            rank=rank,
        ):
            encoded_data = _step5_map_or_load_tokenize_cache(
                datasets=datasets,
                processor=processor,
                nproc=nproc,
                cache_dir=cache_dir,
                cache_fingerprint=cache_fp,
                cache_fingerprint_payload=cache_fp_payload,
                rank=rank,
                show_datasets_progress=show_datasets_progress,
                log_tokenize=(rank == 0),
                phase=split_label,
            )
        setattr(args, "_odcr_eval_tokenize_cache_wall_s", float(time.perf_counter() - _tok_wall0))
        train_dataset = None
    else:
        valid_df = pd.read_csv(valid_path)
        valid_df = normalize_split_indices_to_global(
            valid_df, index_contract, "valid", ctx={**_ictx, "csv_path": valid_path}
        )
        validate_split_indices(valid_df, index_contract, "valid", ctx={**_ictx, "csv_path": valid_path})
        _mm_eval_split = _idx_mm(valid_df)
        valid_df["domain"] = "target"
        valid_df = valid_df.reset_index(drop=True)
        valid_df["sample_id"] = np.arange(len(valid_df), dtype=np.int64)
        valid_df["clean_text"] = valid_df["explanation"].fillna("").astype(str)
        valid_df = _apply_step5_factual_eval_default_controls(valid_df, split_label="valid")

        train_raw = train_df.copy()
        train_df = train_df[train_df["train_keep"] == 1]
        train_df = train_df[train_df["clean_text"].fillna("").astype(str).str.strip() != ""]
        train_df = train_df[
            (train_df["route_scorer"].astype(int) == 1) | (train_df["route_explainer"].astype(int) == 1)
        ]
        gate_cfg_for_data = parse_step5_innovation_config_json(
            str(resolved.step5_innovation_config_json)
        ).explainer_gate
        explainer_only_multiplier = float(gate_cfg_for_data.explainer_only_multiplier)
        # Step4 sample_weight_hint 是 posterior base weight；这里仅应用 Step5B 训练调度倍率。
        train_df["sample_weight_hint"] = np.where(
            train_df["route_scorer"].astype(int) == 1,
            train_df["sample_weight_hint"].astype(float),
            train_df["sample_weight_hint"].astype(float) * explainer_only_multiplier,
        )
        train_df = train_df.reset_index(drop=True)
        train_df["sample_id"] = np.arange(len(train_df), dtype=np.int64)
        validate_split_indices(train_df, index_contract, "train", ctx={**_ictx, "csv_path": train_path})
        _mm_train_after_filter = _idx_mm(train_df)
        if len(train_df) == 0:
            raise ValueError(
                "Step5 训练集在 train_keep==1 且 route_scorer/route_explainer 路由过滤后行数为 0；请检查 Step4 导出阈值。"
            )
        if rank == 0:
            _rank0_step5_train_data_audit(
                train_raw,
                train_df,
                train_label_max_length=tlm,
                train_dynamic_padding=bool(resolved.train_dynamic_padding),
                train_padding_strategy=str(resolved.train_padding_strategy),
                log_path=getattr(args, "log_file", None),
                ddp_find_unused_parameters_effective=bool(resolved.ddp_find_unused_parameters),
            )
        cache_dir, cache_fp, cache_fp_payload = _build_step5_cache_dir(
            get_hf_cache_root(task_idx),
            train_path,
            valid_path,
            processor,
            get_step5_tokenizer(),
            task_idx=task_idx,
            split_label="train+valid",
            eval_only=False,
            index_contract_path=_contract_path,
            step4_export_lineage=_step4_export_lineage,
        )
        if rank == 0:
            _log_step5_tokenize_line(
                f"[Tokenize] step5 cache key | fingerprint={cache_fp} | cache_dir={cache_dir}",
            )
        datasets = DatasetDict(
            {"train": Dataset.from_pandas(train_df), "valid": Dataset.from_pandas(valid_df)}
        )
        _st5 = logging.getLogger(LOGGER_NAME)
        _tok_wall0 = time.perf_counter()
        with odcr_timing_phase(
            _st5,
            "tokenize_pipeline_step5_train_valid",
            route=ROUTE_SUMMARY,
            rank=rank,
        ):
            encoded_data = _step5_map_or_load_tokenize_cache(
                datasets=datasets,
                processor=processor,
                nproc=nproc,
                cache_dir=cache_dir,
                cache_fingerprint=cache_fp,
                cache_fingerprint_payload=cache_fp_payload,
                rank=rank,
                show_datasets_progress=show_datasets_progress,
                log_tokenize=(rank == 0),
                phase="train+valid",
            )
        setattr(args, "_odcr_eval_tokenize_cache_wall_s", float(time.perf_counter() - _tok_wall0))
        train_dataset = encoded_data["train"]
    valid_dataset = encoded_data["valid"]
    _collate_audit = partial(
        _step5_collate_dynamic,
        dynamic_padding=bool(resolved.train_dynamic_padding),
        fixed_max_length=int(resolved.train_label_max_length),
    )
    _step5_audit_first_collate_batches(
        collate_fn=_collate_audit,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        contract=index_contract,
        ctx=_ictx,
        strict=bool(getattr(resolved, "step5_strict_index_batches", False)),
        valid_split_label=split_label,
    )
    if rank == 0 and getattr(args, "log_file", None):
        _log_d = os.path.dirname(os.path.abspath(args.log_file))
        os.makedirs(_log_d, exist_ok=True)
        write_index_contract_audit(
            os.path.join(_log_d, "index_contract_audit.json"),
            {
                "schema_version": "odcr_step5_index_contract_audit/1.0",
                "contract_path": _contract_path,
                "train_csv_resolved": os.path.abspath(train_path),
                "eval_split_csv": os.path.abspath(eval_data_path),
                "step4_run": index_contract.get("step4_run"),
                "step5_run_dir": os.path.abspath(_ckpt_task),
                "task_id": int(task_idx),
                "nuser_global": nuser,
                "nitem_global": nitem,
                "profile_rows": {
                    "user": int(_prof_cpu_user.shape[0]),
                    "item": int(_prof_cpu_item.shape[0]),
                },
                "splits_min_max": {
                    "train_file": _mm_train_file,
                    "train_after_filter": _mm_train_after_filter,
                    "eval_split": _mm_eval_split,
                },
                "local_to_global_applied": True,
                "checks_passed": True,
                "strict_collate_batch_audit": bool(
                    getattr(resolved, "step5_strict_index_batches", False)
                ),
            },
        )
    model = _make_model(base_final, args, local_rank)
    setattr(args, "_odcr_eval_dataset_build_wall_s", float(time.perf_counter() - _dataset_build_t0))
    return base_final, train_dataset, valid_dataset, model


def _metrics_final_dict_from_rows(merged: List[dict]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    all_pred = np.array([r["pred_rating"] for r in merged], dtype=np.float64)
    all_gt = np.array([r["gt_rating"] for r in merged], dtype=np.float64)
    diffs = all_pred - all_gt
    mae = round(float(np.mean(np.abs(diffs))), 4)
    rmse = round(float(np.sqrt(np.mean(np.square(diffs)))), 4)
    pred_tx = [r["pred_text"] for r in merged]
    ref_tx = [r["ref_text"] for r in merged]
    text_results = evaluate_text(pred_tx, ref_tx)
    paper_block = compute_paper_comparable_text_metrics(pred_tx, ref_tx)
    ext = extended_text_metrics_bundle(pred_tx, ref_tx)
    collapse = compute_collapse_stats(pred_tx, ref_tx, top_k_file=20)
    ref_mean = float((ext.get("corpus_level") or {}).get("mean_ref_len_words") or 0.0)
    dirty = compute_dirty_text_stats(pred_tx, ref_mean_len_words=ref_mean or None)
    final = {
        "metrics_schema_version": "odcr_metrics_mainline/2.0",
        "recommendation": {"mae": mae, "rmse": rmse},
        "explanation": text_results,
        "paper_metrics": paper_block,
        "text_metrics_corpus_and_sentence": ext,
        "collapse_stats": collapse,
        "dirty_text": dirty,
    }
    return final, pred_tx, ref_tx


def _run_ddp(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeError("step5 runner 仅支持 CUDA + NCCL DDP。")
    torch.cuda.set_device(local_rank)
    ddp_fast_backends = apply_ddp_fast_torch_backends()
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
    train_logger = _setup["logger"]

    valid_dataloader = None
    eval_only = args.command != "train"
    try:
        base_final, train_dataset, valid_dataset, model = build_odcr_ddp_artefacts(
            args,
            world_size,
            local_rank,
            rank,
            command=args.command,
            show_datasets_progress=(rank == 0),
        )
        if args.command == "generate_samples":
            mxs = max(1, int(args.generate_max_samples))
            n_full = len(valid_dataset)
            mxs = min(mxs, n_full)
            valid_dataset = Subset(valid_dataset, list(range(mxs)))
        final_cfg = replace(
            base_final,
            run_id=run_id,
            logger=train_logger,
            log_file=log_path,
            valid_dataset=valid_dataset,
            ddp_fast_backends=ddp_fast_backends,
            rank0_only_logging=True,
        )
        setattr(args, "_odcr_step5_innovation_config_json", str(final_cfg.step5_innovation_config_json))
        step5_collate_fn = partial(
            _step5_collate_dynamic,
            dynamic_padding=bool(getattr(final_cfg, "train_dynamic_padding", True)),
            fixed_max_length=int(getattr(final_cfg, "train_label_max_length", 64)),
        )
        pin_memory = torch.cuda.is_available()
        _G = int(final_cfg.train_batch_size)
        if rank == 0:
            _decode_meta = {
                "command": args.command,
                "label_smoothing": final_cfg.label_smoothing,
                "repetition_penalty": final_cfg.repetition_penalty,
                "generate_temperature": final_cfg.generate_temperature,
                "generate_top_p": final_cfg.generate_top_p,
                "gap_threshold": float(getattr(final_cfg, "gap_threshold", 0.35)),
                "prefix_greedy_steps": int(getattr(final_cfg, "prefix_greedy_steps", 4)),
                "decode_top_k": int(getattr(final_cfg, "decode_top_k", 5)),
                "max_explanation_length": final_cfg.max_explanation_length,
                "train_label_max_length": int(getattr(final_cfg, "train_label_max_length", 128)),
                "decode_strategy": final_cfg.decode_strategy,
                "decode_seed": final_cfg.decode_seed,
                "no_repeat_ngram_size": final_cfg.no_repeat_ngram_size,
                "min_len": final_cfg.min_len,
                "ntoken_resolved": final_cfg.ntoken,
                "nhead": final_cfg.nhead,
                "nhid": final_cfg.nhid,
                "nlayers": final_cfg.nlayers,
                "dropout": final_cfg.dropout,
                "eval_single_process_safe": bool(getattr(args, "eval_single_process_safe", False)),
                "sanity_compare_ddp_single": bool(getattr(args, "sanity_compare_ddp_single", False)),
                "train_dynamic_padding": bool(getattr(final_cfg, "train_dynamic_padding", True)),
                "train_padding_strategy": str(getattr(final_cfg, "train_padding_strategy", "dynamic_batch")),
            }
            _trd_snap = training_diagnostics_snapshot(
                diagnostics_scope="child",
                effective_training_payload_json=os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""),
                ddp_find_unused_parameters_effective=bool(final_cfg.ddp_find_unused_parameters),
            )
            _tfp = (os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT") or "").strip()
            _gfp = (os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT") or "").strip()
            _rdfp = (os.environ.get("ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip()
            log_run_snapshot(
                train_logger,
                {
                    "auxiliary": args.auxiliary,
                    "target": args.target,
                    "command": args.command,
                    "train_only": getattr(args, "train_only", False),
                    "eval_only": eval_only,
                    "rank": rank,
                    "world_size": world_size,
                    "local_rank": local_rank,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "train_global_batch_size": _G,
                    "train_batch_size_global": final_cfg.batch_size_global,
                    "train_per_device_batch_size": final_cfg.per_device_train_batch_size,
                    "eval_global_batch_size": int(final_cfg.eval_batch_size),
                    "eval_per_gpu_batch_size": (
                        int(final_cfg.eval_batch_size) // world_size
                        if int(final_cfg.eval_batch_size) % world_size == 0
                        else None
                    ),
                    "gradient_accumulation_steps": final_cfg.gradient_accumulation_steps,
                    "effective_global_batch_size": final_cfg.effective_global_batch_size,
                    "train_mode": str(getattr(final_cfg, "train_mode", "lora")),
                    "train_precision": str(getattr(final_cfg, "train_precision", "bf16")),
                    "per_device_eval_batch_size": int(
                        getattr(final_cfg, "per_device_eval_batch_size", 2)
                    ),
                    "peft": getattr(args, "_odcr_step5_peft_meta", None),
                    "distributed_env": collect_distributed_env_for_meta(),
                    "decode_and_model_runtime": _decode_meta,
                    "training_diagnostics": _trd_snap,
                    "training_semantic_fingerprint": _tfp or None,
                    "generation_semantic_fingerprint": _gfp or None,
                    "runtime_diagnostics_fingerprint": _rdfp or None,
                },
                final_cfg.to_log_dict(),
            )
            _cfg_resolved_path = os.path.join(os.path.dirname(log_path), "resolved_config.json")
            _cfg_merged = dict(final_cfg.to_log_dict())
            _ignored = [k for k in _STEP5_IGNORED_FIELDS if k in _cfg_merged]
            for _k in _ignored:
                _cfg_merged.pop(_k, None)
            _cfg_merged["step5_ignored_fields"] = list(_ignored)
            _cfg_merged["training_diagnostics"] = _trd_snap
            _cfg_merged["training_semantic_fingerprint"] = _tfp or None
            _cfg_merged["generation_semantic_fingerprint"] = _gfp or None
            _cfg_merged["runtime_diagnostics_fingerprint"] = _rdfp or None

            _cfg_merged["runtime_env"] = runtime_env_dict_for_config_resolved()

            with open(_cfg_resolved_path, "w", encoding="utf-8") as _cf:
                json.dump(_cfg_merged, _cf, ensure_ascii=False, indent=2, default=str)
                _cf.write("\n")
            train_logger.info(
                "[Config resolved] wrote %s",
                _cfg_resolved_path,
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Config resolved] %s",
                json.dumps(_decode_meta, ensure_ascii=False, default=str),
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Step5 ignored fields] %s",
                ",".join(_ignored) if _ignored else "(none)",
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Fingerprints] training_semantic=%s generation_semantic=%s runtime_diag=%s",
                _tfp or "n/a",
                _gfp or "n/a",
                _rdfp or "n/a",
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            train_logger.info(
                "[Step5 train knobs] train_label_max_length=%s train_dynamic_padding=%s "
                "train_padding_strategy=%s loss_weight_repeat_ul=%s loss_weight_terminal_clean=%s "
                "loss_weight_batch_diversity=%s batch_diversity_warmup_epochs=%s checkpoint_selection_mode=%s",
                int(getattr(final_cfg, "train_label_max_length", 128)),
                bool(getattr(final_cfg, "train_dynamic_padding", True)),
                str(getattr(final_cfg, "train_padding_strategy", "dynamic_batch")),
                float(getattr(final_cfg, "loss_weight_repeat_ul", 0.0)),
                float(getattr(final_cfg, "loss_weight_terminal_clean", 0.0)),
                float(getattr(final_cfg, "loss_weight_batch_diversity", 0.0)),
                int(getattr(final_cfg, "batch_diversity_warmup_epochs", 0)),
                str(getattr(final_cfg, "checkpoint_selection_mode", "guarded_composite")),
                extra=log_route_extra(train_logger, ROUTE_SUMMARY),
            )
            flush_preset_load_events(train_logger)
        valid_sampler = None
        if not eval_only:
            if int(final_cfg.eval_batch_size) % world_size != 0:
                raise ValueError(
                    f"eval_batch_size={int(final_cfg.eval_batch_size)} 与 world_size={world_size} 不整除，无法按卡切分。"
                    "请修改 configs/odcr.yaml 中 eval.profiles.*.eval_batch_size，或调整 hardware.profiles.*.ddp_world_size。"
                )
            valid_per_rank = int(final_cfg.eval_batch_size) // world_size
            valid_sampler = DistributedSampler(
                valid_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=valid_per_rank,
                sampler=valid_sampler,
                shuffle=False,
                num_workers=final_cfg.dataloader_num_workers_valid,
                pin_memory=pin_memory,
                persistent_workers=final_cfg.dataloader_num_workers_valid > 0,
                prefetch_factor=final_cfg.dataloader_prefetch_factor_valid,
                collate_fn=step5_collate_fn,
            )
        if not eval_only:
            _A = max(1, int(final_cfg.gradient_accumulation_steps))
            train_drop_last = _A > 1
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=train_drop_last,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=final_cfg.per_device_train_batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=final_cfg.dataloader_num_workers_train,
                pin_memory=pin_memory,
                persistent_workers=final_cfg.dataloader_num_workers_train > 0,
                prefetch_factor=final_cfg.dataloader_prefetch_factor_train,
                drop_last=train_drop_last,
                collate_fn=step5_collate_fn,
            )
            _n_train_micro = len(train_dataloader)
            if _A > 1 and _n_train_micro % _A != 0:
                raise ValueError(
                    f"train DataLoader 每 epoch 批次数为 {_n_train_micro}，无法被 gradient_accumulation_steps={_A} 整除。"
                    f"请调整全局 batch、--per-device-batch-size、world_size 或数据划分；或令 accum=1。"
                )
            if not bool(final_cfg.ddp_find_unused_parameters):
                run_step5_find_unused_parameters_preflight(
                    model,
                    final_cfg,
                    step5_innov_cfg=parse_step5_innovation_config_json(
                        str(final_cfg.step5_innovation_config_json or "{}")
                    ),
                    logger=train_logger if rank == 0 else None,
                )
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=final_cfg.ddp_find_unused_parameters,
            )
            try:
                trainModel_ddp(
                    model,
                    train_dataloader,
                    valid_dataloader,
                    sampler,
                    valid_sampler,
                    final_cfg,
                    rank,
                    world_size,
                    step5_collate_fn=step5_collate_fn,
                )
            except Exception as exc:
                if rank == 0:
                    log_training_crash(train_logger, exc)
                raise
        if eval_only and not os.path.isfile(final_cfg.save_file):
            raise FileNotFoundError(
                f"eval/test/generate_samples 需要已有权重文件，未找到: {final_cfg.save_file}\n"
                "请确认 ODCR_STAGE_RUN_DIR 指向含 model/best.pth（或显式 --save_file）的训练 run。"
            )
        dist.barrier()
        run_final_eval = eval_only or (args.command == "train" and not getattr(args, "train_only", False))
        if run_final_eval:
            import time as _time

            _eval_t0 = _time.perf_counter()
            _eval_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _split_lab = getattr(args, "_odcr_eval_split_label", "valid")
            _real_n = len(valid_dataset)
            _eval_global_bs = int(final_cfg.eval_batch_size)
            _eval_nw = int(final_cfg.dataloader_num_workers_valid)
            _eval_pf = final_cfg.dataloader_prefetch_factor_valid
            _review_rows: List[str] = []
            _review_meta: Dict[str, Any] = {"fast_path": "not_used", "review_rows_count": 0}
            _review_t0 = time.perf_counter()
            if str(args.command) == "eval-rerank":
                if rank == 0:
                    _review_rows, _review_meta = _load_review_by_sample_id(
                        str(getattr(args, "_odcr_eval_data_path", "") or "")
                    )
                gathered_review: List[Any] = [_review_rows, _review_meta]
                dist.broadcast_object_list(gathered_review, src=0)
                _review_rows = list(gathered_review[0] or [])
                _review_meta = dict(gathered_review[1] or {})
            _review_load_time = float(time.perf_counter() - _review_t0)
            single_safe = bool(getattr(args, "eval_single_process_safe", False)) and world_size > 1
            sanity_cmp = bool(getattr(args, "sanity_compare_ddp_single", False)) and world_size > 1
            if (not single_safe) and world_size > 1 and (_eval_global_bs % world_size != 0):
                raise ValueError(
                    f"eval_batch_size={_eval_global_bs} 与 world_size={world_size} 不整除，DDP 评测非法。"
                    "请修改 configs/odcr.yaml 中 eval.profiles.*.eval_batch_size，或调整 hardware.profiles.*.ddp_world_size。"
                )
            _embedded_eval_log_fh: Optional[logging.Handler] = None
            if rank == 0:
                _emb_lp = (os.environ.get("ODCR_STEP5_EMBEDDED_EVAL_LOG") or "").strip()
                if _emb_lp:
                    _eld = os.path.dirname(os.path.abspath(_emb_lp))
                    if _eld:
                        os.makedirs(_eld, exist_ok=True)
                    _embedded_eval_log_fh = logging.FileHandler(_emb_lp, mode="w", encoding="utf-8")
                    _embedded_eval_log_fh.setLevel(logging.DEBUG)
                    _embedded_eval_log_fh.setFormatter(
                        logging.Formatter(
                            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                        )
                    )
                    train_logger.addHandler(_embedded_eval_log_fh)
                    train_logger.info(
                        "[embedded eval log] 训练后 valid 评估日志写入 %s",
                        _emb_lp,
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )

            def _eval_export_dir() -> str:
                _ed = (os.environ.get("ODCR_EVAL_RUN_DIR") or "").strip()
                if not _ed:
                    raise RuntimeError(
                        "ODCR_EVAL_RUN_DIR 未设置。请使用: python code/odcr.py eval|eval-rerank …"
                    )
                _ed = os.path.abspath(_ed)
                os.makedirs(_ed, exist_ok=True)
                return _ed

            def _rank0_write_eval_artifacts(
                merged: List[dict],
                final: Dict[str, Any],
                *,
                pipeline_tag: str,
                eval_model: Optional[torch.nn.Module] = None,
                eval_perf: Optional[Dict[str, Any]] = None,
            ) -> None:
                ed = _eval_export_dir()
                _eval_tag = eval_decode_tag(
                    decode_strategy=str(final_cfg.decode_strategy),
                    generate_temperature=float(final_cfg.generate_temperature),
                    generate_top_p=float(final_cfg.generate_top_p),
                )
                _is_rerank = str(args.command) == "eval-rerank"
                eval_sub = ed
                os.makedirs(eval_sub, exist_ok=True)
                metrics_output_path = os.path.join(
                    os.path.abspath(eval_sub),
                    path_layout.eval_metrics_filename(rerank=_is_rerank),
                )
                legacy_metrics_path = os.path.join(os.path.abspath(eval_sub), "metrics.json")
                for existing_path in (metrics_output_path, legacy_metrics_path):
                    if not os.path.isfile(existing_path):
                        continue
                    try:
                        existing_metrics = json.loads(open(existing_path, "r", encoding="utf-8").read())
                    except (OSError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"Refusing to reuse unreadable eval metrics output: {existing_path}: {exc}"
                        ) from exc
                    if not isinstance(existing_metrics, dict) or existing_metrics.get("metrics_schema_version") != STEP5_EVAL_OUTPUT_SCHEMA_VERSION:
                        raise RuntimeError(
                            "eval/rerank refused old output schema at "
                            f"{existing_path}: stored={getattr(existing_metrics, 'get', lambda _k, _d=None: None)('metrics_schema_version')!r} "
                            f"expected={STEP5_EVAL_OUTPUT_SCHEMA_VERSION!r}. Use a new eval run id."
                        )
                    raise RuntimeError(
                        f"Refusing to overwrite existing eval metrics output: {existing_path}. "
                        "Use a new eval/rerank run id."
                    )
                checkpoint_lineage = read_checkpoint_lineage(final_cfg.save_file, expected_stage="step5")
                _ebn = (os.environ.get("ODCR_EVAL_PROFILE_NAME") or "").strip() or None
                if _ebn:
                    train_logger.info(
                        "[eval_profile_orchestrator] name=%s hardware_preset=%s decode_preset_stem=%s "
                        "rerank_preset_stem=%s global_eval_batch_size=%s ddp_world_size=%s",
                        _ebn,
                        (os.environ.get("ODCR_HARDWARE_PRESET") or "").strip(),
                        (os.environ.get("ODCR_DECODE_PRESET_STEM") or "").strip(),
                        (os.environ.get("ODCR_RERANK_PRESET_STEM") or "").strip() or "-",
                        int(final_cfg.eval_batch_size),
                        int(getattr(final_cfg, "ddp_world_size", 1) or 1),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                _decode_cfg = {
                    "decode_strategy": final_cfg.decode_strategy,
                    "decode_seed": final_cfg.decode_seed,
                    "repetition_penalty": final_cfg.repetition_penalty,
                    "generate_temperature": final_cfg.generate_temperature,
                    "generate_top_p": final_cfg.generate_top_p,
                    "max_explanation_length": final_cfg.max_explanation_length,
                    "label_smoothing": final_cfg.label_smoothing,
                    "no_repeat_ngram_size": final_cfg.no_repeat_ngram_size,
                    "min_len": final_cfg.min_len,
                    "soft_max_len": getattr(final_cfg, "soft_max_len", None),
                    "hard_max_len": getattr(final_cfg, "hard_max_len", None),
                    "eos_boost_start": getattr(final_cfg, "eos_boost_start", 9999),
                    "eos_boost_value": getattr(final_cfg, "eos_boost_value", 0.0),
                    "tail_temperature": getattr(final_cfg, "tail_temperature", -1.0),
                    "tail_top_p": getattr(final_cfg, "tail_top_p", -1.0),
                    "forbid_eos_after_open_quote": getattr(final_cfg, "forbid_eos_after_open_quote", True),
                    "forbid_eos_after_open_bracket": getattr(final_cfg, "forbid_eos_after_open_bracket", True),
                    "forbid_bad_terminal_tokens": getattr(final_cfg, "forbid_bad_terminal_tokens", True),
                    "decode_token_repeat_window": getattr(final_cfg, "decode_token_repeat_window", 4),
                    "decode_token_repeat_max": getattr(final_cfg, "decode_token_repeat_max", 2),
                    "candidate_family": getattr(final_cfg, "candidate_family", "balanced"),
                    "loss_weight_repeat_ul": getattr(final_cfg, "loss_weight_repeat_ul", 0.0),
                    "loss_weight_terminal_clean": getattr(final_cfg, "loss_weight_terminal_clean", 0.0),
                    "terminal_clean_span": getattr(final_cfg, "terminal_clean_span", 3),
                    "domain_fusion_mode": str(getattr(final_cfg, "domain_fusion_mode", "gate_cross_attn")),
                }
                generation_semantic_resolved, _ = build_generation_semantic_resolved_and_fingerprint(
                    _decode_cfg
                )
                if _is_rerank:
                    _cfp_raw = json.dumps(_decode_cfg, ensure_ascii=False, sort_keys=True, default=str)
                    train_logger.info(
                        "[CheckpointSemantics] eval_rerank candidate_decode_fingerprint_sha1=%s "
                        "decode_strategy=%s num_return_sequences=%s",
                        hashlib.sha1(_cfp_raw.encode("utf-8")).hexdigest()[:16],
                        str(final_cfg.decode_strategy),
                        int(getattr(args, "num_return_sequences", 4) or 4),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                collapse_stats = final.get("collapse_stats") or {}
                _eval_control_contract = step5_factual_eval_control_contract(_split_lab)
                metrics_payload = {
                    "metrics_schema_version": STEP5_EVAL_OUTPUT_SCHEMA_VERSION,
                    "eval_output_schema_version": STEP5_EVAL_OUTPUT_SCHEMA_VERSION,
                    "eval_control_contract": _eval_control_contract,
                    "eval_control_contract_hash": stable_hash(_eval_control_contract),
                    "eval_control_mode": STEP5_CONTROL_MODE_FACTUAL_EVAL_DEFAULT,
                    "checkpoint_compat_schema_version": STEP5_CHECKPOINT_COMPAT_SCHEMA_VERSION,
                    "step5_train_schema_version": STEP5_TRAIN_SCHEMA_VERSION,
                    "step5_checkpoint_lineage_hash": checkpoint_lineage.get("lineage_hash"),
                    "step5_checkpoint_compatibility_hash": checkpoint_lineage.get("checkpoint_compatibility_hash"),
                    "step5_checkpoint_lineage": checkpoint_lineage,
                    "training_semantic_fingerprint": (
                        (os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or None
                    ),
                    "generation_semantic_fingerprint": (
                        (os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or None
                    ),
                    "generation_semantic_resolved": generation_semantic_resolved,
                    "checkpoint": os.path.abspath(str(final_cfg.save_file)),
                    "eval_run_dir": os.path.abspath(eval_sub),
                    "metrics_path": metrics_output_path,
                    "task_idx": int(final_cfg.task_idx),
                    "split": _split_lab,
                    "split_csv": getattr(args, "_odcr_eval_data_path", ""),
                    "review_path": getattr(args, "_odcr_eval_data_path", ""),
                    "review_rows_count": int(_review_meta.get("review_rows_count", 0) or 0),
                    "review_load_fast_path": str(_review_meta.get("fast_path", "not_used")),
                    "seed": int(args.seed),
                    "world_size": world_size,
                    "eval_mode": "single_process_safe" if single_safe else "ddp_sharded",
                    "command": args.command,
                    "eval_profile_name": _ebn,
                    "eval_export_tag": _eval_tag,
                    "decode": _decode_cfg,
                    "collapse_stats": collapse_stats,
                    "paper_metrics": final.get("paper_metrics"),
                    "metrics": final,
                }
                try:
                    _ckp = os.path.abspath(str(final_cfg.save_file))
                    _parent = os.path.dirname(_ckp)
                    if os.path.basename(_parent) == "model":
                        _run_root = os.path.dirname(_parent)
                        metrics_payload["step5_checkpoint_run_dir"] = _run_root
                        metrics_payload["step5_run_id"] = os.path.basename(_run_root)
                except Exception:
                    pass
                merged_for_pred: List[dict] = merged
                if _is_rerank:
                    _rrm = _rerank_eval_cli_resolved(args)
                    _ex_mode = str(_rrm["export_examples_mode"]).strip().lower()
                    _rm = str(_rrm["rerank_method"])
                    rs = _aggregate_rerank_summary(
                        merged,
                        export_examples_mode=_ex_mode,
                        rerank_method=_rm,
                    )
                    rw = build_rerank_weights_dict(
                        weight_logprob=float(_rrm["rerank_weight_logprob"]),
                        weight_length=float(_rrm["rerank_weight_length"]),
                        weight_repeat=float(_rrm["rerank_weight_repeat"]),
                        weight_dirty=float(_rrm["rerank_weight_dirty"]),
                    )
                    _mtail_c = float(_rrm["rerank_malformed_tail_penalty"])
                    _mtok_c = float(_rrm["rerank_malformed_token_penalty"])
                    rs["rerank_weights"] = rw
                    rs["rerank_malformed_tail_coef"] = _mtail_c
                    rs["rerank_malformed_token_coef"] = _mtok_c
                    metrics_payload["rerank_enabled"] = True
                    metrics_payload["rerank_method"] = _rm
                    metrics_payload["num_return_sequences"] = int(_rrm["num_return_sequences"])
                    metrics_payload["rerank_top_k"] = int(_rrm["rerank_top_k"])
                    metrics_payload["rerank_weights"] = rw
                    metrics_payload["rerank_target_len_ratio"] = float(_rrm["rerank_target_len_ratio"])
                    metrics_payload["rerank_malformed_tail_penalty"] = _mtail_c
                    metrics_payload["rerank_malformed_token_penalty"] = _mtok_c
                    metrics_payload["rerank_source_table"] = dict(_rrm.get("rerank_source_table") or {})
                    metrics_payload["rerank_profile_effective"] = dict(_rrm.get("rerank_profile") or {})
                    metrics_payload["export_examples_mode"] = _ex_mode
                    metrics_payload["export_full_rerank_examples"] = bool(
                        getattr(args, "export_full_rerank_examples", False)
                    )
                    metrics_payload["rerank_logprob_source"] = (
                        "per_token_log_softmax_at_chosen_id; same logits/preprocessing as generate "
                        "(nucleus: tempered softmax; greedy: argmax on same logits)"
                    )
                    metrics_payload["rerank_summary"] = rs
                    if "v3" in _rm.replace("_", ""):
                        metrics_payload["rerank_v3_summary"] = dict(rs)
                        _rrpj = (os.environ.get("ODCR_RERANK_PROFILE_JSON") or "").strip()
                        if _rrpj:
                            try:
                                metrics_payload["rerank_v3_profile_effective"] = json.loads(_rrpj)
                            except Exception:
                                metrics_payload["rerank_v3_profile_effective"] = {}
                    merged_for_pred = [{k: v for k, v in r.items() if k != "_rerank"} for r in merged]
                    _export_full = bool(
                        getattr(args, "export_full_rerank_examples", False)
                    ) or (_ex_mode == "full")
                    _rr0 = time.perf_counter()
                    _write_rerank_artifacts(
                        eval_sub,
                        merged,
                        rerank_cfg={
                            "rerank_method": metrics_payload["rerank_method"],
                            "num_return_sequences": metrics_payload["num_return_sequences"],
                            "rerank_top_k": metrics_payload["rerank_top_k"],
                            "rerank_weights": rw,
                            "rerank_target_len_ratio": metrics_payload["rerank_target_len_ratio"],
                            "rerank_source_table": metrics_payload["rerank_source_table"],
                            "rerank_profile_effective": metrics_payload["rerank_profile_effective"],
                            "export_examples_mode": _ex_mode,
                            "export_full_rerank_examples": _export_full,
                            "rerank_malformed_tail_penalty": _mtail_c,
                            "rerank_malformed_token_penalty": _mtok_c,
                        },
                        rerank_summary=rs,
                        export_examples_mode=_ex_mode,
                        export_full_rerank_examples=_export_full,
                    )
                    if eval_perf is not None:
                        eval_perf["rerank_artifacts_write_time"] = float(time.perf_counter() - _rr0)
                if eval_model is not None:
                    _um = get_underlying_model(eval_model)
                    metrics_payload["generate_kwargs_effective"] = _um.get_generate_kwargs_effective()
                    metrics_payload["generate_kwargs_effective_v2"] = _um.get_generate_kwargs_effective_v2()
                    metrics_payload["domain_fusion_mode"] = str(
                        getattr(_um, "domain_fusion_mode", "gate_cross_attn")
                    )
                    metrics_payload["domain_gate_stats"] = getattr(_um, "get_domain_gate_stats", lambda: {})()
                    train_logger.info(
                        "[DMPF gate] mode=%s stats=%s",
                        metrics_payload["domain_fusion_mode"],
                        json.dumps(metrics_payload["domain_gate_stats"], ensure_ascii=False, default=str),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                _ckpt_abs = os.path.abspath(str(final_cfg.save_file))
                _eval_meta = {
                    "checkpoint": _ckpt_abs,
                    "eval_export_tag": _eval_tag,
                    "eval_run_dir": os.path.abspath(eval_sub),
                    "decode": _decode_cfg,
                    "recommendation": final.get("recommendation"),
                    "bleu4": final.get("explanation", {}).get("bleu", {}).get("4"),
                    "meteor": final.get("explanation", {}).get("meteor"),
                    "rouge_l": final.get("explanation", {}).get("rouge", {}).get("l"),
                    "collapse_top1_ratio": collapse_stats.get("top1_pred_ratio"),
                    "collapse_unique_ratio": collapse_stats.get("pred_unique_ratio"),
                    "collapse_stats": collapse_stats,
                }
                try:
                    with open(
                        os.path.join(eval_sub, "eval_checkpoint_sidecar.json"),
                        "w",
                        encoding="utf-8",
                    ) as _emf:
                        json.dump(_eval_meta, _emf, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    pass
                csv_fields = ["sample_id", "pred_rating", "gt_rating", "pred_text", "ref_text"]
                if _is_rerank:
                    csv_fields.extend(["candidate_family", "lp_norm", "completion_ok"])
                _pw0 = time.perf_counter()
                write_predictions_csv(
                    os.path.join(eval_sub, "predictions.csv"), merged_for_pred, csv_fields
                )
                write_predictions_jsonl(os.path.join(eval_sub, "predictions.jsonl"), merged_for_pred)
                if eval_perf is not None:
                    eval_perf["predictions_write_time"] = float(time.perf_counter() - _pw0)
                _perf = dict(eval_perf) if eval_perf is not None else {}
                _perf["total_eval_time"] = float(time.perf_counter() - _eval_t0)
                _summary_keys = (
                    "review_load_time",
                    "tokenize_cache_time",
                    "eval_dataset_build_time",
                    "eval_dataloader_build_time",
                    "decode_time",
                    "rerank_feature_time",
                    "gather_time",
                    "metrics_time",
                    "predictions_write_time",
                    "rerank_scoring_time",
                    "rerank_artifacts_write_time",
                    "total_eval_time",
                )
                _rt_eval_snap = runtime_env_dict_for_config_resolved()
                metrics_payload["eval_performance"] = {
                    "global_eval_batch_size": int(_eval_global_bs),
                    "eval_per_gpu_batch_size": int(
                        min(_eval_global_bs, max(1, _real_n))
                        if single_safe
                        else (_eval_global_bs // world_size)
                    ),
                    "dataloader_num_workers_valid": int(_eval_nw),
                    "dataloader_prefetch_factor_valid": _eval_pf,
                    "hardware_preset": (os.environ.get("ODCR_HARDWARE_PRESET") or "").strip() or None,
                    "runtime_env": _rt_eval_snap,
                    "summary": {k: float(_perf[k]) for k in _summary_keys if k in _perf},
                    "detail": _perf,
                }
                with open(metrics_output_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_payload, f, ensure_ascii=False, indent=2, default=str)
                _cfg_eval = dict(final_cfg.to_log_dict())
                _ignored_eval = [k for k in _STEP5_IGNORED_FIELDS if k in _cfg_eval]
                for _k in _ignored_eval:
                    _cfg_eval.pop(_k, None)
                _cfg_eval["step5_ignored_fields"] = list(_ignored_eval)
                _cfg_eval["training_diagnostics"] = training_diagnostics_snapshot(
                    diagnostics_scope="child",
                    effective_training_payload_json=os.environ.get(
                        "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON", ""
                    ),
                    ddp_find_unused_parameters_effective=bool(final_cfg.ddp_find_unused_parameters),
                )
                _cfg_eval["training_semantic_fingerprint"] = (
                    (os.environ.get("ODCR_TRAINING_SEMANTIC_FINGERPRINT") or "").strip() or None
                )
                _cfg_eval["generation_semantic_fingerprint"] = (
                    (os.environ.get("ODCR_GENERATION_SEMANTIC_FINGERPRINT") or "").strip() or None
                )
                _cfg_eval["runtime_diagnostics_fingerprint"] = (
                    (os.environ.get("ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT") or "").strip() or None
                )

                _cfg_eval["runtime_env"] = _rt_eval_snap

                eval_meta_dir = os.path.join(eval_sub, "meta")
                os.makedirs(eval_meta_dir, exist_ok=True)
                with open(os.path.join(eval_meta_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
                    json.dump(_cfg_eval, f, ensure_ascii=False, indent=2, default=str)
                train_logger.info(
                    "[eval_performance] %s",
                    json.dumps(
                        metrics_payload.get("eval_performance") or {},
                        ensure_ascii=False,
                        default=str,
                    ),
                    extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                )
                log_sample_id_alignment_snippet(merged_for_pred, k=20, logger=train_logger)
                if _is_rerank:
                    _rs = metrics_payload.get("rerank_summary") or {}
                    train_logger.info(
                        "[rerank_effective] K=%s method=%s selected_not_best_logprob_rate=%.6g mean_selected_rerank_score=%s",
                        metrics_payload.get("num_return_sequences"),
                        metrics_payload.get("rerank_method"),
                        float(_rs.get("selected_not_best_logprob_rate", float("nan"))),
                        _rs.get("mean_selected_rerank_score"),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                    train_logger.info(
                        "[rerank_summary] %s",
                        json.dumps(_rs, ensure_ascii=False, default=str),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                _cw = collapse_stats.get("collapse_warnings") or []
                if _cw:
                    train_logger.warning(
                        "[Collapse warning] %s",
                        "; ".join(str(x) for x in _cw),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                _task_desc = (
                    f"Step 5 Task {task_idx} {args.command} (nproc={world_size}, split={_split_lab}): "
                    f"{args.auxiliary} -> {args.target} | eval_tag={_eval_tag}"
                )
                _eval_elapsed = time.perf_counter() - _eval_t0
                _eval_min, _eval_sec = divmod(int(_eval_elapsed), 60)
                _lines = format_final_results_lines(
                    final,
                    task_description=_task_desc,
                    start_time=_eval_start_str,
                    decode_cfg=_decode_cfg,
                    collapse_stats=collapse_stats,
                    eval_run_tag=_eval_tag,
                )
                _lines.append(f"Eval elapsed: {_eval_min}m {_eval_sec}s ({_eval_elapsed:.1f}s)")
                _lines.append(f"Eval artefacts: {eval_sub}")
                log_final_results_block(train_logger, _lines)
                finalize_run_log(train_logger)
                try:
                    flush_odcr_file_handlers(train_logger)
                    _digest_p = write_eval_digest_log(
                        eval_subdir=eval_sub,
                        metrics_final=final,
                        merged_rows=merged_for_pred,
                        final_cfg=final_cfg,
                        decode_cfg=dict(_decode_cfg),
                        active_log_file=(log_path or "").strip() or None,
                        task_idx=int(final_cfg.task_idx),
                        auxiliary=str(args.auxiliary),
                        target=str(args.target),
                        eval_export_tag=_eval_tag,
                        command=str(args.command),
                        eval_timing_summary=dict(
                            (metrics_payload.get("eval_performance") or {}).get("summary") or {}
                        ),
                    )
                    train_logger.info(
                        "[eval_digest] wrote %s",
                        _digest_p,
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                except Exception as _digest_exc:
                    train_logger.warning(
                        "[eval_digest] 生成 eval_digest.log 失败: %s",
                        _digest_exc,
                        exc_info=True,
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                train_logger.info("DONE.")

            if single_safe:
                dist.barrier()
                if rank == 0:
                    eval_model = _make_model(final_cfg, args, local_rank)
                    _load_step5_checkpoint_fail_fast(eval_model, final_cfg.save_file, final_cfg, local_rank)
                    train_logger.info(
                        "[generate_kwargs_effective] %s",
                        json.dumps(
                            get_underlying_model(eval_model).get_generate_kwargs_effective(),
                            ensure_ascii=False,
                            default=str,
                        ),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                    _bs = min(_eval_global_bs, max(1, _real_n))
                    _dl_c0 = time.perf_counter()
                    eval_dataloader = DataLoader(
                        valid_dataset,
                        batch_size=_bs,
                        shuffle=False,
                        num_workers=min(_eval_nw, 2),
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=min(_eval_nw, 2) > 0,
                        prefetch_factor=_eval_pf if min(_eval_nw, 2) > 0 else None,
                        collate_fn=step5_collate_fn,
                    )
                    _eval_perf: Dict[str, Any] = {
                        "review_load_time": _review_load_time,
                        "tokenize_cache_time": float(
                            getattr(args, "_odcr_eval_tokenize_cache_wall_s", 0.0) or 0.0
                        ),
                        "eval_dataset_build_time": float(
                            getattr(args, "_odcr_eval_dataset_build_wall_s", 0.0) or 0.0
                        ),
                        "eval_dataloader_build_time": float(time.perf_counter() - _dl_c0),
                    }
                    rows_local, _t_loc = _eval_rows_local(
                        eval_model, eval_dataloader, local_rank, args, review_rows=_review_rows
                    )
                    _eval_perf["decode_time"] = float(_t_loc.get("decode_time", 0.0))
                    _eval_perf["rerank_feature_time"] = float(_t_loc.get("rerank_feature_time", 0.0))
                    _eval_perf["rerank_scoring_time"] = float(_t_loc.get("rerank_scoring_time", 0.0))
                    _eval_perf["gather_time"] = 0.0
                    _mt0 = time.perf_counter()
                    merged = merge_eval_rows_by_sample_id([rows_local], _real_n)
                    final, _, _ = _metrics_final_dict_from_rows(merged)
                    _eval_perf["metrics_time"] = float(time.perf_counter() - _mt0)
                    _rank0_write_eval_artifacts(
                        merged,
                        final,
                        pipeline_tag=f"run_odcr_{args.command}_single_safe",
                        eval_model=eval_model,
                        eval_perf=_eval_perf,
                    )
                dist.barrier()
                if rank == 0 and sanity_cmp:
                    train_logger.info(
                        "[Eval sanity] 已使用 --eval-single-process-safe，跳过 DDP/单路二次对比（请分别跑 DDP 与 safe 两次对比指标）。",
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
            else:
                eval_model = _make_model(final_cfg, args, local_rank)
                _load_step5_checkpoint_fail_fast(eval_model, final_cfg.save_file, final_cfg, local_rank)
                if rank == 0:
                    train_logger.info(
                        "[generate_kwargs_effective] %s",
                        json.dumps(
                            get_underlying_model(eval_model).get_generate_kwargs_effective(),
                            ensure_ascii=False,
                            default=str,
                        ),
                        extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                    )
                if _eval_global_bs % world_size != 0:
                    raise ValueError(
                        f"eval_batch_size={_eval_global_bs} 与 world_size={world_size} 不整除，DDP 评测非法。"
                        "请修改 configs/odcr.yaml 中 eval.profiles.*.eval_batch_size，或调整 hardware.profiles.*.ddp_world_size。"
                    )
                _eval_per_gpu = _eval_global_bs // world_size
                eval_sampler = DistributedSampler(
                    valid_dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                _dl_c1 = time.perf_counter()
                eval_dataloader = DataLoader(
                    valid_dataset,
                    batch_size=_eval_per_gpu,
                    sampler=eval_sampler,
                    shuffle=False,
                    num_workers=_eval_nw,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=_eval_nw > 0,
                    prefetch_factor=_eval_pf,
                    collate_fn=step5_collate_fn,
                )
                _eval_perf_ddp: Dict[str, Any] = {}
                if rank == 0:
                    _eval_perf_ddp["review_load_time"] = _review_load_time
                    _eval_perf_ddp["tokenize_cache_time"] = float(
                        getattr(args, "_odcr_eval_tokenize_cache_wall_s", 0.0) or 0.0
                    )
                    _eval_perf_ddp["eval_dataset_build_time"] = float(
                        getattr(args, "_odcr_eval_dataset_build_wall_s", 0.0) or 0.0
                    )
                    _eval_perf_ddp["eval_dataloader_build_time"] = float(time.perf_counter() - _dl_c1)
                rows_local, _t_loc_ddp = _eval_rows_local(
                    eval_model, eval_dataloader, local_rank, args, review_rows=_review_rows
                )
                gathered_rows: List[Any] = [None] * world_size
                if rank == 0:
                    _ga0 = time.perf_counter()
                dist.all_gather_object(gathered_rows, rows_local)
                if rank == 0:
                    _eval_perf_ddp["gather_time"] = float(time.perf_counter() - _ga0)
                    _eval_perf_ddp["decode_time"] = float(_t_loc_ddp.get("decode_time", 0.0))
                    _eval_perf_ddp["rerank_feature_time"] = float(
                        _t_loc_ddp.get("rerank_feature_time", 0.0)
                    )
                    _eval_perf_ddp["rerank_scoring_time"] = float(_t_loc_ddp.get("rerank_scoring_time", 0.0))
                    _mm0 = time.perf_counter()
                    merged = merge_eval_rows_by_sample_id(gathered_rows, _real_n)
                    final, _, _ = _metrics_final_dict_from_rows(merged)
                    _eval_perf_ddp["metrics_time"] = float(time.perf_counter() - _mm0)
                    pl = (
                        f"run_odcr_{args.command}_eval"
                        if eval_only
                        else "run_odcr_train_eval"
                    )
                    _rank0_write_eval_artifacts(
                        merged,
                        final,
                        pipeline_tag=pl,
                        eval_model=eval_model,
                        eval_perf=_eval_perf_ddp,
                    )
                    if sanity_cmp:
                        eval_model_s = _make_model(final_cfg, args, local_rank)
                        _load_step5_checkpoint_fail_fast(eval_model_s, final_cfg.save_file, final_cfg, local_rank)
                        _bs2 = min(_eval_global_bs, max(1, _real_n))
                        dl_s = DataLoader(
                            valid_dataset,
                            batch_size=_bs2,
                            shuffle=False,
                            num_workers=min(_eval_nw, 2),
                            pin_memory=torch.cuda.is_available(),
                            collate_fn=step5_collate_fn,
                        )
                        rows_s = evalModel(
                            eval_model_s,
                            dl_s,
                            local_rank,
                            step5_innov_cfg=parse_step5_innovation_config_json(
                                str(final_cfg.step5_innovation_config_json)
                            ),
                        )["rows"]
                        merged_s = merge_eval_rows_by_sample_id([rows_s], _real_n)
                        final_s, _, _ = _metrics_final_dict_from_rows(merged_s)
                        d_mae = abs(float(final["recommendation"]["mae"]) - float(final_s["recommendation"]["mae"]))
                        d_rmse = abs(float(final["recommendation"]["rmse"]) - float(final_s["recommendation"]["rmse"]))
                        d_bleu = abs(float(final["explanation"]["bleu"]["4"]) - float(final_s["explanation"]["bleu"]["4"]))
                        train_logger.info(
                            "[Eval sanity] DDP vs rank0-sequential | d_mae=%.6g d_rmse=%.6g d_bleu4=%.6g",
                            d_mae,
                            d_rmse,
                            d_bleu,
                            extra=log_route_extra(train_logger, ROUTE_SUMMARY),
                        )
            if _embedded_eval_log_fh is not None:
                train_logger.removeHandler(_embedded_eval_log_fh)
                _embedded_eval_log_fh.close()
        elif rank == 0:
            _sf = final_cfg.save_file
            train_logger.info("DONE（train --train-only：已跳过训练后评估；权重: %s）。", _sf)
            finalize_run_log(train_logger)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
