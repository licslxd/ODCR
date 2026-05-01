# -*- coding: utf-8 -*-
"""
Explanation BLEU（quick / full / DDP）单一数据路径：batch → gather → generate → 行 → 聚合算分。
quick 与 full 仅 indices 范围及是否 all_gather_object 不同，禁止重复解包逻辑。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from base_utils import compute_bleu1234_only, get_underlying_model
from odcr_core.gather_schema import GatheredBatch, require_gathered_batch
from odcr_core.mainline_monitor import (
    build_mainline_monitor_bundle_from_merged_rows,
    safe_summarize_uncertainty_decode_aggregate,
)
from executors.decode_controller import (
    empty_uncertainty_decode_aggregate,
    merge_uncertainty_run_into_aggregate,
    reduce_uncertainty_aggregates,
)
from odcr_eval_metrics import merge_eval_rows_by_sample_id
from train_diagnostics import odcr_cuda_bf16_autocast

_SCHEMA_LOGGED = False


def _quick_bleu_safe_collate(batch: List[Any]) -> Any:
    """
    quick BLEU 专用安全 collate：
    - dict 样本：仅抽取 gather 需要的字段，序列字段 explanation_idx 动态 pad；
    - 其他样本：回退 default_collate（兼容旧 6-tuple 路径）。
    """
    if not batch:
        raise ValueError("quick BLEU collate 收到空 batch。")
    first = batch[0]
    if not isinstance(first, dict):
        return default_collate(batch)

    def _stack_scalar(key: str, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([torch.as_tensor(x[key], dtype=dtype) for x in batch], dim=0)

    user_idx = _stack_scalar("user_idx", dtype=torch.long)
    item_idx = _stack_scalar("item_idx", dtype=torch.long)
    rating = _stack_scalar("rating", dtype=torch.float32)
    domain_idx = _stack_scalar("domain_idx", dtype=torch.long)
    sample_id = _stack_scalar("sample_id", dtype=torch.long)
    seqs = [torch.as_tensor(x["explanation_idx"], dtype=torch.long).view(-1) for x in batch]
    max_len = max(1, max(int(s.numel()) for s in seqs))
    tgt_output = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        n = min(max_len, int(s.numel()))
        tgt_output[i, :n] = s[:n]

    if all("exp_sample_weight" in x for x in batch):
        exp_sample_weight = _stack_scalar("exp_sample_weight", dtype=torch.float32)
        return (user_idx, item_idx, rating, tgt_output, domain_idx, sample_id, exp_sample_weight)
    return (user_idx, item_idx, rating, tgt_output, domain_idx, sample_id)


def maybe_log_gather_schema_banner(*, rank: int, logger: Optional[logging.Logger]) -> None:
    """rank0 单次：在 ODCR_LOG_GATHER_SCHEMA=1 时打印 GatheredBatch + 本模块职责（避免默认刷屏）。"""
    global _SCHEMA_LOGGED
    if _SCHEMA_LOGGED or rank != 0:
        return
    flag = os.environ.get("ODCR_LOG_GATHER_SCHEMA", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return
    _SCHEMA_LOGGED = True
    msg = (
        "[Gather/BLEU 协议] gather → GatheredBatch("
        "user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx, sample_id, exp_sample_weight?)；"
        "BLEU 行构建入口: odcr_core.bleu_runtime.build_explanation_bleu_rows_for_indices"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)


def build_explanation_bleu_rows_for_indices(
    underlying_model: torch.nn.Module,
    tokenizer: Any,
    device: int,
    valid_dataset,
    indices: List[int],
    batch_size: int,
    *,
    rank: int = 0,
    logger: Optional[logging.Logger] = None,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: Optional[int],
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    cfg_override: Optional[Mapping[str, Any]] = None,
    include_ratings_for_monitor: bool = False,
    uncertainty_acc: Optional[MutableMapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    在 valid_dataset 上取给定全局 indices，逐批 gather（须为 GatheredBatch）、generate，产出 BLEU 评估行。
    sample_id 取自 GatheredBatch.sample_id（显式字段，不再按 tuple 长度推断）。
    cfg_override：仅当次 generate 的临时解码覆盖（如 full BLEU monitor greedy），不传则沿用模型默认解码。
    """
    maybe_log_gather_schema_banner(rank=rank, logger=logger)
    if not indices:
        return []
    subset = Subset(valid_dataset, indices)
    n = len(subset)
    bs = max(1, min(int(batch_size), n))
    _vn = max(0, int(dataloader_num_workers))
    _pf = dataloader_prefetch_factor if _vn > 0 else None
    dl = DataLoader(
        subset,
        batch_size=bs,
        shuffle=False,
        num_workers=_vn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=_vn > 0,
        prefetch_factor=_pf,
        collate_fn=collate_fn or _quick_bleu_safe_collate,
    )
    out: List[Dict[str, Any]] = []
    underlying_model.eval()
    with torch.inference_mode():
        for batch in dl:
            g = require_gathered_batch(underlying_model.gather(batch, device))
            user_idx = g.user_idx
            item_idx = g.item_idx
            domain_idx = g.domain_idx
            tgt_output = g.tgt_output
            sample_id = g.sample_id
            with odcr_cuda_bf16_autocast():
                pred_rating_t = None
                if include_ratings_for_monitor:
                    pred_rating_t = underlying_model.recommend(user_idx, item_idx, domain_idx)
                gen_pack = underlying_model.generate(
                    user_idx, item_idx, domain_idx, cfg_override=cfg_override
                )
                gen_ids = gen_pack[0]
                if uncertainty_acc is not None:
                    merge_uncertainty_run_into_aggregate(
                        uncertainty_acc,
                        getattr(underlying_model, "_last_uncertainty_decode_stats", None),
                    )
            pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(tgt_output, skip_special_tokens=True)
            gt_rating = g.rating
            bsz = int(user_idx.size(0))
            for i in range(bsz):
                sid = int(sample_id[i].item())
                row: Dict[str, Any] = {
                    "sample_id": sid,
                    "pred_text": pred_texts[i],
                    "ref_text": ref_texts[i],
                }
                if include_ratings_for_monitor and pred_rating_t is not None:
                    row["pred_rating"] = float(pred_rating_t[i].item())
                    row["gt_rating"] = float(gt_rating[i].item())
                out.append(row)
    return out


def explanation_bleu4_score_from_local_rows(rows: List[Dict[str, Any]], expected_n: int) -> float:
    """单进程：行数应覆盖 sample_id 0..expected_n-1，与 merge_eval_rows_by_sample_id 约定一致。"""
    merged = merge_eval_rows_by_sample_id([rows], expected_n)
    preds = [r["pred_text"] for r in merged]
    refs = [r["ref_text"] for r in merged]
    return float(compute_bleu1234_only(preds, refs).get("4", 0.0))


def explanation_bleu4_score_from_ddp_shard_rows(
    rows_per_rank: Sequence[List[Dict[str, Any]]],
    expected_n: int,
) -> float:
    """rank0：各 rank 本地行 all_gather 后按 sample_id 合并再算 BLEU-4。"""
    merged = merge_eval_rows_by_sample_id(rows_per_rank, expected_n)
    preds = [r["pred_text"] for r in merged]
    refs = [r["ref_text"] for r in merged]
    return float(compute_bleu1234_only(preds, refs).get("4", 0.0))


def explanation_bleu4_quick_score(
    model: torch.nn.Module,
    tokenizer: Any,
    valid_dataset,
    device: int,
    max_samples: int,
    *,
    rank: int = 0,
    logger: Optional[logging.Logger] = None,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: Optional[int],
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> float:
    """验证集前 max_samples 条上的 quick BLEU-4（仅 rank0 调用）。"""
    _m = get_underlying_model(model)
    n = min(len(valid_dataset), int(max_samples))
    if n <= 0:
        return 0.0
    indices = list(range(n))
    rows = build_explanation_bleu_rows_for_indices(
        _m,
        tokenizer,
        device,
        valid_dataset,
        indices,
        batch_size=32,
        rank=rank,
        logger=logger,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        collate_fn=collate_fn,
    )
    if not rows:
        return 0.0
    return explanation_bleu4_score_from_local_rows(rows, n)


def bleu4_explanation_full_valid_ddp(
    model: torch.nn.Module,
    valid_dataset,
    *,
    tokenizer: Any,
    device: int,
    rank: int,
    world_size: int,
    batch_size: int = 32,
    dataloader_num_workers: int = 2,
    dataloader_prefetch_factor: Optional[int] = 4,
    logger: Optional[logging.Logger] = None,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    cfg_override: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    完整 valid 上 explanation BLEU-4；各 rank 连续分片，all_gather_object 后 rank0 按 sample_id 合并。
    cfg_override：训练期 full BLEU 监控解码（如 greedy）；quick BLEU 路径勿传。
    """
    _m = get_underlying_model(model)
    n = len(valid_dataset)
    if n <= 0:
        return 0.0

    if world_size <= 1:
        rows = build_explanation_bleu_rows_for_indices(
            _m,
            tokenizer,
            device,
            valid_dataset,
            list(range(n)),
            batch_size,
            rank=rank,
            logger=logger,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            collate_fn=collate_fn,
            cfg_override=cfg_override,
        )
        return explanation_bleu4_score_from_local_rows(rows, n)

    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(n, start + chunk)
    indices: List[int] = list(range(start, end)) if start < n else []

    rows = build_explanation_bleu_rows_for_indices(
        _m,
        tokenizer,
        device,
        valid_dataset,
        indices,
        batch_size,
        rank=rank,
        logger=logger,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        collate_fn=collate_fn,
        cfg_override=cfg_override,
    )
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, rows)

    score = 0.0
    if rank == 0:
        score = explanation_bleu4_score_from_ddp_shard_rows(gathered, n)

    t = torch.zeros(1, dtype=torch.float32, device=device)
    if rank == 0:
        t[0] = score
    dist.broadcast(t, src=0)
    return float(t[0].item())


def mainline_monitor_full_valid_ddp(
    model: torch.nn.Module,
    valid_dataset,
    *,
    tokenizer: Any,
    device: int,
    rank: int,
    world_size: int,
    batch_size: int = 32,
    dataloader_num_workers: int = 2,
    dataloader_prefetch_factor: Optional[int] = 4,
    logger: Optional[logging.Logger] = None,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    cfg_override: Optional[Mapping[str, Any]] = None,
    composite_weights: Optional[Mapping[str, float]] = None,
    uncertainty_high_entropy_threshold: float = 1.0,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    完整 valid：主路径 decode（与 cfg_override / 模型默认一致）下的指标包 + guarded 复合分。
    各 rank 返回相同的 composite（broadcast）；仅 rank0 返回 bundle 字典，其余 rank 返回 (score, None)。
    """
    _m = get_underlying_model(model)
    n = len(valid_dataset)
    if n <= 0:
        return 0.0, {} if rank == 0 else None

    if world_size <= 1:
        unc_acc = empty_uncertainty_decode_aggregate()
        rows = build_explanation_bleu_rows_for_indices(
            _m,
            tokenizer,
            device,
            valid_dataset,
            list(range(n)),
            batch_size,
            rank=rank,
            logger=logger,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            collate_fn=collate_fn,
            cfg_override=cfg_override,
            include_ratings_for_monitor=True,
            uncertainty_acc=unc_acc,
        )
        merged = merge_eval_rows_by_sample_id([rows], n)
        bundle = build_mainline_monitor_bundle_from_merged_rows(
            merged, composite_weights=composite_weights
        )
        bundle["uncertainty_decode_summary"] = safe_summarize_uncertainty_decode_aggregate(
            unc_acc,
            uncertainty_high_entropy_threshold=float(uncertainty_high_entropy_threshold),
            logger=logger,
            ctx="mainline_monitor_full_valid_ddp:world_size=1",
        )
        return float(bundle["mainline_composite_score"]), bundle

    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(n, start + chunk)
    indices: List[int] = list(range(start, end)) if start < n else []

    unc_acc = empty_uncertainty_decode_aggregate()
    rows = build_explanation_bleu_rows_for_indices(
        _m,
        tokenizer,
        device,
        valid_dataset,
        indices,
        batch_size,
        rank=rank,
        logger=logger,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        collate_fn=collate_fn,
        cfg_override=cfg_override,
        include_ratings_for_monitor=True,
        uncertainty_acc=unc_acc,
    )
    gathered: List[Any] = [None] * world_size
    dist.all_gather_object(gathered, rows)
    gathered_u: List[Any] = [None] * world_size
    dist.all_gather_object(gathered_u, unc_acc)

    score = 0.0
    details: Optional[Dict[str, Any]] = None
    if rank == 0:
        merged = merge_eval_rows_by_sample_id(gathered, n)
        details = build_mainline_monitor_bundle_from_merged_rows(
            merged, composite_weights=composite_weights
        )
        merged_u = reduce_uncertainty_aggregates(gathered_u)
        details["uncertainty_decode_summary"] = safe_summarize_uncertainty_decode_aggregate(
            merged_u,
            uncertainty_high_entropy_threshold=float(uncertainty_high_entropy_threshold),
            logger=logger,
            ctx="mainline_monitor_full_valid_ddp:ddp_merge",
        )
        score = float(details["mainline_composite_score"])

    t = torch.zeros(1, dtype=torch.float32, device=device)
    if rank == 0:
        t[0] = score
    dist.broadcast(t, src=0)
    return float(t[0].item()), details


def smoke_bleu_protocol_merge_and_score() -> None:
    """无 GPU：仅校验 DDP 合并 + 打分链路（供 CLI / CI 轻量冒烟）。"""
    rows0 = [
        {"sample_id": 0, "pred_text": "a b", "ref_text": "a b"},
        {"sample_id": 1, "pred_text": "c d", "ref_text": "c d"},
    ]
    rows1 = [
        {"sample_id": 2, "pred_text": "e f", "ref_text": "e f"},
        {"sample_id": 3, "pred_text": "g h", "ref_text": "g h"},
    ]
    s = explanation_bleu4_score_from_ddp_shard_rows([rows0, rows1], 4)
    if s < 0.0:
        raise RuntimeError("smoke: unexpected negative bleu")
    require_gathered_batch(
        GatheredBatch(
            user_idx=torch.zeros(1, dtype=torch.long),
            item_idx=torch.zeros(1, dtype=torch.long),
            rating=torch.zeros(1),
            tgt_input=torch.zeros(1, 1, dtype=torch.long),
            tgt_output=torch.zeros(1, 1, dtype=torch.long),
            domain_idx=torch.zeros(1, dtype=torch.long),
            sample_id=torch.zeros(1, dtype=torch.long),
            exp_sample_weight=None,
        )
    )
