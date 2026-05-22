"""
Step5 official full-valid monitor: paper-greedy text metrics + gate.

Official Step5_e checkpoint selection uses the same 25-token metric_pred /
metric_ref contract as final eval. Diagnostic decode profiles may still call
this helper, but raw pred_text/ref_text are never the preferred metric inputs.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from base_utils import official_paper_metrics
from odcr_eval_dirty_text import compute_dirty_text_stats


def ratings_rmse_mae(
    pred: Sequence[float], gt: Sequence[float]
) -> Tuple[float, float]:
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.size == 0:
        return 0.0, 0.0
    d = p - g
    rmse = float(math.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    return rmse, mae


def default_checkpoint_composite_weights() -> Dict[str, float]:
    return {
        "w_bleu4": 0.32,
        "w_rouge_l": 0.28,
        "w_meteor": 0.22,
        "w_dist1": 0.05,
        "w_dist2": 0.05,
        "w_dirty": 0.12,
    }


def checkpoint_guarded_composite_score(
    *,
    bleu4_pct: float,
    rouge_l_pct: float,
    meteor_pct: float,
    dist1_pct: float,
    dist2_pct: float,
    dirty_hit_rate: float,
    weights: Mapping[str, float],
) -> float:
    """将百分制指标压到 [0,1] 量级后加权；dirty 为比率，直接惩罚。"""
    wb = float(weights.get("w_bleu4", 0.32))
    wr = float(weights.get("w_rouge_l", 0.28))
    wm = float(weights.get("w_meteor", 0.22))
    wd1 = float(weights.get("w_dist1", 0.05))
    wd2 = float(weights.get("w_dist2", 0.05))
    wdr = float(weights.get("w_dirty", 0.12))
    return (
        wb * (bleu4_pct / 100.0)
        + wr * (rouge_l_pct / 100.0)
        + wm * (meteor_pct / 100.0)
        + wd1 * (dist1_pct / 100.0)
        + wd2 * (dist2_pct / 100.0)
        - wdr * float(dirty_hit_rate)
    )


def build_mainline_monitor_bundle_from_merged_rows(
    merged_rows: Sequence[Mapping[str, Any]],
    *,
    composite_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """merged_rows must be sorted by sample_id and include official metric text."""
    preds = [str(r.get("metric_pred_text", r.get("pred_text", "")) or "") for r in merged_rows]
    refs = [str(r.get("metric_ref_text", r.get("ref_text", "")) or "") for r in merged_rows]
    txt = official_paper_metrics(preds, refs)
    mean_ref = sum(len(x.split()) for x in refs) / max(len(refs), 1)
    dirty = compute_dirty_text_stats(preds, ref_mean_len_words=mean_ref)
    dirty_rate = float(dirty.get("hit_rate", 0.0))

    rmse = 0.0
    mae = 0.0
    if merged_rows and "pred_rating" in merged_rows[0] and "gt_rating" in merged_rows[0]:
        pr = [float(r["pred_rating"]) for r in merged_rows]
        gr = [float(r["gt_rating"]) for r in merged_rows]
        rmse, mae = ratings_rmse_mae(pr, gr)

    bleu = txt.get("bleu") or {}
    rouge = txt.get("rouge") or {}
    meteor = float(txt.get("meteor") or 0.0)
    b4 = float(bleu.get("4", 0.0))
    rl = float(rouge.get("rouge_l_f", rouge.get("l", 0.0)) or 0.0)
    distinct = txt.get("distinct_corpus") or {}
    dist = (
        distinct.get("scale_percent_0_100")
        if isinstance(distinct.get("scale_percent_0_100"), Mapping)
        else txt.get("dist") or {}
    )
    d1 = float(dist.get("1", 0.0))
    d2 = float(dist.get("2", 0.0))
    w = dict(default_checkpoint_composite_weights())
    if composite_weights:
        w.update({str(k): float(v) for k, v in composite_weights.items()})
    composite = checkpoint_guarded_composite_score(
        bleu4_pct=b4,
        rouge_l_pct=rl,
        meteor_pct=meteor,
        dist1_pct=d1,
        dist2_pct=d2,
        dirty_hit_rate=dirty_rate,
        weights=w,
    )

    return {
        "bleu": bleu,
        "rouge": rouge,
        "meteor": meteor,
        "dist": dist,
        "official_paper_metrics": txt,
        "official_metric_inputs_used": True,
        "dirty_hit_rate": dirty_rate,
        "dirty_stats": dirty,
        "rmse_rating": rmse,
        "mae_rating": mae,
        "mainline_composite_score": float(composite),
        "checkpoint_composite_weights": w,
    }


def summarize_uncertainty_decode_aggregate(
    agg: Mapping[str, Any],
    *,
    uncertainty_high_entropy_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    将 empty_uncertainty_decode_aggregate / merge 后的累加容器压成可日志字段。
    first_trigger_step_* 仅在发生过至少一次触发的序列上统计；无触发时均记 nan。
    trigger_entropy_* 仅统计真正进入低温 top-k 采样的步；无此类事件时 mean/分位数等为 None。
    """
    td = int(agg.get("uncertainty_total_decision_count", 0))
    tc = int(agg.get("uncertainty_trigger_count", 0))
    rate: Optional[float] = float(tc) / float(td) if td > 0 else None
    fts = list(agg.get("uncertainty_first_trigger_steps") or [])
    arr = np.asarray(fts, dtype=np.float64)
    if arr.size == 0:
        mean_v = None
        p50_v = None
        p90_v = None
    else:
        mean_v = float(np.mean(arr))
        p50_v = float(np.percentile(arr, 50))
        p90_v = float(np.percentile(arr, 90))

    tec = int(agg.get("uncertainty_trigger_entropy_count", 0))
    tes = float(agg.get("uncertainty_trigger_entropy_sum", 0.0))
    te_vals = list(agg.get("uncertainty_trigger_entropy_values") or [])
    mean_te: Optional[float] = float(tes) / float(tec) if tec > 0 else None
    te_arr = np.asarray(te_vals, dtype=np.float64) if te_vals else np.asarray([], dtype=np.float64)
    p50_te: Optional[float] = None
    p90_te: Optional[float] = None
    if te_arr.size > 0:
        p50_te = float(np.percentile(te_arr, 50))
        p90_te = float(np.percentile(te_arr, 90))
    hi_th = float(uncertainty_high_entropy_threshold)
    high_ent_c = int(np.sum(te_arr > hi_th)) if te_arr.size > 0 else 0
    high_ent_rate: Optional[float] = (
        float(high_ent_c) / float(tec) if tec > 0 and te_arr.size > 0 else None
    )
    if tec == 0:
        high_ent_rate = None

    return {
        "uncertainty_trigger_count": tc,
        "uncertainty_total_decision_count": td,
        "uncertainty_trigger_rate": rate,
        "trigger_rate_after_prefix": rate,
        "first_trigger_step_mean": mean_v,
        "first_trigger_step_p50": p50_v,
        "first_trigger_step_p90": p90_v,
        "first_trigger_sequences": int(arr.size),
        "trigger_entropy_sum": float(tes) if tec > 0 else None,
        "trigger_entropy_count": int(tec) if tec > 0 else None,
        "mean_trigger_entropy": mean_te,
        "trigger_entropy_p50": p50_te,
        "trigger_entropy_p90": p90_te,
        "high_entropy_trigger_count": int(high_ent_c) if tec > 0 else None,
        "high_entropy_trigger_rate": high_ent_rate,
    }


def sanity_check_uncertainty_aggregate(
    agg: Mapping[str, Any],
    *,
    logger: Any = None,
    ctx: str = "",
) -> Tuple[bool, Dict[str, Any]]:
    """
    DDP / 合并后一致性检查；失败时打日志并返回 (False, detail)。
    不抛异常，供调用方回退安全 summary。
    """
    td = int(agg.get("uncertainty_total_decision_count", 0))
    tc = int(agg.get("uncertainty_trigger_count", 0))
    tec = int(agg.get("uncertainty_trigger_entropy_count", 0))
    tes = float(agg.get("uncertainty_trigger_entropy_sum", 0.0))
    te_vals = agg.get("uncertainty_trigger_entropy_values") or []
    ok = True
    reasons: List[str] = []
    if td < 0 or tc < 0 or tec < 0:
        ok = False
        reasons.append("negative_count")
    if tc > td:
        ok = False
        reasons.append("trigger_count_exceeds_total_decisions")
    if tec > tc:
        ok = False
        reasons.append("entropy_count_exceeds_trigger_count")
    if len(te_vals) != tec and tec > 0:
        ok = False
        reasons.append("entropy_values_len_mismatch_entropy_count")
    detail = {
        "passed": ok,
        "reasons": reasons,
        "uncertainty_total_decision_count": td,
        "uncertainty_trigger_count": tc,
        "uncertainty_trigger_entropy_count": tec,
    }
    if not ok and logger is not None:
        try:
            logger.warning(
                "[UncertaintyAggregateSanity] failed ctx=%s reasons=%s detail=%s",
                ctx,
                reasons,
                detail,
            )
        except Exception:
            pass
    return ok, detail


def safe_summarize_uncertainty_decode_aggregate(
    agg: Mapping[str, Any],
    *,
    uncertainty_high_entropy_threshold: float = 1.0,
    logger: Any = None,
    ctx: str = "",
) -> Dict[str, Any]:
    """先 sanity，再 summarize；失败时丢弃熵值列表，仅保留计数类字段的安全汇总。"""
    ok, _ = sanity_check_uncertainty_aggregate(agg, logger=logger, ctx=ctx)
    if ok:
        return summarize_uncertainty_decode_aggregate(
            agg, uncertainty_high_entropy_threshold=uncertainty_high_entropy_threshold
        )
    td = max(0, int(agg.get("uncertainty_total_decision_count", 0)))
    tc = max(0, min(int(agg.get("uncertainty_trigger_count", 0)), td))
    fallback = {
        "uncertainty_total_decision_count": td,
        "uncertainty_trigger_count": tc,
        "uncertainty_trigger_entropy_sum": 0.0,
        "uncertainty_trigger_entropy_count": 0,
        "uncertainty_trigger_entropy_values": [],
        "uncertainty_first_trigger_steps": list(agg.get("uncertainty_first_trigger_steps") or []),
    }
    return summarize_uncertainty_decode_aggregate(
        fallback, uncertainty_high_entropy_threshold=uncertainty_high_entropy_threshold
    )


def mainline_selection_gate(
    current: Mapping[str, Any],
    best_prev: Optional[Mapping[str, Any]],
    *,
    dirty_relax: float = 0.04,
    rating_relax_ratio: float = 1.10,
) -> Tuple[bool, Dict[str, Any]]:
    """
    门控：相对上一轮 best，dirty 命中率与 RMSE/MAE 不得明显变差。
    无 best 时直接通过。
    """
    if best_prev is None:
        return True, {"reason": "no_previous_best", "passed": True}

    cur_d = float(current.get("dirty_hit_rate", 0.0))
    prev_d = float(best_prev.get("dirty_hit_rate", 0.0))
    ok_dirty = cur_d <= prev_d + dirty_relax

    cur_rmse = float(current.get("rmse_rating", 0.0))
    prev_rmse = float(best_prev.get("rmse_rating", 0.0))
    ok_rmse = prev_rmse <= 1e-8 or cur_rmse <= prev_rmse * rating_relax_ratio + 1e-8

    cur_mae = float(current.get("mae_rating", 0.0))
    prev_mae = float(best_prev.get("mae_rating", 0.0))
    ok_mae = prev_mae <= 1e-8 or cur_mae <= prev_mae * rating_relax_ratio + 1e-8

    passed = bool(ok_dirty and ok_rmse and ok_mae)
    detail = {
        "passed": passed,
        "ok_dirty": ok_dirty,
        "ok_rmse": ok_rmse,
        "ok_mae": ok_mae,
        "cur_dirty_hit_rate": cur_d,
        "prev_dirty_hit_rate": prev_d,
        "cur_rmse": cur_rmse,
        "prev_rmse": prev_rmse,
        "cur_mae": cur_mae,
        "prev_mae": prev_mae,
    }
    return passed, detail
