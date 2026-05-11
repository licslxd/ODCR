"""Step3 eval protocol, sample-integrity, damping, and effectiveness helpers."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

STEP3_EVAL_PROTOCOL_SCHEMA_VERSION = "odcr_step3_eval_protocol/2"
STEP3_PREDICTION_SHARD_SCHEMA_VERSION = "odcr_step3_prediction_shard/1"
STEP3_SAMPLE_INTEGRITY_SCHEMA_VERSION = "odcr_step3_sample_integrity/1"
STEP3_BATCH_INVARIANCE_SCHEMA_VERSION = "odcr_step3_eval_batch_invariance/1"
STEP3_TRAINING_EFFECTIVENESS_SCHEMA_VERSION = "odcr_step3_training_effectiveness/1"
STEP3_LOSS_DASHBOARD_SCHEMA_VERSION = "odcr_step3_loss_component_dashboard/1"
STEP3_PAPER_METRIC_IMPLEMENTATION_VERSION = "code1_paper_metrics_no_bertscore/1"

MINIMAL_EVAL = "minimal_eval"
ODCR_STEP3_DIAGNOSTIC = "odcr_step3_diagnostic"
PAPER_TARGET_ONLY_EVAL = "paper_target_only_eval"
FULL_PIPELINE_FINAL_EVAL = "full_pipeline_final_eval"

FORMAL_PAPER_METRICS = (
    "mae",
    "rmse",
    "rouge_1",
    "rouge_l",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "dist_1",
    "dist_2",
    "meteor",
)

PREDICTION_SHARD_REQUIRED_FIELDS = (
    "sample_id",
    "row_id",
    "split",
    "domain",
    "user_id",
    "item_id",
    "rating_gold",
    "rating_pred",
    "pred_text",
    "ref_text",
    "decode_status",
    "source_row_index",
    "rank",
)


def normalize_eval_protocol(raw: str | None) -> str:
    value = str(raw or MINIMAL_EVAL).strip().lower().replace("-", "_")
    aliases = {
        "minimal": MINIMAL_EVAL,
        "minimal_eval": MINIMAL_EVAL,
        "diagnostic": ODCR_STEP3_DIAGNOSTIC,
        "odcr_diagnostic": ODCR_STEP3_DIAGNOSTIC,
        "odcr_step3_diagnostic": ODCR_STEP3_DIAGNOSTIC,
        "paper": PAPER_TARGET_ONLY_EVAL,
        "paper_target_only": PAPER_TARGET_ONLY_EVAL,
        "paper_target_only_eval": PAPER_TARGET_ONLY_EVAL,
        "full_pipeline_final": FULL_PIPELINE_FINAL_EVAL,
        "full_pipeline_final_eval": FULL_PIPELINE_FINAL_EVAL,
    }
    if value not in aliases:
        raise ValueError(f"unknown Step3 eval protocol: {raw!r}")
    return aliases[value]


def step3_eval_protocol_spec(
    protocol: str | None,
    *,
    split: str = "valid",
    diagnostic_text_len: int = 48,
    paper_text_len: int = 25,
) -> dict[str, Any]:
    name = normalize_eval_protocol(protocol)
    split_name = str(split or "valid").strip().lower()
    if split_name not in {"valid", "test"}:
        raise ValueError(f"Step3 eval split must be valid or test, got {split!r}")
    base = {
        "schema_version": STEP3_EVAL_PROTOCOL_SCHEMA_VERSION,
        "protocol": name,
        "split": split_name,
        "metric_implementation_version": STEP3_PAPER_METRIC_IMPLEMENTATION_VERSION,
        "formal_metrics": list(FORMAL_PAPER_METRICS),
        "bertscore_enabled": False,
        "bert_score_enabled": False,
    }
    if name == MINIMAL_EVAL:
        base.update(
            {
                "paper_comparable": False,
                "diagnostic_only": False,
                "not_paper_comparable": True,
                "data_protocol": "target_only_if_available_rating_sanity",
                "target_only": True,
                "max_ref_len": None,
                "max_decode_len": None,
                "compute_text_metrics": False,
                "write_samples": False,
                "post_train_default": True,
            }
        )
    elif name == ODCR_STEP3_DIAGNOSTIC:
        base.update(
            {
                "paper_comparable": False,
                "diagnostic_only": True,
                "not_paper_comparable": True,
                "data_protocol": "merged_auxiliary_target",
                "target_only": False,
                "max_ref_len": int(diagnostic_text_len),
                "max_decode_len": int(diagnostic_text_len),
                "text_length_protocol": int(diagnostic_text_len),
                "compute_text_metrics": True,
                "write_samples": True,
            }
        )
    elif name == PAPER_TARGET_ONLY_EVAL:
        base.update(
            {
                "paper_comparable": True,
                "diagnostic_only": False,
                "not_paper_comparable": False,
                "data_protocol": "target_only",
                "target_only": True,
                "max_ref_len": int(paper_text_len),
                "max_decode_len": int(paper_text_len),
                "text_length_protocol": int(paper_text_len),
                "compute_text_metrics": True,
                "write_samples": True,
                "valid_test_supported": True,
            }
        )
    else:
        base.update(
            {
                "paper_comparable": True,
                "diagnostic_only": False,
                "not_paper_comparable": False,
                "data_protocol": "after_step4_step5_final_pipeline",
                "target_only": True,
                "max_ref_len": int(paper_text_len),
                "max_decode_len": int(paper_text_len),
                "text_length_protocol": int(paper_text_len),
                "compute_text_metrics": True,
                "write_samples": True,
                "interface_only": True,
            }
        )
    return base


def stable_step3_sample_id(
    *,
    dataset_name: str,
    split: str,
    source_row_index: int,
    user_id: Any = None,
    item_id: Any = None,
    existing_sample_id: Any = None,
    review_id: Any = None,
) -> str:
    if review_id not in (None, ""):
        return f"{dataset_name}:{split}:review:{review_id}"
    if existing_sample_id not in (None, "") and not str(existing_sample_id).isdigit():
        return f"{dataset_name}:{split}:sample:{existing_sample_id}"
    basis = "|".join(
        [
            str(dataset_name),
            str(split),
            str(int(source_row_index)),
            str(user_id if user_id is not None else ""),
            str(item_id if item_id is not None else ""),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return f"{dataset_name}:{split}:row:{int(source_row_index)}:{digest}"


def validate_prediction_shard_row(row: Mapping[str, Any]) -> list[str]:
    missing = [field for field in PREDICTION_SHARD_REQUIRED_FIELDS if field not in row]
    problems = list(missing)
    if "sample_id" in row and not str(row.get("sample_id") or "").strip():
        problems.append("sample_id_empty")
    if "decode_status" in row and not str(row.get("decode_status") or "").strip():
        problems.append("decode_status_empty")
    return problems


def sort_prediction_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in rows), key=lambda row: str(row.get("sample_id") or ""))


def sample_integrity_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
    expected_sample_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    ids = [str(row.get("sample_id") or "") for row in rows]
    counts: dict[str, int] = defaultdict(int)
    for sid in ids:
        counts[sid] += 1
    duplicates = sorted(sid for sid, count in counts.items() if count > 1)
    empty_ids = [idx for idx, sid in enumerate(ids) if not sid]
    expected = None if expected_sample_ids is None else {str(item) for item in expected_sample_ids}
    actual = set(ids)
    missing = sorted(expected - actual) if expected is not None else []
    unexpected = sorted(actual - expected) if expected is not None else []
    bad_rows = [
        {"index": idx, "problems": validate_prediction_shard_row(row)}
        for idx, row in enumerate(rows)
        if validate_prediction_shard_row(row)
    ]
    count_match = expected_count is None or int(expected_count) == len(rows)
    status = "PASS" if count_match and not duplicates and not empty_ids and not missing and not unexpected and not bad_rows else "FAIL"
    return {
        "schema_version": STEP3_SAMPLE_INTEGRITY_SCHEMA_VERSION,
        "status": status,
        "sample_count": len(rows),
        "expected_count": expected_count,
        "count_match": count_match,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates[:100],
        "empty_sample_id_count": len(empty_ids),
        "missing_count": len(missing),
        "missing_sample_ids": missing[:100],
        "unexpected_count": len(unexpected),
        "unexpected_sample_ids": unexpected[:100],
        "bad_row_count": len(bad_rows),
        "bad_rows": bad_rows[:100],
    }


def rating_metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    diffs = []
    for row in rows:
        diffs.append(float(row.get("rating_pred", 0.0)) - float(row.get("rating_gold", 0.0)))
    if not diffs:
        return {"mae": float("nan"), "rmse": float("nan")}
    mae = sum(abs(diff) for diff in diffs) / len(diffs)
    rmse = math.sqrt(sum(diff * diff for diff in diffs) / len(diffs))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4)}


def metrics_from_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    compute_text_metrics: bool,
    text_metric_fn: Callable[[Sequence[str], Sequence[str]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    sorted_rows = sort_prediction_rows(rows)
    rec = rating_metrics_from_rows(sorted_rows)
    out: dict[str, Any] = {
        "metric_implementation_version": STEP3_PAPER_METRIC_IMPLEMENTATION_VERSION,
        "recommendation": {"mae": rec["mae"], "rmse": rec["rmse"]},
        "bertscore_enabled": False,
    }
    if compute_text_metrics:
        if text_metric_fn is None:
            raise ValueError("text_metric_fn is required when compute_text_metrics=true")
        predictions = [str(row.get("pred_text") or "") for row in sorted_rows]
        references = [str(row.get("ref_text") or "") for row in sorted_rows]
        out.update({"explanation": dict(text_metric_fn(predictions, references))})
    else:
        out.update({"explanation": {"text_metrics_skipped": True}})
    return out


def compare_eval_batch_outputs(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_metrics: Mapping[str, Any] | None = None,
    candidate_metrics: Mapping[str, Any] | None = None,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    base = sort_prediction_rows(baseline_rows)
    cand = sort_prediction_rows(candidate_rows)
    base_ids = [str(row.get("sample_id") or "") for row in base]
    cand_ids = [str(row.get("sample_id") or "") for row in cand]
    missing = sorted(set(base_ids) - set(cand_ids))
    unexpected = sorted(set(cand_ids) - set(base_ids))
    pair_diffs: list[str] = []
    cand_by_id = {str(row.get("sample_id") or ""): row for row in cand}
    for row in base:
        sid = str(row.get("sample_id") or "")
        other = cand_by_id.get(sid)
        if other is None:
            continue
        for key in ("rating_gold", "rating_pred", "pred_text", "ref_text"):
            if str(row.get(key)) != str(other.get(key)):
                pair_diffs.append(sid)
                break
        if len(pair_diffs) >= 100:
            break
    metric_diffs: dict[str, dict[str, Any]] = {}
    if baseline_metrics is not None and candidate_metrics is not None:
        flat_base = _flatten_numbers(baseline_metrics)
        flat_cand = _flatten_numbers(candidate_metrics)
        for key in sorted(set(flat_base) | set(flat_cand)):
            b = flat_base.get(key)
            c = flat_cand.get(key)
            if b is None or c is None or abs(float(b) - float(c)) > tolerance:
                metric_diffs[key] = {"baseline": b, "candidate": c}
    status = "PASS" if base_ids == cand_ids and not pair_diffs and not metric_diffs else "FAIL"
    return {
        "schema_version": STEP3_BATCH_INVARIANCE_SCHEMA_VERSION,
        "status": status,
        "baseline_count": len(base),
        "candidate_count": len(cand),
        "sample_count_identical": len(base) == len(cand),
        "sample_id_set_identical": set(base_ids) == set(cand_ids),
        "sample_id_order_identical_after_sort": base_ids == cand_ids,
        "missing_sample_ids": missing[:100],
        "unexpected_sample_ids": unexpected[:100],
        "prediction_pair_diff_count": len(pair_diffs),
        "prediction_pair_diff_sample_ids": pair_diffs[:100],
        "metric_diff_count": len(metric_diffs),
        "metric_diffs": metric_diffs,
        "tolerance": float(tolerance),
    }


def _flatten_numbers(obj: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_numbers(value, path))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[path] = float(value)
    return out


def scheduler_semantics(
    *,
    scheduler_type: str,
    damping_enabled: bool,
    base_min_lr: float,
    damping_factor_cumulative: float = 1.0,
    effective_min_lr_policy: str = "base_floor",
) -> dict[str, Any]:
    stype = str(scheduler_type or "").strip().lower()
    if stype == "warmup_cosine" and damping_enabled:
        raise ValueError("hidden damping is forbidden: warmup_cosine requires damping_enabled=false")
    if stype == "safe_damping_v2" and not damping_enabled:
        raise ValueError("safe_damping_v2 requires damping_enabled=true")
    if stype == "warmup_cosine_with_damping":
        raise ValueError("warmup_cosine_with_damping is retired from formal Step3; use safe_damping_v2 probe-only.")
    if stype not in {"warmup_cosine", "safe_damping_v2"}:
        raise ValueError(f"unsupported Step3 scheduler_type: {scheduler_type!r}")
    factor = max(0.0, float(damping_factor_cumulative))
    effective_min = float(base_min_lr) if not damping_enabled else float(base_min_lr) * factor
    return {
        "scheduler_type": stype,
        "base_scheduler": "warmup_cosine",
        "damping_enabled": bool(damping_enabled),
        "base_min_lr": float(base_min_lr),
        "damping_factor_cumulative": factor,
        "effective_min_lr": effective_min,
        "effective_min_lr_policy": str(effective_min_lr_policy),
    }


def explain_lr_floor(
    *,
    current_lr: float,
    base_min_lr: float,
    scheduler_type: str,
    damping_enabled: bool,
    effective_min_lr: float | None = None,
) -> dict[str, Any]:
    below = float(current_lr) < float(base_min_lr) - 1.0e-15
    explained = (not below) or (
        str(scheduler_type) in {"safe_damping_v2", "warmup_cosine_with_damping"}
        and bool(damping_enabled)
        and effective_min_lr is not None
        and float(current_lr) >= float(effective_min_lr) - 1.0e-15
    )
    return {
        "current_lr": float(current_lr),
        "base_min_lr": float(base_min_lr),
        "effective_min_lr": None if effective_min_lr is None else float(effective_min_lr),
        "below_base_min_lr": below,
        "floor_explained": explained,
    }


def build_training_effectiveness_record(
    *,
    epoch: int,
    valid_loss: float,
    best_valid_loss: float,
    previous_valid_loss: float | None,
    lr_base: float,
    lr_effective: float,
    base_min_lr: float,
    effective_min_lr: float,
    damping_event: Mapping[str, Any] | None,
    checkpoint_improved: bool,
    grad_finite: bool = True,
    paper_eval_proxy: float | None = None,
    explanation_proxy: float | None = None,
) -> dict[str, Any]:
    delta_from_best = float(valid_loss) - float(best_valid_loss)
    delta_recent = 0.0 if previous_valid_loss is None else float(valid_loss) - float(previous_valid_loss)
    rel_gap = delta_from_best / max(abs(float(best_valid_loss)), 1.0e-12)
    low_lr = float(lr_effective) <= max(float(effective_min_lr) * 1.25, 1.0e-12)
    if bool(checkpoint_improved):
        status = "effective_improvement"
        action = "continue"
    elif low_lr and delta_recent >= -1.0e-3:
        status = "low_lr_no_progress"
        action = "stop_and_select_candidate"
    elif rel_gap <= 0.005 or abs(delta_recent) <= 1.0e-3:
        status = "marginal_improvement"
        action = "run_paper_target_only_eval"
    else:
        status = "no_meaningful_improvement"
        action = "review_loss_rebalance"
    reasons = []
    if low_lr:
        reasons.append("lr_too_low")
    if delta_recent >= -1.0e-3:
        reasons.append("validation_plateau")
    if paper_eval_proxy is None:
        reasons.append("need_protocol_eval")
    if not grad_finite:
        reasons.append("grad_optimization_abnormal")
    return {
        "schema_version": STEP3_TRAINING_EFFECTIVENESS_SCHEMA_VERSION,
        "epoch": int(epoch),
        "valid_loss": float(valid_loss),
        "best_valid_loss": float(best_valid_loss),
        "delta_from_best": float(delta_from_best),
        "delta_recent": float(delta_recent),
        "lr_base": float(lr_base),
        "lr_effective": float(lr_effective),
        "base_min_lr": float(base_min_lr),
        "effective_min_lr": float(effective_min_lr),
        "damping_event": dict(damping_event or {}),
        "checkpoint_improved": bool(checkpoint_improved),
        "paper_eval_proxy": paper_eval_proxy,
        "explanation_proxy": explanation_proxy,
        "grad_finite": bool(grad_finite),
        "effective_improvement_status": status,
        "reasons": reasons,
        "recommended_action": action,
        "action_gate": action,
    }


def summarize_loss_component_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_epoch: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        try:
            epoch = int(row.get("epoch"))
        except Exception:
            continue
        name = str(row.get("loss_name") or "")
        if not name:
            continue
        by_epoch[epoch][name].append(row)
    epoch_rows: list[dict[str, Any]] = []
    component_trends: dict[str, dict[str, Any]] = {}
    component_epochs: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for epoch in sorted(by_epoch):
        for name, items in sorted(by_epoch[epoch].items()):
            raw_values = [float(item.get("raw_value", 0.0) or 0.0) for item in items]
            weighted_values = [float(item.get("weighted_value", 0.0) or 0.0) for item in items]
            raw_mean = sum(raw_values) / len(raw_values)
            weighted_mean = sum(weighted_values) / len(weighted_values)
            component_epochs[name].append((epoch, raw_mean, weighted_mean))
            epoch_rows.append(
                {
                    "schema_version": STEP3_LOSS_DASHBOARD_SCHEMA_VERSION,
                    "epoch": epoch,
                    "loss_name": name,
                    "raw_mean": raw_mean,
                    "weighted_mean": weighted_mean,
                    "count": len(items),
                }
            )
    for name, values in sorted(component_epochs.items()):
        first_epoch, first_raw, first_weighted = values[0]
        last_epoch, last_raw, last_weighted = values[-1]
        saturation_epoch = last_epoch
        for epoch, raw, _weighted in values:
            if abs(raw - last_raw) <= max(abs(first_raw) * 0.01, 1.0e-6):
                saturation_epoch = epoch
                break
        component_trends[name] = {
            "first_epoch": first_epoch,
            "last_epoch": last_epoch,
            "raw_first": first_raw,
            "raw_last": last_raw,
            "raw_delta": last_raw - first_raw,
            "weighted_first": first_weighted,
            "weighted_last": last_weighted,
            "weighted_delta": last_weighted - first_weighted,
            "saturation_epoch": saturation_epoch,
        }
    return {
        "schema_version": STEP3_LOSS_DASHBOARD_SCHEMA_VERSION,
        "epoch_rows": epoch_rows,
        "component_trends": component_trends,
    }


@dataclass(frozen=True)
class EvalBatchProbe:
    batch_size: int
    status: str
    oom: bool = False
    invariance_status: str = "NOT_RUN"
    throughput: float | None = None


def select_largest_safe_eval_batch(probes: Sequence[EvalBatchProbe | Mapping[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for probe in probes:
        if isinstance(probe, EvalBatchProbe):
            item = {
                "batch_size": probe.batch_size,
                "status": probe.status,
                "oom": probe.oom,
                "invariance_status": probe.invariance_status,
                "throughput": probe.throughput,
            }
        else:
            item = dict(probe)
        normalized.append(item)
    safe = [
        item
        for item in normalized
        if str(item.get("status")) == "PASS"
        and not bool(item.get("oom", False))
        and str(item.get("invariance_status", "PASS")) in {"PASS", "BASELINE"}
    ]
    selected = max((int(item["batch_size"]) for item in safe), default=None)
    return {
        "schema_version": STEP3_BATCH_INVARIANCE_SCHEMA_VERSION,
        "selected_eval_batch": selected,
        "invariance_status": "PASS" if selected is not None else "FAIL",
        "probes": normalized,
    }
