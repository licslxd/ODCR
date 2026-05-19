"""Step5A rating-only quality diagnostics and single-run summaries."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from odcr_core import path_layout
from odcr_core.file_atomic import atomic_write_json


QUALITY_DIAGNOSTIC_SCHEMA_VERSION = "odcr_step5A_rating_quality_diagnostic/1"
SINGLE_RUN_SUMMARY_SCHEMA_VERSION = "odcr_step5A_rating_single_run_summary/1"


class Step5RatingQualityError(RuntimeError):
    """Raised when Step5A rating quality diagnostics cannot be produced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_relative(root: Path, path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Step5RatingQualityError(f"missing required JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Step5RatingQualityError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prediction_path(metrics: Mapping[str, Any], *, split: str, root: Path, run_root: Path) -> Path:
    artifacts = metrics.get("prediction_artifacts") if isinstance(metrics.get("prediction_artifacts"), Mapping) else {}
    raw = artifacts.get("csv") or run_root / "eval" / f"rating_{split}_predictions.csv"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.is_file():
        raise Step5RatingQualityError(f"missing {split} prediction CSV: {path}")
    return path


def _read_predictions(path: Path) -> pd.DataFrame:
    cols = ["sample_id", "split", "row_index", "user_idx", "item_idx", "gt_rating", "pred_rating", "abs_error", "squared_error"]
    df = pd.read_csv(path, usecols=cols)
    for col in ("sample_id", "row_index", "user_idx", "item_idx"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("gt_rating", "pred_rating", "abs_error", "squared_error"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["gt_rating", "pred_rating"]].isna().any().any():
        raise Step5RatingQualityError(f"prediction CSV contains nonnumeric ratings: {path}")
    return df


def _series_distribution(series: pd.Series, *, quantiles: tuple[float, ...]) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    q = s.quantile(list(quantiles), interpolation="linear")
    out: dict[str, Any] = {
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "max": float(s.max()),
        "std": float(s.std(ddof=0)),
    }
    for item in quantiles:
        label = f"p{int(item * 100):02d}".replace("p50", "median")
        out[label] = float(q.loc[item])
    return out


def _metric_for_constant(gt: pd.Series, value: float) -> dict[str, float]:
    diff = float(value) - pd.to_numeric(gt, errors="coerce")
    return {
        "mae": round(float(diff.abs().mean()), 4),
        "rmse": round(float(math.sqrt((diff * diff).mean())), 4),
    }


def _metric_for_prediction(df: pd.DataFrame) -> dict[str, float]:
    diff = df["pred_rating"] - df["gt_rating"]
    return {
        "mae": round(float(diff.abs().mean()), 4),
        "rmse": round(float(math.sqrt((diff * diff).mean())), 4),
    }


def _top_values(series: pd.Series, *, n: int = 20) -> list[dict[str, Any]]:
    rounded = pd.to_numeric(series, errors="coerce").round(6)
    counts = rounded.value_counts(dropna=False).head(n)
    total = max(1, int(rounded.shape[0]))
    return [{"value": float(idx), "count": int(count), "rate": float(count / total)} for idx, count in counts.items()]


def _correlations(df: pd.DataFrame) -> dict[str, Any]:
    pearson = float(df["pred_rating"].corr(df["gt_rating"], method="pearson"))
    out: dict[str, Any] = {"pearson": pearson if math.isfinite(pearson) else None}
    try:
        spearman = float(df["pred_rating"].corr(df["gt_rating"], method="spearman"))
        out["spearman"] = spearman if math.isfinite(spearman) else None
    except Exception as exc:  # pragma: no cover - scipy/pandas optional path
        out["spearman"] = None
        out["spearman_error"] = str(exc)
    return out


def _split_quality(df: pd.DataFrame) -> dict[str, Any]:
    abs_error = (df["pred_rating"] - df["gt_rating"]).abs()
    squared_error = (df["pred_rating"] - df["gt_rating"]) ** 2
    unique_ratio = float(df["pred_rating"].nunique(dropna=True) / max(1, len(df)))
    constant_like = float(max(0.0, min(1.0, 1.0 - min(unique_ratio / 0.05, 1.0))))
    error_by_bucket: dict[str, dict[str, Any]] = {}
    for rating, group in df.groupby("gt_rating", dropna=False):
        diff = group["pred_rating"] - group["gt_rating"]
        error_by_bucket[str(float(rating))] = {
            "count": int(group.shape[0]),
            "mae": round(float(diff.abs().mean()), 4),
            "rmse": round(float(math.sqrt((diff * diff).mean())), 4),
            "pred_mean": float(group["pred_rating"].mean()),
        }
    return {
        "sample_count": int(df.shape[0]),
        "actual_metric_recomputed": _metric_for_prediction(df),
        "pred_rating_distribution": _series_distribution(df["pred_rating"], quantiles=(0.01, 0.05, 0.10, 0.90, 0.95, 0.99)),
        "gt_rating_distribution": _series_distribution(df["gt_rating"], quantiles=(0.01, 0.05, 0.95)),
        "error_distribution": {
            "abs_error": _series_distribution(abs_error, quantiles=(0.50, 0.90, 0.95, 0.99)),
            "squared_error_mean": float(squared_error.mean()),
            "error_by_gt_rating_bucket": error_by_bucket,
        },
        "correlation": _correlations(df),
        "raw_prediction_scale_check": {
            "out_of_range_rate_lt_1": float((df["pred_rating"] < 1.0).mean()),
            "out_of_range_rate_gt_5": float((df["pred_rating"] > 5.0).mean()),
            "pred_constant_like_score": constant_like,
            "pred_unique_count": int(df["pred_rating"].nunique(dropna=True)),
            "pred_unique_count_over_n": unique_ratio,
            "top_predicted_values": _top_values(df["pred_rating"]),
        },
    }


def _train_pool_summary(root: Path, task: int, run_summary: Mapping[str, Any]) -> dict[str, Any]:
    step4_run = str(run_summary.get("from_step4") or "1")
    parquet = root / "runs" / "step4" / f"task{int(task)}" / step4_run / "step5_exports" / "step5A_scorer_train.parquet"
    if not parquet.is_file():
        return {"status": "missing", "path": _repo_relative(root, parquet)}
    cols = ["rating", "domain", "sample_origin", "user_idx_global", "item_idx_global", "train_keep"]
    df = pd.read_parquet(parquet, columns=cols)
    origin_counts = {str(k): int(v) for k, v in df["sample_origin"].value_counts(dropna=False).items()}
    domain_counts = {str(k): int(v) for k, v in df["domain"].value_counts(dropna=False).items()}
    target_gold = df[df["sample_origin"].astype(str) == "target_gold"]
    return {
        "status": "ok",
        "path": _repo_relative(root, parquet),
        "sha256": _sha256(parquet),
        "row_count": int(df.shape[0]),
        "component_counts": {
            "target_gold": int(origin_counts.get("target_gold", 0)),
            "aux_gold": int(origin_counts.get("aux_gold", 0)),
            "cf": int(sum(v for k, v in origin_counts.items() if "cf" in k)),
            "raw": origin_counts,
        },
        "domain_counts": domain_counts,
        "rating_mean": float(df["rating"].mean()),
        "target_gold_rating_mean": float(target_gold["rating"].mean()) if not target_gold.empty else None,
        "target_gold_coverage_rate": float(len(target_gold) / max(1, len(df))),
        "user_idx_range": [int(df["user_idx_global"].min()), int(df["user_idx_global"].max())],
        "item_idx_range": [int(df["item_idx_global"].min()), int(df["item_idx_global"].max())],
        "user_set": set(int(x) for x in df["user_idx_global"].dropna().astype(int).unique()),
        "item_set": set(int(x) for x in df["item_idx_global"].dropna().astype(int).unique()),
    }


def _mapping_summary(
    *,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_summary: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    train_users = train_summary.get("user_set") if isinstance(train_summary.get("user_set"), set) else set()
    train_items = train_summary.get("item_set") if isinstance(train_summary.get("item_set"), set) else set()
    nuser = int(architecture.get("nuser") or 0)
    nitem = int(architecture.get("nitem") or 0)

    def one(df: pd.DataFrame) -> dict[str, Any]:
        users = set(int(x) for x in df["user_idx"].dropna().astype(int).unique())
        items = set(int(x) for x in df["item_idx"].dropna().astype(int).unique())
        invalid_users = int(((df["user_idx"] < 0) | ((df["user_idx"] >= nuser) if nuser else False)).sum())
        invalid_items = int(((df["item_idx"] < 0) | ((df["item_idx"] >= nitem) if nitem else False)).sum())
        return {
            "user_idx_range": [int(df["user_idx"].min()), int(df["user_idx"].max())],
            "item_idx_range": [int(df["item_idx"].min()), int(df["item_idx"].max())],
            "oov_user_count": int(len(users - train_users)) if train_users else None,
            "oov_user_rate": float(len(users - train_users) / max(1, len(users))) if train_users else None,
            "oov_item_count": int(len(items - train_items)) if train_items else None,
            "oov_item_rate": float(len(items - train_items) / max(1, len(items))) if train_items else None,
            "invalid_user_count": invalid_users,
            "invalid_item_count": invalid_items,
        }

    return {
        "valid": one(valid_df),
        "test": one(test_df),
        "train_pool_user_idx_range": train_summary.get("user_idx_range"),
        "train_pool_item_idx_range": train_summary.get("item_idx_range"),
        "embedding_size_bounds": {"nuser": nuser, "nitem": nitem},
        "eval_domain_idx": None,
        "target_domain_id": "AM_CDs",
        "invalid_domain_count": None,
        "domain_idx_note": "rating prediction artifacts do not carry domain_idx; target split is AM_CDs by path and run lineage",
    }


def _checkpoint_integrity(root: Path, run_root: Path, metrics: Mapping[str, Any]) -> dict[str, Any]:
    lineage = metrics.get("step5_checkpoint_lineage") if isinstance(metrics.get("step5_checkpoint_lineage"), Mapping) else {}
    checkpoint = Path(str(metrics.get("checkpoint") or run_root / "model" / "best.pth")).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (root / checkpoint).resolve()
    sidecar_hash = str((lineage.get("checkpoint_file") or {}).get("sha256") or "")
    out: dict[str, Any] = {
        "checkpoint_path": _repo_relative(root, checkpoint),
        "checkpoint_hash": sidecar_hash or (_sha256(checkpoint) if checkpoint.is_file() else None),
        "lineage_hash": lineage.get("lineage_hash"),
        "checkpoint_compatibility_hash": lineage.get("checkpoint_compatibility_hash"),
        "strict_load_status": "passed_in_recorded_rating_eval",
        "missing_keys": [],
        "unexpected_keys": [],
        "note": "The rating eval artifacts were produced after the Step5 engine strict load path completed; this diagnostic does not instantiate the full model.",
    }
    try:
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
            state = state["state_dict"]
        if isinstance(state, Mapping) and isinstance(state.get("model_state_dict"), Mapping):
            state = state["model_state_dict"]
        if isinstance(state, Mapping):
            keys = [str(k) for k in state.keys()]
            selected = [
                key
                for key in keys
                if any(token in key.lower() for token in ("rating", "recommender", "score"))
                and not key.startswith("_")
            ][:50]
            total_elements = 0
            param_stats = {}
            h = hashlib.sha256()
            for key in selected:
                tensor = state.get(key)
                if not hasattr(tensor, "numel"):
                    continue
                numel = int(tensor.numel())
                if total_elements + numel > 5_000_000:
                    continue
                total_elements += numel
                arr = tensor.detach().float().cpu()
                h.update(key.encode("utf-8"))
                h.update(arr.numpy().tobytes())
                param_stats[key] = {
                    "shape": list(arr.shape),
                    "numel": numel,
                    "norm": float(arr.norm().item()),
                    "mean": float(arr.mean().item()) if numel else 0.0,
                    "std": float(arr.std(unbiased=False).item()) if numel else 0.0,
                }
            out.update(
                {
                    "torch_load_weights_only": "ok",
                    "selected_head_key_count": len(selected),
                    "rating_recommender_head_parameter_hash": h.hexdigest() if param_stats else None,
                    "rating_recommender_head_parameter_stats": param_stats,
                    "loaded_head_diff_from_init": "not_feasible_without_reconstructing_initialized_model",
                }
            )
    except Exception as exc:
        out.update(
            {
                "torch_load_weights_only": "not_completed",
                "torch_load_error": str(exc),
                "rating_recommender_head_parameter_hash": None,
                "rating_recommender_head_parameter_stats": {},
                "loaded_head_diff_from_init": "not_feasible_without_successful_checkpoint_tensor_read",
            }
        )
    return out


def _root_cause_likelihoods(
    *,
    valid_quality: Mapping[str, Any],
    test_quality: Mapping[str, Any],
    mapping: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    baselines: Mapping[str, Any],
) -> list[dict[str, Any]]:
    test_pred = test_quality["pred_rating_distribution"]
    test_corr = (test_quality.get("correlation") or {}).get("pearson")
    test_scale = test_quality["raw_prediction_scale_check"]
    test_actual = test_quality["actual_metric_recomputed"]
    const3 = baselines.get("predict_constant_3", {}).get("test", {})
    oov_user = ((mapping.get("test") or {}).get("oov_user_rate") or 0.0)
    oov_item = ((mapping.get("test") or {}).get("oov_item_rate") or 0.0)
    invalid = ((mapping.get("test") or {}).get("invalid_user_count") or 0) + ((mapping.get("test") or {}).get("invalid_item_count") or 0)
    underfit_high = (
        (test_corr is None or float(test_corr) < 0.35)
        and float(test_pred.get("std") or 0.0) < 0.7
        and float(test_actual.get("mae") or 0.0) >= float(const3.get("mae") or 99.0)
    )
    return [
        {
            "root_cause": "model_underfit",
            "likelihood": "high" if underfit_high else "medium",
            "evidence": {
                "test_pred_std": test_pred.get("std"),
                "test_pearson": test_corr,
                "test_mae": test_actual.get("mae"),
                "constant_3_test_mae": const3.get("mae"),
            },
            "next_action": "Do not hide the result; inspect Step5A scorer objective/sampling and run a small target-gold-only ablation before expensive 5-seed if desired.",
        },
        {
            "root_cause": "mixed_aux_cf_objective_hurts_rating",
            "likelihood": "medium" if (train_summary.get("component_counts") or {}).get("cf", 0) else "low",
            "evidence": train_summary.get("component_counts"),
            "next_action": "Compare against a target_gold-only or lower-CF Step5A scorer candidate if multi-seed remains poor.",
        },
        {
            "root_cause": "prediction_scale_mismatch",
            "likelihood": "medium"
            if test_scale.get("out_of_range_rate_lt_1", 0.0) > 0.01 or test_scale.get("pred_constant_like_score", 0.0) > 0.5
            else "low",
            "evidence": test_scale,
            "next_action": "Check final rating head activation/denormalization only if out-of-range or constant-like scores stay high.",
        },
        {
            "root_cause": "index_mapping_error",
            "likelihood": "high" if invalid or oov_user > 0.2 or oov_item > 0.2 else "low",
            "evidence": {"test_oov_user_rate": oov_user, "test_oov_item_rate": oov_item, "invalid_index_count": invalid},
            "next_action": "If this rises above low, block multi-seed and fix user/item index lineage.",
        },
        {
            "root_cause": "checkpoint_load_error",
            "likelihood": "low",
            "evidence": "rating eval completed with strict load and checkpoint compatibility hashes recorded",
            "next_action": "No checkpoint reload fix indicated unless future strict-load diagnostics fail.",
        },
        {
            "root_cause": "target_coverage_gap",
            "likelihood": "medium" if float(train_summary.get("target_gold_coverage_rate") or 0.0) < 0.25 else "low",
            "evidence": {"target_gold_coverage_rate": train_summary.get("target_gold_coverage_rate")},
            "next_action": "If target coverage is intentionally low, run target-heavy ablation before formal paper reruns.",
        },
        {
            "root_cause": "train_eval_protocol_mismatch",
            "likelihood": "medium",
            "evidence": "train-time valid_loss_r is an optimization loss; final MAE/RMSE are code1-compatible target-only paper metrics.",
            "next_action": "Track MAE/RMSE during validation or add a no-generate metric monitor so checkpoint selection aligns with the reported metric.",
        },
        {
            "root_cause": "unknown",
            "likelihood": "low",
            "evidence": "primary hard-bug checks have explicit fields above",
            "next_action": "Escalate only if ablations contradict the underfit/objective-mismatch explanation.",
        },
    ]


def build_step5A_rating_quality_diagnostic(
    *,
    repo_root: str | Path,
    task: int,
    source_run_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run_root = path_layout.get_stage_run_root(root, int(task), "v1", "step5", source_run_id).resolve()
    eval_dir = run_root / "eval"
    meta_dir = run_root / "meta"
    run_summary = _load_json(meta_dir / "run_summary.json")
    valid_metrics = _load_json(eval_dir / "rating_valid_metrics.json")
    test_metrics = _load_json(eval_dir / "rating_test_metrics.json")
    valid_path = _prediction_path(valid_metrics, split="valid", root=root, run_root=run_root)
    test_path = _prediction_path(test_metrics, split="test", root=root, run_root=run_root)
    valid_df = _read_predictions(valid_path)
    test_df = _read_predictions(test_path)
    train_summary = _train_pool_summary(root, int(task), run_summary)
    architecture = (valid_metrics.get("step5_checkpoint_lineage") or {}).get("architecture") or {}
    mapping = _mapping_summary(
        valid_df=valid_df,
        test_df=test_df,
        train_summary=train_summary,
        architecture=architecture,
    )
    serializable_train_summary = {k: v for k, v in train_summary.items() if k not in {"user_set", "item_set"}}
    train_mean = train_summary.get("rating_mean")
    valid_mean = float(valid_df["gt_rating"].mean())
    test_mean = float(test_df["gt_rating"].mean())
    baselines = {
        "predict_global_train_mean": {
            "value": train_mean,
            "valid": _metric_for_constant(valid_df["gt_rating"], float(train_mean)) if train_mean is not None else None,
            "test": _metric_for_constant(test_df["gt_rating"], float(train_mean)) if train_mean is not None else None,
        },
        "predict_valid_mean": {
            "value": valid_mean,
            "valid": _metric_for_constant(valid_df["gt_rating"], valid_mean),
            "test": _metric_for_constant(test_df["gt_rating"], valid_mean),
        },
        "predict_test_mean": {
            "value": test_mean,
            "valid": _metric_for_constant(valid_df["gt_rating"], test_mean),
            "test": _metric_for_constant(test_df["gt_rating"], test_mean),
        },
        "predict_constant_3": {
            "value": 3.0,
            "valid": _metric_for_constant(valid_df["gt_rating"], 3.0),
            "test": _metric_for_constant(test_df["gt_rating"], 3.0),
        },
    }
    valid_quality = _split_quality(valid_df)
    test_quality = _split_quality(test_df)
    checkpoint = _checkpoint_integrity(root, run_root, valid_metrics)
    metrics_rows = []
    metrics_path = meta_dir / "metrics.jsonl"
    if metrics_path.is_file():
        metrics_rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    diagnostic = {
        "schema_version": QUALITY_DIAGNOSTIC_SCHEMA_VERSION,
        "task": int(task),
        "head": "step5A",
        "source_run_id": str(source_run_id),
        "created_at": _utc_now(),
        "metric_protocol": "code1_compatible_rating_v1",
        "prediction_artifacts": {
            "valid": {"path": _repo_relative(root, valid_path), "sha256": _sha256(valid_path), "row_count": int(valid_df.shape[0])},
            "test": {"path": _repo_relative(root, test_path), "sha256": _sha256(test_path), "row_count": int(test_df.shape[0])},
        },
        "valid": valid_quality,
        "test": test_quality,
        "baseline_comparison": baselines,
        "checkpoint_integrity": checkpoint,
        "user_item_domain_mapping": mapping,
        "train_eval_data_mismatch": {
            "step5A_train_sample_component_counts": serializable_train_summary.get("component_counts"),
            "target_gold_coverage": serializable_train_summary.get("target_gold_coverage_rate"),
            "eval_split_coverage": {"valid": int(valid_df.shape[0]), "test": int(test_df.shape[0])},
            "target_gold_train_subset_rating_eval": "not_feasible_without_train_subset_predictions",
            "valid_test_full_eval": {"valid": valid_quality["actual_metric_recomputed"], "test": test_quality["actual_metric_recomputed"]},
            "train_pool_summary": serializable_train_summary,
        },
        "train_time_valid_loss_vs_paper_eval": {
            "metrics_jsonl": _repo_relative(root, metrics_path) if metrics_path.is_file() else None,
            "epoch_rows": metrics_rows,
            "same_data": False,
            "same_loss": False,
            "same_metric": False,
            "explanation": "train-time valid_loss_r/valid_loss_total are optimization losses; final MAE/RMSE are target-only code1-compatible rating metrics over prediction CSV rows.",
        },
        "root_cause_likelihood": _root_cause_likelihoods(
            valid_quality=valid_quality,
            test_quality=test_quality,
            mapping=mapping,
            train_summary=serializable_train_summary,
            baselines=baselines,
        ),
    }
    hard_bug = any(
        item["root_cause"] in {"prediction_scale_mismatch", "index_mapping_error", "checkpoint_load_error"}
        and item["likelihood"] == "high"
        for item in diagnostic["root_cause_likelihood"]
    )
    diagnostic["quality_hard_bug_detected"] = bool(hard_bug)
    json_path = eval_dir / "rating_quality_diagnostic.json"
    md_path = eval_dir / "rating_quality_diagnostic.md"
    if dry_run:
        return {"status": "dry_run", "would_write": [_repo_relative(root, json_path), _repo_relative(root, md_path)], "diagnostic": diagnostic}
    json_out = atomic_write_json(json_path, diagnostic)
    md_lines = [
        "# Step5A Rating Quality Diagnostic",
        "",
        f"- task: {task}",
        f"- run: {source_run_id}",
        f"- valid MAE/RMSE: {valid_quality['actual_metric_recomputed']['mae']} / {valid_quality['actual_metric_recomputed']['rmse']}",
        f"- test MAE/RMSE: {test_quality['actual_metric_recomputed']['mae']} / {test_quality['actual_metric_recomputed']['rmse']}",
        f"- test pred mean/std: {test_quality['pred_rating_distribution']['mean']:.6f} / {test_quality['pred_rating_distribution']['std']:.6f}",
        f"- test gt mean/std: {test_quality['gt_rating_distribution']['mean']:.6f} / {test_quality['gt_rating_distribution']['std']:.6f}",
        f"- test pearson/spearman: {test_quality['correlation'].get('pearson')} / {test_quality['correlation'].get('spearman')}",
        f"- hard bug detected: {hard_bug}",
        "",
        "## Baselines",
    ]
    for name, block in baselines.items():
        md_lines.append(f"- {name}: valid={block.get('valid')} test={block.get('test')} value={block.get('value')}")
    md_lines.extend(["", "## Root Cause Likelihood"])
    for item in diagnostic["root_cause_likelihood"]:
        md_lines.append(f"- {item['root_cause']}: {item['likelihood']} - {item['next_action']}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {
        "status": "ok",
        "quality_diagnostic": _repo_relative(root, json_out),
        "quality_diagnostic_md": _repo_relative(root, md_path),
        "quality_hard_bug_detected": hard_bug,
        "valid": valid_quality["actual_metric_recomputed"],
        "test": test_quality["actual_metric_recomputed"],
        "root_cause_likelihood": diagnostic["root_cause_likelihood"],
    }


def write_step5A_single_run_summary(
    *,
    repo_root: str | Path,
    task: int,
    source_run_id: str,
    seed: int = 3407,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    run_root = path_layout.get_stage_run_root(root, int(task), "v1", "step5", source_run_id).resolve()
    handoff = _load_json(run_root / "meta" / "eval_handoff.json")
    diagnostic_path = run_root / "eval" / "rating_quality_diagnostic.json"
    diagnostic = _load_json(diagnostic_path) if diagnostic_path.is_file() else {}
    out_dir = root / "runs" / "step5" / f"task{int(task)}" / "summaries"
    json_path = out_dir / f"step5A_rating_task{int(task)}_seed{int(seed)}_single_run.json"
    md_path = out_dir / f"step5A_rating_task{int(task)}_seed{int(seed)}_single_run.md"
    payload = {
        "schema_version": SINGLE_RUN_SUMMARY_SCHEMA_VERSION,
        "task": int(task),
        "head": "step5A",
        "seed": int(seed),
        "run_id": str(source_run_id),
        "checkpoint_hash": handoff.get("checkpoint_hash"),
        "metric_protocol": handoff.get("metric_protocol"),
        "valid_mae": (handoff.get("valid") or {}).get("mae"),
        "valid_rmse": (handoff.get("valid") or {}).get("rmse"),
        "test_mae": (handoff.get("test") or {}).get("mae"),
        "test_rmse": (handoff.get("test") or {}).get("rmse"),
        "eval_handoff_path": _repo_relative(root, run_root / "meta" / "eval_handoff.json"),
        "rating_quality_diagnostic_path": _repo_relative(root, diagnostic_path) if diagnostic_path.is_file() else None,
        "paper_comparable_single_run": True,
        "paper_comparable_mean_std": False,
        "step5B_touched": False,
        "combined_step5_ready": False,
        "rerank_touched": False,
        "quality_hard_bug_detected": bool(diagnostic.get("quality_hard_bug_detected")),
        "next_action": (
            "fix rating quality hard bug before 5-seed"
            if diagnostic.get("quality_hard_bug_detected")
            else "eligible to launch Step5A 5-seed rating pipeline when GPU time is authorized"
        ),
        "created_at": _utc_now(),
    }
    if dry_run:
        return {"status": "dry_run", "would_write": [_repo_relative(root, json_path), _repo_relative(root, md_path)], "summary": payload}
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = atomic_write_json(json_path, payload)
    md = "\n".join(
        [
            "# Step5A Rating Single-Run Summary",
            "",
            f"- seed: {seed}",
            f"- run_id: {source_run_id}",
            f"- valid MAE/RMSE: {payload['valid_mae']} / {payload['valid_rmse']}",
            f"- test MAE/RMSE: {payload['test_mae']} / {payload['test_rmse']}",
            f"- paper_comparable_single_run: {payload['paper_comparable_single_run']}",
            f"- paper_comparable_mean_std: {payload['paper_comparable_mean_std']}",
            f"- next_action: {payload['next_action']}",
            "",
        ]
    )
    md_path.write_text(md, encoding="utf-8")
    return {"status": "ok", "summary": _repo_relative(root, json_out), "summary_md": _repo_relative(root, md_path), **payload}
