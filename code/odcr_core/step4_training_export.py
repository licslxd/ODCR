"""
Step4 最终训练表：三类来源（aux_gold / target_gold / aux_cf）+ 清洗与质量字段 + manifest。
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from data_contract import CANONICAL_PREPROCESS_ASSET_COLUMNS, PREPROCESS_DERIVED_DEFAULTS
from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_EXPORT_SCHEMA_VERSION,
    STEP4_RCR_FIELD_DEFINITIONS,
    STEP4_RCR_REQUIRED_COLUMNS,
    STEP4_RCR_SCORE_COLUMNS,
    STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION,
    refresh_index_contract_train_csv_fingerprint,
    remap_step4_train_df_to_global_columns,
    step4_prior_boundary_contract,
    step4_rcr_required_fields_hash,
    write_index_contract,
)
from odcr_core.odcr_cf_routing import ODCFRoutingConfig
from odcr_core.text_cleaning import (
    build_sample_quality_flags,
    build_template_stats,
    clean_explanation_text,
    merge_flags_into_row,
)

DEFAULT_ORIGIN_WEIGHTS: Dict[str, float] = {
    "target_gold": 1.0,
    "aux_gold": 0.9,
    "aux_cf": 0.5,
}

ODCR_EXPORT_DEFAULTS: Dict[str, Any] = {
    **PREPROCESS_DERIVED_DEFAULTS,
    "route_scorer": 1,
    "route_explainer": 1,
    "route_reason_scorer": "gold_default",
    "route_reason_explainer": "gold_default",
    "cf_reliability_score": 1.0,
    "content_retention_score": 1.0,
    "style_shift_score": 0.0,
    "rating_stability_score": 1.0,
    "entropy_score": 0.0,
    "uncertainty_score": 0.0,
    "text_quality_score": 1.0,
    "confidence_bucket": 1,
    "preprocess_route_scorer_prior": 0,
    "preprocess_route_explainer_prior": 0,
}

ODCR_ANCHOR_FIELDS: tuple[str, ...] = (
    "content_anchor_score",
    "style_anchor_score",
    *CANONICAL_PREPROCESS_ASSET_COLUMNS,
)

ODCR_ROUTING_RELIABILITY_FIELDS: tuple[str, ...] = (
    *STEP4_RCR_REQUIRED_COLUMNS,
)

ODCR_REQUIRED_CF_POSTERIOR_FIELDS: tuple[str, ...] = (
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "cf_reliability_score",
    "uncertainty_score",
    "text_quality_score",
    "confidence_bucket",
    "route_scorer",
    "route_explainer",
    "route_reason_scorer",
    "route_reason_explainer",
)


def _unit_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(max(0.0, min(1.0, v)))


def _posterior_uncertainty(row: Mapping[str, Any]) -> float:
    return _posterior_uncertainty_with_config(row, ODCFRoutingConfig.for_test_default())


def _posterior_uncertainty_with_config(row: Mapping[str, Any], conf: ODCFRoutingConfig) -> float:
    rs = _unit_float(row.get("rating_stability_score"), 1.0)
    cr = _unit_float(row.get("content_retention_score"), 1.0)
    tq = _unit_float(row.get("text_quality_score"), 1.0)
    ent = _unit_float(row.get("entropy_score"), 0.0)
    w = conf.uncertainty_weights
    return _unit_float(
        float(w["rating_instability"]) * (1.0 - rs)
        + float(w["content_weakness"]) * (1.0 - cr)
        + float(w["text_quality_weakness"]) * (1.0 - tq)
        + float(w["entropy"]) * ent
    )


def _confidence_bucket(row: Mapping[str, Any]) -> int:
    return _confidence_bucket_with_config(row, ODCFRoutingConfig.for_test_default())


def _confidence_bucket_with_config(row: Mapping[str, Any], conf: ODCFRoutingConfig) -> int:
    rel = _unit_float(row.get("cf_reliability_score"), 1.0)
    unc = _unit_float(row.get("uncertainty_score"), 0.0)
    rs = _unit_float(row.get("rating_stability_score"), 1.0)
    cr = _unit_float(row.get("content_retention_score"), 1.0)
    high = conf.confidence_bucket["high"]
    medium = conf.confidence_bucket["medium"]
    if (
        rel >= float(high["min_reliability"])
        and unc <= float(high["max_uncertainty"])
        and rs >= float(high["min_rating_stability"])
        and cr >= float(high["min_content_retention"])
    ):
        return int(high["bucket"])
    if (
        rel >= float(medium["min_reliability"])
        and unc <= float(medium["max_uncertainty"])
        and cr >= float(medium["min_content_retention"])
    ):
        return int(medium["bucket"])
    return int(conf.confidence_bucket.get("low_bucket", 0))


def _preserve_preprocess_route_priors(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if "preprocess_route_scorer_prior" not in out or pd.isna(out.get("preprocess_route_scorer_prior")):
        out["preprocess_route_scorer_prior"] = 0
    if "preprocess_route_explainer_prior" not in out or pd.isna(out.get("preprocess_route_explainer_prior")):
        out["preprocess_route_explainer_prior"] = 0
    return out


def _fill_odcr_export_defaults(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for k, default in ODCR_EXPORT_DEFAULTS.items():
        if k not in out or out[k] is None:
            out[k] = default
    return out


def assemble_step4_training_table(
    train_df: pd.DataFrame,
    filtered_cf_df: pd.DataFrame,
    *,
    origin_weights: Mapping[str, float] | None = None,
    rcr_config: ODCFRoutingConfig | None = None,
    template_min_count: int = 3,
    template_hard_drop_min_count: int = 0,
) -> pd.DataFrame:
    """
    train_df: aug_train 中有 explanation 的子集（与 step4_engine 一致）。
    filtered_cf_df: 已附加 RCR posterior 字段的反事实行（domain 已为 auxiliary，explanation 为生成）。
    """
    ow = dict(origin_weights or DEFAULT_ORIGIN_WEIGHTS)
    if rcr_config is None:
        raise RuntimeError("assemble_step4_training_table requires resolved step4.rcr config from One-Control.")
    conf = rcr_config
    required_cf = tuple(conf.export.get("required_fields") or ODCR_REQUIRED_CF_POSTERIOR_FIELDS)

    aux_gold = train_df[train_df["domain"] == "auxiliary"].copy()
    target_gold = train_df[train_df["domain"] == "target"].copy()
    aux_cf = filtered_cf_df.copy()
    if len(aux_cf) > 0:
        cf_defaultable = {"preprocess_route_scorer_prior", "preprocess_route_explainer_prior"}
        missing_cf = [c for c in required_cf if c not in aux_cf.columns and c not in cf_defaultable]
        if missing_cf:
            raise ValueError(
                "Step4 CF export requires RCR posterior fields before table assembly: "
                + ", ".join(missing_cf)
                + ". Refusing to fall back to entropy/text-only routing."
            )

    aux_gold["_export_role"] = "aux_gold"
    target_gold["_export_role"] = "target_gold"
    aux_cf["_export_role"] = "aux_cf"

    combined = pd.concat([target_gold, aux_gold, aux_cf], ignore_index=True)

    # 第一轮：清洗，准备模板统计
    cleans: List[str] = []
    clean_results = []
    raws: List[str] = []
    for _, row in combined.iterrows():
        raw = row.get("explanation", "")
        raws.append(str(raw) if raw is not None else "")
        cr = clean_explanation_text(raw)
        clean_results.append(cr)
        cleans.append(cr.clean_text)

    tmpl_stats = build_template_stats(cleans)

    rows_out: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(combined.iterrows()):
        d = _preserve_preprocess_route_priors(row.to_dict())
        d = _fill_odcr_export_defaults(d)
        d["template_hard_drop_min_count"] = int(template_hard_drop_min_count)
        role = str(d.pop("_export_role", ""))
        is_cf = 1 if role == "aux_cf" else 0
        if is_cf == 0:
            d["route_scorer"] = 1
            d["route_explainer"] = 1
            d["route_reason_scorer"] = "factual_gold_scorer"
            d["route_reason_explainer"] = "factual_gold_explainer"
            d["cf_reliability_score"] = 1.0
            d["content_retention_score"] = 1.0
            d["style_shift_score"] = 0.0
            d["rating_stability_score"] = 1.0
            d["entropy_score"] = 0.0
            d["uncertainty_score"] = 0.0
            d["text_quality_score"] = 1.0
            d["confidence_bucket"] = 2
        cr = clean_results[i]
        flags = build_sample_quality_flags(
            raw_explanation=raws[i],
            clean_result=cr,
            template_stats=tmpl_stats,
            template_min_count=template_min_count,
        )
        merge_flags_into_row(d, flags, sample_origin=role, is_cf=is_cf, origin_weights=ow)
        d["cf_reliability_score"] = _unit_float(d.get("cf_reliability_score"), 1.0)
        d["content_retention_score"] = _unit_float(d.get("content_retention_score"), 1.0)
        d["style_shift_score"] = _unit_float(d.get("style_shift_score"), 0.0)
        d["rating_stability_score"] = _unit_float(d.get("rating_stability_score"), 1.0)
        d["entropy_score"] = _unit_float(d.get("entropy_score"), 0.0)
        d["text_quality_score"] = _unit_float(d.get("text_quality_score"), 1.0)
        d["uncertainty_score"] = _posterior_uncertainty_with_config(d, conf) if is_cf else 0.0
        d["confidence_bucket"] = _confidence_bucket_with_config(d, conf) if is_cf else 2
        if (
            is_cf
            and bool(conf.train_keep.get("reject_when_both_routes_zero", True))
            and int(d.get("route_scorer", 0)) == 0
            and int(d.get("route_explainer", 0)) == 0
        ):
            d["train_keep"] = 0
            d["train_drop_reason"] = str(conf.train_keep.get("reject_reason", "rcr_route_reject"))
            d["sample_weight_hint"] = 0.0
        elif is_cf and int(d.get("train_keep", 1)) == 1:
            sw_cfg = conf.sample_weight_hint
            route_mult = (
                float(sw_cfg["scorer_route_multiplier"])
                if int(d.get("route_scorer", 0)) == 1
                else float(sw_cfg["explainer_only_route_multiplier"])
            )
            posterior_mult = float(sw_cfg["reliability_floor"]) + float(sw_cfg["reliability_scale"]) * float(d["cf_reliability_score"])
            uncertainty_mult = float(sw_cfg["uncertainty_base"]) + float(sw_cfg["uncertainty_scale"]) * (1.0 - float(d["uncertainty_score"]))
            d["sample_weight_hint"] = round(
                float(d.get("sample_weight_hint", 0.0)) * route_mult * posterior_mult * uncertainty_mult,
                6,
            )
        rows_out.append(d)

    out_df = pd.DataFrame(rows_out)
    out_df["sample_id"] = np.arange(len(out_df), dtype=np.int64)
    return out_df


def validate_odcr_export_schema(df: pd.DataFrame, *, csv_name: str) -> None:
    missing_anchor = [c for c in ODCR_ANCHOR_FIELDS if c not in df.columns]
    missing_routing = [c for c in ODCR_ROUTING_RELIABILITY_FIELDS if c not in df.columns]
    missing = missing_anchor + missing_routing
    if missing:
        raise ValueError(
            f"{csv_name} 缺少 ODCR 必需字段: {', '.join(missing)}。"
            "请确认 Step4 export 输入和 defaults 未被旧逻辑覆盖。"
        )


def build_step4_train_manifest(
    df: pd.DataFrame,
    *,
    n_cf_candidate_input: int,
    n_cf_rcr_scorer_kept: int,
    origin_weights: Mapping[str, float] | None = None,
    rcr_config: ODCFRoutingConfig | None = None,
    index_contract_path: str | None = None,
    index_contract_summary: Mapping[str, Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """rank0 写入 JSON；供 runs/ 下审计与外部脚本读取。"""
    ow = dict(origin_weights or DEFAULT_ORIGIN_WEIGHTS)
    if rcr_config is None:
        raise RuntimeError("build_step4_train_manifest requires resolved step4.rcr config from One-Control.")
    conf = rcr_config
    manifest: Dict[str, Any] = {
        "schema_version": "odcr_step4_train_table/1.2",
        "export_schema": {
            "schema_version": STEP4_RCR_EXPORT_SCHEMA_VERSION,
            "posterior_contract_version": STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION,
            "required_columns": list(STEP4_RCR_REQUIRED_COLUMNS),
            "required_fields_hash": step4_rcr_required_fields_hash(),
            "score_columns": list(STEP4_RCR_SCORE_COLUMNS),
            "field_definitions": dict(STEP4_RCR_FIELD_DEFINITIONS),
        },
        "origin_weights": ow,
        "rcr_routing": {
            "n_target_rows_for_cf": int(n_cf_candidate_input),
            "n_cf_scorer_route": int(n_cf_rcr_scorer_kept),
            "n_cf_scorer_rejected": int(max(0, n_cf_candidate_input - n_cf_rcr_scorer_kept)),
            "entropy_score_role": "auxiliary_generation_uncertainty_not_primary_gate",
            "prior_posterior_boundary": step4_prior_boundary_contract(),
            "one_control_config": conf.to_dict(),
        },
        "row_counts": {
            "total_rows": int(len(df)),
            "by_sample_origin": {},
        },
        "cleaning": {
            "n_clean_changed": int(df["clean_changed"].sum()) if "clean_changed" in df.columns else 0,
            "n_nonempty_clean_text": int((df["clean_text"].fillna("").astype(str).str.strip() != "").sum()),
        },
        "quality_flag_counts": {},
        "template_hit_audit": {
            "template_hit_total": 0,
            "template_hit_kept": 0,
            "template_hit_downweighted": 0,
            "template_hit_dropped": 0,
        },
        "train_keep": {
            "n_keep_1": int((df["train_keep"] == 1).sum()) if "train_keep" in df.columns else 0,
            "n_keep_0": int((df["train_keep"] == 0).sum()) if "train_keep" in df.columns else 0,
        },
        "drop_reason_counts": {},
    }
    if index_contract_path:
        manifest["index_contract_path"] = str(index_contract_path)
        manifest["index_contract_file"] = INDEX_CONTRACT_FILENAME
    if index_contract_summary:
        manifest["index_contract_summary"] = dict(index_contract_summary)
    if lineage:
        manifest["step4_export_lineage"] = dict(lineage)

    if "sample_origin" in df.columns:
        vc = df["sample_origin"].value_counts()
        manifest["row_counts"]["by_sample_origin"] = {str(k): int(v) for k, v in vc.items()}

    for col in (
        "html_entity_hit",
        "bad_tail_hit",
        "template_hit",
        "short_fragment_hit",
        "repeat_tail_hit",
        "clean_changed",
        "route_scorer",
        "route_explainer",
    ):
        if col in df.columns:
            if col in ("route_scorer", "route_explainer"):
                manifest["quality_flag_counts"][col] = int((df[col].astype(int) == 1).sum())
            else:
                manifest["quality_flag_counts"][col] = int(df[col].sum())

    for col in (
        "cf_reliability_score",
        "content_retention_score",
        "style_shift_score",
        "rating_stability_score",
        "uncertainty_score",
        "text_quality_score",
        "evidence_quality_prior",
    ):
        if col in df.columns:
            manifest.setdefault("routing_stats", {})[col] = {
                "mean": float(df[col].astype(float).mean()),
                "min": float(df[col].astype(float).min()),
                "max": float(df[col].astype(float).max()),
            }

    if "train_drop_reason" in df.columns:
        dr = df[df["train_keep"] == 0]["train_drop_reason"].fillna("").astype(str)
        manifest["drop_reason_counts"] = dict(Counter([x for x in dr if x]))
    if "template_hit" in df.columns:
        tmask = df["template_hit"] == 1
        manifest["template_hit_audit"]["template_hit_total"] = int(tmask.sum())
        if "train_keep" in df.columns:
            manifest["template_hit_audit"]["template_hit_kept"] = int(
                ((df["template_hit"] == 1) & (df["train_keep"] == 1)).sum()
            )
            manifest["template_hit_audit"]["template_hit_dropped"] = int(
                ((df["template_hit"] == 1) & (df["train_keep"] == 0)).sum()
            )
        if "template_downweighted" in df.columns:
            manifest["template_hit_audit"]["template_hit_downweighted"] = int(
                ((df["template_hit"] == 1) & (df["template_downweighted"] == 1)).sum()
            )

    return manifest


def write_step4_training_artifacts(
    df: pd.DataFrame,
    manifest: Dict[str, Any],
    out_dir: str,
    *,
    csv_name: str = ODCR_ROUTING_TRAIN_CSV,
    manifest_name: str = "step4_train_table_manifest.json",
    index_contract: Mapping[str, Any] | None = None,
) -> tuple[str, str, str | None]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)
    man_path = os.path.join(out_dir, manifest_name)
    export_df = remap_step4_train_df_to_global_columns(df)
    validate_odcr_export_schema(export_df, csv_name=csv_name)
    export_df.to_csv(csv_path, index=False, encoding="utf-8")
    contract_path: str | None = None
    if index_contract is not None:
        refreshed_contract = refresh_index_contract_train_csv_fingerprint(index_contract, csv_path)
        contract_path = write_index_contract(refreshed_contract, os.path.join(out_dir, INDEX_CONTRACT_FILENAME))
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return csv_path, man_path, contract_path
