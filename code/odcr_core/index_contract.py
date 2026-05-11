"""
Step4 → Step5 索引契约：单一真源、CPU 侧 fail-fast、禁止 local/global 混用列名。

Step4 导出 ``index_contract.json``（与 ``odcr_routing_train.csv`` 同目录）；Step5 仅通过该文件
与解析后的训练 CSV 路径定位契约，不再用 max(CSV)+1 推断全局嵌入大小。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import get_odcr_embed_dim
from odcr_core.training_checkpoint import CheckpointLineageError, file_fingerprint, stable_hash
from paths_config import DEFAULT_SENTENCE_EMBED_MODEL_ID, get_sentence_embed_model_dir

INDEX_CONTRACT_SCHEMA_VERSION = "odcr_index_contract/2.2"
INDEX_CONTRACT_FILENAME = "index_contract.json"
# Step4 → Step5 正式训练表文件名（唯一真源；禁止再默认 factuals_counterfactuals.csv）
ODCR_ROUTING_TRAIN_CSV = "odcr_routing_train.csv"
GLOBAL_COL_USER = "user_idx_global"
GLOBAL_COL_ITEM = "item_idx_global"

STEP4_RCR_EXPORT_SCHEMA_VERSION = "odcr_step4_rcr_export/1.0"
STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION = "odcr_step4_route_posterior/1.0"
STEP4_EXPORT_LINEAGE_SCHEMA_VERSION = "odcr_step4_export_lineage/4A"
STEP4_RCR_SCORE_COLUMNS = (
    "content_retention_score",
    "style_shift_score",
    "rating_stability_score",
    "cf_reliability_score",
    "uncertainty_score",
    "entropy_score",
    "text_quality_score",
    "confidence_bucket",
)
STEP4_RCR_DECISION_COLUMNS = (
    "train_keep",
    "sample_weight_hint",
    "route_scorer",
    "route_explainer",
    "route_reason_scorer",
    "route_reason_explainer",
)
STEP4_RCR_PRIOR_COLUMNS = (
    "preprocess_route_scorer_prior",
    "preprocess_route_explainer_prior",
)
STEP4_RCR_REQUIRED_COLUMNS = STEP4_RCR_DECISION_COLUMNS + STEP4_RCR_SCORE_COLUMNS + STEP4_RCR_PRIOR_COLUMNS

STEP4_RCR_FIELD_DEFINITIONS: Dict[str, str] = {
    "evidence_quality_prior": (
        "Preprocess-side prior only; Step4 preserves it and must not rewrite it "
        "as posterior reliability."
    ),
    "preprocess_route_scorer_prior": (
        "Preprocess-side scorer hint preserved for audit only; Step4 posterior "
        "route_scorer overwrites the active route decision without mixing prior and posterior."
    ),
    "preprocess_route_explainer_prior": (
        "Preprocess-side explainer hint preserved for audit only; Step4 posterior "
        "route_explainer overwrites the active route decision without mixing prior and posterior."
    ),
    "content_retention_score": (
        "Posterior RCR score for shared/content retention under counterfactual "
        "routing; primary source is latent shared similarity plus content evidence strength."
    ),
    "style_shift_score": (
        "Posterior RCR score for specific/style movement toward counterfactual style; "
        "primary source is specific latent shift plus style evidence strength."
    ),
    "rating_stability_score": (
        "Posterior RCR score for scorer invariance when the same shared user/item pair "
        "is evaluated under target vs auxiliary style/domain context."
    ),
    "cf_reliability_score": "Hybrid posterior reliability fusion used by Step4 routing.",
    "uncertainty_score": (
        "Posterior routing uncertainty for UCI metadata; distinct from both "
        "evidence_quality_prior and 1-cf_reliability."
    ),
    "entropy_score": (
        "Auxiliary generation uncertainty normalized from decoder entropy; never the "
        "primary RCR decision signal."
    ),
    "text_quality_score": "Auxiliary text hygiene score used to keep malformed generations out of both routes.",
    "confidence_bucket": (
        "Posterior confidence group derived from RCR reliability, uncertainty, "
        "retention, and rating stability."
    ),
    "route_scorer": (
        "Step4 posterior binary scorer-clean path decision; requires high rating stability, high content "
        "retention, low uncertainty, and clean text."
    ),
    "route_explainer": (
        "Step4 posterior binary explainer-rich path decision; admits high-reliability CFs plus "
        "medium-reliability CFs with meaningful style shift."
    ),
    "train_keep": "Final training row gate after text hygiene and RCR path decisions.",
    "sample_weight_hint": (
        "Final sample weight hint after origin, text hygiene, route class, and posterior "
        "reliability are combined."
    ),
}


class IndexContractError(ValueError):
    """索引契约违反：须在 CPU 数据准备阶段抛出，附可读上下文。"""

    pass


def step4_rcr_export_contract_summary() -> Dict[str, Any]:
    return {
        "schema_version": STEP4_RCR_EXPORT_SCHEMA_VERSION,
        "posterior_contract_version": STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION,
        "train_csv": ODCR_ROUTING_TRAIN_CSV,
        "required_columns": list(STEP4_RCR_REQUIRED_COLUMNS),
        "required_fields_hash": step4_rcr_required_fields_hash(),
        "score_columns": list(STEP4_RCR_SCORE_COLUMNS),
        "decision_columns": list(STEP4_RCR_DECISION_COLUMNS),
        "prior_columns": list(STEP4_RCR_PRIOR_COLUMNS),
        "field_definitions": dict(STEP4_RCR_FIELD_DEFINITIONS),
        "prior_posterior_boundary": {
            "prior_fields": [
                "evidence_quality_prior",
                "preprocess_route_scorer_prior",
                "preprocess_route_explainer_prior",
            ],
            "posterior_fields": [
                "content_retention_score",
                "style_shift_score",
                "rating_stability_score",
                "cf_reliability_score",
                "uncertainty_score",
                "confidence_bucket",
                "route_scorer",
                "route_explainer",
            ],
        },
    }


def step4_rcr_required_fields_hash() -> str:
    return stable_hash(
        {
            "schema_version": STEP4_RCR_EXPORT_SCHEMA_VERSION,
            "posterior_contract_version": STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION,
            "required_columns": list(STEP4_RCR_REQUIRED_COLUMNS),
            "field_definitions": dict(STEP4_RCR_FIELD_DEFINITIONS),
        }
    )


def step4_prior_boundary_contract() -> Dict[str, Any]:
    return {
        "preprocess_prior": [
            "evidence_quality_prior",
            "preprocess_route_scorer_prior",
            "preprocess_route_explainer_prior",
        ],
        "step4_posterior": [
            "content_retention_score",
            "style_shift_score",
            "rating_stability_score",
            "cf_reliability_score",
            "uncertainty_score",
            "confidence_bucket",
            "route_scorer",
            "route_explainer",
        ],
    }


def build_step4_export_lineage(
    *,
    task_id: int,
    auxiliary_domain: str,
    target_domain: str,
    step3_checkpoint_lineage_hash: str,
    step4_rcr_config: Mapping[str, Any],
    step4_run: str | None = None,
    frozen_step3_lineage: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    frozen = dict(
        frozen_step3_lineage
        or {
            "upstream_step3_run_id": "unknown",
            "step3_checkpoint_path": "unknown",
            "step3_checkpoint_hash": str(step3_checkpoint_lineage_hash),
            "step3_checkpoint_lineage_hash": str(step3_checkpoint_lineage_hash),
            "step3_stage_status_hash": "unknown",
            "step3_eval_handoff_hash": "unknown",
        }
    )
    payload: Dict[str, Any] = {
        "schema_version": STEP4_EXPORT_LINEAGE_SCHEMA_VERSION,
        "producer_stage": "step4",
        "step4_run": str(step4_run or "unknown"),
        "step3_checkpoint_lineage_hash": str(step3_checkpoint_lineage_hash),
        "frozen_step3_lineage": frozen,
        "step4_rcr_config_hash": stable_hash(dict(step4_rcr_config)),
        "step4_export_schema_version": STEP4_RCR_EXPORT_SCHEMA_VERSION,
        "rcr_required_fields_hash": step4_rcr_required_fields_hash(),
        "route_posterior_contract_version": STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION,
        "preprocess_route_prior_boundary": step4_prior_boundary_contract(),
        "task": {
            "task_id": int(task_id),
            "auxiliary_domain": str(auxiliary_domain),
            "target_domain": str(target_domain),
        },
    }
    payload["lineage_hash"] = stable_hash(payload)
    return payload


def validate_step4_export_lineage(
    contract: Mapping[str, Any],
    *,
    current_step4_rcr_config: Mapping[str, Any],
    task_id: int,
    auxiliary_domain: str,
    target_domain: str,
) -> Dict[str, Any]:
    lineage = contract.get("step4_export_lineage")
    if not isinstance(lineage, Mapping):
        raise CheckpointLineageError(
            "Step5 refused Step4 export: index_contract.json lacks step4_export_lineage. "
            "Rerun Step4 under Phase 4A lineage gates."
        )
    if lineage.get("schema_version") != STEP4_EXPORT_LINEAGE_SCHEMA_VERSION:
        raise CheckpointLineageError(
            f"Step4 export lineage schema mismatch: {lineage.get('schema_version')!r} "
            f"!= {STEP4_EXPORT_LINEAGE_SCHEMA_VERSION!r}."
        )
    actual_hash = stable_hash({k: v for k, v in dict(lineage).items() if k != "lineage_hash"})
    if lineage.get("lineage_hash") != actual_hash:
        raise CheckpointLineageError(
            f"Step4 export lineage hash mismatch: stored={lineage.get('lineage_hash')} computed={actual_hash}."
        )
    expected = build_step4_export_lineage(
        task_id=int(task_id),
        auxiliary_domain=str(auxiliary_domain),
        target_domain=str(target_domain),
        step3_checkpoint_lineage_hash=str(lineage.get("step3_checkpoint_lineage_hash") or ""),
        step4_rcr_config=current_step4_rcr_config,
        step4_run=str(lineage.get("step4_run") or "unknown"),
        frozen_step3_lineage=dict(lineage.get("frozen_step3_lineage") or {}),
    )
    frozen = lineage.get("frozen_step3_lineage")
    if not isinstance(frozen, Mapping):
        raise CheckpointLineageError("Step5 refused Step4 export: missing frozen Step3 lineage in Step4 export.")
    for required_key in (
        "upstream_step3_run_id",
        "step3_checkpoint_path",
        "step3_checkpoint_hash",
        "step3_stage_status_hash",
        "step3_eval_handoff_hash",
    ):
        if not str(frozen.get(required_key) or "").strip():
            raise CheckpointLineageError(
                f"Step5 refused Step4 export: frozen Step3 lineage missing {required_key}."
            )
    comparisons = {
        "producer_stage": "step4",
        "step4_rcr_config_hash": expected["step4_rcr_config_hash"],
        "step4_export_schema_version": STEP4_RCR_EXPORT_SCHEMA_VERSION,
        "rcr_required_fields_hash": step4_rcr_required_fields_hash(),
        "route_posterior_contract_version": STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION,
        "preprocess_route_prior_boundary": step4_prior_boundary_contract(),
        "frozen_step3_lineage": dict(frozen),
        "task": {
            "task_id": int(task_id),
            "auxiliary_domain": str(auxiliary_domain),
            "target_domain": str(target_domain),
        },
    }
    for key, expected_value in comparisons.items():
        if lineage.get(key) != expected_value:
            raise CheckpointLineageError(
                f"Step4 export lineage mismatch for {key}: export={lineage.get(key)!r} current={expected_value!r}"
            )
    return dict(lineage)


def _ctx_tail(ctx: Mapping[str, Any]) -> str:
    parts = [
        f"task_id={ctx.get('task_id')}",
        f"iter={ctx.get('iteration_id')}",
        f"step4_run={ctx.get('step4_run')}",
        f"step5_run={ctx.get('step5_run')}",
        f"contract_path={ctx.get('contract_path')}",
        f"csv_path={ctx.get('csv_path')}",
        f"profile_path={ctx.get('profile_path')}",
    ]
    return " | ".join(str(p) for p in parts if any(x is not None and str(x) != "None" for x in [p.split("=", 1)[1]]))


def parse_training_run_lineage(stage_run_dir: str) -> Dict[str, Any]:
    """
    从 ``runs/step4/task{T}/{slug}`` 新布局解析 task_id / slug。

    旧 ``runs/task{T}/{iter}/train/step4/{slug}`` 布局不再作为 active 解析兼容；
    看到旧路径必须 fail-fast 后重新经 One-Control 入口生成当前 lineage。
    """
    s = str(Path(stage_run_dir).resolve()).replace("\\", "/")
    m_new = re.search(r"/runs/(step\d+)/task(\d+)/([^/]+)/?$", s)
    if m_new:
        return {
            "task_id": int(m_new.group(2)),
            "iteration_id": "v1",
            "train_stage": m_new.group(1),
            "step4_run": m_new.group(3),
        }
    raise IndexContractError(
        "stage_run_dir must use new ODCR layout runs/<stage>/task<T>/<run_id>; "
        f"got {stage_run_dir!r}. Rerun the producing stage through ./odcr."
    )


def _domain_profile_paths(root: str, domain: str) -> Dict[str, str]:
    base = os.path.abspath(os.path.join(root, domain))
    return {
        "user_content": os.path.join(base, "user_content_profiles.npy"),
        "user_style": os.path.join(base, "user_style_profiles.npy"),
        "item_content": os.path.join(base, "item_content_profiles.npy"),
        "item_style": os.path.join(base, "item_style_profiles.npy"),
        "domain_content": os.path.join(base, "domain_content.npy"),
        "domain_style": os.path.join(base, "domain_style.npy"),
    }


def _dual_channel_required_paths(paths: Mapping[str, str]) -> List[str]:
    return [
        str(paths["user_content"]),
        str(paths["user_style"]),
        str(paths["item_content"]),
        str(paths["item_style"]),
        str(paths["domain_content"]),
        str(paths["domain_style"]),
    ]


def _sentence_embed_backbone_for_contract(*, embed_dim: int) -> Dict[str, Any]:
    """契约「数据/表征」语义：句向量 backbone（不含 Step5 训练 LoRA 等运行时）。"""
    return {
        "model_id": str(DEFAULT_SENTENCE_EMBED_MODEL_ID),
        "local_dir": os.path.abspath(get_sentence_embed_model_dir()),
        "family": "bge_large_en",
        "hidden_size": int(embed_dim),
        "dual_channel": True,
    }


def _fingerprint_for_path(path: str) -> Dict[str, Any]:
    abs_path = os.path.abspath(os.path.expanduser(path))
    try:
        fp = dict(file_fingerprint(path))
    except OSError:
        fp = {"schema_version": "odcr_file_fingerprint/1", "path": abs_path, "exists": False}
    fp.setdefault("path", abs_path)
    fp.setdefault("exists", False)
    fp.setdefault("size", None)
    fp.setdefault("mtime_ns", None)
    fp.setdefault("sha256", None)
    fp.setdefault("sample_sha256", None)
    fp["fingerprint_version"] = str(fp.get("schema_version") or "odcr_file_fingerprint/1")
    return fp


def _raise_if_missing_dual_files(required_paths: List[str]) -> None:
    missing = [p for p in required_paths if not os.path.isfile(p)]
    if missing:
        raise IndexContractError(
            "ODCR 需要完整的双通道 profile/domain 文件，以下路径缺失：\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def _np_to_f32_tensor(arr: np.ndarray, device_idx: int | str) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=torch.float32, device=device_idx)


def _load_one_domain_physical_tensors(
    paths: Mapping[str, str], device_idx: int | str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """单域：content/style 在内存中保持分离，禁止默认主链上的 content/style 算术融合。"""
    _raise_if_missing_dual_files(_dual_channel_required_paths(paths))
    user_content = _np_to_f32_tensor(np.load(paths["user_content"]), device_idx)
    user_style = _np_to_f32_tensor(np.load(paths["user_style"]), device_idx)
    item_content = _np_to_f32_tensor(np.load(paths["item_content"]), device_idx)
    item_style = _np_to_f32_tensor(np.load(paths["item_style"]), device_idx)
    domain_content = _np_to_f32_tensor(np.load(paths["domain_content"]), device_idx)
    domain_style = _np_to_f32_tensor(np.load(paths["domain_style"]), device_idx)
    return user_content, user_style, item_content, item_style, domain_content, domain_style


def _load_profile_tensors_physical_separate_from_paths(
    *,
    target_paths: Mapping[str, str],
    aux_paths: Mapping[str, str],
    device_idx: int | str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """aux 域行 0、target 域行 1；user/item 行序与旧版一致：target 段在前、aux 段在后。"""
    t_uc, t_us, t_ic, t_is, t_dc, t_ds = _load_one_domain_physical_tensors(target_paths, device_idx)
    s_uc, s_us, s_ic, s_is, s_dc, s_ds = _load_one_domain_physical_tensors(aux_paths, device_idx)
    domain_content = torch.stack([s_dc, t_dc], dim=0)
    domain_style = torch.stack([s_ds, t_ds], dim=0)
    user_content = torch.cat([t_uc, s_uc], dim=0)
    user_style = torch.cat([t_us, s_us], dim=0)
    item_content = torch.cat([t_ic, s_ic], dim=0)
    item_style = torch.cat([t_is, s_is], dim=0)
    meta = {
        "profile_mode": "odcr_dual_channel",
        "profile_consumption": "physical_separate",
        "target_user_count": int(t_uc.shape[0]),
        "aux_user_count": int(s_uc.shape[0]),
        "target_item_count": int(t_ic.shape[0]),
        "aux_item_count": int(s_ic.shape[0]),
    }
    return domain_content, domain_style, user_content, user_style, item_content, item_style, meta


def _compare_sample_fingerprint(actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str) -> None:
    for key in ("exists", "is_file", "size", "mtime_ns", "sample_sha256"):
        if key in expected and actual.get(key) != expected.get(key):
            raise IndexContractError(
                f"{context} fingerprint mismatch for {key}: actual={actual.get(key)!r}, expected={expected.get(key)!r}"
            )


def _require_step3_preflight_profile_paths(
    *,
    target_paths: Mapping[str, str],
    aux_paths: Mapping[str, str],
    target_domain: str,
    auxiliary_domain: str,
    preflight_summary: Mapping[str, Any] | None,
) -> None:
    if preflight_summary is None:
        return
    if preflight_summary.get("status") != "ok":
        raise IndexContractError("Step3 upstream preflight summary must have status=ok before profile tensor load.")
    profile_artifacts = preflight_summary.get("profile_artifacts")
    domain_artifacts = preflight_summary.get("domain_artifacts")
    if not isinstance(profile_artifacts, Mapping) or not isinstance(domain_artifacts, Mapping):
        raise IndexContractError("Step3 upstream preflight summary is missing profile/domain artifact contracts.")

    profile_key_map = {
        "user_content": "user_content_profiles",
        "user_style": "user_style_profiles",
        "item_content": "item_content_profiles",
        "item_style": "item_style_profiles",
    }
    domain_key_map = {
        "domain_content": "domain_content",
        "domain_style": "domain_style",
    }
    for domain, paths in ((target_domain, target_paths), (auxiliary_domain, aux_paths)):
        domain_profiles = profile_artifacts.get(domain)
        domain_domains = domain_artifacts.get(domain)
        if not isinstance(domain_profiles, Mapping) or not isinstance(domain_domains, Mapping):
            raise IndexContractError(f"Step3 upstream preflight summary missing artifacts for domain={domain}.")
        for loader_key, summary_key in profile_key_map.items():
            contract = domain_profiles.get(summary_key)
            if not isinstance(contract, Mapping):
                raise IndexContractError(f"Step3 upstream preflight missing profile contract {domain}:{summary_key}.")
            expected_path = os.path.abspath(os.path.expanduser(str(contract.get("path", ""))))
            actual_path = os.path.abspath(os.path.expanduser(str(paths[loader_key])))
            if actual_path != expected_path:
                raise IndexContractError(
                    f"Step3 profile loader path mismatch for {domain}:{summary_key}: {actual_path} != {expected_path}"
                )
            expected_fp = contract.get("fingerprint")
            if isinstance(expected_fp, Mapping):
                _compare_sample_fingerprint(
                    file_fingerprint(actual_path, sample_only=True),
                    expected_fp,
                    context=f"Step3 profile loader {domain}:{summary_key}",
                )
        for loader_key, summary_key in domain_key_map.items():
            contract = domain_domains.get(summary_key)
            if not isinstance(contract, Mapping):
                raise IndexContractError(f"Step3 upstream preflight missing domain contract {domain}:{summary_key}.")
            expected_path = os.path.abspath(os.path.expanduser(str(contract.get("path", ""))))
            actual_path = os.path.abspath(os.path.expanduser(str(paths[loader_key])))
            if actual_path != expected_path:
                raise IndexContractError(
                    f"Step3 domain loader path mismatch for {domain}:{summary_key}: {actual_path} != {expected_path}"
                )
            expected_fp = contract.get("fingerprint")
            if isinstance(expected_fp, Mapping):
                _compare_sample_fingerprint(
                    file_fingerprint(actual_path, sample_only=True),
                    expected_fp,
                    context=f"Step3 domain loader {domain}:{summary_key}",
                )


def load_profile_tensors_dual_first(
    *,
    data_root: str,
    auxiliary_domain: str,
    target_domain: str,
    device_idx: int | str,
    step3_upstream_preflight_summary: Mapping[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """从磁盘双资产加载为六路张量（默认主链物理分离，非融合单向量）。"""
    root = os.path.abspath(os.path.expanduser(data_root))
    target_paths = _domain_profile_paths(root, target_domain)
    aux_paths = _domain_profile_paths(root, auxiliary_domain)
    _require_step3_preflight_profile_paths(
        target_paths=target_paths,
        aux_paths=aux_paths,
        target_domain=target_domain,
        auxiliary_domain=auxiliary_domain,
        preflight_summary=step3_upstream_preflight_summary,
    )
    return _load_profile_tensors_physical_separate_from_paths(
        target_paths=target_paths,
        aux_paths=aux_paths,
        device_idx=device_idx,
    )


def load_profile_tensors_fused_average_legacy(
    *,
    data_root: str,
    auxiliary_domain: str,
    target_domain: str,
    device_idx: int | str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    **显式退役分支**：0.5*(content+style) 单向量，仅用于消融/对照；不得接入 Step4/5 默认主链。

    须同时设置环境变量 ``ODCR_PROFILE_CONSUME_FUSED_LEGACY=1`` 才允许调用（防误用）。
    """
    if os.environ.get("ODCR_PROFILE_CONSUME_FUSED_LEGACY", "").strip() != "1":
        raise IndexContractError(
            "load_profile_tensors_fused_average_legacy 为退役融合加载；设置 ODCR_PROFILE_CONSUME_FUSED_LEGACY=1 后方可调用。"
        )

    def _fused(paths: Mapping[str, str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _raise_if_missing_dual_files(_dual_channel_required_paths(paths))
        uc = np.load(paths["user_content"])
        us = np.load(paths["user_style"])
        ic = np.load(paths["item_content"])
        ist = np.load(paths["item_style"])
        ddc = np.load(paths["domain_content"])
        dds = np.load(paths["domain_style"])
        u = torch.tensor(0.5 * (uc + us), dtype=torch.float32, device=device_idx)
        it = torch.tensor(0.5 * (ic + ist), dtype=torch.float32, device=device_idx)
        dom = torch.tensor(0.5 * (ddc + dds), dtype=torch.float32, device=device_idx)
        return dom, u, it

    root = os.path.abspath(os.path.expanduser(data_root))
    target_paths = _domain_profile_paths(root, target_domain)
    aux_paths = _domain_profile_paths(root, auxiliary_domain)
    sdom, su, si = _fused(aux_paths)
    tdom, tu, ti = _fused(target_paths)
    domain_profiles = torch.stack([sdom, tdom], dim=0)
    user_profiles = torch.cat([tu, su], dim=0)
    item_profiles = torch.cat([ti, si], dim=0)
    meta = {
        "profile_mode": "odcr_dual_channel_fused_legacy",
        "profile_consumption": "fused_average_explicit_legacy",
        "target_user_count": int(tu.shape[0]),
        "aux_user_count": int(su.shape[0]),
        "target_item_count": int(ti.shape[0]),
        "aux_item_count": int(si.shape[0]),
    }
    return domain_profiles, user_profiles, item_profiles, meta


def build_index_contract(
    *,
    task_id: int,
    iteration_id: str,
    step4_run: str,
    auxiliary_domain: str,
    target_domain: str,
    data_root: str,
    train_csv_path: str,
    valid_csv_path: str,
    test_csv_path: str,
    target_user_count: int,
    aux_user_count: int,
    target_item_count: int,
    aux_item_count: int,
) -> Dict[str, Any]:
    """由 Step4 在写出 CSV 的真源（profile 行数）构造契约对象。"""
    nuser_global = int(target_user_count + aux_user_count)
    nitem_global = int(target_item_count + aux_item_count)
    root = os.path.abspath(os.path.expanduser(data_root))
    target_paths = _domain_profile_paths(root, target_domain)
    aux_paths = _domain_profile_paths(root, auxiliary_domain)
    dual_required = _dual_channel_required_paths(target_paths) + _dual_channel_required_paths(aux_paths)
    _raise_if_missing_dual_files(dual_required)

    _probe = np.load(target_paths["user_content"], mmap_mode="r")
    embed_dim = int(_probe.shape[-1])
    expected = int(get_odcr_embed_dim())
    if embed_dim != expected:
        raise IndexContractError(
            f"profile 向量维度={embed_dim} 与 ODCR_EMBED_DIM={expected} 不一致（探测文件: {target_paths['user_content']}）。"
            "请用当前句向量模型重跑 compute_embeddings.py / infer_domain_semantics.py，或调整 ODCR_EMBED_DIM。"
        )

    contract: Dict[str, Any] = {
        "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
        "task_id": int(task_id),
        "iteration_id": str(iteration_id),
        "step4_run": str(step4_run),
        "aux_domain": str(auxiliary_domain),
        "target_domain": str(target_domain),
        "nuser_global": nuser_global,
        "nitem_global": nitem_global,
        "embed_dim": embed_dim,
        "target_user_count": int(target_user_count),
        "aux_user_count": int(aux_user_count),
        "target_item_count": int(target_item_count),
        "aux_item_count": int(aux_item_count),
        "target_user_offset": 0,
        "aux_user_offset": int(target_user_count),
        "target_item_offset": 0,
        "aux_item_offset": int(target_item_count),
        "train_index_space": "global",
        "valid_index_space": "target_local",
        "test_index_space": "target_local",
        "train_csv_path": os.path.abspath(os.path.expanduser(train_csv_path)),
        "valid_csv_path": os.path.abspath(os.path.expanduser(valid_csv_path)),
        "test_csv_path": os.path.abspath(os.path.expanduser(test_csv_path)),
        "target_user_content_profiles_path": target_paths["user_content"],
        "target_user_style_profiles_path": target_paths["user_style"],
        "target_item_content_profiles_path": target_paths["item_content"],
        "target_item_style_profiles_path": target_paths["item_style"],
        "target_domain_content_path": target_paths["domain_content"],
        "target_domain_style_path": target_paths["domain_style"],
        "aux_user_content_profiles_path": aux_paths["user_content"],
        "aux_user_style_profiles_path": aux_paths["user_style"],
        "aux_item_content_profiles_path": aux_paths["item_content"],
        "aux_item_style_profiles_path": aux_paths["item_style"],
        "aux_domain_content_path": aux_paths["domain_content"],
        "aux_domain_style_path": aux_paths["domain_style"],
        "source_paths": {
            "dual_channel": {
                "target": {
                    "user_content_profiles": target_paths["user_content"],
                    "user_style_profiles": target_paths["user_style"],
                    "item_content_profiles": target_paths["item_content"],
                    "item_style_profiles": target_paths["item_style"],
                    "domain_content": target_paths["domain_content"],
                    "domain_style": target_paths["domain_style"],
                },
                "auxiliary": {
                    "user_content_profiles": aux_paths["user_content"],
                    "user_style_profiles": aux_paths["user_style"],
                    "item_content_profiles": aux_paths["item_content"],
                    "item_style_profiles": aux_paths["item_style"],
                    "domain_content": aux_paths["domain_content"],
                    "domain_style": aux_paths["domain_style"],
                },
            },
        },
        "profile_assets": {
            "kind": "odcr_dual_channel",
            "consumption": "physical_separate",
        },
        "step4_export_contract": step4_rcr_export_contract_summary(),
        "backbones": {
            "sentence_embed": _sentence_embed_backbone_for_contract(embed_dim=embed_dim),
        },
        "fingerprints": {
            "train_csv": _fingerprint_for_path(train_csv_path),
            "valid_csv": _fingerprint_for_path(valid_csv_path),
            "test_csv": _fingerprint_for_path(test_csv_path),
            "target_user_content_profiles": _fingerprint_for_path(target_paths["user_content"]),
            "target_user_style_profiles": _fingerprint_for_path(target_paths["user_style"]),
            "target_item_content_profiles": _fingerprint_for_path(target_paths["item_content"]),
            "target_item_style_profiles": _fingerprint_for_path(target_paths["item_style"]),
            "target_domain_content": _fingerprint_for_path(target_paths["domain_content"]),
            "target_domain_style": _fingerprint_for_path(target_paths["domain_style"]),
            "aux_user_content_profiles": _fingerprint_for_path(aux_paths["user_content"]),
            "aux_user_style_profiles": _fingerprint_for_path(aux_paths["user_style"]),
            "aux_item_content_profiles": _fingerprint_for_path(aux_paths["item_content"]),
            "aux_item_style_profiles": _fingerprint_for_path(aux_paths["item_style"]),
            "aux_domain_content": _fingerprint_for_path(aux_paths["domain_content"]),
            "aux_domain_style": _fingerprint_for_path(aux_paths["domain_style"]),
        },
    }
    return contract


def write_index_contract(contract: Mapping[str, Any], out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(contract), f, ensure_ascii=False, indent=2)
        f.write("\n")
    return out_path


def _validate_contract_backbones(raw: Mapping[str, Any], *, path: str) -> None:
    bb = raw.get("backbones")
    if not isinstance(bb, dict):
        raise IndexContractError(f"index_contract 缺少或非法 backbones 对象: path={path}")
    se = bb.get("sentence_embed")
    if not isinstance(se, dict):
        raise IndexContractError(f"index_contract.backbones 缺少 sentence_embed 对象: path={path}")
    for k in ("model_id", "local_dir", "family", "hidden_size", "dual_channel"):
        if k not in se:
            raise IndexContractError(
                f"index_contract.backbones.sentence_embed 缺少字段 {k!r}（须由 Step4 以 schema 2.2 重新导出）。path={path}"
            )
    try:
        hs = int(se["hidden_size"])
    except (TypeError, ValueError) as e:
        raise IndexContractError(
            f"index_contract.backbones.sentence_embed.hidden_size 非法: {se.get('hidden_size')!r} path={path}"
        ) from e
    try:
        ed = int(raw["embed_dim"])
    except (TypeError, ValueError) as e:
        raise IndexContractError(f"index_contract embed_dim 非法: {raw.get('embed_dim')!r} path={path}") from e
    if hs != ed:
        raise IndexContractError(
            f"index_contract embed_dim={ed} 与 backbones.sentence_embed.hidden_size={hs} 不一致；"
            f"须与当前句向量模型维度对齐。path={path}"
        )
    if not isinstance(se.get("dual_channel"), bool):
        raise IndexContractError(
            f"index_contract.backbones.sentence_embed.dual_channel 须为 bool，当前为 {se.get('dual_channel')!r} path={path}"
        )


def load_index_contract(path: str) -> Dict[str, Any]:
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"缺少 index_contract.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise IndexContractError(f"index_contract 根须为 object: {p}")
    if raw.get("schema_version") != INDEX_CONTRACT_SCHEMA_VERSION:
        raise IndexContractError(
            f"不支持的 index_contract schema（须 {INDEX_CONTRACT_SCHEMA_VERSION!r}）: {raw.get('schema_version')!r} path={p}"
        )
    if "embed_dim" not in raw:
        raise IndexContractError(
            f"index_contract 缺少 embed_dim 字段（须由当前 Step4 重新导出）。path={p}"
        )
    try:
        _ed = int(raw["embed_dim"])
    except (TypeError, ValueError) as e:
        raise IndexContractError(f"index_contract embed_dim 非法: {raw.get('embed_dim')!r} path={p}") from e
    if _ed <= 0:
        raise IndexContractError(f"index_contract embed_dim 须为正整数，当前为 {_ed} path={p}")
    _validate_contract_backbones(raw, path=p)
    return raw


def resolve_index_contract_path(train_csv_path: str) -> str:
    """训练 CSV 解析后的目录（随 symlink 指向 Step4 真目录）与契约文件同目录。"""
    real_dir = Path(train_csv_path).resolve().parent
    return str(real_dir / INDEX_CONTRACT_FILENAME)


def remap_step4_train_df_to_global_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step4 导出行：将 aug_train 约定的 ``user_idx``/``item_idx`` 改为全局列名（值不变，语义为 global）。
    """
    if GLOBAL_COL_USER in df.columns and GLOBAL_COL_ITEM in df.columns:
        out = df.drop(columns=[c for c in ("user_idx", "item_idx") if c in df.columns], errors="ignore")
        return out
    if "user_idx" not in df.columns or "item_idx" not in df.columns:
        raise IndexContractError(
            "Step4 导出 DataFrame 须含 user_idx/item_idx 或已含 user_idx_global/item_idx_global"
        )
    out = df.copy()
    out[GLOBAL_COL_USER] = out["user_idx"].astype(np.int64)
    out[GLOBAL_COL_ITEM] = out["item_idx"].astype(np.int64)
    out = out.drop(columns=["user_idx", "item_idx"])
    return out


def convert_target_local_to_global_user(local: np.ndarray | pd.Series, contract: Mapping[str, Any]) -> np.ndarray:
    off = int(contract["target_user_offset"])
    return np.asarray(local, dtype=np.int64) + off


def convert_target_local_to_global_item(local: np.ndarray | pd.Series, contract: Mapping[str, Any]) -> np.ndarray:
    off = int(contract["target_item_offset"])
    return np.asarray(local, dtype=np.int64) + off


def normalize_split_indices_to_global(
    df: pd.DataFrame,
    contract: Mapping[str, Any],
    split_name: str,
    *,
    ctx: Mapping[str, Any],
) -> pd.DataFrame:
    key = f"{split_name}_index_space"
    if key not in contract:
        raise IndexContractError(f"契约缺少字段 {key!r} ({_ctx_tail(ctx)})")
    space = str(contract[key])
    out = df.copy()
    if space == "global":
        if GLOBAL_COL_USER in out.columns and GLOBAL_COL_ITEM in out.columns:
            return out
        if "user_idx" in out.columns and "item_idx" in out.columns:
            out[GLOBAL_COL_USER] = out["user_idx"].astype(np.int64)
            out[GLOBAL_COL_ITEM] = out["item_idx"].astype(np.int64)
            out = out.drop(columns=["user_idx", "item_idx"], errors="ignore")
            return out
        raise IndexContractError(
            f"split={split_name} 为 global 但缺少 {GLOBAL_COL_USER}/{GLOBAL_COL_ITEM} 或 user_idx/item_idx ({_ctx_tail(ctx)})"
        )
    if space == "target_local":
        if "user_idx" not in out.columns or "item_idx" not in out.columns:
            raise IndexContractError(f"split={split_name} 为 target_local 但缺少 user_idx/item_idx ({_ctx_tail(ctx)})")
        out[GLOBAL_COL_USER] = convert_target_local_to_global_user(out["user_idx"], contract)
        out[GLOBAL_COL_ITEM] = convert_target_local_to_global_item(out["item_idx"], contract)
        return out
    raise IndexContractError(f"未知 index_space={space!r} split={split_name} ({_ctx_tail(ctx)})")


def validate_index_contract_against_profiles(
    contract: Mapping[str, Any],
    user_profiles: torch.Tensor,
    item_profiles: torch.Tensor,
    *,
    ctx: Mapping[str, Any],
) -> None:
    nu = int(contract["nuser_global"])
    ni = int(contract["nitem_global"])
    ur, ir = int(user_profiles.shape[0]), int(item_profiles.shape[0])
    if ur != nu:
        raise IndexContractError(
            f"nuser_global={nu} 与拼接 user_profiles.shape[0]={ur} 不一致 ({_ctx_tail(ctx)})"
        )
    if ir != ni:
        raise IndexContractError(
            f"nitem_global={ni} 与拼接 item_profiles.shape[0]={ir} 不一致 ({_ctx_tail(ctx)})"
        )


def validate_split_indices(
    df: pd.DataFrame,
    contract: Mapping[str, Any],
    split_name: str,
    *,
    ctx: Mapping[str, Any],
) -> None:
    if GLOBAL_COL_USER not in df.columns or GLOBAL_COL_ITEM not in df.columns:
        raise IndexContractError(
            f"split={split_name} 缺少 {GLOBAL_COL_USER}/{GLOBAL_COL_ITEM}（应先 normalize）({_ctx_tail(ctx)})"
        )
    nu, ni = int(contract["nuser_global"]), int(contract["nitem_global"])
    u = df[GLOBAL_COL_USER].to_numpy(dtype=np.int64, copy=False)
    it = df[GLOBAL_COL_ITEM].to_numpy(dtype=np.int64, copy=False)
    if u.size == 0:
        return
    u_min, u_max = int(u.min()), int(u.max())
    i_min, i_max = int(it.min()), int(it.max())
    if u_min < 0 or i_min < 0:
        raise IndexContractError(
            f"split={split_name} 索引为负: user[{u_min},{u_max}] item[{i_min},{i_max}] "
            f"合法 user [0,{nu - 1}] item [0,{ni - 1}] ({_ctx_tail(ctx)})"
        )
    if u_max >= nu or i_max >= ni:
        raise IndexContractError(
            f"split={split_name} 索引越界: user min/max={u_min}/{u_max} item min/max={i_min}/{i_max} "
            f"合法 user [0,{nu - 1}] item [0,{ni - 1}] ({_ctx_tail(ctx)})"
        )
    space_key = f"{split_name}_index_space"
    space = str(contract.get(space_key, ""))
    if space == "target_local":
        tc_u = int(contract["target_user_count"])
        tc_i = int(contract["target_item_count"])
        if u_max >= tc_u or i_max >= tc_i:
            raise IndexContractError(
                f"split={split_name} 标为 target_local 但索引超出 target 段: "
                f"user max={u_max} (须 < {tc_u}) item max={i_max} (须 < {tc_i}) ({_ctx_tail(ctx)})"
            )


def validate_domain_indices_tensor(domain_idx: torch.Tensor, *, ctx: Mapping[str, Any]) -> None:
    d = domain_idx.detach().cpu()
    if d.numel() == 0:
        return
    mn, mx = int(d.min().item()), int(d.max().item())
    if mn < 0 or mx > 1:
        raise IndexContractError(
            f"domain_idx 越界: min/max={mn}/{mx} 须落在 {{0,1}} ({_ctx_tail(ctx)})"
        )


def validate_first_batch_indices(
    batch: Tuple[torch.Tensor, ...],
    contract: Mapping[str, Any],
    split_label: str,
    *,
    ctx: Mapping[str, Any],
) -> None:
    """batch 为 step5 collate 输出元组：user_idx, item_idx, ... domain_idx ..."""
    user_idx, item_idx = batch[0], batch[1]
    # domain_idx 索引 4 与 _step5_collate_dynamic 一致
    domain_idx = batch[4]
    nu, ni = int(contract["nuser_global"]), int(contract["nitem_global"])
    u = user_idx.detach().cpu()
    it = item_idx.detach().cpu()
    if u.numel() == 0:
        return
    u_min, u_max = int(u.min().item()), int(u.max().item())
    i_min, i_max = int(it.min().item()), int(it.max().item())
    if u_min < 0 or i_min < 0 or u_max >= nu or i_max >= ni:
        raise IndexContractError(
            f"[batch 审计] split={split_label} user min/max={u_min}/{u_max} item min/max={i_min}/{i_max} "
            f"合法 user [0,{nu - 1}] item [0,{ni - 1}] ({_ctx_tail(ctx)})"
        )
    validate_domain_indices_tensor(domain_idx, ctx=ctx)


def _dual_channel_paths_from_contract(contract: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    def _req(key: str) -> str:
        v = contract.get(key)
        if not v or not str(v).strip():
            raise IndexContractError(f"index_contract 缺少双通道字段 {key!r}")
        return os.path.abspath(str(v))

    target_paths = {
        "user_content": _req("target_user_content_profiles_path"),
        "user_style": _req("target_user_style_profiles_path"),
        "item_content": _req("target_item_content_profiles_path"),
        "item_style": _req("target_item_style_profiles_path"),
        "domain_content": _req("target_domain_content_path"),
        "domain_style": _req("target_domain_style_path"),
    }
    aux_paths = {
        "user_content": _req("aux_user_content_profiles_path"),
        "user_style": _req("aux_user_style_profiles_path"),
        "item_content": _req("aux_item_content_profiles_path"),
        "item_style": _req("aux_item_style_profiles_path"),
        "domain_content": _req("aux_domain_content_path"),
        "domain_style": _req("aux_domain_style_path"),
    }
    return target_paths, aux_paths


def load_profile_tensors_from_contract(
    contract: Mapping[str, Any], device_idx: int | str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """加载双域六路张量：domain_(content|style)、user_(content|style)、item_(content|style)；磁盘缺失则 fail-fast。"""
    target_paths, aux_paths = _dual_channel_paths_from_contract(contract)
    dc, ds, uc, us, ic, ist, _meta = _load_profile_tensors_physical_separate_from_paths(
        target_paths=target_paths,
        aux_paths=aux_paths,
        device_idx=device_idx,
    )
    return dc, ds, uc, us, ic, ist


def write_index_contract_audit(
    out_path: str,
    payload: Mapping[str, Any],
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    return out_path


__all__ = [
    "INDEX_CONTRACT_SCHEMA_VERSION",
    "INDEX_CONTRACT_FILENAME",
    "ODCR_ROUTING_TRAIN_CSV",
    "STEP4_EXPORT_LINEAGE_SCHEMA_VERSION",
    "STEP4_ROUTE_POSTERIOR_CONTRACT_VERSION",
    "STEP4_RCR_EXPORT_SCHEMA_VERSION",
    "GLOBAL_COL_USER",
    "GLOBAL_COL_ITEM",
    "IndexContractError",
    "build_index_contract",
    "build_step4_export_lineage",
    "validate_step4_export_lineage",
    "step4_rcr_required_fields_hash",
    "write_index_contract",
    "load_index_contract",
    "resolve_index_contract_path",
    "remap_step4_train_df_to_global_columns",
    "convert_target_local_to_global_user",
    "convert_target_local_to_global_item",
    "normalize_split_indices_to_global",
    "validate_index_contract_against_profiles",
    "validate_split_indices",
    "validate_domain_indices_tensor",
    "validate_first_batch_indices",
    "load_profile_tensors_dual_first",
    "load_profile_tensors_fused_average_legacy",
    "load_profile_tensors_from_contract",
    "write_index_contract_audit",
    "parse_training_run_lineage",
]
