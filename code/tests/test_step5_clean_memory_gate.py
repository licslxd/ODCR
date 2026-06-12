from __future__ import annotations

from pathlib import Path

import pandas as pd

from odcr_core.step5_clean_memory import (
    STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION,
    STEP5_CLEAN_MEMORY_MODE,
    STEP5_CLEAN_MEMORY_SOURCE,
    apply_step5_clean_memory_controls,
)
from tools.odcr_clean_memory_gate import analyze_frame_contract


def test_clean_memory_gate_rejects_current_row_control_frame_without_source() -> None:
    df = pd.DataFrame(
        {
            "user_idx_global": [1],
            "item_idx_global": [2],
            "review": ["current row review"],
            "explanation": ["current row explanation"],
            "rating": [5.0],
            "content_evidence": ["keywords current answer"],
            "polarity_anchor": ["positive"],
            "content_anchor_score": [0.8],
            "style_anchor_score": [0.7],
            "evidence_quality_prior": [0.9],
        }
    )

    findings = analyze_frame_contract(
        df,
        split="valid",
        path="data/AM_CDs/valid.csv",
        require_leave_one_out=False,
    )

    gates = {finding.gate for finding in findings}
    assert "valid_current_row_control_source" in gates
    assert "valid_gold_rating_polarity_risk" in gates


def test_clean_memory_gate_accepts_train_memory_eval_control_frame() -> None:
    df = pd.DataFrame(
        {
            "user_idx_global": [1],
            "item_idx_global": [2],
            "rating": [5.0],
            "explanation": ["metric reference only"],
            "content_evidence": ["USER_HISTORY: clear bass; ITEM_HISTORY: compact player"],
            "polarity_anchor": ["positive"],
            "content_anchor_score": [0.8],
            "style_anchor_score": [0.7],
            "evidence_quality_prior": [0.9],
            "style_evidence": ["USER_STYLE: clean train style"],
            "domain_style_anchor": ["target:train_memory_style_prior:positive"],
            "local_style_residual_hint": ["source=train_only_memory"],
            "sample_weight_hint": [0.9],
            "route_scorer": [1],
            "route_explainer": [1],
            "step5_clean_control_source": [STEP5_CLEAN_MEMORY_SOURCE],
            "step5_clean_control_contract_version": [STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION],
            "step5_control_mode": [STEP5_CLEAN_MEMORY_MODE],
            "step5_control_source": [STEP5_CLEAN_MEMORY_SOURCE],
            "step5_control_contract_version": [STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION],
            "step5_leave_one_out_memory": [False],
        }
    )

    findings = analyze_frame_contract(
        df,
        split="valid",
        path="data/AM_CDs/valid.csv",
        require_leave_one_out=False,
    )

    assert findings == []


def test_clean_memory_gate_requires_leave_one_out_marker_for_train() -> None:
    df = pd.DataFrame(
        {
            "user_idx_global": [1],
            "item_idx_global": [2],
            "clean_text": ["target explanation"],
            "content_evidence": ["USER_HISTORY: from other train rows"],
            "polarity_anchor": ["neutral"],
            "content_anchor_score": [0.8],
            "style_anchor_score": [0.7],
            "evidence_quality_prior": [0.9],
            "style_evidence": ["USER_STYLE: clean train style"],
            "domain_style_anchor": ["target:train_memory_style_prior:neutral"],
            "local_style_residual_hint": ["source=train_only_memory"],
            "sample_weight_hint": [0.9],
            "route_scorer": [1],
            "route_explainer": [1],
            "step5_clean_control_source": [STEP5_CLEAN_MEMORY_SOURCE],
            "step5_clean_control_contract_version": [STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION],
            "step5_control_mode": [STEP5_CLEAN_MEMORY_MODE],
            "step5_control_source": [STEP5_CLEAN_MEMORY_SOURCE],
            "step5_control_contract_version": [STEP5_CLEAN_MEMORY_CONTROL_SCHEMA_VERSION],
        }
    )

    findings = analyze_frame_contract(
        df,
        split="train",
        path="runs/step4/task2/1/odcr_routing_train.csv",
        require_leave_one_out=True,
    )

    assert {finding.gate for finding in findings} == {"train_leave_one_out_memory"}


def test_clean_memory_controls_exclude_current_train_row(tmp_path: Path) -> None:
    target = tmp_path / "data" / "Target"
    auxiliary = tmp_path / "data" / "Aux"
    target.mkdir(parents=True)
    auxiliary.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "user": "u1",
                "item": "i1",
                "user_idx": 1,
                "item_idx": 2,
                "rating": 5.0,
                "review": "leaky current row review",
                "explanation": "leaky current row explanation",
            },
            {
                "user": "u1",
                "item": "i1",
                "user_idx": 1,
                "item_idx": 2,
                "rating": 2.0,
                "review": "other train memory review",
                "explanation": "other train memory style",
            },
        ]
    ).to_csv(target / "train.csv", index=False)
    pd.DataFrame(
        [
            {
                "user": "aux-user",
                "item": "aux-item",
                "user_idx": 1,
                "item_idx": 2,
                "rating": 3.0,
                "review": "auxiliary memory",
                "explanation": "auxiliary style",
            }
        ]
    ).to_csv(auxiliary / "train.csv", index=False)
    df = pd.DataFrame(
        {
            "user": ["u1"],
            "item": ["i1"],
            "user_idx": [1],
            "item_idx": [2],
            "user_idx_global": [1],
            "item_idx_global": [2],
            "domain": ["target"],
            "rating": [5.0],
            "review": ["leaky current row review"],
            "explanation": ["leaky current row explanation"],
            "clean_text": ["leaky current row explanation"],
        }
    )

    out = apply_step5_clean_memory_controls(
        df,
        repo_root=tmp_path,
        target_domain="Target",
        auxiliary_domain="Aux",
        index_contract={"target_user_offset": 0, "target_item_offset": 0},
        split_label="train",
        leave_one_out=True,
    )

    assert "other train memory review" in str(out.loc[0, "content_evidence"])
    assert "leaky current row review" not in str(out.loc[0, "content_evidence"])
    assert float(out.loc[0, "rating"]) == 2.0
    assert bool(out.loc[0, "step5_leave_one_out_memory"]) is True


def test_clean_memory_controls_use_local_eval_ids_when_global_columns_missing(tmp_path: Path) -> None:
    target = tmp_path / "data" / "Target"
    auxiliary = tmp_path / "data" / "Aux"
    target.mkdir(parents=True)
    auxiliary.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "user": "u-local",
                "item": "i-local",
                "user_idx": 7,
                "item_idx": 11,
                "rating": 4.0,
                "review": "local eval train memory review",
                "explanation": "local eval train memory style",
            }
        ]
    ).to_csv(target / "train.csv", index=False)
    pd.DataFrame(
        [
            {
                "user": "aux-user",
                "item": "aux-item",
                "user_idx": 1,
                "item_idx": 2,
                "rating": 3.0,
                "review": "auxiliary memory",
                "explanation": "auxiliary style",
            }
        ]
    ).to_csv(auxiliary / "train.csv", index=False)
    df = pd.DataFrame(
        {
            "user": ["u-local"],
            "item": ["i-local"],
            "user_idx": [7],
            "item_idx": [11],
            "domain": ["target"],
            "rating": [1.0],
            "review": ["current eval reference only"],
            "explanation": ["current eval answer only"],
        }
    )

    out = apply_step5_clean_memory_controls(
        df,
        repo_root=tmp_path,
        target_domain="Target",
        auxiliary_domain="Aux",
        index_contract={"target_user_offset": 0, "target_item_offset": 0},
        split_label="valid",
        leave_one_out=False,
    )

    assert "local eval train memory review" in str(out.loc[0, "content_evidence"])
    assert "DOMAIN_PRIOR" not in str(out.loc[0, "content_evidence"])
    assert float(out.loc[0, "rating"]) == 4.0
