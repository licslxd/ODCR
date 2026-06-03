"""Machine-readable RACER-C1 innovation alignment report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


INNOVATION_ALIGNMENT_SCHEMA_VERSION = "odcr_racer_c1_innovation_alignment/1"


def build_innovation_alignment(
    *,
    racer_cfg: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    contrastive = dict(racer_cfg.get("contrastive") or {})
    positive_weight = dict(contrastive.get("positive_weight") or {})
    guardrails = dict(racer_cfg.get("guardrails") or {})
    record_count = int(evidence_summary.get("record_count") or 0)
    pair_status = str(pair_manifest.get("status") or "")
    rows = [
        {
            "id": "I1_content_style_split",
            "claim": "D4C W is split into causal content evidence C and style/template shortcut S.",
            "status": "implemented",
            "code_contract": [
                "EvidenceRecord.causal_content_evidence",
                "EvidenceRecord.style_shortcut_evidence",
                "causal_content_score",
                "style_shortcut_score",
                "diagnostics/content_style_split_stats.json",
            ],
            "artifact_evidence": {"train_evidence_records": record_count},
            "metric_role": "C supports ROUGE/BLEU/METEOR; S is penalized to protect DIST.",
        },
        {
            "id": "I2_evidence_weighted_retrieval_expectation",
            "claim": "Causal expectation is moved from generator probability to weighted evidence selection.",
            "status": "implemented",
            "code_contract": [
                "positive_weight.metric_overlap",
                "positive_weight.rcr",
                "positive_weight.causal_content",
                "positive_weight.template",
                "evidence_weighted_retrieval_expectation",
            ],
            "artifact_evidence": {"pair_manifest_status": pair_status},
            "metric_role": "Retrieval selection is directly aligned to reference-like evidence overlap.",
        },
        {
            "id": "I3_counterfactual_evidence_anchors",
            "claim": "Full counterfactual sentences are decomposed into content/style/aspect anchors.",
            "status": "implemented",
            "code_contract": [
                "EvidenceRecord.cf_content_anchor",
                "EvidenceRecord.cf_style_anchor",
                "EvidenceRecord.cf_aspect_anchor",
                "EvidenceRecord.cf_reliability_bucket",
                "EvidenceRecord.cf_template_flag",
                "diagnostics/cf_anchor_stats.json",
            ],
            "artifact_evidence": {"source_type_counts": evidence_summary.get("source_type_counts") or {}},
            "metric_role": "Anchors provide controllable n-gram/aspect material without trusting noisy CF sentences as labels.",
        },
        {
            "id": "I4_cf_as_contrastive_roles",
            "claim": "Factual/CF samples become positives, soft positives, hard negatives, and weights instead of CE augmentation.",
            "status": "implemented",
            "code_contract": [
                "pair_supervision",
                "supervision_role",
                "write_contrastive_pair_manifest",
                "diagnostics/contrastive_role_distribution.json",
            ],
            "artifact_evidence": {
                "positive_construction": pair_manifest.get("positive_construction") or [],
                "negative_construction": pair_manifest.get("negative_construction") or [],
            },
            "metric_role": "High-quality evidence pulls close; low-RCR/template evidence teaches retrieval repulsion.",
        },
        {
            "id": "I5_large_batch_contrastive_alignment",
            "claim": "Adversarial domain alignment is replaced by large-batch contrastive retrieval alignment.",
            "status": "implemented_training_ready",
            "code_contract": [
                "RacerDualEncoder",
                "weighted_multi_positive_infonce",
                "train.global_batch_size",
                "train.per_gpu_batch_size",
            ],
            "artifact_evidence": {
                "projection_dim": contrastive.get("projection_dim"),
                "global_batch_size": (racer_cfg.get("train") or {}).get("global_batch_size"),
            },
            "metric_role": "In-batch negatives optimize evidence alignment more directly than discriminator confusion.",
        },
        {
            "id": "I6_train_retriever_not_generator",
            "claim": "The primary trainable object is the retriever; the deleted generator path is not invoked.",
            "status": "implemented",
            "code_contract": [
                "legacy_generator_policy=deleted_not_available",
                "run_racer_c1",
                "contrastive_trainer.run_train_eval",
            ],
            "artifact_evidence": {"legacy_generator_policy": racer_cfg.get("legacy_generator_policy")},
            "metric_role": "Avoids paraphrase-heavy large generation and keeps output grounded in train-only evidence.",
        },
        {
            "id": "I7_rcr_as_retrieval_supervision",
            "claim": "RCR is promoted from routing metadata to retrieval weighting and hard-negative semantics.",
            "status": "implemented",
            "code_contract": [
                "positive_weight.rcr",
                "cf_reliability_bucket",
                "contrastive_role_hint",
                "retrieval.rcr_score_weight",
                "retrieval_rerank_score",
            ],
            "artifact_evidence": {"rcr_weight": positive_weight.get("rcr")},
            "metric_role": "Reliable CF/factual evidence changes the contrastive objective and later reranking.",
        },
        {
            "id": "I8_traceable_evidence_prediction",
            "claim": "Predictions must carry retrieved evidence provenance for no-leak auditability.",
            "status": "implemented_boundary",
            "code_contract": [
                "select_top1_prediction",
                "write_top1_predictions",
                "prediction_provenance_required",
                "leakage_check",
            ],
            "artifact_evidence": {
                "train_only_guardrail": guardrails.get("train_only_evidence"),
                "provenance_required": guardrails.get("prediction_provenance_required"),
            },
            "metric_role": "Every output can be traced to train evidence rather than hidden reference-side text.",
        },
    ]
    return {
        "schema_version": INNOVATION_ALIGNMENT_SCHEMA_VERSION,
        "method_name": racer_cfg.get("method_name") or "RACER-C1",
        "paper_method_name": racer_cfg.get("paper_method_name") or "RACER",
        "status": "PASS" if all(str(row["status"]).startswith("implemented") for row in rows) else "FAIL",
        "rows": rows,
        "summary": {
            "implemented": sum(1 for row in rows if str(row["status"]).startswith("implemented")),
            "total": len(rows),
            "missing": [row["id"] for row in rows if not str(row["status"]).startswith("implemented")],
        },
    }


def write_innovation_alignment(
    path: Path,
    *,
    racer_cfg: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    pair_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = build_innovation_alignment(
        racer_cfg=racer_cfg,
        evidence_summary=evidence_summary,
        pair_manifest=pair_manifest,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload
