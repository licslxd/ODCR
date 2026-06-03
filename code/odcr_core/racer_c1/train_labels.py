"""Metric-aligned label helpers for RACER-C1 contrastive retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
CONTRASTIVE_PAIR_MANIFEST_SCHEMA_VERSION = "odcr_racer_c1_contrastive_pairs/1"


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def rouge1_f1(prediction: str, reference: str) -> float:
    p = Counter(tokens(prediction))
    r = Counter(tokens(reference))
    if not p or not r:
        return 0.0
    overlap = sum(min(p[k], r[k]) for k in p)
    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(r.values()))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    a = tokens(prediction)
    b = tokens(reference)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    precision = lcs / max(1, len(a))
    recall = lcs / max(1, len(b))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bleu1_precision(prediction: str, reference: str) -> float:
    p = Counter(tokens(prediction))
    r = Counter(tokens(reference))
    if not p:
        return 0.0
    clipped = sum(min(p[k], r[k]) for k in p)
    return clipped / max(1, sum(p.values()))


def meteor_proxy(prediction: str, reference: str) -> float:
    p = set(tokens(prediction))
    r = set(tokens(reference))
    if not p or not r:
        return 0.0
    precision = len(p & r) / len(p)
    recall = len(p & r) / len(r)
    if precision + recall == 0:
        return 0.0
    return (10 * precision * recall) / (recall + 9 * precision)


def metric_overlap(prediction: str, reference: str, weights: Mapping[str, Any]) -> float:
    values = {
        "rouge1": rouge1_f1(prediction, reference),
        "rougeL": rouge_l_f1(prediction, reference),
        "bleu1": bleu1_precision(prediction, reference),
        "meteor": meteor_proxy(prediction, reference),
    }
    score = 0.0
    for key, value in values.items():
        try:
            score += float(weights.get(key, 0.0)) * float(value)
        except Exception:
            continue
    return 0.0 if math.isnan(score) or math.isinf(score) else float(score)


def positive_weight(
    *,
    metric_score: float,
    rcr_score: float,
    same_item: bool,
    same_user: bool,
    template_score: float,
    causal_content_score: float = 0.0,
    style_shortcut_score: float = 0.0,
    cf_reliability_bucket: str = "",
    weights: Mapping[str, Any],
) -> float:
    raw = (
        float(weights.get("metric_overlap", 0.0)) * metric_score
        + float(weights.get("rcr", 0.0)) * rcr_score
        + float(weights.get("same_item", 0.0)) * (1.0 if same_item else 0.0)
        + float(weights.get("same_user", 0.0)) * (1.0 if same_user else 0.0)
        + float(weights.get("causal_content", 0.0)) * causal_content_score
        + float(weights.get("high_reliability_bucket", 0.0)) * (1.0 if cf_reliability_bucket == "high" else 0.0)
        + float(weights.get("style_shortcut", 0.0)) * style_shortcut_score
        + float(weights.get("template", 0.0)) * template_score
    )
    return max(1e-6, float(raw))


def supervision_role(*, rcr_score: float, template_score: float, causal_content_score: float, style_shortcut_score: float) -> str:
    """Map factual/CF evidence into RACER-C1 contrastive supervision roles."""

    if rcr_score >= 0.80 and causal_content_score >= 0.50 and template_score <= 0.35:
        return "positive_high_rcr_content"
    if rcr_score >= 0.55 and causal_content_score >= 0.40 and style_shortcut_score <= 0.65:
        return "soft_positive_medium_rcr"
    if rcr_score < 0.55:
        return "hard_negative_low_rcr"
    if template_score >= 0.65 or style_shortcut_score >= 0.75:
        return "hard_negative_template_style"
    return "candidate_neutral"


def _clean_reference(row: Mapping[str, Any]) -> str:
    return str(row.get("clean_text") or row.get("explanation") or row.get("reference") or "").strip()


def _evidence_text(evidence: Mapping[str, Any]) -> str:
    return str(evidence.get("clean_explanation_25") or evidence.get("clean_explanation") or evidence.get("raw_explanation") or "").strip()


def _same_interaction(query: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    q_sample = str(query.get("sample_id") or query.get("source_sample_id") or "").strip()
    e_sample = str(evidence.get("source_sample_id") or evidence.get("sample_id") or "").strip()
    q_user = str(query.get("user") or query.get("source_user") or "").strip()
    e_user = str(evidence.get("source_user") or evidence.get("user") or "").strip()
    q_item = str(query.get("item") or query.get("source_item") or "").strip()
    e_item = str(evidence.get("source_item") or evidence.get("item") or "").strip()
    return bool(q_sample and e_sample and q_sample == e_sample) or bool(q_user and q_item and q_user == e_user and q_item == e_item)


def pair_supervision(
    *,
    query: Mapping[str, Any],
    evidence: Mapping[str, Any],
    metric_weights: Mapping[str, Any],
    positive_weights: Mapping[str, Any],
    exclude_current_interaction: bool = True,
) -> dict[str, Any]:
    """Return the train-only contrastive role for one query/evidence pair.

    This is the concrete RACER-C1 replacement for D4C's CE augmentation: the
    target explanation is used only inside train split to label whether a
    factual/CF evidence record should be pulled close, kept as a soft positive,
    or treated as a hard negative. Valid/test never call this with references.
    """

    if exclude_current_interaction and _same_interaction(query, evidence):
        return {
            "role": "self_excluded_current_interaction",
            "positive_weight": 0.0,
            "metric_overlap": 0.0,
            "same_item": False,
            "same_user": False,
            "excluded": True,
        }

    reference = _clean_reference(query)
    text = _evidence_text(evidence)
    metric_score = metric_overlap(text, reference, metric_weights)
    q_user = str(query.get("user") or query.get("source_user") or "").strip()
    q_item = str(query.get("item") or query.get("source_item") or "").strip()
    same_item = bool(q_item and q_item == str(evidence.get("source_item") or evidence.get("item") or "").strip())
    same_user = bool(q_user and q_user == str(evidence.get("source_user") or evidence.get("user") or "").strip())
    rcr_score = float(evidence.get("rcr_score") or evidence.get("cf_reliability_score") or 0.0)
    template = float(evidence.get("template_score") or 0.0)
    causal_content = float(evidence.get("causal_content_score") or 0.0)
    style_shortcut = float(evidence.get("style_shortcut_score") or 0.0)
    bucket = str(evidence.get("cf_reliability_bucket") or "")
    role = supervision_role(
        rcr_score=rcr_score,
        template_score=template,
        causal_content_score=causal_content,
        style_shortcut_score=style_shortcut,
    )
    if role == "candidate_neutral":
        if metric_score >= 0.35 or same_item or same_user:
            role = "positive_metric_or_history"
        elif metric_score <= 0.05 and template >= 0.35:
            role = "hard_negative_metric_low_template"
        else:
            role = "in_batch_negative"

    is_positive = role.startswith("positive") or role.startswith("soft_positive")
    weight = (
        positive_weight(
            metric_score=metric_score,
            rcr_score=rcr_score,
            same_item=same_item,
            same_user=same_user,
            template_score=template,
            causal_content_score=causal_content,
            style_shortcut_score=style_shortcut,
            cf_reliability_bucket=bucket,
            weights=positive_weights,
        )
        if is_positive
        else 0.0
    )
    return {
        "role": role,
        "positive_weight": round(float(weight), 6),
        "metric_overlap": round(float(metric_score), 6),
        "same_item": same_item,
        "same_user": same_user,
        "excluded": False,
        "rcr_score": rcr_score,
        "template_score": template,
        "causal_content_score": causal_content,
        "style_shortcut_score": style_shortcut,
        "cf_reliability_bucket": bucket,
    }


def write_contrastive_pair_manifest(
    path: Path,
    *,
    evidence_summary: Mapping[str, Any],
    racer_cfg: Mapping[str, Any],
    train_evidence_path: Path,
) -> dict[str, Any]:
    """Write the machine-readable supervision contract used by GPU training.

    The manifest intentionally records construction rules instead of expanding
    all O(N*K) pairs during prepare. The future CUDA trainer materializes pairs
    from this contract and the train-only evidence pool, while cache identity
    remains content-affecting and lineage keeps epoch/batch/runtime knobs.
    """

    contrastive = dict(racer_cfg.get("contrastive") or {})
    evidence = dict(racer_cfg.get("evidence_pool") or {})
    payload = {
        "schema_version": CONTRASTIVE_PAIR_MANIFEST_SCHEMA_VERSION,
        "status": "rules_materialized_pair_expansion_deferred_to_gpu_trainer",
        "train_only": True,
        "train_evidence_path": train_evidence_path.as_posix(),
        "evidence_pool_record_count": int(evidence_summary.get("record_count") or 0),
        "source_type_counts": dict(evidence_summary.get("source_type_counts") or {}),
        "source_domain_counts": dict(evidence_summary.get("source_domain_counts") or {}),
        "current_interaction_exclusion": bool(evidence.get("exclude_current_train_interaction", True)),
        "metric_overlap_train_label_only": True,
        "valid_test_reference_forbidden": True,
        "positive_construction": [
            "same_item_high_metric_overlap",
            "same_user_preference_consistent",
            "high_rcr_counterfactual_or_factual_content",
            "top_k_metric_overlap_train_only",
        ],
        "negative_construction": list(contrastive.get("hard_negative_types") or []),
        "num_pos_per_query": int(contrastive.get("num_pos_per_query") or 0),
        "num_hard_neg_per_query": int(contrastive.get("num_hard_neg_per_query") or 0),
        "metric_overlap_weights": dict(contrastive.get("metric_overlap_weights") or {}),
        "positive_weight": dict(contrastive.get("positive_weight") or {}),
        "loss": {
            "name": "weighted_multi_positive_infonce",
            "positive_matrix_semantics": "positive_weight>0 pulls evidence close; zero-weight entries remain negatives",
            "hard_negative_semantics": "low-RCR/high-template/wrong-content evidence is retained as contrastive repulsion, not CE text",
        },
        "evidence_weighted_retrieval_expectation": {
            "replaces": "D4C generator expectation over full factual/counterfactual explanation sentences",
            "formal_reading": "P(e_hat | do(U,I)) is approximated by a weighted train-only evidence selection distribution",
            "selection_terms": {
                "factual_evidence": [
                    "same_item_high_metric_overlap",
                    "same_user_preference_consistent",
                    "same_domain_rating_aspect_consistent",
                ],
                "counterfactual_evidence": [
                    "high_rcr_counterfactual_content_anchor",
                    "medium_rcr_soft_positive_anchor",
                    "low_rcr_or_template_counterfactual_hard_negative",
                ],
            },
            "weight_formula": {
                "positive_components": ["metric_overlap_train", "RCR", "same_item", "same_user", "causal_content", "high_reliability_bucket"],
                "penalty_components": ["style_shortcut", "template_score"],
                "valid_test_reference_usage": "forbidden",
            },
        },
        "contrastive_role_policy": {
            "positive_high_rcr_content": "high-RCR, high causal-content, low-template evidence",
            "soft_positive_medium_rcr": "medium-RCR content evidence with controlled style shortcut",
            "hard_negative_low_rcr": "low reliability CF/factual evidence",
            "hard_negative_template_style": "high-template or high-style-shortcut evidence",
            "positive_metric_or_history": "train-only high-overlap or same user/item history evidence",
            "in_batch_negative": "ordinary contrastive negative",
            "self_excluded_current_interaction": "train query's own interaction is excluded from candidate evidence",
        },
        "d4c_chain_replacement": {
            "from": "factual_and_counterfactual_sentence_ce_augmentation",
            "to": "factual_and_counterfactual_evidence_positive_negative_weighting",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload
