"""Top-1 retrieval prediction helpers for RACER-C1.

This module is the no-generator prediction boundary: candidates must already be
train-only evidence records, and the final output is the selected evidence text
with full provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def valid_prediction_text(text: str) -> bool:
    toks = str(text or "").strip().split()
    return 1 <= len(toks) <= 25


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if out == out and out not in {float("inf"), float("-inf")} else default


def retrieval_rerank_score(candidate: Mapping[str, Any], retrieval_cfg: Mapping[str, Any]) -> float:
    """Score one evidence candidate after the contrastive retriever.

    The base score comes from the trained retriever/hybrid retriever. RCR and
    template terms are One-Control weights under ``step5.racer_c1.retrieval``.
    """

    base = _safe_float(
        candidate.get("retrieval_score", candidate.get("contrastive_score", candidate.get("score", 0.0))),
        0.0,
    )
    rcr = _safe_float(candidate.get("rcr_score", candidate.get("cf_reliability_score", 0.0)), 0.0)
    template = _safe_float(candidate.get("template_score", 0.0), 0.0)
    rcr_weight = _safe_float(retrieval_cfg.get("rcr_score_weight"), 0.0)
    template_weight = _safe_float(retrieval_cfg.get("template_penalty_weight"), 0.0)
    return round(base + rcr_weight * rcr - template_weight * template, 6)


def _prediction_text(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("clean_explanation_25")
        or candidate.get("prediction")
        or candidate.get("clean_explanation")
        or candidate.get("raw_explanation")
        or ""
    ).strip()


def _require_train_provenance(candidate: Mapping[str, Any]) -> None:
    source_split = str(candidate.get("source_split") or "").strip()
    if source_split != "train":
        raise ValueError(f"RACER-C1 prediction candidate must come from train split, got {source_split!r}")
    if not str(candidate.get("evidence_id") or candidate.get("retrieved_evidence_id") or "").strip():
        raise ValueError("RACER-C1 prediction candidate missing evidence_id provenance")


def select_top1_prediction(
    *,
    sample_id: str,
    candidates: Iterable[Mapping[str, Any]],
    retrieval_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the top train-only evidence record and return prediction payload.

    Invalid 25-token candidates are skipped according to the configured top2
    fallback policy. No reference-side text is accepted or inspected here.
    """

    ranked: list[tuple[float, int, Mapping[str, Any]]] = []
    for idx, candidate in enumerate(candidates):
        _require_train_provenance(candidate)
        ranked.append((retrieval_rerank_score(candidate, retrieval_cfg), -idx, candidate))
    if not ranked:
        raise ValueError("RACER-C1 top1 prediction requires at least one train evidence candidate")
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = ranked[0][2]
    selected_rank = 1
    fallback_used = False
    if not valid_prediction_text(_prediction_text(selected)):
        for rank, (_score, _neg_idx, candidate) in enumerate(ranked[1:], start=2):
            if valid_prediction_text(_prediction_text(candidate)):
                selected = candidate
                selected_rank = rank
                fallback_used = True
                break
    score = retrieval_rerank_score(selected, retrieval_cfg)
    return {
        "sample_id": str(sample_id),
        "prediction": _prediction_text(selected),
        "retrieved_evidence_id": str(selected.get("evidence_id") or selected.get("retrieved_evidence_id") or ""),
        "retrieval_score": score,
        "base_retrieval_score": _safe_float(
            selected.get("retrieval_score", selected.get("contrastive_score", selected.get("score", 0.0))),
            0.0,
        ),
        "rcr_score": _safe_float(selected.get("rcr_score", selected.get("cf_reliability_score", 0.0)), 0.0),
        "template_score": _safe_float(selected.get("template_score", 0.0), 0.0),
        "source_user": str(selected.get("source_user") or ""),
        "source_item": str(selected.get("source_item") or ""),
        "source_domain": str(selected.get("source_domain") or ""),
        "source_split": "train",
        "source_type": str(selected.get("source_type") or ""),
        "fallback_used": fallback_used,
        "selected_rank": selected_rank,
        "prediction_policy": str(retrieval_cfg.get("prediction_policy") or "top1_clean_explanation_25"),
    }


def write_top1_predictions(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fallback_count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = dict(row)
            if not valid_prediction_text(str(payload.get("prediction") or "")):
                fallback_count += 1
                payload["prediction_invalid"] = True
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return {
        "schema_version": "odcr_racer_c1_prediction_writer/1",
        "path": path.as_posix(),
        "row_count": count,
        "invalid_prediction_count": fallback_count,
        "provenance_required": True,
    }
