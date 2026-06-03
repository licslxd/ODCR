from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.racer_c1.composer import compose_prediction
from odcr_core.racer_c1.retrieve_predict import retrieval_rerank_score, select_top1_prediction
from odcr_core.racer_c1.train_labels import pair_supervision


METRIC_WEIGHTS = {"rouge1": 0.35, "rougeL": 0.25, "bleu1": 0.20, "meteor": 0.20}
POSITIVE_WEIGHTS = {
    "metric_overlap": 0.45,
    "rcr": 0.20,
    "same_item": 0.12,
    "same_user": 0.08,
    "causal_content": 0.20,
    "high_reliability_bucket": 0.08,
    "style_shortcut": -0.15,
    "template": -0.10,
}


def test_pair_supervision_promotes_high_rcr_content_and_penalizes_template_cf() -> None:
    query = {
        "sample_id": "q1",
        "user": "u1",
        "item": "i1",
        "clean_text": "the sound quality is great and the songs are classic",
    }
    high_content = {
        "source_sample_id": "other",
        "source_user": "u2",
        "source_item": "i1",
        "clean_explanation_25": "great sound quality and classic songs",
        "rcr_score": 0.91,
        "template_score": 0.05,
        "causal_content_score": 0.82,
        "style_shortcut_score": 0.10,
        "cf_reliability_bucket": "high",
    }
    template_cf = {
        "source_sample_id": "bad",
        "source_user": "u9",
        "source_item": "i9",
        "clean_explanation_25": "good product and i recommend it",
        "rcr_score": 0.20,
        "template_score": 0.91,
        "causal_content_score": 0.10,
        "style_shortcut_score": 0.90,
        "cf_reliability_bucket": "low",
    }

    pos = pair_supervision(
        query=query,
        evidence=high_content,
        metric_weights=METRIC_WEIGHTS,
        positive_weights=POSITIVE_WEIGHTS,
    )
    neg = pair_supervision(
        query=query,
        evidence=template_cf,
        metric_weights=METRIC_WEIGHTS,
        positive_weights=POSITIVE_WEIGHTS,
    )

    assert pos["role"] == "positive_high_rcr_content"
    assert pos["positive_weight"] > 0
    assert neg["role"] == "hard_negative_low_rcr"
    assert neg["positive_weight"] == 0.0


def test_pair_supervision_excludes_current_train_interaction() -> None:
    query = {"sample_id": "same", "user": "u1", "item": "i1", "clean_text": "great sound"}
    evidence = {
        "source_sample_id": "same",
        "source_user": "u1",
        "source_item": "i1",
        "clean_explanation_25": "great sound",
    }
    result = pair_supervision(
        query=query,
        evidence=evidence,
        metric_weights=METRIC_WEIGHTS,
        positive_weights=POSITIVE_WEIGHTS,
    )
    assert result["role"] == "self_excluded_current_interaction"
    assert result["excluded"] is True


def test_top1_prediction_is_rcr_reranked_train_only_and_traceable() -> None:
    retrieval_cfg = {
        "prediction_policy": "top1_clean_explanation_25",
        "rcr_score_weight": 0.15,
        "template_penalty_weight": 0.15,
    }
    candidates = [
        {
            "evidence_id": "e1",
            "source_split": "train",
            "clean_explanation_25": "good product and i recommend it",
            "retrieval_score": 0.92,
            "rcr_score": 0.10,
            "template_score": 0.95,
            "source_user": "u9",
            "source_item": "i9",
            "source_domain": "AM_CDs",
        },
        {
            "evidence_id": "e2",
            "source_split": "train",
            "clean_explanation_25": "great sound quality and classic songs",
            "retrieval_score": 0.86,
            "rcr_score": 0.90,
            "template_score": 0.05,
            "source_user": "u2",
            "source_item": "i1",
            "source_domain": "AM_CDs",
        },
    ]
    assert retrieval_rerank_score(candidates[1], retrieval_cfg) > retrieval_rerank_score(candidates[0], retrieval_cfg)

    prediction = select_top1_prediction(sample_id="valid-1", candidates=candidates, retrieval_cfg=retrieval_cfg)

    assert prediction["prediction"] == "great sound quality and classic songs"
    assert prediction["retrieved_evidence_id"] == "e2"
    assert prediction["source_split"] == "train"
    assert prediction["rcr_score"] == 0.90
    assert prediction["template_score"] == 0.05


def test_composer_is_official_minimal_rewrite_not_exact_copy() -> None:
    candidates = [
        {
            "evidence_id": "e2",
            "source_split": "train",
            "clean_explanation_25": "great sound quality and classic songs",
            "causal_content_evidence": "keywords sound quality classic songs",
            "cf_content_anchor": "sound quality classic songs",
            "retrieval_score": 0.86,
            "rcr_score": 0.90,
            "template_score": 0.05,
            "source_user": "u2",
            "source_item": "i1",
            "source_domain": "AM_CDs",
            "source_type": "target_gold",
        }
    ]
    payload = compose_prediction(
        sample_id="valid-1",
        candidates=candidates,
        retrieval_cfg={"top_k": 3},
        composer_cfg={
            "policy": "rule_based_minimal_rewrite",
            "max_output_tokens": 25,
            "max_input_evidence": 3,
            "forbid_exact_copy": True,
            "max_lcs_ratio": 0.85,
        },
    )
    assert payload["source_split"] == "train"
    assert payload["prediction"] != candidates[0]["clean_explanation_25"]
    assert payload["retrieved_evidence_id"] == "e2"
    assert "copy_ratio" in payload


def test_top1_prediction_rejects_non_train_candidate() -> None:
    candidate = {
        "evidence_id": "bad",
        "source_split": "valid",
        "clean_explanation_25": "reference side text",
    }
    try:
        select_top1_prediction(sample_id="x", candidates=[candidate], retrieval_cfg={})
    except ValueError as exc:
        assert "train split" in str(exc)
    else:
        raise AssertionError("valid/test candidate was accepted as RACER-C1 evidence")
