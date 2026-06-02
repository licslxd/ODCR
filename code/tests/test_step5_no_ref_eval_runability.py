from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import odcr as odcr_cli  # noqa: E402
from executors import step5_engine as s5  # noqa: E402
from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.step5_no_ref_history_cache import (  # noqa: E402
    ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF,
    SCHEMA_VERSION as NO_REF_HISTORY_CACHE_SCHEMA_VERSION,
    build_or_load_no_ref_evidence_for_frame,
    collect_phrase_filter_audit,
    extract_evidence_phrases,
)
from tools import odcr_rebuild_step5_post_train_eval  # noqa: E402


def _eval_cfg() -> dict:
    return {
        "generation_input_policy": s5.STEP5_GENERATION_INPUT_POLICY_NO_REF,
        "content_evidence_policy": s5.STEP5_CONTENT_EVIDENCE_POLICY_TRAIN_ONLY_HISTORY,
        "reference_usage": s5.STEP5_REFERENCE_USAGE_METRIC_ONLY,
        "neutral_content_evidence": s5.STEP5_NEUTRAL_CONTENT_EVIDENCE,
        "forbid_current_eval_fields_in_generation": list(s5._STEP5_NO_REF_FORBIDDEN_EVAL_FIELDS),
    }


def test_eval_progress_heartbeat_writes_atomic_progress_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        meta = Path(td) / "eval" / "meta"
        payload = s5._step5_write_eval_progress(
            meta_dir=str(meta),
            split="valid",
            task=8,
            run_id="1_19",
            rank=0,
            batch_idx=3,
            total_batches=8,
            rows_done=384,
            elapsed_sec=12.0,
            last_batch_decode_sec=2.5,
            device="cpu",
            stage="decode",
        )
        saved = json.loads((meta / "eval_progress.json").read_text(encoding="utf-8"))
        assert saved["schema_version"] == s5.STEP5_EVAL_PROGRESS_SCHEMA_VERSION
        assert saved["rows_done"] == 384
        assert saved["samples_per_sec"] == payload["samples_per_sec"]


def test_bounded_eval_resolves_max_rows_and_non_official_output_dir() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="eval",
        task_id=2,
        set_overrides=["eval.split=valid", "eval.max_rows=512"],
        dry_run=True,
        run_id="512",
        from_step5="1_18",
    )
    assert cfg.eval_max_rows == 512
    assert snapshot["eval"]["max_rows"] == 512
    assert str(cfg.eval_run_dir).endswith("runs/rerank/task2/512")


def test_no_current_eval_row_reference_review_or_content_evidence_in_generation_input() -> None:
    with tempfile.TemporaryDirectory() as td:
        data_root = Path(td) / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,explanation\nu1,i1,train history lobby breakfast helpful staff\n",
            encoding="utf-8",
        )
        with mock.patch("executors.step5_engine.get_data_dir", return_value=str(data_root)), mock.patch(
            "executors.step5_engine.get_odcr_root", return_value=str(Path(td))
        ):
            df = pd.DataFrame(
                {
                    "user_idx_global": [0],
                    "item_idx_global": [1],
                    "user": ["u1"],
                    "item": ["i1"],
                    "rating": [4.0],
                    "explanation": ["UNIQUE_CURRENT_REFERENCE_SHOULD_NOT_APPEAR"],
                    "review": ["UNIQUE_CURRENT_REVIEW_SHOULD_NOT_APPEAR"],
                    "clean_text": ["UNIQUE_CURRENT_REFERENCE_SHOULD_NOT_APPEAR"],
                    "content_evidence": ["UNIQUE_CURRENT_CONTENT_EVIDENCE_SHOULD_NOT_APPEAR"],
                    "style_evidence": ["UNIQUE_CURRENT_STYLE_EVIDENCE_SHOULD_NOT_APPEAR"],
                }
            )
            out = s5._apply_step5_factual_eval_default_controls(
                df,
                split_label="valid",
                auxiliary="Aux",
                target="Target",
                final_eval_config=_eval_cfg(),
            )
        generation_input = str(out.loc[0, "content_evidence"])
        assert "UNIQUE_CURRENT_REFERENCE_SHOULD_NOT_APPEAR" not in generation_input
        assert "UNIQUE_CURRENT_REVIEW_SHOULD_NOT_APPEAR" not in generation_input
        assert "UNIQUE_CURRENT_CONTENT_EVIDENCE_SHOULD_NOT_APPEAR" not in generation_input


def test_generation_input_samples_record_visible_prompt_without_reference_leak() -> None:
    class TinyTok:
        eos_token_id = 1
        pad_token_id = 0

        def __call__(self, text, padding=False, max_length=None, truncation=False):
            ids = list(range(2, 2 + len(str(text).split())))
            if truncation and max_length is not None:
                ids = ids[: int(max_length)]
            return {"input_ids": ids}

        def decode(self, ids, skip_special_tokens=True):
            return "decoded " + " ".join(str(x) for x in ids)

    eval_frame = pd.DataFrame(
        {
            "sample_id": [7],
            "source_dataset": ["Yelp"],
            "source_split": ["valid"],
            "source_row_id": ["row-7"],
            "sample_origin": ["target_gold"],
            "content_evidence_source": ["compact_train_only_history"],
            "content_evidence": ["Item evidence: fresh sushi.\nTask: Write one concise review reason using the evidence."],
        }
    )
    rows = [
        {
            "sample_id": 7,
            "pred_text": "fresh sushi",
            "raw_ref_text": "REFERENCE_METRIC_ONLY",
            "encoder_raw_input_token_len": 16,
            "encoder_input_token_len": 12,
            "encoder_truncated": False,
        }
    ]
    with mock.patch("executors.step5_engine.get_step5_tokenizer", return_value=TinyTok()):
        samples = s5._build_step5_generation_input_samples(
            rows,
            eval_frame=eval_frame,
            split_label="valid",
            split_csv="data/Yelp/valid.csv",
            no_ref_input_protocol="text_clean_item_only_no_ref",
            encoder_content_token_budget=128,
        )
    assert len(samples) == 1
    sample = samples[0]
    assert "C: fresh sushi" in sample["raw_prompt"]
    assert "Task:" not in sample["raw_prompt"]
    assert sample["prediction"] == "fresh sushi"
    assert sample["reference_metric_only"] == "REFERENCE_METRIC_ONLY"
    assert sample["reference_before_generation"] is False
    assert sample["current_eval_row_reference_used_in_generation"] is False
    assert sample["row_lineage"]["source_split"] == "valid"


def test_current_eval_overlap_guard_neutralizes_history_text() -> None:
    with tempfile.TemporaryDirectory() as td:
        data_root = Path(td) / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,explanation\n"
            "u1,i1,uniquerefalpha uniquerefbeta uniquerefgamma\n"
            "u1,i1,uniquerefalpha uniquerefbeta uniquerefgamma\n",
            encoding="utf-8",
        )
        no_ref_cfg = {
            "input_protocol": "text_clean_item_user_no_ref",
            "min_df": 1,
            "user_history_top_k": 6,
            "item_history_top_k": 6,
            "domain_prior_display_top_k": 6,
        }
        with mock.patch("executors.step5_engine.get_data_dir", return_value=str(data_root)), mock.patch(
            "executors.step5_engine.get_odcr_root", return_value=str(Path(td))
        ):
            df = pd.DataFrame(
                {
                    "user_idx_global": [0],
                    "item_idx_global": [1],
                    "user": ["u1"],
                    "item": ["i1"],
                    "rating": [4.0],
                    "explanation": ["uniquerefalpha uniquerefbeta uniquerefgamma"],
                    "review": [""],
                    "clean_text": ["uniquerefalpha uniquerefbeta uniquerefgamma"],
                    "content_evidence": [""],
                    "style_evidence": [""],
                }
            )
            out = s5._apply_step5_factual_eval_default_controls(
                df,
                split_label="valid",
                auxiliary="Aux",
                target="Target",
                final_eval_config=_eval_cfg(),
                no_ref_evidence_config=no_ref_cfg,
            )
        assert "uniquerefalpha uniquerefbeta uniquerefgamma" not in str(out.loc[0, "content_evidence"])
        assert "user_id" not in str(out.loc[0, "content_evidence"])
        assert "item_id" not in str(out.loc[0, "content_evidence"])


def test_selected_sample_no_ref_cache_row_count_and_max_rows_identity() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\n"
            "u1,i1,1,10,alpha service staff breakfast\n"
            "u2,i2,2,20,beta pizza crust friendly\n"
            "u3,i3,3,30,gamma coffee table clean\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0, 1, 2, 3],
                "domain": ["target", "target", "target", "target"],
                "user_idx_global": [1, 2, 3, 1],
                "item_idx_global": [10, 20, 30, 10],
                "clean_text": ["target one", "target two", "target three", "target four"],
                "effective_epoch": [0, 0, 1, 1],
            }
        )
        active = selected[selected["effective_epoch"] < 1].copy()
        texts, _metas, manifest = build_or_load_no_ref_evidence_for_frame(
            active,
            split_label="train",
            task_id=8,
            auxiliary="Aux",
            target="Target",
            config={"input_protocol": "text_clean_item_only_no_ref", "min_df": 1},
            data_root=data_root,
            cache_root=root / "cache" / "step5_no_ref_history",
        )
        evidence_path = Path(manifest["files"]["train_evidence_selected.parquet"])
        cached = pd.read_parquet(evidence_path)
        assert len(texts) == 2
        assert len(cached) == 2
        assert manifest["row_count"] == 2
        assert "Item evidence:" in texts[0]
        assert "Context:" not in texts[0]
        assert "Yelp restaurant" not in texts[0]
        assert "user_id" not in texts[0]
        assert "item_id" not in texts[0]
        assert "user 1" not in texts[0]
        assert "item 10" not in texts[0]

        smoke, _smoke_meta, smoke_manifest = build_or_load_no_ref_evidence_for_frame(
            active.head(1),
            split_label="valid",
            task_id=8,
            auxiliary="Aux",
            target="Target",
            config={"input_protocol": "text_clean_item_only_no_ref", "min_df": 1},
            data_root=data_root,
            cache_root=root / "cache" / "step5_no_ref_history",
            max_rows=1,
        )
        assert smoke
        assert smoke_manifest["identity"]["frame_identity"]["max_rows"] == 1


def test_compact_route_prompt_formats_are_clean_and_route_specific() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\n"
            "u1,i1,1,10,fresh sushi friendly service reasonable price\n"
            "u1,i2,1,20,quiet patio fast staff\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [1],
                "item_idx_global": [10],
                "clean_text": ["target label"],
            }
        )
        prompts: dict[str, str] = {}
        for protocol in ("text_clean_item_only_no_ref", "text_clean_item_user_no_ref", ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF):
            texts, _metas, _manifest = build_or_load_no_ref_evidence_for_frame(
                selected,
                split_label="train",
                task_id=8,
                auxiliary="Aux",
                target="Target",
                config={"input_protocol": protocol, "min_df": 1},
                data_root=data_root,
                cache_root=root / "cache" / protocol,
            )
            prompts[protocol] = texts[0]
            assert "user_id" not in texts[0]
            assert "item_id" not in texts[0]
            assert "user 1" not in texts[0]
            assert "item 10" not in texts[0]

        assert "Item evidence:" in prompts["text_clean_item_only_no_ref"]
        assert "User preference:" not in prompts["text_clean_item_only_no_ref"]
        assert "User preference:" in prompts["text_clean_item_user_no_ref"]
        assert "Item evidence:" in prompts[ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF]
        assert "route_explainer" not in prompts[ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF]
        assert "Yelp restaurant" not in prompts[ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF]


def test_evidence_v3_filters_generic_singletons_and_keeps_concrete_phrases() -> None:
    phrases = extract_evidence_phrases(
        "good great nice n't food service Yelp restaurant beef sandwich which pain au pain au chocolat "
        "crispy which think makes keeps getting some brunch each part own starbucks good drinks great taste "
        "fresh sushi friendly service sweet potato fries nice fine excellent products drinks late server every time "
        "re open hours both medium drink twice",
        token_limit=64,
    )
    joined = " | ".join(phrases)
    assert "good" not in phrases
    assert "great" not in phrases
    assert "nice" not in phrases
    assert "n't" not in joined
    assert "yelp restaurant" not in joined
    assert "beef sandwich which" not in joined
    assert "pain au |" not in f"{joined} |"
    assert "crispy which" not in joined
    assert "think makes" not in joined
    assert "keeps getting" not in joined
    assert "some brunch" not in joined
    assert "each part" not in joined
    assert "own starbucks" not in joined
    assert "good drinks" not in joined
    assert "great taste" not in joined
    assert "nice fine" not in joined
    assert "excellent products" not in joined
    assert "drinks late" not in joined
    assert "server every time" not in joined
    assert "re open hours" not in joined
    assert "both medium" not in joined
    assert "drink twice" not in joined
    assert any("pain au chocolat" in phrase for phrase in phrases)
    assert any("fresh sushi" in phrase for phrase in phrases)
    assert any("friendly service" in phrase for phrase in phrases)
    assert any("sweet potato fries" in phrase for phrase in phrases)
    rejections = collect_phrase_filter_audit(
        "nice fine excellent products drinks late server every time re open hours both medium drink twice",
        token_limit=64,
        max_examples=80,
    )
    rejected = {item["phrase"]: item["reason"] for item in rejections}
    assert rejected["nice fine"] in {"sentiment_adjacent", "sentiment_only"}
    assert rejected["excellent products"] == "sentiment_generic_head"
    assert rejected["drinks late"] in {"weak_bigram", "generic_event_phrase"}


def test_token_cache_identity_tracks_evidence_v3_phrase_quality_builder() -> None:
    assert "evidence_v3" in NO_REF_HISTORY_CACHE_SCHEMA_VERSION
    assert "evidence_v3" in s5.STEP5_NO_REF_HISTORY_SCHEMA_VERSION
    assert "compact96_anchor" in s5.ODCR_TOKENIZE_CACHE_VERSION


def test_phrase_scorer_prompt_uses_concrete_aspect_phrase_not_context_label() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\n"
            "u1,i1,1,10,good great nice food service fresh sushi friendly service\n"
            "u2,i1,2,10,fresh sushi spicy tuna roll crispy fries\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [3],
                "item_idx_global": [10],
                "clean_text": ["target label"],
            }
        )
        texts, metas, manifest = build_or_load_no_ref_evidence_for_frame(
            selected,
            split_label="train",
            task_id=8,
            auxiliary="Aux",
            target="Target",
            config={"input_protocol": "text_clean_item_only_no_ref", "min_df": 1},
            data_root=data_root,
            cache_root=root / "cache" / "phrase_quality",
        )
    assert "fresh sushi" in texts[0] or "spicy tuna roll" in texts[0]
    assert "Yelp restaurant" not in texts[0]
    assert "Context:" not in texts[0]
    assert "good; great; nice" not in texts[0]
    assert metas[0]["item_evidence_phrases"]
    audit = manifest["history"]["phrase_filter_audit"]
    assert audit["sampled_reason_counts"]
    assert any(item["reason"] for item in audit["examples"])
    assert metas[0]["item_evidence_phrases"][0]["score"] > 0


def test_route_weighted_phrase_selection_does_not_use_current_reference() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\n"
            "u1,i1,1,10,red robin burgers toasted bun crispy fries\n"
            "u2,i1,2,10,red robin burgers friendly staff quick service\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [1],
                "item_idx_global": [10],
                "clean_text": ["UNIQUE_CURRENT_REFERENCE_SHOULD_NOT_APPEAR"],
                "explanation": ["UNIQUE_CURRENT_REFERENCE_SHOULD_NOT_APPEAR"],
                "route_explainer": [1],
                "confidence_bucket": [2],
                "sample_weight_hint": [1.2],
                "sample_origin": ["target_gold"],
            }
        )
        texts, metas, _manifest = build_or_load_no_ref_evidence_for_frame(
            selected,
            split_label="train",
            task_id=8,
            auxiliary="Aux",
            target="Target",
            config={"input_protocol": ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF, "min_df": 1},
            data_root=data_root,
            cache_root=root / "cache" / "route_weighted",
        )
    assert "UNIQUE_CURRENT_REFERENCE_SHOULD_NOT_APPEAR" not in texts[0]
    assert "route_explainer" not in texts[0]
    assert "sample_weight_hint" not in texts[0]
    assert "red robin burgers" in texts[0]
    assert metas[0]["route_weighted"] is True
    assert metas[0]["item_evidence_phrases"][0]["route_confidence_bonus"] > 0
    assert metas[0]["item_evidence_phrases"][0]["target_gold_bonus"] > 0


def test_soft_prefix_route_requires_real_latent_metadata() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\n"
            "u1,i1,1,10,fresh sushi friendly service\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [1],
                "item_idx_global": [10],
                "clean_text": ["fresh sushi"],
            }
        )
        with pytest.raises(ValueError, match="soft_prefix_light_no_ref requires"):
            build_or_load_no_ref_evidence_for_frame(
                selected,
                split_label="train",
                task_id=8,
                auxiliary="Aux",
                target="Target",
                config={"input_protocol": "soft_prefix_light_no_ref", "min_df": 1},
                data_root=data_root,
                cache_root=root / "cache" / "soft_prefix",
            )


def test_retired_singleton_and_contrastive_protocols_fail_fast() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\nu1,i1,1,10,fresh sushi friendly service\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [1],
                "item_idx_global": [10],
                "clean_text": ["fresh sushi"],
            }
        )
        for protocol in ("selected_history_lite_no_ref", "target_domain_phrase_no_ref", "contrastive_evidence_no_ref"):
            with pytest.raises(ValueError, match="retired Step5 no-ref input_protocol"):
                build_or_load_no_ref_evidence_for_frame(
                    selected,
                    split_label="train",
                    task_id=8,
                    auxiliary="Aux",
                    target="Target",
                    config={"input_protocol": protocol, "min_df": 1},
                    data_root=data_root,
                    cache_root=root / "cache" / protocol,
                )


def test_train_current_label_token_cannot_return_via_domain_prior() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\n"
            "u1,i1,1,10,rareanswerkw\n"
            "u2,i2,2,20,service staff breakfast\n"
            "u3,i3,3,30,service patio friendly\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [1],
                "item_idx_global": [10],
                "clean_text": ["rareanswerkw"],
                "route_scorer": [1],
                "route_explainer": [1],
            }
        )
        texts, _metas, manifest = build_or_load_no_ref_evidence_for_frame(
            selected,
            split_label="train",
            task_id=8,
            auxiliary="Aux",
            target="Target",
            config={"input_protocol": "text_clean_item_only_no_ref", "min_df": 2, "domain_prior_display_top_k": 6},
            data_root=data_root,
            cache_root=root / "cache" / "step5_no_ref_history",
        )
        prior = json.loads(Path(manifest["files"]["domain_prior"]).read_text(encoding="utf-8"))
        prior_tokens = {str(x["token"]) for x in prior["domains"]["target"]}
        assert "rareanswerkw" not in prior_tokens
        assert "rareanswerkw" not in texts[0]


def test_compact_evidence_loader_drops_stopword_only_domain_prior() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\n"
            "u1,i1,1,10,the and place really just\n"
            "u2,i2,2,20,the and place really just\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [99],
                "item_idx_global": [99],
                "clean_text": ["target label"],
            }
        )
        texts, metas, _manifest = build_or_load_no_ref_evidence_for_frame(
            selected,
            split_label="train",
            task_id=8,
            auxiliary="Aux",
            target="Target",
            config={"input_protocol": "text_clean_item_user_no_ref", "min_df": 1},
            data_root=data_root,
            cache_root=root / "cache" / "stopwords",
        )
        assert "Item evidence: none." in texts[0]
        assert "Domain prior:" not in texts[0]
        assert metas[0]["domain_prior_terms"] == 0


def test_step5_odcr_native_no_ref_policy_resolves_from_one_control() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=8,
        set_overrides=[],
        dry_run=True,
    )
    final_eval = snapshot["step5_final_eval"]
    assert final_eval["schema_version"] == "odcr_step5_final_eval_config/5_odcr_native_no_ref_k5"
    assert final_eval["official_profile"] == "odcr_no_ref_k5_25"
    assert final_eval["generation_input_policy"] == "history_conditioned_no_reference_evidence"
    assert final_eval["content_evidence_policy"] == "train_only_history"
    assert cfg.step5_train_generation_input_policy == "history_conditioned_no_reference_evidence"
    assert cfg.step5_train_content_evidence_policy == "train_only_history"
    policy = snapshot["step5_no_ref_evidence"]
    assert policy["diagnostic_only"] is False
    assert policy["formal_allowed"] is True
    assert policy["paper_table_allowed"] is True
    assert policy["input_protocol"] == ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF
    assert "text_clean_item_only_no_ref" in policy["allowed_input_protocols"]
    assert ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF in policy["allowed_input_protocols"]
    assert policy["encoder_content_token_budget"] == 96
    assert policy["domain_prior_display_top_k"] == 6
    assert cfg.step5_no_ref_encoder_content_token_budget == 96
    assert json.loads(cfg.step5_no_ref_evidence_config_json)["input_protocol"] == ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF


def test_forcedscale_sanity_metrics_are_not_recorded_as_performance_stop_policy() -> None:
    audit = (ROOT / "AI_analysis/03_evidence_ledgers/step5_task8_forcedscale_initial_audit.md").read_text(
        encoding="utf-8"
    )
    assert "low 512/2048 metrics must not stop 50k/100k/1-effective-epoch" in audit


def test_step5_replay_config_preserves_no_ref_evidence_policy() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        meta = root / "runs" / "step5" / "task8" / "1_99" / "meta"
        meta.mkdir(parents=True)
        resolved = meta / "resolved_config.json"
        target_only_candidate = (
            "STEP5_RATIO_TARGET_GOLD_ONLY+WEAK_CROSS_PLATFORM_LOW_WEIGHTED_CF_V1+"
            "TG_MIX_WEAK_MEDIUM_ONLY+AG_MIX_WEAK_MEDIUM_ONLY+LR_1e-3+W0"
        )
        resolved.write_text(
            json.dumps(
                {
                    "step5_selected_tuning_candidate": target_only_candidate,
                    "step5_no_ref_evidence_config_json": json.dumps(
                        {
                            "schema_version": "odcr_step5_no_ref_evidence_config/1",
                            "input_protocol": "neutral_core_no_ref",
                            "allowed_input_protocols": ["neutral_core_no_ref", "selected_history_lite_no_ref"],
                            "min_df": 3,
                        },
                        sort_keys=True,
                    ),
                    "step5_sampler_config_json": json.dumps(
                        {
                            "explanation": {
                                "target_gold_ratio": 1.0,
                                "aux_gold_ratio": 0.0,
                                "cf_ratio": 0.0,
                                "target_gold_tier_mix": {"high": 0.0, "medium": 1.0},
                                "aux_gold_tier_mix": {"high": 0.0, "medium": 1.0},
                                "cf_tier_mix": {"high": 0.0, "medium": 0.0, "low_weighted": 1.0},
                            }
                        },
                        sort_keys=True,
                    ),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (meta / "run_summary.json").write_text(
            json.dumps(
                {
                    "command": (
                        "./odcr step5 --task 8 --run-id 1_99 "
                        "--set step5.no_ref_evidence.input_protocol=neutral_core_no_ref "
                        f"--set step5.tasks.8.tuning.selected_tuning_candidate={target_only_candidate} "
                        "--set step5.sampler.explanation.target_gold_ratio=1.0"
                    ),
                    "resolved_config_path": str(resolved.relative_to(root)),
                    "selected_tuning_candidate": target_only_candidate,
                    "step5_effective_samples": {"explanation": 10000},
                    "step5_optimizer_steps": {"explanation": 156},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        args = mock.Mock(task=8, from_step5="1_99", from_step5_run=None)
        with mock.patch.object(odcr_cli, "REPO_ROOT", root):
            replay_sets, manifest = odcr_cli._step5_run_summary_replay_sets(args)
    assert "step5.no_ref_evidence.input_protocol=neutral_core_no_ref" in replay_sets
    assert f"step5.tasks.8.tuning.selected_tuning_candidate={target_only_candidate}" in replay_sets
    assert "step5.sampler.explanation.target_gold_ratio=1.0" in replay_sets
    assert "step5.sampler.explanation.aux_gold_ratio=0.0" in replay_sets
    assert "step5.sampler.explanation.cf_ratio=0.0" in replay_sets
    assert any(item.startswith("step5.no_ref_evidence.allowed_input_protocols=") for item in replay_sets)
    assert manifest["schema_version"] == "odcr_step5_run_config_replay/2_no_ref_evidence"


def test_step5_target_gold_only_sampler_override_resolves_for_pilot() -> None:
    cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=8,
        set_overrides=[
            "step5.tasks.8.tuning.selected_tuning_candidate=STEP5_RATIO_TARGET_GOLD_ONLY+WEAK_CROSS_PLATFORM_LOW_WEIGHTED_CF_V1+TG_MIX_WEAK_MEDIUM_ONLY+AG_MIX_WEAK_MEDIUM_ONLY+LR_5e-4+W0",
            "step5.tasks.8.sampler.explanation.target_gold_ratio=1.0",
            "step5.tasks.8.sampler.explanation.aux_gold_ratio=0.0",
            "step5.tasks.8.sampler.explanation.cf_ratio=0.0",
            "step5.tasks.8.sampler.explanation.target_gold_tier_mix.high=0.0",
            "step5.tasks.8.sampler.explanation.target_gold_tier_mix.medium=1.0",
            "step5.tasks.8.sampler.explanation.aux_gold_tier_mix.high=0.0",
            "step5.tasks.8.sampler.explanation.aux_gold_tier_mix.medium=1.0",
            "step5.tasks.8.sampler.explanation.cf_tier_mix.high=0.0",
            "step5.tasks.8.sampler.explanation.cf_tier_mix.medium=0.0",
            "step5.tasks.8.sampler.explanation.cf_tier_mix.low_weighted=1.0",
        ],
        dry_run=True,
    )
    sampler = json.loads(cfg.step5_sampler_config_json)
    assert sampler["explanation"]["target_gold_ratio"] == 1.0
    assert sampler["explanation"]["aux_gold_ratio"] == 0.0
    assert sampler["explanation"]["cf_ratio"] == 0.0
    assert snapshot["step5_sampler"]["explanation"]["target_gold_ratio"] == 1.0


def test_missing_history_uses_neutral_no_reference_fallback() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "data"
        (data_root / "Target").mkdir(parents=True)
        (data_root / "Target" / "train.csv").write_text(
            "user,item,user_idx,item_idx,explanation\nu1,i1,1,10,the and just really\n",
            encoding="utf-8",
        )
        selected = pd.DataFrame(
            {
                "sample_id": [0],
                "domain": ["target"],
                "user_idx_global": [99],
                "item_idx_global": [99],
                "clean_text": ["target label"],
            }
        )
        texts, metas, _manifest = build_or_load_no_ref_evidence_for_frame(
            selected,
            split_label="valid",
            task_id=8,
            auxiliary="Aux",
            target="Target",
            config={"input_protocol": "text_clean_item_only_no_ref", "min_df": 1},
            data_root=data_root,
            cache_root=root / "cache" / "missing",
        )
    assert texts[0].startswith("Item evidence: none.")
    assert metas[0]["content_evidence_source"] == "neutral_no_train_history"


def test_no_fake_metrics_static_contract() -> None:
    text = (CODE_DIR / "executors" / "step5_engine.py").read_text(encoding="utf-8")
    assert "fake_metrics" not in text
    assert "candidate_paper_metrics_available\": _paper_table_allowed" in text
    assert "_metrics_final_dict_from_rows" in text


def test_old_oracle_rebuild_tool_fails_fast() -> None:
    with mock.patch.object(sys, "argv", ["odcr_rebuild_step5_post_train_eval.py"]):
        with pytest.raises(SystemExit, match="old post_train_eval protocol"):
            odcr_rebuild_step5_post_train_eval.main()
