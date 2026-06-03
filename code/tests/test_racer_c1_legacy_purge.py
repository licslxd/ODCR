from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_old_step5_generator_files_are_physically_removed() -> None:
    deleted = [
        "code/executors/step5_engine.py",
        "code/executors/step5_entry.py",
        "code/odcr_core/step5_explanation_flan_bridge.py",
        "code/odcr_core/step5_native_lora.py",
        "code/odcr_core/step5_no_ref_history_cache.py",
        "code/odcr_core/step5_prompt_templates.py",
        "code/odcr_core/step5_runtime_probe.py",
        "code/odcr_core/generation/decoder_kv.py",
        "code/odcr_core/generation/cache_types.py",
        "code/odcr_core/ablation/probe.py",
        "code/odcr_core/bleu_runtime.py",
        "code/odcr_core/mainline_monitor.py",
        "code/odcr_core/step5_auto_budget.py",
        "code/odcr_core/step5_code1_text_eval.py",
        "code/odcr_core/step5_eval_summary.py",
        "code/odcr_core/step5_explanation_control_projection.py",
        "code/odcr_core/step5_grad_contract.py",
        "code/odcr_core/step5_innovation.py",
        "code/odcr_core/step5_word_losses.py",
        "code/tools/compact_step5_post_train_eval.py",
        "code/tools/odcr_rebuild_step5_post_train_eval.py",
        "code/tests/test_full_bleu_monitor_decode.py",
        "code/tests/test_bleu_quick_collate.py",
        "code/tests/test_index_contract.py",
        "code/tests/test_step3_no_accum_training_loop.py",
    ]
    assert [rel for rel in deleted if (ROOT / rel).exists()] == []


def test_step5_executor_is_fail_fast_only() -> None:
    text = (ROOT / "code" / "executors" / "step5.py").read_text(encoding="utf-8")
    assert "Old Step5 generator/eval/rerank code has been deleted" in text
    assert "step5_entry" not in text
    assert "step5_engine" not in text


def test_racer_c1_policy_has_no_baseline_language() -> None:
    cfg = (ROOT / "configs" / "odcr.yaml").read_text(encoding="utf-8")
    assert "legacy_generator_policy: deleted_not_available" in cfg
    assert "baseline_only_not_invoked" not in cfg


def test_racer_c1_evidence_pool_encodes_content_style_split() -> None:
    text = (ROOT / "code" / "odcr_core" / "racer_c1" / "evidence_pool.py").read_text(encoding="utf-8")
    for needle in (
        "causal_content_evidence",
        "style_shortcut_evidence",
        "cf_content_anchor",
        "cf_reliability_bucket",
        "contrastive_role_hint",
    ):
        assert needle in text
