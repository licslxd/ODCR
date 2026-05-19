from __future__ import annotations

import json

from odcr_core.step5_runtime_probe import candidate_decision_from_result, _validate_existing_probe_result


def _formal_entry_result(**overrides):
    payload = {
        "schema_version": "odcr_step5_e4_bounded_probe/1",
        "stage": "step5A",
        "task_id": 2,
        "candidate_id": "B224-unit",
        "success": True,
        "evidence_level": "E4_gpu_shard_forward_bounded_formal_entry_with_validation",
        "formal_entry_lifecycle": True,
        "forward_executed": True,
        "loss_backward_executed": True,
        "optimizer_step_executed": True,
        "preflight_executed": True,
        "scratch_cleanup_status": "pass",
        "graph_tensor_audit_status": "pass",
        "graph_scratch_before_ema": [],
        "ema_init_pass": True,
        "ema_init_executed_in_E4": True,
        "ddp_wrap_pass": True,
        "first_train_step_pass": True,
        "validation_pass_executed": True,
        "validation_forward_pass": True,
        "validation_loss_finite": True,
        "validation_oom": False,
        "step5A_validation_scorer_only": True,
        "flan_explainer_called_in_step5A_validation": False,
        "out_logits_materialized_in_step5A_validation": False,
        "valid_forward_micro_batch_size": 192,
        "train_per_gpu_batch_size": 224,
        "all_trainable_grad_status": "pass",
        "trainable_param_count": 4,
        "grad_present_count": 4,
        "lora_trainable_count": 2,
        "lora_grad_present_count": 2,
        "missing_grad_params": [],
        "real_forward_backward_executed": True,
        "finite_loss_sync_ok": True,
        "graph_safe_backward_ok": True,
        "rank_sample_balance_ok": True,
        "loss_component_keys_per_rank_identical": True,
        "formal_namespace_pollution": False,
        "latest_json_created": False,
        "checkpoint_written": False,
        "long_window": {"steps_executed_max": 1, "rank_results": [{"steps_executed": 1}]},
        "memory_truth": {"oom": False, "reserved_is_diagnostic_only": True},
    }
    payload.update(overrides)
    return payload


def test_candidate_decision_requires_ema_and_first_train_step() -> None:
    assert candidate_decision_from_result(_formal_entry_result())["correctness_pass"] is True
    assert candidate_decision_from_result(_formal_entry_result(ema_init_pass=False))["correctness_pass"] is False
    assert candidate_decision_from_result(_formal_entry_result(first_train_step_pass=False))["correctness_pass"] is False
    assert candidate_decision_from_result(_formal_entry_result(validation_pass_executed=False))["correctness_pass"] is False
    assert candidate_decision_from_result(_formal_entry_result(out_logits_materialized_in_step5A_validation=True))[
        "correctness_pass"
    ] is False
    assert (
        candidate_decision_from_result(_formal_entry_result(graph_scratch_before_ema=[{"path": "m._last"}]))[
            "correctness_pass"
        ]
        is False
    )


def test_existing_result_reuse_requires_formal_entry_lifecycle(tmp_path) -> None:  # noqa: ANN001
    out = tmp_path / "probe"
    out.mkdir()
    (out / "resolved_config.json").write_text("{}", encoding="utf-8")
    (out / "source_table.json").write_text("{}", encoding="utf-8")
    (out / "sample_plan" / "sample_plan_manifest.json").parent.mkdir()
    (out / "sample_plan" / "sample_plan_manifest.json").write_text("{}", encoding="utf-8")
    (out / "bounded_token_cache").mkdir()
    (out / "bounded_token_cache" / "rank0_manifest.json").write_text("{}", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(_formal_entry_result()), encoding="utf-8")

    valid = _validate_existing_probe_result(
        out / "result.json",
        stage="step5A",
        task=2,
        candidate_id="B224-unit",
        min_steps=1,
    )
    assert valid["valid"] is True

    (out / "result.json").write_text(json.dumps(_formal_entry_result(ema_init_executed_in_E4=False)), encoding="utf-8")
    invalid = _validate_existing_probe_result(
        out / "result.json",
        stage="step5A",
        task=2,
        candidate_id="B224-unit",
        min_steps=1,
    )
    assert invalid["valid"] is False
    assert "ema_init_failed" in invalid["reasons"]
