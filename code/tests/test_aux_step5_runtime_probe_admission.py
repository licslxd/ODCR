from __future__ import annotations

import json
from types import SimpleNamespace

from odcr_core.aux.runtime.bounded_probe import build_probe_payload
from odcr_core.aux.runtime.command_registry import get_registry
from odcr_core.step5_runtime_probe import (
    baseline_candidate_from_config,
    candidate_decision_from_result,
    candidate_overrides,
    expand_scan_candidates,
    rank_batch_candidates,
    validate_memory_truth_schema,
    _worker_budget,
    _finalize_per_tier_loss,
    _new_per_tier_accumulator,
    _per_tier_loss_keys,
    _patch_candidate_resolution_contract,
    _prelaunch_compute_app_guard,
    _classify_step5_runtime_probe_failure,
    _ddp_ready_hook_policy_payload,
    _resolve_candidate,
    _summarize_runtime_transformers_signatures,
    _validate_existing_probe_result,
)


def _memory_truth(**overrides):
    payload = {
        "device_total_gb": 79.0,
        "max_memory_allocated_gb": 30.0,
        "max_memory_reserved_gb": 76.0,
        "reserved_minus_allocated_gb": 46.0,
        "allocated_to_total_ratio": 30.0 / 79.0,
        "reserved_to_total_ratio": 76.0 / 79.0,
        "nvidia_smi_process_used_gb": 32.0,
        "param_memory_gb": 12.0,
        "trainable_param_memory_gb": 0.5,
        "frozen_param_memory_gb": 11.5,
        "grad_memory_gb": 0.5,
        "optimizer_state_memory_gb": 1.0,
        "activation_peak_estimated_gb": 16.5,
        "fragmentation_hint": {"diagnostic_only": True},
        "memory_creep_detected": False,
        "oom": False,
        "oom_error_message": "",
        "cuda_allocator_backend": "native",
        "torch_cuda_alloc_conf": "",
        "reserved_is_diagnostic_only": True,
    }
    payload.update(overrides)
    return payload


def _probe_result(**overrides):
    payload = {
        "schema_version": "odcr_step5_e4_bounded_probe/1",
        "success": True,
        "evidence_level": "E4_gpu_shard_forward_bounded_formal_entry_with_validation",
        "formal_entry_lifecycle": True,
        "actual_gpu_forward_executed": True,
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
        "real_forward_backward_executed": True,
        "real_task_data_used": True,
        "real_ccv_packet_used": True,
        "all_trainable_grad_status": "pass",
        "trainable_param_count": 4,
        "grad_present_count": 4,
        "lora_trainable_count": 2,
        "lora_grad_present_count": 2,
        "missing_grad_params": [],
        "memory_truth": _memory_truth(),
        "candidate_decision": {"reserved_memory_used_for_rejection": False},
        "formal_namespace_pollution": False,
        "latest_json_created": False,
        "checkpoint_written": False,
    }
    payload.update(overrides)
    return payload


def test_step5a_step5b_probes_registered() -> None:
    assert get_registry().get("probe.step5A.bounded") is not None
    assert get_registry().get("probe.step5B.bounded") is not None


def test_step5_runtime_probe_resolves_candidate_with_stage_head(monkeypatch) -> None:  # noqa: ANN001
    import odcr_core.step5_runtime_probe as probe_mod

    captured = {}

    def fake_resolve_config(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return SimpleNamespace(step5_head=kwargs["step5_head"]), {}, {"field_sources": {}}

    monkeypatch.setattr(probe_mod, "resolve_config", fake_resolve_config)
    monkeypatch.setattr(probe_mod, "build_formal_source_table_snapshot", lambda _snapshot: {"records": []})
    cfg, _snapshot, _source = _resolve_candidate(
        stage="step5A",
        task=2,
        config_path="configs/odcr.yaml",
        set_overrides=[],
        from_step4="1",
    )
    assert captured["step5_head"] == "step5A"
    assert cfg.step5_head == "step5A"


def test_step5_runtime_probe_patches_candidate_contract_artifacts(tmp_path) -> None:  # noqa: ANN001
    resolved_path = tmp_path / "resolved_config.json"
    source_path = tmp_path / "source_table.json"
    resolved_path.write_text("{}", encoding="utf-8")
    source_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    result = _probe_result(
        lora_target_policy_id="step5_head_aware_lora_allowlist/1",
        head_specific_lora_allowlist_id="step5_head_aware_lora_allowlist/1:step5A",
        final_lora_target_modules=["domain_gate"],
        forbidden_lora_targets=["domain_cross_attn.out_proj"],
        deleted_legacy_modules=["recommender", "flan_soft_prompt_stack", "hidden2token"],
        head_specific_trainable_policy="step5_head_specific_trainable_contract/1:step5A",
        head_gated_loss_contract={"head": "step5A"},
        all_trainable_grad={"status": "pass", "evidence_context": {"evidence_id": "e4-id"}},
        trainable_parameter_names_hash="trainable-hash",
    )

    _patch_candidate_resolution_contract(
        resolution_paths={
            "resolved_config_path": str(resolved_path),
            "source_table_path": str(source_path),
        },
        result=result,
    )

    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert resolved["final_lora_target_modules"] == ["domain_gate"]
    assert resolved["forbidden_lora_targets"] == ["domain_cross_attn.out_proj"]
    assert resolved["runtime_e4_evidence_id"] == "e4-id"
    assert resolved["scratch_cleanup_status"] == "pass"
    assert resolved["graph_tensor_audit_status"] == "pass"
    assert resolved["ema_init_status"] == "pass"
    assert resolved["formal_entry_E4_evidence_id"] == "e4-id"
    assert resolved["first_train_step_evidence_id"] == "e4-id"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    records = {item["key"]: item["value"] for item in source["records"]}
    assert records["final_lora_target_modules_hash"]
    assert records["forbidden_lora_targets"] == ["domain_cross_attn.out_proj"]
    assert records["runtime_e4_evidence_id"] == "e4-id"
    assert records["scratch_cleanup_status"] == "pass"
    assert records["graph_tensor_audit_status"] == "pass"
    assert records["ema_init_status"] == "pass"
    assert records["formal_entry_E4_evidence_id"] == "e4-id"
    assert records["first_train_step_evidence_id"] == "e4-id"
    assert records["synthetic_used"] is False


def test_step5_probe_requires_successful_cuda_handshake() -> None:
    failed = build_probe_payload("step5A", 2, handshake={"torch.cuda.is_available": False, "torch.cuda.device_count": 0})
    assert not failed["success"]
    handshake_only = build_probe_payload(
        "step5B",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 1,
        },
        target_source="current_gpu_pane_handoff",
    )
    assert not handshake_only["success"]
    assert handshake_only["evidence_level"] == "E3_gpu_transport"


def test_old_e4_without_formal_entry_lifecycle_cannot_allow_step5_gate() -> None:
    payload = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(
            evidence_level="E4_gpu_shard_forward_bounded",
            formal_entry_lifecycle=False,
            ema_init_pass=False,
            first_train_step_pass=False,
        ),
        target_source="current_gpu_pane_handoff",
    )
    assert payload["success"] is False
    assert payload["evidence_level"] == "E3_gpu_transport"
    assert payload["formal_entry_lifecycle_e4_required"] is True
    assert payload["formal_entry_validation_e4_required"] is True


def test_step5_probe_requires_forward_backward_for_e4() -> None:
    forward_only = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result={
            "success": True,
            "evidence_level": "E3_gpu_transport",
            "actual_gpu_forward_executed": True,
            "real_forward_backward_executed": False,
            "memory_truth": _memory_truth(),
            "candidate_decision": {"reserved_memory_used_for_rejection": False},
        },
        target_source="current_gpu_pane_handoff",
    )
    assert not forward_only["success"]
    real_e4 = build_probe_payload(
        "step5B",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(),
        target_source="current_gpu_pane_handoff",
    )
    assert real_e4["success"]
    assert real_e4["evidence_level"] == "E4_gpu_shard_forward_bounded_formal_entry_with_validation"


def test_step5_probe_rejects_e4_without_validation_pass() -> None:
    payload = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(validation_pass_executed=False),
        target_source="current_gpu_pane_handoff",
    )
    assert payload["success"] is False
    assert payload["evidence_level"] == "E3_gpu_transport"


def test_step5_artifact_build_preflight_admits_true_ddp_artifact_path() -> None:
    payload = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result={
            "schema_version": "odcr_step5_e4_bounded_probe/1",
            "success": True,
            "evidence_level": "E4_gpu_shard_forward_bounded",
            "artifact_build_only": True,
            "artifact_build_preflight_pass": True,
            "ddp_world_size": 2,
            "rank_ids": [0, 1],
            "rank0_rank1_cache_fingerprint_match": True,
            "rank1_used_rank0_cache_dir": True,
            "missing_dataset_absent": True,
            "token_cache_lineage_success": True,
            "index_contract_audit_pass": True,
            "formal_namespace_pollution": False,
            "latest_json_created": False,
            "checkpoint_written": False,
        },
        target_source="current_gpu_pane_handoff",
    )
    assert payload["artifact_build_preflight_admitted"] is True
    assert payload["backward_validation_claimed"] is False
    assert payload["artifact_build_only_does_not_validate_backward"] is True
    assert payload["current_real_batch_backward_e4_required"] is True
    assert payload["evidence_level"] == "E3_gpu_transport"
    assert payload["formal_training"] == "not run"
    assert payload["success"] is False


def test_bounded_probe_success_does_not_imply_formal_namespace_write() -> None:
    payload = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(),
        target_source="cli_explicit",
    )
    assert payload["success"] is True
    assert payload["writes_formal_runs"] is False
    assert payload["formal_training"] == "not run"
    assert payload["probe_result"]["latest_json_created"] is False
    assert payload["probe_result"]["checkpoint_written"] is False


def test_old_or_partial_e4_cannot_substitute_current_real_batch_backward() -> None:
    payload = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(loss_backward_executed=False),
        target_source="current_gpu_pane_handoff",
    )
    assert payload["success"] is False
    assert payload["evidence_level"] == "E3_gpu_transport"


def test_step5_e4_admission_requires_all_trainable_grad_pass() -> None:
    payload = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(
            all_trainable_grad_status="fail",
            grad_present_count=3,
            missing_grad_params=["domain_gate.lora_A"],
        ),
        target_source="current_gpu_pane_handoff",
    )
    assert payload["success"] is False
    assert payload["evidence_level"] == "E3_gpu_transport"


def test_e3_transport_and_flagged_fake_batch_cannot_allow_step5a_formal_gate() -> None:
    e3_only = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        target_source="current_gpu_pane_handoff",
    )
    assert e3_only["success"] is False
    assert e3_only["evidence_level"] == "E3_gpu_transport"
    fake_batch = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(synthetic_batch_used_for_formal_gate=True),
        target_source="current_gpu_pane_handoff",
    )
    assert fake_batch["success"] is False
    assert fake_batch["backward_validation_claimed"] is False


def test_step5_gate_requires_fresh_handoff_or_cli_source() -> None:
    missing = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "gpu03",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "torch.cuda.is_available": True,
            "torch.cuda.device_count": 2,
        },
        probe_result=_probe_result(),
    )
    assert missing["success"] is False
    assert missing["fresh_gpu_pane_handoff_required"] is True
    assert missing["target_source_allowed_for_step5_gate"] is False
    admin = build_probe_payload(
        "step5A",
        2,
        handshake={
            "hostname": "admin",
            "CUDA_VISIBLE_DEVICES": "",
            "torch.cuda.is_available": False,
            "torch.cuda.device_count": 0,
        },
        probe_result=_probe_result(),
        target_source="current_gpu_pane_handoff",
    )
    assert admin["success"] is False
    assert admin["validated_hostname_non_admin"] is False


def test_step5_backward_preflight_source_executes_backward_and_optimizer_step() -> None:
    import inspect
    import odcr_core.step5_runtime_probe as step5_runtime_probe

    src = inspect.getsource(step5_runtime_probe._run_rank_probe)
    assert "total_loss.backward()" in src
    assert "optimizer.step()" in src
    assert "loss_backward_executed" in src
    assert "optimizer_step_executed" in src
    assert "real_forward_backward_executed" in src
    assert "latest_json_created" in src
    assert "checkpoint_written" in src


def test_runtime_signature_summary_requires_rank_match_and_kwargs_support() -> None:
    summary = _summarize_runtime_transformers_signatures(
        [
            {
                "runtime_transformers_signature": {
                    "rank": 0,
                    "transformers_version": "4.57.6",
                    "torch_version": "2.5.1+cu121",
                    "transformers_module_path": "/env/site-packages/transformers/__init__.py",
                    "gradient_checkpointing_kwargs_supported": True,
                    "use_reentrant_false_supported": True,
                    "torch_and_transformers_same_conda_env": True,
                }
            },
            {
                "runtime_transformers_signature": {
                    "rank": 1,
                    "transformers_version": "4.57.6",
                    "torch_version": "2.5.1+cu121",
                    "transformers_module_path": "/env/site-packages/transformers/__init__.py",
                    "gradient_checkpointing_kwargs_supported": True,
                    "use_reentrant_false_supported": True,
                    "torch_and_transformers_same_conda_env": True,
                }
            },
        ]
    )
    assert summary["all_ranks_transformers_version_match"] is True
    assert summary["gradient_checkpointing_kwargs_supported"] is True
    assert summary["use_reentrant_false_supported"] is True
    assert summary["torch_and_transformers_same_conda_env"] is True


def test_ddp_ready_hook_policy_payload_records_effective_values() -> None:
    payload = _ddp_ready_hook_policy_payload(
        SimpleNamespace(ddp_find_unused_parameters=False, ddp_static_graph=False)
    )
    assert payload["ddp_find_unused_parameters_effective"] is False
    assert payload["ddp_static_graph_effective"] is False
    assert payload["ddp_static_graph_reason"]
    assert payload["ddp_ready_hook_policy"] == "find_unused_false_static_graph_false_non_reentrant_checkpointing_required"


def test_ready_hook_runtime_classifier_maps_required_fields() -> None:
    payload = _classify_step5_runtime_probe_failure(
        "RuntimeError: Expected to mark a variable ready only once. "
        "Parameter at index 876 with name "
        "flan_explainer.decoder.block.23.layer.2.DenseReluDense.wo.lora_B has been marked as ready twice."
    )
    assert payload["failure_phase"] == "train_backward"
    assert payload["failure_type"] == "ddp_parameter_ready_twice"
    assert payload["root_cause"] == "ddp_lora_checkpointing_ready_hook_conflict"
    assert payload["parameter_name"] == "flan_explainer.decoder.block.23.layer.2.DenseReluDense.wo.lora_B"


def test_gradient_checkpointing_policy_unsupported_classifier() -> None:
    payload = _classify_step5_runtime_probe_failure(
        "transformers_runtime_api_mismatch: gradient_checkpointing_kwargs is unsupported"
    )
    assert payload["failure_phase"] == "model_init"
    assert payload["failure_type"] == "gradient_checkpointing_policy_unsupported"
    assert payload["root_cause"] == "transformers_runtime_api_mismatch"


def test_ccv_shape_classifier_does_not_report_tokenization_cache() -> None:
    payload = _classify_step5_runtime_probe_failure(
        "RuntimeError: CCV control ids must be [B,T], got (4,)"
    )
    assert payload["failure_phase"] == "data_collate"
    assert payload["failure_type"] == "ccv_control_packet_shape_contract"
    assert payload["root_cause"] == "real_batch_control_packet_shape_invalid"


def test_step5_candidate_overrides_are_one_control_keys() -> None:
    overrides = candidate_overrides(
        hardware_profile="default",
        per_gpu_batch_size=24,
        global_batch_size=48,
        workers_per_rank=5,
        prefetch_factor=4,
        bounded_rows=4096,
        chunk_rows=100000,
    )
    assert "step5.train.per_gpu_batch_size=24" in overrides
    assert "step5.train.batch_size=48" in overrides
    assert "hardware.profiles.default.dataloader_num_workers_train=5" in overrides
    assert "hardware.profiles.default.dataloader_prefetch_factor_train=4" in overrides
    assert "step5.export_loader.bounded_max_rows=4096" in overrides
    assert "step5.export_loader.chunk_rows=100000" in overrides


def test_step5_e4_cpu_worker_budget_rejects_oversubscription() -> None:
    cfg = SimpleNamespace(
        ddp_world_size=2,
        hardware_profile_json=json.dumps(
            {
                "dataloader_num_workers_train": 6,
                "max_parallel_cpu": 12,
                "worker_budget_formula": {"reserved_cpu": 2},
            }
        ),
    )
    budget = _worker_budget(cfg)
    assert not budget["ok"]
    assert budget["formula"] == "6 * 2 + 2 <= 12"


def test_reserved_memory_is_diagnostic_only_and_cannot_reject() -> None:
    result = _probe_result(memory_truth=_memory_truth(max_memory_reserved_gb=78.0, reserved_to_total_ratio=0.987))
    decision = candidate_decision_from_result(result, {"reject_on_oom": True, "reject_on_allocated_ratio": 0.92})
    assert decision["reject_reason"] is None
    assert decision["reserved_memory_used_for_rejection"] is False


def test_oom_still_rejects_candidate() -> None:
    result = _probe_result(memory_truth=_memory_truth(oom=True, oom_error_message="CUDA out of memory"))
    decision = candidate_decision_from_result(result, {"reject_on_oom": True, "reject_on_allocated_ratio": 0.92})
    assert decision["reject_reason"] == "oom"
    assert decision["reject_reason_category"] == "oom"


def test_allocated_ratio_rejects_when_configured() -> None:
    result = _probe_result(memory_truth=_memory_truth(max_memory_allocated_gb=75.0, allocated_to_total_ratio=0.95))
    decision = candidate_decision_from_result(result, {"reject_on_oom": True, "reject_on_allocated_ratio": 0.92})
    assert decision["reject_reason"] == "allocated_memory_ratio_exceeds_configured_limit"
    assert decision["reserved_memory_used_for_rejection"] is False


def test_a4_is_not_skipped_when_a3_reserved_is_high() -> None:
    cfg = {
        "batch_candidates": [
            {"id": "A3", "per_gpu_batch_size": 48, "global_batch_size": 96},
            {"id": "A4", "per_gpu_batch_size": 64, "global_batch_size": 128},
        ],
        "dataloader_candidates": [{"id": "C2", "workers_per_rank": 4, "prefetch_factor": 4}],
        "row_candidates": [{"id": "R0", "bounded_rows": 1024, "chunk_rows": 100000}],
    }
    candidates = expand_scan_candidates(cfg)
    assert [item["candidate_id"] for item in candidates] == ["A3_C2_R0", "A4_C2_R0"]


def test_candidate_id_selects_requested_batch_loader_and_rows_for_probe_child() -> None:
    cfg = {
        "batch_candidates": [
            {"id": "B96", "per_gpu_batch_size": 96, "global_batch_size": 192},
            {"id": "B224", "per_gpu_batch_size": 224, "global_batch_size": 448},
        ],
        "dataloader_candidates": [
            {"id": "C0", "workers_per_rank": 4, "prefetch_factor": 2},
            {"id": "C2", "workers_per_rank": 4, "prefetch_factor": 4},
        ],
        "row_candidates": [
            {"id": "R0", "bounded_rows": 1024, "chunk_rows": 100000},
            {"id": "R2", "bounded_rows": 16384, "chunk_rows": 100000},
        ],
    }
    candidate = baseline_candidate_from_config(
        cfg,
        candidate_id="B224_C2_R0_real_batch_confirmed_pane",
    )
    assert candidate["candidate_id"] == "B224_C2_R0"
    assert "step5.train.per_gpu_batch_size=224" in candidate["overrides"]
    assert "step5.train.batch_size=448" in candidate["overrides"]
    assert "hardware.profiles.default.dataloader_prefetch_factor_train=4" in candidate["overrides"]


def test_memory_truth_schema_complete_and_ranking_prefers_throughput_after_correctness() -> None:
    validate_memory_truth_schema(_memory_truth())
    high_reserved = {
        "candidate_id": "B48",
        "per_gpu_batch_size": 48,
        "candidate_decision": {"reject_reason": None, "selected_score": 1.0},
        "throughput_samples_per_sec": 100.0,
        "data_wait_ratio": 0.01,
        "max_reserved_gb": 78.0,
    }
    low_reserved = {
        "candidate_id": "B32",
        "per_gpu_batch_size": 32,
        "candidate_decision": {"reject_reason": None, "selected_score": 10.0},
        "throughput_samples_per_sec": 120.0,
        "data_wait_ratio": 0.01,
        "max_reserved_gb": 30.0,
    }
    ranked = rank_batch_candidates([low_reserved, high_reserved])
    assert ranked[0]["candidate_id"] == "B32"
    assert ranked[1]["candidate_id"] == "B48"


def test_missing_per_tier_loss_emits_graph_tied_zero_keys() -> None:
    report = _finalize_per_tier_loss(_new_per_tier_accumulator())
    assert report["all_tiers_emitted"] is True
    assert report["tiers"]["target_gold_high"]["tier_count"] == 0
    assert report["tiers"]["target_gold_high"]["metrics"]["scorer_main_loss_raw"]["zero_kind"] == "graph_tied_zero"
    assert "cf_low_weighted.fca_weighted_loss" in _per_tier_loss_keys(report)


def test_existing_probe_result_reuse_requires_floor_artifacts_and_per_tier(tmp_path) -> None:
    out = tmp_path / "candidate"
    (out / "sample_plan").mkdir(parents=True)
    (out / "bounded_token_cache").mkdir()
    (out / "resolved_config.json").write_text("{}", encoding="utf-8")
    (out / "source_table.json").write_text("{}", encoding="utf-8")
    (out / "candidate.json").write_text("{}", encoding="utf-8")
    (out / "request.json").write_text("{}", encoding="utf-8")
    (out / "sample_plan" / "sample_plan_manifest.json").write_text('{"plan_hash":"abc"}', encoding="utf-8")
    (out / "bounded_token_cache" / "rank0_manifest.json").write_text("{}", encoding="utf-8")
    result = _probe_result(
        stage="step5B",
        task_id=2,
        candidate_id="CAND_LW1000",
        long_window={"steps_executed_max": 1000, "rank_results": [{"steps_executed": 1000}]},
        finite_loss_sync_ok=True,
        graph_safe_backward_ok=True,
        rank_sample_balance_ok=True,
        loss_component_keys_per_rank_identical=True,
        per_tier_loss=_finalize_per_tier_loss(_new_per_tier_accumulator()),
        per_tier_loss_keys_per_rank_identical=True,
    )
    (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    validation = _validate_existing_probe_result(
        out / "result.json",
        stage="step5B",
        task=2,
        candidate_id="CAND_LW1000",
        min_steps=1000,
        require_per_tier=True,
    )
    assert validation["valid"] is True
    assert validation["reused"] is True
    result.pop("per_tier_loss")
    (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    validation = _validate_existing_probe_result(
        out / "result.json",
        stage="step5B",
        task=2,
        candidate_id="CAND_LW1000",
        min_steps=1000,
        require_per_tier=True,
    )
    assert validation["valid"] is False
    assert "per_tier_loss_failed" in validation["reasons"]


def test_compute_app_guard_blocks_unowned_duplicate_launch(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "odcr_core.step5_runtime_probe._nvidia_smi_compute_apps",
        lambda: {"available": True, "rows": [{"pid": 999999, "process_name": "python", "used_memory_mib": 100.0}], "stdout": "999999, python, 100"},
    )
    monkeypatch.setattr("odcr_core.step5_runtime_probe._proc_cmdline", lambda pid: "python unrelated.py")
    guard = _prelaunch_compute_app_guard(candidate_id="CAND", out_dir=tmp_path)
    assert guard["pass"] is False
    assert guard["blocked"] is True
    assert guard["duplicate_launch_prevented"] is True
