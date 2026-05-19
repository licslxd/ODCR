from __future__ import annotations

import json
from types import SimpleNamespace

from executors.step5_engine import _patch_step5_runtime_contract_artifacts


def test_runtime_contract_patches_source_table_resolved_config_and_manifest(tmp_path) -> None:  # noqa: ANN001
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "resolved_config.json").write_text("{}", encoding="utf-8")
    (meta / "source_table.json").write_text(
        json.dumps({"source_table_schema_version": "unit", "records": []}),
        encoding="utf-8",
    )
    (meta / "manifest.json").write_text("{}", encoding="utf-8")
    final_cfg = SimpleNamespace(step5_head="step5A")
    args = SimpleNamespace(
        _odcr_step5_peft_meta={
            "target_policy_id": "step5_head_aware_lora_allowlist/1",
            "head_specific_lora_allowlist_id": "step5_head_aware_lora_allowlist/1:step5A",
            "target_modules": ["domain_gate", "transformer_encoder.layers.0.linear1"],
            "forbidden_lora_targets": ["transformer_encoder.layers.0.self_attn.out_proj"],
            "deleted_legacy_modules": ["recommender", "flan_soft_prompt_stack", "hidden2token"],
        },
        _odcr_step5_trainable_contract_meta={
            "policy_id": "step5_head_specific_trainable_contract/1:step5A",
            "final_lora_target_modules_hash": "targets-hash",
            "trainable_parameter_names_hash": "trainable-hash",
            "frozen_parameter_names_hash": "frozen-hash",
            "head_gated_loss_contract": {"head": "step5A", "active_losses": ["scorer_mse", "lci"]},
        },
    )
    preflight = {
        "all_trainable_grad_status": "pass",
        "all_trainable_grad_preflight_result": {
            "status": "pass",
            "trainable_param_count": 4,
            "grad_present_count": 4,
            "lora_trainable_count": 2,
            "lora_grad_present_count": 2,
            "missing_grad_params": [],
        },
        "trainable_param_count": 4,
        "grad_present_count": 4,
        "lora_trainable_count": 2,
        "lora_grad_present_count": 2,
        "missing_grad_params": [],
        "scratch_cleared_after_preflight": True,
        "graph_scratch_before_ema": [],
    }

    _patch_step5_runtime_contract_artifacts(str(meta), final_cfg, args, preflight_result=preflight)

    resolved = json.loads((meta / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["lora_target_policy_id"] == "step5_head_aware_lora_allowlist/1"
    assert resolved["head_specific_lora_allowlist_id"].endswith(":step5A")
    assert resolved["final_lora_target_modules"] == ["domain_gate", "transformer_encoder.layers.0.linear1"]
    assert resolved["forbidden_lora_targets"] == ["transformer_encoder.layers.0.self_attn.out_proj"]
    assert resolved["deleted_legacy_modules"] == ["recommender", "flan_soft_prompt_stack", "hidden2token"]
    assert resolved["all_trainable_grad_required"] is True
    assert resolved["all_trainable_grad_status"] == "pass"
    assert resolved["scratch_cleanup_required"] is True
    assert resolved["formal_entry_E4_required"] is True
    assert resolved["scratch_cleanup_status"] is True
    assert resolved["graph_tensor_audit_status"] == "pass"
    assert resolved["ema_init_strategy"] == "AveragedModel_after_scratch_cleanup"

    source = json.loads((meta / "source_table.json").read_text(encoding="utf-8"))
    records = {row["key"]: row["value"] for row in source["records"]}
    assert records["final_lora_target_modules_hash"] == "targets-hash"
    assert records["trainable_parameter_names_hash"] == "trainable-hash"
    assert records["deleted_legacy_modules"] == ["recommender", "flan_soft_prompt_stack", "hidden2token"]
    assert records["head_specific_trainable_policy"] == "step5_head_specific_trainable_contract/1:step5A"
    assert records["all_trainable_grad_preflight_result"]["status"] == "pass"
    assert records["scratch_cleanup_status"] is True
    assert records["graph_tensor_audit_status"] == "pass"
    assert records["graph_tensor_audit_phase"] == "before_ema_init"
    assert records["ema_init_strategy"] == "AveragedModel_after_scratch_cleanup"
    assert records["formal_entry_E4_required"] is True
    assert records["missing_grad_params"] == []
    assert records["synthetic_used"] is False

    manifest = json.loads((meta / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_trainable_grad_status"] == "pass"
    assert manifest["step5_trainable_contract"]["trainable_param_count"] == 4
