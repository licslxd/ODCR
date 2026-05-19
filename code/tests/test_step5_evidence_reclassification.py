from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE = REPO_ROOT / "AI_analysis" / "05_final_reports" / "step5_evidence_reclassification_table.json"
CONTRACT = REPO_ROOT / "AI_analysis" / "05_final_reports" / "formal_evidence_gate_contract.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reclassification_table_exists_and_covers_required_categories() -> None:
    payload = _read(TABLE)
    entries = payload["evidence"]
    categories = {item["evidence_id"] for item in entries}
    required = {
        "gpu_bridge_handoff_v2",
        "step5A_B224_real_batch_E4",
        "step5B_batch_evidence",
        "lr_tuning_evidence",
        "sample_budget_evidence",
        "cf_gold_sampling_evidence",
        "token_cache_evidence",
        "artifact_build_evidence",
        "admin_no_cuda_evidence",
        "synthetic_evidence",
        "cpu_preview_evidence",
        "old_transformers_evidence",
        "old_bridge_evidence",
        "old_code_branch_evidence",
    }
    assert required <= categories


def test_invalid_evidence_cannot_gate_or_select() -> None:
    payload = _read(TABLE)
    for item in payload["evidence"]:
        invalid_flag = any(
            bool(item.get(key))
            for key in (
                "uses_synthetic",
                "uses_cpu_preview",
                "uses_fake_proxy",
                "uses_admin_no_cuda",
                "uses_old_bridge",
                "uses_old_transformers",
                "artifact_build_only",
            )
        )
        if invalid_flag:
            assert item["decision"] in {"invalid", "needs_rerun"}
            assert item["can_gate_formal"] is False
            assert item["can_select_batch"] is False
            assert item["can_select_lr"] is False
            assert item["can_select_samples"] is False
        if item["evidence_level"] == "E4":
            assert item["forward_executed"] and item["loss_backward_executed"] and item["optimizer_step_executed"]
        if item["evidence_level"] == "E3":
            assert item["can_gate_formal"] is False


def test_formal_evidence_gate_contract_records_hard_rules() -> None:
    payload = _read(CONTRACT)
    assert payload["schema_version"] == "odcr_formal_evidence_gate_contract/1"
    assert payload["e3_can_gate_formal"] is False
    assert payload["artifact_build_only_can_be_e4"] is False
    assert payload["synthetic_can_gate_formal"] is False
    assert payload["required_for_e4"] == [
        "uses_real_data",
        "uses_real_model",
        "uses_current_gpu_handoff_v2_or_cli_explicit",
        "real_ccv_packet_used",
        "forward_executed",
        "loss_backward_executed",
        "optimizer_step_executed",
    ]
