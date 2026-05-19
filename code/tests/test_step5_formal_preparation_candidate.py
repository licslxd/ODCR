from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "tests"))

import odcr_core.config_resolver as config_resolver
from odcr_core.step5_formal_preparation import (
    REPORT_PATHS,
    build_step5_formal_split_execution_plan,
    low_weighted_consistency,
    parse_step5_candidate_id,
    reconcile_step5_formal_sample_budget,
)
from test_step5_auto_budget_tuning import _manifest


SELECTED = "A_RATIO_0+B_RATIO_0+A_CF_MIX_FORMAL_HIGH_ONLY+B_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"


def _result(low_count: int) -> dict:
    return {
        "success": True,
        "real_forward_backward_executed": True,
        "actual_gpu_backward_executed": True,
        "finite_loss_sync_ok": True,
        "graph_safe_backward_ok": True,
        "long_window": {"memory_creep_detected": False, "steps_executed_max": 100},
        "formal_namespace_pollution": False,
        "latest_json_created": False,
        "checkpoint_written": False,
        "per_tier_loss": {
            "cf_high": {"tier_count": 10},
            "cf_medium": {"tier_count": 4},
            "cf_low_weighted": {"tier_count": int(low_count)},
        },
        "rank_results": [
            {
                "losses": {
                    "lci_raw_loss": 0.1,
                    "lci_weighted_loss": 0.01,
                    "uci_weight_mean": 1.0,
                    "fca_raw_loss": 0.1,
                    "fca_weighted_loss": 0.01,
                },
                "ccv_control_packet": {"present": True},
            }
        ],
    }


def test_selected_formal_prep_candidate_parses_lr_and_weights() -> None:
    parsed = parse_step5_candidate_id(SELECTED)
    assert parsed["step5A_ratio"] == "A_RATIO_0"
    assert parsed["step5B_ratio"] == "B_RATIO_0"
    assert parsed["step5A_cf_mix"] == "A_CF_MIX_FORMAL_HIGH_ONLY"
    assert parsed["step5B_cf_mix"] == "B_CF_MIX_FORMAL_HIGH_MEDIUM"
    assert parsed["lr"] == 0.001
    assert parsed["innovation_weights"] == "W0"


def test_formal_preparation_reports_are_ai_analysis_only() -> None:
    for path in REPORT_PATHS.values():
        assert str(path).startswith("AI_analysis/")
    assert "runs/step5" not in "\n".join(str(path) for path in REPORT_PATHS.values())


def test_low_weighted_active_mismatch_fails() -> None:
    report = low_weighted_consistency(
        sampler_config={
            "step5A": {"cf_tier_mix": {"high": 0.95, "medium": 0.0, "low_weighted": 0.05}},
            "step5B": {"cf_tier_mix": {"high": 0.85, "medium": 0.0, "low_weighted": 0.15}},
        },
        step5a_result=_result(0),
        step5b_result=_result(3),
    )
    assert report["low_weighted_mismatch"] is True
    assert report["low_weighted_consistency_pass"] is False


def test_low_weighted_zero_mainline_allows_graph_tied_zero() -> None:
    report = low_weighted_consistency(
        sampler_config={
            "step5A": {"cf_tier_mix": {"high": 1.0, "medium": 0.0, "low_weighted": 0.0}},
            "step5B": {"cf_tier_mix": {"high": 0.6, "medium": 0.4, "low_weighted": 0.0}},
        },
        step5a_result=_result(0),
        step5b_result=_result(0),
    )
    assert report["low_weighted_policy"] == "disabled_for_mainline"
    assert report["low_weighted_active_in_config"] is False
    assert report["low_weighted_consistency_pass"] is True


def test_low_weighted_positive_requires_nonzero_sampled_low() -> None:
    report = low_weighted_consistency(
        sampler_config={
            "step5A": {"cf_tier_mix": {"high": 0.95, "medium": 0.0, "low_weighted": 0.05}},
            "step5B": {"cf_tier_mix": {"high": 0.85, "medium": 0.0, "low_weighted": 0.15}},
        },
        step5a_result=_result(1),
        step5b_result=_result(3),
    )
    assert report["low_weighted_policy"] == "active"
    assert report["low_weighted_sampled_nonzero"] is True
    assert report["low_weighted_consistency_pass"] is True


def test_formal_active_cf_mix_recomputes_balanced_capacity_and_samples() -> None:
    cfg = config_resolver.load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
    sampler = config_resolver._resolve_step5_sampler_config(cfg)
    batch = config_resolver._resolve_step5_batch_candidates_config(cfg)
    tuning = config_resolver._resolve_step5_tuning_config(cfg, batch)
    old_report = {
        "heads": {
            "step5A": {"selected": {"effective_samples": 1290582, "optimizer_steps": 2881}},
            "step5B": {"selected": {"effective_samples": 1092873, "optimizer_steps": 2440}},
        }
    }
    report = reconcile_step5_formal_sample_budget(
        manifest=_manifest(),
        sampler_config=sampler,
        batch_candidates_config=batch,
        tuning_config=tuning,
        old_auto_budget_report=old_report,
    )
    assert report["heads"]["step5A"]["formal_active_balanced_capacity"] == 238308
    assert report["heads"]["step5B"]["formal_active_balanced_capacity"] == 533176
    assert report["heads"]["step5A"]["recommended_effective_samples"] == 190646
    assert report["heads"]["step5B"]["recommended_effective_samples"] == 426541
    assert report["heads"]["step5A"]["recommended_optimizer_steps"] == 426
    assert report["heads"]["step5B"]["recommended_optimizer_steps"] == 953
    assert report["heads"]["step5A"]["reported_auto_budget_effective_samples"] == 1290582
    assert report["heads"]["step5B"]["reported_auto_budget_effective_samples"] == 1092873
    assert report["heads"]["step5A"]["old_effective_samples_still_valid"] is False
    assert report["heads"]["step5B"]["old_effective_samples_still_valid"] is False
    assert report["replacement_rate_ok"] is True


def test_split_execution_plan_keeps_step5a_step5b_independent() -> None:
    candidate = {
        "effective_samples": {"step5A": 190646, "step5B": 426541},
        "optimizer_steps": {"step5A": 426, "step5B": 953},
        "step4_pool_manifest": "runs/step4/task2/1/step5_pools/step5_pool_manifest.json",
        "step4_sampling_contract": "runs/step4/task2/1/step5_pools/step5_sampling_contract.json",
        "selected_tuning_candidate": SELECTED,
    }
    plan = build_step5_formal_split_execution_plan(candidate=candidate, budget_reconciliation={"heads": {}})
    assert plan["can_run_heads_independently"] is True
    assert "Step5A formal train" in plan["recommended_split"]
    assert "Step5B formal train" in plan["recommended_split"]
    assert plan["namespace_plan"]["latest_pointer"].startswith("write only after both heads")
    assert plan["eval_export_policy"].startswith("eval/rerank is separate authorization")
