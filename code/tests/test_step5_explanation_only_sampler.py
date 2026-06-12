from pathlib import Path
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import OneControlConfigError
from odcr_core.config_resolver import resolve_config
from odcr_core.step5_pool_sampler import resolve_step5_pool_source, validate_step5_formal_sample_plan_for_source


def test_step5_sampler_uses_explanation_pools_and_route_explainer() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=5,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    source = resolve_step5_pool_source(step4_run_dir=ROOT / "runs" / "step4" / "task5" / "1", repo_root=ROOT)
    report = validate_step5_formal_sample_plan_for_source(
        source,
        sampler_config=snapshot["step5_sampler"],
        batch_candidates_config=snapshot["step5_batch_candidates"],
        tuning_config=snapshot["step5_tuning"],
        task_head="explanation",
        bounded_max_rows=128,
        no_write=True,
    )
    head = report["heads"]["explanation"]
    ratios = snapshot["step5_sampler"]["explanation"]
    assert ratios["target_gold_ratio"] == 0.50
    assert ratios["aux_gold_ratio"] == 0.15
    assert ratios["cf_ratio"] == 0.35
    assert snapshot["step5_tuning"]["effective_samples"]["explanation"] == 250000
    assert snapshot["step5_tuning"]["optimizer_steps"]["explanation"] == 1954
    assert snapshot["step5_tuning"]["selected_warmup_fraction"] == 0.05
    assert snapshot["step5_formal_active_candidate"]["warmup_fraction"] == 0.05
    intent = snapshot["step5_formal_active_candidate"]["sample_plan_intent"]
    assert intent["component_ratios"] == {"target_gold": 0.5, "aux_gold": 0.15, "cf": 0.35}
    assert intent["sample_plan_intent_hash"] == snapshot["step5_formal_active_candidate"]["sample_plan_intent_hash"]
    payload = json.loads(_cfg.effective_training_payload_json)
    assert payload["training_row"]["warmup_ratio"] == 0.05
    assert payload["training_row"]["warmup_epochs"] == 0.0
    assert head["components"]["cf"]["route_filter"]["route_column"] == "route_explainer"
    assert head["task_decoupled_policy"]["rating_training"]["enabled"] is False
    assert report["formal_namespace_write"] is False


def test_step5_task1_uses_b32_memory_limited_batch_after_oom() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=1,
        set_overrides=[],
        dry_run=True,
        from_step4="2",
    )
    assert snapshot["train"]["global_batch_size"] == 64
    assert snapshot["train"]["per_gpu_batch_size"] == 32
    assert snapshot["train"]["batch_candidate"] == "B32"
    assert snapshot["step5_tuning"]["batch_candidate"] == "B32"
    assert snapshot["step5_tuning"]["optimizer_steps"]["explanation"] == 3907
    active = snapshot["step5_formal_active_candidate"]
    assert active["batch_candidate"] == "B32"
    assert active["sample_plan_intent"]["selected_batch"] == {
        "id": "B32",
        "per_gpu_batch_size": 32,
        "global_batch_size": 64,
    }
    assert snapshot["eval"]["step5_train_validation"]["valid_per_gpu_batch_size"] == 128
    assert snapshot["eval"]["step5_train_validation"]["valid_global_batch_size"] == 256
    assert snapshot["eval"]["step5_train_validation"]["valid_forward_micro_batch_size"] == 128


def _step5_ratio_override(candidate_id: str, *, target_gold: float, aux_gold: float, cf: float) -> list[str]:
    return [
        f"step5.tuning.ratio_candidates.explanation.{candidate_id}.target_gold={target_gold}",
        f"step5.tuning.ratio_candidates.explanation.{candidate_id}.aux_gold={aux_gold}",
        f"step5.tuning.ratio_candidates.explanation.{candidate_id}.cf={cf}",
        f"step5.sampler.explanation.target_gold_ratio={target_gold}",
        f"step5.sampler.explanation.aux_gold_ratio={aux_gold}",
        f"step5.sampler.explanation.cf_ratio={cf}",
        (
            "step5.tuning.selected_tuning_candidate="
            f"{candidate_id}+STEP5_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"
        ),
    ]


def test_step5_sampler_accepts_target_gold_only_overfit_ratio() -> None:
    candidate_id = "STEP5_RATIO_TARGET_ONLY"
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=5,
        set_overrides=_step5_ratio_override(candidate_id, target_gold=1.0, aux_gold=0.0, cf=0.0),
        dry_run=True,
        from_step4="1",
    )
    ratios = snapshot["step5_sampler"]["explanation"]
    assert ratios["target_gold_ratio"] == 1.0
    assert ratios["aux_gold_ratio"] == 0.0
    assert ratios["cf_ratio"] == 0.0
    assert snapshot["step5_formal_active_candidate"]["candidate_parts"]["ratio_id"] == candidate_id


def test_step5_sampler_accepts_target_gold_heavy_warm_ratio() -> None:
    candidate_id = "STEP5_RATIO_TARGET_HEAVY"
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=5,
        set_overrides=_step5_ratio_override(candidate_id, target_gold=0.70, aux_gold=0.10, cf=0.20),
        dry_run=True,
        from_step4="1",
    )
    ratios = snapshot["step5_sampler"]["explanation"]
    assert ratios["target_gold_ratio"] == 0.70
    assert ratios["aux_gold_ratio"] == 0.10
    assert ratios["cf_ratio"] == 0.20
    assert snapshot["step5_formal_active_candidate"]["candidate_parts"]["ratio_id"] == candidate_id


def test_step5_selected_ratio_candidate_must_match_runtime_sampler_ratios() -> None:
    candidate_id = "STEP5_RATIO_TARGET_ONLY"
    overrides = [
        f"step5.tuning.ratio_candidates.explanation.{candidate_id}.target_gold=1.0",
        f"step5.tuning.ratio_candidates.explanation.{candidate_id}.aux_gold=0.0",
        f"step5.tuning.ratio_candidates.explanation.{candidate_id}.cf=0.0",
        (
            "step5.tuning.selected_tuning_candidate="
            f"{candidate_id}+STEP5_CF_MIX_FORMAL_HIGH_MEDIUM+TG_MIX_0+AG_MIX_0+LR_1e-3+W0"
        ),
    ]
    with pytest.raises(OneControlConfigError, match="does not match selected Step5"):
        resolve_config(
            config_path=ROOT / "configs" / "odcr.yaml",
            command="step5",
            task_id=5,
            set_overrides=overrides,
            dry_run=True,
            from_step4="1",
        )


WEAK_PROTOCOL_OVERRIDES = [
    "step5.tuning.selected_tuning_candidate=STEP5_RATIO_0+WEAK_CROSS_PLATFORM_LOW_WEIGHTED_CF_V1+TG_MIX_WEAK_MEDIUM_ONLY+AG_MIX_WEAK_MEDIUM_ONLY+LR_1e-3+W0",
    "step5.sampler.explanation.target_gold_tier_mix.high=0.0",
    "step5.sampler.explanation.target_gold_tier_mix.medium=1.0",
    "step5.sampler.explanation.aux_gold_tier_mix.high=0.0",
    "step5.sampler.explanation.aux_gold_tier_mix.medium=1.0",
    "step5.sampler.explanation.cf_tier_mix.high=0.0",
    "step5.sampler.explanation.cf_tier_mix.medium=0.0",
    "step5.sampler.explanation.cf_tier_mix.low_weighted=1.0",
    "step5.tuning.batch_candidate=B32",
    "step5.tasks.8.lr=0.001",
]


def test_step5_weak_cross_platform_low_weighted_candidate_is_explicit_for_task8() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=8,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    active = snapshot["step5_formal_active_candidate"]
    assert active["sampler_protocol"] == "WEAK_CROSS_PLATFORM_LOW_WEIGHTED_CF_V1"
    assert active["weak_cross_platform_protocol"] is True
    assert active["explanation_cf_mix"] == {"high": 0.0, "medium": 0.0, "low_weighted": 1.0}
    assert active["target_gold_tier_mix"] == {"high": 0.0, "medium": 1.0}
    assert active["aux_gold_tier_mix"] == {"high": 0.0, "medium": 1.0}
    assert active["lr"] == 0.001
    assert snapshot["step5_tuning"]["selected_tuning_candidate"].endswith("LR_1e-3+W0")
    assert snapshot["field_sources"]["step5_tuning"].startswith("step5.tasks.8.tuning")
    assert snapshot["field_sources"]["step5_sampler"].startswith("step5.tasks.8.sampler")


def test_step5_weak_cross_platform_low_weighted_candidate_is_explicit_for_task7() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=7,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    active = snapshot["step5_formal_active_candidate"]
    assert active["sampler_protocol"] == "WEAK_CROSS_PLATFORM_LOW_WEIGHTED_CF_V1"
    assert active["weak_cross_platform_protocol"] is True
    assert active["explanation_cf_mix"] == {"high": 0.0, "medium": 0.0, "low_weighted": 1.0}
    assert active["lr"] == 0.001
    assert snapshot["step5_tuning"]["selected_tuning_candidate"].endswith("LR_1e-3+W0")


def test_step5_weak_cross_platform_low_weighted_candidate_is_rejected_for_task2() -> None:
    task2_overrides = [item.replace("step5.tasks.8.lr", "step5.tasks.2.lr") for item in WEAK_PROTOCOL_OVERRIDES]
    with pytest.raises(OneControlConfigError, match="only allowed for task7/task8 weak_cross_platform"):
        resolve_config(
            config_path=ROOT / "configs" / "odcr.yaml",
            command="step5",
            task_id=2,
            set_overrides=task2_overrides,
            dry_run=True,
            from_step4="1",
        )
