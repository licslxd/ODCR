from pathlib import Path
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
        task_id=2,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    source = resolve_step5_pool_source(step4_run_dir=ROOT / "runs" / "step4" / "task2" / "1", repo_root=ROOT)
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
    assert head["components"]["cf"]["route_filter"]["route_column"] == "route_explainer"
    assert head["task_decoupled_policy"]["rating_training"]["enabled"] is False
    assert report["formal_namespace_write"] is False


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
    "step5.tuning.fallback_batch_candidate=B32",
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
