from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from odcr_core.config_resolver import OneControlConfigError
from odcr_core.config_resolver import resolve_config
from odcr_core.step5_pool_sampler import (
    Step5PoolSource,
    _route_cap_cf_tier_counts,
    _sample_component,
    resolve_step5_pool_source,
    validate_step5_formal_sample_plan_for_source,
)


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
WEAK_PROTOCOL_TASK8_OVERRIDES = [
    item.replace("step5.tuning.", "step5.tasks.8.tuning.")
    .replace("step5.sampler.", "step5.tasks.8.sampler.")
    for item in WEAK_PROTOCOL_OVERRIDES
] + [
    "step5.tasks.8.sampler.explanation.target_gold_ratio=0.43",
    "step5.tasks.8.sampler.explanation.aux_gold_ratio=0.23",
    "step5.tasks.8.sampler.explanation.cf_ratio=0.34",
]


def test_step5_task8_default_uses_task_adaptive_route_cap_candidate() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=8,
        set_overrides=[],
        dry_run=True,
        from_step4="1",
    )
    active = snapshot["step5_formal_active_candidate"]
    assert active["sampler_protocol"] == "STEP5_TASK_ADAPTIVE_ROUTE_CAP_MAINLINE"
    assert active["weak_cross_platform_protocol"] is False
    assert active["candidate_parts"]["ratio_id"] == "STEP5_RATIO_T8_ODCR_NATIVE_CF_CAP"
    assert active["candidate_parts"]["cf_mix_id"] == "STEP5_CF_MIX_T8_ROUTE_CAP"
    assert active["effective_samples"] == {"explanation": 250000}
    assert active["optimizer_steps"] == {"explanation": 3907}
    assert active["explanation_cf_mix"] == {
        "high": 0.00392,
        "medium": 0.03032,
        "low_weighted": 0.96576,
    }
    assert active["target_gold_tier_mix"] == {"high": 0.6, "medium": 0.4}
    assert active["aux_gold_tier_mix"] == {"high": 0.7, "medium": 0.3}
    assert active["lr"] == 0.0005
    assert snapshot["step5_tuning"]["selected_tuning_candidate"].endswith("LR_5e-4+W0")
    assert snapshot["field_sources"]["step5_tuning"].startswith("step5.tasks.8.tuning")
    assert snapshot["field_sources"]["step5_sampler"].startswith("step5.tasks.8.sampler")
    assert int(getattr(_cfg, "train_label_max_length")) == 48
    assert _cfg.step5_train_generation_input_policy == "history_conditioned_no_reference_evidence"
    assert _cfg.step5_train_content_evidence_policy == "train_only_history"


def test_step5_task8_route_cap_cf_counts_use_total_budget_then_priority_fallback() -> None:
    counts, report = _route_cap_cf_tier_counts(
        12500,
        requested_counts={"high": 49, "medium": 379, "low_weighted": 12072},
        component_mix={"high": 0.00392, "medium": 0.03032, "low_weighted": 0.96576},
        available={"high": 49, "medium": 379, "low_weighted": 246270},
        enabled=True,
    )
    assert counts == {"high": 49, "medium": 379, "low_weighted": 12072}
    assert report["policy"] == "cf_total_cap_high_then_medium_then_low_weighted"
    assert report["component_total_cap"] == 12500
    assert report["low_weighted_fallback_rows"] == 12072


def test_step5_task8_route_cap_cf_sampling_uses_single_budget_gate(tmp_path: Path) -> None:
    pool_dir = tmp_path / "step5_pools"
    pool_dir.mkdir()

    def write_pool(name: str, tier: str, n: int, *, route_positive: int) -> dict:
        rows = []
        for idx in range(n):
            routed = 1 if idx < route_positive else 0
            rows.append(
                {
                    "sample_id": f"{tier}-{idx}",
                    "sample_origin": "aux_cf",
                    "clean_text": f"{tier} text {idx}",
                    "content_evidence": f"{tier} evidence",
                    "train_keep": routed,
                    "route_scorer": 0,
                    "route_explainer": routed,
                    "sample_weight_hint": 0.05,
                    "uncertainty_score": 0.2,
                    "confidence_bucket": 1,
                }
            )
        df = pytest.importorskip("pandas").DataFrame(rows)
        path = pool_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return {"path": str(path), "row_count": n, "columns": list(df.columns), "tier": tier}

    pools = {
        "step5_explanation_cf_explainer_high": write_pool(
            "step5_explanation_cf_explainer_high",
            "high",
            1,
            route_positive=1,
        ),
        "step5_explanation_cf_explainer_medium": write_pool(
            "step5_explanation_cf_explainer_medium",
            "medium",
            2,
            route_positive=0,
        ),
        "step5_explanation_cf_explainer_low_weighted": write_pool(
            "step5_explanation_cf_explainer_low_weighted",
            "low_weighted",
            10,
            route_positive=0,
        ),
    }
    manifest_path = pool_dir / "step5_pool_manifest.json"
    contract_path = pool_dir / "step5_sampling_contract.json"
    manifest = {"repo_root": str(tmp_path), "pools": pools}
    manifest_path.write_text("{}", encoding="utf-8")
    contract_path.write_text("{}", encoding="utf-8")
    source = Step5PoolSource(
        pool_dir=pool_dir,
        manifest_path=manifest_path,
        sampling_contract_path=contract_path,
        manifest=manifest,
        sampling_contract={},
        source_full_export=None,
    )

    df, report = _sample_component(
        source,
        head="explanation",
        component="cf",
        count=6,
        component_mix={"high": 0.00392, "medium": 0.03032, "low_weighted": 0.96576},
        weights={"high": 1.2, "medium": 0.9, "low_weighted": 0.3},
        seed=3407,
        epoch=0,
        columns=None,
        task8_route_cap_cf_protocol=True,
    )

    assert len(df) == 6
    assert report["tier_counts"] == {"high": 1, "medium": 2, "low_weighted": 3}
    assert report["route_filter"]["policy"] == "single_gate_cf_budget"
    assert report["route_filter"]["available_route_compatible_by_tier"] == {"high": 1, "medium": 0, "low_weighted": 0}
    assert report["route_filter"]["sampling_capacity_by_tier"] == {"high": 1, "medium": 2, "low_weighted": 10}
    assert set(df["route_explainer"].astype(int)) == {1}
    assert set(df["train_keep"].astype(int)) == {1}
    assert int((df["posterior_route_explainer"].astype(int) == 0).sum()) == 5
    assert int((df["cf_budget_route_override"].astype(bool)).sum()) == 5


def test_step5_task8_weak_low_weighted_candidate_remains_explicit_diagnostic() -> None:
    _cfg, _sources, snapshot = resolve_config(
        config_path=ROOT / "configs" / "odcr.yaml",
        command="step5",
        task_id=8,
        set_overrides=WEAK_PROTOCOL_TASK8_OVERRIDES,
        dry_run=True,
        from_step4="1",
    )
    active = snapshot["step5_formal_active_candidate"]
    assert active["sampler_protocol"] == "WEAK_CROSS_PLATFORM_LOW_WEIGHTED_CF_V1"
    assert active["weak_cross_platform_protocol"] is True
    assert active["explanation_cf_mix"] == {"high": 0.0, "medium": 0.0, "low_weighted": 1.0}
    assert active["target_gold_tier_mix"] == {"high": 0.0, "medium": 1.0}
    assert active["aux_gold_tier_mix"] == {"high": 0.0, "medium": 1.0}


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
