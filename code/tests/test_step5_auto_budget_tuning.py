from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(CODE_DIR / "tests"))

import odcr_core.config_resolver as config_resolver  # noqa: E402
from odcr_core.step5_auto_budget import (  # noqa: E402
    compute_step5_auto_budget_report,
    pool_capacity_from_manifest,
    replacement_summary_for_effective_samples,
)
from test_step4_step5_pool_sampler import (  # noqa: E402
    _batch_candidates_config,
    _sampler_config,
    _tuning_config,
    _write_run,
)
from odcr_core.step4_pool_exports import export_step4_pool_exports  # noqa: E402
from odcr_core.step5_pool_sampler import resolve_step5_pool_source, sample_effective_epochs_from_pools  # noqa: E402


def _manifest() -> dict:
    pools = {}
    counts = {
        "target_gold_anchor_high": 244267,
        "target_gold_anchor_medium": 562347,
        "aux_gold_anchor_high": 482147,
        "aux_gold_anchor_medium": 762799,
        "cf_scorer_high": 59577,
        "cf_scorer_medium": 118025,
        "cf_scorer_low_weighted": 286869,
        "cf_explainer_high": 110410,
        "cf_explainer_medium": 70870,
        "cf_explainer_low_weighted": 283191,
    }
    for head in ("step5A", "step5B"):
        for prefix in ("target_gold_anchor", "aux_gold_anchor"):
            for tier in ("high", "medium"):
                name = f"{head}_{prefix}_{tier}"
                pools[name] = {"row_count": counts[f"{prefix}_{tier}"]}
        suffix = "cf_scorer" if head == "step5A" else "cf_explainer"
        for tier in ("high", "medium", "low_weighted"):
            name = f"{head}_{suffix}_{tier}"
            pools[name] = {"row_count": counts[f"{suffix}_{tier}"]}
    return {"schema_version": "odcr_step5_pool_manifest/1", "pools": pools}


class Step5AutoBudgetTuningTest(unittest.TestCase):
    def test_auto_budget_balanced_capacity_multipliers_and_steps(self) -> None:
        cfg = config_resolver.load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        sampler = config_resolver._resolve_step5_sampler_config(cfg)
        batch = config_resolver._resolve_step5_batch_candidates_config(cfg)
        tuning = config_resolver._resolve_step5_tuning_config(cfg, batch)
        report = compute_step5_auto_budget_report(
            _manifest(),
            sampler_config=sampler,
            batch_candidates_config=batch,
            tuning_config=tuning,
            throughput_samples_per_sec=223.8946,
        )
        self.assertEqual(report["batch_candidate"], "B224")
        self.assertEqual(report["global_batch_size"], 448)
        self.assertEqual(report["heads"]["step5A"]["balanced_capacity"], 806614)
        self.assertEqual(report["heads"]["step5B"]["balanced_capacity"], 533176)
        self.assertEqual(set(report["heads"]["step5A"]["budget_candidates"]), {"small", "medium", "full", "large"})
        selected_a = report["heads"]["step5A"]["selected"]
        selected_b = report["heads"]["step5B"]["selected"]
        self.assertEqual(selected_a["effective_samples"], 190646)
        self.assertEqual(selected_b["effective_samples"], 426541)
        self.assertEqual(selected_a["optimizer_steps"], 426)
        self.assertEqual(selected_b["optimizer_steps"], 953)
        self.assertTrue(selected_a["replacement_rate_ok"])
        self.assertTrue(selected_b["replacement_rate_ok"])
        self.assertTrue(selected_a["explicit_reconciled_effective_samples"])
        self.assertTrue(selected_b["explicit_reconciled_effective_samples"])
        self.assertEqual(report["heads"]["step5A"]["active_tiers"]["cf"], ["high"])
        self.assertEqual(report["heads"]["step5B"]["active_tiers"]["cf"], ["high", "medium"])

    def test_capacity_comes_from_manifest_not_hardcoded_samples(self) -> None:
        cfg = config_resolver.load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        sampler = config_resolver._resolve_step5_sampler_config(cfg)
        capacity = pool_capacity_from_manifest(_manifest(), "step5A", head_cfg=sampler["step5A"])
        self.assertEqual(capacity["available"]["target_gold"], 806614)
        self.assertEqual(capacity["available"]["aux_gold"], 1244946)
        self.assertEqual(capacity["available"]["cf"], 59577)
        self.assertEqual(capacity["available_all"]["cf"], 464471)
        serialized = json.dumps(config_resolver._resolve_step5_sampler_config(config_resolver.load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")))
        self.assertNotIn("400000", serialized)
        self.assertNotIn("350000", serialized)
        self.assertNotIn("effective_samples_per_epoch_candidates", serialized)

    def test_old_formal_samples_exceed_tier_shortage_guard_under_selected_cf_mix(self) -> None:
        cfg = config_resolver.load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        sampler = config_resolver._resolve_step5_sampler_config(cfg)
        old_a = replacement_summary_for_effective_samples(
            _manifest(),
            head="step5A",
            head_cfg=sampler["step5A"],
            effective_samples=1290582,
        )
        old_b = replacement_summary_for_effective_samples(
            _manifest(),
            head="step5B",
            head_cfg=sampler["step5B"],
            effective_samples=1092873,
        )
        self.assertGreater(old_a["replacement_rate"], 0.20)
        self.assertGreater(old_a["tier_shortage_rate_before_reallocation"], 0.20)
        self.assertGreater(old_a["tier_shortage_rate_by_component_before_reallocation"]["target_gold"], 0.20)
        self.assertEqual(old_a["tier_shortage_rate_by_component_before_reallocation"]["cf"], 0.0)
        self.assertGreater(old_b["tier_shortage_rate_by_component_before_reallocation"]["cf"], 0.35)

    def test_replacement_guard_clamps_large_candidate(self) -> None:
        cfg = config_resolver.load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        sampler = config_resolver._resolve_step5_sampler_config(cfg)
        batch = config_resolver._resolve_step5_batch_candidates_config(cfg)
        tuning = config_resolver._resolve_step5_tuning_config(cfg, batch)
        sampler = copy.deepcopy(sampler)
        sampler["auto_budget"]["budget_multipliers"]["large"] = 2.5
        sampler["auto_budget"]["max_steps_per_effective_epoch"] = 10000
        tuning_no_override = {**tuning, "selected_budget_candidate": "large"}
        tuning_no_override.pop("effective_samples", None)
        tuning_no_override.pop("optimizer_steps", None)
        report = compute_step5_auto_budget_report(
            _manifest(),
            sampler_config=sampler,
            batch_candidates_config=batch,
            tuning_config=tuning_no_override,
        )
        large = report["heads"]["step5A"]["selected"]
        self.assertIn("clamped_to_replacement_guard", large["adjustments"])
        self.assertLessEqual(large["replacement_rate"], 0.20)

    def test_low_weighted_cf_sampling_for_both_heads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            sampler = _sampler_config()
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=sampler,
            )
            source = resolve_step5_pool_source(step4_run_dir=run, repo_root=root)
            sampled_a = sample_effective_epochs_from_pools(
                source,
                sampler_config=sampler,
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                task_head="step5A",
                bounded_max_rows=20,
            )
            sampled_b = sample_effective_epochs_from_pools(
                source,
                sampler_config=sampler,
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                task_head="step5B",
                bounded_max_rows=20,
            )
            self.assertGreaterEqual(
                sampled_a.stats["epoch_reports"][0]["heads"]["step5A"]["cf_tier_counts"].get("low_weighted", 0),
                0,
            )
            self.assertGreater(
                sampled_b.stats["epoch_reports"][0]["heads"]["step5B"]["cf_tier_counts"].get("low_weighted", 0),
                0,
            )
            self.assertFalse((root / "runs" / "step5" / "task2" / "latest.json").exists())

    def test_tuning_candidates_are_one_control_and_prompt_fixed(self) -> None:
        cfg = config_resolver.load_yaml_config(REPO_ROOT / "configs" / "odcr.yaml")
        batch = config_resolver._resolve_step5_batch_candidates_config(cfg)
        tuning = config_resolver._resolve_step5_tuning_config(cfg, batch)
        self.assertEqual([row["id"] for row in tuning["ratio_candidates"]["step5A"]], ["A_TARGET_ONLY"])
        self.assertEqual([row["id"] for row in tuning["ratio_candidates"]["step5B"]], ["B_RATIO_0", "B_RATIO_1", "B_RATIO_2"])
        self.assertIn(0.001, tuning["lr_candidates"])
        self.assertIn(0.05, tuning["warmup_fraction_candidates"])
        self.assertEqual([row["id"] for row in tuning["innovation_weight_candidates"]], ["W0", "W1", "W2", "W3", "W4"])
        resolved, _sources, snapshot = config_resolver.resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step5",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            from_step4="1",
            eval_profile="balanced_2gpu",
            mode="train_only",
        )
        self.assertEqual(snapshot["step5_prompt_templates"]["allowed_template_count"], 4)
        self.assertEqual(json.loads(resolved.step5_tuning_config_json)["batch_candidate"], "B224")


if __name__ == "__main__":
    unittest.main()
