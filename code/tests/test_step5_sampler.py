from __future__ import annotations

import unittest
import sys
import tempfile
import copy
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from odcr_core.gold_quality import default_cf_tier_config  # noqa: E402
from odcr_core.step4_pool_exports import export_step4_pool_exports  # noqa: E402
from odcr_core.step5_pool_sampler import (  # noqa: E402
    Step5PoolSamplerError,
    resolve_step5_pool_source,
    sample_effective_epochs_from_pools,
    validate_step5_formal_sample_plan_for_source,
)
from test_step4_step5_pool_sampler import (  # noqa: E402
    Step4Step5PoolSamplerTest,
    _batch_candidates_config,
    _sampler_config,
    _tuning_config,
    _write_run,
)


class Step5SamplerContractTest(Step4Step5PoolSamplerTest):
    """Alias the pool-sampler contract tests under the active sampler name."""

    def test_step5a_aux_cf_injection_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            sampler = _sampler_config()
            sampler["step5A"]["target_gold_ratio"] = 0.50
            sampler["step5A"]["aux_gold_ratio"] = 0.25
            sampler["step5A"]["cf_ratio"] = 0.25
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=sampler,
            )
            source = resolve_step5_pool_source(step4_run_dir=run, repo_root=root)
            with self.assertRaisesRegex(Exception, "Step5A scorer-clean"):
                validate_step5_formal_sample_plan_for_source(
                    source,
                    sampler_config=sampler,
                    batch_candidates_config=_batch_candidates_config(),
                    tuning_config=_tuning_config(),
                    task_head="step5A",
                    mode="bounded",
                    bounded_max_rows=8,
                )

    def test_target_only_step5a_and_explainer_high_medium_preflight_pass(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            sampler = _sampler_config()
            sampler["step5A"]["target_gold_tier_mix"] = {"high": 0.00, "medium": 1.00}
            sampler["step5B"]["cf_tier_mix"] = {"high": 0.50, "medium": 0.50, "low_weighted": 0.00}
            cf_cfg = default_cf_tier_config()
            cf_cfg["step5A"]["high"].update(
                {
                    "min_rating_stability": 0.90,
                    "min_content_retention": 0.90,
                    "max_uncertainty": 0.12,
                    "min_reliability": 0.65,
                }
            )
            cf_cfg["step5B"]["high"].update(
                {
                    "min_style_shift": 0.15,
                    "min_reliability": 0.65,
                    "max_uncertainty": 0.12,
                    "min_text_quality": 0.80,
                }
            )
            cf_cfg["step5B"]["medium"].update(
                {
                    "min_style_shift": 0.10,
                    "min_reliability": 0.65,
                    "max_uncertainty": 0.40,
                    "min_text_quality": 0.70,
                }
            )
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                cf_tier_config=cf_cfg,
                sampler_config=sampler,
            )
            source = resolve_step5_pool_source(step4_run_dir=run, repo_root=root)
            report_a = validate_step5_formal_sample_plan_for_source(
                source,
                sampler_config=sampler,
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                task_head="step5A",
                mode="bounded",
                bounded_max_rows=1,
            )
            report_b = validate_step5_formal_sample_plan_for_source(
                source,
                sampler_config=sampler,
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                task_head="step5B",
                mode="bounded",
                bounded_max_rows=1,
            )
            self.assertEqual(report_a["status"], "pass")
            self.assertEqual(report_b["status"], "pass")
            self.assertEqual(
                report_a["heads"]["step5A"]["requested_counts"]["target_gold"],
                report_a["heads"]["step5A"]["effective_samples_per_epoch"],
            )
            self.assertEqual(report_a["heads"]["step5A"]["requested_counts"]["aux_gold"], 0)
            self.assertEqual(report_a["heads"]["step5A"]["requested_counts"]["cf"], 0)
            self.assertEqual(report_b["heads"]["step5B"]["components"]["cf"]["requested_tier_counts"]["low_weighted"], 0)

    def test_sample_plan_preflight_rejects_retired_fallback_switches(self) -> None:
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
            sampler["full_audit_default_allowed"] = True
            with self.assertRaisesRegex(Step5PoolSamplerError, "full_audit_default_allowed must be false"):
                validate_step5_formal_sample_plan_for_source(
                    source,
                    sampler_config=sampler,
                    batch_candidates_config=_batch_candidates_config(),
                    tuning_config=_tuning_config(),
                    task_head="step5A",
                    mode="bounded",
                    bounded_max_rows=4,
                )

    def test_low_weighted_zero_mainline_keeps_low_tier_absent_by_design(self) -> None:
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
            sampled = sample_effective_epochs_from_pools(
                source,
                sampler_config=sampler,
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                task_head="step5A",
                bounded_max_rows=8,
            )
            report = sampled.stats["epoch_reports"][0]["heads"]["step5A"]["components"]["cf"]
            self.assertEqual(report["requested_tier_counts"]["low_weighted"], 0)
            self.assertEqual(report["tier_counts"]["low_weighted"], 0)
            self.assertEqual(
                sampled.stats["epoch_reports"][0]["heads"]["step5A"]["cf_tier_counts"].get("low_weighted", 0),
                0,
            )
            self.assertEqual(sampled.stats["epoch_reports"][0]["heads"]["step5A"]["actual_counts"]["aux_gold"], 0)
            self.assertEqual(sampled.stats["epoch_reports"][0]["heads"]["step5A"]["actual_counts"]["cf"], 0)

    def test_low_weighted_active_requires_nonzero_route_compatible_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            sampler = copy.deepcopy(_sampler_config())
            sampler["step5B"]["cf_tier_mix"] = {"high": 0.00, "medium": 0.00, "low_weighted": 1.00}
            export_step4_pool_exports(
                repo_root=root,
                task=2,
                from_run="1",
                pool_config={"enabled": True, "output_dir_name": "step5_pools", "chunk_rows": 2},
                sampler_config=sampler,
            )
            source = resolve_step5_pool_source(step4_run_dir=run, repo_root=root)
            sampled = sample_effective_epochs_from_pools(
                source,
                sampler_config=sampler,
                batch_candidates_config=_batch_candidates_config(),
                tuning_config=_tuning_config(),
                mode="bounded",
                task_head="step5B",
                bounded_max_rows=20,
            )
            low_count = sampled.stats["epoch_reports"][0]["heads"]["step5B"]["cf_tier_counts"].get("low_weighted", 0)
            self.assertGreater(low_count, 0)


if __name__ == "__main__":
    unittest.main()
