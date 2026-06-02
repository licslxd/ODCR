from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.ablation.binding import (  # noqa: E402
    apply_binding_to_step5_runtime_configs,
    load_ablation_binding,
)
from odcr_core.manifests import write_run_summary_json  # noqa: E402


BASE_SAMPLER = {
    "enabled": True,
    "contract_source": "step4_pool_manifest",
    "full_audit_default_allowed": False,
    "legacy_gold_heavy_exports_allowed": False,
    "explanation": {
        "target_gold_ratio": 0.43,
        "aux_gold_ratio": 0.23,
        "cf_ratio": 0.34,
        "target_gold_tier_mix": {"high": 0.0, "medium": 1.0},
        "aux_gold_tier_mix": {"high": 0.0, "medium": 1.0},
        "cf_tier_mix": {"high": 0.0, "medium": 0.0, "low_weighted": 1.0},
        "aux_gold_weight": 0.5,
        "cf_high_weight": 1.2,
        "cf_medium_weight": 0.9,
        "cf_low_weight": 0.3,
    },
}
BASE_BATCH = {"selected_default": "B32", "candidates": [{"id": "B32", "global_batch_size": 64}]}
BASE_TUNING = {"batch_candidate": "B32", "selected_budget_candidate": "medium"}
BASE_INNOVATION = {
    "lci": {
        "enabled": False,
        "weight": 0.0,
        "confidence_schedule": {"high": 1.0, "medium": 1.0, "low": 0.0},
        "min_reliability": 0.5,
        "max_uncertainty": 0.6,
        "perturb_std": 0.1,
        "counterfactual_label_weight": 0.0,
        "robustness_weight": 0.0,
    },
    "uci": {
        "enabled": False,
        "bucket_weights": {"high": 1.0, "medium": 1.0, "low": 0.0},
        "uncertainty_temperature": 1.0,
        "low_confidence_floor": 0.0,
    },
    "explainer_gate": {
        "bucket_weights": {"high": 1.0, "medium": 1.0, "low": 0.5},
        "uncertainty_exponent": 1.0,
        "style_shift_diversity_boost": 0.0,
        "min_weight": 0.0,
        "max_weight": 2.0,
        "explainer_only_multiplier": 1.0,
    },
    "ccv": {
        "enabled": True,
        "control_fields": ["content_evidence"],
        "uncertainty_tone_control": True,
        "route_conditioning": True,
        "numeric_control_weight": 1.0,
        "control_packet_field_policy": "strict_required",
        "verbalizer_adapter_policy": "ccv_control_adapter",
        "soft_prompt_len": 16,
        "numeric_control_dim": 13,
        "control_adapter_input_blocks": 6,
        "native_lora": {"enabled": True, "r": 16, "alpha": 32.0, "dropout": 0.05, "target_modules": []},
    },
    "fca": {
        "enabled": True,
        "weight": 0.08,
        "min_reliability": 0.5,
        "max_uncertainty": 0.62,
        "evidence_alignment_mode": "evidence_basis",
    },
}


class AblationRuntimeBindingTest(unittest.TestCase):
    def test_wo_rcr_binding_sets_uniform_sampler_runtime(self) -> None:
        binding = load_ablation_binding(REPO_ROOT, task=8, variant="wo_rcr")
        applied = apply_binding_to_step5_runtime_configs(
            binding,
            sampler_config=copy.deepcopy(BASE_SAMPLER),
            batch_candidates_config=copy.deepcopy(BASE_BATCH),
            tuning_config=copy.deepcopy(BASE_TUNING),
            innovation_config=copy.deepcopy(BASE_INNOVATION),
        )
        sampler = applied["sampler_config"]
        self.assertEqual(sampler["ablation_runtime"]["rcr"]["pool_weight_source"], "uniform")
        self.assertEqual(sampler["ablation_runtime"]["rcr"]["route_filter"], "disabled_for_ablation")
        self.assertEqual(sampler["explanation"]["aux_gold_weight"], 1.0)
        self.assertEqual(sampler["explanation"]["cf_low_weight"], 1.0)

    def test_wo_cf_binding_forces_target_gold_only_ratios(self) -> None:
        binding = load_ablation_binding(REPO_ROOT, task=8, variant="wo_cf")
        applied = apply_binding_to_step5_runtime_configs(
            binding,
            sampler_config=copy.deepcopy(BASE_SAMPLER),
            batch_candidates_config=copy.deepcopy(BASE_BATCH),
            tuning_config=copy.deepcopy(BASE_TUNING),
            innovation_config=copy.deepcopy(BASE_INNOVATION),
        )
        head = applied["sampler_config"]["explanation"]
        self.assertEqual(head["target_gold_ratio"], 1.0)
        self.assertEqual(head["aux_gold_ratio"], 0.0)
        self.assertEqual(head["cf_ratio"], 0.0)
        self.assertTrue(applied["application_evidence"]["cf"]["target_gold_only"])

    def test_wo_ccv_fca_binding_disables_loss_controls(self) -> None:
        binding = load_ablation_binding(REPO_ROOT, task=8, variant="wo_ccv_fca")
        applied = apply_binding_to_step5_runtime_configs(
            binding,
            sampler_config=copy.deepcopy(BASE_SAMPLER),
            batch_candidates_config=copy.deepcopy(BASE_BATCH),
            tuning_config=copy.deepcopy(BASE_TUNING),
            innovation_config=copy.deepcopy(BASE_INNOVATION),
        )
        innovation = applied["innovation_config"]
        self.assertFalse(innovation["ccv"]["enabled"])
        self.assertEqual(innovation["ccv"]["numeric_control_weight"], 0.0)
        self.assertFalse(innovation["fca"]["enabled"])
        self.assertEqual(innovation["fca"]["weight"], 0.0)

    def test_ablation_eval_run_summary_does_not_write_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "step5" / "task8" / "ablation_wo_rcr_1" / "post_train_eval_no_ref" / "valid"
            meta_dir = run_dir / "meta"
            meta_dir.mkdir(parents=True)
            write_run_summary_json(
                {
                    "schema_version": "test",
                    "stage": "eval",
                    "task_id": 8,
                    "run_id": "valid",
                    "run_dir": str(run_dir),
                    "meta_dir": str(meta_dir),
                    "status": "ok",
                },
                repo_root=root,
            )
            self.assertTrue((meta_dir / "run_summary.json").is_file())
            self.assertFalse((run_dir.parent / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
