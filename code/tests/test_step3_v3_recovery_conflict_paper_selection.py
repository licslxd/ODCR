from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.step3_v3_policy import (  # noqa: E402
    STEP3_TOTAL_LOSS_COMPONENTS,
    apply_loss_multipliers,
    build_recovery_plan,
    detect_objective_drift,
    resolve_phase_for_epoch,
    select_paper_aware_candidates,
    validate_loss_group_mapping,
)


def _resolve_step3():
    return resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        run_id="auto",
        mode="full",
    )


class Step3V3RecoveryConflictPaperSelectionTest(unittest.TestCase):
    def test_formal_default_is_pure_warmup_cosine_and_no_hidden_damping(self) -> None:
        cfg, _sources, snapshot = _resolve_step3()
        payload = json.loads(cfg.effective_training_payload_json)
        scheduler = payload["step3_scheduler"]
        self.assertEqual(scheduler["name"], "warmup_cosine")
        self.assertFalse(scheduler["damping_enabled"])
        self.assertFalse(scheduler["validation_aware_lr_damping"]["enabled"])
        self.assertIn("step3_objective_drift", payload)
        self.assertIn("step3_recovery", payload)
        self.assertIn("step3_phase_loss_schedule", payload)
        self.assertEqual(snapshot["step3_scheduler"]["name"], "warmup_cosine")

    def test_objective_drift_detects_epoch2_to_epoch3_case(self) -> None:
        record = detect_objective_drift(
            epoch=3,
            valid_loss=5.964049042885467,
            best_valid_loss=4.8578043832568705,
            previous_valid_loss=4.8578043832568705,
            component_deltas={
                "L_rating_shared": 0.21,
                "L_content_alignment": 0.03,
                "L_specific_separation": 0.04,
                "L_variance": 0.02,
            },
            config={},
        )
        self.assertEqual(record["status"], "severe_objective_drift")
        self.assertIn("L_rating_shared", record["drift_components"])
        self.assertEqual(record["action"], "start_recovery")

    def test_objective_drift_ignores_tiny_noise(self) -> None:
        record = detect_objective_drift(
            epoch=3,
            valid_loss=4.861,
            best_valid_loss=4.8578,
            previous_valid_loss=4.858,
            component_deltas={"L_rating_shared": 0.0001},
            config={},
        )
        self.assertEqual(record["status"], "none")

    def test_recovery_plan_uses_best_observed_short_cosine_and_not_latest(self) -> None:
        plan = build_recovery_plan(
            epoch=3,
            drift_record={"status": "severe_objective_drift", "valid_loss": 5.9},
            config={"enabled": True, "restart_lr_ratio": 0.25, "recovery_epochs": 8, "max_recoveries": 1},
            best_observed_checkpoint="model/best_observed.pth",
            latest_checkpoint="model/latest.pth",
        )
        self.assertEqual(plan["source_checkpoint_scope"], "best_observed")
        self.assertEqual(plan["source_checkpoint"], "model/best_observed.pth")
        self.assertEqual(plan["forbidden_source_checkpoint"], "model/latest.pth")
        self.assertEqual(plan["recovery_scheduler"], "short_cosine")
        self.assertFalse(plan["damping_enabled"])
        self.assertTrue(plan["max_recoveries_prevents_infinite_loop"])

    def test_phase_schedule_resolves_and_reduces_late_structure_losses(self) -> None:
        _cfg, _sources, snapshot = _resolve_step3()
        schedule = snapshot["step3_phase_loss_schedule"]
        phase = resolve_phase_for_epoch(epoch=30, config=schedule)
        self.assertEqual(phase["phase"], "light_regularization")
        weights = {"L_specific_separation": 0.16, "L_variance": 0.10, "L_light_explainer": 0.03}
        adjusted = apply_loss_multipliers(weights, phase["loss_multipliers"])
        self.assertLess(adjusted["L_specific_separation"], weights["L_specific_separation"])
        self.assertLess(adjusted["L_variance"], weights["L_variance"])
        self.assertGreater(adjusted["L_light_explainer"], weights["L_light_explainer"])

    def test_loss_group_mapping_covers_active_components_and_schema(self) -> None:
        payload = validate_loss_group_mapping(STEP3_TOTAL_LOSS_COMPONENTS)
        self.assertEqual(payload["status"], "pass")
        self.assertFalse(payload["unmapped_components"])
        self.assertTrue(payload["real_data_only"])
        self.assertFalse(payload["writes_formal_checkpoint"])

    def test_paper_candidate_selection_dist_guard_and_scorer_explainer_split(self) -> None:
        collapsed = {
            "candidate_id": "valid_loss_best",
            "checkpoint": "best_observed.pth",
            "checkpoint_scope": "best_observed",
            "metrics": {
                "MAE": 0.575,
                "RMSE": 0.847,
                "ROUGE-L": 10.7,
                "BLEU-4": 1.02,
                "METEOR": 14.4,
                "DIST-1": 0.03,
                "DIST-2": 0.12,
            },
        }
        diverse = {
            "candidate_id": "recovery_candidate",
            "checkpoint": "recovery.pth",
            "checkpoint_scope": "recovery",
            "metrics": {
                "MAE": 0.590,
                "RMSE": 0.860,
                "ROUGE-L": 10.5,
                "BLEU-4": 1.35,
                "METEOR": 14.2,
                "DIST-1": 0.08,
                "DIST-2": 0.25,
            },
        }
        selection = select_paper_aware_candidates([collapsed, diverse])
        self.assertEqual(selection["scorer_downstream_checkpoint"]["checkpoint"], "best_observed.pth")
        self.assertEqual(selection["explainer_downstream_checkpoint"]["checkpoint"], "recovery.pth")
        self.assertTrue(selection["scorer_explainer_can_differ"])

    def test_no_paper_eval_means_no_paper_aware_selection(self) -> None:
        selection = select_paper_aware_candidates([])
        self.assertFalse(selection["selection_available"])
        self.assertTrue(selection["no_paper_eval_no_selection"])

    def test_adapter_gating_defaults_off_and_conflict_modes_probe_gated(self) -> None:
        _cfg, _sources, snapshot = _resolve_step3()
        self.assertFalse(snapshot["step3_adapter_gating"]["enabled"])
        self.assertFalse(snapshot["step3_adapter_gating"]["formal_allowed"])
        self.assertEqual(snapshot["step3_conflict_aware"]["mode"], "off")
        self.assertFalse(snapshot["step3_conflict_aware"]["formal_allowed"])


if __name__ == "__main__":
    unittest.main()
