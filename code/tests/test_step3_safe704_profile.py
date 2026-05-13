from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.config_resolver import OneControlConfigError, resolve_config


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step3Safe704ProfileTest(unittest.TestCase):
    def test_safe704_profile_resolves_no_accum_batch_formula(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=["experiment_profile=csb_odcr_full_safe"],
            dry_run=True,
            run_id="auto",
            mode="full",
        )

        self.assertEqual(cfg.per_gpu_batch_size, 704)
        self.assertEqual(cfg.global_batch_size, 1408)
        self.assertEqual(cfg.ddp_world_size, 2)
        self.assertEqual(cfg.learning_rate, 0.0007)
        self.assertEqual(cfg.max_grad_norm, 0.5)
        self.assertEqual(cfg.max_epochs, 40)
        self.assertEqual(snapshot["train"]["precision"], "bf16")
        self.assertEqual(snapshot["train"]["step3_batch_candidate_role"], "G1S-safe")
        self.assertIs(snapshot["train"]["grad_accum_removed"], True)
        self.assertEqual(
            snapshot["train"]["global_batch_size"],
            snapshot["train"]["per_gpu_batch_size"] * snapshot["train"]["ddp_world_size"],
        )
        self.assertIs(snapshot["experiment_profiles"]["csb_odcr_full_safe640"]["fallback_only"], True)
        self.assertEqual(snapshot["experiment_profiles"]["csb_odcr_full"]["train"], {})


    def test_safe_profile_formula_fail_fast_for_bad_override(self) -> None:
        with self.assertRaisesRegex(OneControlConfigError, "batch formula failed"):
            resolve_config(
                config_path=REPO_ROOT / "configs" / "odcr.yaml",
                command="step3",
                task_id=2,
                set_overrides=[
                    "experiment_profile=csb_odcr_full_safe",
                    "step3.experiment_profiles.csb_odcr_full_safe.train.batch_size=1409",
                ],
                dry_run=True,
                run_id="auto",
                mode="full",
            )


if __name__ == "__main__":
    unittest.main()
