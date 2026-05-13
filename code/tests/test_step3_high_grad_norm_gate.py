from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step3HighGradNormGateTest(unittest.TestCase):
    def test_high_grad_norm_thresholds_resolve_from_one_control(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=["experiment_profile=csb_odcr_full_safe"],
            dry_run=True,
            run_id="auto",
            mode="full",
        )

        grad = snapshot["step3_grad_finite"]
        self.assertEqual(grad["high_grad_norm_warn_threshold"], 100.0)
        self.assertEqual(grad["high_grad_norm_skip_threshold"], 10000.0)
        self.assertEqual(grad["high_grad_norm_abort_threshold"], 100000000.0)
        self.assertEqual(grad["continuous_high_grad_abort_threshold"], 3)
        self.assertIn("high_grad_norm_skip_threshold", cfg.grad_finite_config_json)

    def test_train_loop_skip_branch_zeroes_grad_and_does_not_step_scheduler(self) -> None:
        source = (REPO_ROOT / "code" / "executors" / "step3_train_core.py").read_text(encoding="utf-8")
        branch_start = source.index('skipped_step_reason = "high_grad_norm_abort"')
        branch = source[branch_start : source.index("else:", branch_start)]

        self.assertIn("optimizer.zero_grad(set_to_none=True)", branch)
        self.assertNotIn("sched.step()", branch)
        self.assertIn("scheduler_step_executed", source)
        self.assertIn("Step3 high gradient norm gate aborted after", branch)


if __name__ == "__main__":
    unittest.main()
