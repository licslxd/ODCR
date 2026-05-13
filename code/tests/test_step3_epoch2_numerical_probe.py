from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.runtime.stage_dispatch import runtime_probe_bridge_args  # noqa: E402
from odcr_core.step3_runtime_probe import (  # noqa: E402
    GRAD_REQUIRED_FIELDS,
    STEP3_RUNTIME_PROBE_TYPES,
    Step3ValidationWindowRequest,
)


class Step3Epoch2NumericalProbeTest(unittest.TestCase):
    def test_epoch2_probe_registered_with_bounded_bridge_args(self) -> None:
        args = runtime_probe_bridge_args(
            stage="step3",
            task=2,
            profile="csb_odcr_full_safe",
            bounded=True,
            probe_kind="epoch2-numerical-stability",
            dry_run=True,
            no_send=True,
        )

        self.assertIn("epoch2-numerical-stability", STEP3_RUNTIME_PROBE_TYPES)
        self.assertEqual(args[0], "step3-performance-probe")
        self.assertIn("--probe-type", args)
        self.assertEqual(args[args.index("--probe-type") + 1], "epoch2-numerical-stability")
        self.assertEqual(args[args.index("--warmup-steps") + 1], "500")
        self.assertEqual(args[args.index("--measured-steps") + 1], "500")
        self.assertEqual(args[args.index("--max-seconds") + 1], "895")
        self.assertIn("--dry-run", args)
        self.assertIn("--no-send", args)

    def test_probe_contract_requires_new_grad_diagnostics(self) -> None:
        request = Step3ValidationWindowRequest(
            task_id=2,
            validation_slug="epoch2_numerical_stability",
            run_id="unit",
            probe_type="epoch2-numerical-stability",
            warmup_steps=500,
            measured_steps=500,
            max_wall_seconds=895,
        )

        self.assertEqual(request.probe_type, "epoch2-numerical-stability")
        for field in (
            "pre_clip_grad_norm",
            "post_clip_grad_norm",
            "rank_global_grad_finite",
            "rank_global_grad_norm_finite",
            "topk_grad_norm_params",
            "high_grad_skip_count",
            "nonfinite_skip_count",
            "warmup_multipliers",
        ):
            self.assertIn(field, GRAD_REQUIRED_FIELDS)


if __name__ == "__main__":
    unittest.main()
