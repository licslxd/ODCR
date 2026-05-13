from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_quality import inspect_gradients, sync_grad_gate_diagnostics  # noqa: E402


class Step3GradFiniteDiagnosticsTest(unittest.TestCase):
    def test_top_finite_grad_norm_params_are_reported(self) -> None:
        small = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        large = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        small.grad = torch.ones_like(small) * 2.0
        large.grad = torch.ones_like(large) * 1000.0

        inspection = inspect_gradients([("small.weight", small), ("large.weight", large)], topk=2)

        self.assertTrue(inspection.grad_tensor_finite)
        self.assertTrue(inspection.grad_norm_finite)
        self.assertTrue(inspection.grad_finite)
        self.assertGreater(inspection.grad_norm_pre_clip, 1000.0)
        self.assertEqual(inspection.topk_grad_norm_params[0]["name"], "large.weight")
        self.assertEqual(inspection.topk_nonfinite_params, [])
        self.assertEqual(inspection.first_bad_param_name, "")

    def test_raw_nonfinite_gradient_records_first_bad_param(self) -> None:
        param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        param.grad = torch.tensor([1.0, float("inf")])

        inspection = inspect_gradients([("csb.projection.weight", param)], topk=5)

        self.assertFalse(inspection.grad_tensor_finite)
        self.assertFalse(inspection.grad_finite)
        self.assertEqual(inspection.first_bad_param_name, "csb.projection.weight")
        self.assertEqual(inspection.first_bad_param_reason, "raw_grad_tensor_nonfinite")
        self.assertGreaterEqual(inspection.nonfinite_param_count, 1)
        self.assertEqual(inspection.topk_nonfinite_params[0]["name"], "csb.projection.weight")

    def test_world_size_one_sync_reports_norm_and_high_grad_flags(self) -> None:
        diag = sync_grad_gate_diagnostics(
            local_tensor_finite=True,
            local_norm_finite=True,
            local_high_grad_skip=True,
            local_high_grad_abort=False,
            device=torch.device("cpu"),
            world_size=1,
            rank=0,
        )

        self.assertTrue(diag["rank_global_grad_finite"])
        self.assertTrue(diag["rank_global_grad_norm_finite"])
        self.assertTrue(diag["rank_global_high_grad_skip"])
        self.assertFalse(diag["rank_global_high_grad_abort"])
        self.assertEqual(diag["offender_rank"], 0)


if __name__ == "__main__":
    unittest.main()
