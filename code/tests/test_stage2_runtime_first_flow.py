from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.stage2_runtime_first import (  # noqa: E402
    FAST_SANITY_COMMANDS,
    GPU_VALIDATION_STEPS,
    can_enter_gpu_runtime,
    post_edit_diagnostic_blocks_gpu_probe,
    runtime_first_flow_order,
)


class Stage2RuntimeFirstFlowTest(unittest.TestCase):
    def test_fast_sanity_precedes_gpu_probe(self) -> None:
        order = runtime_first_flow_order()
        self.assertLess(order.index("fast_sanity"), order.index("fresh_gpu_discover_validate"))
        self.assertIn("./odcr doctor", FAST_SANITY_COMMANDS)
        self.assertIn("validate-only", GPU_VALIDATION_STEPS)

    def test_post_edit_full_not_required_before_gpu(self) -> None:
        diagnostic = {"classification": "flaky_resource_kill", "exit_code": -9}
        self.assertFalse(post_edit_diagnostic_blocks_gpu_probe(diagnostic))
        self.assertTrue(can_enter_gpu_runtime(fast_sanity_pass=True, post_edit_diagnostic=diagnostic))

    def test_fast_sanity_failure_blocks_gpu_runtime(self) -> None:
        self.assertFalse(can_enter_gpu_runtime(fast_sanity_pass=False, post_edit_diagnostic=None))


if __name__ == "__main__":
    unittest.main()
