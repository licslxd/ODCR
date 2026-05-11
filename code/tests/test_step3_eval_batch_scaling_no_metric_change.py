from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import EvalBatchProbe, select_largest_safe_eval_batch  # noqa: E402


class TestStep3EvalBatchScalingNoMetricChange(unittest.TestCase):
    def test_largest_safe_batch_requires_invariance_pass(self) -> None:
        selected = select_largest_safe_eval_batch(
            [
                EvalBatchProbe(batch_size=1536, status="PASS", invariance_status="PASS"),
                EvalBatchProbe(batch_size=3072, status="PASS", invariance_status="PASS"),
                EvalBatchProbe(batch_size=6144, status="PASS", invariance_status="PASS"),
                EvalBatchProbe(batch_size=8192, status="FAIL", oom=True),
            ]
        )
        self.assertEqual(selected["selected_eval_batch"], 6144)
        self.assertEqual(selected["invariance_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
