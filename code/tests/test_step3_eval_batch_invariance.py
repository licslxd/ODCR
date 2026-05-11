from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import compare_eval_batch_outputs  # noqa: E402


class TestStep3EvalBatchInvariance(unittest.TestCase):
    def test_same_rows_after_sort_are_invariant(self) -> None:
        rows = [
            {"sample_id": "b", "rating_gold": 2, "rating_pred": 2.1, "pred_text": "x", "ref_text": "x"},
            {"sample_id": "a", "rating_gold": 4, "rating_pred": 3.9, "pred_text": "y", "ref_text": "y"},
        ]
        report = compare_eval_batch_outputs(rows, list(reversed(rows)))
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
