from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import PREDICTION_SHARD_REQUIRED_FIELDS, sample_integrity_report  # noqa: E402


class TestStep3EvalPredictionShardsHaveSampleId(unittest.TestCase):
    def test_required_fields_and_integrity_pass(self) -> None:
        self.assertIn("sample_id", PREDICTION_SHARD_REQUIRED_FIELDS)
        row = {
            "sample_id": "task2:target:row:1:abc",
            "row_id": 1,
            "split": "valid",
            "domain": "target",
            "user_id": 7,
            "item_id": 9,
            "rating_gold": 4.0,
            "rating_pred": 3.5,
            "pred_text": "good",
            "ref_text": "good",
            "decode_status": "decoded",
            "source_row_index": 1,
            "rank": 0,
        }
        report = sample_integrity_report([row], expected_count=1, expected_sample_ids=[row["sample_id"]])
        self.assertEqual(report["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
