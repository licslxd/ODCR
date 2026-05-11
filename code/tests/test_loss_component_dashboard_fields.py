from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import STEP3_LOSS_DASHBOARD_SCHEMA_VERSION, summarize_loss_component_rows  # noqa: E402


class TestLossComponentDashboardFields(unittest.TestCase):
    def test_dashboard_contains_epoch_rows_and_trends(self) -> None:
        summary = summarize_loss_component_rows(
            [
                {"epoch": 2, "loss_name": "rating_loss", "raw_value": 2.0, "weighted_value": 2.0},
                {"epoch": 40, "loss_name": "rating_loss", "raw_value": 1.9, "weighted_value": 1.9},
            ]
        )
        self.assertEqual(summary["schema_version"], STEP3_LOSS_DASHBOARD_SCHEMA_VERSION)
        self.assertTrue(summary["epoch_rows"])
        self.assertIn("rating_loss", summary["component_trends"])


if __name__ == "__main__":
    unittest.main()
