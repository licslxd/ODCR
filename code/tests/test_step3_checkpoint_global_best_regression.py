from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_quality import metric_improved  # noqa: E402


class TestStep3CheckpointGlobalBestRegression(unittest.TestCase):
    def test_epoch2_global_best_beats_epoch7_local_improvement(self) -> None:
        metrics = {2: 4.8578, 6: 8.4000, 7: 8.1553}
        best_epoch = None
        best_metric = None
        for epoch in (2, 6, 7):
            if metric_improved(metrics[epoch], best_metric, direction="min"):
                best_epoch = epoch
                best_metric = metrics[epoch]
        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_metric, 4.8578)
        self.assertTrue(metric_improved(metrics[7], metrics[6], direction="min"))
        self.assertFalse(metric_improved(metrics[7], metrics[2], direction="min"))


if __name__ == "__main__":
    unittest.main()
