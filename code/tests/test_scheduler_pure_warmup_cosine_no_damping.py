from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import scheduler_semantics  # noqa: E402


class TestSchedulerPureWarmupCosine(unittest.TestCase):
    def test_pure_scheduler_uses_base_floor(self) -> None:
        state = scheduler_semantics(scheduler_type="warmup_cosine", base_min_lr=1e-6)
        self.assertEqual(state["effective_min_lr"], 1e-6)
        with self.assertRaises(ValueError):
            scheduler_semantics(scheduler_type="unsupported", base_min_lr=1e-6)


if __name__ == "__main__":
    unittest.main()
