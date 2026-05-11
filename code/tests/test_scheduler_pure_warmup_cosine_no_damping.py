from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import scheduler_semantics  # noqa: E402


class TestSchedulerPureWarmupCosineNoDamping(unittest.TestCase):
    def test_pure_scheduler_rejects_hidden_damping(self) -> None:
        state = scheduler_semantics(scheduler_type="warmup_cosine", damping_enabled=False, base_min_lr=1e-6)
        self.assertFalse(state["damping_enabled"])
        self.assertEqual(state["effective_min_lr"], 1e-6)
        with self.assertRaises(ValueError):
            scheduler_semantics(scheduler_type="warmup_cosine", damping_enabled=True, base_min_lr=1e-6)


if __name__ == "__main__":
    unittest.main()
