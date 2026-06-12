from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import explain_lr_floor  # noqa: E402


class TestSchedulerCurrentLrFloorExplained(unittest.TestCase):
    def test_below_base_floor_is_not_explained(self) -> None:
        pure = explain_lr_floor(current_lr=5e-7, base_min_lr=1e-6, scheduler_type="warmup_cosine")
        self.assertFalse(pure["floor_explained"])
        at_floor = explain_lr_floor(current_lr=1e-6, base_min_lr=1e-6, scheduler_type="warmup_cosine")
        self.assertTrue(at_floor["floor_explained"])


if __name__ == "__main__":
    unittest.main()
