from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import scheduler_semantics  # noqa: E402


class TestSchedulerExplicitDampingSemantics(unittest.TestCase):
    def test_retired_warmup_cosine_with_damping_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "retired"):
            scheduler_semantics(
                scheduler_type="warmup_cosine_with_damping",
                damping_enabled=True,
                base_min_lr=1e-6,
                damping_factor_cumulative=0.25,
            )

    def test_safe_damping_v2_sets_effective_floor(self) -> None:
        state = scheduler_semantics(
            scheduler_type="safe_damping_v2",
            damping_enabled=True,
            base_min_lr=1e-6,
            damping_factor_cumulative=0.25,
        )
        self.assertEqual(state["base_scheduler"], "warmup_cosine")
        self.assertTrue(state["damping_enabled"])
        self.assertEqual(state["effective_min_lr"], 2.5e-7)


if __name__ == "__main__":
    unittest.main()
