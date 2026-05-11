from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import build_training_effectiveness_record  # noqa: E402


class TestTrainingEffectivenessGatePlateau(unittest.TestCase):
    def test_low_lr_plateau_gets_actionable_status(self) -> None:
        row = build_training_effectiveness_record(
            epoch=40,
            valid_loss=4.7231,
            best_valid_loss=4.7231,
            previous_valid_loss=4.7240,
            lr_base=5e-7,
            lr_effective=2.6e-7,
            base_min_lr=5e-7,
            effective_min_lr=2.5e-7,
            damping_event={},
            checkpoint_improved=False,
        )
        self.assertIn(row["effective_improvement_status"], {"low_lr_no_progress", "marginal_improvement"})
        self.assertIn("need_protocol_eval", row["reasons"])


if __name__ == "__main__":
    unittest.main()
