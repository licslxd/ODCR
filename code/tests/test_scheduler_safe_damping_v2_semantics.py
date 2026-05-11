from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_v3_policy import safe_damping_v2_decision  # noqa: E402


class TestSchedulerSafeDampingV2Semantics(unittest.TestCase):
    def test_safe_damping_has_cap_floor_and_cooldown(self) -> None:
        cfg = {
            "enabled": True,
            "max_damping_events": 2,
            "start_epoch": 4,
            "cooldown_epochs": 3,
            "worsen_abs_threshold": 0.25,
            "worsen_ratio_threshold": 0.10,
            "effective_lr_floor_ratio": 0.25,
            "effective_lr_floor_abs": 0.0,
            "lr_decay_factor": 0.5,
        }
        decision = safe_damping_v2_decision(
            epoch=4,
            valid_loss=5.5,
            best_valid_loss=4.8,
            previous_valid_loss=5.4,
            current_lr=1.0e-4,
            base_min_lr=1.0e-5,
            event_count=0,
            cooldown_remaining=0,
            config=cfg,
        )
        self.assertTrue(decision["apply"])
        self.assertGreaterEqual(decision["effective_lr_floor"], 2.5e-6)
        self.assertEqual(decision["cooldown_remaining"], 3)

        skipped = safe_damping_v2_decision(
            epoch=5,
            valid_loss=5.6,
            best_valid_loss=4.8,
            previous_valid_loss=5.5,
            current_lr=5.0e-5,
            base_min_lr=1.0e-5,
            event_count=1,
            cooldown_remaining=3,
            config=cfg,
        )
        self.assertFalse(skipped["apply"])
        self.assertEqual(skipped["reason"], "cooldown_active")

        capped = safe_damping_v2_decision(
            epoch=8,
            valid_loss=5.6,
            best_valid_loss=4.8,
            previous_valid_loss=5.5,
            current_lr=5.0e-5,
            base_min_lr=1.0e-5,
            event_count=2,
            cooldown_remaining=0,
            config=cfg,
        )
        self.assertFalse(capped["apply"])
        self.assertEqual(capped["reason"], "max_damping_events_reached")


if __name__ == "__main__":
    unittest.main()

