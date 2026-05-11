from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import MINIMAL_EVAL, PAPER_TARGET_ONLY_EVAL, step3_eval_protocol_spec  # noqa: E402


class TestStep3EvalRun2CheckpointEvalOnly(unittest.TestCase):
    def test_run2_checkpoint_exists_and_eval_protocols_are_declared(self) -> None:
        self.assertTrue(ROOT.joinpath("runs/step3/task2/2/model/best_observed.pth").is_file())
        self.assertFalse(step3_eval_protocol_spec(MINIMAL_EVAL)["compute_text_metrics"])
        self.assertTrue(step3_eval_protocol_spec(PAPER_TARGET_ONLY_EVAL)["compute_text_metrics"])


if __name__ == "__main__":
    unittest.main()
