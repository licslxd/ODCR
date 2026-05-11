from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import FORMAL_PAPER_METRICS, PAPER_TARGET_ONLY_EVAL, step3_eval_protocol_spec  # noqa: E402


class TestStep3EvalNoBertscoreInPaperProtocol(unittest.TestCase):
    def test_bertscore_is_excluded(self) -> None:
        spec = step3_eval_protocol_spec(PAPER_TARGET_ONLY_EVAL)
        self.assertFalse(spec["bertscore_enabled"])
        self.assertFalse(spec["bert_score_enabled"])
        self.assertNotIn("bertscore", {metric.lower() for metric in FORMAL_PAPER_METRICS})


if __name__ == "__main__":
    unittest.main()
