from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import PAPER_TARGET_ONLY_EVAL, step3_eval_protocol_spec  # noqa: E402


class TestStep3EvalPaperTargetOnlyProtocol(unittest.TestCase):
    def test_paper_protocol_is_target_only_and_paper_comparable(self) -> None:
        spec = step3_eval_protocol_spec(PAPER_TARGET_ONLY_EVAL, split="test")
        self.assertTrue(spec["paper_comparable"])
        self.assertTrue(spec["target_only"])
        self.assertEqual(spec["data_protocol"], "target_only")
        self.assertEqual(spec["split"], "test")


if __name__ == "__main__":
    unittest.main()
