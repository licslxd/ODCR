from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import PAPER_TARGET_ONLY_EVAL, step3_eval_protocol_spec  # noqa: E402


class TestStep3Eval25TokenPaperProtocol(unittest.TestCase):
    def test_ref_and_decode_are_25_tokens(self) -> None:
        spec = step3_eval_protocol_spec(PAPER_TARGET_ONLY_EVAL)
        self.assertEqual(spec["max_ref_len"], 25)
        self.assertEqual(spec["max_decode_len"], 25)
        self.assertEqual(spec["text_length_protocol"], 25)


if __name__ == "__main__":
    unittest.main()
