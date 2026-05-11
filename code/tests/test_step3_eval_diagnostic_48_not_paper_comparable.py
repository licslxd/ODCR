from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_eval_protocol import ODCR_STEP3_DIAGNOSTIC, step3_eval_protocol_spec  # noqa: E402


class TestStep3EvalDiagnostic48NotPaperComparable(unittest.TestCase):
    def test_diagnostic_protocol_is_marked_non_comparable(self) -> None:
        spec = step3_eval_protocol_spec(ODCR_STEP3_DIAGNOSTIC)
        self.assertTrue(spec["diagnostic_only"])
        self.assertTrue(spec["not_paper_comparable"])
        self.assertFalse(spec["paper_comparable"])
        self.assertEqual(spec["data_protocol"], "merged_auxiliary_target")
        self.assertEqual(spec["text_length_protocol"], 48)


if __name__ == "__main__":
    unittest.main()
