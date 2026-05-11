from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_runtime_probe import evidence_level_runtime_verified  # noqa: E402


class EvidenceLevelNoOverclaimTest(unittest.TestCase):
    def test_level_1_2_cannot_claim_runtime_verified(self) -> None:
        self.assertFalse(evidence_level_runtime_verified(1, code_present=True, active_path=True))
        self.assertFalse(evidence_level_runtime_verified(2, code_present=True, active_path=True))
        self.assertTrue(evidence_level_runtime_verified(3, code_present=True, active_path=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)

