from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import step4_runtime


class Step4PreflightNoFormalPollutionTest(unittest.TestCase):
    def test_preflight_checks_latest_hash_and_writes_nonformal_artifacts(self) -> None:
        src = inspect.getsource(step4_runtime.run_step4_bounded_preflight)
        self.assertIn("step4_preflight", inspect.getsource(step4_runtime._preflight_dir))
        self.assertIn("before_latest_hash", src)
        self.assertIn("after_latest_hash", src)
        self.assertIn("bounded preflight changed formal Step4 latest.json", src)
        self.assertNotIn("write_latest_pointer_json", src)


if __name__ == "__main__":
    unittest.main()
