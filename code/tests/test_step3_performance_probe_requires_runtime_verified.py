from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_runtime_probe import child_status_from_report  # noqa: E402


class Step3PerformanceProbeRequiresRuntimeVerifiedTest(unittest.TestCase):
    def test_runtime_verified_false_exits_nonzero(self) -> None:
        status = child_status_from_report(
            {
                "runtime_verified": False,
                "evidence_complete": True,
                "runtime_probe_ok": True,
                "formal_namespace_polluted": False,
                "runtime_started": True,
            },
            run_id="unit",
            elapsed_s=1.0,
            max_seconds=30,
        )
        self.assertFalse(status["success"])
        self.assertNotEqual(status["exit_code"], 0)

    def test_success_true_runtime_false_is_normalized_to_fail(self) -> None:
        status = child_status_from_report(
            {
                "success": True,
                "runtime_verified": False,
                "evidence_complete": True,
                "runtime_probe_ok": True,
                "formal_namespace_polluted": False,
                "runtime_started": True,
            },
            run_id="unit",
            elapsed_s=1.0,
            max_seconds=30,
        )
        self.assertFalse(status["success"])
        self.assertFalse(status["runtime_verified"])
        self.assertEqual(status["exit_code"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

