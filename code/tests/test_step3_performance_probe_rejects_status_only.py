from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = CODE_DIR / "tools"
for path in (CODE_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import odcr_step3_performance_probe as probe  # noqa: E402
from odcr_core.step3_runtime_probe import normalize_bridge_runtime_success  # noqa: E402


class Step3PerformanceProbeRejectsStatusOnlyTest(unittest.TestCase):
    def test_plan_only_report_cannot_pass_performance_probe(self) -> None:
        status = normalize_bridge_runtime_success(
            {
                "kind": "step3-performance-probe",
                "success": True,
                "exit_code": 0,
                "first_result_seen": True,
                "plan_only": True,
                "runtime_started": False,
                "runtime_probe_ok": False,
                "runtime_verified": False,
                "evidence_complete": False,
                "formal_namespace_polluted": False,
                "metrics": {},
            },
            bridge_transport_ok=True,
        )
        self.assertFalse(status["success"])
        self.assertNotEqual(status["exit_code"], 0)
        self.assertIn("plan_only", status["stop_reason"])

    def test_tool_source_calls_runtime_window_not_plan_writer(self) -> None:
        source = inspect.getsource(probe)
        self.assertIn("run_step3_validation_window", source)
        self.assertNotIn("build_probe_plan", source)
        self.assertNotIn("status writer", source.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)

