from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = CODE_DIR / "tools"
for path in (CODE_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402


class TmuxGpuBridgeRuntimeSuccessSemanticsTest(unittest.TestCase):
    def test_transport_ok_runtime_probe_false_fails(self) -> None:
        status = bridge.normalize_bridge_runtime_success(
            {
                "kind": "step3-performance-probe",
                "success": True,
                "exit_code": 0,
                "first_result_seen": True,
                "runtime_started": True,
                "runtime_probe_ok": False,
                "runtime_verified": False,
                "evidence_complete": True,
                "formal_namespace_polluted": False,
                "metrics": {},
            },
            bridge_transport_ok=True,
        )
        self.assertTrue(status["bridge_transport_ok"])
        self.assertFalse(status["runtime_probe_ok"])
        self.assertFalse(status["success"])
        self.assertEqual(status["exit_code"], 1)

    def test_child_exit_zero_evidence_false_fails(self) -> None:
        status = bridge.normalize_bridge_runtime_success(
            {
                "kind": "step3-performance-probe",
                "success": True,
                "exit_code": 0,
                "first_result_seen": True,
                "runtime_started": True,
                "runtime_probe_ok": True,
                "runtime_verified": True,
                "evidence_complete": False,
                "formal_namespace_polluted": False,
                "metrics": {},
            },
            bridge_transport_ok=True,
        )
        self.assertTrue(status["child_process_ok"])
        self.assertFalse(status["evidence_complete"])
        self.assertFalse(status["final_success"])
        self.assertFalse(status["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

