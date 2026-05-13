from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_runtime_probe import (  # noqa: E402
    Step3RuntimeEvidenceSink,
    Step3ValidationNamespaceGuard,
    Step3ValidationWindowRequest,
)


class Step3PerformanceProbeMetricsRequiredTest(unittest.TestCase):
    def test_empty_required_rows_fail_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = Step3ValidationWindowRequest(
                task_id=2,
                validation_slug="truth_probe",
                run_id="unit",
                probe_type="timing-profile-window",
                measured_steps=1,
                max_wall_seconds=30,
            )
            guard = Step3ValidationNamespaceGuard(root, 2, "truth_probe", "unit")
            sink = Step3RuntimeEvidenceSink(request=request, guard=guard)
            ok, findings = sink.validate(
                state={
                    "runtime_started": True,
                    "components_built": True,
                    "dataloader_built": True,
                    "ddp_initialized": True,
                    "batch_executed": True,
                    "forward_executed": True,
                    "loss_executed": True,
                    "backward_executed": True,
                    "optimizer_executed_or_intentionally_skipped": True,
                    "formal_namespace_polluted": False,
                }
            )
        self.assertFalse(ok)
        self.assertTrue(any("timing" in item for item in findings))
        self.assertTrue(any("memory" in item for item in findings))
        self.assertTrue(any("prefetch" in item for item in findings))
        self.assertTrue(any("grad" in item for item in findings))
        self.assertTrue(any("ddp" in item.lower() for item in findings))
        self.assertTrue(any("loss" in item.lower() for item in findings))
        self.assertTrue(any("csb" in item.lower() for item in findings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
