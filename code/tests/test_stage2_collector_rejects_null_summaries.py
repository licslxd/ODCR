from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_runtime_probe import (  # noqa: E402
    MEMORY_REQUIRED_FIELDS,
    TIMING_REQUIRED_FIELDS,
    evaluate_stage2_probe_evidence,
)


class Stage2CollectorRejectsNullSummariesTest(unittest.TestCase):
    def test_null_metrics_are_hard_fail_and_ladder_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "timing_breakdown.csv").write_text(",".join(TIMING_REQUIRED_FIELDS) + "\n", encoding="utf-8")
            memory_row = {field: "NA" for field in MEMORY_REQUIRED_FIELDS}
            (root / "memory_phase_summary.csv").write_text(
                ",".join(MEMORY_REQUIRED_FIELDS) + "\n"
                + ",".join(str(memory_row[field]) for field in MEMORY_REQUIRED_FIELDS)
                + "\n",
                encoding="utf-8",
            )
            for name in (
                "prefetch_overlap_summary.json",
                "grad_monitor_validation.json",
                "ddp_gather_sync_summary.json",
            ):
                (root / name).write_text(json.dumps({"runtime_verified": False, "h2d_wait_ms": None}), encoding="utf-8")
            (root / "run_summary_validation.json").write_text(
                json.dumps({"runtime_verified": False, "evidence_complete": False}),
                encoding="utf-8",
            )
            verdict = evaluate_stage2_probe_evidence(root, probe_type="timing-profile-window")
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["candidate_selection_allowed"])
        self.assertEqual(verdict["G1-M"], "skipped_by_gate")
        self.assertEqual(verdict["G2-C"], "skipped_by_gate")
        self.assertEqual(verdict["G3"], "skipped_by_gate")


if __name__ == "__main__":
    unittest.main(verbosity=2)

