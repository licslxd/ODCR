from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import step4_runtime  # noqa: E402
from odcr_core.evidence_level import mark_schema_preview  # noqa: E402
from odcr_core.step4_tuning_evidence import (  # noqa: E402
    build_best_candidate_record,
    build_patch_suggestion_text,
    rank_step4_candidates,
)


class Step4CpuPreviewNotTuningTest(unittest.TestCase):
    def test_cpu_preview_runtime_path_carries_not_for_tuning_terms(self) -> None:
        src = inspect.getsource(step4_runtime.run_step4_bounded_preflight)
        self.assertIn("mark_schema_preview", src)
        self.assertIn("cpu_preview_proxy_fields", src)
        self.assertIn("not_step4_runtime_evidence", src)

    def test_cpu_preview_cannot_rank_best_or_patch(self) -> None:
        cpu = mark_schema_preview({"candidate_id": "cpu_preview", "score": 1.0})
        with self.assertRaises(Exception):
            rank_step4_candidates([cpu])
        with self.assertRaises(Exception):
            build_best_candidate_record(cpu, source_artifacts=["cpu.json"])
        with self.assertRaises(Exception):
            build_patch_suggestion_text(cpu, body="unsafe")


if __name__ == "__main__":
    unittest.main()
