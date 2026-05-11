from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.evidence_level import mark_gpu_shard_forward, mark_schema_preview  # noqa: E402
from odcr_core.step4_tuning_evidence import (  # noqa: E402
    build_best_candidate_record,
    build_patch_suggestion_text,
    rank_step4_candidates,
)


class Step4CandidateRequiresRealGpuTest(unittest.TestCase):
    def test_e4_candidate_can_rank_and_write_gated_records(self) -> None:
        e4 = mark_gpu_shard_forward({"candidate_id": "G1", "score": 0.9})
        ranked = rank_step4_candidates([e4])
        self.assertEqual(ranked[0]["candidate_id"], "G1")
        best = build_best_candidate_record(e4, source_artifacts=["runs/step4_preflight/task2/g1/preflight_summary.json"])
        self.assertEqual(best["evidence_level"], "E4_gpu_shard_forward_bounded")
        self.assertTrue(best["candidate_actual_gpu_confirmed"])
        patch = build_patch_suggestion_text(e4, body="candidate: G1")
        self.assertIn("real GPU-forward posterior evidence", patch)

    def test_old_c9_cpu_preview_artifact_is_rejected(self) -> None:
        old_c9 = mark_schema_preview({"candidate_id": "C9_bucket_balanced", "score": 1.0, "candidate_source_is_cpu_preview": True})
        with self.assertRaises(Exception):
            rank_step4_candidates([old_c9])


if __name__ == "__main__":
    unittest.main()
