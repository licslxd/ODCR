from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.evidence_level import (  # noqa: E402
    E1_SCHEMA_PREVIEW,
    E3_GPU_TRANSPORT,
    E4_GPU_SHARD_FORWARD_BOUNDED,
    EvidenceLevelError,
    assert_not_schema_only_for_tuning,
    evidence_level_rank,
    is_tuning_eligible,
    mark_gpu_shard_forward,
    mark_schema_preview,
    parse_evidence_level,
    require_min_evidence_level,
)


class Step4EvidenceLevelTest(unittest.TestCase):
    def test_parse_and_rank_levels(self) -> None:
        self.assertEqual(parse_evidence_level("E1"), E1_SCHEMA_PREVIEW)
        self.assertLess(evidence_level_rank(E3_GPU_TRANSPORT), evidence_level_rank(E4_GPU_SHARD_FORWARD_BOUNDED))

    def test_cpu_preview_is_marked_e1_not_tuning(self) -> None:
        payload = mark_schema_preview({"candidate_id": "cpu"})
        self.assertEqual(payload["evidence_level"], E1_SCHEMA_PREVIEW)
        self.assertTrue(payload["proxy_score_present"])
        self.assertTrue(payload["fake_score_present"])
        self.assertTrue(payload["not_for_tuning"])
        self.assertFalse(is_tuning_eligible(payload))
        with self.assertRaises(EvidenceLevelError):
            assert_not_schema_only_for_tuning(payload)

    def test_e3_transport_cannot_tune_but_e4_can(self) -> None:
        e3 = {
            "evidence_level": E3_GPU_TRANSPORT,
            "gpu_runtime_evidence": False,
            "actual_gpu_forward_executed": False,
        }
        with self.assertRaises(EvidenceLevelError):
            require_min_evidence_level(e3, E4_GPU_SHARD_FORWARD_BOUNDED, "transport")
        e4 = mark_gpu_shard_forward({"candidate_id": "g1"})
        self.assertTrue(is_tuning_eligible(e4))


if __name__ == "__main__":
    unittest.main()
