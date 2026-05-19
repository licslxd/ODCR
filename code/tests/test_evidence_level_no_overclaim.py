from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_runtime_probe import evidence_level_runtime_verified  # noqa: E402
from odcr_core.aux.runtime.bounded_probe import build_probe_payload  # noqa: E402


def _memory_truth() -> dict:
    return {"reserved_is_diagnostic_only": True}


class EvidenceLevelNoOverclaimTest(unittest.TestCase):
    def test_level_1_2_cannot_claim_runtime_verified(self) -> None:
        self.assertFalse(evidence_level_runtime_verified(1, code_present=True, active_path=True))
        self.assertFalse(evidence_level_runtime_verified(2, code_present=True, active_path=True))
        self.assertTrue(evidence_level_runtime_verified(3, code_present=True, active_path=True))

    def test_step5_handshake_and_forward_only_cannot_claim_e4(self) -> None:
        handshake = build_probe_payload(
            "step5A",
            2,
            handshake={"torch.cuda.is_available": True, "torch.cuda.device_count": 2},
        )
        self.assertFalse(handshake["success"])
        self.assertEqual(handshake["evidence_level"], "E3_gpu_transport")
        forward_only = build_probe_payload(
            "step5B",
            2,
            handshake={"torch.cuda.is_available": True, "torch.cuda.device_count": 2},
            probe_result={
                "success": True,
                "evidence_level": "E4_gpu_shard_forward_bounded",
                "actual_gpu_forward_executed": True,
                "real_forward_backward_executed": False,
                "memory_truth": _memory_truth(),
                "candidate_decision": {"reserved_memory_used_for_rejection": False},
            },
        )
        self.assertFalse(forward_only["success"])
        self.assertEqual(forward_only["evidence_level"], "E3_gpu_transport")


if __name__ == "__main__":
    unittest.main(verbosity=2)
