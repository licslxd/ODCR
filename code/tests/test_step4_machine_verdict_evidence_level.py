from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step4_evidence_machine_verdict import build_step4_evidence_machine_verdict  # noqa: E402


def _passing() -> dict:
    return {
        "candidate_ranking_evidence_level": "E4_gpu_shard_forward_bounded",
        "schema_only_evidence_used_for_tuning": False,
        "proxy_score_present": False,
        "fake_score_used_for_tuning": False,
        "final_candidate_actual_gpu_confirmed": True,
        "actual_gpu_forward_executed": True,
        "gpu_runtime_evidence": True,
        "candidate_source_is_cpu_preview": False,
        "candidate_source_is_real_gpu_forward": True,
        "eligible_for_formal_prompt": True,
        "guardrail_r116_status": "passed",
    }


class Step4MachineVerdictEvidenceLevelTest(unittest.TestCase):
    def test_e4_real_gpu_can_be_a_when_clean(self) -> None:
        verdict = build_step4_evidence_machine_verdict(_passing())
        self.assertEqual(verdict["verdict"], "A")

    def test_cpu_preview_cannot_verdict_a(self) -> None:
        fields = _passing()
        fields.update(
            {
                "candidate_ranking_evidence_level": "E1_schema_preview",
                "schema_only_evidence_used_for_tuning": True,
                "proxy_score_present": True,
                "fake_score_used_for_tuning": True,
                "actual_gpu_forward_executed": False,
                "gpu_runtime_evidence": False,
                "candidate_source_is_cpu_preview": True,
                "candidate_source_is_real_gpu_forward": False,
            }
        )
        verdict = build_step4_evidence_machine_verdict(fields)
        self.assertEqual(verdict["verdict"], "C")

    def test_missing_evidence_level_cannot_verdict_a(self) -> None:
        fields = _passing()
        fields["candidate_ranking_evidence_level"] = ""
        verdict = build_step4_evidence_machine_verdict(fields)
        self.assertEqual(verdict["verdict"], "D")

    def test_proxy_or_schema_usage_forces_c(self) -> None:
        fields = _passing()
        fields["proxy_score_present"] = True
        fields["proxy_score_used_for_tuning"] = True
        self.assertEqual(build_step4_evidence_machine_verdict(fields)["verdict"], "C")
        fields = _passing()
        fields["schema_only_evidence_used_for_tuning"] = True
        self.assertEqual(build_step4_evidence_machine_verdict(fields)["verdict"], "C")


if __name__ == "__main__":
    unittest.main()
