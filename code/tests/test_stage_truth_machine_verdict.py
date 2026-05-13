from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.stage_truth_machine_verdict import build_machine_verdict  # noqa: E402


def _passing_fields() -> dict:
    return {
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 0,
        "required_positive_tests_passed": True,
        "required_negative_tests_passed": True,
        "forged_status_rejected": True,
        "missing_artifact_rejected": True,
        "hash_mismatch_rejected": True,
        "stale_exists_rejected": True,
        "quality_audit_cannot_override_stage_status": True,
        "latest_json_pointer_only": True,
        "manual_alias_parity_passed": True,
        "promotion_malformed_target_rejected": True,
        "resolver_recomputes_checkpoint_hash": True,
        "resolver_validates_readiness_audit": True,
        "resolver_validates_lineage": True,
        "resolver_validates_source_table": True,
        "formal_namespace_pollution_check_passed": True,
        "comparison_probe_namespace_status": "not_implemented_fail_closed",
        "pytest_status": "passed",
        "unittest_fallback_status": "not_needed",
        "guardrail_strict_status": "passed",
    }


class StageTruthMachineVerdictTest(unittest.TestCase):
    def test_all_required_flags_pass_means_a(self) -> None:
        verdict = build_machine_verdict(_passing_fields())
        self.assertEqual(verdict["verdict"], "A")
        self.assertFalse(verdict["blocks_step4_formal_preparation"])

    def test_required_negative_failure_forces_c(self) -> None:
        fields = _passing_fields()
        fields["forged_status_rejected"] = False
        verdict = build_machine_verdict(fields)
        self.assertEqual(verdict["verdict"], "C")

    def test_unimplemented_comparison_probe_fail_closed_is_not_p0(self) -> None:
        fields = _passing_fields()
        fields["comparison_probe_namespace_status"] = "not_implemented_fail_closed"
        fields["p1_count"] = 1
        verdict = build_machine_verdict(fields)
        self.assertEqual(verdict["verdict"], "B")
        self.assertFalse(verdict["blocks_step4_formal_preparation"])

    def test_no_tests_ran_cannot_be_a(self) -> None:
        fields = _passing_fields()
        fields["pytest_status"] = "unavailable"
        fields["unittest_fallback_status"] = "not_needed"
        verdict = build_machine_verdict(fields)
        self.assertEqual(verdict["verdict"], "D")


if __name__ == "__main__":
    unittest.main()
