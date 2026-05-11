from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.stage2_runtime_first import (  # noqa: E402
    CANDIDATE_PROBES,
    PREREQUISITE_RUNTIME_PROBES,
    select_stage2_candidate,
)


def _ok() -> dict[str, object]:
    return {
        "runtime_verified": True,
        "evidence_complete": True,
        "formal_namespace_polluted": False,
    }


class Stage2CandidateSelectionUsesRuntimeEvidenceTest(unittest.TestCase):
    def test_no_prerequisite_evidence_means_no_candidate(self) -> None:
        verdict = select_stage2_candidate(prerequisite_results={}, candidate_results={})
        self.assertEqual(verdict.verdict, "B")
        self.assertIsNone(verdict.selected)
        self.assertFalse(verdict.runtime_verified)

    def test_candidate_selection_requires_candidate_runtime_evidence(self) -> None:
        prereq = {name: _ok() for name in PREREQUISITE_RUNTIME_PROBES}
        verdict = select_stage2_candidate(prerequisite_results=prereq, candidate_results={})
        self.assertEqual(verdict.verdict, "B")
        self.assertIsNone(verdict.selected)
        self.assertFalse(verdict.evidence_complete)

    def test_post_edit_flaky_does_not_erase_runtime_evidence(self) -> None:
        prereq = {name: _ok() for name in PREREQUISITE_RUNTIME_PROBES}
        candidates = {name: _ok() for name in CANDIDATE_PROBES}
        verdict = select_stage2_candidate(
            prerequisite_results=prereq,
            candidate_results=candidates,
            post_edit_diagnostic={"classification": "flaky_resource_kill"},
        )
        self.assertEqual(verdict.verdict, "A")
        self.assertEqual(verdict.selected, "G1-M")

    def test_semantic_p0_blocks_formal_candidate(self) -> None:
        prereq = {name: _ok() for name in PREREQUISITE_RUNTIME_PROBES}
        candidates = {name: _ok() for name in CANDIDATE_PROBES}
        verdict = select_stage2_candidate(
            prerequisite_results=prereq,
            candidate_results=candidates,
            post_edit_diagnostic={"classification": "P0_semantic_blocker"},
        )
        self.assertEqual(verdict.verdict, "C")
        self.assertIsNone(verdict.selected)


if __name__ == "__main__":
    unittest.main()
