from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.check_one_control_guardrails import RULE_GROUP_BY_ID, run_checks  # noqa: E402

pytestmark = pytest.mark.slow


class NoAccumGuardrailTest(unittest.TestCase):
    def test_r117_no_accum_guardrail_is_strict_and_passing(self) -> None:
        report = run_checks(repo_root=REPO_ROOT, strict=True)
        statuses = {item.rule_id: item.status for item in report.results}
        self.assertEqual(statuses.get("R117"), "PASS")
        self.assertEqual(RULE_GROUP_BY_ID.get("R117"), "no-accum-architecture")


if __name__ == "__main__":
    unittest.main()
