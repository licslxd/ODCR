from __future__ import annotations

import unittest
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.governance.rule_registry import RULE_GROUP_BY_ID, hook_scope_for_path, suggest_scope_for_paths


class AuxGovernanceRegistryTest(unittest.TestCase):
    def test_rule_groups_are_single_source(self) -> None:
        self.assertEqual(RULE_GROUP_BY_ID["R051"], "post-edit-workflow")
        self.assertEqual(RULE_GROUP_BY_ID["R115"], "step4-runtime-preflight")

    def test_scope_helpers_cover_hook_and_post_edit(self) -> None:
        self.assertEqual(hook_scope_for_path("code/executors/step4_engine.py"), "step4")
        self.assertEqual(hook_scope_for_path("code/data_contract.py"), "all")
        self.assertEqual(suggest_scope_for_paths(["code/odcr_core/path_layout.py"]), "logging")


if __name__ == "__main__":
    unittest.main()
