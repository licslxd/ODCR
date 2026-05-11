from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


class TestStage2SelectedCandidateBoundToResolver(unittest.TestCase):
    def test_performance_candidate_matrix_binds_g1s(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        matrix = snapshot["step3_performance_candidates"]
        self.assertEqual(matrix["selected_candidate"], "G1S")
        self.assertEqual(matrix["formal_default_profile"], "task2_strong_forward_g1s")
        self.assertTrue(matrix["batch_ladder"]["G1S"]["formal_allowed"])
        self.assertFalse(matrix["batch_ladder"]["G1S"]["probe_only"])


if __name__ == "__main__":
    unittest.main()
