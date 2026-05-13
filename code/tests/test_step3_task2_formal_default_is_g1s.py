from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


class TestStep3Task2FormalDefaultIsG1S(unittest.TestCase):
    def test_task2_resolves_g1s_as_formal_default(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        self.assertEqual(cfg.task_profile_id, "task2_strong_forward_g1s")
        self.assertEqual(snapshot["train"]["candidate"], "G1S-sidecar-stable")
        self.assertEqual(snapshot["step3_task_profile"]["candidate"], "G1S")
        self.assertTrue(snapshot["step3_task_profile"]["formal_allowed"])
        self.assertFalse(snapshot["step3_task_profile"]["probe_only"])


if __name__ == "__main__":
    unittest.main()
