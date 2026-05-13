from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


class TestStep3G1BackupNotDefault(unittest.TestCase):
    def test_g1_is_backup_only(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        self.assertEqual(snapshot["train"]["candidate"], "G1S-sidecar-stable")
        g1 = snapshot["step3_backup_profiles"]["task2_g1_backup"]
        self.assertEqual(g1["candidate"], "G1")
        self.assertTrue(g1["backup_only"])
        self.assertTrue(g1["manual_selection_required"])
        self.assertFalse(g1["formal_allowed"])


if __name__ == "__main__":
    unittest.main()
