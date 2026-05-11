from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


class TestStep3TokenizationNumProcAuto12Core(unittest.TestCase):
    def test_auto_selects_8_on_12_core_default(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        self.assertEqual(cfg.num_proc, 8)
        self.assertEqual(snapshot["hardware"]["tokenization_num_proc"], 8)
        self.assertEqual(snapshot["hardware"]["worker_budget_formula"]["tokenization_active_processes"], 10)


if __name__ == "__main__":
    unittest.main()
