from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import resolve_config  # noqa: E402


class TestStep3G2CNotFormal(unittest.TestCase):
    def test_g1m_and_g2c_remain_probe_only(self) -> None:
        _cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        ladder = snapshot["step3_performance_candidates"]["batch_ladder"]
        for name in ("G1-M", "G2-C"):
            self.assertFalse(ladder[name]["formal_allowed"])
            self.assertTrue(ladder[name]["probe_only"])


if __name__ == "__main__":
    unittest.main()
