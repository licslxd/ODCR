from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_rating_seed_runner import Step3RatingSeedRunnerError, build_rating_seed_plan  # noqa: E402


class Step3RatingSeedRunnerNoLegacyModesTest(unittest.TestCase):
    def test_legacy_modes_are_rejected(self) -> None:
        for mode in (
            "missing4seeds",
            "run_missing",
            "multiseed_run_missing",
            "multiseed_plan",
            "strict_rerun_all",
            "reuse_existing",
            "coverage",
        ):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(Step3RatingSeedRunnerError, "single or multi"):
                    build_rating_seed_plan(task=2, mode=mode, run_id_start=10)

    def test_old_task2_missing_launcher_removed(self) -> None:
        self.assertFalse((ROOT / "test" / "step3_task2_missing4seeds_gpu_launcher.sh").exists())
        self.assertFalse((ROOT / "test" / "step3_rating_task2_multi_run10_5seed_gpu_launcher.sh").exists())


if __name__ == "__main__":
    unittest.main()
