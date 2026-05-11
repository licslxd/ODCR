from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.odcr_step3_cache_check import run_cache_check  # noqa: E402


class TestStep3CacheCheckSelectedNumProc(unittest.TestCase):
    def test_cache_check_reports_rebuild_num_proc_8(self) -> None:
        result = run_cache_check(task_id=2, expected_profile="task2_strong_forward_g1s", expect_num_proc=8)
        self.assertEqual(result["selected_num_proc_if_rebuild"], 8)
        self.assertIn("max_parallel_cpu(12)", result["num_proc_formula"])


if __name__ == "__main__":
    unittest.main()
