from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.odcr_step3_cache_check import run_cache_check  # noqa: E402


class TestStep3CacheCheckHit(unittest.TestCase):
    def test_task2_cache_check_reports_hit(self) -> None:
        result = run_cache_check(task_id=2, expected_profile="task2_strong_forward_g1s", expect_num_proc=8)
        self.assertTrue(result["would_hit_cache"])
        self.assertEqual(result["formal_profile"], "task2_strong_forward_g1s")
        self.assertTrue(result["read_only"])


if __name__ == "__main__":
    unittest.main()
