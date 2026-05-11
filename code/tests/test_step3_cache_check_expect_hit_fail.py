from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.odcr_step3_cache_check import run_cache_check  # noqa: E402


class TestStep3CacheCheckExpectHitFail(unittest.TestCase):
    def test_expect_cache_hit_fails_for_missing_namespace(self) -> None:
        with self.assertRaises(SystemExit):
            run_cache_check(task_id=8, expected_profile="task8_weak_forward_init", expect_cache_hit=True)


if __name__ == "__main__":
    unittest.main()
