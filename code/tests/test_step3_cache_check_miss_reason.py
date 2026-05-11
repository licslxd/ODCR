from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.odcr_step3_cache_check import run_cache_check  # noqa: E402


class TestStep3CacheCheckMissReason(unittest.TestCase):
    def test_missing_task_namespace_reports_miss_reason(self) -> None:
        result = run_cache_check(task_id=8, expected_profile="task8_weak_forward_init", allow_cold_build=True)
        self.assertIn("miss_reason", result)
        if not result["would_hit_cache"]:
            self.assertTrue(result["miss_reason"])


if __name__ == "__main__":
    unittest.main()
