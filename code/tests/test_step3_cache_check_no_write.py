from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.odcr_step3_cache_check import run_cache_check  # noqa: E402


class TestStep3CacheCheckNoWrite(unittest.TestCase):
    def test_cache_check_does_not_touch_latest(self) -> None:
        latest = REPO_ROOT / "runs" / "step3" / "task2" / "latest.json"
        before = latest.stat().st_mtime_ns if latest.is_file() else None
        result = run_cache_check(task_id=2, expected_profile="task2_strong_forward_g1s")
        after = latest.stat().st_mtime_ns if latest.is_file() else None
        self.assertTrue(result["read_only"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
