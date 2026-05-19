from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tools.odcr_step3_cache_check as cache_check  # noqa: E402


class TestStep3CacheCheckNoWrite(unittest.TestCase):
    def test_cache_check_does_not_touch_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            latest = repo / "runs" / "step3" / "task2" / "latest.json"
            latest.parent.mkdir(parents=True)
            latest.write_text('{"latest_run_id":"1"}\n', encoding="utf-8")
            before = latest.stat().st_mtime_ns
            old_root = cache_check.REPO_ROOT
            try:
                cache_check.REPO_ROOT = repo
                result = cache_check.run_cache_check(
                    task_id=2,
                    expected_profile="task2_strong_forward_g1s",
                    resolved_snapshot={
                        "task": {
                            "source": "A",
                            "target": "T",
                            "task_profile_id": "task2_strong_forward_g1s",
                        },
                        "hardware": {
                            "tokenization_num_proc": 8,
                            "reserved_cpu": 2,
                            "max_parallel_cpu": 12,
                        },
                    },
                )
            finally:
                cache_check.REPO_ROOT = old_root
            after = latest.stat().st_mtime_ns
        self.assertTrue(result["read_only"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
