from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestStep3ExpectProfileAssertion(unittest.TestCase):
    def test_expect_profile_passes_for_g1s_and_fails_for_g1(self) -> None:
        ok = subprocess.run(
            [sys.executable, "code/odcr.py", "step3", "--task", "2", "--dry-run", "--expect-profile", "task2_strong_forward_g1s"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        bad = subprocess.run(
            [sys.executable, "code/odcr.py", "step3", "--task", "2", "--dry-run", "--expect-profile", "task2_strong_forward_g1"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("expected task2_strong_forward_g1 but resolved task2_strong_forward_g1s", bad.stderr)


if __name__ == "__main__":
    unittest.main()
