from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class AuxRuntimeCliTest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "code/odcr.py", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )

    def test_runtime_help_surfaces_bridge_and_probe(self) -> None:
        proc = self._run("runtime", "--help")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("bridge", proc.stdout)
        self.assertIn("probe", proc.stdout)

    def test_runtime_bridge_help_surfaces_registered_modes(self) -> None:
        proc = self._run("runtime", "bridge", "--help")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        for mode in ("validate-only", "marker-probe", "cuda-probe"):
            self.assertIn(mode, proc.stdout)

    def test_runtime_probe_help_surfaces_all_stages(self) -> None:
        proc = self._run("runtime", "probe", "--help")
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("preprocess", proc.stdout)
        self.assertIn("step3", proc.stdout)
        self.assertIn("--bounded", proc.stdout)


if __name__ == "__main__":
    unittest.main()

