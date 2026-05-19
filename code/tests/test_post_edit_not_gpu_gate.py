from __future__ import annotations

import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from tools.odcr_post_edit_check import (  # noqa: E402
    CheckCommand,
    classify_exit,
    classification_blocks_formal,
    classification_blocks_gpu_probe,
    post_edit_results_block_formal,
    post_edit_results_block_gpu_probe,
    run_commands,
)


class PostEditNotGpuGateTest(unittest.TestCase):
    def test_exit_minus_9_classified_as_resource_kill_not_gpu_gate(self) -> None:
        classification, sig = classify_exit(-9)
        self.assertEqual(classification, "resource_kill")
        self.assertEqual(sig, signal.SIGKILL)
        self.assertFalse(classification_blocks_gpu_probe(classification))
        self.assertTrue(classification_blocks_formal(classification))

    def test_semantic_p0_can_block_formal_but_not_gpu_probe(self) -> None:
        self.assertFalse(classification_blocks_gpu_probe("P0_semantic_blocker"))
        self.assertTrue(classification_blocks_formal("P0_semantic_blocker"))

    def test_timeout_fails_closed_for_formal_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_commands(
                [
                    CheckCommand(
                        "sleeping command",
                        (sys.executable, "-c", "import time; time.sleep(2)"),
                    )
                ],
                repo_root=root,
                max_seconds=1,
            )[0]
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.classification, "timeout")
        self.assertTrue(result.blocks_formal)
        self.assertFalse(result.blocks_gpu_probe)

    def test_resource_kill_fails_closed_without_rerun_pass_masking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "code" / "tests"
            test_dir.mkdir(parents=True)
            marker = root / "marker"
            test_file = test_dir / "test_flaky.py"
            test_file.write_text(
                "import os, signal, pathlib\n"
                f"m = pathlib.Path({str(marker)!r})\n"
                "if not m.exists():\n"
                "    m.write_text('seen')\n"
                "    os.kill(os.getpid(), signal.SIGKILL)\n",
                encoding="utf-8",
            )
            result = run_commands(
                [CheckCommand("code/tests/test_flaky.py", (sys.executable, "code/tests/test_flaky.py"))],
                repo_root=root,
                max_seconds=10,
            )[0]
        self.assertEqual(result.classification, "resource_kill")
        self.assertEqual(result.status, "FAIL")
        self.assertFalse(post_edit_results_block_gpu_probe([result]))
        self.assertTrue(post_edit_results_block_formal([result]))

    def test_resource_kill_rerun_fail_becomes_semantic_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "code" / "tests"
            test_dir.mkdir(parents=True)
            marker = root / "marker"
            test_file = test_dir / "test_flaky_fail.py"
            test_file.write_text(
                "import os, signal, pathlib, sys\n"
                f"m = pathlib.Path({str(marker)!r})\n"
                "if not m.exists():\n"
                "    m.write_text('seen')\n"
                "    os.kill(os.getpid(), signal.SIGKILL)\n"
                "sys.exit(1)\n",
                encoding="utf-8",
            )
            result = run_commands(
                [CheckCommand("code/tests/test_flaky_fail.py", (sys.executable, "code/tests/test_flaky_fail.py"))],
                repo_root=root,
                max_seconds=10,
            )[0]
        self.assertEqual(result.classification, "resource_kill")
        self.assertTrue(result.blocks_formal)
        self.assertFalse(result.blocks_gpu_probe)


if __name__ == "__main__":
    unittest.main()
