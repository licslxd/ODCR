from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step4_engine


class Step4NoRank0CpuTailWithPgAliveTest(unittest.TestCase):
    def test_rank0_export_after_destroy_and_no_barrier_after(self) -> None:
        src = inspect.getsource(step4_engine._run_one_task)
        destroy = src.index("dist.destroy_process_group()")
        export = src.index("write_step4_training_artifacts")
        self.assertLess(destroy, export)
        self.assertNotIn("dist.barrier()", src[export:])
        self.assertIn("return", src[src.index("if rank != 0:") : export])


if __name__ == "__main__":
    unittest.main()
