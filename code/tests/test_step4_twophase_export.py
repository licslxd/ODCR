from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step4_engine


class Step4TwoPhaseExportTest(unittest.TestCase):
    def test_cpu_export_occurs_after_process_group_destroy(self) -> None:
        src = inspect.getsource(step4_engine._run_one_task)
        destroy_idx = src.index("destroy_process_group_before_cpu_export=True")
        export_idx = src.index("build_step4_train_manifest")
        self.assertLess(destroy_idx, export_idx)
        self.assertIn("non_rank0_gpu_released_before_cpu_tail=True", src)
        self.assertNotIn("dist.gather_object", src)
        self.assertNotIn("all_gather_object", src)


if __name__ == "__main__":
    unittest.main()
