from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step4_engine
from odcr_core import step4_runtime


class Step4CachePreDdpNoNcclTest(unittest.TestCase):
    def test_prepare_cache_helper_has_no_torch_distributed_collective(self) -> None:
        src = inspect.getsource(step4_runtime.prepare_step4_encoded_cache)
        forbidden = ("init_process_group", "dist.barrier", "all_reduce", "broadcast", "all_gather")
        for token in forbidden:
            self.assertNotIn(token, src)
        self.assertIn("no_torch_distributed_collective", src)

    def test_ddp_runtime_refuses_cold_cache_build(self) -> None:
        src = inspect.getsource(step4_engine._run_one_task)
        self.assertIn("cold_build_allowed=False", src)
        self.assertIn("Step4 encoded cache is not ready before DDP inference", src)
        self.assertNotIn("target_dataset.map", src)


if __name__ == "__main__":
    unittest.main()
