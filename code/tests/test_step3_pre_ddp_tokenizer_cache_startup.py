from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402
from odcr_core import runners  # noqa: E402


class Step3PreDdpTokenizerCacheStartupTest(unittest.TestCase):
    def test_parent_builds_tokenizer_cache_before_torchrun(self) -> None:
        src = inspect.getsource(runners._run_step3_train)
        self.assertLess(
            src.index("_ensure_step3_pre_ddp_tokenizer_cache"),
            src.index("_run_torchrun"),
        )

    def test_train_ddp_does_not_init_process_group_before_data_build(self) -> None:
        src = inspect.getsource(step3._run_train_ddp)
        self.assertNotIn("dist.init_process_group", src)
        self.assertIn("build_config_and_data_ddp", src)

    def test_data_builder_initializes_ddp_after_cache_load(self) -> None:
        src = inspect.getsource(step3.build_config_and_data_ddp)
        self.assertLess(
            src.index("_load_step3_artefacts"),
            src.index("init_step3_ddp_after_cache_ready"),
        )
        self.assertLess(
            src.index("init_step3_ddp_after_cache_ready"),
            src.index("DistributedDataParallel"),
        )


if __name__ == "__main__":
    unittest.main()
