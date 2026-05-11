from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402


class Step3NoDistributedCollectiveInCachePhaseTest(unittest.TestCase):
    def test_cache_phase_functions_do_not_use_distributed_collectives(self) -> None:
        functions = [
            step3.ensure_step3_tokenizer_cache_ready_pre_ddp,
            step3.build_or_reuse_step3_tokenizer_cache_atomic,
            step3.wait_for_completed_cache_manifest_file_polling,
            step3.validate_completed_step3_tokenizer_cache,
            step3.load_completed_step3_tokenizer_cache_for_rank,
            step3._map_tokenize_train_valid_to_hf_cache,
        ]
        for fn in functions:
            with self.subTest(function=fn.__name__):
                src = inspect.getsource(fn)
                self.assertNotIn("dist.barrier", src)
                self.assertNotIn("dist.all_reduce", src)
                self.assertNotIn("broadcast_object_list", src)
                self.assertNotIn("init_process_group", src)


if __name__ == "__main__":
    unittest.main()
