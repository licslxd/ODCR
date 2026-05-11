from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
for path in (CODE_DIR, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_step3_tokenizer_cache_manifest import Step3TokenizerCacheManifestTest  # noqa: E402


class TestStep3TokenizerCacheIgnoresTrainingParams(unittest.TestCase):
    def test_lr_batch_optimizer_do_not_change_compat_hash(self) -> None:
        helper = Step3TokenizerCacheManifestTest(methodName="run")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            base, _ = helper._fingerprint(repo)
            changed, _ = helper._fingerprint(
                repo,
                training_row_updates={"batch_size": 1536, "per_gpu_batch_size": 768, "lr": 0.0007},
                optimizer_name="sgd",
            )
            self.assertEqual(base["tokenizer_cache_compat_hash"], changed["tokenizer_cache_compat_hash"])


if __name__ == "__main__":
    unittest.main()
