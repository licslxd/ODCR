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


class TestStep3TokenizerCacheReusesAcrossG1G1S(unittest.TestCase):
    def test_profile_record_only_change_does_not_change_compat_hash(self) -> None:
        helper = Step3TokenizerCacheManifestTest(methodName="run")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            g1, _ = helper._fingerprint(repo, training_row_updates={"task_profile_id": "task2_strong_forward_g1"})
            g1s, _ = helper._fingerprint(repo, training_row_updates={"task_profile_id": "task2_strong_forward_g1s"})
            self.assertEqual(g1["tokenizer_cache_compat_hash"], g1s["tokenizer_cache_compat_hash"])


if __name__ == "__main__":
    unittest.main()
