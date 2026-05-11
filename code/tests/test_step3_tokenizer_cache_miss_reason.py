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

from executors import step3_train_core as step3  # noqa: E402
from test_step3_tokenizer_cache_manifest import Step3TokenizerCacheManifestTest  # noqa: E402


class TestStep3TokenizerCacheMissReason(unittest.TestCase):
    def test_tokenizer_change_reports_hard_miss_reason(self) -> None:
        helper = Step3TokenizerCacheManifestTest(methodName="run")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = helper._fingerprint(repo, tokenizer_name="tok-a")
            expected, _ = helper._fingerprint(repo, tokenizer_name="tok-b")
            cache_dir = helper._write_manifest(repo, fp)
            decision = step3._step3_tokenize_cache_manifest_decision(str(cache_dir), expected_fingerprint=expected)
            self.assertFalse(decision["would_hit_cache"])
            self.assertEqual(decision["miss_reason"], "tokenizer_cache_compat_hash_mismatch")
            self.assertIn("tokenizer_cache_compat_hash", decision["rejected_fields"])


if __name__ == "__main__":
    unittest.main()
