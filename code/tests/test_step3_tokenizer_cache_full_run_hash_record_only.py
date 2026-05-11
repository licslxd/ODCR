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


class TestStep3TokenizerCacheFullRunHashRecordOnly(unittest.TestCase):
    def test_full_run_config_hash_mismatch_hits_cache(self) -> None:
        helper = Step3TokenizerCacheManifestTest(methodName="run")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = helper._fingerprint(repo)
            cache_dir = helper._write_manifest(repo, fp)
            helper._mutate_manifest(cache_dir, lambda manifest: manifest.__setitem__("full_run_config_hash", "other-run"))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "hit_record_only_mismatch")


if __name__ == "__main__":
    unittest.main()
