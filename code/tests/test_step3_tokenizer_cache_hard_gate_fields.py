from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402


class TestStep3TokenizerCacheHardGateFields(unittest.TestCase):
    def test_training_and_lineage_hashes_are_not_hard_gate_fields(self) -> None:
        gate_source = Path(step3.__file__).read_text(encoding="utf-8")
        start = gate_source.index("def _step3_tokenize_cache_manifest_gate_fields")
        end = gate_source.index("def _step3_tokenize_cache_manifest_sections")
        block = gate_source[start:end]
        for record_only in (
            "full_run_config_hash",
            "resolved_config",
            "source_table_hash",
            "train_runtime_config_hash",
            "optimizer_config_hash",
            "performance_profile_hash",
            "task_profile_id",
            "profile_artifact_fingerprints",
            "domain_artifact_fingerprints",
        ):
            self.assertNotIn(record_only, block)
        self.assertIn("tokenizer_cache_compat_hash", block)


if __name__ == "__main__":
    unittest.main()
