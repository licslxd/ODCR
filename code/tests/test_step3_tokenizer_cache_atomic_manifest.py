from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402


class _FakeDatasetDict:
    def map(self, _fn, *, num_proc: int, desc: str):
        self.num_proc = num_proc
        self.desc = desc
        return self

    def save_to_disk(self, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "dataset_dict.json").write_text("{}", encoding="utf-8")
        (root / "state.json").write_text(json.dumps({"desc": getattr(self, "desc", "")}), encoding="utf-8")


class Step3TokenizerCacheAtomicManifestTest(unittest.TestCase):
    def test_atomic_builder_publishes_completed_manifest_and_cleans_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache" / "step3" / "tokenizer" / "task2" / "AM_Movies_to_AM_CDs" / "abc"
            fp = {
                "cache_version": step3.ODCR_TOKENIZE_CACHE_VERSION,
                "task_id": 2,
                "source_domain": "AM_Movies",
                "target_domain": "AM_CDs",
                "mode": "train_valid",
                "tokenizer_cache_compat_hash": "abc",
                "data_contract_hash": "data",
                "preprocessing_artifact_hash": "pre",
                "step3_tokenizer_config": {"x": 1},
                "dataset_split_info": {"splits": {"train": {"row_count": 1}}},
                "source_csv_fingerprints": {"train": {"sha256": "a"}},
                "preprocess_latest_run_ids": {"a": "1", "b": "1", "c": "1"},
                "preprocess_manifest_fingerprints": {"a": "x"},
                "preprocess_metrics_verify_fingerprints": {"b": "x"},
                "schema_contract": {"schema": "x"},
                "upstream_gate_hash": "up",
                "tokenizer_cache_compat_payload": {"cache": "payload"},
                "compatibility_key": "abc",
                "fingerprint_hash": "abc",
                "full_run_config_hash": "resolved",
                "source_table_hash": "source-table",
                "train_runtime_config_hash": "train-runtime",
                "optimizer_config_hash": "optim",
                "performance_profile_hash": "perf",
                "record_only_lineage": {"resolved": "x"},
                "profile_artifact_fingerprints": {"profile": "x"},
                "domain_artifact_fingerprints": {"domain": "x"},
                "env": {"embed_dim": 8},
            }
            manifest = step3.build_or_reuse_step3_tokenizer_cache_atomic(
                datasets=_FakeDatasetDict(),
                processor=lambda sample: sample,
                nproc=1,
                cache_dir=str(cache_dir),
                cache_fingerprint="abc",
                cache_fingerprint_payload=fp,
                build_allowed=True,
                rank="test",
                show_datasets_progress=False,
                log_tokenize=False,
                phase="unit",
                log_file=None,
            )
            self.assertTrue((cache_dir / step3.STEP3_TOKENIZE_CACHE_MANIFEST).is_file())
            self.assertTrue((cache_dir / step3.STEP3_TOKENIZE_CACHE_COMPLETED_MARKER).is_file())
            self.assertTrue(manifest["completed"])
            self.assertFalse(list(cache_dir.parent.glob(f"{cache_dir.name}.partial.*")))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
