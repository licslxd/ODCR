import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step5_engine


class _DummyTokenizer:
    name_or_path = "dummy-tokenizer"


class Step5CacheManifestTest(unittest.TestCase):
    def _env(self, repo: Path, model_dir: Path) -> mock._patch_dict:
        payload = {
            "schema_version": "test",
            "runtime_roots": {
                "step5_text_model": str(model_dir),
                "data_dir": str(repo / "data"),
                "merged_dir": str(repo / "merged"),
                "models_dir": str(repo / "models"),
                "sentence_embed_model": str(repo / "models" / "sent"),
                "embed_dim": 8,
            },
            "step5_innovation": {
                "lci": {"weight": 0.1},
                "uci": {},
                "ccv": {},
                "fca": {"weight": 0.2},
                "explainer_gate": {},
            },
        }
        return mock.patch.dict(
            os.environ,
            {
                "ODCR_ROOT": str(repo),
                "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON": json.dumps(payload, sort_keys=True),
                "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(model_dir),
                "ODCR_RESOLVED_DATA_DIR": str(repo / "data"),
                "ODCR_RESOLVED_MERGED_DIR": str(repo / "merged"),
                "ODCR_RESOLVED_MODELS_DIR": str(repo / "models"),
                "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(repo / "models" / "sent"),
                "ODCR_RESOLVED_EMBED_DIM": "8",
            },
            clear=False,
        )

    def _fingerprint(self, repo: Path) -> dict:
        train = repo / "step4_export" / "train.csv"
        valid = repo / "step4_export" / "valid.csv"
        index_contract = repo / "step4_export" / "index_contract.json"
        train.parent.mkdir(parents=True, exist_ok=True)
        train.write_text("global_user_idx,global_item_idx,rating,clean_text\n1,2,5,ok\n", encoding="utf-8")
        valid.write_text("global_user_idx,global_item_idx,rating,clean_text\n3,4,4,ok\n", encoding="utf-8")
        index_contract.write_text(json.dumps({"step4_export_lineage": {"lineage_hash": "abc"}}), encoding="utf-8")
        model_dir = repo / "models" / "flan"
        model_dir.mkdir(parents=True)
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        with self._env(repo, model_dir):
            return step5_engine._build_tokenize_cache_fingerprint(
                train_path=str(train),
                eval_split_path=str(valid),
                task_idx=4,
                split_label="train+valid",
                tok=_DummyTokenizer(),
                max_length=25,
                cache_version=step5_engine.ODCR_TOKENIZE_CACHE_VERSION,
                eval_only=False,
                index_contract_path=str(index_contract),
                step4_export_lineage={"lineage_hash": "abc"},
            )

    def test_manifest_gate_accepts_matching_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp = self._fingerprint(repo)
            cache_dir = repo / "cache" / "step5"
            cache_dir.mkdir(parents=True)
            (cache_dir / "dataset_dict.json").write_text("{}", encoding="utf-8")
            step5_engine._write_step5_tokenize_cache_manifest(
                str(cache_dir),
                fingerprint=fp,
                splits={"train": 1, "valid": 1},
            )
            ok, reason = step5_engine._step5_tokenize_cache_manifest_matches(
                str(cache_dir),
                expected_fingerprint=fp,
            )
            self.assertTrue(ok, reason)

    def test_manifest_gate_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp = self._fingerprint(repo)
            cache_dir = repo / "cache" / "step5"
            cache_dir.mkdir(parents=True)
            (cache_dir / "dataset_dict.json").write_text("{}", encoding="utf-8")
            ok, reason = step5_engine._step5_tokenize_cache_manifest_matches(
                str(cache_dir),
                expected_fingerprint=fp,
            )
            self.assertFalse(ok)
            self.assertEqual(reason, "missing_manifest")

    def test_manifest_gate_rejects_step5_innovation_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp = self._fingerprint(repo)
            cache_dir = repo / "cache" / "step5"
            cache_dir.mkdir(parents=True)
            (cache_dir / "dataset_dict.json").write_text("{}", encoding="utf-8")
            step5_engine._write_step5_tokenize_cache_manifest(
                str(cache_dir),
                fingerprint=fp,
                splits={"train": 1, "valid": 1},
            )
            manifest_path = Path(step5_engine._step5_tokenize_cache_manifest_path(str(cache_dir)))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["step5_innovation_config_hash"] = "bad"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            ok, reason = step5_engine._step5_tokenize_cache_manifest_matches(
                str(cache_dir),
                expected_fingerprint=fp,
            )
            self.assertFalse(ok)
            self.assertEqual(reason, "step5_innovation_config_hash_mismatch")

    def test_eval_only_fingerprint_labels_factual_default_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            train = repo / "step4_export" / "train.csv"
            valid = repo / "step4_export" / "valid.csv"
            index_contract = repo / "step4_export" / "index_contract.json"
            train.parent.mkdir(parents=True, exist_ok=True)
            train.write_text("user_idx_global,item_idx_global,rating,clean_text\n1,2,5,ok\n", encoding="utf-8")
            valid.write_text("user_idx_global,item_idx_global,rating,explanation\n3,4,4,ok\n", encoding="utf-8")
            index_contract.write_text(json.dumps({"step4_export_lineage": {"lineage_hash": "abc"}}), encoding="utf-8")
            model_dir = repo / "models" / "flan"
            model_dir.mkdir(parents=True)
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            with self._env(repo, model_dir):
                fp = step5_engine._build_tokenize_cache_fingerprint(
                    train_path=str(train),
                    eval_split_path=str(valid),
                    task_idx=4,
                    split_label="valid",
                    tok=_DummyTokenizer(),
                    max_length=25,
                    cache_version=step5_engine.ODCR_TOKENIZE_CACHE_VERSION,
                    eval_only=True,
                    index_contract_path=str(index_contract),
                    step4_export_lineage={"lineage_hash": "abc"},
                )
            contract = fp["eval_control_contract"]
            self.assertEqual(contract["mode"], "factual_eval_default")
            self.assertFalse(contract["is_rcr_posterior"])
            self.assertFalse(contract["is_step4_export_posterior"])
            self.assertTrue(fp["eval_control_contract_hash"])


if __name__ == "__main__":
    unittest.main()
