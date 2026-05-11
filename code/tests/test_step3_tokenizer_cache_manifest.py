from __future__ import annotations

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

from executors import step3_train_core as step3  # noqa: E402


class _DummyTokenizer:
    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path


class Step3TokenizerCacheManifestTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    def _env(
        self,
        repo: Path,
        model_dir: Path,
        meta_dir: Path,
        *,
        embed_dim: int = 8,
        training_row_updates: dict | None = None,
        optimizer_name: str = "adamw",
    ) -> mock._patch_dict:
        training_row = {
            "auxiliary": "AM_Movies",
            "target": "AM_Electronics",
            "batch_size": 8,
            "per_gpu_batch_size": 4,
            "batch_semantics_version": "odcr_no_accum/1",
            "step3_batch_formula": "global_batch_size = per_gpu_batch_size * ddp_world_size",
            "lr": 0.001,
        }
        training_row.update(training_row_updates or {})
        payload = {
            "schema_version": "test",
            "training_row": training_row,
            "runtime_roots": {
                "data_dir": str(repo / "data"),
                "merged_dir": str(repo / "merged"),
                "runs_dir": str(repo / "runs"),
                "cache_dir": str(repo / "cache"),
                "models_dir": str(repo / "models"),
                "step5_text_model": str(model_dir),
                "sentence_embed_model": str(repo / "models" / "sent"),
                "embed_dim": embed_dim,
            },
            "step3_structured_losses": {"evidence": 0.1},
            "step3_optimizer": {"name": optimizer_name},
            "step3_scheduler": {"name": "warmup_cosine"},
        }
        field_sources = {
            "task": "tasks.4",
            "train": "step3.train",
            "step3_structured_losses": "step3.structured_losses",
            "embed_dim": "env.embed_dim",
        }
        return mock.patch.dict(
            os.environ,
            {
                "ODCR_ROOT": str(repo),
                "ODCR_MANIFEST_DIR": str(meta_dir),
                "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON": json.dumps(payload, sort_keys=True),
                "ODCR_CONFIG_FIELD_SOURCES_JSON": json.dumps(field_sources, sort_keys=True),
                "ODCR_RESOLVED_DATA_DIR": str(repo / "data"),
                "ODCR_RESOLVED_MERGED_DIR": str(repo / "merged"),
                "ODCR_RESOLVED_RUNS_DIR": str(repo / "runs"),
                "ODCR_RESOLVED_MODELS_DIR": str(repo / "models"),
                "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(model_dir),
                "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(repo / "models" / "sent"),
                "ODCR_RESOLVED_EMBED_DIM": str(embed_dim),
                "ODCR_TRAINING_SEMANTIC_FINGERPRINT": "train-fp",
                "ODCR_GENERATION_SEMANTIC_FINGERPRINT": "gen-fp",
                "ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT": "runtime-fp",
            },
            clear=False,
        )

    def _repo_fixture(self, repo: Path) -> tuple[Path, Path, Path, Path]:
        train = repo / "merged" / "4" / "aug_train.csv"
        valid = repo / "merged" / "4" / "aug_valid.csv"
        train.parent.mkdir(parents=True, exist_ok=True)
        if not train.exists():
            train.write_text("user_idx,item_idx,rating,explanation\n1,2,5,good\n", encoding="utf-8")
        if not valid.exists():
            valid.write_text("user_idx,item_idx,rating,explanation\n3,4,4,ok\n", encoding="utf-8")
        model_dir = repo / "models" / "flan"
        model_dir.mkdir(parents=True, exist_ok=True)
        if not (model_dir / "tokenizer_config.json").exists():
            (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        meta_dir = repo / "runs" / "step3" / "task4" / "unit" / "meta"
        if not (meta_dir / "source_table.json").exists():
            self._write_json(meta_dir / "source_table.json", {"records": [{"key": "task", "source": "tasks.4"}]})
        if not (meta_dir / "resolved_config.json").exists():
            self._write_json(meta_dir / "resolved_config.json", {"task": {"id": 4}, "train": {"stage": "step3"}})
        return train, valid, model_dir, meta_dir

    def _upstream_evidence(self) -> dict:
        preprocess = {}
        for unit in ("a", "b", "c"):
            preprocess[unit] = {
                "run_id": "1",
                "fingerprint_hash": f"fp-{unit}",
                "run_summary_fingerprint": {"path": f"runs/preprocess/{unit}/1/meta/run_summary.json", "sample_sha256": f"rs-{unit}"},
                "stage_status_fingerprint": {"path": f"runs/preprocess/{unit}/1/meta/stage_status.json", "sample_sha256": f"ss-{unit}"},
                "stage_manifest_fingerprint": {"path": f"runs/preprocess/{unit}/1/meta/stage_manifest.json", "sample_sha256": f"sm-{unit}"},
                "source_table_fingerprint": {"path": f"runs/preprocess/{unit}/1/meta/source_table.json", "sample_sha256": f"st-{unit}"},
                "metrics_fingerprint": {"path": f"runs/preprocess/{unit}/1/meta/metrics.json", "sample_sha256": f"mt-{unit}"} if unit in {"b", "c"} else None,
                "verify_report_fingerprint": {"path": f"runs/preprocess/{unit}/1/meta/verify_report.json", "sample_sha256": f"vr-{unit}"} if unit in {"b", "c"} else None,
            }
        return {
            "schema_version": "odcr_step3_upstream_gate/1",
            "contract_schema_version": "odcr_step3_preprocess_upstream_contract/1",
            "fingerprint_hash": "upstream-hash",
            "preprocess": preprocess,
            "source_csv_artifacts": {
                "AM_Movies": {"sample_sha256": "source-a"},
                "AM_Electronics": {"sample_sha256": "source-b"},
            },
            "merged_artifacts": {
                "aug_train_csv": {"sample_sha256": "merged-train"},
                "aug_valid_csv": {"sample_sha256": "merged-valid"},
            },
            "profile_artifact_fingerprints": {
                "AM_Movies": {"user_content_profiles": {"sample_sha256": "profile-a"}},
                "AM_Electronics": {"user_content_profiles": {"sample_sha256": "profile-b"}},
            },
            "domain_artifact_fingerprints": {
                "AM_Movies": {"domain_content": {"sample_sha256": "domain-a"}},
                "AM_Electronics": {"domain_content": {"sample_sha256": "domain-b"}},
            },
        }

    def _fingerprint(
        self,
        repo: Path,
        tokenizer_name: str = "dummy-tokenizer",
        *,
        max_length: int = 48,
        evidence_length: int = 48,
        training_row_updates: dict | None = None,
        optimizer_name: str = "adamw",
    ) -> tuple[dict, Path]:
        train, valid, model_dir, meta_dir = self._repo_fixture(repo)
        with self._env(repo, model_dir, meta_dir, training_row_updates=training_row_updates, optimizer_name=optimizer_name):
            fp = step3._build_tokenize_cache_fingerprint(
                train_path=str(train),
                valid_path=str(valid),
                task_idx=4,
                source_domain="AM_Movies",
                target_domain="AM_Electronics",
                mode="train_valid",
                split_row_counts={"train": 1, "valid": 1},
                upstream_evidence=self._upstream_evidence(),
                tok=_DummyTokenizer(tokenizer_name),
                max_length=max_length,
                evidence_length=evidence_length,
                cache_version=step3.ODCR_TOKENIZE_CACHE_VERSION,
            )
        return fp, model_dir

    def _cache_dir(self, repo: Path) -> Path:
        cache_dir = repo / "cache" / "step3" / "tokenizer" / "task4" / "AM_Movies_to_AM_Electronics" / "toy_cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "dataset_dict.json").write_text("{}", encoding="utf-8")
        (cache_dir / "state.json").write_text('{"splits":["train","valid"]}', encoding="utf-8")
        return cache_dir

    def _write_manifest(self, repo: Path, fp: dict) -> Path:
        cache_dir = self._cache_dir(repo)
        step3._write_step3_tokenize_cache_manifest(str(cache_dir), fingerprint=fp)
        return cache_dir

    def _mutate_manifest(self, cache_dir: Path, mutate) -> None:
        manifest_path = Path(step3._step3_tokenize_cache_manifest_path(str(cache_dir)))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mutate(manifest)
        self._write_json(manifest_path, manifest)

    def test_manifest_gate_accepts_matching_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertTrue(ok, reason)

    def test_cache_path_uses_one_control_formal_namespace(self) -> None:
        class _Processor:
            max_length = 48
            evidence_length = 48

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            train, valid, model_dir, meta_dir = self._repo_fixture(repo)
            with self._env(repo, model_dir, meta_dir):
                cache_dir, cache_key, _payload = step3._build_step3_cache_dir(
                    4,
                    str(train),
                    str(valid),
                    _Processor(),
                    _DummyTokenizer("dummy-tokenizer"),
                    source_domain="AM_Movies",
                    target_domain="AM_Electronics",
                    split_row_counts={"train": 1, "valid": 1},
                    upstream_evidence=self._upstream_evidence(),
                )
            self.assertIn("/cache/step3/tokenizer/task4/AM_Movies_to_AM_Electronics/", cache_dir)
            self.assertTrue(Path(cache_dir).name.startswith(step3.ODCR_TOKENIZE_CACHE_VERSION))
            self.assertEqual(Path(cache_dir).name, cache_key)
            self.assertNotIn("/cache/task4/hf/", cache_dir)

    def test_manifest_gate_rejects_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._cache_dir(repo)
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "missing_manifest")

    def test_dataset_dict_alone_is_not_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._cache_dir(repo)
            self.assertTrue(step3._hf_dataset_cache_ready(str(cache_dir)))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "missing_manifest")

    def test_manifest_gate_rejects_old_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(cache_dir, lambda manifest: manifest.__setitem__("manifest_schema_version", "odcr_step3_tokenizer_cache/1"))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "retired_v1_schema_rebuild_required")

    def test_manifest_gate_rejects_completed_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(cache_dir, lambda manifest: manifest.__setitem__("completed", False))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "completed_false")

    def test_manifest_gate_rejects_failed_marker_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            (cache_dir / step3.STEP3_TOKENIZE_CACHE_FAILED_MARKER).write_text("{}", encoding="utf-8")
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "failed_marker_present")

    def test_manifest_gate_rejects_partial_dir_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = repo / "cache" / "step3" / "tokenizer" / "task4" / "AM_Movies_to_AM_Electronics" / "toy_cache"
            partial = cache_dir.parent / f"{cache_dir.name}.partial.123"
            partial.mkdir(parents=True)
            (partial / "dataset_dict.json").write_text("{}", encoding="utf-8")
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "partial_dir_only")

    def test_manifest_gate_rejects_tokenizer_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo, tokenizer_name="tok-a")
            expected, _ = self._fingerprint(repo, tokenizer_name="tok-b")
            cache_dir = self._write_manifest(repo, fp)
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=expected)
            self.assertFalse(ok)
            self.assertEqual(reason, "tokenizer_cache_compat_hash_mismatch")

    def test_manifest_gate_rejects_source_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(
                cache_dir,
                lambda manifest: manifest["source_csv_fingerprints"]["merged_csvs"]["train"].__setitem__("sha256", "bad"),
            )
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "source_csv_fingerprints_mismatch")

    def test_manifest_gate_rejects_preprocess_run_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(
                cache_dir,
                lambda manifest: manifest["preprocess_latest_run_ids"].__setitem__("b", "2"),
            )
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "preprocess_latest_run_ids_mismatch")

    def test_manifest_gate_records_profile_domain_fingerprint_mismatch_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(
                cache_dir,
                lambda manifest: manifest["profile_artifact_fingerprints"]["AM_Movies"]["user_content_profiles"].__setitem__("sample_sha256", "bad"),
            )
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "hit_record_only_mismatch")
            decision = step3._step3_tokenize_cache_manifest_decision(str(cache_dir), expected_fingerprint=fp)
            self.assertIn("profile_artifact_fingerprints", decision["record_only_mismatches"])

    def test_manifest_gate_records_resolved_config_hash_mismatch_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(cache_dir, lambda manifest: manifest.__setitem__("full_run_config_hash", "bad"))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "hit_record_only_mismatch")
            decision = step3._step3_tokenize_cache_manifest_decision(str(cache_dir), expected_fingerprint=fp)
            self.assertIn("full_run_config_hash", decision["record_only_mismatches"])

    def test_manifest_gate_records_source_table_hash_mismatch_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(cache_dir, lambda manifest: manifest.__setitem__("source_table_hash", "bad"))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "hit_record_only_mismatch")
            decision = step3._step3_tokenize_cache_manifest_decision(str(cache_dir), expected_fingerprint=fp)
            self.assertIn("source_table_hash", decision["record_only_mismatches"])

    def test_manifest_gate_records_embed_dim_mismatch_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            self._mutate_manifest(cache_dir, lambda manifest: manifest["env"].__setitem__("embed_dim", 99))
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "hit_record_only_mismatch")
            decision = step3._step3_tokenize_cache_manifest_decision(str(cache_dir), expected_fingerprint=fp)
            self.assertIn("env", decision["record_only_mismatches"])

    def test_training_runtime_changes_do_not_change_tokenizer_compat_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            base, _ = self._fingerprint(repo)
            for update in (
                {"batch_size": 16},
                {"per_gpu_batch_size": 8, "batch_size": 16},
                {"batch_semantics_version": "odcr_no_accum/1"},
                {"lr": 0.01},
            ):
                changed, _ = self._fingerprint(repo, training_row_updates=update)
                self.assertEqual(changed["tokenizer_cache_compat_hash"], base["tokenizer_cache_compat_hash"])
            changed_optimizer, _ = self._fingerprint(repo, optimizer_name="sgd")
            self.assertEqual(changed_optimizer["tokenizer_cache_compat_hash"], base["tokenizer_cache_compat_hash"])

    def test_tokenization_inputs_change_tokenizer_compat_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            base, _ = self._fingerprint(repo)
            for kwargs in (
                {"max_length": 64},
                {"evidence_length": 64},
                {"tokenizer_name": "other-tokenizer"},
            ):
                changed, _ = self._fingerprint(repo, **kwargs)
                self.assertNotEqual(changed["tokenizer_cache_compat_hash"], base["tokenizer_cache_compat_hash"])

    def test_manifest_gate_rejects_content_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp, _ = self._fingerprint(repo)
            cache_dir = self._write_manifest(repo, fp)
            (cache_dir / "state.json").write_text('{"splits":["changed"]}', encoding="utf-8")
            ok, reason = step3._step3_tokenize_cache_manifest_matches(str(cache_dir), expected_fingerprint=fp)
            self.assertFalse(ok)
            self.assertEqual(reason, "cache_content_fingerprint_mismatch")

    def test_pre_ddp_cache_contract_has_no_distributed_collective(self) -> None:
        core_text = Path(step3.__file__).read_text(encoding="utf-8")
        cache_start = core_text.index("def build_or_reuse_step3_tokenizer_cache_atomic")
        cache_end = core_text.index("def _load_step3_artefacts", cache_start)
        cache_block = core_text[cache_start:cache_end]
        self.assertIn("wait_for_completed_cache_manifest_file_polling", cache_block)
        self.assertIn("save_to_disk(partial_dir)", cache_block)
        self.assertNotIn("dist.barrier", cache_block)
        self.assertNotIn("dist.all_reduce", cache_block)
        self.assertNotIn("save_to_disk(cache_dir)", cache_block)


if __name__ == "__main__":
    unittest.main()
