import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasets import Dataset, DatasetDict


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step5_engine


class _DummyTokenizer:
    name_or_path = "dummy-tokenizer"


class _FakeDist:
    def __init__(self, control):
        self.control = control

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def broadcast_object_list(self, obj, src=0):
        obj[0] = self.control

    def barrier(self):
        return None


class Step5TokenCacheDeterminismTest(unittest.TestCase):
    def _payload(self, selected="CANDIDATE_A"):
        return {
            "schema_version": 3,
            "task_id": 2,
            "preset_name": "step5",
            "training_row": {"lr": 0.001, "coef": 0.1, "epochs": 3},
            "runtime_roots": {},
            "step5_head": "step5A",
            "selected_tuning_candidate": selected,
            "step5_effective_samples": {"step5A": 190646},
            "step5_optimizer_steps": {"step5A": 426},
            "step5_innovation": {"lci": {"weight": 0.12}},
            "step5_sampler": {"seed": 3407},
            "step5_tuning": {"selected_tuning_candidate": selected, "batch_candidate": "B224"},
            "step5_formal_active_candidate": {
                "selected_tuning_candidate": selected,
                "batch_candidate": "B224",
            },
        }

    def _env(self, repo: Path, model_dir: Path, *, selected="CANDIDATE_A"):
        return mock.patch.dict(
            os.environ,
            {
                "ODCR_ROOT": str(repo),
                "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON": json.dumps(self._payload(selected), sort_keys=True),
                "ODCR_RESOLVED_STEP5_TEXT_MODEL": str(model_dir),
                "ODCR_RESOLVED_DATA_DIR": str(repo / "data"),
                "ODCR_RESOLVED_MERGED_DIR": str(repo / "merged"),
                "ODCR_RESOLVED_MODELS_DIR": str(repo / "models"),
                "ODCR_RESOLVED_SENTENCE_EMBED_MODEL": str(repo / "models" / "sent"),
                "ODCR_RESOLVED_EMBED_DIM": "1024",
                "ODCR_TRAINING_SEMANTIC_FINGERPRINT": "train-sem",
                "ODCR_GENERATION_SEMANTIC_FINGERPRINT": "gen-sem",
                "ODCR_RUNTIME_DIAGNOSTICS_FINGERPRINT": "runtime-must-not-enter-token-cache",
            },
            clear=False,
        )

    def _repo_files(self, repo: Path, *, tokenizer_text="{}"):
        train = repo / "step4" / "odcr_routing_train.csv"
        valid = repo / "data" / "AM_CDs" / "valid.csv"
        index_contract = repo / "step4" / "index_contract.json"
        train.parent.mkdir(parents=True, exist_ok=True)
        valid.parent.mkdir(parents=True, exist_ok=True)
        train.write_text("user_idx_global,item_idx_global,rating,clean_text\n1,2,5,ok\n", encoding="utf-8")
        valid.write_text("user_idx_global,item_idx_global,rating,explanation\n3,4,4,ok\n", encoding="utf-8")
        index_contract.write_text(json.dumps({"step4_run": "1"}), encoding="utf-8")
        model_dir = repo / "models" / "flan"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "tokenizer_config.json").write_text(tokenizer_text, encoding="utf-8")
        return train, valid, index_contract, model_dir

    def _fingerprint(
        self,
        repo: Path,
        *,
        sampler_summary=None,
        runtime_diagnostics=None,
        selected="CANDIDATE_A",
        tokenizer_text="{}",
        task_head="step5A",
    ):
        train, valid, index_contract, model_dir = self._repo_files(repo, tokenizer_text=tokenizer_text)
        with self._env(repo, model_dir, selected=selected):
            return step5_engine._build_tokenize_cache_fingerprint(
                train_path=str(train),
                eval_split_path=str(valid),
                task_idx=2,
                split_label="train+valid",
                tok=_DummyTokenizer(),
                max_length=36,
                cache_version=step5_engine.ODCR_TOKENIZE_CACHE_VERSION,
                eval_only=False,
                index_contract_path=str(index_contract),
                step4_export_lineage={"lineage_hash": "lineage", "from_step3": "2", "from_step4": "1"},
                step5_sampler_summary=sampler_summary or {},
                step5_runtime_diagnostics=runtime_diagnostics or {},
                task_head=task_head,
            )

    def test_timing_fields_and_runtime_diagnostics_do_not_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            semantic = {"sample_plan_hash": "plan-a", "stats": {"planned_total_rows": 10}}
            noisy_a = {**semantic, "stats": {**semantic["stats"], "sampler_plan_time_s": 1.0}}
            noisy_b = {**semantic, "stats": {**semantic["stats"], "sampler_plan_time_s": 99.0, "prompt_build_time_s": 33.0}}
            fp_a = self._fingerprint(repo, sampler_summary=noisy_a, runtime_diagnostics={"tokenize_wall_time_s": 1.0})
            fp_b = self._fingerprint(repo, sampler_summary=noisy_b, runtime_diagnostics={"tokenize_wall_time_s": 999.0})
            self.assertEqual(fp_a["fingerprint_hash"], fp_b["fingerprint_hash"])
            self.assertEqual(fp_a["semantic_payload_hash"], fp_b["semantic_payload_hash"])
            self.assertNotEqual(fp_a["runtime_diagnostics_hash"], fp_b["runtime_diagnostics_hash"])

    def test_semantic_changes_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            base = self._fingerprint(repo, sampler_summary={"sample_plan_hash": "plan-a"})
            plan_b = self._fingerprint(repo, sampler_summary={"sample_plan_hash": "plan-b"})
            cand_b = self._fingerprint(repo, sampler_summary={"sample_plan_hash": "plan-a"}, selected="CANDIDATE_B")
            tok_b = self._fingerprint(repo, sampler_summary={"sample_plan_hash": "plan-a"}, tokenizer_text='{"version": 2}')
            big_base = self._fingerprint(repo, sampler_summary={"sample_plan_hash": "plan-a"}, task_head="step5B")
            big_tok_b = self._fingerprint(
                repo,
                sampler_summary={"sample_plan_hash": "plan-a"},
                tokenizer_text='{"version": 2}',
                task_head="step5B",
            )
            self.assertNotEqual(base["fingerprint_hash"], plan_b["fingerprint_hash"])
            self.assertNotEqual(base["fingerprint_hash"], cand_b["fingerprint_hash"])
            self.assertEqual(base["fingerprint_hash"], tok_b["fingerprint_hash"])
            self.assertNotEqual(big_base["fingerprint_hash"], big_tok_b["fingerprint_hash"])

    def test_rank1_uses_rank0_broadcast_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            fp = self._fingerprint(repo, sampler_summary={"sample_plan_hash": "rank0-plan"})
            rank0_dir = repo / "cache" / "rank0"
            rank0_dir.mkdir(parents=True)
            DatasetDict({"train": Dataset.from_dict({"x": [1]}), "valid": Dataset.from_dict({"x": [2]})}).save_to_disk(str(rank0_dir))
            step5_engine._write_step5_tokenize_cache_manifest(
                str(rank0_dir),
                fingerprint=fp,
                splits={"train": 1, "valid": 1},
                rank0_builder_id="rank0:test",
            )
            control = {
                "cache_dir": str(rank0_dir),
                "cache_fingerprint": "rank0-fp",
                "cache_fingerprint_payload": fp,
                "fingerprint_hash": fp["fingerprint_hash"],
                "semantic_payload_hash": fp["semantic_payload_hash"],
            }
            local_fp = dict(fp)
            local_fp["fingerprint_hash"] = "divergent-local"
            with mock.patch.object(step5_engine, "dist", _FakeDist(control)):
                encoded = step5_engine._step5_map_or_load_tokenize_cache(
                    datasets=DatasetDict(),
                    processor=lambda row: row,
                    nproc=1,
                    cache_dir=str(repo / "cache" / "rank1_missing"),
                    cache_fingerprint="rank1-local",
                    cache_fingerprint_payload=local_fp,
                    rank=1,
                    show_datasets_progress=False,
                    log_tokenize=False,
                    phase="train+valid",
                )
            self.assertEqual(len(encoded["train"]), 1)
            self.assertEqual(len(encoded["valid"]), 1)


if __name__ == "__main__":
    unittest.main()
