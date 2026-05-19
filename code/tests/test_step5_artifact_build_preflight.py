import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import step5_runtime_probe
from odcr_core.index_contract import ODCR_ROUTING_TRAIN_CSV


class Step5ArtifactBuildPreflightTest(unittest.TestCase):
    def test_artifact_build_mode_uses_existing_bounded_probe_candidate_id(self):
        self.assertTrue(step5_runtime_probe._artifact_build_preflight_requested("artifact_build_B224"))
        self.assertTrue(step5_runtime_probe._artifact_build_preflight_requested("step5A-artifact-build"))
        self.assertFalse(step5_runtime_probe._artifact_build_preflight_requested("A0_C0_R0"))

    def test_artifact_build_train_link_stays_in_ai_analysis_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export = root / "runs" / "step4" / "task2" / "1" / ODCR_ROUTING_TRAIN_CSV
            export.parent.mkdir(parents=True)
            export.write_text("x\n1\n", encoding="utf-8")
            run_dir = root / "AI_analysis" / "artifact_build_preflight_run"
            link = step5_runtime_probe._ensure_artifact_build_train_link(run_dir, export)
            self.assertEqual(link, run_dir / ODCR_ROUTING_TRAIN_CSV)
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), export.resolve())
            self.assertFalse((root / "runs" / "step5").exists())
            self.assertFalse((run_dir / "latest.json").exists())
            self.assertFalse(any(run_dir.glob("model/*.pth")))

    def test_artifact_build_args_are_head_only_and_checkpoint_forbidden(self):
        cfg = SimpleNamespace(
            auxiliary="AM_Movies",
            target="AM_CDs",
            seed=3407,
            num_proc=8,
            decode_profile_json="{}",
            label_smoothing=0.02,
            repetition_penalty=1.12,
            generate_temperature=0.2,
            generate_top_p=1.0,
            max_explanation_length=36,
            decode_strategy="uncertainty_low_temp_top_k",
            decode_seed=3407,
            no_repeat_ngram_size=3,
            min_len=4,
        )
        args = step5_runtime_probe._artifact_build_args(
            cfg,
            run_dir=Path("/tmp/ai_artifact_run"),
            log_file=Path("/tmp/ai_artifact_run/meta/full.log"),
            save_file=Path("/tmp/ai_artifact_run/model/FORBIDDEN_checkpoint_not_written.pth"),
            stage="step5A",
        )
        self.assertEqual(args.task_head, "step5A")
        self.assertTrue(args.train_only)
        self.assertIn("FORBIDDEN_checkpoint_not_written", args.save_file)

    def test_artifact_build_cache_log_evidence_detects_broadcast_and_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "artifact_build_rank1.log"
            log.write_text(
                "[Tokenize] rank using rank0 broadcast cache | "
                "broadcast_fingerprint=v8_step5_semantic_lineage_abc | "
                "broadcast_dir=/cache/hf_cache_step5_v8_step5_semantic_lineage_abc\n",
                encoding="utf-8",
            )
            payload = step5_runtime_probe._extract_artifact_build_cache_evidence(log)
            self.assertTrue(payload["rank_used_rank0_broadcast_cache"])
            self.assertTrue(payload["missing_dataset_absent"])
            self.assertEqual(payload["cache_fingerprint"], "v8_step5_semantic_lineage_abc")
            self.assertEqual(payload["cache_dir"], "/cache/hf_cache_step5_v8_step5_semantic_lineage_abc")

            log.write_text("RuntimeError: reason=missing_dataset\n", encoding="utf-8")
            payload = step5_runtime_probe._extract_artifact_build_cache_evidence(log)
            self.assertFalse(payload["missing_dataset_absent"])

    def test_artifact_build_rank_summary_verifies_shared_cache_from_rank_payloads(self):
        cache_dir = "/cache/hf_cache_step5_v8_step5_semantic_lineage_abc"
        payloads = []
        for rank in (0, 1):
            payloads.append(
                {
                    "rank": rank,
                    "artifact_build_preflight": {
                        "cache_dir_candidates": [cache_dir],
                        "cache_lineage_semantic_hashes": {cache_dir: "abc"},
                        "index_contract_audit_exists": True,
                        "token_cache_success_marker_exists": True,
                        "token_cache_lineage_exists": True,
                        "missing_dataset_absent": True,
                        "rank_used_rank0_broadcast_cache": rank == 0,
                    },
                }
            )
        summary = step5_runtime_probe._summarize_artifact_build_rank_payloads(payloads, world_size=2)
        self.assertTrue(summary["rank_payloads_complete"])
        self.assertTrue(summary["rank0_rank1_cache_dir_match"])
        self.assertTrue(summary["rank0_rank1_cache_fingerprint_match"])
        self.assertTrue(summary["rank1_used_rank0_cache_dir"])
        self.assertTrue(summary["rank1_broadcast_cache_inferred_from_rank_payloads"])
        self.assertTrue(summary["token_cache_lineage_success"])


if __name__ == "__main__":
    unittest.main()
