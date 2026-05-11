from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import OneControlConfigError, _latest_run  # noqa: E402
from odcr_core.step3_eval_handoff import (  # noqa: E402
    Step3EvalHandoffError,
    accept_step3_eval_handoff,
    validate_step3_eval_handoff_evidence,
)
from odcr_core.training_checkpoint import (  # noqa: E402
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    checkpoint_file_sha256,
    write_checkpoint_lineage,
)


def _metrics(split: str) -> dict[str, object]:
    return {
        "bertscore_enabled": False,
        "paper_comparable": True,
        "protocol": "paper_target_only_eval",
        "split": split,
        "target_only": True,
        "recommendation": {"mae": 0.63, "rmse": 0.90},
        "explanation": {
            "rouge": {"1": 14.7, "l": 10.5},
            "bleu": {"1": 15.2, "2": 5.2, "3": 2.1, "4": 1.0},
            "dist": {"1": 0.2, "2": 0.9},
            "meteor": 13.8,
        },
    }


class Step3EvalHandoffTest(unittest.TestCase):
    def _write_fixture(self, repo: Path, *, integrity_status: str = "PASS", omit_test: bool = False) -> None:
        run = repo / "runs" / "step3" / "task2" / "2"
        meta = run / "meta"
        model = run / "model"
        model.mkdir(parents=True)
        meta.mkdir(parents=True)
        for name in ("best_observed.pth", "best.pth", "latest.pth"):
            ckpt = model / name
            payload = b"synthetic-step3-latest-checkpoint" if name == "latest.pth" else b"synthetic-step3-checkpoint"
            ckpt.write_bytes(payload)
            write_checkpoint_lineage(
                ckpt,
                {
                    "stage": "step3",
                    "task_id": 2,
                    "run_id": "2",
                    "sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
                    "checkpoint_path": str(ckpt),
                    "checkpoint_file_hash": checkpoint_file_sha256(ckpt),
                    "reason": "global_best_improved",
                    "replaced_previous": False,
                    "selection_scope": "best_observed" if name != "latest.pth" else "latest",
                },
            )
        (meta / "run_summary.json").write_text(
            json.dumps(
                {
                    "run_id": "2",
                    "stage": "step3",
                    "task_id": 2,
                    "status": "failed",
                    "run_dir": "runs/step3/task2/2",
                    "meta_dir": "runs/step3/task2/2/meta",
                    "latest_error": "WorkNCCL(SeqNum=3, OpType=ALLREDUCE, Timeout(ms)=600000)",
                    "fatal_signature": "WorkNCCL(SeqNum=3, OpType=ALLREDUCE, Timeout(ms)=600000)",
                    "failure_phase": "post_train_eval",
                    "checkpoint_created": True,
                    "training_loop_started": True,
                }
            ),
            encoding="utf-8",
        )
        (meta / "source_table.json").write_text(json.dumps({"records": [{"key": "task", "source": "unit"}]}), encoding="utf-8")
        (meta / "resolved_config.json").write_text(json.dumps({"task": {"id": 2}, "train": {"batch_semantics_version": "odcr_no_accum/1"}}), encoding="utf-8")
        latest = repo / "runs" / "step3" / "task2" / "latest.json"
        latest.write_text(
            json.dumps(
                {
                    "latest_run_id": "2",
                    "latest_run_dir": "runs/step3/task2/2",
                    "latest_summary_path": "runs/step3/task2/2/meta/run_summary.json",
                    "latest_status": "failed",
                }
            ),
            encoding="utf-8",
        )
        registry_dir = repo / "runs" / "task2" / "meta"
        registry_dir.mkdir(parents=True)
        registry_lines: list[str] = []
        for split, label, count in (
            ("valid", "paper_valid_b6144_full_detached", 10),
            ("test", "paper_test_b6144_full_detached", 9),
        ):
            if omit_test and split == "test":
                continue
            root = meta / "eval_only" / label / f"eval_paper_target_only_eval_{split}"
            root.mkdir(parents=True)
            summary = {
                "eval_status": "completed",
                "eval_protocol": "paper_target_only_eval",
                "target_only": True,
                "bertscore_enabled": False,
                "max_ref_len": 25,
                "max_decode_len": 25,
                "sample_count": count,
                "eval_batch_global": 6144,
                "metrics": _metrics(split),
            }
            protocol = {
                "protocol": "paper_target_only_eval",
                "target_only": True,
                "bertscore_enabled": False,
                "max_ref_len": 25,
                "max_decode_len": 25,
                "schema_version": "odcr_step3_eval_protocol/2",
            }
            integrity = {"status": integrity_status, "count_match": integrity_status == "PASS", "sample_count": count}
            (root / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (root / "eval_protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
            (root / "sample_integrity_report.json").write_text(json.dumps(integrity), encoding="utf-8")
            (root.parent / "full.log").write_text("completed without distributed timeout\n", encoding="utf-8")
            (root.parent / "errors.log").write_text("", encoding="utf-8")
            registry_lines.append(
                json.dumps(
                    {
                        "pipeline": "Step3_two_phase_paper_target_only_eval",
                        "task_description": f"Step 3 two-phase eval Task 2 protocol=paper_target_only_eval split={split}",
                        "log_file": str(root.parent / "full.log"),
                        "mae": 0.63,
                        "rmse": 0.90,
                    }
                )
            )
        (registry_dir / "eval_registry.jsonl").write_text("\n".join(registry_lines) + "\n", encoding="utf-8")

    def test_accept_eval_handoff_updates_summary_latest_and_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_fixture(repo)
            dry = accept_step3_eval_handoff(repo_root=repo, task_id=2, run_id="2", dry_run=True)
            self.assertTrue(dry["accepted"])
            self.assertFalse((repo / "runs" / "step3" / "task2" / "2" / "meta" / "eval_handoff.json").exists())
            result = accept_step3_eval_handoff(repo_root=repo, task_id=2, run_id="2")
            self.assertTrue(result["accepted"])
            handoff = json.loads((repo / "runs" / "step3" / "task2" / "2" / "meta" / "eval_handoff.json").read_text())
            self.assertEqual(handoff["paper_eval_status"], "completed")
            summary = json.loads((repo / "runs" / "step3" / "task2" / "2" / "meta" / "run_summary.json").read_text())
            self.assertEqual(summary["status"], "completed_with_eval_handoff")
            self.assertTrue(summary["downstream_ready"])
            self.assertEqual(summary["failure_history"][0]["failure_type"], "nccl_timeout")
            latest = json.loads((repo / "runs" / "step3" / "task2" / "latest.json").read_text())
            self.assertNotIn("latest_status", latest)
            self.assertEqual(latest["latest_run_id"], "2")
            self.assertEqual(_latest_run(repo, 2, "step3", dry_run=False), "2")

    def test_old_failed_without_handoff_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_fixture(repo)
            with self.assertRaisesRegex(OneControlConfigError, "not eligible for Step4 formal upstream"):
                _latest_run(repo, 2, "step3", dry_run=False)

    def test_missing_test_eval_rejected_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_fixture(repo, omit_test=True)
            with self.assertRaises(Step3EvalHandoffError):
                validate_step3_eval_handoff_evidence(repo_root=repo, task_id=2, run_id="2", require_test=True)

    def test_checkpoint_hash_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_fixture(repo)
            (repo / "runs" / "step3" / "task2" / "2" / "model" / "best_observed.pth").write_bytes(b"changed")
            with self.assertRaisesRegex(Step3EvalHandoffError, "lineage hash mismatch"):
                validate_step3_eval_handoff_evidence(repo_root=repo, task_id=2, run_id="2")

    def test_sample_integrity_fail_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_fixture(repo, integrity_status="FAIL")
            with self.assertRaisesRegex(Step3EvalHandoffError, "sample integrity"):
                validate_step3_eval_handoff_evidence(repo_root=repo, task_id=2, run_id="2")


if __name__ == "__main__":
    unittest.main()
