"""Latest-run resolver tests.

this test proves: latest.json uses the new meta/run_summary + stage_status handoff.
this test does not prove: controlled GPU runtime behavior, formal training quality, or performance closure.
whether formal hot path is covered: no, these tests use synthetic filesystem fixtures.
whether runtime evidence is required: yes for Level 3/4 claims outside these tests.
regression bug it prevents: blocked or malformed latest pointers being consumed as downstream checkpoints.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import OneControlConfigError, _latest_run
from odcr_core.csb_contract import csb_contract_hash, default_csb_contract_payload, method_payload
from odcr_core.training_checkpoint import (
    STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
    checkpoint_file_sha256,
    write_checkpoint_lineage,
)


class LatestRunResolutionTest(unittest.TestCase):
    def _write_summary(self, repo: Path, stage: str = "step3", task_id: int = 4, run_id: str = "1") -> Path:
        meta = repo / "runs" / stage / f"task{task_id}" / run_id / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        summary = meta / "run_summary.json"
        csb_contract = default_csb_contract_payload()
        csb_contract["contract_hash"] = csb_contract_hash(csb_contract)
        summary.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "stage": stage,
                    "task_id": task_id,
                    "method_name": "CSB-ODCR",
                    "method": method_payload(),
                    "csb_contract": csb_contract,
                    "csb_contract_hash": csb_contract["contract_hash"],
                    "source_domain": "AM_Movies" if task_id == 4 else "AM_CDs",
                    "target_domain": "AM_CDs" if task_id == 4 else "AM_Movies",
                    "status": "step4_ready",
                    "train_status": "completed",
                    "paper_eval_status": "not_applicable",
                    "downstream_ready": True,
                    "selected_checkpoint": f"runs/{stage}/task{task_id}/{run_id}/model/best.pth",
                    "selected_checkpoint_hash": "",
                    "selected_downstream_checkpoint": f"runs/{stage}/task{task_id}/{run_id}/model/best.pth",
                }
            ),
            encoding="utf-8",
        )
        return summary

    def _write_step3_checkpoint_sidecar(self, repo: Path, task_id: int = 4, run_id: str = "1") -> Path:
        ckpt = repo / "runs" / "step3" / f"task{task_id}" / run_id / "model" / "best.pth"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"dummy-checkpoint")
        write_checkpoint_lineage(
            ckpt,
            {
                "stage": "step3",
                "run_id": run_id,
                "task_id": task_id,
                "source_domain": "AM_Movies" if task_id == 4 else "AM_CDs",
                "target_domain": "AM_CDs" if task_id == 4 else "AM_Movies",
                "sidecar_schema_version": STEP3_CHECKPOINT_COMPAT_SCHEMA_VERSION,
                "checkpoint_file_hash": checkpoint_file_sha256(ckpt),
                "checkpoint_path": str(ckpt),
                "reason": "global_best_improved",
                "replaced_previous": False,
            },
        )
        readiness = ckpt.parents[1] / "meta" / "readiness_audit.json"
        readiness.parent.mkdir(parents=True, exist_ok=True)
        readiness.write_text(
            json.dumps(
                {
                    "schema_version": "odcr_step3_readiness_audit/1",
                    "readiness_gate": "step3_upstream_readiness_gate",
                    "readiness_status": "pass",
                    "quality_status": "pass",
                    "downstream_ready": True,
                    "ready_for": ["step4"],
                    "paper_metrics_excluded_from_readiness": ["BLEU", "ROUGE", "DIST", "METEOR"],
                    "selected_downstream_checkpoint": str(ckpt),
                    "selected_downstream_checkpoint_hash": checkpoint_file_sha256(ckpt),
                    "selected_downstream_checkpoint_scope": "best_observed",
                    "selected_downstream_checkpoint_epoch": 1,
                    "selected_downstream_checkpoint_metric": 1.0,
                    "csb_contract_health": {
                        "required_z_fields": ["z_content", "z_style", "z_domain", "z_uncertainty"],
                        "missing_z_fields": [],
                        "csb_contract_hash_present": True,
                        "sidecar_only": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (ckpt.parents[1] / "meta" / "source_table.json").write_text(
            json.dumps({"source_table_schema_version": "1.0", "view": "formal", "records": []}),
            encoding="utf-8",
        )
        (ckpt.parents[1] / "meta" / "resolved_config.json").write_text(
            json.dumps(
                {
                    "task": {
                        "id": task_id,
                        "source": "AM_Movies" if task_id == 4 else "AM_CDs",
                        "target": "AM_CDs" if task_id == 4 else "AM_Movies",
                    }
                }
            ),
            encoding="utf-8",
        )
        summary = ckpt.parents[1] / "meta" / "run_summary.json"
        if summary.is_file():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["selected_checkpoint_hash"] = checkpoint_file_sha256(ckpt)
            payload["selected_downstream_checkpoint_hash"] = checkpoint_file_sha256(ckpt)
            payload["failure_history"] = [{"status": "failed", "source": "fixture"}]
            summary.write_text(json.dumps(payload), encoding="utf-8")
        return ckpt

    def _write_latest(self, repo: Path, summary: Path, stage: str = "step3", task_id: int = 4, run_id: str = "1") -> None:
        parent = repo / "runs" / stage / f"task{task_id}"
        parent.mkdir(parents=True, exist_ok=True)
        latest = parent / "latest.json"
        latest.write_text(
            json.dumps(
                {
                    "latest_run_id": run_id,
                    "latest_summary_path": summary.relative_to(repo).as_posix(),
                    "latest_status": "ok",
                }
            ),
            encoding="utf-8",
        )

    def test_latest_json_normal_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="7")
            self._write_latest(repo, summary, run_id="7")
            self._write_step3_checkpoint_sidecar(repo, run_id="7")
            self.assertEqual(_latest_run(repo, 4, "step3", dry_run=False), "7")

    def test_missing_latest_json_fails_fast_even_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            with self.assertRaisesRegex(OneControlConfigError, "missing .*latest.json"):
                _latest_run(repo, 4, "step3", dry_run=True)

    def test_missing_run_summary_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = repo / "runs" / "step3" / "task4" / "2" / "meta" / "run_summary.json"
            self._write_latest(repo, summary, run_id="2")
            with self.assertRaisesRegex(OneControlConfigError, "missing run_summary.json"):
                _latest_run(repo, 4, "step3", dry_run=False)

    def test_old_runs_task_layout_is_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            old = repo / "runs" / "task4" / "step3" / "999" / "meta"
            old.mkdir(parents=True)
            (old / "run_summary.json").write_text(json.dumps({"run_id": "999"}), encoding="utf-8")
            with self.assertRaisesRegex(OneControlConfigError, "missing .*latest.json"):
                _latest_run(repo, 4, "step3", dry_run=False)

    def test_latest_pointer_wins_over_larger_directory_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="1")
            self._write_latest(repo, summary, run_id="1")
            self._write_step3_checkpoint_sidecar(repo, run_id="1")
            self._write_summary(repo, run_id="999")
            self.assertEqual(_latest_run(repo, 4, "step3", dry_run=False), "1")

    def test_failed_latest_status_is_ignored_after_stage_status_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="1")
            self._write_latest(repo, summary, run_id="1")
            self._write_step3_checkpoint_sidecar(repo, run_id="1")
            latest = repo / "runs" / "step3" / "task4" / "latest.json"
            payload = json.loads(latest.read_text(encoding="utf-8"))
            payload["latest_status"] = "failed"
            latest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_latest_run(repo, 4, "step3", dry_run=False), "1")

    def test_running_and_partial_latest_status_are_ignored_after_stage_status_validation(self) -> None:
        for status in ("running", "partial"):
            with self.subTest(status=status):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = Path(tmp)
                    summary = self._write_summary(repo, run_id="1")
                    self._write_latest(repo, summary, run_id="1")
                    self._write_step3_checkpoint_sidecar(repo, run_id="1")
                    latest = repo / "runs" / "step3" / "task4" / "latest.json"
                    payload = json.loads(latest.read_text(encoding="utf-8"))
                    payload["latest_status"] = status
                    latest.write_text(json.dumps(payload), encoding="utf-8")
                    self.assertEqual(_latest_run(repo, 4, "step3", dry_run=False), "1")

    def test_failed_run_summary_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="1")
            summary.write_text(json.dumps({"run_id": "1", "stage": "step3", "task_id": 4, "status": "failed"}), encoding="utf-8")
            self._write_latest(repo, summary, run_id="1")
            with self.assertRaisesRegex(OneControlConfigError, "run1 is not eligible for Step4 formal upstream"):
                _latest_run(repo, 4, "step3", dry_run=False)

    def test_checkpoint_missing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="1")
            self._write_latest(repo, summary, run_id="1")
            with self.assertRaisesRegex(OneControlConfigError, "selected_checkpoint_missing|downstream_ready=False"):
                _latest_run(repo, 4, "step3", dry_run=False)

    def test_sidecar_missing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            summary = self._write_summary(repo, run_id="1")
            self._write_latest(repo, summary, run_id="1")
            ckpt = repo / "runs" / "step3" / "task4" / "1" / "model" / "best.pth"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(b"dummy-checkpoint")
            readiness = repo / "runs" / "step3" / "task4" / "1" / "meta" / "readiness_audit.json"
            readiness.parent.mkdir(parents=True, exist_ok=True)
            readiness.write_text(
                json.dumps(
                    {
                        "schema_version": "odcr_step3_readiness_audit/1",
                        "readiness_gate": "step3_upstream_readiness_gate",
                        "readiness_status": "pass",
                        "quality_status": "pass",
                        "downstream_ready": True,
                        "paper_metrics_excluded_from_readiness": ["BLEU", "ROUGE", "METEOR"],
                        "selected_downstream_checkpoint": str(ckpt),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OneControlConfigError, "checkpoint_lineage_invalid|downstream_ready=False"):
                _latest_run(repo, 4, "step3", dry_run=False)


if __name__ == "__main__":
    unittest.main()
