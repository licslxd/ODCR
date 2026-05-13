from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.stage_promotion import promote_upstream  # noqa: E402
from odcr_core.stage_truth_antiforgery import mutate_status, write_json, write_step3_fixture  # noqa: E402
from odcr_core.upstream_resolver import UpstreamResolutionError, resolve_upstream  # noqa: E402


class StageTruthAntiForgeryTest(unittest.TestCase):
    def test_minimal_forged_stage_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            run = write_step3_fixture(repo, task=2, run_id="3", eligible=True)
            write_json(
                run / "meta" / "stage_status.json",
                {
                    "schema_version": "odcr_stage_status/1",
                    "stage": "step3",
                    "task": 2,
                    "task_id": 2,
                    "run_id": "3",
                    "run_dir": "runs/step3/task2/3",
                    "final_status": "step4_ready",
                    "downstream_ready": True,
                    "ready_for": ["step4"],
                    "artifacts": {},
                },
            )
            with self.assertRaisesRegex(UpstreamResolutionError, "stage_status_strict_validation_failed|csb_contract_gate_failed"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="3", consumer_stage="step4")

    def test_missing_artifacts_hash_mismatch_and_stale_exists_are_rejected(self) -> None:
        cases = {
            "missing_readiness_audit": lambda repo, run: (run / "meta" / "readiness_audit.json").unlink(),
            "missing_checkpoint": lambda repo, run: (run / "model" / "best_observed.pth").unlink(),
            "hash_mismatch": lambda repo, run: mutate_status(
                repo,
                task=2,
                run_id=run.name,
                mutate=lambda payload: payload.__setitem__("selected_checkpoint_hash", "0" * 64),
            ),
            "missing_checkpoint_lineage": lambda repo, run: (run / "state" / "checkpoint_lineage.json").unlink(),
            "missing_source_table": lambda repo, run: (run / "meta" / "source_table.json").unlink(),
            "missing_resolved_config": lambda repo, run: (run / "meta" / "resolved_config.json").unlink(),
        }
        for idx, (name, mutate) in enumerate(cases.items(), start=10):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = Path(tmp)
                    write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
                    run = write_step3_fixture(repo, task=2, run_id=str(idx), eligible=True)
                    mutate(repo, run)
                    with self.assertRaisesRegex(UpstreamResolutionError, "stage_status_strict_validation_failed"):
                        resolve_upstream(repo_root=repo, stage="step3", task=2, from_run=str(idx), consumer_stage="step4")

    def test_task_run_path_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            write_step3_fixture(repo, task=2, run_id="3", eligible=True)
            mutate_status(repo, task=2, run_id="3", mutate=lambda payload: payload.__setitem__("task", 5))
            with self.assertRaisesRegex(UpstreamResolutionError, "stage_status task mismatch"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="3", consumer_stage="step4")

    def test_latest_status_conflict_is_pointer_only_warning_not_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True, latest_status="failed")
            res = resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="latest", consumer_stage="step4")
            self.assertEqual(res.run_id, "2")
            warnings = (res.validation or {}).get("latest_warnings") or []
            self.assertTrue(any("deprecated latest_status" in item for item in warnings))

    def test_strict_resolver_rejects_ineligible_and_accepts_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            blocked = "11"
            ready = "12"
            write_step3_fixture(repo, task=2, run_id=blocked, eligible=False)
            write_step3_fixture(repo, task=2, run_id=ready, active=True, eligible=True)
            for requested in (blocked, blocked):
                with self.assertRaisesRegex(UpstreamResolutionError, "not eligible"):
                    resolve_upstream(repo_root=repo, stage="step3", task=2, from_run=requested, consumer_stage="step4")
            for requested in (ready, "latest"):
                self.assertEqual(
                    resolve_upstream(repo_root=repo, stage="step3", task=2, from_run=requested, consumer_stage="step4").run_id,
                    ready,
                )

    def test_generic_task_runs_are_not_hardcoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="3", eligible=True)
            write_step3_fixture(repo, task=2, run_id="4", active=True, eligible=True)
            write_step3_fixture(repo, task=5, run_id="1", active=True, eligible=True)
            self.assertEqual(resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4").run_id, "4")
            self.assertEqual(resolve_upstream(repo_root=repo, stage="step3", task=5, consumer_stage="step4").run_id, "1")

    def test_quality_audit_fake_true_or_false_cannot_override_stage_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True, quality_downstream_ready=False)
            self.assertEqual(resolve_upstream(repo_root=repo, stage="step3", task=2, consumer_stage="step4").run_id, "2")
            run = write_step3_fixture(repo, task=2, run_id="3", eligible=True, quality_downstream_ready=True)
            (run / "meta" / "readiness_audit.json").unlink()
            with self.assertRaisesRegex(UpstreamResolutionError, "stage_status_strict_validation_failed"):
                resolve_upstream(repo_root=repo, stage="step3", task=2, from_run="3", consumer_stage="step4")

    def test_promotion_rejects_malformed_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            run = write_step3_fixture(repo, task=2, run_id="3", eligible=True)
            payload = json.loads((run / "meta" / "stage_status.json").read_text(encoding="utf-8"))
            payload["artifacts"] = {}
            write_json(run / "meta" / "stage_status.json", payload)
            with self.assertRaisesRegex(UpstreamResolutionError, "stage_status_strict_validation_failed"):
                promote_upstream(repo_root=repo, stage="step3", task=2, run_id="3", dry_run=True)


if __name__ == "__main__":
    unittest.main()
