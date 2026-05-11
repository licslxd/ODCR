from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.manifests import (  # noqa: E402
    build_run_summary,
    training_runtime_config_path,
    write_resolved_config_artifacts,
    write_run_summary_json,
    write_training_runtime_config_artifact,
)


class TestRunSummaryLogging(unittest.TestCase):
    def test_run_summary_helper_writes_required_fields_and_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_dir = repo / "runs" / "step3" / "task4" / "1"
            meta_dir = run_dir / "meta"
            summary = build_run_summary(
                repo_root=repo,
                run_dir=run_dir,
                meta_dir=meta_dir,
                run_id="1",
                stage="step3",
                task_id=4,
                source_domain="AM_Movies",
                target_domain="AM_Electronics",
                status="running",
                started_at="2026-04-27T00:00:00Z",
                command="./odcr step3 --task 4",
            )
            out = write_run_summary_json(summary, repo_root=repo)
            self.assertEqual(out, meta_dir / "run_summary.json")
            payload = json.loads(out.read_text(encoding="utf-8"))
            for key in (
                "run_id",
                "stage",
                "task_id",
                "status",
                "started_at",
                "resolved_config_path",
                "training_runtime_config_path",
                "source_table_path",
                "console_log_path",
                "full_log_path",
                "authoritative_full_log_path",
                "errors_log_path",
                "debug_log_path",
                "manifest_path",
                "key_artifacts",
                "validation_status",
                "post_edit_scope",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["resolved_config_path"], "runs/step3/task4/1/meta/resolved_config.json")
            self.assertIsNone(payload["training_runtime_config_path"])
            self.assertEqual(payload["console_log_path"], "runs/step3/task4/1/meta/console.log")
            self.assertEqual(payload["full_log_path"], "runs/step3/task4/1/meta/full.log")
            self.assertEqual(payload["authoritative_full_log_path"], "runs/step3/task4/1/meta/full.log")
            self.assertEqual(payload["errors_log_path"], "runs/step3/task4/1/meta/errors.log")
            self.assertEqual(payload["debug_log_path"], "runs/step3/task4/1/meta/debug.log")
            self.assertNotIn("debug_log", payload["key_artifacts"])
            self.assertNotIn("samples_log", payload["key_artifacts"])
            self.assertEqual(
                payload["optional_artifacts"]["debug_log"]["path"],
                "runs/step3/task4/1/meta/debug.log",
            )
            self.assertEqual(
                payload["optional_artifacts"]["training_runtime_config"]["path"],
                "runs/step3/task4/1/meta/training_runtime_config.json",
            )
            self.assertTrue(payload["optional_artifacts"]["debug_log"]["missing_ok"])
            self.assertEqual(
                payload["optional_artifacts"]["samples_log"]["reason"],
                "samples_not_requested",
            )
            latest = json.loads((repo / "runs" / "step3" / "task4" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["latest_run_id"], "1")
            self.assertEqual(latest["latest_summary_path"], "runs/step3/task4/1/meta/run_summary.json")
            self.assertEqual(latest["latest_stage_status_path"], "runs/step3/task4/1/meta/stage_status.json")
            self.assertEqual(latest["status_claim_source"], "stage_status_strict_verifier")
            self.assertNotIn("latest_status", latest)

    def test_resolved_config_artifacts_use_canonical_names_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "runs" / "step5" / "task4" / "2_1_1" / "meta"
            config_path, source_path = write_resolved_config_artifacts(
                meta,
                {"field_sources": {"step5": "step5.train"}, "value": 7},
            )
            self.assertEqual(config_path.name, "resolved_config.json")
            self.assertEqual(source_path.name, "source_table.json")
            self.assertTrue(config_path.is_file())
            self.assertTrue(source_path.is_file())
            self.assertFalse((meta / "config_resolved.json").exists())
            self.assertFalse((meta / "resolved_config_snapshot.json").exists())
            self.assertFalse((meta / "config_snapshot.json").exists())
            source_table = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(source_table["field_sources"]["step5"], "step5.train")

    def test_resolved_config_not_overwritten_by_training_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "runs" / "step3" / "task2" / "dry_run" / "meta"
            parent_snapshot = {
                "train": {"stage": "step3", "batch_size": 1536},
                "field_sources": {"step3.train.batch_size": "configs/odcr.yaml"},
            }
            runtime_snapshot = {
                "FinalTrainingConfig": True,
                "train_batch_size": 1536,
                "training_diagnostics": {"diagnostics_scope": "child"},
            }
            config_path, _ = write_resolved_config_artifacts(meta, parent_snapshot, formal_only_source_table=True)
            before = json.loads(config_path.read_text(encoding="utf-8"))
            runtime_path = write_training_runtime_config_artifact(meta, runtime_snapshot)
            after = json.loads(config_path.read_text(encoding="utf-8"))
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertEqual(before, after)
            self.assertNotEqual(after, runtime)
            self.assertEqual(runtime_path, training_runtime_config_path(meta))
            self.assertEqual(runtime["training_runtime_config_schema_version"], "odcr_training_runtime_config/1")

    def test_preprocess_summary_records_metrics_and_verify_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_dir = repo / "runs" / "preprocess" / "c" / "1"
            meta_dir = run_dir / "meta"
            metrics_path = meta_dir / "metrics.json"
            verify_path = meta_dir / "verify_report.json"
            summary = build_run_summary(
                repo_root=repo,
                run_dir=run_dir,
                meta_dir=meta_dir,
                run_id="1",
                stage="preprocess",
                unit="c",
                status="ok",
                started_at="2026-05-01T00:00:00Z",
                finished_at="2026-05-01T00:00:10Z",
                command="./odcr preprocess c",
                metrics_path=metrics_path,
                key_artifacts={"metrics": metrics_path, "verify_report": verify_path},
            )
            out = write_run_summary_json(summary, repo_root=repo, update_latest=True)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["metrics_path"], "runs/preprocess/c/1/meta/metrics.json")
            self.assertEqual(payload["key_artifacts"]["metrics"], "runs/preprocess/c/1/meta/metrics.json")
            self.assertEqual(
                payload["key_artifacts"]["verify_report"],
                "runs/preprocess/c/1/meta/verify_report.json",
            )
            self.assertIsNotNone(payload["metrics_path"])


if __name__ == "__main__":
    unittest.main()
