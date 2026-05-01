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
    write_resolved_config_artifacts,
    write_run_summary_json,
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
                "source_table_path",
                "console_log_path",
                "full_log_path",
                "errors_log_path",
                "manifest_path",
                "key_artifacts",
                "validation_status",
                "post_edit_scope",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["resolved_config_path"], "runs/step3/task4/1/meta/resolved_config.json")
            self.assertEqual(payload["console_log_path"], "runs/step3/task4/1/meta/console.log")
            self.assertEqual(payload["full_log_path"], "runs/step3/task4/1/meta/full.log")
            self.assertEqual(payload["errors_log_path"], "runs/step3/task4/1/meta/errors.log")
            self.assertNotIn("debug_log", payload["key_artifacts"])
            self.assertNotIn("samples_log", payload["key_artifacts"])
            self.assertEqual(
                payload["optional_artifacts"]["debug_log"]["path"],
                "runs/step3/task4/1/meta/debug.log",
            )
            self.assertTrue(payload["optional_artifacts"]["debug_log"]["missing_ok"])
            self.assertEqual(
                payload["optional_artifacts"]["samples_log"]["reason"],
                "samples_not_requested",
            )
            latest = json.loads((repo / "runs" / "step3" / "task4" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["latest_run_id"], "1")
            self.assertEqual(latest["latest_summary_path"], "runs/step3/task4/1/meta/run_summary.json")
            self.assertEqual(latest["latest_status"], "running")

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


if __name__ == "__main__":
    unittest.main()
