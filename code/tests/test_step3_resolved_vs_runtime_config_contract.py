"""Parent resolved config and child runtime config stay split."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
_REPO_ROOT = Path(_CODE_DIR).resolve().parent

from odcr_core.manifests import (  # noqa: E402
    build_run_summary,
    source_table_path,
    source_table_verbose_path,
    training_runtime_config_path,
    write_resolved_config_artifacts,
    write_training_runtime_config_artifact,
)


class TestStep3ResolvedVsRuntimeConfigContract(unittest.TestCase):
    def test_parent_and_runtime_snapshots_have_distinct_paths_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = root / "runs" / "step3" / "task2" / "dry_run" / "meta"
            snapshot = {"train": {"stage": "step3"}, "field_sources": {"train": "step3.train"}}
            resolved_path, source_path = write_resolved_config_artifacts(
                meta,
                snapshot,
                formal_only_source_table=True,
                write_verbose_source_table=True,
            )
            runtime_path = write_training_runtime_config_artifact(meta, {"runtime": {"rank": 0}})
            self.assertNotEqual(resolved_path.name, runtime_path.name)
            self.assertEqual(runtime_path, training_runtime_config_path(meta))
            self.assertEqual(source_path, source_table_path(meta))
            self.assertTrue(source_table_verbose_path(meta).is_file())

            summary = build_run_summary(
                repo_root=root,
                run_dir=meta.parent,
                meta_dir=meta,
                run_id="dry_run",
                stage="step3",
                status="ok",
                started_at="2026-05-06T00:00:00Z",
                finished_at="2026-05-06T00:00:01Z",
                task_id=2,
            )
            self.assertTrue(summary["resolved_config_hash"])
            self.assertTrue(summary["source_table_hash"])
            self.assertTrue(summary["training_runtime_config_hash"])
            self.assertNotEqual(summary["resolved_config_hash"], summary["training_runtime_config_hash"])


if __name__ == "__main__":
    unittest.main()
