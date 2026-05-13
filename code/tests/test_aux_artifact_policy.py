from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.artifacts import cache_namespace, lineage_fingerprint, meta_log_paths, resolve_latest_summary_path, run_meta_dir


class AuxArtifactPolicyTest(unittest.TestCase):
    def test_run_log_and_cache_paths_are_canonical(self) -> None:
        root = Path("/tmp/odcr-test")
        meta = run_meta_dir(root, "step3", 2, "1_1")
        self.assertEqual(meta.as_posix(), "/tmp/odcr-test/runs/step3/task2/1_1/meta")
        self.assertEqual(meta_log_paths(meta)["console"].name, "console.log")
        self.assertIn("/cache/step4/task2/encoded", cache_namespace(root, "step4", 2, "encoded").as_posix())

    def test_latest_resolver_requires_run_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "runs" / "step3" / "task2"
            summary = task_dir / "1_1" / "meta" / "run_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text("{}", encoding="utf-8")
            (task_dir / "latest.json").write_text(json.dumps({"latest_summary_path": str(summary)}), encoding="utf-8")
            self.assertEqual(resolve_latest_summary_path(task_dir), summary.resolve())

    def test_lineage_fingerprint_is_stable(self) -> None:
        self.assertEqual(lineage_fingerprint({"b": 2, "a": 1}), lineage_fingerprint({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
