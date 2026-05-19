from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(CODE_DIR))

from odcr_core import path_layout  # noqa: E402
from odcr_core.logging_meta import run_log_paths  # noqa: E402
from helpers.test_artifacts import (  # noqa: E402
    assert_not_formal_runs_path,
    assert_path_under_test_artifacts,
    explain_artifact_policy,
    make_test_run_root,
)


class TestPathLayoutBoundaries(unittest.TestCase):
    def test_artifact_role_registry_has_required_roles(self) -> None:
        registry = path_layout.artifact_role_registry()
        for role in (
            "run_meta",
            "console_log",
            "full_log",
            "errors_log",
            "metrics",
            "lineage",
            "manifest",
            "cache",
            "ai_analysis",
            "data_artifact",
            "merged_artifact",
        ):
            self.assertIn(role, registry)
            spec = path_layout.validate_artifact_role_spec(registry[role])
            self.assertEqual(spec.role, role)
            self.assertTrue(spec.default_directory)
            self.assertTrue(spec.filename_convention)
            self.assertTrue(spec.producer)
            self.assertTrue(spec.consumer)
            self.assertTrue(spec.retention_note)

    def test_cache_paths_are_not_under_runs_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            b = path_layout.preprocess_cache_entry_dir(repo, "b", "abc123")
            c = path_layout.preprocess_cache_entry_dir(repo, "preprocess_c", "def456")
            self.assertEqual(b, repo / "cache" / "preprocess_b" / "abc123")
            self.assertEqual(c, repo / "cache" / "preprocess_c" / "def456")
            for path in (b, c):
                rel = path.relative_to(repo).as_posix()
                self.assertTrue(rel.startswith("cache/"))
                self.assertNotIn("/meta/", rel)
                self.assertFalse(rel.startswith("runs/"))

    def test_run_log_paths_are_run_meta_not_cache_data_or_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            meta = repo / "runs" / "step3" / "task4" / "1" / "meta"
            cfg = SimpleNamespace(manifest_dir=str(meta))
            paths = run_log_paths(cfg)
            for key in ("console", "full", "errors"):
                rel = paths[key].relative_to(repo).as_posix()
                self.assertTrue(rel.startswith("runs/step3/task4/1/meta/"))
                self.assertFalse(rel.startswith(("cache/", "data/", "merged/")))

    def test_ai_analysis_is_not_active_full_log_sink(self) -> None:
        registry = path_layout.artifact_role_registry()
        self.assertFalse(registry["full_log"].ai_analysis_may_copy)
        self.assertFalse(registry["console_log"].ai_analysis_may_copy)
        self.assertTrue(registry["ai_analysis"].ai_analysis_may_copy)
        self.assertIn("do not mirror full training logs", registry["ai_analysis"].retention_note)

    def test_post_edit_hook_logs_only_under_ai_analysis_codex_hooks(self) -> None:
        hook = (
            REPO_ROOT / "code" / "odcr_core" / "aux" / "governance" / "hook_scope.py"
        ).read_text(encoding="utf-8")
        self.assertIn('Path("AI_analysis") / "01_raw_logs" / "codex_hooks"', hook)
        self.assertNotIn('Path("runs")', hook)
        self.assertNotIn("_launcher_logs", hook)

    def test_test_run_like_artifacts_stay_out_of_formal_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = make_test_run_root("step4", 2, "unit", repo_root=repo)
            self.assertEqual(run_root, repo / "test_artifacts" / "runs_like" / "step4" / "task2" / "unit")
            assert_path_under_test_artifacts(run_root / "meta" / "run_summary.json", repo_root=repo)
            with self.assertRaises(AssertionError):
                assert_not_formal_runs_path(repo / "runs" / "step4" / "task2" / "unit", repo_root=repo)
            self.assertIn("Run-like test files", explain_artifact_policy())

    def test_metrics_filename_helper(self) -> None:
        self.assertEqual(path_layout.metrics_filename("metrics"), "metrics.jsonl")
        self.assertEqual(path_layout.metrics_filename("epoch_summary"), "epoch_summary.csv")
        self.assertEqual(path_layout.metrics_filename("loss_breakdown"), "loss_breakdown.jsonl")
        self.assertEqual(path_layout.metrics_filename("timing_profile"), "timing_profile.jsonl")
        self.assertEqual(path_layout.metrics_filename("gpu_profile"), "gpu_profile.jsonl")
        self.assertEqual(path_layout.metrics_filename("rcr_distribution"), "rcr_distribution.json")
        self.assertEqual(path_layout.metrics_filename("eval_metrics"), "eval_metrics.json")
        self.assertEqual(path_layout.metrics_filename("rerank_summary"), "rerank_summary.json")
        self.assertEqual(path_layout.metrics_filename("data_audit_summary"), "data_audit_summary.csv")


if __name__ == "__main__":
    unittest.main()
