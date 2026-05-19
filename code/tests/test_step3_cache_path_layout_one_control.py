from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402
from odcr_core import path_layout  # noqa: E402
from odcr_core.config_resolver import resolve_config  # noqa: E402
from odcr_core.manifests import build_formal_source_table_snapshot  # noqa: E402


class Step3CachePathLayoutOneControlTest(unittest.TestCase):
    def test_formal_cache_path_comes_only_from_one_control_path_layout(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        cache_policy = snapshot["step3_cache_policy"]
        cache_dir = path_layout.step3_tokenizer_cache_entry_dir(
            REPO_ROOT,
            formal_cache_namespace=cache_policy["formal_cache_namespace"],
            task_id=2,
            source_domain=cfg.auxiliary,
            target_domain=cfg.target,
            compatibility_key="compat_key",
        )
        self.assertTrue(cache_dir.as_posix().endswith("/cache/step3/tokenizer/task2/AM_Movies_to_AM_CDs/compat_key"))
        self.assertNotIn("/cache/task2/hf", cache_dir.as_posix())
        self.assertTrue(cache_dir.as_posix().startswith(Path(cfg.cache_dir).resolve().as_posix()))

    def test_resolved_config_source_table_and_runtime_cache_dir_are_consistent(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=[],
            dry_run=True,
            run_id="auto",
            mode="full",
        )
        source_table = build_formal_source_table_snapshot(snapshot)
        self.assertEqual(Path(snapshot["roots"]["cache_dir"]).resolve(), Path(cfg.cache_dir).resolve())
        self.assertEqual(source_table["field_sources"]["cache_dir"], "project.cache_dir")
        self.assertEqual(snapshot["step3_cache_policy"]["formal_cache_namespace"], "cache/step3/tokenizer")
        self.assertEqual(
            snapshot["field_sources"]["step3_cache_policy"],
            "step3.cache",
        )

    def test_validation_namespace_is_separate_from_formal_namespace(self) -> None:
        """The validation namespace stays outside formal runs/step3/task2 state."""
        formal = path_layout.step3_tokenizer_cache_entry_dir(
            REPO_ROOT,
            formal_cache_namespace="cache/step3/tokenizer",
            task_id=2,
            source_domain="AM_Movies",
            target_domain="AM_CDs",
            compatibility_key="compat_key",
        )
        validation = path_layout.step3_validation_tokenizer_cache_entry_dir(
            REPO_ROOT,
            validation_slug="step3_tmux_gpu_bridge_startup_validation_closeout",
            run_id="bridge_test",
            task_id=2,
            source_domain="AM_Movies",
            target_domain="AM_CDs",
            compatibility_key="compat_key",
        )
        self.assertIn("AI_analysis/01_raw_logs", validation.as_posix())
        self.assertIn("/cache/step3/tokenizer/task2/AM_Movies_to_AM_CDs/compat_key", validation.as_posix())
        self.assertNotEqual(formal.resolve(), validation.resolve())
        self.assertNotIn("runs/step3/task2", validation.as_posix())

    def test_old_path_fallback_is_not_active_in_step3_cache_build(self) -> None:
        for fn in (
            step3._build_step3_cache_dir,
            step3._build_step3_eval_cache_dir,
        ):
            source = inspect.getsource(fn)
            with self.subTest(function=fn.__name__):
                self.assertIn("path_layout.step3_tokenizer_cache_entry_dir", source)
                self.assertNotIn("get_hf_cache_root(task_idx)", source)
                self.assertNotIn("cache/task{", source)
        ensure_source = inspect.getsource(step3.ensure_step3_tokenizer_cache_ready_pre_ddp)
        self.assertIn("_build_step3_cache_dir", ensure_source)

    def test_manifest_path_contract_includes_task_domains_schema_and_compat_key(self) -> None:
        source = inspect.getsource(step3._step3_tokenize_cache_manifest_gate_fields)
        for token in (
            "manifest_schema_version",
            "cache_schema_version",
            "task_id",
            "source_domain",
            "target_domain",
            "compatibility_key",
            "tokenizer_cache_compat_hash",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_guardrail_contains_old_path_regression_checks(self) -> None:
        guardrail = (
            CODE_DIR / "odcr_core" / "aux" / "governance" / "guardrail_runner.py"
        ).read_text(encoding="utf-8")
        registry = (
            CODE_DIR / "odcr_core" / "aux" / "governance" / "rule_registry.py"
        ).read_text(encoding="utf-8")
        self.assertIn("R119", registry)
        self.assertIn("R126", registry)
        self.assertIn("run_checks", guardrail)


if __name__ == "__main__":
    unittest.main()
