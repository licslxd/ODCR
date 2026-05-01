from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import odcr  # noqa: E402
from odcr_core.logging_meta import (  # noqa: E402
    CONSOLE_LEVEL_DEBUG,
    CONSOLE_LEVEL_SUMMARY,
    CONSOLE_LEVEL_VERBOSE,
    console_level_from_flags,
    print_pre_run_banner,
)
from odcr_core.manifests import write_resolved_config_artifacts  # noqa: E402
from paths_config import DEFAULT_MIRROR_LOG, append_log_dual  # noqa: E402
from tools.odcr_post_edit_check import build_plan, plan_safety_violations  # noqa: E402


def _summary_cfg(tmp: Path) -> SimpleNamespace:
    meta = tmp / "runs" / "step3" / "task4" / "1" / "meta"
    return SimpleNamespace(
        repo_root=tmp,
        manifest_dir=str(meta),
        checkpoint_dir=str(meta.parent),
        command="step3",
        task_id=4,
        auxiliary="AM_Movies",
        target="AM_Electronics",
        run_name="1",
        step4_run=None,
        step5_run=None,
        eval_run_dir=None,
        train_batch_size=16,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        ddp_world_size=2,
        epochs=3,
        learning_rate=0.001,
        hardware_preset_id="local_2gpu",
        train_precision="bf16",
        launcher_env_effective_json=json.dumps({"CUDA_VISIBLE_DEVICES": "0,1"}),
    )


class TestLoggingConsoleFile(unittest.TestCase):
    def test_default_console_summary_omits_full_source_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            cfg = _summary_cfg(Path(tmp_raw))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_pre_run_banner(
                    "step3",
                    cfg,
                    console_level=CONSOLE_LEVEL_SUMMARY,
                    started_at="2026-04-28T00:00:00Z",
                )
            text = buf.getvalue()
            self.assertIn("stage=step3 task=4", text)
            self.assertIn("run_summary=", text)
            self.assertIn("console.log", text)
            self.assertNotIn("Source table:", text)
            self.assertNotIn("field_sources", text)
            self.assertNotIn("resolved_snapshot", text)

    def test_resolved_config_and_source_table_retain_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            meta = Path(tmp_raw) / "runs" / "step3" / "task4" / "1" / "meta"
            config_path, source_path = write_resolved_config_artifacts(
                meta,
                {
                    "train": {"batch_size": 16, "micro_batch_size": 2},
                    "field_sources": {
                        "step3.train.batch_size": "configs/odcr.yaml",
                        "tasks.4.source": "configs/odcr.yaml",
                    },
                },
            )
            resolved = json.loads(config_path.read_text(encoding="utf-8"))
            source = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(resolved["train"]["batch_size"], 16)
            self.assertEqual(source["field_sources"]["step3.train.batch_size"], "configs/odcr.yaml")
            self.assertEqual(source_path.name, "source_table.json")

    def test_verbose_debug_flags_do_not_change_resolved_training_payload(self) -> None:
        base_args = odcr.build_parser().parse_args(["show", "--stage", "step3", "--task", "4"])
        verbose_args = odcr.build_parser().parse_args(["show", "--stage", "step3", "--task", "4", "--verbose"])
        debug_args = odcr.build_parser().parse_args(["show", "--stage", "step3", "--task", "4", "--debug"])
        cfg_base, _, snap_base = odcr._resolve_for_args(base_args, "step3")
        cfg_verbose, _, snap_verbose = odcr._resolve_for_args(verbose_args, "step3")
        cfg_debug, _, snap_debug = odcr._resolve_for_args(debug_args, "step3")
        self.assertEqual(snap_base["train"], snap_verbose["train"])
        self.assertEqual(snap_base["train"], snap_debug["train"])
        self.assertEqual(cfg_base.effective_training_payload_json, cfg_verbose.effective_training_payload_json)
        self.assertEqual(cfg_base.effective_training_payload_json, cfg_debug.effective_training_payload_json)
        self.assertEqual(console_level_from_flags(verbose=False, debug=False), CONSOLE_LEVEL_SUMMARY)
        self.assertEqual(console_level_from_flags(verbose=True, debug=False), CONSOLE_LEVEL_VERBOSE)
        self.assertEqual(console_level_from_flags(verbose=False, debug=True), CONSOLE_LEVEL_DEBUG)

    def test_code_log_out_is_not_active_default(self) -> None:
        self.assertEqual(DEFAULT_MIRROR_LOG, "")
        old = os.environ.pop("ODCR_LOG_DIR", None)
        try:
            import train_logging

            self.assertIn("runs/internal", Path(train_logging._log_dir()).as_posix())
        finally:
            if old is not None:
                os.environ["ODCR_LOG_DIR"] = old

    def test_fallback_mirror_log_is_retired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            tmp = Path(tmp_raw)
            primary = tmp / "runs" / "step4" / "task4" / "1_1" / "meta" / "full.log"
            append_log_dual(str(primary), "hello\n", mirror=True)
            self.assertEqual(primary.read_text(encoding="utf-8"), "hello\n")
            self.assertFalse((tmp / "runs" / "_legacy_logs").exists())

    def test_post_edit_dry_run_has_logging_scope(self) -> None:
        plan = build_plan("logging", repo_root=REPO_ROOT, python_executable=sys.executable)
        displays = [cmd.display() for cmd in plan]
        self.assertTrue(any("test_run_summary_logging.py" in item for item in displays))
        self.assertTrue(any("test_logging_console_file.py" in item for item in displays))
        self.assertTrue(any("test_logging_tail.py" in item for item in displays))
        self.assertTrue(any("test_path_layout_boundaries.py" in item for item in displays))
        self.assertEqual(plan_safety_violations(plan), [])


if __name__ == "__main__":
    unittest.main()
