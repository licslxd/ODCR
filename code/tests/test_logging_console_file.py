from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import odcr  # noqa: E402
from odcr_core.logging_meta import (  # noqa: E402
    CONSOLE_LEVEL_DEBUG,
    CONSOLE_LEVEL_SUMMARY,
    CONSOLE_LEVEL_VERBOSE,
    console_level_from_flags,
    initialize_run_log_files,
    print_pre_run_banner,
)
from odcr_core.manifests import write_resolved_config_artifacts  # noqa: E402
from odcr_core.runners import _record_child_output  # noqa: E402
from paths_config import DEFAULT_MIRROR_LOG, append_log_dual  # noqa: E402
from train_logging import (  # noqa: E402
    append_step3_epoch_summary_csv,
    append_step3_gpu_profile_jsonl,
    append_step3_loss_breakdown_jsonl,
    append_step3_timing_profile_jsonl,
    append_train_epoch_metrics_jsonl,
)
from tools.odcr_post_edit_check import build_plan, plan_safety_violations  # noqa: E402


def _summary_cfg(tmp: Path):
    meta = tmp / "runs" / "step3" / "task2" / "1" / "meta"
    cfg, _, _ = odcr.resolve_config(
        config_path=REPO_ROOT / "configs" / "odcr.yaml",
        command="step3",
        task_id=2,
        set_overrides=[],
        dry_run=True,
        run_id="1",
        mode="full",
    )
    return replace(
        cfg,
        repo_root=tmp,
        manifest_dir=str(meta),
        log_dir=str(meta),
        checkpoint_dir=str(meta.parent),
        run_name="1",
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
            self.assertIn("stage=step3 task=2", text)
            self.assertIn("run_summary=", text)
            self.assertIn("console.log", text)
            self.assertNotIn("Source table:", text)
            self.assertNotIn("field_sources", text)
            self.assertNotIn("resolved_snapshot", text)
            self.assertNotIn("RUN_CONFIG", text)
            self.assertIn("profile=task2_strong_forward_g1s", text)
            self.assertIn("global/per_gpu/world=", text)

    def test_full_log_contains_launcher_and_raw_stream_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            cfg = _summary_cfg(Path(tmp_raw))
            initialize_run_log_files(
                cfg,
                {"train": {"stage": "step3"}, "field_sources": {"step3.train.batch_size": "configs/odcr.yaml"}},
                command_line="./odcr step3 --task 2",
                started_at="2026-04-28T00:00:00Z",
                console_level=CONSOLE_LEVEL_SUMMARY,
            )
            _record_child_output(cfg, "[child] hello detail stream", console_level=CONSOLE_LEVEL_SUMMARY)
            full = (Path(cfg.manifest_dir) / "full.log").read_text(encoding="utf-8")
            debug = (Path(cfg.manifest_dir) / "debug.log").read_text(encoding="utf-8")
            self.assertIn("ODCR RUN LOGGING POLICY odcr_step3_logging/2", full)
            self.assertIn("RAW CHILD STDOUT/STDERR STREAM", full)
            self.assertIn("[raw child] [child] hello detail stream", full)
            self.assertIn("[child] hello detail stream", debug)
            self.assertGreater(len(full), len(debug))

    def test_errors_log_context_includes_parent_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            cfg = _summary_cfg(Path(tmp_raw))
            initialize_run_log_files(
                cfg,
                {"train": {"stage": "step3"}},
                command_line="./odcr step3 --task 2",
                started_at="2026-04-28T00:00:00Z",
                console_level=CONSOLE_LEVEL_SUMMARY,
            )
            _record_child_output(cfg, "WARNING: cuda out of memory context", console_level=CONSOLE_LEVEL_SUMMARY)
            errors = (Path(cfg.manifest_dir) / "errors.log").read_text(encoding="utf-8")
            self.assertIn("rank=parent", errors)
            self.assertIn("local_rank=parent", errors)
            self.assertIn("pid=", errors)
            self.assertIn("hostname=", errors)
            self.assertIn("run_id=1", errors)
            self.assertIn("cuda out of memory", errors)

    def test_step3_structured_metric_writers_emit_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            log_file = str(Path(tmp_raw) / "runs" / "step3" / "task2" / "dry_run" / "meta" / "full.log")
            append_train_epoch_metrics_jsonl(
                log_file=log_file,
                row={"global_step": 50, "epoch": 1, "train_loss": 1.25, "valid_loss": None, "throughput": 10.0, "lr": 7e-4},
            )
            append_step3_loss_breakdown_jsonl(
                log_file=log_file,
                row={
                    "global_step": 50,
                    "epoch": 1,
                    "total_loss": 1.25,
                    "components": {"L_rating_shared": 1.0},
                    "component_weights": {"L_rating_shared": 1.0},
                    "weighted_contributions": {"L_rating_shared": 1.0},
                    "participates_in_total": {"L_rating_shared": True},
                    "finite_status": {"L_rating_shared": True},
                },
            )
            append_step3_timing_profile_jsonl(
                log_file=log_file,
                row={
                    "global_step": 50,
                    "epoch": 1,
                    "dataloader_wait": 0.1,
                    "h2d": 0.2,
                    "forward": 0.3,
                    "loss": 0.1,
                    "backward": 0.4,
                    "optimizer": 0.1,
                    "scheduler": 0.01,
                    "sync": 0.02,
                    "logging": 0.01,
                    "total_step_time": 1.25,
                },
            )
            append_step3_gpu_profile_jsonl(
                log_file=log_file,
                row={
                    "rank": 0,
                    "device": "cuda:0",
                    "peak_allocated": 10,
                    "peak_reserved": 20,
                    "current_allocated": 5,
                    "current_reserved": 12,
                },
            )
            append_step3_epoch_summary_csv(
                log_file=log_file,
                row={
                    "epoch": 1,
                    "train_loss": 1.2,
                    "valid_loss": 1.1,
                    "best_metric": 1.1,
                    "elapsed_s": 123.0,
                    "samples_per_sec": 42.0,
                    "checkpoint_path": "checkpoint.pt",
                    "status": "ok",
                },
            )
            meta = Path(log_file).parent
            self.assertTrue((meta / "metrics.jsonl").is_file())
            self.assertTrue((meta / "loss_breakdown.jsonl").is_file())
            self.assertTrue((meta / "timing_profile.jsonl").is_file())
            self.assertTrue((meta / "gpu_profile.jsonl").is_file())
            self.assertTrue((meta / "epoch_summary.csv").is_file())
            loss_row = json.loads((meta / "loss_breakdown.jsonl").read_text(encoding="utf-8").splitlines()[0])
            timing_row = json.loads((meta / "timing_profile.jsonl").read_text(encoding="utf-8").splitlines()[0])
            gpu_row = json.loads((meta / "gpu_profile.jsonl").read_text(encoding="utf-8").splitlines()[0])
            header = (meta / "epoch_summary.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertIn("component_weights", loss_row)
            self.assertIn("weighted_contributions", loss_row)
            self.assertIn("total_step_time", timing_row)
            self.assertIn("peak_reserved", gpu_row)
            self.assertEqual(
                header,
                [
                    "epoch",
                    "train_loss",
                    "valid_loss",
                    "best_metric",
                    "delta_from_best",
                    "delta_recent",
                    "lr_base",
                    "lr_effective",
                    "base_min_lr",
                    "effective_min_lr",
                    "damping_event",
                    "objective_drift_status",
                    "loss_phase",
                    "checkpoint_improved",
                    "effective_improvement_status",
                    "recommended_action",
                    "elapsed_s",
                    "samples_per_sec",
                    "checkpoint_path",
                    "status",
                ],
            )

    def test_resolved_config_and_source_table_retain_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_raw:
            meta = Path(tmp_raw) / "runs" / "step3" / "task4" / "1" / "meta"
            config_path, source_path = write_resolved_config_artifacts(
                meta,
                {
                    "train": {"batch_size": 16, "per_gpu_batch_size": 2},
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
        base_args = odcr.build_parser().parse_args(["show", "--stage", "step3", "--task", "2"])
        verbose_args = odcr.build_parser().parse_args(["show", "--stage", "step3", "--task", "2", "--verbose"])
        debug_args = odcr.build_parser().parse_args(["show", "--stage", "step3", "--task", "2", "--debug"])
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
