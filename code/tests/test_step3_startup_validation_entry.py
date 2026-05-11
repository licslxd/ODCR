from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
TOOLS_DIR = CODE_DIR / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from odcr_core import path_layout  # noqa: E402
import odcr_step3_startup_validation as startup  # noqa: E402


class Step3StartupValidationEntryTest(unittest.TestCase):
    def test_parser_is_closed_choice_startup_only_validation_namespace(self) -> None:
        parser = startup.build_parser()
        args = parser.parse_args(["--run-id", "bridge_test"])
        self.assertEqual(args.task, 2)
        self.assertEqual(args.mode, "startup-only")
        self.assertEqual(args.namespace, "validation")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--run-id", "bridge_test", "--mode", "full"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--run-id", "bridge_test", "--namespace", "formal"])

    def test_validation_paths_are_isolated_from_formal_namespace(self) -> None:
        paths = startup._validation_paths(startup.REPO_ROOT, startup.DEFAULT_SLUG, "bridge_test")
        self.assertIn("runs/step3_validation/step3_tmux_gpu_bridge_startup_validation_closeout/bridge_test", paths["run_root"].as_posix())
        self.assertIn(
            "AI_analysis/06_probe_evidence/step3_tmux_gpu_bridge_startup_validation_closeout/bridge_test",
            paths["evidence"].as_posix(),
        )
        formal_root = path_layout.get_stage_task_root(startup.REPO_ROOT, "step3", 2)
        self.assertNotEqual(paths["run_root"].resolve(), formal_root.resolve())
        self.assertNotIn("runs/step3/task2", paths["run_root"].as_posix())
        self.assertNotIn("runs/step3/task2", paths["evidence"].as_posix())

    def test_summary_writer_does_not_update_formal_latest(self) -> None:
        source = inspect.getsource(startup._write_summary)
        self.assertIn("write_run_summary_json", source)
        self.assertIn("update_latest=False", source)
        run_source = inspect.getsource(startup.run_validation)
        self.assertIn("formal_latest_updated", run_source)
        self.assertIn("formal_namespace_polluted", run_source)
        self.assertIn("checkpoint_created", run_source)
        self.assertIn("training_loop_full_epoch_started", run_source)
        self.assertNotIn("best.pth", run_source)
        self.assertNotIn("./odcr step3", run_source)

    def test_runtime_summary_tracks_thread_and_cache_startup_contract(self) -> None:
        source = inspect.getsource(startup.run_validation) + inspect.getsource(startup._thread_trace)
        for token in (
            "TOKENIZERS_PARALLELISM",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "num_proc",
            "max_parallel_cpu",
            "reserved_cpu",
            "tokenization_formula",
            "worker_formula",
            "distributed_collective_calls_in_cache_phase",
            "nccl_init_after_cache_ready",
            "ranks_seen",
            "cache_dir",
            "cache_key",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
