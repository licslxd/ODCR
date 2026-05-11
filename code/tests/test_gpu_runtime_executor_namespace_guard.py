from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402


class GpuRuntimeExecutorNamespaceGuardTest(unittest.TestCase):
    def test_validation_output_namespace_allowed(self) -> None:
        path = bridge.resolve_runtime_output_dir(
            "AI_analysis/06_probe_evidence/test_gpu_runtime_executor",
            "bridge_test_runtime",
        )
        self.assertTrue(str(path).endswith("AI_analysis/06_probe_evidence/test_gpu_runtime_executor"))
        step4_path = bridge.resolve_runtime_output_dir(
            "runs/step4_preflight/task2/step4_preflight_smoke",
            "bridge_test_runtime",
        )
        self.assertTrue(str(step4_path).endswith("runs/step4_preflight/task2/step4_preflight_smoke"))

    def test_formal_output_namespace_blocked_without_confirmation(self) -> None:
        with self.assertRaises(bridge.BridgeError) as ctx:
            bridge.resolve_runtime_output_dir("runs/step3/task2/1/model", "bridge_test_runtime")
        self.assertEqual(ctx.exception.stop_reason, "formal_namespace_blocked")

    def test_bounded_step4_allowed_but_formal_step4_step5_eval_rerank_blocked(self) -> None:
        output_dir = bridge.resolve_runtime_output_dir(
            "AI_analysis/06_probe_evidence/test_gpu_runtime_executor",
            "bridge_test_runtime",
        )
        bridge.validate_runtime_command_safety(
            "./odcr step4 --task 2 --preflight --max-samples 128 --validation-namespace step4_preflight_smoke",
            output_dir=output_dir,
        )
        for command in (
            "./odcr step4 --task 4",
            "./odcr step5 --task 4 --dry-run",
            "./odcr eval --task 4",
            "./odcr rerank --task 4",
        ):
            with self.subTest(command=command):
                with self.assertRaises(bridge.BridgeError):
                    bridge.validate_runtime_command_safety(command, output_dir=output_dir)

    def test_command_file_must_live_in_validation_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_file = Path(tmp) / "cmd.sh"
            command_file.write_text("python code/tools/odcr_step3_performance_probe.py --help\n", encoding="utf-8")
            with self.assertRaises(bridge.BridgeError):
                bridge.runtime_command_argv(
                    bridge.BridgeOptions(
                        mode="command-file",
                        command_file=str(command_file),
                        output_dir="AI_analysis/06_probe_evidence/test_gpu_runtime_executor",
                    ),
                    "bridge_test_cmdfile",
                )


if __name__ == "__main__":
    unittest.main()
