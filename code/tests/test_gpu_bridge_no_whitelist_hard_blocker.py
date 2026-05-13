from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402


class GpuBridgeRepoCommandAllowlistTest(unittest.TestCase):
    def test_repo_command_modes_are_metadata_registered(self) -> None:
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        self.assertNotIn("MODE_SPECS", source)
        for mode in ("repo-command", "repo-script", "repo-module", "command-file"):
            self.assertIn(mode, bridge.OPERATION_SPECS)
        from odcr_core.aux.runtime.stage_dispatch import REPO_COMMAND_REGISTRY

        self.assertIn("step3_bounded_probe", REPO_COMMAND_REGISTRY)
        self.assertIn("step4_bounded_preflight", REPO_COMMAND_REGISTRY)

    def test_unregistered_repo_local_module_and_script_denied(self) -> None:
        output_dir = "AI_analysis/06_probe_evidence/test_gpu_bridge_no_whitelist"
        with self.assertRaises(bridge.BridgeError):
            bridge.runtime_command_argv(
                bridge.BridgeOptions(
                    mode="repo-module",
                    module_name="tools.odcr_step3_performance_probe",
                    command_argv=("--help",),
                    output_dir=output_dir,
                ),
                "bridge_test_module",
            )

        with self.assertRaises(bridge.BridgeError):
            bridge.runtime_command_argv(
                bridge.BridgeOptions(
                    mode="repo-script",
                    script_path="code/tools/odcr_step3_performance_probe.py",
                    command_argv=("--help",),
                    output_dir=output_dir,
                ),
                "bridge_test_script",
            )

    def test_registered_repo_command_allows_exact_probe_help(self) -> None:
        argv = bridge.runtime_command_argv(
            bridge.BridgeOptions(
                mode="repo-command",
                command_argv=("python", "code/tools/odcr_step3_performance_probe.py", "--help"),
                output_dir="AI_analysis/06_probe_evidence/test_gpu_bridge_no_whitelist",
            ),
            "bridge_test_repo_command",
        )
        self.assertEqual(argv[:2], ("python", "code/tools/odcr_step3_performance_probe.py"))


if __name__ == "__main__":
    unittest.main()
