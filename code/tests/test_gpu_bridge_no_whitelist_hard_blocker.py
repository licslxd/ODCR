from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402


class GpuBridgeNoWhitelistHardBlockerTest(unittest.TestCase):
    def test_mode_specs_whitelist_hard_blocker_absent(self) -> None:
        source = Path(bridge.__file__).read_text(encoding="utf-8")
        self.assertNotIn("MODE_SPECS", source)
        for mode in ("repo-command", "repo-script", "repo-module", "command-file"):
            self.assertIn(mode, bridge.OPERATION_SPECS)

    def test_repo_local_module_and_script_allowed(self) -> None:
        output_dir = "AI_analysis/06_probe_evidence/test_gpu_bridge_no_whitelist"
        module_argv = bridge.runtime_command_argv(
            bridge.BridgeOptions(
                mode="repo-module",
                module_name="tools.odcr_step3_performance_probe",
                command_argv=("--help",),
                output_dir=output_dir,
            ),
            "bridge_test_module",
        )
        self.assertIn("-m", module_argv)
        self.assertIn("tools.odcr_step3_performance_probe", module_argv)

        script_argv = bridge.runtime_command_argv(
            bridge.BridgeOptions(
                mode="repo-script",
                script_path="code/tools/odcr_step3_performance_probe.py",
                command_argv=("--help",),
                output_dir=output_dir,
            ),
            "bridge_test_script",
        )
        self.assertIn("code/tools/odcr_step3_performance_probe.py", script_argv)

    def test_mode_mismatch_no_longer_blocks_repo_command(self) -> None:
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
