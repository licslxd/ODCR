from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.aux.runtime.stage_dispatch import classify_repo_command, get_runtime_command, runtime_probe_bridge_args


class AuxStageDispatchTest(unittest.TestCase):
    def test_registry_declares_required_metadata(self) -> None:
        spec = get_runtime_command("step3_bounded_probe")
        self.assertEqual(spec.stage, "step3")
        self.assertTrue(spec.allow_gpu)
        self.assertFalse(spec.allow_formal_run)
        self.assertIn("AI_analysis", spec.output_policy)

    def test_arbitrary_python_and_shell_are_denied(self) -> None:
        self.assertFalse(classify_repo_command(("python", "-c", "print(1)")).allowed)
        self.assertFalse(classify_repo_command(("bash", "-c", "echo hi")).allowed)
        self.assertFalse(classify_repo_command(("nohup", "./odcr", "step4")).allowed)

    def test_bounded_step4_validation_is_registered(self) -> None:
        admission = classify_repo_command(
            (
                "./odcr",
                "step4",
                "--task",
                "2",
                "--preflight",
                "--max-samples",
                "128",
                "--validation-namespace",
                "step4_preflight_smoke",
            )
        )
        self.assertTrue(admission.allowed, admission.reason)
        self.assertEqual(admission.command_id, "step4_bounded_preflight")
        self.assertEqual(admission.bounded_limit_value, 128)

    def test_step3_probe_builds_bridge_mode(self) -> None:
        args = runtime_probe_bridge_args(stage="step3", task=2, profile="csb_odcr_full", bounded=True, dry_run=True)
        self.assertEqual(args[0], "step3-performance-probe")
        self.assertIn("--candidate-name", args)
        self.assertIn("--dry-run", args)


if __name__ == "__main__":
    unittest.main()

