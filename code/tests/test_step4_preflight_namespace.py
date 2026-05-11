from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step4_runtime import run_step4_bounded_preflight


class Step4PreflightNamespaceTest(unittest.TestCase):
    def test_dry_run_uses_non_formal_namespace(self) -> None:
        cfg = SimpleNamespace(repo_root=Path("/tmp/odcr"), task_id=2, step4_runtime_config_json="{}")
        payload = run_step4_bounded_preflight(
            cfg,
            max_samples=8,
            validation_namespace="step4_preflight_smoke",
            dry_run=True,
        )
        self.assertIn("runs/step4_preflight/task2/step4_preflight_smoke", payload["output_dir"])
        self.assertNotIn("runs/step4/task2", payload["output_dir"])
        self.assertFalse(payload["formal_latest_write"])
        self.assertFalse(payload["formal_export_write"])

    def test_rejects_formal_namespace_tokens(self) -> None:
        cfg = SimpleNamespace(repo_root=Path("/tmp/odcr"), task_id=2, step4_runtime_config_json="{}")
        with self.assertRaises(Exception):
            run_step4_bounded_preflight(cfg, validation_namespace="../step4/task2/latest", dry_run=True)


if __name__ == "__main__":
    unittest.main()
