from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_runtime_probe import Step3RuntimeProbeError, Step3ValidationNamespaceGuard  # noqa: E402


class Step3ValidationNamespaceGuardTest(unittest.TestCase):
    def test_validation_path_is_allowed_but_formal_latest_write_attempt_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = Step3ValidationNamespaceGuard(root, 2, "truth_probe", "unit")
            allowed = guard.evidence_dir / "report.json"
            self.assertEqual(guard.assert_validation_path(allowed), allowed.resolve())
            formal_latest = root / "runs" / "step3" / "task2" / "latest.json"
            with self.assertRaises(Step3RuntimeProbeError):
                guard.assert_validation_path(formal_latest, role="formal latest")

    def test_formal_checkpoint_and_step4_tokens_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = Step3ValidationNamespaceGuard(root, 2, "truth_probe", "unit")
            with self.assertRaises(Step3RuntimeProbeError):
                guard.assert_no_formal_token("runs/step3/task2/1/model/best.pth")
            with self.assertRaises(Step3RuntimeProbeError):
                guard.assert_no_formal_token("./odcr step4 --task 2 --dry-run")

    def test_formal_file_change_is_pollution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_latest = root / "runs" / "step3" / "task2" / "latest.json"
            formal_latest.parent.mkdir(parents=True)
            formal_latest.write_text('{"before": true}\n', encoding="utf-8")
            guard = Step3ValidationNamespaceGuard(root, 2, "truth_probe", "unit")
            formal_latest.write_text('{"after": true}\n', encoding="utf-8")
            self.assertTrue(guard.formal_namespace_polluted())


if __name__ == "__main__":
    unittest.main(verbosity=2)

