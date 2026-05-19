from __future__ import annotations

import unittest
import sys

import tempfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from test_step4_export_validator import _write_ready_fixture  # noqa: E402
from odcr_core.stage_status import build_stage_status
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST


class Step4ReadinessRequiresManifestTest(unittest.TestCase):
    def test_stage_status_blocks_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "runs" / "step4" / "task2" / "2_1"
            meta = run / "meta"
            meta.mkdir(parents=True)
            _write_ready_fixture(run)
            (run / STEP4_EXPORT_MANIFEST).unlink()
            (meta / "run_summary.json").write_text(
                '{"stage":"step4","task_id":2,"run_id":"2_1","status":"ok"}',
                encoding="utf-8",
            )
            (meta / "source_table.json").write_text("{}", encoding="utf-8")
            (meta / "resolved_config.json").write_text("{}", encoding="utf-8")
            status = build_stage_status(repo_root=root, stage="step4", task=2, run_id="2_1")
            self.assertFalse(status["downstream_ready"])
            self.assertNotIn("step5", status["ready_for"])


if __name__ == "__main__":
    unittest.main()
