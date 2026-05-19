from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step4_engine, step4_entry  # noqa: E402
from odcr_core import runners, step4_runtime  # noqa: E402
from odcr_core.step4_checkpoint_lineage import require_step4_lineage_field  # noqa: E402


class Step4NoRawLineageKeyErrorTest(unittest.TestCase):
    def test_step4_engine_uses_required_field_helper_for_lineage_hash(self) -> None:
        src = inspect.getsource(step4_engine._run_one_task)
        self.assertNotIn('step3_lineage["lineage_hash"]', src)
        self.assertNotIn("step3_lineage['lineage_hash']", src)
        self.assertIn("require_step4_lineage_field", src)
        with self.assertRaisesRegex(Exception, "available_keys"):
            require_step4_lineage_field({"checkpoint_sha256": "x"}, "lineage_hash")

    def test_run_id_and_log_file_are_propagated_to_child_entry(self) -> None:
        runner_src = inspect.getsource(runners.run_step4)
        entry_src = inspect.getsource(step4_entry.run_step4_cli)
        engine_src = inspect.getsource(step4_engine._run_one_task)
        runtime_src = inspect.getsource(step4_runtime.step4_runtime_env)
        self.assertIn("--run-id", runner_src)
        self.assertIn("--run-id", entry_src)
        self.assertIn("ODCR_STEP4_RUN_ID", runtime_src + engine_src)
        self.assertIn("log_file/run_dir propagation mismatch", engine_src)

    def test_no_lineage_bypass_switches_introduced(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                CODE_DIR / "odcr_core" / "step4_checkpoint_lineage.py",
                CODE_DIR / "odcr_core" / "step4_runtime.py",
                CODE_DIR / "executors" / "step4_engine.py",
            )
        )
        for forbidden in ("allow_mismatch", "skip_lineage", "force_load"):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
