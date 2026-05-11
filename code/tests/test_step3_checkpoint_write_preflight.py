from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.odcr_step3_checkpoint_write_preflight import REQUIRED_EVENT_FIELDS, run_preflight  # noqa: E402


class TestStep3CheckpointWritePreflight(unittest.TestCase):
    def test_preflight_writes_temp_lineage_event_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(task_id=2, output_root=tmp)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["reason"], "latest_epoch_snapshot")
            self.assertFalse(result["replaced_previous"])
            self.assertTrue(Path(result["lineage_path"]).is_file())
            self.assertEqual(tuple(result["required_event_fields"]), REQUIRED_EVENT_FIELDS)


if __name__ == "__main__":
    unittest.main()
