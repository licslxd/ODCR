from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from tools.odcr_step3_checkpoint_write_preflight import REQUIRED_EVENT_FIELDS, run_preflight  # noqa: E402


class TestStep3CheckpointLineageEventRequiredFields(unittest.TestCase):
    def test_checkpoint_lineage_event_contains_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(task_id=2, output_root=tmp)
            ledger = json.loads(Path(result["ledger_path"]).read_text(encoding="utf-8"))
            event = ledger["saved_checkpoint_events"][-1]
            missing = [field for field in REQUIRED_EVENT_FIELDS if field not in event]
            self.assertEqual(missing, [])
            self.assertEqual(event["selection_scope"], "latest")
            self.assertEqual(event["reason"], "latest_epoch_snapshot")


if __name__ == "__main__":
    unittest.main()
