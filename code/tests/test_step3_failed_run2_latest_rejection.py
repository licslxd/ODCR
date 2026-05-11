from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import _latest_run  # noqa: E402
from odcr_core.config_schema import OneControlConfigError  # noqa: E402


class TestStep3FailedRun2LatestRejection(unittest.TestCase):
    def test_failed_latest_and_failure_audit_block_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            meta = repo / "runs" / "step3" / "task2" / "2" / "meta"
            meta.mkdir(parents=True)
            summary = meta / "run_summary.json"
            summary.write_text(json.dumps({"run_id": "2", "status": "failed"}), encoding="utf-8")
            (meta / "failure_audit.json").write_text(
                json.dumps(
                    {
                        "run_id": "2",
                        "status": "failed",
                        "do_not_downstream": True,
                        "root_cause": "checkpoint_event_from_sidecar_missing_reason_replaced_previous",
                    }
                ),
                encoding="utf-8",
            )
            latest = repo / "runs" / "step3" / "task2" / "latest.json"
            latest.write_text(
                json.dumps(
                    {
                        "latest_run_id": "2",
                        "latest_status": "failed",
                        "latest_summary_path": str(summary),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OneControlConfigError, "not eligible for Step4 formal upstream"):
                _latest_run(repo, 2, "step3", dry_run=False)


if __name__ == "__main__":
    unittest.main()
