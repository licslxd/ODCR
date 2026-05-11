from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.config_resolver import OneControlConfigError, _latest_run  # noqa: E402


class Step3FailedLatestGateTest(unittest.TestCase):
    def test_failed_step3_latest_is_rejected_before_checkpoint_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            meta = repo / "runs" / "step3" / "task2" / "1" / "meta"
            meta.mkdir(parents=True)
            summary = meta / "run_summary.json"
            summary.write_text(
                json.dumps({"run_id": "1", "stage": "step3", "task_id": 2, "status": "failed"}),
                encoding="utf-8",
            )
            latest = repo / "runs" / "step3" / "task2" / "latest.json"
            latest.write_text(
                json.dumps(
                    {
                        "latest_run_id": "1",
                        "latest_summary_path": summary.relative_to(repo).as_posix(),
                        "latest_status": "failed",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OneControlConfigError, "not eligible for Step4 formal upstream"):
                _latest_run(repo, 2, "step3", dry_run=False)


if __name__ == "__main__":
    unittest.main()
