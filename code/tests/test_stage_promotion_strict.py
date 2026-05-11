from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.stage_promotion import promote_upstream  # noqa: E402
from odcr_core.stage_status import read_stage_status  # noqa: E402
from odcr_core.stage_truth_antiforgery import mutate_status, write_step3_fixture  # noqa: E402
from odcr_core.upstream_resolver import UpstreamResolutionError  # noqa: E402


class StagePromotionStrictTest(unittest.TestCase):
    def test_dry_run_validates_target_and_writes_pointer_only_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            result = promote_upstream(repo_root=repo, stage="step3", task=2, run_id="2", dry_run=True)
            latest = result["latest_payload"]
            self.assertEqual(latest["active_run_id"], "2")
            self.assertIn("latest_stage_status_path", latest)
            self.assertNotIn("latest_status", latest)
            self.assertTrue(any(str(item).endswith("promotion_history.jsonl") for item in result["would_write"]))

    def test_malformed_target_rejected_before_pointer_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            write_step3_fixture(repo, task=2, run_id="3", eligible=True)
            mutate_status(repo, task=2, run_id="3", mutate=lambda payload: payload.__setitem__("artifacts", {}))
            with self.assertRaisesRegex(UpstreamResolutionError, "stage_status_strict_validation_failed"):
                promote_upstream(repo_root=repo, stage="step3", task=2, run_id="3", dry_run=True)

    def test_actual_promotion_does_not_rewrite_old_stage_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            write_step3_fixture(repo, task=2, run_id="3", eligible=True)
            old_before = read_stage_status(repo / "runs" / "step3" / "task2" / "2")
            result = promote_upstream(repo_root=repo, stage="step3", task=2, run_id="3", dry_run=False)
            old_after = read_stage_status(repo / "runs" / "step3" / "task2" / "2")
            self.assertEqual(old_after["final_status"], old_before["final_status"])
            self.assertTrue(result["historical_stage_status_immutable"])
            latest = json.loads((repo / "runs" / "step3" / "task2" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["latest_run_id"], "3")
            self.assertFalse((repo / "runs" / "step3" / "task2" / "2" / "meta" / "stage_status.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
