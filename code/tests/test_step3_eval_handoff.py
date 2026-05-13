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
from odcr_core.stage_truth_antiforgery import write_json, write_step3_fixture  # noqa: E402


class Step3EvalHandoffRetiredTest(unittest.TestCase):
    def test_readiness_audit_not_eval_handoff_selects_step3_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            self.assertEqual(_latest_run(repo, 2, "step3", dry_run=False), "2")

    def test_paper_metric_handoff_sidecar_cannot_make_step3_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run = write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            (run / "meta" / "readiness_audit.json").unlink()
            write_json(
                run / "meta" / "eval_handoff.json",
                {
                    "schema_version": "odcr_step3_eval_handoff/1",
                    "paper_eval_protocol": "paper_target_only_eval",
                    "paper_eval_status": "completed",
                    "checkpoint_path": str(run / "model" / "best_observed.pth"),
                },
            )
            with self.assertRaisesRegex(OneControlConfigError, "readiness_audit"):
                _latest_run(repo, 2, "step3", dry_run=False)

    def test_readiness_gate_excludes_paper_text_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run = write_step3_fixture(repo, task=2, run_id="2", active=True, eligible=True)
            readiness = json.loads((run / "meta" / "readiness_audit.json").read_text(encoding="utf-8"))
            readiness["paper_eval_valid_metrics"] = {"BLEU-4": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}
            readiness["paper_eval_test_metrics"] = {"BLEU-4": 0.0, "ROUGE-L": 0.0, "METEOR": 0.0}
            write_json(run / "meta" / "readiness_audit.json", readiness)
            self.assertEqual(_latest_run(repo, 2, "step3", dry_run=False), "2")


if __name__ == "__main__":
    unittest.main()
