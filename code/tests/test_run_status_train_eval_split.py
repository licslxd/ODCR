from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.manifests import build_run_summary_for_config  # noqa: E402


class TestRunStatusTrainEvalSplit(unittest.TestCase):
    def test_eval_sidecar_overrides_train_eval_status(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            run = Path(tmp)
            meta = run / "meta"
            model = run / "model"
            state = run / "state"
            meta.mkdir()
            model.mkdir()
            state.mkdir()
            (model / "best.pth").write_bytes(b"x")
            (meta / "step3_eval_status.json").write_text(
                json.dumps(
                    {
                        "train_status": "completed",
                        "eval_status": "failed",
                        "quality_status": "not_evaluated",
                        "downstream_ready": False,
                        "failure_phase": "post_train_eval",
                    }
                ),
                encoding="utf-8",
            )
            cfg = SimpleNamespace(
                repo_root=ROOT,
                command="step3",
                checkpoint_dir=str(run),
                manifest_dir=str(meta),
                log_dir=str(meta),
                run_name=run.name,
                task_id=2,
                auxiliary="AM_Movies",
                target="AM_CDs",
                step3_mode="full",
                step3_eval_protocol="minimal_eval",
                step3_eval_split="valid",
            )
            summary = build_run_summary_for_config(cfg, status="failed", started_at="2026-05-08T00:00:00Z")
            self.assertEqual(summary["train_status"], "completed")
            self.assertEqual(summary["eval_status"], "failed")
            self.assertEqual(summary["failure_phase"], "post_train_eval")


if __name__ == "__main__":
    unittest.main()
