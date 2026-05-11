from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.manifests import build_run_summary, training_runtime_config_path, write_training_runtime_config_artifact  # noqa: E402


class Step3RuntimeConfigFailureArtifactTest(unittest.TestCase):
    def test_failed_summary_does_not_require_missing_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_dir = repo / "runs" / "step3" / "task2" / "1"
            meta = run_dir / "meta"
            meta.mkdir(parents=True)
            (meta / "errors.log").write_text(
                "[Tokenize] step3 pre-DDP cache key | fingerprint=abc | cache_dir=/tmp/cache-x\n"
                "WorkNCCL(SeqNum=5, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)\n",
                encoding="utf-8",
            )
            summary = build_run_summary(
                repo_root=repo,
                run_dir=run_dir,
                meta_dir=meta,
                run_id="1",
                stage="step3",
                status="failed",
                started_at="2026-05-06T00:00:00Z",
                finished_at="2026-05-06T00:01:00Z",
                task_id=2,
                key_artifacts={
                    "training_runtime_config": training_runtime_config_path(meta),
                    "model": run_dir / "model" / "best.pth",
                },
                latest_error="torchrun failed",
            )
            self.assertIsNone(summary["training_runtime_config_path"])
            self.assertIsNone(summary["training_runtime_config_hash"])
            self.assertIn("training_runtime_config", summary["optional_artifacts"])
            self.assertEqual(summary["failure_phase"], "tokenization_cache")
            self.assertFalse(summary["training_loop_started"])
            self.assertFalse(summary["checkpoint_created"])
            self.assertEqual(summary["failure_root_signature"]["nccl"]["seq_num"], 5)

    def test_runtime_config_exists_before_cache_build_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "runs" / "step3" / "task2" / "1" / "meta"
            path = write_training_runtime_config_artifact(
                meta,
                {
                    "phase": "parent_pre_ddp_initial",
                    "cache_status": "not_started",
                    "training_loop_started": False,
                },
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "parent_pre_ddp_initial")
            self.assertEqual(payload["cache_status"], "not_started")
            self.assertFalse(payload["training_loop_started"])


if __name__ == "__main__":
    unittest.main()
