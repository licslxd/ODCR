from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import path_layout  # noqa: E402
from odcr_core.manifests import _extract_failure_root_signature  # noqa: E402


def _classify(meta: Path, repo: Path, *, checkpoint: Path | None = None, latest_error: str = "") -> dict[str, object]:
    return _extract_failure_root_signature(
        meta=meta,
        latest_error=latest_error,
        repo_root=repo,
        checkpoint_path=checkpoint,
    )


class Step3TrainingFirstFailureClassificationTest(unittest.TestCase):
    def test_nonfinite_gradient_beats_tokenizer_cache_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            meta = repo / "runs" / "step3" / "task2" / "1" / "meta"
            meta.mkdir(parents=True)
            (meta / "full.log").write_text(
                "\n".join(
                    [
                        "[Tokenize] fingerprint=abc cache_dir=/tmp/token-cache",
                        "[Train/no_accum] n_optimizer_steps=3000",
                        "loss_breakdown finite through optimizer_step=2600",
                        "Step3 nonfinite gradient gate aborted after 3 continuous skipped steps.",
                    ]
                ),
                encoding="utf-8",
            )

            details = _classify(meta, repo, latest_error="CalledProcessError: torchrun failed")

            self.assertEqual(details["failure_phase"], "training_loop_nonfinite_gradient_gate")
            self.assertEqual(details["fatal_source"], "Step3 grad finite gate")
            self.assertIn("post-backward gradient norm explosion", str(details["root_cause"]))
            self.assertNotEqual(details["failure_phase"], "tokenization_cache")
            self.assertIs(details["training_loop_started"], True)

    def test_checkpoint_created_blocks_tokenization_cache_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            meta = repo / "runs" / "step3" / "task2" / "1" / "meta"
            meta.mkdir(parents=True)
            checkpoint = repo / "runs" / "step3" / "task2" / "1" / "model" / "best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            (meta / "full.log").write_text(
                "[Tokenize] fingerprint=abc cache_dir=/tmp/token-cache\n"
                "torch.distributed.elastic.multiprocessing.errors.ChildFailedError\n",
                encoding="utf-8",
            )

            details = _classify(meta, repo, checkpoint=checkpoint, latest_error="ChildFailedError")

            self.assertIs(details["checkpoint_created"], True)
            self.assertIs(details["training_loop_started"], True)
            self.assertEqual(details["failure_phase"], "train_ddp_runtime_error")

    def test_epoch_summary_and_train_metrics_block_tokenization_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            meta = repo / "runs" / "step3" / "task2" / "1" / "meta"
            meta.mkdir(parents=True)
            (meta / path_layout.metrics_filename("epoch_summary")).write_text(
                "epoch,train_loss,valid_loss\n1,1.0,1.1\n",
                encoding="utf-8",
            )
            (meta / path_layout.metrics_filename("loss_breakdown")).write_text(
                json.dumps({"split": "train", "epoch": 2, "loss_name": "L_rating_shared"}) + "\n",
                encoding="utf-8",
            )
            (meta / "full.log").write_text(
                "[Tokenize] fingerprint=abc cache_dir=/tmp/token-cache\nChildFailedError\n",
                encoding="utf-8",
            )

            details = _classify(meta, repo, latest_error="ChildFailedError")

            self.assertTrue(details["epoch_summary_nonempty"])
            self.assertTrue(details["loss_breakdown_train_rows"])
            self.assertEqual(details["failure_phase"], "train_ddp_runtime_error")

    def test_backward_oom_classification_stays_training_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            meta = repo / "runs" / "step3" / "task2" / "1" / "meta"
            meta.mkdir(parents=True)
            (meta / "full.log").write_text(
                "[Tokenize] fingerprint=abc cache_dir=/tmp/token-cache\n"
                "[Train/no_accum] n_optimizer_steps=3000\n"
                "loss.backward()\n"
                "torch.OutOfMemoryError: CUDA out of memory\n",
                encoding="utf-8",
            )

            details = _classify(meta, repo, latest_error="rank=1 torch.OutOfMemoryError")

            self.assertEqual(details["failure_phase"], "train_backward_oom")
            self.assertEqual(details["fatal_source"], "torchrun_child_rank_oom")


if __name__ == "__main__":
    unittest.main()
