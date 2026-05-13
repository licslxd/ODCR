from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.manifests import _extract_failure_root_signature


class Step3OomFailureClassificationTest(unittest.TestCase):
    def test_cuda_backward_oom_beats_tokenization_cache_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meta = tmp_path / "meta"
            meta.mkdir()
            (meta / "full.log").write_text(
                "\n".join(
                    [
                        "[Tokenize] fingerprint=abc cache_dir=cache/step3/tokenizer/abc",
                        "[Train/no_accum] n_optimizer_steps=100",
                        "[Epoch Summary] epoch=1 valid_loss=1.0",
                        "Traceback (most recent call last):",
                        "  File \"code/executors/step3_train_core.py\", line 5587, in trainModel",
                        "    loss.backward()",
                        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.41 GiB.",
                        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:",
                    ]
                ),
                encoding="utf-8",
            )
            checkpoint = tmp_path / "model" / "best.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")

            details = _extract_failure_root_signature(
                meta=meta,
                latest_error="Command '['torchrun']' returned non-zero exit status 1.",
                repo_root=tmp_path,
                checkpoint_path=checkpoint,
            )

            self.assertEqual(details["failure_phase"], "epoch_boundary_backward_oom")
            self.assertEqual(details["fatal_source"], "torchrun_child_rank_oom")
            self.assertEqual(details["root_cause"], "cuda_out_of_memory_during_loss_backward")
            self.assertIs(details["training_loop_started"], True)
            self.assertIs(details["checkpoint_created"], True)
            self.assertEqual(details["cache_status"], "failed_or_missing")


if __name__ == "__main__":
    unittest.main()
