from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executors import step3_train_core as s3


class Step3EpochBoundaryMemoryCleanupTest(unittest.TestCase):
    def test_epoch_boundary_cleanup_emits_cpu_json_memory_row(self) -> None:
        cfg = SimpleNamespace(run_id="cleanup_probe", task_idx=2, task_profile_id="task2_strong_forward_g1s", log_file=None)

        row = s3._cleanup_cuda_epoch_boundary(
            final_cfg=cfg,
            rank=0,
            device="cpu",
            global_step=12,
            epoch=1,
            reset_peak=True,
        )

        self.assertEqual(row["phase"], "after_epoch_boundary_cleanup")
        self.assertIs(row["cleanup_gc_collect"], True)
        self.assertEqual(row["allocated_gib"], 0.0)
        self.assertEqual(row["reserved_gib"], 0.0)
        self.assertEqual(row["global_step"], 12)


    def test_to_scalar_cpu_json_detaches_nested_tensors(self) -> None:
        import torch

        payload = s3._to_scalar_cpu_json({"loss": torch.tensor(1.25), "vector": torch.tensor([1, 2])})

        self.assertEqual(payload, {"loss": 1.25, "vector": [1, 2]})


if __name__ == "__main__":
    unittest.main()
