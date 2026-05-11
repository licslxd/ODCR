from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step3NoAccumTrainingLoopTest(unittest.TestCase):
    def test_step3_and_step5_training_loops_have_no_accumulation_paths(self) -> None:
        for rel in ("code/executors/step3_train_core.py", "code/executors/step5_engine.py"):
            with self.subTest(rel=rel):
                text = (REPO_ROOT / rel).read_text(encoding="utf-8")
                self.assertNotIn(".no_sync(", text)
                self.assertNotIn("_ddp_no_sync_model", text)
                self.assertNotIn("gradient_accumulation_steps", text)
                self.assertNotIn("micro_step_count", text)
                self.assertNotIn("inv_accum", text)
                self.assertNotIn("odcr_step3_no_accum/1", text)
        step3 = (REPO_ROOT / "code/executors/step3_train_core.py").read_text(encoding="utf-8")
        step5 = (REPO_ROOT / "code/executors/step5_engine.py").read_text(encoding="utf-8")
        self.assertIn("batches_per_epoch", step3)
        self.assertIn("n_steps = max(1, n_batches)", step5)
        self.assertIn("optimizer.step()", step5)
        self.assertIn("sched.step()", step5)
        self.assertIn("batch_semantics_version=odcr_no_accum/1", step5)


if __name__ == "__main__":
    unittest.main()
