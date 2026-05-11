from __future__ import annotations

import unittest
from pathlib import Path


class TestStep3EvalTwoPhaseNoBarrierAfterCpuMetric(unittest.TestCase):
    def test_cpu_metric_runs_after_destroy_without_later_barrier(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("executors/step3_train_core.py").read_text(encoding="utf-8")
        eval_block = text[text.index("def _run_eval_ddp"):]
        destroy_idx = eval_block.index("dist.destroy_process_group()")
        metric_idx = eval_block.index("metrics_from_prediction_rows(")
        self.assertLess(destroy_idx, metric_idx)
        self.assertNotIn("dist.gather_object", eval_block)
        self.assertNotIn("rank_gathered_", eval_block)
        self.assertNotIn("dist.barrier()", eval_block[metric_idx:])


if __name__ == "__main__":
    unittest.main()
