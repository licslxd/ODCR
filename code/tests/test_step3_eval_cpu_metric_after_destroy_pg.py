from __future__ import annotations

import unittest
from pathlib import Path


class TestStep3EvalCpuMetricAfterDestroyPg(unittest.TestCase):
    def test_destroy_process_group_precedes_cpu_metrics(self) -> None:
        text = Path(__file__).resolve().parents[1].joinpath("executors/step3_train_core.py").read_text(encoding="utf-8")
        block = text[text.index("def _run_eval_ddp"):]
        self.assertLess(block.index("dist.destroy_process_group()"), block.index("metrics_from_prediction_rows("))
        self.assertLess(block.index("dist.destroy_process_group()"), block.index("_write_eval_results_log("))


if __name__ == "__main__":
    unittest.main()
