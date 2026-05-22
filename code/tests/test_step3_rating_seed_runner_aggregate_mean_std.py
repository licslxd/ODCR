from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_rating_seed_runner import DEFAULT_RATING_SEEDS, aggregate_step3_rating_eval_five_seed  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class Step3RatingSeedRunnerAggregateMeanStdTest(unittest.TestCase):
    def _write_source_run(self, repo: Path, *, run_id: int = 2) -> None:
        run = repo / "runs" / "step3" / "task2" / str(run_id)
        ckpt = run / "model" / "best.pth"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"checkpoint")
        _write_json(
            run / "meta" / "stage_status.json",
            {
                "stage": "step3",
                "task": 2,
                "run_id": str(run_id),
                "final_status": "completed_with_eval_handoff",
            },
        )

    def _write_eval_split(self, repo: Path, *, seed: int, split: str, mae: float, rmse: float) -> None:
        sample_count = 109732 if split == "valid" else 109720
        artifact = (
            repo
            / "runs"
            / "step3"
            / "task2"
            / "eval"
            / "5"
            / f"seed{seed}"
            / split
            / f"eval_paper_target_only_eval_{split}"
        )
        _write_json(
            artifact / "eval_summary.json",
            {
                "eval_status": "completed",
                "eval_protocol": "paper_target_only_eval",
                "target_only": True,
                "sample_count": sample_count,
                "metrics": {"recommendation": {"mae": mae, "rmse": rmse}},
            },
        )
        _write_json(
            artifact / "sample_integrity_report.json",
            {"status": "PASS", "sample_count": sample_count, "count_match": True},
        )

    def test_aggregate_writes_sample_std_with_ddof_1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self._write_source_run(repo, run_id=2)
            valid_maes = [0.50, 0.52, 0.54, 0.56, 0.58]
            for idx, seed in enumerate(DEFAULT_RATING_SEEDS):
                self._write_eval_split(repo, seed=seed, split="valid", mae=valid_maes[idx], rmse=0.80 + idx * 0.01)
                self._write_eval_split(repo, seed=seed, split="test", mae=0.51 + idx * 0.01, rmse=0.81 + idx * 0.01)
            result = aggregate_step3_rating_eval_five_seed(repo, task=2, source_run_id="2")
            self.assertTrue(result["paper_comparable_mean_std"])
            self.assertEqual(result["std_ddof"], 1)
            expected_mean = sum(valid_maes) / len(valid_maes)
            expected_std = math.sqrt(sum((x - expected_mean) ** 2 for x in valid_maes) / 4)
            self.assertAlmostEqual(result["valid"]["mae_mean"], expected_mean)
            self.assertAlmostEqual(result["valid"]["mae_std"], expected_std)
            self.assertTrue((repo / "runs" / "step3" / "task2" / "eval" / "5" / "step3_rating_task2_eval_5seed_runs.csv").is_file())
            self.assertTrue((repo / "runs" / "step3" / "task2" / "eval" / "5" / "step3_rating_task2_eval_5seed_mean_std.json").is_file())
            self.assertTrue((repo / "runs" / "step3" / "task2" / "eval" / "5" / "step3_rating_task2_eval_5seed_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
