from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_rating_seed_runner import (  # noqa: E402
    DEFAULT_RATING_SEEDS,
    build_rating_seed_plan,
    run_step3_rating_seed_runner,
)


class Step3RatingSeedRunnerSingleMultiTest(unittest.TestCase):
    def test_single_plan_is_one_explicit_seed_and_run(self) -> None:
        plan = build_rating_seed_plan(task=2, mode="single", seed=3407, run_id="10")
        self.assertEqual([(item.seed, item.run_id) for item in plan], [(3407, "10")])

    def test_multi_plan_is_fixed_five_seed_eval_against_one_source_run(self) -> None:
        plan = build_rating_seed_plan(task=2, mode="multi", run_id="2")
        self.assertEqual([item.seed for item in plan], list(DEFAULT_RATING_SEEDS))
        self.assertEqual([item.run_id for item in plan], ["2", "2", "2", "2", "2"])

    def test_multi_accepts_legacy_run_id_start_as_source_run_for_dry_run(self) -> None:
        plan = build_rating_seed_plan(task=2, mode="multi", run_id_start=2)
        self.assertEqual([item.seed for item in plan], list(DEFAULT_RATING_SEEDS))
        self.assertEqual([item.run_id for item in plan], ["2", "2", "2", "2", "2"])

    def test_dry_run_reports_commands_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "odcr").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            result = run_step3_rating_seed_runner(
                repo,
                task=2,
                mode="single",
                config_path="configs/odcr.yaml",
                seed=3407,
                run_id="10",
                dry_run=True,
            )
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["mode"], "single")
            self.assertFalse(result["formal_training_executed"])
            self.assertEqual(result["eval_namespace"], "runs/step3/task2/eval/1")
            self.assertEqual(result["commands"][0]["paper_eval_valid"]["set"][0], "project.seed=3407")
            self.assertIn("paper_target_only_eval", result["commands"][0]["paper_eval_valid"]["set"][1])
            self.assertIn("runs/step3/task2/eval/1/seed3407/valid", result["commands"][0]["paper_eval_valid"]["log_dir"])
            self.assertNotIn("train", result["commands"][0])


if __name__ == "__main__":
    unittest.main()
