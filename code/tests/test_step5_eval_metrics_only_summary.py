from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.step5_eval_summary import compact_post_train_eval_layout, write_post_train_eval_metrics_log


class Step5EvalMetricsOnlySummaryTest(unittest.TestCase):
    def test_writes_single_metrics_entrypoint_for_valid_and_test(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "post_train_eval"
            for split, bleu4 in (("valid", 0.0066), ("test", 0.0068)):
                split_dir = root / split
                (split_dir / "meta").mkdir(parents=True)
                (split_dir / "meta" / "run_summary.json").write_text(
                    json.dumps({"status": "ok", "duration_sec": 12}),
                    encoding="utf-8",
                )
                (split_dir / "eval_handoff.json").write_text(
                    json.dumps({"run_id": "1_2", "checkpoint": "/tmp/best.pth"}),
                    encoding="utf-8",
                )
                (split_dir / "eval_metrics.json").write_text(
                    json.dumps({"decode": {"decode_strategy": "uncertainty_low_temp_top_k"}}),
                    encoding="utf-8",
                )
                (split_dir / "paper_metrics.json").write_text(
                    json.dumps(
                        {
                            "split": split,
                            "rating_metrics": {
                                "valid": {"mae": 0.575, "rmse": 0.8473},
                                "test": {"mae": 0.5764, "rmse": 0.8494},
                            },
                            "explanation_metrics": {
                                "sample_count": 100,
                                "explanation": {
                                    "bleu": {"1": 0.3, "2": 0.1, "3": 0.02, "4": 0.01},
                                    "rouge": {"1": 2.8, "2": 0.1, "l": 1.8},
                                    "dist": {"1": 0.5, "2": 2.4},
                                    "meteor": 1.9,
                                },
                                "paper_metrics": {
                                    "bleu": {"1": 0.3, "2": 0.1, "3": 0.02, "4": bleu4},
                                    "rouge": {"rouge_1_f": 3.5, "rouge_2_f": 0.13, "rouge_l_f": 2.19},
                                    "distinct_corpus": {"scale_percent_0_100": {"1": 0.54, "2": 2.46}},
                                },
                                "collapse_stats": {
                                    "top1_pred_ratio": 0.6,
                                    "pred_unique_ratio": 0.09,
                                    "collapse_warnings": ["top1_pred_ratio>=0.2"],
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            out = Path(write_post_train_eval_metrics_log(root))
            content = out.read_text(encoding="utf-8")
            self.assertEqual(out.name, "metrics.log")
            self.assertIn("[valid]", content)
            self.assertIn("[test]", content)
            self.assertIn("[Recommendation]", content)
            self.assertIn("MAE = 0.575 | RMSE = 0.8473", content)
            self.assertIn("MAE = 0.5764 | RMSE = 0.8494", content)
            self.assertIn("ROUGE: 3.50, 0.13, 2.19", content)
            self.assertIn("BLEU: 0.30, 0.10, 0.02, 0.01", content)
            self.assertIn("DIST-1/DIST-2 (evaluate_text, paper-compatible): 0.54, 2.46", content)
            self.assertIn("METEOR: 1.90", content)

    def test_compacts_layout_and_moves_cache_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "runs" / "step5" / "task2" / "1_2" / "post_train_eval"
            root.mkdir(parents=True)
            (root / "metrics_only.json").write_text("{}", encoding="utf-8")
            for split in ("valid", "test"):
                split_dir = root / split
                (split_dir / "meta").mkdir(parents=True)
                (split_dir / "meta" / f"cache_reuse_decision_{split}.json").write_text(
                    json.dumps({"split": split}),
                    encoding="utf-8",
                )
                (split_dir / "meta" / "run_summary.json").write_text(
                    json.dumps({"status": "ok", "duration_sec": 12}),
                    encoding="utf-8",
                )
                (split_dir / "eval_handoff.json").write_text(
                    json.dumps({"run_id": "1_2", "checkpoint": "/tmp/best.pth"}),
                    encoding="utf-8",
                )
                (split_dir / "eval_metrics.json").write_text(
                    json.dumps({"decode": {"decode_strategy": "uncertainty_low_temp_top_k"}}),
                    encoding="utf-8",
                )
                (split_dir / "predictions.jsonl").write_text("{}", encoding="utf-8")
                (split_dir / "paper_metrics.json").write_text(
                    json.dumps(
                        {
                            "split": split,
                            "rating_metrics": {split: {"mae": 0.5, "rmse": 0.8}},
                            "explanation_metrics": {
                                "sample_count": 10,
                                "explanation": {"meteor": 1.9},
                                "paper_metrics": {
                                    "bleu": {"4": 0.01},
                                    "rouge": {"rouge_l_f": 2.2},
                                    "distinct_corpus": {"scale_percent_0_100": {"2": 2.4}},
                                },
                                "collapse_stats": {"top1_pred_ratio": 0.6, "pred_unique_ratio": 0.1},
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            cache_root = Path(td) / "cache"
            summary = compact_post_train_eval_layout(root, cache_root=cache_root)
            self.assertTrue(Path(summary["metrics_log"]).is_file())
            self.assertTrue((root / "layout.log").is_file())
            self.assertFalse((root / "metrics_only.json").exists())
            for split in ("valid", "test"):
                split_dir = root / split
                self.assertTrue((split_dir / "metrics.log").is_file())
                self.assertFalse((split_dir / "evidence.tar.gz").exists())
                self.assertTrue((split_dir / "evidence").is_dir())
                self.assertTrue((split_dir / "evidence" / "metrics" / "paper_metrics.json").is_file())
                self.assertTrue((split_dir / "evidence" / "metrics" / "eval_metrics.json").is_file())
                self.assertTrue((split_dir / "evidence" / "predictions" / "predictions.jsonl").is_file())
                self.assertTrue((split_dir / "evidence" / "meta" / "run_summary.json").is_file())
                self.assertFalse((split_dir / "paper_metrics.json").exists())
                compacted_metrics = (split_dir / "metrics.log").read_text(encoding="utf-8")
                self.assertIn("[Recommendation]", compacted_metrics)
                self.assertIn("[Explanation]", compacted_metrics)
                moved_cache = cache_root / "step5" / "task2" / "1_2" / "post_train_eval" / split / f"cache_reuse_decision_{split}.json"
                self.assertTrue(moved_cache.is_file())
            compacted_again = compact_post_train_eval_layout(root, cache_root=cache_root)
            self.assertTrue(Path(compacted_again["metrics_log"]).read_text(encoding="utf-8").startswith("20"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
