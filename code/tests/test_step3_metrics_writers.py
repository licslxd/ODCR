"""Step3 structured metric writer schema tests."""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core import path_layout  # noqa: E402
from train_logging import (  # noqa: E402
    append_step3_epoch_summary_csv,
    append_step3_gpu_profile_jsonl,
    append_step3_loss_breakdown_jsonl,
    append_step3_timing_profile_jsonl,
    append_train_epoch_metrics_jsonl,
)


class TestStep3MetricsWriters(unittest.TestCase):
    def test_jsonl_and_csv_writers_create_expected_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_file = str(Path(tmp) / "meta" / "full.log")
            Path(log_file).parent.mkdir(parents=True)
            Path(log_file).write_text("", encoding="utf-8")
            metric_row = {
                "run_id": "dry_run",
                "task_id": 2,
                "profile_id": "task2_strong_forward_g1s",
                "epoch": 1,
                "global_step": 50,
                "rank": 0,
                "split": "train",
                "loss_total": 1.0,
                "lr": 7e-4,
                "finite": True,
                "timestamp": "2026-05-06T00:00:00Z",
            }
            append_train_epoch_metrics_jsonl(log_file=log_file, row=metric_row)
            append_step3_loss_breakdown_jsonl(
                log_file=log_file,
                row={**metric_row, "loss_name": "L_rating_shared", "raw_value": 1.0, "weight": 1.0, "weighted_value": 1.0},
            )
            append_step3_timing_profile_jsonl(
                log_file=log_file,
                row={**metric_row, "data_time": 0.1, "forward_time": 0.2, "backward_time": 0.3, "optimizer_time": 0.4, "step_time": 1.0, "samples_per_sec": 1536.0},
            )
            append_step3_gpu_profile_jsonl(
                log_file=log_file,
                row={**metric_row, "device": "cuda:0", "allocated_gib": 1.0, "reserved_gib": 2.0, "max_allocated_gib": 3.0, "max_reserved_gib": 4.0},
            )
            append_step3_epoch_summary_csv(
                log_file=log_file,
                row={
                    "epoch": 1,
                    "train_loss": 1.0,
                    "valid_loss": 0.9,
                    "best_metric": 0.9,
                    "elapsed_s": 12.3,
                    "samples_per_sec": 1536.0,
                    "checkpoint_path": "runs/task2/v1/step3/1/model/best.pth",
                    "status": "best",
                },
            )
            meta = Path(log_file).parent
            for kind in ("metrics", "loss_breakdown", "timing_profile", "gpu_profile"):
                path = meta / path_layout.metrics_filename(kind)
                self.assertTrue(path.is_file())
                row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(row["run_id"], "dry_run")
                self.assertEqual(row["task_id"], 2)
            with (meta / path_layout.metrics_filename("epoch_summary")).open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["best_metric"], "0.9")
            self.assertEqual(rows[0]["status"], "best")


if __name__ == "__main__":
    unittest.main()
