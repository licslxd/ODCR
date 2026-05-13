from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core import step4_runtime


class Step4BoundedPreflightRealFieldsTest(unittest.TestCase):
    def test_preflight_writes_required_preview_files_and_real_data_flags(self) -> None:
        src = inspect.getsource(step4_runtime.run_step4_bounded_preflight)
        for name in (
            "rcr_distribution.json",
            "required_fields_check.json",
            "manifest_preview.json",
            "index_contract_preview.json",
            "lineage_preview.json",
            "cpu_gpu_utilization_snapshot.json",
            "preflight_summary.json",
        ):
            self.assertIn(name, src)
        self.assertIn("uses_real_task_data", src)
        self.assertIn("uses_selected_step3_checkpoint", src)
        self.assertIn("sample_weight_hint", src)
        self.assertIn("confidence_bucket_distribution", src)
        self.assertIn("gpu-shard", src)
        self.assertIn("force_gpu_forward", src)
        self.assertIn("candidate_config", src)

    def test_gpu_shard_runner_writes_validation_only_proof_artifacts(self) -> None:
        runner_src = (CODE_DIR / "odcr_core" / "step4_gpu_preflight_runner.py").read_text(encoding="utf-8")
        for name in (
            "gpu_shard_path_proof.json",
            "per_rank_summary.json",
            "timing_breakdown.json",
            "cpu_export_after_pg_destroy",
            "actual_gpu_forward_executed",
            "actual_model_loaded_on_gpu",
            "two_gpus_used",
        ):
            self.assertIn(name, runner_src)
        self.assertNotIn("write_latest_pointer_json", runner_src)


if __name__ == "__main__":
    unittest.main()
