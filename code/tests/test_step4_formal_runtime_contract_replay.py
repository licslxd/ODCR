from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import odcr  # noqa: E402
from odcr_core import step4_runtime  # noqa: E402


class Step4FormalRuntimeContractReplayTest(unittest.TestCase):
    def test_dry_run_uses_formal_runtime_contract_replay(self) -> None:
        src = inspect.getsource(odcr._run_resolved)
        self.assertIn("validate_step4_formal_runtime_contract_replay", src)
        self.assertIn("step4_formal_runtime_contract_replay", src)
        self.assertIn("step4_prelaunch_lineage_validation", src)

    def test_replay_reports_formal_runtime_contract_fields(self) -> None:
        src = inspect.getsource(step4_runtime.validate_step4_formal_runtime_contract_replay)
        for term in (
            "planned_run_id",
            "planned_run_dir",
            "planned_full_log",
            "selected_checkpoint",
            "lineage_hash",
            "checkpoint_sha256",
            "model_architecture_config_hash",
            "effective_model_ntoken",
            "required_fields_status",
            "run_id_overwrite_status",
            "will_write_formal_on_actual_run",
            "dry_run_no_formal_write",
        ):
            self.assertIn(term, src)

    def test_dry_run_artifacts_are_written_under_ai_analysis(self) -> None:
        src = inspect.getsource(step4_runtime.step4_formal_dry_run_meta_dir)
        self.assertIn("AI_analysis", src)
        self.assertIn("01_raw_logs", src)
        self.assertIn("step4_formal_runtime_contract", src)
        replay_src = inspect.getsource(step4_runtime.validate_step4_formal_runtime_contract_replay)
        self.assertIn("formal_dry_run_replay_report.json", replay_src)
        self.assertIn("formal_namespace_polluted", replay_src)


if __name__ == "__main__":
    unittest.main()
