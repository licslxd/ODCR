from __future__ import annotations

import unittest
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.index_contract import build_step4_export_lineage, validate_step4_export_lineage
from odcr_core.csb_contract import default_csb_contract_payload
from odcr_core.training_checkpoint import CheckpointLineageError


class Step4Step5FrozenLineageTest(unittest.TestCase):
    def test_step5_rejects_missing_frozen_lineage(self) -> None:
        lineage = build_step4_export_lineage(
            task_id=2,
            auxiliary_domain="A",
            target_domain="T",
            step3_checkpoint_lineage_hash="abc",
            step4_rcr_config={"x": 1},
            csb_contract=default_csb_contract_payload(),
        )
        lineage.pop("frozen_step3_lineage")
        contract = {"step4_export_lineage": lineage}
        with self.assertRaises(CheckpointLineageError):
            validate_step4_export_lineage(
                contract,
                current_step4_rcr_config={"x": 1},
                task_id=2,
                auxiliary_domain="A",
                target_domain="T",
            )

    def test_frozen_lineage_passes_without_current_latest_override(self) -> None:
        lineage = build_step4_export_lineage(
            task_id=2,
            auxiliary_domain="A",
            target_domain="T",
            step3_checkpoint_lineage_hash="abc",
            step4_rcr_config={"x": 1},
            frozen_step3_lineage={
                "upstream_step3_run_id": "2",
                "step3_checkpoint_path": "runs/step3/task2/2/model/best.pth",
                "step3_checkpoint_hash": "h",
                "step3_stage_status_hash": "s",
                "step3_readiness_audit_hash": "e",
            },
            csb_contract=default_csb_contract_payload(),
        )
        out = validate_step4_export_lineage(
            {"step4_export_lineage": lineage},
            current_step4_rcr_config={"x": 1},
            task_id=2,
            auxiliary_domain="A",
            target_domain="T",
        )
        self.assertEqual(out["frozen_step3_lineage"]["upstream_step3_run_id"], "2")


if __name__ == "__main__":
    unittest.main()
