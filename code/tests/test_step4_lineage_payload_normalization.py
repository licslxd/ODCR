from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step4_checkpoint_lineage import (  # noqa: E402
    normalize_step3_lineage_for_step4,
    validate_step4_formal_lineage_contract,
)


UPSTREAM = {
    "stage_status_validation": {
        "status_path": "runs/step3/task2/2/meta/stage_status.json",
        "eval_handoff": "runs/step3/task2/2/meta/eval_handoff.json",
        "run_summary": "runs/step3/task2/2/meta/run_summary.json",
        "source_table": "runs/step3/task2/2/meta/source_table.json",
        "resolved_config": "runs/step3/task2/2/meta/resolved_config.json",
        "selected_checkpoint": "runs/step3/task2/2/model/best_observed.pth",
        "selected_checkpoint_hash": "checkpoint-sha",
    }
}


def _base_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "checkpoint_lineage_hash": "lineage-from-validator",
        "checkpoint_path": "runs/step3/task2/2/model/best_observed.pth",
        "checkpoint_file_hash": "checkpoint-sha",
        "model_architecture_config_hash": "arch-hash",
        "effective_model_ntoken": 32128,
        "sidecar_ntoken": 32100,
        "checkpoint_tensor_ntoken": 32128,
        "checkpoint_source": "stage_status.selected_checkpoint",
        "checkpoint_binding": {
            "checkpoint_source": "stage_status.selected_checkpoint",
            "selected_checkpoint_path": "runs/step3/task2/2/model/best_observed.pth",
            "selected_checkpoint_hash": "checkpoint-sha",
            "best_pth_alias": {"alias_consistent": True, "used_as_primary": False},
        },
        "source_table_hash_scope_report": {"schema_version": "scope/1"},
        "live_vs_frozen_step3_config_drift": {"policy": "frozen Step3 policy"},
    }


class Step4LineagePayloadNormalizationTest(unittest.TestCase):
    def test_normalize_returns_lineage_hash(self) -> None:
        payload = validate_step4_formal_lineage_contract(
            normalize_step3_lineage_for_step4(
                _base_payload(),
                upstream_resolution=UPSTREAM,
                checkpoint_lineage_path="runs/step3/task2/2/model/best_observed.pth.lineage.json",
            )
        )
        self.assertEqual(payload["lineage_hash"], "lineage-from-validator")
        self.assertEqual(payload["lineage_hash_source"], "validated.checkpoint_lineage_hash")
        self.assertEqual(payload["checkpoint_sha256"], "checkpoint-sha")
        self.assertTrue(payload["best_pth_alias_consistent"])

    def test_missing_lineage_hash_is_derived_with_source(self) -> None:
        raw = _base_payload()
        raw.pop("checkpoint_lineage_hash")
        payload = validate_step4_formal_lineage_contract(
            normalize_step3_lineage_for_step4(
                raw,
                upstream_resolution=UPSTREAM,
                checkpoint_lineage_path="runs/step3/task2/2/model/best_observed.pth.lineage.json",
            )
        )
        self.assertTrue(payload["lineage_hash"])
        self.assertEqual(payload["lineage_hash_source"], "derived_from_normalized_step3_lineage_evidence")

    def test_missing_nonderivable_required_field_fails(self) -> None:
        raw = _base_payload()
        raw.pop("effective_model_ntoken")
        payload = normalize_step3_lineage_for_step4(
            raw,
            upstream_resolution=UPSTREAM,
            checkpoint_lineage_path="runs/step3/task2/2/model/best_observed.pth.lineage.json",
        )
        with self.assertRaisesRegex(Exception, "effective_model_ntoken"):
            validate_step4_formal_lineage_contract(payload)


if __name__ == "__main__":
    unittest.main()
