from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd

from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    build_step4_export_lineage,
)
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST, validate_step4_export_ready


def _write_ready_fixture(run: Path) -> None:
    row = {col: 1 for col in STEP4_RCR_REQUIRED_COLUMNS}
    row.update(
        {
            "route_reason_scorer": "rcr_scorer_clean",
            "route_reason_explainer": "rcr_explainer_rich",
            "content_retention_score": 0.9,
            "style_shift_score": 0.7,
            "rating_stability_score": 0.95,
            "cf_reliability_score": 0.88,
            "uncertainty_score": 0.1,
            "entropy_score": 0.0,
            "text_quality_score": 1.0,
            "confidence_bucket": 2,
            "sample_weight_hint": 1.0,
            "preprocess_route_scorer_prior": 0,
            "preprocess_route_explainer_prior": 0,
        }
    )
    pd.DataFrame([row]).to_csv(run / ODCR_ROUTING_TRAIN_CSV, index=False)
    lineage = build_step4_export_lineage(
        task_id=2,
        auxiliary_domain="A",
        target_domain="T",
        step3_checkpoint_lineage_hash="lineage",
        step4_rcr_config={"x": 1},
        step4_run="2_1",
        frozen_step3_lineage={
            "upstream_step3_run_id": "2",
            "step3_checkpoint_path": "runs/step3/task2/2/model/best.pth",
            "step3_checkpoint_hash": "h",
            "step3_stage_status_hash": "s",
            "step3_eval_handoff_hash": "e",
        },
    )
    (run / INDEX_CONTRACT_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
                "embed_dim": 1024,
                "backbones": {
                    "sentence_embed": {
                        "model_id": "m",
                        "local_dir": "/tmp/m",
                        "family": "bge_large_en",
                        "hidden_size": 1024,
                        "dual_channel": True,
                    }
                },
                "step4_export_lineage": lineage,
            }
        ),
        encoding="utf-8",
    )
    (run / STEP4_EXPORT_MANIFEST).write_text(
        json.dumps({"schema_version": "odcr_step4_train_table/1.2", "step4_export_lineage": lineage}),
        encoding="utf-8",
    )


class Step4ExportValidatorTest(unittest.TestCase):
    def test_ready_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            _write_ready_fixture(run)
            result = validate_step4_export_ready(run, repo_root=run)
            self.assertTrue(result.ready, result.errors)
            self.assertEqual(result.train_keep_count, 1)

    def test_csv_without_manifest_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            _write_ready_fixture(run)
            (run / STEP4_EXPORT_MANIFEST).unlink()
            result = validate_step4_export_ready(run, repo_root=run)
            self.assertFalse(result.ready)
            self.assertIn("export manifest missing", result.errors[0])


if __name__ == "__main__":
    unittest.main()
