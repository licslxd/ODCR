from __future__ import annotations

import json
import hashlib
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
    refresh_index_contract_train_csv_fingerprint,
)
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST, validate_step4_export_ready


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ready_row(**overrides: object) -> dict[str, object]:
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
    row.update(overrides)
    return row


def _write_ready_fixture(run: Path) -> None:
    pd.DataFrame([_ready_row()]).to_csv(run / ODCR_ROUTING_TRAIN_CSV, index=False)
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
    contract = refresh_index_contract_train_csv_fingerprint(
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
        },
        str(run / ODCR_ROUTING_TRAIN_CSV),
    )
    (run / INDEX_CONTRACT_FILENAME).write_text(
        json.dumps(contract),
        encoding="utf-8",
    )
    (run / STEP4_EXPORT_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": "odcr_step4_train_table/1.2",
                "row_counts": {"total_rows": 1, "by_sample_origin": {"aux_cf": 1}},
                "step4_export_lineage": lineage,
            }
        ),
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

    def test_partial_cf_rows_do_not_need_to_equal_final_total(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            run.mkdir(parents=True, exist_ok=True)
            rows = [
                _ready_row(sample_origin="target_gold", train_keep=1),
                _ready_row(sample_origin="aux_gold", train_keep=1),
                _ready_row(sample_origin="aux_cf", train_keep=1),
                _ready_row(sample_origin="aux_cf", train_keep=1),
            ]
            pd.DataFrame(rows).to_csv(run / ODCR_ROUTING_TRAIN_CSV, index=False)
            partial_dir = run / "step4_partials"
            partial_dir.mkdir()
            partials = []
            for rank, idx in enumerate(([0], [1])):
                parquet = partial_dir / f"step4_partial_task2_rank{rank}.parquet"
                pd.DataFrame(
                    {
                        "row_idx": idx,
                        "entropy": [0.0],
                        "explanation": ["ok"],
                        "rating_target": [5.0],
                        "rating_counterfactual": [5.0],
                        "rating_delta": [0.0],
                        "rating_stability_score": [1.0],
                        "shared_latent_similarity": [1.0],
                        "specific_latent_shift": [0.5],
                    }
                ).to_parquet(parquet, index=False)
                sha = _sha256(parquet)
                manifest = {
                    "schema_version": "odcr_step4_partial_artifact/1",
                    "status": "ok",
                    "rank": rank,
                    "world_size": 2,
                    "shard_id": rank,
                    "path": str(parquet),
                    "row_count": 1,
                    "format": "parquet",
                    "sha256": sha,
                }
                partials.append(manifest)
                (parquet.with_suffix(parquet.suffix + ".manifest.json")).write_text(json.dumps(manifest), encoding="utf-8")
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
            contract = refresh_index_contract_train_csv_fingerprint(
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
                },
                str(run / ODCR_ROUTING_TRAIN_CSV),
            )
            (run / INDEX_CONTRACT_FILENAME).write_text(json.dumps(contract), encoding="utf-8")
            (run / STEP4_EXPORT_MANIFEST).write_text(
                json.dumps(
                    {
                        "schema_version": "odcr_step4_train_table/1.2",
                        "row_counts": {
                            "total_rows": 4,
                            "by_sample_origin": {"target_gold": 1, "aux_gold": 1, "aux_cf": 2},
                        },
                        "rcr_routing": {"n_target_rows_for_cf": 2},
                        "partial_artifacts": partials,
                        "step4_export_lineage": lineage,
                    }
                ),
                encoding="utf-8",
            )
            result = validate_step4_export_ready(run, repo_root=run)
            self.assertTrue(result.ready, result.errors)
            self.assertEqual(result.row_count, 4)
            self.assertEqual(result.diagnostics["partial_artifacts"]["row_count"], 2)
            self.assertEqual(result.diagnostics["partial_artifacts"]["target_cf_rows"], 2)

    def test_stale_train_csv_contract_fails_until_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run = Path(td)
            _write_ready_fixture(run)
            contract_path = run / INDEX_CONTRACT_FILENAME
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["fingerprints"]["train_csv"]["exists"] = False
            contract["fingerprints"]["train_csv"]["is_file"] = False
            contract["fingerprints"]["train_csv"]["sha256"] = None
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            stale = validate_step4_export_ready(run, repo_root=run)
            self.assertFalse(stale.ready)
            self.assertIn("contract_stale", stale.errors[0])
            refreshed = refresh_index_contract_train_csv_fingerprint(contract, str(run / ODCR_ROUTING_TRAIN_CSV))
            contract_path.write_text(json.dumps(refreshed), encoding="utf-8")
            ready = validate_step4_export_ready(run, repo_root=run)
            self.assertTrue(ready.ready, ready.errors)


if __name__ == "__main__":
    unittest.main()
