from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.index_contract import (  # noqa: E402
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    build_step4_export_lineage,
    refresh_index_contract_train_csv_fingerprint,
)
from odcr_core.step4_dedicated_exports import (  # noqa: E402
    FULL_AUDIT_PARQUET,
    ROUTE_INTERSECTION_REPORT,
    STEP4_DEDICATED_EXPORTS_STATUS,
    STEP5_TRAIN_MANIFEST,
    STEP5A_SCORER_TRAIN_PARQUET,
    STEP5B_EXPLAINER_TRAIN_PARQUET,
    Step4DedicatedExportError,
    export_step4_dedicated_exports,
    validate_step4_dedicated_exports,
)
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _config() -> dict[str, object]:
    return {
        "enabled": True,
        "output_dir_name": "step5_exports",
        "full_audit_format": "parquet",
        "scorer_train_format": "parquet",
        "explainer_train_format": "parquet",
        "scorer_filter": {"train_keep": True, "route_scorer": True, "min_sample_weight_hint": 0.0},
        "explainer_filter": {"train_keep": True, "route_explainer": True, "min_sample_weight_hint": 0.0},
        "write_gold_cf_subsplits": True,
        "full_audit_role": "audit_only",
        "atomic_write": True,
        "validate_after_write": True,
        "chunk_rows": 2,
    }


def _row(origin: str, *, scorer: int, explainer: int, keep: int, weight: float, bucket: int) -> dict[str, object]:
    return {
        "user": f"user_{origin}_{scorer}_{explainer}",
        "item": f"item_{origin}",
        "rating": 5.0,
        "review": "source review",
        "explanation": "clean explanation",
        "content_evidence": "content evidence",
        "content_anchor_score": 0.7,
        "polarity_anchor": "neutral",
        "domain_style_anchor": "domain:style",
        "local_style_residual_hint": "local hint",
        "style_evidence": "style evidence",
        "style_anchor_score": 0.6,
        "evidence_quality_prior": 0.5,
        "preprocess_route_scorer_prior": 0,
        "preprocess_route_explainer_prior": 0,
        "domain": "target" if origin == "target_gold" else "auxiliary",
        "sample_id": 0,
        "entropy": 0.0,
        "rating_target": 5.0,
        "rating_counterfactual": 5.0,
        "rating_delta": 0.0,
        "rating_stability_score": 0.95,
        "shared_latent_similarity": 0.9,
        "specific_latent_shift": 0.4,
        "content_retention_score": 0.92,
        "style_shift_score": 0.3,
        "cf_reliability_score": 0.88,
        "uncertainty_score": 0.05,
        "entropy_score": 0.0,
        "text_quality_score": 1.0,
        "confidence_bucket": bucket,
        "route_scorer": scorer,
        "route_explainer": explainer,
        "route_reason_scorer": "factual_gold_scorer" if origin != "aux_cf" else "rcr_scorer_clean",
        "route_reason_explainer": "factual_gold_explainer" if origin != "aux_cf" else "rcr_explainer_rich",
        "train_keep": keep,
        "sample_weight_hint": weight,
        "sample_origin": origin,
        "is_counterfactual": 1 if origin == "aux_cf" else 0,
        "clean_text": "clean explanation",
        "clean_changed": 0,
        "html_entity_hit": 0,
        "bad_tail_hit": 0,
        "bad_tail_types": "",
        "template_hit": 0,
        "template_count": 1,
        "template_hard_drop_hit": 0,
        "template_downweighted": 0,
        "noisy_tail_downweighted": 0,
        "short_fragment_hit": 0,
        "repeat_tail_hit": 0,
        "train_drop_reason": "" if keep else "rcr_route_reject",
        "user_idx_global": 1,
        "item_idx_global": 2,
    }


def _write_run(root: Path, *, drop_column: str | None = None) -> Path:
    run = root / "runs" / "step4" / "task2" / "1"
    (run / "meta").mkdir(parents=True)
    rows = [
        _row("target_gold", scorer=1, explainer=1, keep=1, weight=1.0, bucket=2),
        _row("aux_gold", scorer=1, explainer=1, keep=1, weight=0.9, bucket=2),
        _row("aux_cf", scorer=1, explainer=0, keep=1, weight=0.5, bucket=1),
        _row("aux_cf", scorer=0, explainer=1, keep=1, weight=0.4, bucket=0),
        _row("aux_cf", scorer=0, explainer=0, keep=0, weight=0.0, bucket=0),
    ]
    df = pd.DataFrame(rows)
    if drop_column:
        df = df.drop(columns=[drop_column])
    df.to_csv(run / ODCR_ROUTING_TRAIN_CSV, index=False)
    lineage = build_step4_export_lineage(
        task_id=2,
        auxiliary_domain="A",
        target_domain="T",
        step3_checkpoint_lineage_hash="lineage",
        step4_rcr_config={"fixture": True},
        step4_run="1",
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
                "row_counts": {"total_rows": len(df), "by_sample_origin": {"target_gold": 1, "aux_gold": 1, "aux_cf": 3}},
                "step4_export_lineage": lineage,
            }
        ),
        encoding="utf-8",
    )
    (run / "meta" / "stage_status.json").write_text(
        json.dumps(
            {
                "schema_version": "odcr_stage_status/1",
                "validator_version": "odcr_stage_status_validator/2",
                "stage": "step4",
                "task": 2,
                "task_id": 2,
                "run_id": "1",
                "run_dir": "runs/step4/task2/1",
                "selected_export": "runs/step4/task2/1/odcr_routing_train.csv",
                "export_manifest": "runs/step4/task2/1/step4_train_table_manifest.json",
                "index_contract": "runs/step4/task2/1/index_contract.json",
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    return run


class Step4DedicatedExportsTest(unittest.TestCase):
    def test_step4_core_export_not_concat_only_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            result = export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            manifest = json.loads((run / "step5_exports" / STEP5_TRAIN_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["exports"]["full_audit"]["role"], "audit_only")
            self.assertTrue(manifest["do_not_use_full_audit_as_default_step5_train"])
            self.assertEqual(result["exports"]["full_audit"]["row_count"], 5)

    def test_step4_route_intersection_report_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            report = json.loads((run / "step5_exports" / ROUTE_INTERSECTION_REPORT).read_text(encoding="utf-8"))
            self.assertEqual(report["intersections"]["train_keep_and_route_scorer"], 3)
            self.assertEqual(report["intersections"]["train_keep_and_route_explainer"], 3)
            self.assertEqual(report["intersections"]["train_keep_and_route_scorer_and_aux_cf"], 1)
            self.assertEqual(report["intersections"]["train_keep_and_route_explainer_and_aux_cf"], 1)

    def test_step4_exports_step5A_scorer_train_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            df = pd.read_parquet(run / "step5_exports" / STEP5A_SCORER_TRAIN_PARQUET)
            self.assertEqual(len(df), 3)
            self.assertLess(len(df.columns), 54)
            self.assertTrue(((df["train_keep"] == 1) & (df["route_scorer"] == 1) & (df["sample_weight_hint"] > 0)).all())

    def test_step4_exports_step5B_explainer_train_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            df = pd.read_parquet(run / "step5_exports" / STEP5B_EXPLAINER_TRAIN_PARQUET)
            self.assertEqual(len(df), 3)
            self.assertLess(len(df.columns), 54)
            self.assertTrue(((df["train_keep"] == 1) & (df["route_explainer"] == 1) & (df["sample_weight_hint"] > 0)).all())

    def test_step4_dedicated_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            manifest = json.loads((run / "step5_exports" / STEP5_TRAIN_MANIFEST).read_text(encoding="utf-8"))
            for item in manifest["exports"].values():
                path = root / item["path"]
                self.assertEqual(_sha256(path), item["sha256"])

    def test_step4_dedicated_exports_status_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            before = _sha256(run / "meta" / "stage_status.json")
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            sidecar = json.loads((run / "meta" / STEP4_DEDICATED_EXPORTS_STATUS).read_text(encoding="utf-8"))
            self.assertTrue(sidecar["step5_dedicated_exports_ready"])
            self.assertEqual(sidecar["previous_stage_status_sha256"], before)

    def test_step4_readiness_distinguishes_full_audit_from_step5_train(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            status = json.loads((run / "meta" / "stage_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["full_audit_table_role"], "audit_only")
            self.assertEqual(status["step5_train_input_role"], "dedicated_split_exports")
            self.assertTrue(status["step5A_scorer_train_export"].endswith(STEP5A_SCORER_TRAIN_PARQUET))

    def test_step4_full_audit_is_not_default_train_role(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            manifest = json.loads((run / "step5_exports" / STEP5_TRAIN_MANIFEST).read_text(encoding="utf-8"))
            self.assertNotEqual(manifest["exports"]["full_audit"]["role"], "step5A_train")
            self.assertNotEqual(manifest["exports"]["full_audit"]["role"], "step5B_train")

    def test_step4_route_intersections_by_origin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            report = json.loads((run / "step5_exports" / ROUTE_INTERSECTION_REPORT).read_text(encoding="utf-8"))
            self.assertEqual(report["origin_breakdown"]["step5A_scorer_train"]["target_gold"], 1)
            self.assertEqual(report["origin_breakdown"]["step5A_scorer_train"]["aux_gold"], 1)
            self.assertEqual(report["origin_breakdown"]["step5A_scorer_train"]["aux_cf"], 1)
            self.assertEqual(report["origin_breakdown"]["step5B_explainer_train"]["aux_cf"], 1)

    def test_step4_dedicated_exports_atomic_write(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            leftovers = list((run / "step5_exports").glob("*.tmp*"))
            self.assertEqual(leftovers, [])

    def test_step4_dedicated_exports_no_step5_namespace_pollution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            self.assertFalse((root / "runs" / "step5").exists())

    def test_step4_dedicated_exports_missing_fields_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_run(root, drop_column="clean_text")
            with self.assertRaises(Step4DedicatedExportError):
                export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())

    def test_step4_dedicated_validator_accepts_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = _write_run(root)
            export_step4_dedicated_exports(repo_root=root, task=2, from_run="1", config=_config())
            result = validate_step4_dedicated_exports(run, repo_root=root)
            self.assertTrue(result.ready, result.errors)
            self.assertEqual(result.diagnostics["exports"]["full_audit"]["path"], f"runs/step4/task2/1/step5_exports/{FULL_AUDIT_PARQUET}")


if __name__ == "__main__":
    unittest.main()

