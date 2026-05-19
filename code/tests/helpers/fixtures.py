from __future__ import annotations

import json
from pathlib import Path

from odcr_core.index_contract import (
    INDEX_CONTRACT_FILENAME,
    INDEX_CONTRACT_SCHEMA_VERSION,
    ODCR_ROUTING_TRAIN_CSV,
    STEP4_RCR_REQUIRED_COLUMNS,
    build_step4_export_lineage,
    refresh_index_contract_train_csv_fingerprint,
)
from odcr_core.manifests import write_latest_pointer_json
from odcr_core.stage_status import build_and_write_stage_status
from odcr_core.step4_export_validator import STEP4_EXPORT_MANIFEST


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_step4_upstream_fixture(repo: Path, *, task_id: int = 4, run_id: str = "1") -> Path:
    """Write the smallest real Step4-ready contract for Step5 admission tests."""

    run = repo / "runs" / "step4" / f"task{task_id}" / run_id
    meta = run / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    export = run / ODCR_ROUTING_TRAIN_CSV
    row = {col: 1 for col in STEP4_RCR_REQUIRED_COLUMNS}
    row.update(
        {
            "route_reason_scorer": "rcr_scorer_clean",
            "route_reason_explainer": "rcr_explainer_rich",
            "confidence_bucket": 2,
            "preprocess_route_scorer_prior": 0,
            "preprocess_route_explainer_prior": 0,
        }
    )
    headers = list(STEP4_RCR_REQUIRED_COLUMNS)
    export.write_text(
        ",".join(headers) + "\n" + ",".join(str(row[col]) for col in headers) + "\n",
        encoding="utf-8",
    )
    lineage = build_step4_export_lineage(
        task_id=task_id,
        auxiliary_domain="A",
        target_domain="T",
        step3_checkpoint_lineage_hash="lineage",
        step4_rcr_config={"fixture": True},
        step4_run=run_id,
        frozen_step3_lineage={
            "upstream_step3_run_id": "2",
            "step3_checkpoint_path": f"runs/step3/task{task_id}/2/model/best_observed.pth",
            "step3_checkpoint_hash": "fixture_checkpoint_hash",
            "step3_stage_status_hash": "fixture_stage_status_hash",
            "step3_eval_handoff_hash": "fixture_eval_handoff_hash",
        },
    )
    contract = refresh_index_contract_train_csv_fingerprint(
        {
            "schema_version": INDEX_CONTRACT_SCHEMA_VERSION,
            "embed_dim": 1024,
            "backbones": {
                "sentence_embed": {
                    "model_id": "fixture",
                    "local_dir": "/tmp/fixture",
                    "family": "bge_large_en",
                    "hidden_size": 1024,
                    "dual_channel": True,
                }
            },
            "step4_export_lineage": lineage,
        },
        str(export),
    )
    write_json(run / INDEX_CONTRACT_FILENAME, contract)
    write_json(
        run / STEP4_EXPORT_MANIFEST,
        {
            "schema_version": "odcr_step4_train_table/1.2",
            "row_counts": {"total_rows": 1, "by_sample_origin": {"aux_cf": 1}},
            "step4_export_lineage": lineage,
        },
    )
    write_json(meta / "source_table.json", {"records": []})
    write_json(meta / "resolved_config.json", {"task": {"id": task_id}})
    write_json(
        meta / "run_summary.json",
        {
            "run_id": run_id,
            "stage": "step4",
            "task_id": task_id,
            "status": "ok",
            "run_dir": f"runs/step4/task{task_id}/{run_id}",
            "meta_dir": f"runs/step4/task{task_id}/{run_id}/meta",
        },
    )
    build_and_write_stage_status(repo_root=repo, stage="step4", task=task_id, run_id=run_id)
    write_latest_pointer_json(
        repo_root=repo,
        stage_unit_dir=repo / "runs" / "step4" / f"task{task_id}",
        run_id=run_id,
        run_dir=run,
        summary_path=meta / "run_summary.json",
        status="ok",
    )
    return run
