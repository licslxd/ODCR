from __future__ import annotations

import json
from pathlib import Path

from odcr_eval_metrics import (
    CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
    code1_compatible_rating_metrics,
    compare_code1_rating_prediction_rows,
)
from odcr_core.step5_rating_handoff import finalize_step5A_rating_eval_handoff


def test_code1_compatible_rating_metric_parity_no_clip_no_preround() -> None:
    pred = [1.0, 2.5, 4.0, 5.2]
    ref = [1.5, 2.0, 3.0, 5.0]

    metrics = code1_compatible_rating_metrics(pred, ref)

    assert metrics["metric_protocol"] == CODE1_COMPATIBLE_RATING_PROTOCOL_ID
    assert metrics["prediction_clipping"] is False
    assert metrics["prediction_rounding_before_metric"] is False
    assert metrics["rating_normalization"] is False
    assert metrics["rating_denormalization"] is False
    assert metrics["mae"] == 0.55
    assert metrics["rmse"] == 0.6205
    assert metrics["sample_count"] == 4


def test_rating_prediction_row_batch_invariance_comparator() -> None:
    rows = [
        {"sample_id": 0, "pred_rating": 1.0, "gt_rating": 1.5},
        {"sample_id": 1, "pred_rating": 2.5, "gt_rating": 2.0},
        {"sample_id": 2, "pred_rating": 4.0, "gt_rating": 3.0},
    ]
    reordered = [rows[2], rows[0], rows[1]]

    result = compare_code1_rating_prediction_rows(rows, reordered, tolerance=1e-5)

    assert result["sample_count_equal"] is True
    assert result["sample_id_set_equal"] is True
    assert result["max_abs_pred_delta"] == 0.0
    assert result["passed"] is True


def test_rating_prediction_row_batch_invariance_rejects_drift() -> None:
    rows = [
        {"sample_id": 0, "pred_rating": 1.0, "gt_rating": 1.5},
        {"sample_id": 1, "pred_rating": 2.5, "gt_rating": 2.0},
    ]
    drifted = [
        {"sample_id": 0, "pred_rating": 1.0, "gt_rating": 1.5},
        {"sample_id": 1, "pred_rating": 2.5002, "gt_rating": 2.0},
    ]

    result = compare_code1_rating_prediction_rows(rows, drifted, tolerance=1e-5)

    assert result["passed"] is False
    assert result["max_abs_pred_delta"] > 1e-5


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_prediction_artifacts(eval_dir: Path, split: str) -> dict[str, str]:
    csv_path = eval_dir / f"rating_{split}_predictions.csv"
    jsonl_path = eval_dir / f"rating_{split}_predictions.jsonl"
    rows = [
        "sample_id,split,row_index,user_idx,item_idx,gt_rating,pred_rating,abs_error,squared_error\n",
        f"0,{split},0,1,2,4.0,2.0,2.0,4.0\n",
        f"1,{split},1,3,4,2.0,3.0,1.0,1.0\n",
    ]
    csv_path.write_text("".join(rows), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": 0, "split": split, "gt_rating": 4.0, "pred_rating": 2.0}),
                json.dumps({"sample_id": 1, "split": split, "gt_rating": 2.0, "pred_rating": 3.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        eval_dir / f"rating_{split}_digest.json",
        {
            "schema_version": "odcr_step5_rating_only_digest/1",
            "split": split,
            "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
            "recommendation": {"mae": 1.5, "rmse": 1.5811},
            "sample_count": 2,
            "no_generate": True,
        },
    )
    return {"csv": str(csv_path), "jsonl": str(jsonl_path)}


def test_step5a_rating_handoff_dry_run_ignores_failed_batch_diagnostic(tmp_path: Path) -> None:
    repo = tmp_path
    run_root = repo / "runs" / "step5" / "task2" / "1_2_step5A"
    eval_dir = run_root / "eval"
    eval_dir.mkdir(parents=True)
    checkpoint = run_root / "model" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    split_dir = repo / "data" / "AM_CDs"
    split_dir.mkdir(parents=True)
    for split in ("valid", "test"):
        (split_dir / f"{split}.csv").write_text(
            "user,item,rating,user_idx,item_idx\nu1,i1,4.0,1,2\nu2,i2,2.0,3,4\n",
            encoding="utf-8",
        )
        artifacts = _write_prediction_artifacts(eval_dir, split)
        _write_json(
            eval_dir / f"rating_{split}_metrics.json",
            {
                "metrics_schema_version": "odcr_step5_rating_only_eval_output/1",
                "rating_only": True,
                "no_generate": True,
                "target_only": True,
                "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
                "generation_touched": False,
                "step5B_touched": False,
                "rerank_touched": False,
                "checkpoint": str(checkpoint),
                "metrics_path": str(eval_dir / f"rating_{split}_metrics.json"),
                "sample_count": 2,
                "rating_metrics": {
                    "mae": 1.5,
                    "rmse": 1.5811,
                    "sample_count": 2,
                    "metric_protocol": CODE1_COMPATIBLE_RATING_PROTOCOL_ID,
                },
                "prediction_artifacts": artifacts,
                "batch_invariance": {"passed": False, "max_abs_pred_delta": 0.015625},
            },
        )

    result = finalize_step5A_rating_eval_handoff(
        repo_root=repo,
        task=2,
        source_run_id="1_2_step5A",
        checkpoint=checkpoint,
        valid_metrics_path=eval_dir / "rating_valid_metrics.json",
        test_metrics_path=eval_dir / "rating_test_metrics.json",
        valid_file=split_dir / "valid.csv",
        test_file=split_dir / "test.csv",
        expected_valid_count=2,
        expected_test_count=2,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["batch_invariance_required"] is False
    assert result["batch_invariance_gate_removed"] is True
