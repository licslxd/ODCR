from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.preprocess_runtime import write_preprocess_gpu_metrics_verify_artifacts  # noqa: E402
from odcr_core.preprocess_schema import PREPROCESS_C_DOMAIN_CONTRACT_VERSION  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_preprocess_b_metrics_verify_artifacts_are_written(tmp_path: Path) -> None:
    root = tmp_path
    meta = root / "runs" / "preprocess" / "b" / "1" / "meta"
    dataset_dir = root / "data" / "D"
    paths = {
        "user_content_profiles": dataset_dir / "user_content_profiles.npy",
        "user_style_profiles": dataset_dir / "user_style_profiles.npy",
        "item_content_profiles": dataset_dir / "item_content_profiles.npy",
        "item_style_profiles": dataset_dir / "item_style_profiles.npy",
    }
    for index, path in enumerate(paths.values(), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.full((2, 4), index, dtype=np.float32))

    write_preprocess_gpu_metrics_verify_artifacts(
        repo_root=root,
        meta_root=meta,
        stage="preprocess_b",
        unit="b",
        run_id="1",
        stage_metadata={
            "stage_specific": {
                "embed_dim": 4,
                "profile_output_paths": {"D": {key: str(path) for key, path in paths.items()}},
            }
        },
        dataset_statuses={"D": {"status": "ok"}},
        worker_results=[{"worker_id": 1, "exit_code": 0, "handled_units": ["D"]}],
        status="ok",
    )

    metrics = _load(meta / "metrics.json")
    verify = _load(meta / "verify_report.json")
    assert metrics["stage"] == "preprocess_b"
    assert metrics["dataset_status_counts"] == {"ok": 1}
    assert metrics["artifact_count"] == 4
    assert verify["status"] == "pass"
    assert len(verify["artifacts"]) == 4
    assert {item["spec"] for item in verify["artifacts"]} == {
        "user_content",
        "user_style",
        "item_content",
        "item_style",
    }
    assert all(item["shape"] == [2, 4] for item in verify["artifacts"])
    assert all(item["dtype"] == "float32" for item in verify["artifacts"])


def test_preprocess_c_metrics_verify_artifacts_are_written(tmp_path: Path) -> None:
    root = tmp_path
    meta = root / "runs" / "preprocess" / "c" / "1" / "meta"
    dataset_dir = root / "data" / "D"
    paths = {
        "domain_content": dataset_dir / "domain_content.npy",
        "domain_style": dataset_dir / "domain_style.npy",
    }
    for index, path in enumerate(paths.values(), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.full((4,), index, dtype=np.float32))

    write_preprocess_gpu_metrics_verify_artifacts(
        repo_root=root,
        meta_root=meta,
        stage="preprocess_c",
        unit="c",
        run_id="1",
        stage_metadata={
            "stage_specific": {
                "embed_dim": 4,
                "domain_output_paths": {"D": {key: str(path) for key, path in paths.items()}},
            }
        },
        dataset_statuses={"D": {"status": "ok"}},
        worker_results=[{"worker_id": 1, "exit_code": 0, "handled_units": ["D"]}],
        status="ok",
    )

    metrics = _load(meta / "metrics.json")
    verify = _load(meta / "verify_report.json")
    assert metrics["stage"] == "preprocess_c"
    assert metrics["artifact_count"] == 2
    assert verify["status"] == "pass"
    assert {item["domain"] for item in verify["artifacts"]} == {"content", "style"}
    assert all(item["shape"] == [4] for item in verify["artifacts"])
    assert all(
        item["domain_shape_contract_version"] == PREPROCESS_C_DOMAIN_CONTRACT_VERSION
        for item in verify["artifacts"]
    )
