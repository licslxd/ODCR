from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "code", REPO_ROOT / "code" / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402
from odcr_core.evidence_level import mark_gpu_shard_forward, mark_schema_preview  # noqa: E402


class GpuBridgeRuntimeEvidenceLevelTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_required_artifacts(self, root: Path, summary: dict, gpu_snapshot: dict) -> None:
        self._write_json(root / "preflight_summary.json", summary)
        self._write_json(
            root / "rcr_distribution.json",
            {
                "sample_count": 4,
                "route_scorer_count": 2,
                "route_explainer_count": 2,
                "train_keep_count": 2,
                "confidence_bucket_distribution": {"1": 4},
                "sample_weight_hint": {"mean": 1.0},
            },
        )
        self._write_json(root / "required_fields_check.json", {"passed": True, "missing": []})
        self._write_json(root / "manifest_preview.json", {"schema_version": "x"})
        self._write_json(root / "index_contract_preview.json", {"schema_version": "x"})
        self._write_json(root / "lineage_preview.json", {"lineage_hash": "abc"})
        self._write_json(root / "cpu_gpu_utilization_snapshot.json", gpu_snapshot)

    def test_cuda_available_without_gpu_forward_is_not_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_required_artifacts(
                root,
                mark_schema_preview(
                    {
                        "validation_namespace": "x",
                        "sample_count": 4,
                        "max_samples": 8,
                        "formal_latest_write": False,
                        "formal_export_write": False,
                        "upstream_step3_run_id": "2",
                    }
                ),
                mark_schema_preview({"cuda_available": True, "gpu_runtime_evidence": False}),
            )
            evidence = bridge.parse_step4_preflight_evidence(output_dir=root)
        self.assertTrue(evidence["evidence_complete"])
        self.assertFalse(evidence["runtime_evidence_ok"])
        self.assertEqual(evidence["evidence_level"], "E1_schema_preview")

    def test_cuda_probe_only_is_e3_transport_not_step4_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_json(root / "report.json", {"cuda_available": True})
            evidence = bridge.parse_step4_preflight_evidence(output_dir=root)
        self.assertEqual(evidence["evidence_level"], "E3_gpu_transport")
        self.assertTrue(evidence["gpu_transport_ok"])
        self.assertFalse(evidence["runtime_evidence_ok"])

    def test_e4_gpu_forward_artifacts_are_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_required_artifacts(
                root,
                mark_gpu_shard_forward(
                    {
                        "validation_namespace": "x",
                        "sample_count": 4,
                        "max_samples": 8,
                        "formal_latest_write": False,
                        "formal_export_write": False,
                        "upstream_step3_run_id": "2",
                    }
                ),
                mark_gpu_shard_forward({"cuda_available": True, "device_count": 2}),
            )
            evidence = bridge.parse_step4_preflight_evidence(output_dir=root)
        self.assertEqual(evidence["evidence_level"], "E4_gpu_shard_forward_bounded")
        self.assertTrue(evidence["runtime_evidence_ok"])


if __name__ == "__main__":
    unittest.main()
