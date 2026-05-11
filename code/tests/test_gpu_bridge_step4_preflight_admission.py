from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

import odcr_tmux_gpu_bridge as bridge  # noqa: E402


SAFE_PREFLIGHT = (
    "./odcr",
    "step4",
    "--task",
    "2",
    "--preflight",
    "--max-samples",
    "128",
    "--validation-namespace",
    "step4_preflight_smoke",
)

GPU_SHARD_PREFLIGHT = (
    "./odcr",
    "step4",
    "--task",
    "2",
    "--preflight",
    "--preflight-mode",
    "gpu-shard",
    "--force-gpu-forward",
    "--profile-utilization",
    "--max-samples",
    "16384",
    "--validation-namespace",
    "step4_gpu_c9_bucket_balanced_s16384",
    "--candidate-config",
    "AI_analysis/06_step4_tuning_c9_neighborhood/best_candidate.yaml",
)


class GpuBridgeStep4PreflightAdmissionTest(unittest.TestCase):
    def test_step4_preflight_repo_command_allowed(self) -> None:
        classification = bridge.BridgeCommandPolicy.classify_repo_command(SAFE_PREFLIGHT)
        self.assertTrue(classification.allowed, classification.reason)
        self.assertEqual(classification.stage, "step4")
        self.assertEqual(classification.operation, "preflight")
        self.assertEqual(classification.validation_namespace, "step4_preflight_smoke")
        self.assertEqual(classification.bounded_limit_value, 128)
        bridge.validate_script_safety(" ".join(SAFE_PREFLIGHT))

    def test_step4_gpu_shard_preflight_repo_command_allowed(self) -> None:
        classification = bridge.BridgeCommandPolicy.classify_repo_command(GPU_SHARD_PREFLIGHT)
        self.assertTrue(classification.allowed, classification.reason)
        self.assertEqual(classification.stage, "step4")
        self.assertEqual(classification.operation, "preflight")
        self.assertEqual(classification.validation_namespace, "step4_gpu_c9_bucket_balanced_s16384")
        self.assertEqual(classification.bounded_limit_value, 16384)

    def test_step4_prepare_cache_repo_command_allowed(self) -> None:
        command = (
            "./odcr",
            "step4",
            "--task",
            "2",
            "--prepare-cache",
            "--max-samples",
            "128",
            "--validation-namespace",
            "step4_preflight_smoke",
        )
        classification = bridge.BridgeCommandPolicy.classify_repo_command(command)
        self.assertTrue(classification.allowed, classification.reason)
        self.assertEqual(classification.operation, "prepare_cache")

    def test_step4_preflight_output_root_allowed(self) -> None:
        output_dir = bridge.resolve_runtime_output_dir(
            "runs/step4_preflight/task2/step4_preflight_smoke",
            "bridge_step4_policy",
        )
        self.assertTrue(str(output_dir).endswith("runs/step4_preflight/task2/step4_preflight_smoke"))

    def test_formal_and_unbounded_step4_commands_rejected(self) -> None:
        cases = {
            "formal": ("./odcr", "step4", "--task", "2"),
            "missing_limit": (
                "./odcr",
                "step4",
                "--task",
                "2",
                "--preflight",
                "--validation-namespace",
                "step4_preflight_smoke",
            ),
            "missing_namespace": ("./odcr", "step4", "--task", "2", "--preflight", "--max-samples", "128"),
            "formal_namespace": (
                "./odcr",
                "step4",
                "--task",
                "2",
                "--preflight",
                "--max-samples",
                "128",
                "--validation-namespace",
                "formal",
            ),
            "path_namespace": (
                "./odcr",
                "step4",
                "--task",
                "2",
                "--preflight",
                "--max-samples",
                "128",
                "--validation-namespace",
                "../bad",
            ),
            "formal_output": (
                "./odcr",
                "step4",
                "--task",
                "2",
                "--preflight",
                "--max-samples",
                "128",
                "--validation-namespace",
                "step4_preflight_smoke",
                "--output",
                "runs/step4/task2/latest.json",
            ),
        }
        for name, command in cases.items():
            with self.subTest(name=name):
                classification = bridge.BridgeCommandPolicy.classify_repo_command(command)
                self.assertFalse(classification.allowed)

    def test_step5_eval_rerank_background_allocation_and_destructive_rejected(self) -> None:
        commands = (
            ("./odcr", "step5", "--task", "2"),
            ("./odcr", "eval", "--task", "2"),
            ("./odcr", "rerank", "--task", "2"),
            ("nohup", "./odcr", "step4"),
            ("./odcr", "step4", "--task", "2", "&"),
            ("disown",),
            ("srun", "--pty", "bash"),
            ("sbatch", "job.sh"),
            ("scancel", "1"),
            ("odcr-enter-gpu", "1"),
            ("rm", "-rf", "runs/step4/task2"),
        )
        for command in commands:
            with self.subTest(command=command):
                classification = bridge.BridgeCommandPolicy.classify_repo_command(command)
                self.assertFalse(classification.allowed)


class GpuBridgeStep4EvidenceParserTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_step4_preflight_evidence_parser_recognizes_required_artifacts(self) -> None:
        classification = bridge.BridgeCommandPolicy.classify_repo_command(SAFE_PREFLIGHT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_json(
                root / "preflight_summary.json",
                {
                    "evidence_level": "E4_gpu_shard_forward_bounded",
                    "validation_namespace": "step4_preflight_smoke",
                    "sample_count": 16,
                    "max_samples": 128,
                    "formal_latest_write": False,
                    "formal_export_write": False,
                    "upstream_step3_run_id": "2",
                    "gpu_runtime_evidence": True,
                    "actual_gpu_forward_executed": True,
                    "actual_model_loaded_on_gpu": True,
                    "force_gpu_forward": True,
                },
            )
            self._write_json(
                root / "rcr_distribution.json",
                {
                    "sample_count": 16,
                    "route_scorer_count": 8,
                    "route_explainer_count": 8,
                    "train_keep_count": 16,
                    "confidence_bucket_distribution": {"1": 16},
                    "sample_weight_hint": {"min": 0.5, "mean": 0.7, "max": 1.0},
                },
            )
            self._write_json(root / "required_fields_check.json", {"passed": True, "missing": []})
            self._write_json(root / "manifest_preview.json", {"schema_version": "manifest"})
            self._write_json(root / "index_contract_preview.json", {"schema_version": "index"})
            self._write_json(root / "lineage_preview.json", {"lineage_hash": "abc"})
            self._write_json(
                root / "cpu_gpu_utilization_snapshot.json",
                {
                    "evidence_level": "E4_gpu_shard_forward_bounded",
                    "cuda_available": True,
                    "device_count": 1,
                    "gpu_runtime_evidence": True,
                    "actual_gpu_forward_executed": True,
                    "actual_model_loaded_on_gpu": True,
                    "force_gpu_forward": True,
                },
            )
            evidence = bridge.parse_step4_preflight_evidence(output_dir=root, classification=classification)
        self.assertTrue(evidence["evidence_complete"], evidence)
        self.assertTrue(evidence["runtime_evidence_ok"], evidence)
        self.assertEqual(evidence["route_scorer_count"], 8)

    def test_cuda_probe_only_is_not_step4_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_json(root / "report.json", {"cuda_available": True})
            evidence = bridge.parse_step4_preflight_evidence(output_dir=root)
        self.assertFalse(evidence["evidence_complete"])
        self.assertFalse(evidence["runtime_evidence_ok"])

    def test_formal_pollution_snapshot_change_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = bridge.snapshot_formal_namespace(root)
            latest = root / "runs" / "step4" / "task2" / "latest.json"
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_text("{}", encoding="utf-8")
            after = bridge.snapshot_formal_namespace(root)
        self.assertTrue(bridge.formal_namespace_polluted(before, after))


if __name__ == "__main__":
    unittest.main()
