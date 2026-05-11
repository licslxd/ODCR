"""Unit tests for the bridge-only Step3 DDP smoke probe schema."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

import odcr_step3_real_data_probe as probe  # noqa: E402


class Step3RealDataProbeTests(unittest.TestCase):
    def test_help_does_not_expose_formal_train_command(self) -> None:
        proc = subprocess.run(
            [sys.executable, "code/tools/odcr_step3_real_data_probe.py", "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("bridge-only", proc.stdout)
        self.assertNotIn("./odcr step3", proc.stdout)
        self.assertNotIn("torchrun", proc.stdout)
        self.assertNotIn("start formal", proc.stdout.lower())

    def test_cpu12_worker_formula_guard(self) -> None:
        _cfg, _sources, snapshot = probe.resolve_one_control(2)
        result = probe.validate_cpu_budget(snapshot, strict=True)
        formula = result["formula"]
        self.assertEqual(formula["max_parallel_cpu"], 12)
        self.assertEqual(formula["dataloader_workers_per_rank"]["train"], 4)
        self.assertEqual(formula["dataloader_workers_per_rank"]["valid"], 2)
        self.assertEqual(formula["train_total_with_reserved_cpu"], 10)
        self.assertEqual(formula["tokenization_total_with_reserved_cpu"], 10)
        self.assertTrue(formula["train_safe"])
        self.assertTrue(formula["tokenization_safe"])

    def test_smoke_candidate_resolves_from_one_control_ladder(self) -> None:
        _cfg, sources, snapshot = probe.resolve_one_control(2, smoke_candidate="G0")
        self.assertEqual(snapshot["train"]["batch_size"], 1024)
        self.assertEqual(snapshot["train"]["per_gpu_batch_size"], 512)
        self.assertNotIn("grad_accum", snapshot["train"])
        self.assertEqual(snapshot["train"]["batch_semantics_version"], "odcr_no_accum/1")
        selection = snapshot["step3_probe_candidate_selection"]
        self.assertEqual(selection["name"], "G0")
        self.assertIn("step3.backup_profiles", selection["source"])
        self.assertTrue(selection["cross_rank_structured_gather"])
        self.assertTrue(any(row["key"] == "step3_probe_candidate_selection" for row in sources))

    def test_performance_probe_candidate_and_worker_resolve_from_one_control(self) -> None:
        _cfg, sources, snapshot = probe.resolve_one_control(
            2,
            mode="step3-performance-probe",
            candidate_name="G1S",
            worker_profile="W2",
        )
        self.assertEqual(snapshot["train"]["batch_size"], 1536)
        self.assertEqual(snapshot["train"]["per_gpu_batch_size"], 768)
        self.assertNotIn("grad_accum", snapshot["train"])
        self.assertEqual(snapshot["hardware"]["dataloader_num_workers_train"], 5)
        self.assertEqual(snapshot["hardware"]["dataloader_prefetch_factor_train"], 2)
        self.assertEqual(snapshot["step3_probe_candidate_selection"]["name"], "G1S")
        self.assertTrue(snapshot["step3_cross_rank_structured_gather"]["enabled"])
        self.assertEqual(snapshot["step3_cross_rank_structured_gather"]["mode"], "local_gradient_context")
        self.assertEqual(snapshot["step3_worker_profile_selection"]["name"], "W2")
        self.assertTrue(any(row["key"] == "step3_probe_candidate_selection" for row in sources))

    def test_unknown_smoke_candidate_rejected(self) -> None:
        with self.assertRaises(probe.ProbeError):
            probe.resolve_one_control(2, smoke_candidate="missing_candidate")

    def test_schema_only_writes_not_verified_json_under_ai_analysis(self) -> None:
        run_id = "unit_step3_probe_schema"
        rc = probe.main(
            [
                "--mode",
                "step3-ddp-smoke",
                "--task-id",
                "2",
                "--run-id",
                run_id,
                "--strict",
                "--schema-only",
                "--max-batches",
                "1",
                "--max-steps",
                "1",
                "--no-formal-checkpoint",
            ]
        )
        self.assertEqual(rc, 0)
        paths = probe.make_paths(run_id, mode="step3-ddp-smoke", bridge_log_path=None, bridge_status_path=None)
        payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], probe.SCHEMA_VERSION)
        self.assertEqual(payload["mode"], "step3-ddp-smoke")
        self.assertEqual(payload["task_id"], 2)
        self.assertEqual(payload["one_control"]["optimizer"]["name"], "adamw")
        self.assertEqual(payload["one_control"]["precision"]["train_precision"], "bf16")
        self.assertEqual(payload["one_control"]["max_parallel_cpu"], 12)
        self.assertEqual(payload["verdict"]["status"], "NOT_VERIFIED")
        self.assertIn("schema_only requested", " ".join(payload["verdict"]["blockers"]))
        self.assertTrue(str(paths.json_path.resolve()).startswith(str(probe.AI_ANALYSIS.resolve())))
        self.assertTrue(paths.md_path.is_file())

    def test_performance_probe_schema_requires_optimizer_step_without_real_run(self) -> None:
        run_id = "unit_step3_perf_probe_schema"
        rc = probe.main(
            [
                "--mode",
                "step3-performance-probe",
                "--task-id",
                "2",
                "--candidate-name",
                "task2_g2_effective_pool_2048",
                "--worker-profile",
                "W0",
                "--run-id",
                run_id,
                "--strict",
                "--schema-only",
                "--no-formal-checkpoint",
            ]
        )
        self.assertEqual(rc, 0)
        paths = probe.make_paths(run_id, mode="step3-performance-probe", bridge_log_path=None, bridge_status_path=None)
        payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "step3-performance-probe")
        self.assertTrue(payload["one_control"]["optimizer_step_required"])
        self.assertEqual(payload["one_control"]["probe_candidate"]["name"], "task2_g2_effective_pool_2048")
        self.assertTrue(payload["one_control"]["probe_candidate"]["probe_only"])
        self.assertFalse(payload["one_control"]["probe_candidate"]["formal_allowed"])
        self.assertTrue(payload["outputs"]["formal_writes_forbidden"]["formal_checkpoint"])

    def test_no_formal_checkpoint_flag_is_mandatory(self) -> None:
        with self.assertRaises(SystemExit):
            probe.build_parser().parse_args(["--run-id", "unit_missing_no_checkpoint", "--schema-only"])

    def test_unsafe_run_id_rejected(self) -> None:
        with self.assertRaises(probe.ProbeError):
            probe.make_paths("../bad", mode="step3-ddp-smoke", bridge_log_path=None, bridge_status_path=None)

    def test_probe_uses_shared_step3_loss_builder_not_side_channel(self) -> None:
        text = (REPO_ROOT / "code" / "tools" / "odcr_step3_real_data_probe.py").read_text(encoding="utf-8")
        self.assertIn("compose_step3_loss_from_forward_output", text)
        self.assertIn("validate_step3_graph_safety_preflight", text)
        self.assertIn("memory_phase_snapshots", text)
        self.assertIn("oom_allocation_request", text)
        self.assertNotIn("underlying.last_odcr_latents", text)
        self.assertNotIn("ddp_model.module", text)
        self.assertNotIn("domain_style_proto.weight", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
