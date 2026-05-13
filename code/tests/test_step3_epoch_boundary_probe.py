from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.aux.runtime import gpu_bridge
from odcr_core.aux.runtime.stage_dispatch import runtime_probe_bridge_args
from odcr_core.step3_runtime_probe import (
    EPOCH_BOUNDARY_MEMORY_REQUIRED_PHASES,
    STEP3_RUNTIME_PROBE_TYPES,
    Step3RuntimeEvidenceSink,
    Step3ValidationNamespaceGuard,
    Step3ValidationWindowRequest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step3EpochBoundaryProbeTest(unittest.TestCase):
    def test_runtime_probe_registers_epoch_boundary_memory_for_safe704(self) -> None:
        args = runtime_probe_bridge_args(
            stage="step3",
            task=2,
            profile="csb_odcr_full_safe",
            bounded=True,
            no_send=True,
        )

        self.assertIn("epoch-boundary-memory", STEP3_RUNTIME_PROBE_TYPES)
        self.assertIn("--probe-type", args)
        self.assertIn("epoch-boundary-memory", args)
        self.assertIn("--candidate-name", args)
        self.assertIn("csb_odcr_full_safe", args)

    def test_bridge_script_uses_epoch_boundary_probe_type(self) -> None:
        target = gpu_bridge.PaneCandidate(
            socket="/sock/gpu",
            session="odcr",
            target="odcr:0.0",
            pane_id="%0",
            pane_pid=100,
            pane_command="srun",
            cwd=str(REPO_ROOT),
            active=True,
            dead=False,
            in_mode=False,
            srun_pid=200,
            srun_command="srun --jobid=10 --pty bash",
            job_id="10",
            step_id="10.0",
            node="gpu01",
            gpu="gres/gpu=1,TresPerNode=gpu:A100:1",
            step_state="RUNNING",
            job_state="RUNNING",
        )
        paths = gpu_bridge.make_generated_paths("bridge_step3_epoch_boundary_test")
        script = gpu_bridge.build_step3_performance_probe_script(
            "bridge_step3_epoch_boundary_test",
            paths,
            gpu_bridge.resolve_timeouts("step3-performance-probe"),
            target,
            task_id=2,
            candidate_name="csb_odcr_full_safe",
            probe_type="epoch-boundary-memory",
        )

        gpu_bridge.validate_script_safety(script)
        self.assertIn("--probe-type epoch-boundary-memory", script)
        self.assertIn("--candidate-name csb_odcr_full_safe", script)
        self.assertNotIn("latest.json", script)
        self.assertNotIn("torchrun", script)

    def test_epoch_boundary_probe_requires_boundary_memory_phases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = Step3ValidationWindowRequest(
                task_id=2,
                validation_slug="validation_slug",
                run_id="run_probe",
                probe_type="epoch-boundary-memory",
                candidate_name="csb_odcr_full_safe",
                warmup_steps=0,
                measured_steps=1,
                max_wall_seconds=20,
                bridge_dispatched=True,
            )
            guard = Step3ValidationNamespaceGuard(Path(tmp), 2, "validation_slug", "run_probe")
            sink = Step3RuntimeEvidenceSink(request=request, guard=guard)
            base = {
                "validation_run_id": "run_probe",
                "task_id": 2,
                "profile_id": "task2_strong_forward_g1s",
                "probe_type": "epoch-boundary-memory",
                "rank": 0,
                "world_size": 2,
                "device": "cuda:0",
                "allocated_gib": 1.0,
                "max_allocated_gib": 1.0,
                "reserved_gib": 1.0,
                "max_reserved_gib": 1.0,
                "reserved_minus_allocated_gib": 0.0,
                "cuda_malloc_retry_count": 0,
                "non_releasable_gib": 0.0,
                "inactive_split_gib": 0.0,
            }
            sink.memory_rows = [{**base, "phase": phase} for phase in EPOCH_BOUNDARY_MEMORY_REQUIRED_PHASES]

            ok, findings = sink.validate(state={"formal_namespace_polluted": False})

            self.assertFalse(ok)
            self.assertTrue(any("after_batch_cpu" in item for item in findings))


if __name__ == "__main__":
    unittest.main()

