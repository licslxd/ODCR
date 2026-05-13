from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odcr_core.config_resolver import resolve_config
from odcr_core.runners import _torchrun_hardware_env


REPO_ROOT = Path(__file__).resolve().parents[2]


class Step3AllocatorTransportTest(unittest.TestCase):
    def test_step3_allocator_resolves_from_one_control_and_transports_to_launcher_env(self) -> None:
        cfg, _sources, snapshot = resolve_config(
            config_path=REPO_ROOT / "configs" / "odcr.yaml",
            command="step3",
            task_id=2,
            set_overrides=["experiment_profile=csb_odcr_full_safe"],
            dry_run=True,
            run_id="auto",
            mode="full",
        )

        allocator = snapshot["step3_memory"]["allocator"]
        self.assertEqual(allocator["source"], "step3.memory.allocator")
        self.assertEqual(allocator["cuda_alloc_conf"], "expandable_segments:True")
        self.assertEqual(
            snapshot["runtime_env"]["launcher_env_effective"]["PYTORCH_CUDA_ALLOC_CONF"],
            "expandable_segments:True",
        )
        self.assertEqual(snapshot["field_sources"]["step3_memory_allocator"], "step3.memory.allocator")
        self.assertEqual(_torchrun_hardware_env(cfg)["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")


    def test_step4_does_not_inherit_step3_allocator_transport(self) -> None:
        cfg = SimpleNamespace(
            train_precision="bf16",
            allow_tf32=True,
            amp_autocast=True,
            grad_scaler=False,
            tokenizer_max_length=48,
            evidence_max_length=48,
            omp_num_threads=1,
            mkl_num_threads=1,
            tokenizers_parallelism=False,
            launcher_env_effective_json='{"CUDA_VISIBLE_DEVICES": "0,1"}',
            hardware_profile_json="{}",
            hardware_preset_id="default",
        )

        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", _torchrun_hardware_env(cfg))


if __name__ == "__main__":
    unittest.main()
