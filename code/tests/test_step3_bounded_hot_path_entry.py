from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402
from odcr_core import step3_runtime_probe  # noqa: E402
from odcr_core.step3_runtime_probe import Step3ValidationWindowRequest  # noqa: E402


class Step3BoundedHotPathEntryTest(unittest.TestCase):
    def test_validation_entry_names_exist(self) -> None:
        self.assertTrue(callable(step3.run_step3_validation_window))
        self.assertTrue(callable(step3.build_step3_training_components))
        self.assertTrue(callable(step3.run_step3_measured_steps))

    def test_runtime_worker_uses_real_step3_hot_path_symbols(self) -> None:
        source = inspect.getsource(step3_runtime_probe._runtime_rank_worker)
        for token in (
            "build_config_and_data_ddp",
            "Step3CUDAPrefetcher",
            "compose_step3_loss_from_forward_output",
            "backward_step3_primary_and_sidecar_losses",
            "optimizer.step()",
            "step3_sync_loss_bundle_finite_status",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_measured_steps_must_be_positive(self) -> None:
        request = Step3ValidationWindowRequest(
            task_id=2,
            validation_slug="truth_probe",
            run_id="unit",
            probe_type="timing-profile-window",
            measured_steps=1,
            max_wall_seconds=30,
        )
        self.assertGreater(request.measured_steps, 0)
        with self.assertRaises(Exception):
            Step3ValidationWindowRequest(
                task_id=2,
                validation_slug="truth_probe",
                run_id="unit",
                probe_type="timing-profile-window",
                measured_steps=0,
                max_wall_seconds=30,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
