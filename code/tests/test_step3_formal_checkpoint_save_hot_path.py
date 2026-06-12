from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from executors import step3_train_core as step3  # noqa: E402


class TestStep3FormalCheckpointSaveHotPath(unittest.TestCase):
    def test_recovery_checkpoint_load_map_location_converts_int_rank(self) -> None:
        map_location = step3._torch_load_map_location_for_device(1)
        self.assertIsInstance(map_location, torch.device)
        self.assertEqual(map_location.type, "cuda")
        self.assertEqual(map_location.index, 1)
        self.assertEqual(step3._torch_load_map_location_for_device(torch.device("cpu")), torch.device("cpu"))

    def test_hot_path_passes_explicit_checkpoint_event_semantics(self) -> None:
        source = Path(step3.__file__).read_text(encoding="utf-8")
        self.assertIn("checkpoint_event_from_sidecar(lineage, reason=reason, replaced_previous=replaced_previous)", source)
        for reason in (
            "global_best_improved",
            "after_min_epochs_best_improved",
            "latest_epoch_snapshot",
            "topk_retained",
            "topk_replaced",
        ):
            self.assertIn(reason, source)
        self.assertNotIn("checkpoint_event_from_sidecar(lineage)", source)


if __name__ == "__main__":
    unittest.main()
