from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step3_quality import checkpoint_event_from_sidecar  # noqa: E402


class TestStep3CheckpointEventFromSidecarSignature(unittest.TestCase):
    def _sidecar(self) -> dict:
        return {
            "checkpoint_path": "/tmp/latest.pth",
            "checkpoint_file_hash": "hash",
            "checkpoint_epoch": 2,
            "selection_scope": "best_observed",
            "selection_metric": "valid_loss",
            "selection_metric_value": 4.8578,
            "selection_direction": "min",
            "global_best_epoch": 2,
            "global_best_metric": 4.8578,
            "after_min_epochs_best_epoch": 7,
            "after_min_epochs_best_metric": 8.1553,
            "resolved_config_hash": "resolved",
            "training_runtime_config_hash": "runtime",
            "epoch_summary_hash": "epoch-summary",
            "metrics_jsonl_hash": "metrics-jsonl",
            "quality_status": "not_evaluated",
            "downstream_ready": False,
        }

    def test_missing_reason_and_replaced_previous_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            checkpoint_event_from_sidecar(self._sidecar())  # type: ignore[call-arg]

    def test_explicit_reason_and_replaced_previous_are_required_event_fields(self) -> None:
        event = checkpoint_event_from_sidecar(
            self._sidecar(),
            reason="global_best_improved",
            replaced_previous=False,
        )
        for key in (
            "event_id",
            "checkpoint_file",
            "checkpoint_file_hash",
            "checkpoint_epoch",
            "selection_scope",
            "selection_metric",
            "selection_metric_value",
            "selection_direction",
            "reason",
            "replaced_previous",
            "global_best_epoch",
            "global_best_metric",
            "after_min_epochs_best_epoch",
            "after_min_epochs_best_metric",
            "resolved_config_hash",
            "training_runtime_config_hash",
            "epoch_summary_hash",
            "metrics_jsonl_hash",
            "quality_status",
            "downstream_ready",
            "created_at",
        ):
            self.assertIn(key, event)
        self.assertEqual(event["reason"], "global_best_improved")
        self.assertFalse(event["replaced_previous"])


if __name__ == "__main__":
    unittest.main()
