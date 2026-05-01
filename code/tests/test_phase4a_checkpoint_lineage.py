from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from odcr_core.training_checkpoint import (  # noqa: E402
    CheckpointLineageError,
    checkpoint_lineage_path_for_weight,
    read_checkpoint_lineage,
    write_checkpoint_lineage,
)


class TestPhase4ACheckpointLineage(unittest.TestCase):
    def test_checkpoint_lineage_sidecar_hash_and_stage_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "run" / "model" / "best.pth"
            ckpt.parent.mkdir(parents=True)
            ckpt.write_bytes(b"not-a-real-torch-state-dict")
            sidecar = write_checkpoint_lineage(
                ckpt,
                {
                    "stage": "step5",
                    "compat_schema_version": "test",
                    "checkpoint_compatibility_hash": "abc",
                },
            )
            self.assertEqual(sidecar, checkpoint_lineage_path_for_weight(ckpt))
            loaded = read_checkpoint_lineage(ckpt, expected_stage="step5")
            self.assertEqual(loaded["stage"], "step5")
            self.assertIn("lineage_hash", loaded)
            with self.assertRaises(CheckpointLineageError):
                read_checkpoint_lineage(ckpt, expected_stage="step3")

            data = json.loads(sidecar.read_text(encoding="utf-8"))
            data["checkpoint_compatibility_hash"] = "tampered"
            sidecar.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(CheckpointLineageError):
                read_checkpoint_lineage(ckpt, expected_stage="step5")


if __name__ == "__main__":
    unittest.main()
