from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step4_checkpoint_lineage import (  # noqa: E402
    architecture_hash_diff_report,
    ntoken_sidecar_checkpoint_compatibility_report,
    require_step4_checkpoint_architecture_compatible,
)
from odcr_core.training_checkpoint import (  # noqa: E402
    CheckpointLineageError,
    compute_model_architecture_config_hash,
    extract_checkpoint_state_dict_architecture_payload,
)


ARCH = {
    "nuser": 199218,
    "nitem": 114495,
    "ntoken": 32128,
    "emsize": 1024,
    "nlayers": 2,
    "nhead": 2,
    "nhid": 2048,
    "dropout": 0.2,
}


def _lineage(arch: dict[str, object] | None = None) -> dict[str, object]:
    payload = dict(arch or ARCH)
    return {
        "stage": "step3",
        "model_architecture_config": payload,
        "model_architecture_config_hash": compute_model_architecture_config_hash(payload),
    }


class Step4CheckpointArchitectureHashDiffTest(unittest.TestCase):
    def test_diff_reports_field_level_mismatch(self) -> None:
        expected = dict(ARCH)
        expected["ntoken"] = 32100
        report = architecture_hash_diff_report(
            checkpoint_lineage=_lineage(),
            expected_architecture_payload=expected,
            checkpoint_lineage_path="runs/step3/task2/2/model/best.pth.lineage.json",
        )
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["mismatch_keys"], ["ntoken"])
        self.assertEqual(report["mismatches"]["ntoken"]["checkpoint"], 32128)
        self.assertEqual(report["mismatches"]["ntoken"]["expected"], 32100)
        self.assertIn("hash_source_paths", report)

    def test_true_architecture_change_raises(self) -> None:
        expected = dict(ARCH)
        expected["emsize"] = 768
        with self.assertRaises(CheckpointLineageError):
            require_step4_checkpoint_architecture_compatible(
                checkpoint_lineage=_lineage(),
                expected_architecture_payload=expected,
            )

    def test_failed_run1_preserved_next_flat_run_id_is_2(self) -> None:
        from odcr_core.run_naming import next_run_id

        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            (parent / "1").mkdir()
            self.assertEqual(next_run_id(parent), "2")

    def test_state_dict_shape_supersedes_stale_sidecar_ntoken_for_load_arch(self) -> None:
        import torch

        sidecar = {
            "nuser": 5,
            "nitem": 6,
            "ntoken": 9,
            "emsize": 4,
            "nlayers": 2,
            "nhead": 2,
            "nhid": 8,
            "dropout": 0.2,
        }
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "best.pth"
            torch.save(
                {
                    "user_embeddings.weight": torch.zeros(5, 4),
                    "item_embeddings.weight": torch.zeros(6, 4),
                    "word_embeddings.weight": torch.zeros(7, 4),
                    "hidden2token.weight": torch.zeros(7, 4),
                    "hidden2token.bias": torch.zeros(7),
                    "transformer_encoder.layers.0.linear1.weight": torch.zeros(8, 4),
                    "transformer_encoder.layers.0.linear2.weight": torch.zeros(4, 8),
                    "transformer_encoder.layers.1.linear1.weight": torch.zeros(8, 4),
                    "transformer_encoder.layers.1.linear2.weight": torch.zeros(4, 8),
                },
                checkpoint,
            )
            tensor_arch = extract_checkpoint_state_dict_architecture_payload(
                checkpoint,
                fallback_payload=sidecar,
            )
        self.assertEqual(tensor_arch["ntoken"], 7)
        report = architecture_hash_diff_report(
            checkpoint_lineage=_lineage(sidecar),
            expected_architecture_payload=tensor_arch,
            checkpoint_architecture_payload=tensor_arch,
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["mismatch_keys"], [])
        self.assertEqual(report["sidecar_metadata_diff"]["mismatch_keys"], ["ntoken"])

    def test_ntoken_mismatch_is_explicit_compatible_policy_not_silent(self) -> None:
        tensor_arch = dict(ARCH)
        tensor_arch["ntoken"] = 32100
        report = ntoken_sidecar_checkpoint_compatibility_report(
            sidecar_architecture_payload=ARCH,
            checkpoint_architecture_payload=tensor_arch,
        )
        self.assertEqual(report["status"], "compatible")
        self.assertEqual(report["sidecar_ntoken"], 32128)
        self.assertEqual(report["checkpoint_tensor_ntoken"], 32100)
        self.assertEqual(report["effective_model_ntoken"], 32100)
        self.assertFalse(report["silent_ignore"])
        self.assertIn("checkpoint tensor shape", report["compatibility_note"])

    def test_incompatible_token_projection_shape_fails(self) -> None:
        import torch

        sidecar = {
            "nuser": 5,
            "nitem": 6,
            "ntoken": 7,
            "emsize": 4,
            "nlayers": 2,
            "nhead": 2,
            "nhid": 8,
            "dropout": 0.2,
        }
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "bad.pth"
            torch.save(
                {
                    "user_embeddings.weight": torch.zeros(5, 4),
                    "item_embeddings.weight": torch.zeros(6, 4),
                    "word_embeddings.weight": torch.zeros(7, 4),
                    "hidden2token.weight": torch.zeros(8, 4),
                    "hidden2token.bias": torch.zeros(7),
                    "transformer_encoder.layers.0.linear1.weight": torch.zeros(8, 4),
                    "transformer_encoder.layers.0.linear2.weight": torch.zeros(4, 8),
                    "transformer_encoder.layers.1.linear1.weight": torch.zeros(8, 4),
                    "transformer_encoder.layers.1.linear2.weight": torch.zeros(4, 8),
                },
                checkpoint,
            )
            with self.assertRaisesRegex(CheckpointLineageError, "token projection shape"):
                extract_checkpoint_state_dict_architecture_payload(checkpoint, fallback_payload=sidecar)


if __name__ == "__main__":
    unittest.main()
