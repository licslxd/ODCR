from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from odcr_core.step4_checkpoint_lineage import (  # noqa: E402
    STEP4_ARCHITECTURE_IGNORED_CONFIG_KEYS,
    architecture_hash_diff_report,
    build_step4_expected_checkpoint_arch_payload,
    live_vs_frozen_step3_config_drift_report,
    source_table_hash_scope_report,
)
from odcr_core.training_checkpoint import compute_model_architecture_config_hash, stable_hash  # noqa: E402


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


def _lineage() -> dict[str, object]:
    return {
        "stage": "step3",
        "model_architecture_config": dict(ARCH),
        "model_architecture_config_hash": compute_model_architecture_config_hash(ARCH),
    }


class Step4CheckpointLineageHashScopeTest(unittest.TestCase):
    def test_step4_rcr_changes_do_not_change_step3_arch_hash(self) -> None:
        base = dict(ARCH)
        changed = dict(ARCH)
        changed["step4_rcr"] = {"route_scorer": {"min_reliability": 0.99}}
        self.assertEqual(
            compute_model_architecture_config_hash(base),
            compute_model_architecture_config_hash(changed),
        )

    def test_step4_runtime_p3_changes_do_not_change_step3_arch_hash(self) -> None:
        changed = dict(ARCH)
        changed["step4_runtime"] = {"partial_format": "parquet", "decode_chunk": 128}
        changed["eval"] = {"profile": "balanced_2gpu"}
        self.assertEqual(
            compute_model_architecture_config_hash(ARCH),
            compute_model_architecture_config_hash(changed),
        )

    def test_evidence_no_accum_docs_guardrail_do_not_change_step3_arch_hash(self) -> None:
        changed = dict(ARCH)
        changed.update(
            {
                "evidence_level": "E4_gpu_shard_forward_bounded",
                "batch_semantics": "odcr_no_accum/1",
                "docs": {"note": "documentation-only"},
                "guardrail": {"strict": True},
            }
        )
        self.assertEqual(
            compute_model_architecture_config_hash(ARCH),
            compute_model_architecture_config_hash(changed),
        )

    def test_expected_payload_comes_from_checkpoint_lineage(self) -> None:
        expected = build_step4_expected_checkpoint_arch_payload(_lineage())
        self.assertEqual(expected, ARCH)
        report = architecture_hash_diff_report(
            checkpoint_lineage=_lineage(),
            expected_architecture_payload=expected,
            observed_loader_architecture_payload={**ARCH, "ntoken": 32100},
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["mismatch_keys"], [])
        self.assertEqual(report["observed_current_loader"]["mismatch_keys"], ["ntoken"])
        self.assertIn("step4_rcr", STEP4_ARCHITECTURE_IGNORED_CONFIG_KEYS)
        self.assertIn("step4_runtime", STEP4_ARCHITECTURE_IGNORED_CONFIG_KEYS)

    def test_expected_payload_comes_from_checkpoint_tensor_shapes_when_available(self) -> None:
        import tempfile

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
            expected = build_step4_expected_checkpoint_arch_payload(
                {
                    "stage": "step3",
                    "model_architecture_config": sidecar,
                    "model_architecture_config_hash": compute_model_architecture_config_hash(sidecar),
                },
                checkpoint_path=checkpoint,
            )
        self.assertEqual(expected["ntoken"], 7)
        self.assertEqual(expected["nhead"], 2)

    def test_source_table_hash_scope_labels_frozen_vs_live(self) -> None:
        current_payload = {
            "task_id": 2,
            "step4_rcr": {"route_scorer": {"min_reliability": 0.9}},
            "step4_runtime": {"partial_format": "parquet"},
        }
        frozen_source_table = {
            "field_sources": {
                "task": "tasks.2",
                "step3_structured_losses": "step3.structured_losses",
                "embed_dim": "env.embed_dim",
            }
        }
        expected_hash = stable_hash(
            {
                "schema_version": "odcr_step3_checkpoint_compat/2",
                "field_sources": dict(frozen_source_table["field_sources"]),
            }
        )
        report = source_table_hash_scope_report(
            checkpoint_lineage={
                **_lineage(),
                "source_table_hash": "frozen-source-table",
                "source_table_compatibility_hash": expected_hash,
                "resolved_config_compatibility_hash": "frozen-training",
                "source_table_path": "runs/step3/task2/2/meta/source_table.json",
            },
            source_table_compatibility_payload={"schema_version": "x", "field_sources": {"step4_rcr": "live"}},
            current_payload=current_payload,
            checkpoint_architecture_payload=ARCH,
        )
        self.assertFalse(report["blocking"])
        self.assertEqual(report["hash_scopes"]["step3_checkpoint_arch_hash"]["severity"], "block")
        self.assertEqual(report["hash_scopes"]["step4_rcr_config_hash"]["severity"], "display-only")
        self.assertEqual(report["field_diffs"][0]["severity"], "display-only")

    def test_live_vs_frozen_config_drift_is_labeled_non_blocking(self) -> None:
        report = live_vs_frozen_step3_config_drift_report(
            checkpoint_lineage={
                **_lineage(),
                "resolved_config_compatibility_hash": "frozen",
                "checkpoint_path": "runs/step3/task2/2/model/best_observed.pth",
            },
            current_payload={"step4_rcr": {"a": 1}, "step4_runtime": {"b": 2}},
        )
        self.assertEqual(report["status"], "allowed_historical_vs_live_drift")
        self.assertFalse(report["blocks_step4"])
        self.assertFalse(report["checkpoint_compatibility_uses_current_live_step3_config"])
        self.assertIn("Current configs/odcr.yaml Step3 settings", report["policy"])


if __name__ == "__main__":
    unittest.main()
