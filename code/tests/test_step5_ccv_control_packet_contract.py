from __future__ import annotations

import os
import sys
import unittest
import tempfile
from pathlib import Path

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.gather_schema import GatheredBatch  # noqa: E402
from odcr_core.step5_innovation import (  # noqa: E402
    STEP5_EVIDENCE_FEATURE_DIM,
    build_ccv_control_packet,
    for_test_default_step5_innovation_config,
    validate_ccv_control_packet_shapes,
)
from odcr_core.manifests import _extract_failure_root_signature  # noqa: E402


def _batch(*, polarity_rank1: bool = False) -> GatheredBatch:
    bsz = 2
    ids = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    ev = torch.zeros(bsz, STEP5_EVIDENCE_FEATURE_DIM)
    ev[:, 1] = 0.9
    ev[:, 2] = 0.2
    ev[:, 3] = 0.8
    ev[:, 4] = 0.7
    ev[:, 5] = 1.0
    ev[:, 6] = 0.1
    ev[:, 7] = 1.0
    return GatheredBatch(
        user_idx=torch.arange(bsz),
        item_idx=torch.arange(bsz),
        rating=torch.ones(bsz),
        tgt_input=ids,
        tgt_output=ids,
        domain_idx=torch.zeros(bsz, dtype=torch.long),
        sample_id=torch.arange(bsz),
        exp_sample_weight=torch.ones(bsz),
        route_scorer_mask=torch.ones(bsz),
        route_explainer_mask=torch.ones(bsz),
        uncertainty_score=torch.full((bsz,), 0.1),
        confidence_bucket=torch.full((bsz,), 2.0),
        content_anchor_score=torch.full((bsz,), 0.8),
        style_anchor_score=torch.full((bsz,), 0.7),
        evidence_features=ev,
        content_evidence_ids=ids,
        style_evidence_ids=ids,
        domain_style_anchor_ids=ids,
        local_style_hint_ids=ids,
        polarity_ids=torch.tensor([2, 1], dtype=torch.long) if polarity_rank1 else torch.tensor([[2], [1]], dtype=torch.long),
    )


class TestStep5CCVControlPacketContract(unittest.TestCase):
    def test_real_test_fixture_batch_uses_rank2_text_controls(self) -> None:
        batch = _batch()
        for name in (
            "content_evidence_ids",
            "style_evidence_ids",
            "domain_style_anchor_ids",
            "local_style_hint_ids",
            "polarity_ids",
        ):
            value = getattr(batch, name)
            self.assertEqual(value.dim(), 2, name)
            self.assertEqual(value.size(0), batch.user_idx.size(0), name)
        self.assertEqual(tuple(batch.polarity_ids.shape), (2, 1))

    def test_rank1_polarity_ids_fail_at_packet_boundary_with_producer(self) -> None:
        cfg = for_test_default_step5_innovation_config()
        with self.assertRaisesRegex(RuntimeError, r"producer=build_ccv_control_packet.*polarity_ids.*\[B,T\]"):
            build_ccv_control_packet(_batch(polarity_rank1=True), cfg)

    def test_validator_reports_test_fixture_producer(self) -> None:
        batch = _batch(polarity_rank1=True)
        with self.assertRaisesRegex(RuntimeError, r"producer=test_fixture_bad_packet.*polarity_ids.*\[B,T\]"):
            validate_ccv_control_packet_shapes(
                batch,
                producer="test_fixture_bad_packet",
                head="step5A",
                strict=True,
            )

    def test_run_summary_classifier_prefers_ccv_shape_over_tokenization_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta"
            meta.mkdir()
            (meta / "errors.log").write_text(
                "[Tokenize] historical cache line\n"
                "producer=test_fixture_bad_packet RuntimeError: CCV control ids must be [B,T], got (4,)\n",
                encoding="utf-8",
            )
            payload = _extract_failure_root_signature(
                meta=meta,
                latest_error="torchrun failed",
                repo_root=Path(tmp),
                checkpoint_path=None,
            )
        self.assertEqual(payload["failure_phase"], "data_collate")
        self.assertEqual(payload["failure_type"], "ccv_control_packet_shape_contract")
        self.assertEqual(payload["root_cause"], "real_batch_control_packet_shape_invalid")


if __name__ == "__main__":
    unittest.main()
