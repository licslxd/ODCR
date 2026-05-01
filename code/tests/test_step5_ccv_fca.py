from __future__ import annotations

import os
import sys
import unittest

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.gather_schema import GatheredBatch  # noqa: E402
from odcr_core.step5_innovation import (  # noqa: E402
    STEP5_EVIDENCE_FEATURE_DIM,
    build_ccv_control_packet,
    build_step5b_explainer_gate,
    evidence_basis_fca_loss,
    for_test_default_step5_innovation_config,
    parse_step5_innovation_config_json,
)


def _batch(with_ids: bool = True) -> GatheredBatch:
    ev = torch.zeros(2, STEP5_EVIDENCE_FEATURE_DIM)
    ev[:, 0] = torch.tensor([0.8, 0.6])
    ev[:, 1] = torch.tensor([0.9, 0.7])
    ev[:, 2] = torch.tensor([0.3, 0.8])
    ev[:, 3] = torch.tensor([0.9, 0.6])
    ev[:, 4] = torch.tensor([0.9, 0.7])
    ev[:, 5] = torch.tensor([1.0, 0.8])
    ev[:, 6] = torch.tensor([0.1, 0.3])
    kwargs = {}
    if with_ids:
        kwargs.update(
            content_evidence_ids=torch.tensor([[1, 2], [3, 0]], dtype=torch.long),
            style_evidence_ids=torch.tensor([[4], [5]], dtype=torch.long),
            domain_style_anchor_ids=torch.tensor([[6], [7]], dtype=torch.long),
            local_style_hint_ids=torch.tensor([[8], [9]], dtype=torch.long),
            polarity_ids=torch.tensor([[10], [11]], dtype=torch.long),
        )
    return GatheredBatch(
        user_idx=torch.arange(2),
        item_idx=torch.arange(2),
        rating=torch.tensor([4.0, 2.0]),
        tgt_input=torch.ones(2, 2, dtype=torch.long),
        tgt_output=torch.ones(2, 2, dtype=torch.long),
        domain_idx=torch.zeros(2, dtype=torch.long),
        sample_id=torch.arange(2),
        exp_sample_weight=torch.ones(2),
        route_scorer_mask=torch.tensor([1.0, 0.0]),
        route_explainer_mask=torch.tensor([1.0, 1.0]),
        uncertainty_score=torch.tensor([0.1, 0.3]),
        confidence_bucket=torch.tensor([2.0, 1.0]),
        content_anchor_score=torch.tensor([0.9, 0.7]),
        style_anchor_score=torch.tensor([0.3, 0.8]),
        evidence_features=ev,
        **kwargs,
    )


class TestStep5CCVFCA(unittest.TestCase):
    def test_explainer_gate_config_changes_step5b_weight(self) -> None:
        batch = _batch()
        base_cfg = for_test_default_step5_innovation_config()
        changed_cfg = parse_step5_innovation_config_json(
            {
                "explainer_gate": {
                    "bucket_weights": {"high": 2.0, "medium": 1.0, "low": 0.55},
                    "uncertainty_exponent": 0.85,
                    "style_shift_diversity_boost": 0.15,
                    "min_weight": 0.0,
                    "max_weight": 4.0,
                    "explainer_only_multiplier": 0.7,
                }
            },
            allow_test_defaults=True,
        )
        base_gate = build_step5b_explainer_gate(batch, base_cfg)
        changed_gate = build_step5b_explainer_gate(batch, changed_cfg)
        self.assertGreater(
            float(changed_gate.explainer_weight[0]),
            float(base_gate.explainer_weight[0]),
        )

    def test_ccv_control_packet_is_required_and_structured(self) -> None:
        cfg = for_test_default_step5_innovation_config()
        packet = build_ccv_control_packet(_batch(), cfg)
        controls = packet.numeric_controls()
        self.assertEqual(tuple(controls.shape), (2, cfg.ccv.numeric_control_dim))
        self.assertGreater(float(controls[:, 3].mean()), 0.0)
        self.assertEqual(cfg.ccv.control_packet_field_policy, "strict_required")
        self.assertEqual(cfg.ccv.verbalizer_adapter_policy, "ccv_control_adapter")
        with self.assertRaisesRegex(RuntimeError, "CCV control packet missing"):
            build_ccv_control_packet(_batch(with_ids=False), cfg)

    def test_fca_aligns_evidence_bases_and_is_weighted(self) -> None:
        cfg = for_test_default_step5_innovation_config()
        batch = _batch()
        packet = build_ccv_control_packet(batch, cfg)
        gate = build_step5b_explainer_gate(batch, cfg)
        torch.manual_seed(0)
        scorer_h = torch.randn(2, 4)
        explainer_h = scorer_h + 0.05 * torch.randn(2, 4)
        shared = torch.randn(2, 4)
        content_profile = torch.randn(2, 4)
        content_evidence = torch.randn(2, 4)
        bundle = evidence_basis_fca_loss(
            scorer_hidden=scorer_h,
            explainer_hidden=explainer_h,
            shared_latent=shared,
            content_profile=content_profile,
            content_evidence_latent=content_evidence,
            packet=packet,
            gate=gate,
            cfg=cfg,
        )
        self.assertGreaterEqual(float(bundle.fca_loss), 0.0)
        self.assertAlmostEqual(
            float(bundle.fca_weighted_loss),
            float(bundle.fca_loss) * float(cfg.fca.weight),
            places=6,
        )
        self.assertEqual(tuple(bundle.scorer_evidence_basis.shape), (2, 4))
        self.assertEqual(tuple(bundle.explainer_evidence_basis.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
