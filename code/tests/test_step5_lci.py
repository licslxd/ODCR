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
    build_step5a_scorer_gate,
    for_test_default_step5_innovation_config,
    lci_score_invariance_loss,
)


def _batch() -> GatheredBatch:
    ev = torch.zeros(3, STEP5_EVIDENCE_FEATURE_DIM)
    ev[:, 1] = torch.tensor([0.90, 0.70, 0.95])  # cf_reliability_score
    ev[:, 3] = torch.tensor([0.90, 0.70, 0.95])  # rating_stability_score
    ev[:, 4] = torch.tensor([0.90, 0.70, 0.95])  # content_retention_score
    ev[:, 6] = torch.tensor([0.10, 0.40, 0.05])  # uncertainty_score
    return GatheredBatch(
        user_idx=torch.arange(3),
        item_idx=torch.arange(3),
        rating=torch.tensor([4.0, 3.0, 5.0]),
        tgt_input=torch.ones(3, 2, dtype=torch.long),
        tgt_output=torch.ones(3, 2, dtype=torch.long),
        domain_idx=torch.ones(3, dtype=torch.long),
        sample_id=torch.arange(3),
        exp_sample_weight=torch.ones(3),
        route_scorer_mask=torch.tensor([1.0, 1.0, 0.0]),
        route_explainer_mask=torch.tensor([0.0, 1.0, 1.0]),
        uncertainty_score=torch.tensor([0.10, 0.40, 0.05]),
        confidence_bucket=torch.tensor([2.0, 1.0, 2.0]),
        content_anchor_score=torch.ones(3),
        style_anchor_score=torch.ones(3),
        evidence_features=ev,
    )


class TestStep5LCIUCI(unittest.TestCase):
    def test_uci_weights_high_confidence_lci_more_than_medium_and_route_zero(self) -> None:
        cfg = for_test_default_step5_innovation_config()
        gate = build_step5a_scorer_gate(_batch(), cfg)
        self.assertGreater(float(gate.lci_weight[0]), float(gate.lci_weight[1]))
        self.assertEqual(float(gate.lci_weight[2]), 0.0)
        self.assertGreater(float(gate.scorer_weight[0]), 0.0)

    def test_lci_weighted_loss_uses_gate_and_weight(self) -> None:
        cfg = for_test_default_step5_innovation_config()
        gate = build_step5a_scorer_gate(_batch(), cfg)
        bundle = lci_score_invariance_loss(
            factual_score=torch.tensor([4.0, 3.0, 5.0]),
            cf_score=torch.tensor([4.2, 3.4, 1.0]),
            robust_score=torch.tensor([4.1, 3.1, 5.5]),
            target_rating=torch.tensor([4.0, 3.0, 5.0]),
            gate=gate,
            cfg=cfg,
        )
        self.assertGreater(float(bundle.lci_loss), 0.0)
        self.assertAlmostEqual(
            float(bundle.lci_weighted_loss),
            float(bundle.lci_loss) * float(cfg.lci.weight),
            places=6,
        )
        self.assertGreater(float(bundle.uci_weight_mean), 0.0)


if __name__ == "__main__":
    unittest.main()
