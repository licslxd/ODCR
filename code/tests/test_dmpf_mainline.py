"""DMPF 主线：gate warm-start / fusion mode / step4 decode 接线 / manifest 字段。"""
import os
import sys
import unittest
import json
from dataclasses import asdict

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)
os.environ["ODCR_STEP5_INIT_FLAN_STUB"] = "1"

from base_utils import T5_shift_right  # noqa: E402
from executors.step5_engine import Model as Step5Model  # noqa: E402
from odcr_core.config_resolver import needs_decode_layer  # noqa: E402
from odcr_core.step5_innovation import (  # noqa: E402
    CCVControlPacket,
    STEP5_EVIDENCE_FEATURE_DIM,
    for_test_default_step5_innovation_config,
)


def _ccv_packet(batch_size: int, *, device: torch.device | str = "cpu") -> CCVControlPacket:
    ids = torch.ones(batch_size, 2, dtype=torch.long, device=device)
    ones = torch.ones(batch_size, dtype=torch.float32, device=device)
    zeros = torch.zeros(batch_size, dtype=torch.float32, device=device)
    return CCVControlPacket(
        content_evidence_ids=ids,
        style_evidence_ids=ids,
        domain_style_anchor_ids=ids,
        local_style_hint_ids=ids,
        polarity_ids=ids,
        route_scorer_mask=ones,
        route_explainer_mask=ones,
        sample_weight_hint=ones,
        cf_reliability_score=ones,
        content_retention_score=ones,
        style_shift_score=zeros,
        rating_stability_score=ones,
        uncertainty_score=zeros,
        confidence_bucket=ones * 2.0,
        evidence_quality_prior=ones,
        content_anchor_score=ones,
        style_anchor_score=ones,
    )


def _build_step5_model() -> Step5Model:
    torch.manual_seed(7)
    nuser, nitem, ntoken, d = 8, 8, 32, 16
    uc = torch.randn(nuser, d)
    us = torch.randn(nuser, d)
    ic = torch.randn(nitem, d)
    ist = torch.randn(nitem, d)
    dc = torch.randn(2, d)
    ds = torch.randn(2, d)
    return Step5Model(
        nuser=nuser,
        nitem=nitem,
        ntoken=ntoken,
        emsize=d,
        nhead=2,
        nhid=32,
        nlayers=1,
        dropout=0.1,
        user_content_profiles=uc,
        user_style_profiles=us,
        item_content_profiles=ic,
        item_style_profiles=ist,
        domain_content_profiles=dc,
        domain_style_profiles=ds,
        step5_innovation_config_json=json.dumps(asdict(for_test_default_step5_innovation_config())),
    )


class TestDmpfMainline(unittest.TestCase):
    def test_identity_warm_start_gate_and_modulation(self) -> None:
        m = _build_step5_model()
        u = torch.tensor([1, 2], dtype=torch.long)
        i = torch.tensor([3, 4], dtype=torch.long)
        d = torch.tensor([0, 1], dtype=torch.long)
        domain_raw = m.domain_content_profiles[d].unsqueeze(1)
        gate = m._compute_domain_gate(domain_raw)
        self.assertTrue(torch.allclose(gate, torch.ones_like(gate), atol=1e-6, rtol=0.0))
        ctx = m._build_context_tokens(u, i)
        mod = m._apply_domain_modulation(ctx, gate)
        self.assertTrue(torch.allclose(mod, ctx, atol=1e-6, rtol=0.0))

    def test_fusion_modes_forward_shapes(self) -> None:
        m = _build_step5_model()
        u = torch.tensor([1, 2], dtype=torch.long)
        i = torch.tensor([3, 4], dtype=torch.long)
        d = torch.tensor([0, 1], dtype=torch.long)
        tgt_out = torch.randint(1, 31, (2, 5), dtype=torch.long)
        tgt_in = T5_shift_right(tgt_out)
        for mode in ("cross_attn_only", "gate_only", "gate_cross_attn"):
            m.domain_fusion_mode = mode
            rating, context_dist, word_dist = m(
                u,
                i,
                tgt_in,
                d,
                target_tokens=tgt_out,
                evidence_features=torch.zeros(2, STEP5_EVIDENCE_FEATURE_DIM),
                ccv_control_packet=_ccv_packet(2),
            )
            self.assertEqual(tuple(rating.shape), (2,))
            self.assertEqual(tuple(context_dist.shape), (2, 32))
            self.assertEqual(tuple(word_dist.shape), (2, 5, 32))

    def test_step4_is_decode_layer(self) -> None:
        self.assertTrue(needs_decode_layer("step4", step5_train_only=False))

    def test_manifest_includes_domain_fusion_mode(self) -> None:
        p = os.path.join(_CODE_DIR, "odcr_core", "manifests.py")
        with open(p, "r", encoding="utf-8") as fh:
            txt = fh.read()
        self.assertIn('"domain_fusion_mode"', txt)


if __name__ == "__main__":
    unittest.main()
