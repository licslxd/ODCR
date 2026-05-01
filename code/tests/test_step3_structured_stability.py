"""Step3 structured disentanglement: synthetic stability smoke and anti-collapse checks."""
from __future__ import annotations

import math
import os
import sys
import unittest

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from executors.step3_train_core import (  # noqa: E402
    Model,
    compose_step3_structured_loss,
    parse_step3_structured_loss_weights,
    step3_global_finite_decision_from_local,
)
from odcr_core.odcr_losses import (  # noqa: E402
    anchor_score_alignment_loss,
    build_orthogonal_losses,
    cosine_pull_loss,
    domain_style_prototype_separation,
    residual_l2_penalty,
    shared_invariance_loss,
    shared_prototype_pull_loss,
    specific_separation_loss,
    variance_floor_loss,
)


def _build_model() -> Model:
    torch.manual_seed(7)
    nuser, nitem, ntoken, d = 16, 20, 100, 32
    uc = torch.randn(nuser, d)
    us = torch.randn(nuser, d)
    ic = torch.randn(nitem, d)
    ist = torch.randn(nitem, d)
    dc = torch.randn(2, d)
    ds = torch.randn(2, d)
    return Model(
        nuser=nuser,
        nitem=nitem,
        ntoken=ntoken,
        emsize=d,
        nhead=2,
        nhid=64,
        nlayers=1,
        dropout=0.1,
        user_content_profiles=uc,
        user_style_profiles=us,
        item_content_profiles=ic,
        item_style_profiles=ist,
        domain_content_profiles=dc,
        domain_style_profiles=ds,
    )


def _synthetic_loss_bundle(model: Model) -> tuple[torch.Tensor, dict[str, float]]:
    torch.manual_seed(17)
    bsz, seq_len = 8, 12
    user = torch.randint(0, 16, (bsz,))
    item = torch.randint(0, 20, (bsz,))
    domain = torch.randint(0, 2, (bsz,))
    tgt = torch.randint(1, 100, (bsz, seq_len))
    rating = torch.randn(bsz)
    content_anchor = torch.rand(bsz)
    style_anchor = torch.rand(bsz)
    content_ids = torch.randint(0, 100, (bsz, 24))
    style_ids = torch.randint(0, 100, (bsz, 24))
    domain_style_ids = torch.randint(0, 100, (bsz, 24))
    local_style_ids = torch.randint(0, 100, (bsz, 24))
    polarity_ids = torch.randint(0, 3, (bsz,))
    evidence_quality = torch.rand(bsz)

    pred_rating, _, word_dist, _, _ = model(
        user,
        item,
        tgt,
        domain,
        content_anchor=content_anchor,
        style_anchor=style_anchor,
        content_evidence_ids=content_ids,
        style_evidence_ids=style_ids,
        domain_style_anchor_ids=domain_style_ids,
        local_style_hint_ids=local_style_ids,
        polarity_ids=polarity_ids,
        evidence_quality_prior=evidence_quality,
    )
    lat = model.last_odcr_latents
    q_w = (0.55 + 0.45 * evidence_quality.clamp(0.0, 1.0)).detach()
    q_content = q_w * (0.40 + 0.60 * content_anchor.clamp(0.0, 1.0))
    q_style = q_w * (0.40 + 0.60 * style_anchor.clamp(0.0, 1.0))

    l_anchor_sh = anchor_score_alignment_loss(lat.anchor_pred_content, content_anchor, sample_weight=q_content)
    l_anchor_sp = anchor_score_alignment_loss(lat.anchor_pred_style, style_anchor, sample_weight=q_style)
    l_content_align = cosine_pull_loss(lat.shared_latent, lat.content_evidence_target, sample_weight=q_content)
    l_style_align = cosine_pull_loss(lat.specific_latent, lat.style_evidence_target, sample_weight=q_style)
    l_shared_proto = shared_prototype_pull_loss(lat.shared_latent, lat.shared_prototype, sample_weight=q_content)
    l_domain_style_align = cosine_pull_loss(lat.domain_style_component, lat.domain_style_target, sample_weight=q_style)
    l_local_style_align = cosine_pull_loss(lat.residual_local, lat.local_style_target, sample_weight=q_style)
    l_polarity_align = cosine_pull_loss(lat.specific_latent, lat.polarity_target, sample_weight=q_style)
    l_residual = residual_l2_penalty(lat.residual_local)
    l_proto = domain_style_prototype_separation(model.odcr_disentangler.domain_style_proto.weight)
    l_rating = model.rating_loss_fn(pred_rating, rating)
    l_explainer = 0.15 * model.exp_loss_fn(word_dist.view(-1, model.ntoken), tgt.reshape(-1))
    orth = build_orthogonal_losses(lat.shared_latent, lat.specific_latent)
    var = variance_floor_loss(lat.shared_latent, lat.specific_latent)
    l_shared_inv = shared_invariance_loss(lat.shared_latent, domain)
    l_specific_sep = specific_separation_loss(lat.specific_latent, domain)

    loss = (
        l_rating
        + 0.2 * l_explainer
        + 0.2 * orth.loss_ortho_total
        + 0.10 * var.loss_var_total
        + 0.18 * l_shared_inv
        + 0.16 * l_specific_sep
        + 0.08 * l_anchor_sh
        + 0.08 * l_anchor_sp
        + 0.12 * l_content_align
        + 0.12 * l_style_align
        + 0.08 * l_shared_proto
        + 0.06 * l_domain_style_align
        + 0.06 * l_local_style_align
        + 0.05 * l_polarity_align
        + 0.025 * l_residual
        + 0.04 * l_proto
    )
    stats = {
        "loss_total": float(loss.detach().item()),
        "ortho_xcov": float(orth.loss_ortho_xcov.detach().item()),
        "var_total": float(var.loss_var_total.detach().item()),
        "shared_std_mean": float(var.shared_std_mean.detach().item()),
        "specific_std_mean": float(var.specific_std_mean.detach().item()),
    }
    return loss, stats


class TestStep3StructuredStability(unittest.TestCase):
    def test_global_finite_decision_helper_is_all_rank_min(self) -> None:
        self.assertTrue(step3_global_finite_decision_from_local(True, world_size=1))
        self.assertTrue(
            step3_global_finite_decision_from_local(
                True,
                world_size=2,
                reduce_min=lambda local: min(local, 1),
            )
        )
        self.assertFalse(
            step3_global_finite_decision_from_local(
                True,
                world_size=2,
                reduce_min=lambda local: min(local, 0),
            )
        )
        self.assertFalse(
            step3_global_finite_decision_from_local(
                False,
                world_size=2,
                reduce_min=lambda local: min(local, 1),
            )
        )

    def test_single_domain_auxiliary_zeros_are_graph_tied(self) -> None:
        shared = torch.randn(4, 8, requires_grad=True)
        specific = torch.randn(4, 8, requires_grad=True)
        domain = torch.zeros(4, dtype=torch.long)
        l_shared = shared_invariance_loss(shared, domain)
        l_specific = specific_separation_loss(specific, domain)
        self.assertTrue(l_shared.requires_grad)
        self.assertTrue(l_specific.requires_grad)
        (l_shared + l_specific).backward()
        self.assertIsNotNone(shared.grad)
        self.assertIsNotNone(specific.grad)
        self.assertTrue(torch.equal(shared.grad, torch.zeros_like(shared.grad)))
        self.assertTrue(torch.equal(specific.grad, torch.zeros_like(specific.grad)))

    def test_structured_loss_weight_config_changes_weighted_loss(self) -> None:
        base = {
            "orthogonal": {"weight": 0.20, "xcov_weight": 1.0, "cosine_weight": 0.25},
            "variance_weight": 0.10,
            "shared_invariance_weight": 0.18,
            "specific_separation_weight": 0.16,
            "anchor_alignment_weight": 0.08,
            "content_alignment_weight": 0.12,
            "style_alignment_weight": 0.12,
            "shared_prototype_weight": 0.08,
            "domain_style_alignment_weight": 0.06,
            "local_style_alignment_weight": 0.06,
            "polarity_alignment_weight": 0.05,
            "residual_specific_weight": 0.025,
            "prototype_separation_weight": 0.04,
            "light_explainer_weight": 0.03,
        }
        changed = {**base, "content_alignment_weight": 0.50}
        terms = {
            "rating_shared": torch.tensor(1.0),
            "light_explainer": torch.tensor(2.0),
            "orthogonal_total": torch.tensor(3.0),
            "variance_total": torch.tensor(4.0),
            "shared_invariance": torch.tensor(5.0),
            "specific_separation": torch.tensor(6.0),
            "anchor_shared": torch.tensor(7.0),
            "anchor_specific": torch.tensor(8.0),
            "content_alignment": torch.tensor(9.0),
            "style_alignment": torch.tensor(10.0),
            "shared_prototype": torch.tensor(11.0),
            "domain_style_alignment": torch.tensor(12.0),
            "local_style_alignment": torch.tensor(13.0),
            "polarity_alignment": torch.tensor(14.0),
            "residual_specific": torch.tensor(15.0),
            "prototype_separation": torch.tensor(16.0),
        }
        loss_base = compose_step3_structured_loss(
            weights=parse_step3_structured_loss_weights(base),
            **terms,
        )
        loss_changed = compose_step3_structured_loss(
            weights=parse_step3_structured_loss_weights(changed),
            **terms,
        )
        self.assertAlmostEqual(
            float(loss_changed - loss_base),
            float((0.50 - 0.12) * terms["content_alignment"]),
            places=6,
        )

    def test_synthetic_forward_backward_is_finite_and_bounded(self) -> None:
        model = _build_model()
        loss, stats = _synthetic_loss_bundle(model)
        loss.backward()

        grad_norms = [
            float(param.grad.norm().item())
            for _, param in model.named_parameters()
            if param.grad is not None
        ]
        self.assertTrue(torch.isfinite(loss).item())
        self.assertLess(stats["loss_total"], 5.0)
        self.assertLess(stats["ortho_xcov"], 1.0)
        self.assertGreater(stats["shared_std_mean"], 0.4)
        self.assertGreater(stats["specific_std_mean"], 0.4)
        self.assertTrue(all(math.isfinite(v) for v in grad_norms))
        self.assertLess(max(grad_norms), 10.0)

    def test_variance_floor_penalizes_collapsed_latents(self) -> None:
        shared = torch.zeros(4, 8)
        specific = torch.ones(4, 8)
        var = variance_floor_loss(shared, specific, target_std=0.7)
        self.assertGreater(float(var.loss_var_total.item()), 1.0)
        self.assertLess(float(var.shared_std_mean.item()), 0.1)
        self.assertLess(float(var.specific_std_mean.item()), 0.1)


if __name__ == "__main__":
    unittest.main()
