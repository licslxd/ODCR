"""Step3 structured disentanglement: synthetic stability smoke and anti-collapse checks."""
from __future__ import annotations

import math
import json
import os
import sys
import unittest

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from executors.step3_train_core import (  # noqa: E402
    Model,
    STEP3_TOTAL_LOSS_COMPONENT_KEYS,
    Step3ForwardOutput,
    compose_step3_loss_from_forward_output,
    duplicate_step3_loss_check,
    step3_global_finite_decision_from_local,
    step3_sync_loss_bundle_finite_status,
    summarize_step3_profile_buffers,
    validate_step3_graph_safety_preflight,
)
from odcr_core.csb_contract import (  # noqa: E402
    CSB_FORWARD_OUTPUT_SCHEMA_VERSION,
    csb_contract_hash,
    default_csb_contract_payload,
    validate_csb_forward_output_schema,
)
from odcr_core.odcr_losses import (  # noqa: E402
    build_orthogonal_losses,
    shared_invariance_loss,
    specific_separation_loss,
    variance_floor_loss,
)


class _Cfg:
    step3_structured_loss_weights_json = json.dumps(
        {
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
        },
        sort_keys=True,
    )
    step3_loss_semantics_json = json.dumps(
        {
            "specific_separation_margin": 0.6,
            "variance_target_std": 0.7,
            "variance_eps": 1e-4,
            "orthogonal_eps": 1e-8,
            "cosine_eps": 1e-8,
            "sample_weight_eps": 1e-6,
            "prototype_separation_eps": 1e-8,
            "quality_weight": {
                "evidence_base": 0.55,
                "evidence_scale": 0.45,
                "anchor_base": 0.40,
                "anchor_scale": 0.60,
            },
        },
        sort_keys=True,
    )
    csb_odcr_config_json = json.dumps(
        {
            "enabled": True,
            "controlled_injection": {"enabled": True, "gate_init": 0.35, "rating_safe_injection": True},
            "conflict_routing": {
                "enabled": True,
                "mode": "rating_anchor_projection",
                "rating_anchor": "L_rating_shared",
                "explanation_anchor": "L_light_explainer",
                "diversity_guard": ["DIST-1", "DIST-2"],
                "aux_soft_cap": 0.75,
                "dynamic_downweight": True,
            },
        },
        sort_keys=True,
    )


class _Batch:
    pass


def _build_model() -> Model:
    torch.manual_seed(7)
    nuser, nitem, ntoken, d = 16, 20, 100, 32
    uc = torch.randn(nuser, d)
    us = torch.randn(nuser, d)
    ic = torch.randn(nitem, d)
    ist = torch.randn(nitem, d)
    dc = torch.randn(2, d)
    ds = torch.randn(2, d)
    model = Model(
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
    contract = default_csb_contract_payload()
    contract["contract_hash"] = csb_contract_hash(contract)
    model.csb_odcr_bottleneck.set_csb_contract_payload(contract)
    return model


def _synthetic_forward_batch(model: Model) -> tuple[Step3ForwardOutput, _Batch]:
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

    out = model(
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
    self_batch = _Batch()
    self_batch.rating = rating
    self_batch.tgt_output = tgt
    self_batch.domain_idx = domain
    self_batch.content_anchor_score = content_anchor
    self_batch.style_anchor_score = style_anchor
    self_batch.evidence_quality_prior = evidence_quality
    return out, self_batch


def _synthetic_loss_bundle(model: Model) -> tuple[torch.Tensor, dict[str, float]]:
    out, self_batch = _synthetic_forward_batch(model)
    bundle = compose_step3_loss_from_forward_output(
        forward_output=out,
        batch=self_batch,
        final_cfg=_Cfg(),
    )
    validate_step3_graph_safety_preflight(
        forward_output=out,
        loss_bundle=bundle,
        underlying_model=model,
        ctx="unit",
    )
    loss = bundle.total_loss
    stats = {
        "loss_total": float(loss.detach().item()),
        "ortho_xcov": float(bundle.diagnostics["L_orthogonal_xcov"].detach().item()),
        "var_total": float(bundle.components["L_variance"].detach().item()),
        "shared_std_mean": float(bundle.diagnostics["shared_std_mean"].detach().item()),
        "specific_std_mean": float(bundle.diagnostics["specific_std_mean"].detach().item()),
    }
    return loss, stats


class TestStep3StructuredStability(unittest.TestCase):
    def test_forward_returns_step3_output_and_detached_debug_latents(self) -> None:
        model = _build_model()
        bsz, seq_len = 4, 6
        out = model(
            torch.randint(0, 16, (bsz,)),
            torch.randint(0, 20, (bsz,)),
            torch.randint(1, 100, (bsz, seq_len)),
            torch.randint(0, 2, (bsz,)),
            content_anchor=torch.rand(bsz),
            style_anchor=torch.rand(bsz),
            content_evidence_ids=torch.randint(0, 100, (bsz, 24)),
            style_evidence_ids=torch.randint(0, 100, (bsz, 24)),
            domain_style_anchor_ids=torch.randint(0, 100, (bsz, 24)),
            local_style_hint_ids=torch.randint(0, 100, (bsz, 24)),
            polarity_ids=torch.randint(0, 3, (bsz,)),
            evidence_quality_prior=torch.rand(bsz),
        )
        self.assertIsInstance(out, Step3ForwardOutput)
        validate_csb_forward_output_schema(out)
        self.assertEqual(out.csb_schema_version, CSB_FORWARD_OUTPUT_SCHEMA_VERSION)
        for key in ("z_content", "z_style", "z_domain", "z_uncertainty"):
            self.assertIn(key, out.structured_loss_inputs)
            self.assertEqual(tuple(getattr(out, key).shape), (bsz, model.emsize))
        self.assertEqual(out.csb_packet["method_name"], "CSB-ODCR")
        self.assertFalse(out.csb_diagnostics["controlled_injection_enabled"])
        for key in ("shared_prototype", "domain_style_proto", "shared_latent", "specific_latent"):
            self.assertIn(key, out.structured_loss_inputs)
            self.assertTrue(out.structured_loss_inputs[key].requires_grad)
        self.assertIsNotNone(model.last_csb_latents)
        self.assertFalse(model.last_csb_latents.shared_prototype.requires_grad)
        self.assertFalse(model.last_shared_proj.requires_grad)

    def test_profile_domain_artifacts_are_frozen_buffers(self) -> None:
        model = _build_model()
        summary = summarize_step3_profile_buffers(model)
        self.assertTrue(summary["profile_domain_requires_grad_false"])
        self.assertTrue(summary["profile_domain_not_trainable_parameters"])
        self.assertGreater(summary["profile_domain_memory_bytes"], 0)
        params = dict(model.named_parameters())
        for name in (
            "domain_content_profiles",
            "domain_style_profiles",
            "user_content_profiles",
            "user_style_profiles",
            "item_content_profiles",
            "item_style_profiles",
        ):
            self.assertNotIn(name, params)
            self.assertIn(name, dict(model.named_buffers()))

    def test_duplicate_loss_check_detects_semantic_duplicates(self) -> None:
        clean = duplicate_step3_loss_check(STEP3_TOTAL_LOSS_COMPONENT_KEYS)
        self.assertEqual(clean["status"], "unique_semantic_components")
        dup = duplicate_step3_loss_check(["L_rating_shared", "L_rating_shared"])
        self.assertEqual(dup["duplicates"], ["L_rating_shared"])

    def test_global_finite_decision_helper_is_all_rank_min(self) -> None:
        self.assertTrue(step3_global_finite_decision_from_local(True, world_size=1))
        self.assertTrue(
            step3_global_finite_decision_from_local(
                True,
                world_size=2,
                reduce_min=lambda local: min(local, 1),
            )
        )

    def test_loss_bundle_finite_status_has_single_vector_sync_contract(self) -> None:
        model = _build_model()
        out, batch = _synthetic_forward_batch(model)
        bundle = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=_Cfg())
        sync = step3_sync_loss_bundle_finite_status(bundle, world_size=1)
        self.assertTrue(sync["global_total_finite"])
        self.assertEqual(sync["sync_method"], "single_all_reduce_min_vector")
        self.assertEqual(set(sync["global_component_finite_status"]), set(STEP3_TOTAL_LOSS_COMPONENT_KEYS))
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
        class _CfgChanged(_Cfg):
            _weights = json.loads(_Cfg.step3_structured_loss_weights_json)
            _weights["content_alignment_weight"] = 0.50
            step3_structured_loss_weights_json = json.dumps(_weights, sort_keys=True)

        model = _build_model()
        out, batch = _synthetic_forward_batch(model)
        base = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=_Cfg())
        changed = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=_CfgChanged())
        self.assertAlmostEqual(float((changed.total_loss - base.total_loss).detach().item()), 0.0, places=6)
        self.assertAlmostEqual(
            float((changed.sidecar_loss - base.sidecar_loss).detach().item()),
            float(
                (changed.weights["L_content_alignment"] - base.weights["L_content_alignment"])
                * base.components["L_content_alignment"].detach().item()
            ),
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

    def test_csb_specific_branch_is_sidecar_loss_only(self) -> None:
        model = _build_model()
        out, batch = _synthetic_forward_batch(model)
        bundle = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=_Cfg())
        bundle.sidecar_loss.backward()

        params = dict(model.named_parameters())
        for name in (
            "csb_odcr_bottleneck.csb_content_head.0.weight",
            "csb_odcr_bottleneck.csb_style_head.0.weight",
            "csb_odcr_bottleneck.shared_projector.0.weight",
            "csb_odcr_bottleneck.anchor_head_content.weight",
        ):
            with self.subTest(name=name):
                grad = params[name].grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all().item())

    def test_variance_floor_penalizes_collapsed_latents(self) -> None:
        shared = torch.zeros(4, 8)
        specific = torch.ones(4, 8)
        var = variance_floor_loss(shared, specific, target_std=0.7)
        self.assertGreater(float(var.loss_var_total.item()), 1.0)
        self.assertLess(float(var.shared_std_mean.item()), 0.1)
        self.assertLess(float(var.specific_std_mean.item()), 0.1)


if __name__ == "__main__":
    unittest.main()
