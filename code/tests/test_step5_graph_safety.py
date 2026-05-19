from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import torch

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = Path(_CODE_DIR).parent
sys.path.insert(0, _CODE_DIR)

import odcr_core.config_resolver as config_resolver  # noqa: E402
from odcr_core.config_resolver import OneControlConfigError, resolve_config  # noqa: E402
from odcr_core.gather_schema import GatheredBatch  # noqa: E402
from odcr_core.manifests import build_formal_source_table_snapshot  # noqa: E402
from odcr_core.step5_innovation import (  # noqa: E402
    STEP5_EVIDENCE_FEATURE_DIM,
    build_ccv_control_packet,
    build_step5a_scorer_gate,
    build_step5b_explainer_gate,
    evidence_basis_fca_loss,
    for_test_default_step5_innovation_config,
    lci_score_invariance_loss,
)
from odcr_core.step5_word_losses import route_weighted_mean  # noqa: E402
from executors.step5_engine import compose_step5_total_loss  # noqa: E402
from helpers.fixtures import write_step4_upstream_fixture  # noqa: E402


def _batch(route_scorer: torch.Tensor, route_explainer: torch.Tensor, domain: torch.Tensor) -> GatheredBatch:
    bsz = int(route_scorer.numel())
    ev = torch.zeros(bsz, STEP5_EVIDENCE_FEATURE_DIM)
    ev[:, 0] = torch.tensor([0.7, 0.8, 0.6, 0.9])[:bsz]
    ev[:, 1] = torch.tensor([0.95, 0.85, 0.75, 0.90])[:bsz]
    ev[:, 2] = torch.tensor([0.2, 0.7, 0.3, 0.8])[:bsz]
    ev[:, 3] = torch.tensor([0.95, 0.85, 0.70, 0.90])[:bsz]
    ev[:, 4] = torch.tensor([0.95, 0.80, 0.70, 0.90])[:bsz]
    ev[:, 5] = torch.ones(bsz)
    ev[:, 6] = torch.tensor([0.05, 0.15, 0.35, 0.10])[:bsz]
    ev[:, 7] = torch.ones(bsz)
    ids = torch.arange(1, bsz * 3 + 1, dtype=torch.long).view(bsz, 3)
    return GatheredBatch(
        user_idx=torch.arange(bsz),
        item_idx=torch.arange(bsz),
        rating=torch.tensor([4.0, 3.0, 2.0, 5.0])[:bsz],
        tgt_input=ids,
        tgt_output=ids,
        domain_idx=domain.long(),
        sample_id=torch.arange(bsz),
        exp_sample_weight=torch.ones(bsz),
        route_scorer_mask=route_scorer.float(),
        route_explainer_mask=route_explainer.float(),
        uncertainty_score=ev[:, 6],
        confidence_bucket=torch.tensor([2.0, 1.0, 0.0, 2.0])[:bsz],
        content_anchor_score=torch.tensor([0.9, 0.8, 0.6, 0.85])[:bsz],
        style_anchor_score=torch.tensor([0.2, 0.7, 0.3, 0.8])[:bsz],
        evidence_features=ev,
        content_evidence_ids=ids,
        style_evidence_ids=ids,
        domain_style_anchor_ids=ids,
        local_style_hint_ids=ids,
        polarity_ids=torch.tensor([[2], [1], [0], [2]])[:bsz],
    )


class TestStep5GraphSafety(unittest.TestCase):
    def _run_route_case(self, route_scorer: list[float], route_explainer: list[float], domain: list[int]) -> None:
        cfg = for_test_default_step5_innovation_config()
        batch = _batch(torch.tensor(route_scorer), torch.tensor(route_explainer), torch.tensor(domain))
        gate_a = build_step5a_scorer_gate(batch, cfg)
        gate_b = build_step5b_explainer_gate(batch, cfg)
        packet = build_ccv_control_packet(batch, cfg)
        bsz = len(route_scorer)
        factual = torch.linspace(2.0, 5.0, bsz, requires_grad=True)
        cf_score = factual + 0.1
        robust = factual - 0.1
        scorer_h = torch.randn(bsz, 5, requires_grad=True)
        explainer_h = torch.randn(bsz, 5, requires_grad=True)
        shared = torch.randn(bsz, 5, requires_grad=True)
        content_profile = torch.randn(bsz, 5, requires_grad=True)
        content_evidence = torch.randn(bsz, 5, requires_grad=True)

        lci = lci_score_invariance_loss(
            factual_score=factual,
            cf_score=cf_score,
            robust_score=robust,
            target_rating=batch.rating,
            gate=gate_a,
            cfg=cfg,
        )
        fca = evidence_basis_fca_loss(
            scorer_hidden=scorer_h,
            explainer_hidden=explainer_h,
            shared_latent=shared,
            content_profile=content_profile,
            content_evidence_latent=content_evidence,
            packet=packet,
            gate=gate_b,
            cfg=cfg,
        )
        dom = batch.domain_idx.view(-1)
        loss_factual = route_weighted_mean((factual - batch.rating).pow(2), gate_a.scorer_weight, dom == 1)
        loss_counterfactual = route_weighted_mean(factual.pow(2), gate_b.explainer_weight, dom == 0)
        total = compose_step5_total_loss(
            loss_factual=loss_factual,
            loss_counterfactual=loss_counterfactual,
            loss_repeat_ul=factual.sum() * 0.0,
            loss_terminal_clean=factual.sum() * 0.0,
            loss_batch_diversity=factual.sum() * 0.0,
            repeat_ul_weight=0.0,
            terminal_clean_weight=0.0,
            batch_diversity_weight=0.0,
            lci_weighted_loss=lci.lci_weighted_loss,
            fca_weighted_loss=fca.fca_weighted_loss,
            ortho_keep_loss=factual.sum() * 0.0,
            ortho_keep_weight=0.0,
        )
        self.assertTrue(total.requires_grad)
        total.backward()
        self.assertIsNotNone(factual.grad)
        self.assertIsNotNone(scorer_h.grad)
        self.assertIsNotNone(explainer_h.grad)

    def test_empty_route_variants_keep_graph(self) -> None:
        cases = {
            "all_scorer": ([1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]),
            "all_explainer": ([0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]),
            "neither_route": ([0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]),
            "mixed_route": ([1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]),
        }
        for name, (rs, re_, dom) in cases.items():
            with self.subTest(name=name):
                self._run_route_case(rs, re_, dom)

    def test_route_weighted_mean_zero_denominator_is_safe(self) -> None:
        values = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        out = route_weighted_mean(values, torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float32))
        self.assertEqual(out.dtype, values.dtype)
        self.assertEqual(out.device, values.device)
        self.assertTrue(out.requires_grad)
        self.assertEqual(float(out.detach()), 0.0)
        out.backward()
        self.assertIsNotNone(values.grad)

    def test_find_unused_false_requires_real_sample_plan_preflight_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step4_upstream_fixture(repo, task_id=4, run_id="1")
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                cfg, _, snapshot = resolve_config(
                    config_path=_REPO_ROOT / "configs" / "odcr.yaml",
                    command="step5",
                    task_id=4,
                    set_overrides=[],
                    dry_run=True,
                    from_step4="1",
                    eval_profile="balanced_2gpu",
                    mode="train_only",
                )
            finally:
                config_resolver._REPO_ROOT = old_root
        self.assertFalse(cfg.ddp_find_unused_parameters)
        self.assertEqual(cfg.ddp_find_unused_false_preflight, "real_sample_plan_one_batch")
        self.assertFalse(cfg.ddp_static_graph)
        self.assertEqual(cfg.step5_gradient_checkpointing_reentrant_policy, "non_reentrant")
        self.assertFalse(snapshot["step5_ddp"]["ddp_find_unused_parameters"])
        self.assertFalse(snapshot["step5_ddp"]["ddp_static_graph"])
        self.assertEqual(
            snapshot["step5_memory_truth"]["gradient_checkpointing_reentrant_policy"],
            "non_reentrant",
        )
        source_table = build_formal_source_table_snapshot(snapshot)
        records = {row["key"]: row for row in source_table["records"]}
        self.assertEqual(records["step5_ddp_find_unused_parameters"]["value"], False)
        self.assertEqual(records["step5_ddp_static_graph"]["value"], False)
        self.assertEqual(records["step5_ddp_find_unused_false_preflight"]["value"], "real_sample_plan_one_batch")
        self.assertEqual(
            records["step5_ddp_find_unused_false_preflight"]["source"],
            "configs/odcr.yaml:step5.ddp.find_unused_false_preflight",
        )
        self.assertNotIn("synthetic_preflight_role", records)
        self.assertEqual(records["formal_preflight_uses_real_data"]["value"], True)
        control_fields = records["step5_ccv.control_fields"]["value"]
        self.assertIn("polarity_anchor", control_fields)
        self.assertIn("content_anchor_score", control_fields)
        self.assertIn("style_anchor_score", control_fields)
        self.assertIn("evidence_quality_prior", control_fields)
        self.assertEqual(
            records["step5_ccv.derived_control_input.polarity_ids"]["value"],
            "polarity_anchor -> Processor._control_text_to_ids -> CCVControlPacket.polarity_ids",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write_step4_upstream_fixture(repo, task_id=4, run_id="1")
            old_root = config_resolver._REPO_ROOT
            try:
                config_resolver._REPO_ROOT = repo
                with self.assertRaises(OneControlConfigError):
                    resolve_config(
                        config_path=_REPO_ROOT / "configs" / "odcr.yaml",
                        command="step5",
                        task_id=4,
                        set_overrides=[
                            "step5.ddp.find_unused_parameters=false",
                            "step5.ddp.find_unused_false_preflight=synthetic_one_batch",
                        ],
                        dry_run=True,
                        from_step4="1",
                        eval_profile="balanced_2gpu",
                        mode="train_only",
                    )
            finally:
                config_resolver._REPO_ROOT = old_root

    def test_step5_ddp_wrapper_uses_resolved_find_unused_and_static_graph(self) -> None:
        source = (_REPO_ROOT / "code" / "executors" / "step5_engine.py").read_text(encoding="utf-8")
        self.assertIn("find_unused_parameters=bool(final_cfg.ddp_find_unused_parameters)", source)
        self.assertIn("static_graph=bool(getattr(final_cfg, \"ddp_static_graph\", False))", source)

    def test_flan_forward_has_no_hf_labels_and_lci_fca_are_not_repeated(self) -> None:
        source = (_REPO_ROOT / "code" / "executors" / "step5_engine.py").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\blabels\s*=", source))
        total = compose_step5_total_loss(
            loss_factual=torch.tensor(1.0),
            loss_counterfactual=torch.tensor(2.0),
            loss_repeat_ul=torch.tensor(3.0),
            loss_terminal_clean=torch.tensor(5.0),
            loss_batch_diversity=torch.tensor(7.0),
            repeat_ul_weight=0.1,
            terminal_clean_weight=0.2,
            batch_diversity_weight=0.3,
            lci_weighted_loss=torch.tensor(11.0),
            fca_weighted_loss=torch.tensor(13.0),
            ortho_keep_loss=torch.tensor(17.0),
            ortho_keep_weight=0.4,
        )
        expected = 1.0 + 2.0 + 0.3 + 1.0 + 2.1 + 11.0 + 13.0 + 6.8
        self.assertAlmostEqual(float(total), expected, places=5)


if __name__ == "__main__":
    unittest.main()
