from __future__ import annotations

import json
import os
import unittest
from dataclasses import asdict
from types import SimpleNamespace

import torch

from executors.step5_engine import (
    _assert_step5a_no_big_model,
    _head_gated_step5_loss_terms,
    build_step5A_small_scorer_model,
    build_step5B_large_explainer_model,
    normalize_step5_task_head,
)
from odcr_core.step5_innovation import for_test_default_step5_innovation_config


def _tiny_step5_cfg(**overrides):
    base = {
        "nuser": 3,
        "nitem": 4,
        "ntoken": 11,
        "emsize": 8,
        "nhead": 2,
        "nhid": 16,
        "nlayers": 1,
        "dropout": 0.0,
        "label_smoothing": 0.0,
        "step5_innovation_config_json": json.dumps(
            asdict(for_test_default_step5_innovation_config()),
            sort_keys=True,
        ),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _tiny_profiles(cfg):
    return (
        torch.zeros(2, cfg.emsize),
        torch.zeros(2, cfg.emsize),
        torch.zeros(cfg.nuser, cfg.emsize),
        torch.zeros(cfg.nuser, cfg.emsize),
        torch.zeros(cfg.nitem, cfg.emsize),
        torch.zeros(cfg.nitem, cfg.emsize),
    )


class Step5HeadSplitEngineTest(unittest.TestCase):
    def test_head_gate_keeps_only_step5a_losses_for_step5a(self) -> None:
        base = torch.tensor(2.0, requires_grad=True)
        factual, counter, lci, fca, raw_lci, raw_fca = _head_gated_step5_loss_terms(
            task_head="step5A",
            zero_like=base,
            loss_factual=base + 1.0,
            loss_counterfactual=base + 2.0,
            lci_weighted_loss=base + 3.0,
            fca_weighted_loss=base + 4.0,
            l_lci=base + 5.0,
            l_fca=base + 6.0,
        )
        self.assertGreater(float(factual.detach()), 0.0)
        self.assertEqual(float(counter.detach()), 0.0)
        self.assertGreater(float(lci.detach()), 0.0)
        self.assertEqual(float(fca.detach()), 0.0)
        self.assertGreater(float(raw_lci.detach()), 0.0)
        self.assertEqual(float(raw_fca.detach()), 0.0)
        (factual + counter + lci + fca).backward()
        self.assertIsNotNone(base.grad)

    def test_head_gate_keeps_only_step5b_losses_for_step5b(self) -> None:
        base = torch.tensor(2.0, requires_grad=True)
        factual, counter, lci, fca, raw_lci, raw_fca = _head_gated_step5_loss_terms(
            task_head="step5B",
            zero_like=base,
            loss_factual=base + 1.0,
            loss_counterfactual=base + 2.0,
            lci_weighted_loss=base + 3.0,
            fca_weighted_loss=base + 4.0,
            l_lci=base + 5.0,
            l_fca=base + 6.0,
        )
        self.assertEqual(float(factual.detach()), 0.0)
        self.assertGreater(float(counter.detach()), 0.0)
        self.assertEqual(float(lci.detach()), 0.0)
        self.assertGreater(float(fca.detach()), 0.0)
        self.assertEqual(float(raw_lci.detach()), 0.0)
        self.assertGreater(float(raw_fca.detach()), 0.0)
        (factual + counter + lci + fca).backward()
        self.assertIsNotNone(base.grad)

    def test_combined_preserves_both_heads_and_invalid_head_fails(self) -> None:
        base = torch.tensor(2.0, requires_grad=True)
        factual, counter, lci, fca, raw_lci, raw_fca = _head_gated_step5_loss_terms(
            task_head="combined",
            zero_like=base,
            loss_factual=base + 1.0,
            loss_counterfactual=base + 2.0,
            lci_weighted_loss=base + 3.0,
            fca_weighted_loss=base + 4.0,
            l_lci=base + 5.0,
            l_fca=base + 6.0,
        )
        for tensor in (factual, counter, lci, fca, raw_lci, raw_fca):
            self.assertGreater(float(tensor.detach()), 0.0)
        with self.assertRaisesRegex(RuntimeError, "invalid Step5 task head"):
            normalize_step5_task_head("runtime_probe")

    def test_step5a_small_scorer_factory_has_no_big_model_or_word_logits(self) -> None:
        cfg = _tiny_step5_cfg(ntoken=1)
        model = build_step5A_small_scorer_model(cfg, profile_tensors=_tiny_profiles(cfg))
        preflight = _assert_step5a_no_big_model(model)
        self.assertFalse(preflight["step5A_uses_big_model"])
        self.assertFalse(hasattr(model, "flan_explainer"))
        self.assertIsNone(model.word_embeddings)
        forbidden = [
            name
            for name, _module in model.named_modules()
            if any(token in name.lower() for token in ("flan", "t5", "explainer", "decoder"))
        ]
        self.assertEqual(forbidden, [])
        rating, context_dist, word_dist = model(
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.zeros(2, 1, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            target_tokens=None,
            return_explainer_logits=False,
            scorer_only=True,
        )
        self.assertEqual(tuple(rating.shape), (2,))
        self.assertIsNone(context_dist)
        self.assertIsNone(word_dist)
        self.assertFalse(model._last_step5_forward_contract["flan_explainer_called"])
        self.assertFalse(model._last_step5_forward_contract["word_dist_returned"])

    def test_step5b_large_factory_keeps_explainer_with_stub(self) -> None:
        cfg = _tiny_step5_cfg()
        old = os.environ.get("ODCR_STEP5_INIT_FLAN_STUB")
        os.environ["ODCR_STEP5_INIT_FLAN_STUB"] = "1"
        try:
            model = build_step5B_large_explainer_model(cfg, profile_tensors=_tiny_profiles(cfg))
        finally:
            if old is None:
                os.environ.pop("ODCR_STEP5_INIT_FLAN_STUB", None)
            else:
                os.environ["ODCR_STEP5_INIT_FLAN_STUB"] = old
        self.assertTrue(hasattr(model, "flan_explainer"))
        self.assertIsNotNone(model.word_embeddings)
        self.assertTrue(model.step5_big_model_enabled)


if __name__ == "__main__":
    unittest.main()
