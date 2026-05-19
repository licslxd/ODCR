from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch
from torch import nn

_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _CODE_DIR)

from odcr_core.gather_schema import GatheredBatch  # noqa: E402
from odcr_core.step5_innovation import (  # noqa: E402
    STEP5_EVIDENCE_FEATURE_DIM,
    for_test_default_step5_innovation_config,
)
from executors.step5_engine import (  # noqa: E402
    _step5_collate_dynamic,
    run_step5_find_unused_parameters_preflight,
)


def _gathered_batch() -> GatheredBatch:
    bsz = 4
    tgt = torch.tensor([[1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 5, 0]], dtype=torch.long)
    ids = torch.tensor([[1, 2], [2, 0], [3, 0], [4, 0]], dtype=torch.long)
    ev = torch.zeros(bsz, STEP5_EVIDENCE_FEATURE_DIM)
    ev[:, 1] = torch.tensor([0.95, 0.85, 0.75, 0.9])
    ev[:, 2] = torch.tensor([0.2, 0.7, 0.3, 0.8])
    ev[:, 3] = torch.tensor([0.95, 0.85, 0.7, 0.9])
    ev[:, 4] = torch.tensor([0.95, 0.8, 0.7, 0.9])
    ev[:, 5] = 1.0
    ev[:, 6] = torch.tensor([0.05, 0.15, 0.3, 0.1])
    ev[:, 7] = 1.0
    return GatheredBatch(
        user_idx=torch.arange(bsz),
        item_idx=torch.arange(bsz),
        rating=torch.tensor([4.5, 3.0, 2.0, 5.0]),
        tgt_input=tgt,
        tgt_output=tgt,
        domain_idx=torch.tensor([1, 0, 1, 0], dtype=torch.long),
        sample_id=torch.arange(bsz),
        exp_sample_weight=torch.ones(bsz),
        route_scorer_mask=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        route_explainer_mask=torch.tensor([0.0, 1.0, 0.0, 1.0]),
        entropy_score=torch.zeros(bsz),
        uncertainty_score=ev[:, 6],
        confidence_bucket=torch.tensor([2.0, 2.0, 0.0, 1.0]),
        content_anchor_score=torch.tensor([0.9, 0.8, 0.6, 0.85]),
        style_anchor_score=torch.tensor([0.2, 0.75, 0.35, 0.8]),
        evidence_features=ev,
        content_evidence_ids=ids,
        style_evidence_ids=ids,
        domain_style_anchor_ids=ids,
        local_style_hint_ids=ids,
        polarity_ids=torch.tensor([[2], [1], [0], [2]], dtype=torch.long),
    )


class _TinyStep5Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.ntoken = 8
        self.user_content_profiles = torch.ones(16, 3)

    def gather(self, batch, device, *, non_blocking_h2d: bool | None = None):
        return batch

    def odcr_scorer(self, shared, profile, specific):
        return (shared + profile + specific).mean(dim=1) * self.scale

    def forward(
        self,
        user_idx,
        item_idx,
        tgt_input,
        domain_idx,
        *,
        target_tokens,
        evidence_features,
        content_anchor_score,
        style_anchor_score,
        ccv_control_packet,
    ):
        bsz = int(user_idx.size(0))
        latent = torch.ones(bsz, 3) * self.scale
        latent = latent + evidence_features[:, :3] * 0.01
        numeric = ccv_control_packet.numeric_controls()[:, :3].to(dtype=latent.dtype)
        self._last_shared_latent = latent + numeric * 0.01
        self._last_specific_latent = latent + 0.02
        self._last_h_score = latent + 0.03
        self._last_h_explain_aligned = latent + 0.04
        self._last_content_profile = latent + 0.05
        self._last_content_evidence_latent = latent + 0.06
        rating = self._last_shared_latent.mean(dim=1) * self.scale
        vocab = int(self.ntoken)
        word_dist = torch.zeros(bsz, int(target_tokens.size(1)), vocab)
        word_dist = word_dist + self.scale.view(1, 1, 1)
        return rating, word_dist[:, 0, :] * 0.0, word_dist


def _final_cfg(policy: str = "real_sample_plan_one_batch") -> SimpleNamespace:
    return SimpleNamespace(
        ddp_find_unused_parameters=False,
        ddp_find_unused_false_preflight=policy,
        train_precision="fp32",
        step5_head="step5A",
        label_smoothing=0.0,
        coef=0.1,
        explainer_loss_weight=0.005,
        lambda_ortho_step5=0.0,
        lambda_ortho_xcov=1.0,
        lambda_ortho_cos=0.25,
        pin_memory=False,
        non_blocking_h2d=False,
    )


class TestStep5RealBatchPreflight(unittest.TestCase):
    def test_real_batch_preflight_executes_forward_backward_without_optimizer_step(self) -> None:
        model = _TinyStep5Model()
        before = float(model.scale.detach())
        result = run_step5_find_unused_parameters_preflight(
            model,
            _final_cfg(),
            step5_innov_cfg=for_test_default_step5_innovation_config(),
            train_dataloader=[_gathered_batch()],
        )
        self.assertTrue(result["real_data_batch_used"])
        self.assertFalse(result["synthetic_batch_used"])
        self.assertTrue(result["forward_success"])
        self.assertTrue(result["backward_success"])
        self.assertFalse(result["optimizer_success"])
        self.assertFalse(result["optimizer_step_executed"])
        self.assertFalse(result["formal_model_optimizer_step_executed"])
        self.assertFalse(result["formal_model_weights_changed_by_preflight"])
        self.assertTrue(result["scratch_cleared_after_preflight"])
        self.assertTrue(result["grads_cleared_after_preflight"])
        self.assertEqual(result["graph_scratch_before_ema"], [])
        self.assertEqual(result["all_trainable_grad_status"], "pass")
        self.assertEqual(result["trainable_param_count"], result["grad_present_count"])
        self.assertEqual(result["lora_trainable_count"], result["lora_grad_present_count"])
        self.assertEqual(result["missing_grad_params"], [])
        self.assertEqual(result["polarity_ids_shape"], [4, 1])
        self.assertEqual(float(model.scale.detach()), before)

    def test_synthetic_policy_is_rejected_as_formal_gate(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be one of"):
            run_step5_find_unused_parameters_preflight(
                _TinyStep5Model(),
                _final_cfg("synthetic_one_batch"),
                step5_innov_cfg=for_test_default_step5_innovation_config(),
                train_dataloader=[_gathered_batch()],
            )

    def test_real_collate_generates_rank2_polarity_ids(self) -> None:
        sample = {
            "user_idx": torch.tensor(0),
            "item_idx": torch.tensor(0),
            "rating": torch.tensor(4.0),
            "explanation_idx": torch.tensor([1, 2]),
            "domain_idx": torch.tensor(1),
            "sample_id": torch.tensor(0),
            "exp_sample_weight": torch.tensor(1.0),
            "route_scorer_mask": torch.tensor(1.0),
            "route_explainer_mask": torch.tensor(0.0),
            "entropy_score": torch.tensor(0.0),
            "uncertainty_score": torch.tensor(0.1),
            "confidence_bucket": torch.tensor(2.0),
            "content_anchor_score": torch.tensor(0.9),
            "style_anchor_score": torch.tensor(0.8),
            "evidence_features": torch.zeros(STEP5_EVIDENCE_FEATURE_DIM),
            "content_evidence_ids": torch.tensor([1, 2]),
            "style_evidence_ids": torch.tensor([3]),
            "domain_style_anchor_ids": torch.tensor([4]),
            "local_style_hint_ids": torch.tensor([5]),
            "polarity_ids": torch.tensor([2]),
        }
        batch = _step5_collate_dynamic([sample, sample], dynamic_padding=True, fixed_max_length=8)
        polarity_ids = batch[19]
        self.assertEqual(tuple(polarity_ids.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
