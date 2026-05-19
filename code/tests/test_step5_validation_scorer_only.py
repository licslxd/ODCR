from __future__ import annotations

from types import SimpleNamespace

import torch

from odcr_core.gather_schema import GatheredBatch
from odcr_core.step5_innovation import STEP5_EVIDENCE_FEATURE_DIM, parse_step5_innovation_config_json
from executors.step5_engine import validModel


class _FakeScorer:
    def __init__(self) -> None:
        self.last_hidden = None

    def __call__(self, shared, profile, specific):  # noqa: ANN001
        self.last_hidden = shared + specific
        return (shared + specific + profile).mean(dim=1)


class _FakeStep5Model(torch.nn.Module):
    def __init__(self, batch: GatheredBatch) -> None:
        super().__init__()
        self.batch = batch
        self.exp_loss_fn = SimpleNamespace(label_smoothing=0.0)
        self.odcr_scorer = _FakeScorer()
        self.user_content_profiles = torch.ones(16, 4)
        self.scorer_only_calls = 0
        self.full_calls = 0
        self.logits_materialized = False

    def eval(self):  # noqa: ANN001
        return self

    def gather(self, _batch, _device, *, non_blocking_h2d):  # noqa: ANN001
        assert non_blocking_h2d is False
        return self.batch

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
        return_explainer_logits=True,
        scorer_only=False,
        validation_mode=False,
    ):
        del item_idx, tgt_input, domain_idx, evidence_features, content_anchor_score, style_anchor_score
        bsz = int(user_idx.shape[0])
        shared = torch.ones(bsz, 4)
        specific = torch.full((bsz, 4), 0.5)
        self._last_shared_latent = shared
        self._last_specific_latent = specific
        self._last_content_profile = self.user_content_profiles[user_idx]
        rating = self.odcr_scorer(shared, self._last_content_profile, specific)
        if scorer_only:
            assert return_explainer_logits is False
            assert validation_mode is True
            assert ccv_control_packet is None
            self.scorer_only_calls += 1
            self._last_validation_contract = {
                "step5A_validation_scorer_only": True,
                "flan_explainer_called_in_step5A_validation": False,
                "out_logits_materialized_in_step5A_validation": False,
                "word_dist_returned_in_step5A_validation": False,
            }
            return rating, None, None
        self.full_calls += 1
        self.logits_materialized = True
        word_dist = torch.zeros(bsz, int(target_tokens.shape[1]), 7)
        word_dist[..., 1] = 4.0
        return rating, torch.zeros(bsz, 7), word_dist


def _batch(size: int, *, explainer: bool = False) -> GatheredBatch:
    tgt = torch.ones(size, 3, dtype=torch.long)
    return GatheredBatch(
        user_idx=torch.arange(size, dtype=torch.long),
        item_idx=torch.arange(size, dtype=torch.long),
        rating=torch.ones(size),
        tgt_input=tgt.clone(),
        tgt_output=tgt.clone(),
        domain_idx=torch.zeros(size, dtype=torch.long) if explainer else torch.ones(size, dtype=torch.long),
        sample_id=torch.arange(size, dtype=torch.long),
        exp_sample_weight=torch.ones(size),
        route_scorer_mask=torch.zeros(size) if explainer else torch.ones(size),
        route_explainer_mask=torch.ones(size) if explainer else torch.zeros(size),
        entropy_score=torch.zeros(size),
        uncertainty_score=torch.zeros(size),
        confidence_bucket=torch.full((size,), 2.0),
        content_anchor_score=torch.full((size,), 0.5),
        style_anchor_score=torch.full((size,), 0.5),
        evidence_features=torch.ones(size, STEP5_EVIDENCE_FEATURE_DIM),
        content_evidence_ids=torch.ones(size, 3, dtype=torch.long),
        style_evidence_ids=torch.ones(size, 3, dtype=torch.long),
        domain_style_anchor_ids=torch.ones(size, 3, dtype=torch.long),
        local_style_hint_ids=torch.ones(size, 3, dtype=torch.long),
        polarity_ids=torch.ones(size, 3, dtype=torch.long),
    )


def test_step5a_validation_uses_scorer_only_forward_without_logits() -> None:
    innov = parse_step5_innovation_config_json(None, allow_test_defaults=True)
    batch = _batch(4)
    model = _FakeStep5Model(batch)

    loss_sum, n, *_ = validModel(
        model,
        [object()],
        torch.device("cpu"),
        coef=0.1,
        explainer_loss_weight=0.005,
        step5_innov_cfg=innov,
        non_blocking_h2d=False,
        task_head="step5A",
        valid_forward_micro_batch_size=2,
    )

    assert n == 4
    assert torch.isfinite(torch.tensor(loss_sum))
    assert model.scorer_only_calls == 2
    assert model.full_calls == 0
    assert model.logits_materialized is False
    assert model._last_validation_contract["step5A_validation_scorer_only"] is True


def test_step5b_validation_still_uses_explainer_path_but_respects_microbatch() -> None:
    innov = parse_step5_innovation_config_json(None, allow_test_defaults=True)
    batch = _batch(5, explainer=True)
    model = _FakeStep5Model(batch)

    loss_sum, n, *_ = validModel(
        model,
        [object()],
        torch.device("cpu"),
        coef=0.1,
        explainer_loss_weight=0.005,
        step5_innov_cfg=innov,
        non_blocking_h2d=False,
        task_head="step5B",
        valid_forward_micro_batch_size=2,
    )

    assert n == 5
    assert torch.isfinite(torch.tensor(loss_sum))
    assert model.full_calls == 3
    assert model.logits_materialized is True
