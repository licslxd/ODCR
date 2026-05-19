# -*- coding: utf-8 -*-
"""主线 batch gather 命名协议：取代 tuple 位置语义，供训练 / 验证 / BLEU 共用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class GatheredBatch:
    """Model.gather(batch, device) 的统一返回值（主线唯一协议）。"""

    user_idx: torch.Tensor
    item_idx: torch.Tensor
    rating: torch.Tensor
    tgt_input: torch.Tensor
    tgt_output: torch.Tensor
    domain_idx: torch.Tensor
    sample_id: torch.Tensor
    exp_sample_weight: Optional[torch.Tensor] = None
    route_scorer_mask: Optional[torch.Tensor] = None
    route_explainer_mask: Optional[torch.Tensor] = None
    entropy_score: Optional[torch.Tensor] = None
    uncertainty_score: Optional[torch.Tensor] = None
    confidence_bucket: Optional[torch.Tensor] = None
    content_anchor_score: Optional[torch.Tensor] = None
    style_anchor_score: Optional[torch.Tensor] = None
    evidence_features: Optional[torch.Tensor] = None
    content_evidence_ids: Optional[torch.Tensor] = None
    style_evidence_ids: Optional[torch.Tensor] = None
    domain_style_anchor_ids: Optional[torch.Tensor] = None
    local_style_hint_ids: Optional[torch.Tensor] = None
    polarity_ids: Optional[torch.Tensor] = None
    evidence_quality_prior: Optional[torch.Tensor] = None
    sampler_component_id: Optional[torch.Tensor] = None
    sampler_tier_id: Optional[torch.Tensor] = None

    def assert_uniform_batch_dim(self) -> None:
        """校验各张量首维（batch）一致；失败时指出字段名。"""
        b = int(self.user_idx.shape[0])
        pairs = [
            ("user_idx", self.user_idx),
            ("item_idx", self.item_idx),
            ("rating", self.rating),
            ("tgt_input", self.tgt_input),
            ("tgt_output", self.tgt_output),
            ("domain_idx", self.domain_idx),
            ("sample_id", self.sample_id),
        ]
        for name, t in pairs:
            if int(t.shape[0]) != b:
                raise ValueError(
                    f"GatheredBatch 批量维不一致：基准 batch_size={b}（来自 user_idx），"
                    f"字段 {name!r} 的首维为 {int(t.shape[0])}。"
                )
        w = self.exp_sample_weight
        if w is not None and int(w.shape[0]) != b:
            raise ValueError(
                f"GatheredBatch 批量维不一致：基准 batch_size={b}（来自 user_idx），"
                f"字段 'exp_sample_weight' 的首维为 {int(w.shape[0])}。"
            )
        rs = self.route_scorer_mask
        if rs is not None and int(rs.shape[0]) != b:
            raise ValueError(
                f"GatheredBatch 批量维不一致：基准 batch_size={b}（来自 user_idx），"
                f"字段 'route_scorer_mask' 的首维为 {int(rs.shape[0])}。"
            )
        re = self.route_explainer_mask
        if re is not None and int(re.shape[0]) != b:
            raise ValueError(
                f"GatheredBatch 批量维不一致：基准 batch_size={b}（来自 user_idx），"
                f"字段 'route_explainer_mask' 的首维为 {int(re.shape[0])}。"
            )
        for name, t in (
            ("entropy_score", self.entropy_score),
            ("uncertainty_score", self.uncertainty_score),
            ("confidence_bucket", self.confidence_bucket),
            ("content_anchor_score", self.content_anchor_score),
            ("style_anchor_score", self.style_anchor_score),
            ("evidence_features", self.evidence_features),
            ("content_evidence_ids", self.content_evidence_ids),
            ("style_evidence_ids", self.style_evidence_ids),
            ("domain_style_anchor_ids", self.domain_style_anchor_ids),
            ("local_style_hint_ids", self.local_style_hint_ids),
            ("polarity_ids", self.polarity_ids),
            ("evidence_quality_prior", self.evidence_quality_prior),
            ("sampler_component_id", self.sampler_component_id),
            ("sampler_tier_id", self.sampler_tier_id),
        ):
            if t is not None and int(t.shape[0]) != b:
                raise ValueError(
                    f"GatheredBatch 批量维不一致：基准 batch_size={b}（来自 user_idx），"
                    f"字段 {name!r} 的首维为 {int(t.shape[0])}。"
                )


def require_gathered_batch(obj: Any) -> GatheredBatch:
    """将 gather 返回值约束为 GatheredBatch；否则抛出清晰 TypeError。"""
    if not isinstance(obj, GatheredBatch):
        raise TypeError(
            "model.gather(batch, device) 必须返回 odcr_core.gather_schema.GatheredBatch；"
            f"实际为 {type(obj).__name__!r}。主线已移除 tuple 位置协议，请统一改为 GatheredBatch。"
        )
    obj.assert_uniform_batch_dim()
    return obj
