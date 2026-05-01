from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ODCRLatentBundle:
    shared: torch.Tensor
    specific: torch.Tensor
    specific_core: torch.Tensor
    shared_proj: torch.Tensor
    specific_proj: torch.Tensor
    residual_local: torch.Tensor
    domain_style_component: torch.Tensor
    """逐样本 domain-global style prototype lookup。"""
    domain_style_proto: torch.Tensor
    """形状 (num_domains, H) 的 domain style 原型表。"""
    shared_prototype: torch.Tensor
    """逐样本 broadcast 后的 shared global prototype。"""
    content_evidence_target: torch.Tensor
    style_evidence_target: torch.Tensor
    domain_style_target: torch.Tensor
    local_style_target: torch.Tensor
    polarity_target: torch.Tensor
    anchor_pred_content: torch.Tensor
    anchor_pred_style: torch.Tensor

    @property
    def shared_latent(self) -> torch.Tensor:
        return self.shared

    @property
    def specific_latent(self) -> torch.Tensor:
        return self.specific


def _mlp(width: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
    )


class ODCRDisentangler(nn.Module):
    """
    Step3 主解耦器：
    - shared 只接 content-guided source
    - specific 只接 style-guided source
    - specific = domain-global style prototype + local residual
    """

    def __init__(self, hidden_size: int, proj_size: int | None = None, dropout: float = 0.1, num_domains: int = 2):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_domains = int(num_domains)
        p = int(proj_size or hidden_size)
        self.shared_encoder = _mlp(self.hidden_size, dropout)
        self.specific_encoder = _mlp(self.hidden_size, dropout)
        self.residual_mlp = _mlp(self.hidden_size, dropout)
        self.content_target_proj = _mlp(self.hidden_size, dropout)
        self.style_target_proj = _mlp(self.hidden_size, dropout)
        self.domain_style_target_proj = _mlp(self.hidden_size, dropout)
        self.local_style_target_proj = _mlp(self.hidden_size, dropout)
        self.polarity_target_proj = _mlp(self.hidden_size, dropout)
        self.shared_input_norm = nn.LayerNorm(self.hidden_size)
        self.specific_input_norm = nn.LayerNorm(self.hidden_size)
        self.content_target_norm = nn.LayerNorm(self.hidden_size)
        self.style_target_norm = nn.LayerNorm(self.hidden_size)
        self.domain_style_target_norm = nn.LayerNorm(self.hidden_size)
        self.local_style_target_norm = nn.LayerNorm(self.hidden_size)
        self.polarity_target_norm = nn.LayerNorm(self.hidden_size)
        self.shared_output_norm = nn.LayerNorm(self.hidden_size)
        self.specific_output_norm = nn.LayerNorm(self.hidden_size)
        self.residual_output_norm = nn.LayerNorm(self.hidden_size)
        self.domain_proto_norm = nn.LayerNorm(self.hidden_size)
        self.shared_proto_norm = nn.LayerNorm(self.hidden_size)
        self.shared_projector = nn.Sequential(nn.Linear(self.hidden_size, p), nn.ReLU(), nn.Linear(p, p))
        self.specific_projector = nn.Sequential(nn.Linear(self.hidden_size, p), nn.ReLU(), nn.Linear(p, p))
        self.domain_style_proto = nn.Embedding(self.num_domains, self.hidden_size)
        self.shared_global_proto = nn.Parameter(torch.empty(1, self.hidden_size))
        self.gate_wc = nn.Parameter(torch.tensor(1.25))
        self.gate_ws = nn.Parameter(torch.tensor(1.25))
        self.anchor_head_content = nn.Linear(p, 1)
        self.anchor_head_style = nn.Linear(p, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.shared_global_proto, -0.08, 0.08)

    def forward(
        self,
        shared_source: torch.Tensor,
        specific_source: torch.Tensor,
        domain_idx: torch.Tensor,
        *,
        content_guide: torch.Tensor,
        style_guide: torch.Tensor,
        domain_style_guide: torch.Tensor,
        local_style_guide: torch.Tensor,
        polarity_guide: torch.Tensor,
        content_anchor_score: torch.Tensor,
        style_anchor_score: torch.Tensor,
    ) -> ODCRLatentBundle:
        d = domain_idx.long().view(-1).clamp(0, self.num_domains - 1)
        content_target = self.content_target_norm(self.content_target_proj(content_guide))
        style_target = self.style_target_norm(self.style_target_proj(style_guide))
        domain_style_target = self.domain_style_target_norm(self.domain_style_target_proj(domain_style_guide))
        local_style_target = self.local_style_target_norm(self.local_style_target_proj(local_style_guide))
        polarity_target = self.polarity_target_norm(self.polarity_target_proj(polarity_guide))

        ca = content_anchor_score.to(dtype=shared_source.dtype).view(-1, 1).clamp(0.0, 1.0)
        sa = style_anchor_score.to(dtype=specific_source.dtype).view(-1, 1).clamp(0.0, 1.0)
        gate_c = torch.sigmoid(self.gate_wc * (ca - 0.5))
        gate_s = torch.sigmoid(self.gate_ws * (sa - 0.5))

        shared_seed = self.shared_input_norm(shared_source + 0.45 * content_target)
        shared_base = self.shared_encoder(shared_seed)
        shared = self.shared_output_norm(
            shared_base
            + 0.35 * gate_c * content_target
            + 0.15 * (1.0 - gate_c) * shared_source
        )

        specific_seed = self.specific_input_norm(
            specific_source
            + 0.35 * style_target
            + 0.25 * domain_style_target
            + 0.15 * local_style_target
            + 0.15 * polarity_target
        )
        specific_core = self.specific_encoder(specific_seed)
        residual = self.residual_output_norm(
            self.residual_mlp(
                self.specific_input_norm(specific_core + 0.5 * local_style_target + 0.15 * style_target)
            )
        )
        domain_vec = self.domain_proto_norm(self.domain_style_proto(d))
        specific = self.specific_output_norm(
            specific_core
            + 0.45 * gate_s * domain_vec
            + 0.35 * residual
            + 0.25 * style_target
            + 0.20 * domain_style_target
            + 0.10 * local_style_target
            + 0.10 * polarity_target
        )

        shared_proj = self.shared_projector(shared)
        specific_proj = self.specific_projector(specific)
        pred_c = torch.sigmoid(self.anchor_head_content(shared_proj)).squeeze(-1)
        pred_s = torch.sigmoid(self.anchor_head_style(specific_proj)).squeeze(-1)
        shared_prototype = self.shared_proto_norm(self.shared_global_proto).expand(shared.size(0), -1)
        return ODCRLatentBundle(
            shared=shared,
            specific=specific,
            specific_core=specific_core,
            shared_proj=shared_proj,
            specific_proj=specific_proj,
            residual_local=residual,
            domain_style_component=domain_vec,
            domain_style_proto=self.domain_style_proto.weight,
            shared_prototype=shared_prototype,
            content_evidence_target=content_target,
            style_evidence_target=style_target,
            domain_style_target=domain_style_target,
            local_style_target=local_style_target,
            polarity_target=polarity_target,
            anchor_pred_content=pred_c,
            anchor_pred_style=pred_s,
        )
