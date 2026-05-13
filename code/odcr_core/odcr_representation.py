from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from odcr_core.csb_contract import (
    CSB_FORWARD_OUTPUT_SCHEMA_VERSION,
    CSB_PACKET_SCHEMA_VERSION,
    build_csb_packet_summary,
    csb_contract_hash,
)


@dataclass
class CSBODCRLatentBundle:
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
    z_content: torch.Tensor
    z_style: torch.Tensor
    z_domain: torch.Tensor
    z_uncertainty: torch.Tensor
    csb_packet: dict[str, Any]
    csb_diagnostics: dict[str, Any]
    csb_schema_version: str
    csb_contract_hash: str

    @property
    def shared_latent(self) -> torch.Tensor:
        return self.z_content

    @property
    def specific_latent(self) -> torch.Tensor:
        return self.z_style


def _mlp(width: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
    )


class CSBODCRBottleneck(nn.Module):
    """
    CSB-ODCR Step3 causal structure bottleneck:
    - primary shared/specific streams remain the rating-clean backbone inputs
    - z_content/z_style/z_domain/z_uncertainty are CSB branch variables
    - CSB branch projections are fed by stop-gradient primary seeds so EASD,
      HSS, and geometry train the bottleneck first instead of hard-pulling the
      primary rating path.
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
        self.csb_content_head = _mlp(self.hidden_size, dropout)
        self.csb_style_head = _mlp(self.hidden_size, dropout)
        self.csb_domain_head = _mlp(self.hidden_size, dropout)
        self.csb_uncertainty_head = _mlp(self.hidden_size, dropout)
        self.csb_content_norm = nn.LayerNorm(self.hidden_size)
        self.csb_style_norm = nn.LayerNorm(self.hidden_size)
        self.csb_domain_norm = nn.LayerNorm(self.hidden_size)
        self.csb_uncertainty_norm = nn.LayerNorm(self.hidden_size)
        self.domain_style_proto = nn.Embedding(self.num_domains, self.hidden_size)
        self.shared_global_proto = nn.Parameter(torch.empty(1, self.hidden_size))
        self.gate_wc = nn.Parameter(torch.tensor(1.25))
        self.gate_ws = nn.Parameter(torch.tensor(1.25))
        self.anchor_head_content = nn.Linear(p, 1)
        self.anchor_head_style = nn.Linear(p, 1)
        self.csb_contract_payload: dict[str, Any] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.shared_global_proto, -0.08, 0.08)

    def set_csb_contract_payload(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError("CSBODCRBottleneck requires resolved CSB contract payload.")
        contract = dict(payload)
        stored_hash = str(contract.get("contract_hash") or "").strip()
        if not stored_hash:
            raise ValueError("CSBODCRBottleneck requires CSB contract_hash.")
        computed_hash = csb_contract_hash(contract)
        if stored_hash != computed_hash:
            raise ValueError(f"CSB contract hash mismatch: stored={stored_hash} computed={computed_hash}.")
        self.csb_contract_payload = contract

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
    ) -> CSBODCRLatentBundle:
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

        shared_stop = shared.detach()
        specific_stop = specific.detach()
        content_stop = content_target.detach()
        style_stop = style_target.detach()
        domain_stop = domain_style_target.detach()
        local_stop = local_style_target.detach()
        reliability = ((ca + sa) * 0.5).clamp(0.0, 1.0)
        uncertainty = (1.0 - reliability).clamp(0.0, 1.0)

        z_content = self.csb_content_norm(self.csb_content_head(shared_stop + 0.35 * content_stop))
        z_style = self.csb_style_norm(self.csb_style_head(specific_stop + 0.35 * style_stop + 0.15 * local_stop))
        z_domain = self.csb_domain_norm(
            self.csb_domain_head(domain_vec.detach() + 0.35 * domain_stop + 0.15 * style_stop)
        )
        z_uncertainty = self.csb_uncertainty_norm(
            self.csb_uncertainty_head(
                uncertainty * (shared_stop + specific_stop)
                + (1.0 - uncertainty) * (content_stop + style_stop) * 0.5
            )
        )

        shared_proj = self.shared_projector(z_content)
        specific_proj = self.specific_projector(z_style)
        pred_c = torch.sigmoid(self.anchor_head_content(shared_proj)).squeeze(-1)
        pred_s = torch.sigmoid(self.anchor_head_style(specific_proj)).squeeze(-1)
        shared_prototype = self.shared_proto_norm(self.shared_global_proto).expand(shared.size(0), -1)
        if self.csb_contract_payload is None:
            raise RuntimeError("CSBODCRBottleneck forward requires resolved CSB contract payload.")
        contract_payload = dict(self.csb_contract_payload)
        contract_hash = csb_contract_hash(contract_payload)
        csb_diagnostics = {
            "schema_version": "csb_odcr_step3_diagnostics/1",
            "content_anchor_source": "preprocess_content_anchor_score",
            "style_anchor_source": "preprocess_style_anchor_score",
            "uncertainty_source": "1 - mean(content_anchor_score, style_anchor_score)",
            "primary_to_csb_boundary": "stop_gradient_primary_seed_to_csb_heads",
            "packet_schema_version": CSB_PACKET_SCHEMA_VERSION,
        }
        csb_packet = build_csb_packet_summary(
            z_content=z_content,
            z_style=z_style,
            z_domain=z_domain,
            z_uncertainty=z_uncertainty,
            diagnostics=csb_diagnostics,
            contract_payload=contract_payload,
        )
        return CSBODCRLatentBundle(
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
            z_content=z_content,
            z_style=z_style,
            z_domain=z_domain,
            z_uncertainty=z_uncertainty,
            csb_packet=csb_packet,
            csb_diagnostics=csb_diagnostics,
            csb_schema_version=CSB_FORWARD_OUTPUT_SCHEMA_VERSION,
            csb_contract_hash=contract_hash,
        )
