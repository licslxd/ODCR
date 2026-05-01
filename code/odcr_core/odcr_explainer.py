from __future__ import annotations

import torch
from torch import nn


class ODCRExplainerBridge(nn.Module):
    """Explainer path adapter: shared + specific + style control -> hidden logits input."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        shared_latent: torch.Tensor,
        specific_latent: torch.Tensor,
        style_control: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([shared_latent, specific_latent, style_control], dim=-1)
        return self.adapter(x)

