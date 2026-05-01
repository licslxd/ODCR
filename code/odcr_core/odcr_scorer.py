from __future__ import annotations

import torch
from torch import nn


class ODCRScorer(nn.Module):
    """Scorer：shared + content profile + specific → rating；保留 penultimate 供 FCA。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        h = int(hidden_size)
        self.fc1 = nn.Linear(h * 3, h * 2)
        self.fc2 = nn.Linear(h * 2, h)
        self.fc3 = nn.Linear(h, 1)
        self.last_hidden: torch.Tensor | None = None

    def forward(
        self,
        shared_latent: torch.Tensor,
        content_profile: torch.Tensor,
        specific_latent: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([shared_latent, content_profile, specific_latent], dim=-1)
        h = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h))
        self.last_hidden = h2
        return self.fc3(h2).view(-1)
