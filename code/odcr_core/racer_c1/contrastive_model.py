"""Compact dual-encoder model for RACER-C1."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int, *, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(num_layers) - 1)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, int(projection_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class RacerDualEncoder(nn.Module):
    def __init__(
        self,
        *,
        query_input_dim: int,
        evidence_input_dim: int,
        hidden_dim: int,
        projection_dim: int,
        num_layers: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.query_tower = ProjectionTower(query_input_dim, hidden_dim, projection_dim, num_layers=num_layers, dropout=dropout)
        self.evidence_tower = ProjectionTower(evidence_input_dim, hidden_dim, projection_dim, num_layers=num_layers, dropout=dropout)
        self.temperature = float(temperature)

    def encode_query(self, query_features: torch.Tensor) -> torch.Tensor:
        return self.query_tower(query_features)

    def encode_evidence(self, evidence_features: torch.Tensor) -> torch.Tensor:
        return self.evidence_tower(evidence_features)

    def logits(self, query_features: torch.Tensor, evidence_features: torch.Tensor) -> torch.Tensor:
        q = self.encode_query(query_features)
        e = self.encode_evidence(evidence_features)
        return q @ e.t() / max(self.temperature, 1e-6)

    def forward(self, query_features: torch.Tensor, evidence_features: torch.Tensor) -> torch.Tensor:
        return self.logits(query_features, evidence_features)


def weighted_multi_positive_infonce(logits: torch.Tensor, positive_weights: torch.Tensor) -> torch.Tensor:
    """Weighted multi-positive InfoNCE.

    ``positive_weights`` is a dense BxB matrix. Zero means negative, positive
    values mean one or more positives for that query.
    """

    if logits.ndim != 2 or positive_weights.shape != logits.shape:
        raise ValueError("logits and positive_weights must both be BxB")
    log_den = torch.logsumexp(logits, dim=1)
    pos = positive_weights.clamp_min(0.0)
    pos_mass = pos.sum(dim=1).clamp_min(1e-12)
    log_num = torch.logsumexp(logits + torch.log(pos.clamp_min(1e-12)), dim=1)
    has_pos = pos_mass > 1e-8
    if not torch.any(has_pos):
        return logits.sum() * 0.0
    return (log_den[has_pos] - log_num[has_pos]).mean()
