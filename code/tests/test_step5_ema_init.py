from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

from executors.step5_engine import clear_step5_graph_cache, initialize_step5_ema_model


class _EmaRepro(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self._last_h_score = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        self._last_h_score = out + 1.0
        return out.sum()


def test_averaged_model_fails_before_cleanup_and_passes_after_cleanup() -> None:
    model = _EmaRepro()
    _ = model(torch.ones(2, 2))

    with pytest.raises(RuntimeError, match="Only Tensors created explicitly by the user"):
        AveragedModel(model)

    clear_step5_graph_cache(model, reason="unit_after_preflight")
    ema_model, report = initialize_step5_ema_model(model, ema_decay=0.999)

    assert ema_model is not None
    assert report["ema_strategy"] == "AveragedModel_after_scratch_cleanup"
    assert report["ema_deepcopy_success"] is True
    assert report["ema_init_pass"] is True
    assert report["graph_scratch_before_ema"] == []
