from __future__ import annotations

import pytest
import torch
from torch import nn

from executors.step5_engine import (
    assert_no_step5_graph_tensors_attached,
    clear_step5_graph_cache,
    find_step5_graph_tensors_attached,
)


class _ScratchChild(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 3)
        self.last_hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        self.last_hidden = out
        return out


class _ScratchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.child = _ScratchChild()
        self._last_h_score = None
        self._last_nested_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.child(x)
        self._last_h_score = hidden + 1.0
        self._last_nested_cache = {"tuple": (hidden * 2.0,)}
        return hidden.sum()


def test_forward_writes_last_graph_tensors_and_assert_detects_them() -> None:
    model = _ScratchModel()
    loss = model(torch.ones(2, 3))
    assert loss.grad_fn is not None

    findings = find_step5_graph_tensors_attached(model, phase="unit")
    paths = {item["path"] for item in findings}
    assert "_ScratchModel._last_h_score" in paths
    assert "_ScratchModel._last_nested_cache.tuple[0]" in paths
    assert "child.last_hidden" in paths
    with pytest.raises(RuntimeError, match="Step5 graph tensor scratch attached"):
        assert_no_step5_graph_tensors_attached(model, phase="before_ema_init")


def test_clear_step5_graph_cache_clears_nested_and_child_scratch() -> None:
    model = _ScratchModel()
    _ = model(torch.ones(2, 3))

    report = clear_step5_graph_cache(model, reason="unit")

    assert report["remaining_graph_tensor_count"] == 0
    assert model._last_h_score is None
    assert model._last_nested_cache is None
    assert model.child.last_hidden is None
    assert_no_step5_graph_tensors_attached(model, phase="before_ema_init")
