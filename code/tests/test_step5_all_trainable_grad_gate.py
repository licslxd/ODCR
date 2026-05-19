from __future__ import annotations

import torch
from torch import nn

from odcr_core.step5_grad_contract import validate_all_trainable_params_receive_grad


class _TinyGradModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(3, 1)
        self.unused = nn.Linear(3, 1)
        self.lora_A = nn.Parameter(torch.randn(2, 3))
        self.lora_B = nn.Parameter(torch.randn(1, 2))

    def forward_used_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x).sum()

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        lora = (x @ self.lora_A.T @ self.lora_B.T).sum()
        return self.used(x).sum() + self.unused(x).sum() + lora


def test_all_trainable_grad_gate_reports_missing_lora_and_non_lora_params() -> None:
    model = _TinyGradModel()
    loss = model.forward_used_only(torch.ones(4, 3))
    loss.backward()
    report = validate_all_trainable_params_receive_grad(
        model,
        loss,
        head="step5A",
        evidence_context={"evidence_id": "unit"},
        fail_on_missing=False,
    )
    assert report["status"] == "fail"
    assert report["trainable_param_count"] == 6
    assert report["grad_present_count"] == 2
    assert report["lora_trainable_count"] == 2
    assert report["lora_grad_present_count"] == 0
    assert set(report["missing_grad_params"]) == {"unused.weight", "unused.bias", "lora_A", "lora_B"}


def test_all_trainable_grad_gate_passes_when_lora_and_non_lora_all_receive_grad() -> None:
    model = _TinyGradModel()
    loss = model.forward_all(torch.ones(4, 3))
    loss.backward()
    report = validate_all_trainable_params_receive_grad(
        model,
        loss,
        head="step5B",
        evidence_context={"evidence_id": "unit"},
        fail_on_missing=True,
    )
    assert report["status"] == "pass"
    assert report["trainable_param_count"] == report["grad_present_count"]
    assert report["lora_trainable_count"] == report["lora_grad_present_count"] == 2


def test_all_trainable_grad_gate_raises_on_missing_when_strict() -> None:
    model = _TinyGradModel()
    loss = model.forward_used_only(torch.ones(4, 3))
    loss.backward()
    try:
        validate_all_trainable_params_receive_grad(
            model,
            loss,
            head="step5A",
            evidence_context={"evidence_id": "unit"},
            fail_on_missing=True,
        )
    except RuntimeError as exc:
        assert "trainable params without grad" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing trainable gradients should fail the strict gate")
