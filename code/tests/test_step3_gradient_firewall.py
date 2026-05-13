from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executors.step3_train_core import compose_step3_loss_from_forward_output, step3_parameter_training_role
from step3_sidecar_test_utils import Cfg, build_model, forward_batch


def _role_grad_abs(model, role: str) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if step3_parameter_training_role(name) != role or param.grad is None:
            continue
        total += float(param.grad.detach().abs().sum().item())
    return total


def _named_grad_abs(model, text: str) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if text in name and param.grad is not None:
            total += float(param.grad.detach().abs().sum().item())
    return total


def test_rating_loss_updates_primary_scorer_but_not_sidecar() -> None:
    model = build_model(apply_runtime=True)
    out, batch = forward_batch(model)
    bundle = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=Cfg())

    model.zero_grad(set_to_none=True)
    bundle.primary_loss.backward()

    assert _role_grad_abs(model, "primary_scorer") > 0.0
    assert _role_grad_abs(model, "csb_sidecar") == 0.0
    assert torch.isclose(bundle.total_loss.detach(), bundle.components["L_rating_shared"].detach()).item()


def test_sidecar_loss_updates_sidecar_but_not_primary_or_injection() -> None:
    model = build_model(apply_runtime=True)
    out, batch = forward_batch(model)
    bundle = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=Cfg())

    model.zero_grad(set_to_none=True)
    bundle.sidecar_loss.backward()

    assert _role_grad_abs(model, "csb_sidecar") > 0.0
    assert _role_grad_abs(model, "primary_scorer") == 0.0
    assert _named_grad_abs(model, "csb_rating_safe_adapter") == 0.0
    assert _named_grad_abs(model, "csb_prefix") == 0.0

