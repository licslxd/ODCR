from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executors.step3_train_core import compose_step3_loss_from_forward_output
from step3_sidecar_test_utils import Cfg, build_model, forward_batch


def test_controlled_injection_and_rating_safe_adapter_are_frozen_for_step3_formal() -> None:
    model = build_model(apply_runtime=True)

    assert model.csb_controlled_injection_enabled is False
    assert model.csb_rating_safe_injection_enabled is False
    assert all(not p.requires_grad for p in model.csb_prefix_adapter.parameters())
    assert all(not p.requires_grad for p in model.csb_prefix_gate.parameters())
    assert all(not p.requires_grad for p in model.csb_rating_safe_adapter.parameters())

    out, batch = forward_batch(model)
    bundle = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=Cfg())
    model.zero_grad(set_to_none=True)
    bundle.primary_loss.backward()

    for module in (model.csb_prefix_adapter, model.csb_prefix_gate, model.csb_rating_safe_adapter):
        assert all(param.grad is None for param in module.parameters())
    assert out.csb_diagnostics["controlled_injection_enabled"] is False
    assert out.csb_diagnostics["light_explainer_step3_loss"] is False

