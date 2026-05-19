from __future__ import annotations

import torch

from odcr_core.step5_innovation import for_test_default_step5_innovation_config
from executors.step5_engine import run_step5_find_unused_parameters_preflight
from test_step5_real_batch_preflight import _final_cfg, _gathered_batch, _TinyStep5Model


def test_find_unused_preflight_does_not_change_formal_model_weights_or_leave_state() -> None:
    model = _TinyStep5Model()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    result = run_step5_find_unused_parameters_preflight(
        model,
        _final_cfg(),
        step5_innov_cfg=for_test_default_step5_innovation_config(),
        train_dataloader=[_gathered_batch()],
    )

    after = model.state_dict()
    assert result["optimizer_step_executed"] is False
    assert result["formal_model_optimizer_step_executed"] is False
    assert result["formal_model_weights_changed_by_preflight"] is False
    assert result["scratch_cleared_after_preflight"] is True
    assert result["grads_cleared_after_preflight"] is True
    assert result["graph_scratch_before_ema"] == []
    for name, expected in before.items():
        assert torch.equal(after[name], expected), name
