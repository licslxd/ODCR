from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from executors.step3_train_core import compose_step3_loss_from_forward_output
from step3_sidecar_test_utils import Cfg, build_model, forward_batch


def test_light_explainer_is_not_step3_formal_total_loss() -> None:
    model = build_model(apply_runtime=True)
    out, batch = forward_batch(model)
    bundle = compose_step3_loss_from_forward_output(forward_output=out, batch=batch, final_cfg=Cfg())

    assert bundle.weights["L_light_explainer"] == 0.0
    assert bundle.participates_in_total["L_light_explainer"] is False
    assert bundle.component_roles["L_light_explainer"] == "disabled_moved_to_step5"
    assert bundle.logging_summary["gradient_firewall"]["disabled_step3_formal_losses"] == ["L_light_explainer"]
    assert float((bundle.total_loss - bundle.components["L_rating_shared"]).detach().abs().item()) == 0.0

