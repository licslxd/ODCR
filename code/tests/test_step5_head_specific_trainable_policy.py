from __future__ import annotations

import torch
from torch import nn

from executors.step5_engine import _apply_step5_head_trainable_contract
from odcr_core.step5_grad_contract import head_gated_loss_contract


class _FakeFlan(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.lora_A = nn.Parameter(torch.randn(2, 4))
        self.lora_B = nn.Parameter(torch.randn(4, 2))


class _TinyPolicyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.domain_gate = nn.Linear(4, 4)
        self.transformer_encoder = nn.Linear(4, 4)
        self.flan_explainer = _FakeFlan()
        self.ccv_numeric_adapter = nn.Linear(3, 4)
        self.ccv_control_adapter = nn.Linear(8, 4)
        self.fca_score_align = nn.Linear(4, 4)
        self.fca_explain_align = nn.Linear(4, 4)


def test_step5a_trainable_contract_freezes_step5b_only_modules() -> None:
    model = _TinyPolicyModel()
    meta = _apply_step5_head_trainable_contract(
        model,
        head="step5A",
        peft_meta={"target_modules": ["domain_gate", "transformer_encoder"]},
    )
    assert meta["head"] == "step5A"
    assert meta["combined_formal_enabled"] is False
    assert meta["all_trainable_grad_required"] is True
    assert "flan_explainer.lora_A" in meta["frozen_parameter_names"]
    assert "ccv_numeric_adapter.weight" in meta["frozen_parameter_names"]
    assert "domain_gate.weight" in meta["trainable_parameter_names"]
    assert meta["head_gated_loss_contract"]["active_losses"] == ["scorer_mse", "lci", "orthogonal_keep"]


def test_step5b_trainable_contract_keeps_only_flan_lora_params_inside_flan() -> None:
    model = _TinyPolicyModel()
    meta = _apply_step5_head_trainable_contract(
        model,
        head="step5B",
        peft_meta={"target_modules": ["flan_explainer.base"]},
    )
    assert meta["head"] == "step5B"
    assert "flan_explainer.lora_A" in meta["trainable_parameter_names"]
    assert "flan_explainer.lora_B" in meta["trainable_parameter_names"]
    assert "flan_explainer.base.weight" in meta["frozen_parameter_names"]
    assert meta["head_gated_loss_contract"]["active_losses"] == ["explainer_ce", "fca", "orthogonal_keep"]


def test_combined_formal_contract_is_disabled_until_audited() -> None:
    contract = head_gated_loss_contract("combined")
    assert contract["combined_formal_enabled"] is False
    assert "scorer_mse" in contract["active_losses"]
    assert "explainer_ce" in contract["active_losses"]
