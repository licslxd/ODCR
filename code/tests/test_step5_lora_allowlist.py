from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from odcr_core.config_resolver import _step5_forbidden_lora_targets_from_model_config
from odcr_core.step5_native_lora import (
    HEAD_AWARE_LORA_TARGET_SENTINEL,
    LoRALinear,
    apply_native_lora_to_step5_model,
    resolve_step5_lora_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class _TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(4, 1, batch_first=True)
        self.ff = nn.Linear(4, 4)


class _TinyStep5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.domain_gate = nn.Linear(4, 4)
        self.transformer_encoder = nn.Module()
        self.transformer_encoder.layers = nn.ModuleList([_TinyLayer()])
        self.ccv_numeric_adapter = nn.Linear(3, 4)
        self.ccv_control_adapter = nn.Sequential(nn.Linear(8, 4), nn.GELU())
        self.fca_score_align = nn.Linear(4, 4)
        self.fca_explain_align = nn.Linear(4, 4)
        self.flan_explainer = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))


def test_empty_target_modules_is_retired_and_sentinel_resolves_to_policy() -> None:
    model = _TinyStep5()
    with pytest.raises(RuntimeError, match=r"target_modules=\[\] is retired"):
        resolve_step5_lora_targets(model, head="step5A", configured_target_modules=[])

    resolved = resolve_step5_lora_targets(
        model,
        head="step5A",
        configured_target_modules=[HEAD_AWARE_LORA_TARGET_SENTINEL],
    )
    assert resolved["target_policy_id"] == "step5_head_aware_lora_allowlist/1"
    assert resolved["head_specific_lora_allowlist_id"].endswith(":step5A")
    assert resolved["final_lora_target_modules"]


def test_head_aware_targets_exclude_legacy_and_mha_out_proj() -> None:
    model = _TinyStep5()
    resolved_a = resolve_step5_lora_targets(
        model,
        head="step5A",
        configured_target_modules=[HEAD_AWARE_LORA_TARGET_SENTINEL],
    )
    targets_a = set(resolved_a["final_lora_target_modules"])
    assert "domain_gate" in targets_a
    assert "transformer_encoder.layers.0.ff" in targets_a
    assert "transformer_encoder.layers.0.self_attn.out_proj" not in targets_a
    assert not any("hidden2token" in name or "recommender" in name or "flan_soft_prompt_stack" in name for name in targets_a)
    assert "transformer_encoder.layers.0.self_attn.out_proj" in set(resolved_a["forbidden_lora_targets"])

    resolved_b = resolve_step5_lora_targets(
        model,
        head="step5B",
        configured_target_modules=[HEAD_AWARE_LORA_TARGET_SENTINEL],
    )
    targets_b = set(resolved_b["final_lora_target_modules"])
    assert "ccv_numeric_adapter" in targets_b
    assert "flan_explainer.0" in targets_b


def test_lora_injection_never_wraps_mha_out_proj_and_persists_final_targets() -> None:
    torch.manual_seed(0)
    model = _TinyStep5()
    meta = apply_native_lora_to_step5_model(
        model,
        r=2,
        alpha=4,
        dropout=0.0,
        head="step5A",
        target_modules_override=[HEAD_AWARE_LORA_TARGET_SENTINEL],
    )
    assert meta["target_modules"]
    assert meta["target_policy_id"] == "step5_head_aware_lora_allowlist/1"
    assert isinstance(model.domain_gate, LoRALinear)
    assert not isinstance(model.transformer_encoder.layers[0].self_attn.out_proj, LoRALinear)
    assert not any("out_proj.lora_A" in name or "out_proj.lora_B" in name for name, _ in model.named_parameters())


def test_explicit_forbidden_targets_fail_fast() -> None:
    model = _TinyStep5()
    with pytest.raises(RuntimeError, match="deleted legacy module hidden2token"):
        resolve_step5_lora_targets(model, head="step5A", configured_target_modules=["hidden2token"])
    with pytest.raises(RuntimeError, match="MultiheadAttention.out_proj"):
        resolve_step5_lora_targets(
            model,
            head="step5A",
            configured_target_modules=["transformer_encoder.layers.0.self_attn.out_proj"],
        )


def test_resolved_forbidden_targets_include_all_mha_out_proj() -> None:
    forbidden = _step5_forbidden_lora_targets_from_model_config(4)
    assert forbidden == [
        "domain_cross_attn.out_proj",
        "transformer_encoder.layers.0.self_attn.out_proj",
        "transformer_encoder.layers.1.self_attn.out_proj",
        "transformer_encoder.layers.2.self_attn.out_proj",
        "transformer_encoder.layers.3.self_attn.out_proj",
    ]
